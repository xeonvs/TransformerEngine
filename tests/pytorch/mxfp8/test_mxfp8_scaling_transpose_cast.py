# Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.

"""Tests for MXFP8 cast-and-transpose helper."""

from __future__ import annotations

import pytest
import torch

te = pytest.importorskip("transformer_engine")
tex = pytest.importorskip("transformer_engine_torch")

if not torch.cuda.is_available():
    pytest.skip("CUDA required", allow_module_level=True)
if not hasattr(tex, "mxfp8_scaling_transpose_cast"):
    pytest.skip("Built TE missing mxfp8_scaling_transpose_cast", allow_module_level=True)

from transformer_engine.pytorch.constants import MXFP8_BLOCK_SCALING_SIZE, DType
from transformer_engine.pytorch.tensor.mxfp8_tensor import MXFP8Quantizer


def _make_source(rows: int, cols: int, dtype=torch.bfloat16, seed: int = 1234) -> torch.Tensor:
    generator = torch.Generator(device="cuda").manual_seed(seed)
    return torch.randn((rows, cols), dtype=dtype, device="cuda", generator=generator) * 4.0


def _make_quantizer(fp8_dtype: DType = DType.kFloat8E4M3) -> MXFP8Quantizer:
    quantizer = MXFP8Quantizer(fp8_dtype=fp8_dtype, rowwise=True, columnwise=True)
    quantizer.optimize_for_gemm = False
    return quantizer


def _copy_adapter_transpose(mxfp8_tensor) -> tuple[torch.Tensor, torch.Tensor]:
    return (
        mxfp8_tensor._columnwise_data.contiguous().t().contiguous(),
        mxfp8_tensor._columnwise_scale_inv.contiguous().t().contiguous(),
    )


def _swizzle_mxfp8_scale(compact_scale: torch.Tensor, source_rows: int) -> torch.Tensor:
    """Reference GEMM swizzle for MXFP8 E8M0 scale tensors."""

    output = torch.empty_like(compact_scale.flatten())
    num_tiles_x = (source_rows + 127) // 128
    tile_size = 4 * 128
    for i in range(compact_scale.shape[0]):
        for j in range(compact_scale.shape[1]):
            tile_idx_x = j // 4
            tile_idx_y = i // 128
            idx_in_tile_x = j % 4
            idx_in_tile_y = i % 128
            idx = (tile_idx_y * num_tiles_x + tile_idx_x) * tile_size
            idx += (idx_in_tile_y % 32) * 16 + (idx_in_tile_y // 32) * 4 + idx_in_tile_x
            output[idx] = compact_scale[i, j]
    return output.view_as(compact_scale)


@pytest.mark.parametrize("rows,cols", [(64, 128), (128, 256)])
@pytest.mark.parametrize("fp8_dtype", [DType.kFloat8E4M3, DType.kFloat8E5M2])
def test_transpose_cast_matches_columnwise_copy_adapter(rows, cols, fp8_dtype):
    source = _make_source(rows, cols)
    quantizer = _make_quantizer(fp8_dtype)
    quantizer.set_usage(rowwise=True, columnwise=True)
    mxfp8 = quantizer.quantize(source)

    expected_payload, expected_scale = _copy_adapter_transpose(mxfp8)

    helper_quantizer = MXFP8Quantizer(fp8_dtype=fp8_dtype, rowwise=True, columnwise=False)
    helper_quantizer.optimize_for_gemm = False
    transposed = helper_quantizer.quantize_rowwise_transpose(
        source,
        mxfp8._columnwise_scale_inv.contiguous(),
    )

    assert tuple(transposed.shape) == (cols, rows)
    assert transposed._rowwise_data is not None
    assert transposed._columnwise_data is None
    assert torch.equal(transposed._rowwise_data.view(torch.uint8), expected_payload.view(torch.uint8))
    assert torch.equal(transposed._rowwise_scale_inv, expected_scale)


def test_transpose_cast_can_emit_gemm_swizzled_scales():
    rows, cols = 64, 128
    source = _make_source(rows, cols)
    quantizer = _make_quantizer()
    mxfp8 = quantizer.quantize(source)
    expected_payload, compact_scale = _copy_adapter_transpose(mxfp8)

    helper_quantizer = MXFP8Quantizer(fp8_dtype=DType.kFloat8E4M3, rowwise=True, columnwise=False)
    helper_quantizer.optimize_for_gemm = True
    transposed = helper_quantizer.quantize_rowwise_transpose(
        source,
        mxfp8._columnwise_scale_inv.contiguous(),
        with_gemm_swizzled_scales=True,
    )

    assert transposed._with_gemm_swizzled_scales
    assert torch.equal(transposed._rowwise_data.view(torch.uint8), expected_payload.view(torch.uint8))
    assert torch.equal(
        transposed._rowwise_scale_inv,
        _swizzle_mxfp8_scale(compact_scale.cpu(), rows).to(device="cuda"),
    )


def test_transpose_cast_rejects_unaligned_dims():
    source = _make_source(48, 128)
    quantizer = MXFP8Quantizer(fp8_dtype=DType.kFloat8E4M3, rowwise=True, columnwise=False)
    bad_scale = torch.zeros(
        (1, 128),
        dtype=torch.uint8,
        device="cuda",
    )

    with pytest.raises(ValueError, match=f"divisible by {MXFP8_BLOCK_SCALING_SIZE}"):
        quantizer.quantize_rowwise_transpose(source, bad_scale)
