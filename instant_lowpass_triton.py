from __future__ import annotations

import torch

try:
    import triton
    import triton.language as tl
except ImportError as exc:  # pragma: no cover - import guard exercised by caller.
    raise RuntimeError("Triton is required for instant_lowpass_triton") from exc


@triton.jit
def _piecewise_project_kernel(
    x_ptr,
    coeff_ptr,
    out_ptr,
    n_items: tl.constexpr,
    seq_len: tl.constexpr,
    channels: tl.constexpr,
    rank: tl.constexpr,
    segment_count: tl.constexpr,
    segment_len: tl.constexpr,
    BLOCK_R: tl.constexpr,
    BLOCK_C: tl.constexpr,
    BLOCK_L: tl.constexpr,
):
    pid_n = tl.program_id(0)
    pid_r = tl.program_id(1)
    pid_c = tl.program_id(2)

    offs_r = pid_r * BLOCK_R + tl.arange(0, BLOCK_R)
    offs_c = pid_c * BLOCK_C + tl.arange(0, BLOCK_C)
    offs_l = tl.arange(0, BLOCK_L)

    accum = tl.zeros((BLOCK_R, BLOCK_C), dtype=tl.float32)
    for segment in range(0, segment_count):
        segment_accum = tl.zeros((BLOCK_C,), dtype=tl.float32)
        segment_start = segment * segment_len
        for start_l in range(0, segment_len, BLOCK_L):
            cur_l = segment_start + start_l + offs_l
            x_vals = tl.load(
                x_ptr + pid_n * seq_len * channels + cur_l[:, None] * channels + offs_c[None, :],
                mask=(cur_l[:, None] < segment_start + segment_len) & (offs_c[None, :] < channels),
                other=0.0,
            )
            segment_accum += tl.sum(x_vals.to(tl.float32), axis=0)
        coeff = tl.load(
            coeff_ptr + offs_r * segment_count + segment,
            mask=offs_r < rank,
            other=0.0,
        )
        accum += coeff[:, None].to(tl.float32) * segment_accum[None, :]

    valid = (offs_r[:, None] < rank) & (offs_c[None, :] < channels)
    out_offsets = pid_n * rank * channels + offs_r[:, None] * channels + offs_c[None, :]
    tl.store(out_ptr + out_offsets, accum, mask=valid)


@triton.jit
def _gralora_mix_hidden_k2_kernel(
    hidden_ptr,
    out_ptr,
    total: tl.constexpr,
    rank: tl.constexpr,
    BLOCK: tl.constexpr,
):
    offsets = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    mask = offsets < total
    sub_rank = rank // 2
    token_rank = offsets % rank
    block = (offsets // rank) % 2
    token = offsets // (2 * rank)
    source_block = token_rank // sub_rank
    source_rank = block * sub_rank + (token_rank % sub_rank)
    source_offsets = token * (2 * rank) + source_block * rank + source_rank
    values = tl.load(hidden_ptr + source_offsets, mask=mask)
    tl.store(out_ptr + offsets, values, mask=mask)


@triton.jit
def _gralora_unmix_hidden_k2_kernel(
    grad_mixed_ptr,
    out_ptr,
    total: tl.constexpr,
    rank: tl.constexpr,
    BLOCK: tl.constexpr,
):
    offsets = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    mask = offsets < total
    sub_rank = rank // 2
    hidden_rank = offsets % rank
    hidden_block = (offsets // rank) % 2
    token = offsets // (2 * rank)
    mixed_block = hidden_rank // sub_rank
    mixed_rank = hidden_block * sub_rank + (hidden_rank % sub_rank)
    source_offsets = token * (2 * rank) + mixed_block * rank + mixed_rank
    values = tl.load(grad_mixed_ptr + source_offsets, mask=mask)
    tl.store(out_ptr + offsets, values, mask=mask)


def piecewise_project(
    x: torch.Tensor,
    coefficients: torch.Tensor,
    *,
    segment_len: int,
    block_channels: int = 64,
) -> torch.Tensor:
    if not x.is_cuda or not coefficients.is_cuda:
        raise ValueError("piecewise_project requires CUDA tensors")
    if x.ndim != 3 or coefficients.ndim != 2:
        raise ValueError(f"expected x [N,L,C] and coefficients [R,K], got {tuple(x.shape)} and {tuple(coefficients.shape)}")
    if segment_len <= 0:
        raise ValueError("segment_len must be positive")
    x_work = x.contiguous()
    coeff_work = coefficients.contiguous()
    n_items, seq_len, channels = x_work.shape
    rank, segment_count = coeff_work.shape
    if segment_count * int(segment_len) != seq_len:
        raise ValueError(
            f"segment coefficients imply L={segment_count * int(segment_len)}, but x has L={seq_len}"
        )
    block_channels = max(16, int(block_channels))
    out = torch.empty((n_items, rank, channels), device=x.device, dtype=x.dtype)
    block_r = 16
    grid = (n_items, triton.cdiv(rank, block_r), triton.cdiv(channels, block_channels))
    _piecewise_project_kernel[grid](
        x_work,
        coeff_work,
        out,
        n_items,
        seq_len,
        channels,
        rank,
        segment_count,
        int(segment_len),
        BLOCK_R=block_r,
        BLOCK_C=block_channels,
        BLOCK_L=64,
        num_warps=4,
        num_stages=4,
    )
    return out


def _validate_k2_hidden(hidden: torch.Tensor, name: str) -> torch.Tensor:
    if not hidden.is_cuda:
        raise ValueError(f"{name} requires a CUDA tensor")
    if hidden.ndim != 3:
        raise ValueError(f"{name} expected [tokens, 2, rank], got {tuple(hidden.shape)}")
    if int(hidden.shape[1]) != 2:
        raise ValueError(f"{name} only supports gralora_k=2")
    if int(hidden.shape[2]) % 2:
        raise ValueError(f"{name} requires an even GraLoRA rank")
    return hidden.contiguous()


def gralora_mix_hidden_k2(hidden: torch.Tensor, *, block_size: int = 256) -> torch.Tensor:
    hidden_work = _validate_k2_hidden(hidden, "gralora_mix_hidden_k2")
    out = torch.empty_like(hidden_work)
    total = hidden_work.numel()
    grid = (triton.cdiv(total, int(block_size)),)
    _gralora_mix_hidden_k2_kernel[grid](
        hidden_work,
        out,
        total,
        int(hidden_work.shape[2]),
        BLOCK=int(block_size),
        num_warps=4,
    )
    return out


def gralora_unmix_hidden_k2(grad_mixed: torch.Tensor, *, block_size: int = 256) -> torch.Tensor:
    grad_mixed_work = _validate_k2_hidden(grad_mixed, "gralora_unmix_hidden_k2")
    out = torch.empty_like(grad_mixed_work)
    total = grad_mixed_work.numel()
    grid = (triton.cdiv(total, int(block_size)),)
    _gralora_unmix_hidden_k2_kernel[grid](
        grad_mixed_work,
        out,
        total,
        int(grad_mixed_work.shape[2]),
        BLOCK=int(block_size),
        num_warps=4,
    )
    return out
