from __future__ import annotations

import math
import os
import types
from dataclasses import dataclass
from typing import Callable

import torch
import torch.nn.functional as F
from torch import Tensor


_TRITON_GRALORA_MIX: Callable[[Tensor], Tensor] | None = None
_TRITON_GRALORA_UNMIX: Callable[[Tensor], Tensor] | None = None
_TRITON_GRALORA_IMPORT_ERROR: Exception | None = None
_TRITON_PIECEWISE_PROJECT: Callable[[Tensor, Tensor], Tensor] | None = None
_TRITON_PIECEWISE_PROJECT_IMPORT_ERROR: Exception | None = None
_PROJECTOR_CACHE: dict[tuple[str, int, int, str, int, str], Tensor] = {}


@dataclass(frozen=True)
class InstantLowpassConfig:
    projector_kind: str = "hadamard"
    chunk_size: int = 64
    keep: int = 32
    min_hidden_dim: int = 64
    hadamard_backend: str = "auto"
    parameter_gradient: str = "exact"
    exact_input_grad: bool = True
    enabled: bool = True

    def __post_init__(self) -> None:
        projector_kind = str(self.projector_kind).lower()
        hadamard_backend = str(self.hadamard_backend).lower().replace("_", "-")
        parameter_gradient = str(self.parameter_gradient).lower().replace("-", "_")
        if hadamard_backend in {"fast", "triton"}:
            hadamard_backend = "piecewise"
        if parameter_gradient in {"projected", "lowpass", "low_pass"}:
            parameter_gradient = "projected_lowpass"
        object.__setattr__(self, "projector_kind", projector_kind)
        object.__setattr__(self, "hadamard_backend", hadamard_backend)
        object.__setattr__(self, "parameter_gradient", parameter_gradient)
        if projector_kind not in {"hadamard", "dct", "haar"}:
            raise ValueError(f"unknown projector_kind {self.projector_kind!r}")
        if hadamard_backend not in {"auto", "piecewise", "dense"}:
            raise ValueError(f"unknown hadamard_backend {self.hadamard_backend!r}")
        if parameter_gradient not in {"exact", "projected_lowpass"}:
            raise ValueError(f"unknown parameter_gradient {self.parameter_gradient!r}")
        if self.chunk_size < 1:
            raise ValueError("chunk_size must be positive")
        if self.keep < 1 or self.keep > self.chunk_size:
            raise ValueError("keep must be in [1, chunk_size]")
        if self.min_hidden_dim < 0:
            raise ValueError("min_hidden_dim must be non-negative")
        if projector_kind in {"hadamard", "haar"} and not _is_power_of_two(self.chunk_size):
            raise ValueError(f"{projector_kind} requires power-of-two chunk_size")


def _torch_is_compiling() -> bool:
    compiler = getattr(torch, "compiler", None)
    if compiler is not None and hasattr(compiler, "is_compiling"):
        return bool(compiler.is_compiling())
    dynamo = getattr(torch, "_dynamo", None)
    if dynamo is not None and hasattr(dynamo, "is_compiling"):
        return bool(dynamo.is_compiling())
    return False


def _is_power_of_two(value: int) -> bool:
    return value > 0 and (value & (value - 1)) == 0


def _next_power_of_two(value: int) -> int:
    if value <= 1:
        return 1
    return 1 << (int(value) - 1).bit_length()


def _bit_reverse(value: int, width: int) -> int:
    result = 0
    for _ in range(width):
        result = (result << 1) | (value & 1)
        value >>= 1
    return result


def _hadamard_index_for_sequency(sequency: int, width: int) -> int:
    gray = sequency ^ (sequency >> 1)
    return _bit_reverse(gray, width)


def _projector_cache_key(
    kind: str,
    seq_len: int,
    rank: int,
    device: torch.device,
    dtype: torch.dtype,
) -> tuple[str, int, int, str, int, str]:
    device_index = -1 if device.index is None else int(device.index)
    return kind, int(seq_len), int(rank), device.type, device_index, str(dtype)


def _dct_projector(seq_len: int, rank: int, device: torch.device, dtype: torch.dtype) -> Tensor:
    positions = torch.arange(seq_len, device=device, dtype=torch.float32).add_(0.5)
    freqs = torch.arange(rank, device=device, dtype=torch.float32).unsqueeze(1)
    projector = torch.cos((math.pi / float(seq_len)) * freqs * positions.unsqueeze(0))
    if rank > 0:
        projector[0].mul_(math.sqrt(1.0 / float(seq_len)))
    if rank > 1:
        projector[1:].mul_(math.sqrt(2.0 / float(seq_len)))
    return projector.to(dtype=dtype).contiguous()


def _hadamard_projector(seq_len: int, rank: int, device: torch.device, dtype: torch.dtype) -> Tensor:
    if not _is_power_of_two(seq_len):
        raise ValueError(f"Hadamard token projection requires power-of-two seq_len, got {seq_len}")
    width = int(math.log2(seq_len))
    row_indices = torch.tensor(
        [_hadamard_index_for_sequency(index, width) for index in range(rank)],
        device=device,
        dtype=torch.long,
    )
    columns = torch.arange(seq_len, device=device, dtype=torch.long)
    parity = torch.zeros((rank, seq_len), device=device, dtype=torch.bool)
    for bit in range(width):
        row_bit = torch.bitwise_and(torch.bitwise_right_shift(row_indices[:, None], bit), 1).bool()
        col_bit = torch.bitwise_and(torch.bitwise_right_shift(columns[None, :], bit), 1).bool()
        parity.logical_xor_(row_bit & col_bit)
    projector = torch.where(
        parity,
        torch.tensor(-1.0, device=device, dtype=torch.float32),
        torch.tensor(1.0, device=device, dtype=torch.float32),
    )
    projector.mul_(1.0 / math.sqrt(float(seq_len)))
    return projector.to(dtype=dtype).contiguous()


def _haar_projector(seq_len: int, rank: int, device: torch.device, dtype: torch.dtype) -> Tensor:
    if not _is_power_of_two(seq_len):
        raise ValueError(f"Haar token projection requires power-of-two seq_len, got {seq_len}")
    projector = torch.zeros((rank, seq_len), device=device, dtype=torch.float32)
    projector[0].fill_(1.0 / math.sqrt(float(seq_len)))
    row = 1
    block = seq_len
    while row < rank and block > 1:
        half = block // 2
        value = 1.0 / math.sqrt(float(block))
        for start in range(0, seq_len, block):
            if row >= rank:
                break
            projector[row, start : start + half].fill_(value)
            projector[row, start + half : start + block].fill_(-value)
            row += 1
        block //= 2
    return projector.to(dtype=dtype).contiguous()


def _fixed_projector(kind: str, seq_len: int, rank: int, device: torch.device, dtype: torch.dtype) -> Tensor:
    rank = min(max(int(rank), 0), int(seq_len))
    if rank <= 0:
        return torch.empty((0, seq_len), device=device, dtype=dtype)
    key = _projector_cache_key(kind, seq_len, rank, device, dtype)
    cached = _PROJECTOR_CACHE.get(key)
    if cached is not None:
        return cached
    if kind == "dct":
        projector = _dct_projector(seq_len, rank, device, dtype)
    elif kind == "hadamard":
        projector = _hadamard_projector(seq_len, rank, device, dtype)
    elif kind == "haar":
        projector = _haar_projector(seq_len, rank, device, dtype)
    else:
        raise ValueError(f"unknown fixed projector kind {kind!r}")
    _PROJECTOR_CACHE[key] = projector
    return projector


def _piecewise_segment_count(kind: str, seq_len: int, rank: int) -> int | None:
    rank = min(max(int(rank), 0), int(seq_len))
    if kind not in {"hadamard", "haar"} or rank <= 0:
        return None
    if not _is_power_of_two(seq_len):
        return None
    segment_count = min(_next_power_of_two(rank), seq_len)
    if seq_len % segment_count != 0:
        return None
    return segment_count


def _piecewise_projector_coefficients(
    kind: str,
    seq_len: int,
    rank: int,
    device: torch.device,
    dtype: torch.dtype,
) -> tuple[Tensor, int] | None:
    segment_count = _piecewise_segment_count(kind, seq_len, rank)
    if segment_count is None:
        return None
    segment_len = seq_len // segment_count
    projector = _fixed_projector(kind, seq_len, rank, device, dtype)
    coefficients = projector[:, ::segment_len].contiguous()
    return coefficients, segment_len


def _load_triton_gralora_mix() -> tuple[Callable[[Tensor], Tensor], Callable[[Tensor], Tensor]]:
    global _TRITON_GRALORA_IMPORT_ERROR, _TRITON_GRALORA_MIX, _TRITON_GRALORA_UNMIX
    if _TRITON_GRALORA_MIX is not None and _TRITON_GRALORA_UNMIX is not None:
        return _TRITON_GRALORA_MIX, _TRITON_GRALORA_UNMIX
    if _TRITON_GRALORA_IMPORT_ERROR is not None:
        raise RuntimeError("instant GraLoRA Triton kernels are unavailable") from _TRITON_GRALORA_IMPORT_ERROR
    try:
        from instant_lowpass_triton import gralora_mix_hidden_k2, gralora_unmix_hidden_k2
    except Exception as exc:  # pragma: no cover - depends on optional CUDA/Triton runtime.
        _TRITON_GRALORA_IMPORT_ERROR = exc
        raise RuntimeError("instant GraLoRA Triton kernels are unavailable") from exc
    _TRITON_GRALORA_MIX = gralora_mix_hidden_k2
    _TRITON_GRALORA_UNMIX = gralora_unmix_hidden_k2
    return gralora_mix_hidden_k2, gralora_unmix_hidden_k2


def _load_triton_piecewise_project() -> Callable[[Tensor, Tensor], Tensor]:
    global _TRITON_PIECEWISE_PROJECT_IMPORT_ERROR, _TRITON_PIECEWISE_PROJECT
    if _TRITON_PIECEWISE_PROJECT is not None:
        return _TRITON_PIECEWISE_PROJECT
    if _TRITON_PIECEWISE_PROJECT_IMPORT_ERROR is not None:
        raise RuntimeError("instant token projection Triton kernel is unavailable") from (
            _TRITON_PIECEWISE_PROJECT_IMPORT_ERROR
        )
    try:
        from instant_lowpass_triton import piecewise_project
    except Exception as exc:  # pragma: no cover - depends on optional CUDA/Triton runtime.
        _TRITON_PIECEWISE_PROJECT_IMPORT_ERROR = exc
        raise RuntimeError("instant token projection Triton kernel is unavailable") from exc
    _TRITON_PIECEWISE_PROJECT = piecewise_project
    return piecewise_project


def _can_use_triton_mix(hidden: Tensor, use_triton: bool) -> bool:
    return (
        use_triton
        and hidden.is_cuda
        and hidden.ndim == 3
        and int(hidden.shape[1]) == 2
        and int(hidden.shape[2]) % 2 == 0
        and os.environ.get("INSTANT_LOWPASS_DISABLE_TRITON", "0") != "1"
    )


def _fallback_gralora_mix(hidden: Tensor) -> Tensor:
    token_count, block_count, rank = hidden.shape
    if rank % block_count:
        raise ValueError("GraLoRA rank must be divisible by block count")
    sub_rank = rank // block_count
    return (
        hidden.view(token_count, block_count, block_count, sub_rank)
        .permute(0, 2, 1, 3)
        .reshape(token_count, block_count, rank)
        .contiguous()
    )


def _fallback_gralora_unmix(grad_mixed: Tensor) -> Tensor:
    token_count, block_count, rank = grad_mixed.shape
    if rank % block_count:
        raise ValueError("GraLoRA rank must be divisible by block count")
    sub_rank = rank // block_count
    return (
        grad_mixed.view(token_count, block_count, block_count, sub_rank)
        .permute(0, 2, 1, 3)
        .reshape(token_count, block_count, rank)
        .contiguous()
    )


def _gralora_mix(hidden: Tensor, use_triton: bool = True) -> Tensor:
    if _can_use_triton_mix(hidden, use_triton):
        try:
            mix, _unmix = _load_triton_gralora_mix()
            return mix(hidden.contiguous())
        except Exception:
            if os.environ.get("INSTANT_LOWPASS_REQUIRE_TRITON", "0") == "1":
                raise
    return _fallback_gralora_mix(hidden)


def _gralora_unmix(grad_mixed: Tensor, use_triton: bool = True) -> Tensor:
    if _can_use_triton_mix(grad_mixed, use_triton):
        try:
            _mix, unmix = _load_triton_gralora_mix()
            return unmix(grad_mixed.contiguous())
        except Exception:
            if os.environ.get("INSTANT_LOWPASS_REQUIRE_TRITON", "0") == "1":
                raise
    return _fallback_gralora_unmix(grad_mixed)


def _gralora_hidden_from_x(x: Tensor, gralora_a: Tensor) -> Tensor:
    batch, seq_len, in_features = x.shape
    block_count = int(gralora_a.shape[0])
    in_per_block = int(gralora_a.shape[1])
    if in_features != block_count * in_per_block:
        raise ValueError(
            f"input hidden dim {in_features} is incompatible with GraLoRA blocks "
            f"{block_count} x {in_per_block}"
        )
    token_count = batch * seq_len
    x_blocks = x.reshape(token_count, block_count, in_per_block).permute(1, 0, 2).contiguous()
    hidden = torch.bmm(x_blocks, gralora_a).permute(1, 0, 2).contiguous()
    return hidden


def _gralora_forward_from_x(
    x: Tensor,
    gralora_a: Tensor,
    gralora_b: Tensor,
    *,
    use_triton: bool = True,
) -> Tensor:
    batch, seq_len, _in_features = x.shape
    block_count = int(gralora_a.shape[0])
    token_count = batch * seq_len
    out_per_block = int(gralora_b.shape[2])
    hidden = _gralora_hidden_from_x(x, gralora_a)
    mixed = _gralora_mix(hidden, use_triton=use_triton)
    output_blocks = torch.bmm(mixed.permute(1, 0, 2).contiguous(), gralora_b)
    return output_blocks.permute(1, 0, 2).reshape(
        token_count,
        block_count * out_per_block,
    ).view(batch, seq_len, block_count * out_per_block)


def _gralora_grads_from_x_go(
    x: Tensor,
    grad_output: Tensor,
    gralora_a: Tensor,
    gralora_b: Tensor,
    *,
    use_triton: bool = True,
) -> tuple[Tensor, Tensor, Tensor]:
    batch, seq_len, in_features = x.shape
    block_count = int(gralora_a.shape[0])
    in_per_block = int(gralora_a.shape[1])
    out_per_block = int(gralora_b.shape[2])
    token_count = batch * seq_len

    x_blocks = x.reshape(token_count, block_count, in_per_block).permute(1, 0, 2).contiguous()
    hidden = torch.bmm(x_blocks, gralora_a).permute(1, 0, 2).contiguous()
    mixed = _gralora_mix(hidden, use_triton=use_triton)

    go_blocks = grad_output.reshape(token_count, block_count, out_per_block).permute(1, 0, 2).contiguous()
    mixed_blocks = mixed.permute(1, 0, 2).contiguous()
    grad_b = torch.bmm(mixed_blocks.transpose(1, 2), go_blocks)

    grad_mixed = torch.bmm(go_blocks, gralora_b.transpose(1, 2)).permute(1, 0, 2).contiguous()
    grad_hidden = _gralora_unmix(grad_mixed, use_triton=use_triton)
    grad_hidden_blocks = grad_hidden.permute(1, 0, 2).contiguous()

    grad_a = torch.bmm(x_blocks.transpose(1, 2), grad_hidden_blocks)
    grad_x = torch.bmm(grad_hidden_blocks, gralora_a.transpose(1, 2)).permute(1, 0, 2)
    grad_x = grad_x.reshape(batch, seq_len, in_features).contiguous()
    return grad_x, grad_a, grad_b


def _gralora_param_grads_from_x_go(
    x: Tensor,
    grad_output: Tensor,
    gralora_a: Tensor,
    gralora_b: Tensor,
    *,
    use_triton: bool = True,
) -> tuple[Tensor, Tensor]:
    batch, seq_len, _in_features = x.shape
    block_count = int(gralora_a.shape[0])
    in_per_block = int(gralora_a.shape[1])
    out_per_block = int(gralora_b.shape[2])
    token_count = batch * seq_len

    x_blocks = x.reshape(token_count, block_count, in_per_block).permute(1, 0, 2).contiguous()
    hidden = torch.bmm(x_blocks, gralora_a).permute(1, 0, 2).contiguous()
    mixed = _gralora_mix(hidden, use_triton=use_triton)

    go_blocks = grad_output.reshape(token_count, block_count, out_per_block).permute(1, 0, 2).contiguous()
    mixed_blocks = mixed.permute(1, 0, 2).contiguous()
    grad_b = torch.bmm(mixed_blocks.transpose(1, 2), go_blocks)

    grad_mixed = torch.bmm(go_blocks, gralora_b.transpose(1, 2)).permute(1, 0, 2).contiguous()
    grad_hidden = _gralora_unmix(grad_mixed, use_triton=use_triton)
    grad_hidden_blocks = grad_hidden.permute(1, 0, 2).contiguous()
    grad_a = torch.bmm(x_blocks.transpose(1, 2), grad_hidden_blocks)
    return grad_a, grad_b


def _gralora_input_grad(
    grad_output: Tensor,
    gralora_a: Tensor,
    gralora_b: Tensor,
    *,
    use_triton: bool = True,
) -> Tensor:
    batch, seq_len, out_features = grad_output.shape
    block_count = int(gralora_a.shape[0])
    in_per_block = int(gralora_a.shape[1])
    out_per_block = int(gralora_b.shape[2])
    token_count = batch * seq_len
    if out_features != block_count * out_per_block:
        raise ValueError(
            f"output hidden dim {out_features} is incompatible with GraLoRA blocks "
            f"{block_count} x {out_per_block}"
        )
    go_blocks = grad_output.reshape(token_count, block_count, out_per_block).permute(1, 0, 2).contiguous()
    grad_mixed = torch.bmm(go_blocks, gralora_b.transpose(1, 2)).permute(1, 0, 2).contiguous()
    grad_hidden = _gralora_unmix(grad_mixed, use_triton=use_triton).permute(1, 0, 2).contiguous()
    grad_x = torch.bmm(grad_hidden, gralora_a.transpose(1, 2)).permute(1, 0, 2)
    return grad_x.reshape(batch, seq_len, block_count * in_per_block).contiguous()


def _project_fixed_token_basis(
    kind: str,
    x: Tensor,
    rank: int,
    *,
    hadamard_backend: str,
) -> Tensor:
    if (
        x.is_cuda
        and x.ndim == 3
        and kind in {"hadamard", "haar"}
        and hadamard_backend != "dense"
        and os.environ.get("INSTANT_LOWPASS_DISABLE_TRITON", "0") != "1"
    ):
        try:
            coefficients_and_segment_len = _piecewise_projector_coefficients(
                kind,
                int(x.shape[-2]),
                int(rank),
                x.device,
                x.dtype,
            )
            if coefficients_and_segment_len is not None:
                coefficients, segment_len = coefficients_and_segment_len
                return _load_triton_piecewise_project()(x.contiguous(), coefficients, segment_len=segment_len)
        except Exception:
            if os.environ.get("INSTANT_LOWPASS_REQUIRE_TRITON", "0") == "1":
                raise
    projector = _fixed_projector(kind, int(x.shape[-2]), int(rank), x.device, x.dtype)
    return torch.einsum("rl,nlc->nrc", projector, x)


def _project_token_chunks(
    x: Tensor,
    *,
    projector_kind: str,
    chunk_size: int,
    keep: int,
    hadamard_backend: str,
) -> Tensor:
    if x.ndim != 3:
        raise ValueError(f"expected [batch, seq, hidden], got shape {tuple(x.shape)}")
    batch, seq_len, hidden_dim = x.shape
    chunk_size = int(chunk_size)
    keep = int(keep)
    if seq_len % chunk_size:
        raise ValueError(f"sequence length {seq_len} is not divisible by chunk size {chunk_size}")
    chunk_count = seq_len // chunk_size
    chunks = x.reshape(batch, chunk_count, chunk_size, hidden_dim)
    chunks = chunks.reshape(batch * chunk_count, chunk_size, hidden_dim).contiguous()
    return _project_fixed_token_basis(
        projector_kind,
        chunks,
        keep,
        hadamard_backend=hadamard_backend,
    ).contiguous()


def _gralora_delta_weight(gralora_a: Tensor, gralora_b: Tensor) -> Tensor:
    block_count = int(gralora_a.shape[0])
    in_per_block = int(gralora_a.shape[1])
    rank = int(gralora_a.shape[2])
    out_per_block = int(gralora_b.shape[2])
    if rank % block_count:
        raise ValueError("GraLoRA rank must be divisible by block count")
    if int(gralora_b.shape[0]) != block_count or int(gralora_b.shape[1]) != rank:
        raise ValueError(
            f"GraLoRA B shape {tuple(gralora_b.shape)} is incompatible with A shape "
            f"{tuple(gralora_a.shape)}"
        )
    sub_rank = rank // block_count
    a_chunks = gralora_a.view(block_count, in_per_block, block_count, sub_rank)
    b_chunks = gralora_b.view(block_count, block_count, sub_rank, out_per_block)
    # blocks[c, b] is the dense weight block from input block b to output block c.
    blocks = torch.matmul(
        b_chunks.permute(0, 1, 3, 2).contiguous(),
        a_chunks.permute(2, 0, 3, 1).contiguous(),
    )
    return blocks.permute(0, 2, 1, 3).reshape(
        block_count * out_per_block,
        block_count * in_per_block,
    ).contiguous()


class _InstantLowpassLinearFunction(torch.autograd.Function):
    @staticmethod
    def forward(  # type: ignore[override]
        ctx,
        x: Tensor,
        weight: Tensor,
        bias: Tensor | None,
        exact_input_grad: bool,
    ) -> Tensor:
        y = F.linear(x, weight, bias)
        ctx.input_shape = tuple(x.shape)
        ctx.input_dtype = x.dtype
        ctx.weight_dtype = weight.dtype
        ctx.has_bias = bias is not None
        ctx.exact_input_grad = bool(exact_input_grad)
        ctx.save_for_backward(x.contiguous(), weight)
        return y

    @staticmethod
    def backward(ctx, grad_output: Tensor):  # type: ignore[override]
        x, weight = ctx.saved_tensors
        work_dtype = weight.dtype if weight.is_floating_point() else grad_output.dtype
        go = grad_output.to(work_dtype)
        x_work = x.to(work_dtype)
        grad_x = grad_weight = grad_bias = None

        if ctx.needs_input_grad[1]:
            grad_weight = go.reshape(-1, go.shape[-1]).T.matmul(
                x_work.reshape(-1, x_work.shape[-1])
            ).to(ctx.weight_dtype)

        if ctx.needs_input_grad[0]:
            if not ctx.exact_input_grad:
                raise RuntimeError("instant low-pass currently requires exact_input_grad=True")
            grad_x = go.matmul(weight.to(work_dtype)).to(ctx.input_dtype)

        if ctx.has_bias and ctx.needs_input_grad[2]:
            reduce_dims = tuple(range(grad_output.ndim - 1))
            grad_bias = grad_output.sum(dim=reduce_dims)

        return grad_x, grad_weight, grad_bias, None


class _InstantLowpassGraloraFunction(torch.autograd.Function):
    @staticmethod
    def forward(  # type: ignore[override]
        ctx,
        x: Tensor,
        gralora_a: Tensor,
        gralora_b: Tensor,
        use_triton: bool,
    ) -> Tensor:
        output = _gralora_forward_from_x(x, gralora_a, gralora_b, use_triton=bool(use_triton))
        ctx.input_shape = tuple(x.shape)
        ctx.input_dtype = x.dtype
        ctx.a_dtype = gralora_a.dtype
        ctx.b_dtype = gralora_b.dtype
        ctx.use_triton = bool(use_triton)
        ctx.save_for_backward(x.contiguous(), gralora_a, gralora_b)
        return output

    @staticmethod
    def backward(ctx, grad_output: Tensor):  # type: ignore[override]
        x, gralora_a, gralora_b = ctx.saved_tensors
        work_dtype = gralora_a.dtype if gralora_a.is_floating_point() else grad_output.dtype
        go = grad_output.to(work_dtype)
        gralora_a_work = gralora_a.to(work_dtype)
        gralora_b_work = gralora_b.to(work_dtype)
        x_work = x.to(work_dtype)
        grad_x = grad_a = grad_b = None

        if ctx.needs_input_grad[0] and (ctx.needs_input_grad[1] or ctx.needs_input_grad[2]):
            grad_x, grad_a, grad_b = _gralora_grads_from_x_go(
                x_work,
                go,
                gralora_a_work,
                gralora_b_work,
                use_triton=ctx.use_triton,
            )
        else:
            if ctx.needs_input_grad[0]:
                grad_x = _gralora_input_grad(
                    go,
                    gralora_a_work,
                    gralora_b_work,
                    use_triton=ctx.use_triton,
                )
            if ctx.needs_input_grad[1] or ctx.needs_input_grad[2]:
                _grad_x, grad_a, grad_b = _gralora_grads_from_x_go(
                    x_work,
                    go,
                    gralora_a_work,
                    gralora_b_work,
                    use_triton=ctx.use_triton,
                )

        if grad_x is not None:
            grad_x = grad_x.reshape(ctx.input_shape).to(ctx.input_dtype)
        if grad_a is not None:
            grad_a = grad_a.to(ctx.a_dtype)
        if grad_b is not None:
            grad_b = grad_b.to(ctx.b_dtype)
        return grad_x, grad_a, grad_b, None


class _InstantMergedGraloraFunction(torch.autograd.Function):
    @staticmethod
    def forward(  # type: ignore[override]
        ctx,
        x: Tensor,
        merged_weight: Tensor,
        bias: Tensor | None,
        gralora_a: Tensor,
        gralora_b: Tensor,
        scaling: float,
        use_triton: bool,
    ) -> Tensor:
        output = F.linear(x, merged_weight, bias)
        ctx.input_shape = tuple(x.shape)
        ctx.input_dtype = x.dtype
        ctx.weight_dtype = merged_weight.dtype
        ctx.a_dtype = gralora_a.dtype
        ctx.b_dtype = gralora_b.dtype
        ctx.has_bias = bias is not None
        ctx.scaling = float(scaling)
        ctx.use_triton = bool(use_triton)
        ctx.save_for_backward(x.contiguous(), merged_weight, gralora_a, gralora_b)
        return output

    @staticmethod
    def backward(ctx, grad_output: Tensor):  # type: ignore[override]
        x, merged_weight, gralora_a, gralora_b = ctx.saved_tensors
        grad_x = grad_bias = grad_a = grad_b = None

        if ctx.needs_input_grad[0]:
            go_for_x = grad_output.to(ctx.weight_dtype)
            grad_x = go_for_x.matmul(merged_weight.to(ctx.weight_dtype)).reshape(
                ctx.input_shape
            ).to(ctx.input_dtype)

        if ctx.needs_input_grad[3] or ctx.needs_input_grad[4]:
            work_dtype = gralora_a.dtype if gralora_a.is_floating_point() else grad_output.dtype
            _grad_x, grad_a, grad_b = _gralora_grads_from_x_go(
                x.to(work_dtype),
                grad_output.to(work_dtype),
                gralora_a.to(work_dtype),
                gralora_b.to(work_dtype),
                use_triton=ctx.use_triton,
            )
            scale = float(ctx.scaling)
            if grad_a is not None:
                grad_a = grad_a.mul(scale).to(ctx.a_dtype)
            if grad_b is not None:
                grad_b = grad_b.mul(scale).to(ctx.b_dtype)

        if ctx.has_bias and ctx.needs_input_grad[2]:
            reduce_dims = tuple(range(grad_output.ndim - 1))
            grad_bias = grad_output.sum(dim=reduce_dims)

        return grad_x, None, grad_bias, grad_a, grad_b, None, None


class _AttachGraloraAdapterGradient(torch.autograd.Function):
    @staticmethod
    def forward(  # type: ignore[override]
        ctx,
        result: Tensor,
        x: Tensor,
        gralora_a: Tensor,
        gralora_b: Tensor,
        scaling: float,
        use_triton: bool,
    ) -> Tensor:
        ctx.a_dtype = gralora_a.dtype
        ctx.b_dtype = gralora_b.dtype
        ctx.scaling = float(scaling)
        ctx.use_triton = bool(use_triton)
        ctx.save_for_backward(x.contiguous(), gralora_a, gralora_b)
        return result

    @staticmethod
    def backward(ctx, grad_output: Tensor):  # type: ignore[override]
        x, gralora_a, gralora_b = ctx.saved_tensors
        grad_a = grad_b = None
        if ctx.needs_input_grad[2] or ctx.needs_input_grad[3]:
            work_dtype = gralora_a.dtype if gralora_a.is_floating_point() else grad_output.dtype
            grad_a, grad_b = _gralora_param_grads_from_x_go(
                x.to(work_dtype),
                grad_output.to(work_dtype),
                gralora_a.to(work_dtype),
                gralora_b.to(work_dtype),
                use_triton=ctx.use_triton,
            )
            scale = float(ctx.scaling)
            grad_a = grad_a.mul(scale).to(ctx.a_dtype)
            grad_b = grad_b.mul(scale).to(ctx.b_dtype)
        return grad_output, None, grad_a, grad_b, None, None


class _AttachLowpassGraloraAdapterGradient(torch.autograd.Function):
    @staticmethod
    def forward(  # type: ignore[override]
        ctx,
        result: Tensor,
        x: Tensor,
        gralora_a: Tensor,
        gralora_b: Tensor,
        scaling: float,
        use_triton: bool,
        projector_kind: str,
        chunk_size: int,
        keep: int,
        hadamard_backend: str,
    ) -> Tensor:
        ctx.a_dtype = gralora_a.dtype
        ctx.b_dtype = gralora_b.dtype
        ctx.scaling = float(scaling)
        ctx.use_triton = bool(use_triton)
        ctx.projector_kind = str(projector_kind)
        ctx.chunk_size = int(chunk_size)
        ctx.keep = int(keep)
        ctx.hadamard_backend = str(hadamard_backend)
        x_hat = _project_token_chunks(
            x,
            projector_kind=ctx.projector_kind,
            chunk_size=ctx.chunk_size,
            keep=ctx.keep,
            hadamard_backend=ctx.hadamard_backend,
        )
        ctx.save_for_backward(x_hat.contiguous(), gralora_a, gralora_b)
        return result

    @staticmethod
    def backward(ctx, grad_output: Tensor):  # type: ignore[override]
        x_hat, gralora_a, gralora_b = ctx.saved_tensors
        grad_a = grad_b = None
        if ctx.needs_input_grad[2] or ctx.needs_input_grad[3]:
            go_hat = _project_token_chunks(
                grad_output,
                projector_kind=ctx.projector_kind,
                chunk_size=ctx.chunk_size,
                keep=ctx.keep,
                hadamard_backend=ctx.hadamard_backend,
            )
            work_dtype = gralora_a.dtype if gralora_a.is_floating_point() else grad_output.dtype
            grad_a, grad_b = _gralora_param_grads_from_x_go(
                x_hat.to(work_dtype),
                go_hat.to(work_dtype),
                gralora_a.to(work_dtype),
                gralora_b.to(work_dtype),
                use_triton=ctx.use_triton,
            )
            scale = float(ctx.scaling)
            grad_a = grad_a.mul(scale).to(ctx.a_dtype)
            grad_b = grad_b.mul(scale).to(ctx.b_dtype)
        return grad_output, None, grad_a, grad_b, None, None, None, None, None, None


class _LowpassGraloraAdapterFunction(torch.autograd.Function):
    @staticmethod
    def forward(  # type: ignore[override]
        ctx,
        x: Tensor,
        gralora_a: Tensor,
        gralora_b: Tensor,
        use_triton: bool,
        projector_kind: str,
        chunk_size: int,
        keep: int,
        hadamard_backend: str,
    ) -> Tensor:
        output = _gralora_forward_from_x(x, gralora_a, gralora_b, use_triton=bool(use_triton))
        ctx.input_shape = tuple(x.shape)
        ctx.input_dtype = x.dtype
        ctx.a_dtype = gralora_a.dtype
        ctx.b_dtype = gralora_b.dtype
        ctx.use_triton = bool(use_triton)
        ctx.projector_kind = str(projector_kind)
        ctx.chunk_size = int(chunk_size)
        ctx.keep = int(keep)
        ctx.hadamard_backend = str(hadamard_backend)
        x_hat = _project_token_chunks(
            x,
            projector_kind=ctx.projector_kind,
            chunk_size=ctx.chunk_size,
            keep=ctx.keep,
            hadamard_backend=ctx.hadamard_backend,
        )
        ctx.save_for_backward(x_hat.contiguous(), gralora_a, gralora_b)
        return output

    @staticmethod
    def backward(ctx, grad_output: Tensor):  # type: ignore[override]
        x_hat, gralora_a, gralora_b = ctx.saved_tensors
        work_dtype = gralora_a.dtype if gralora_a.is_floating_point() else grad_output.dtype
        grad_x = grad_a = grad_b = None

        if ctx.needs_input_grad[0]:
            grad_x = _gralora_input_grad(
                grad_output.to(work_dtype),
                gralora_a.to(work_dtype),
                gralora_b.to(work_dtype),
                use_triton=ctx.use_triton,
            ).reshape(ctx.input_shape).to(ctx.input_dtype)

        if ctx.needs_input_grad[1] or ctx.needs_input_grad[2]:
            go_hat = _project_token_chunks(
                grad_output,
                projector_kind=ctx.projector_kind,
                chunk_size=ctx.chunk_size,
                keep=ctx.keep,
                hadamard_backend=ctx.hadamard_backend,
            )
            grad_a, grad_b = _gralora_param_grads_from_x_go(
                x_hat.to(work_dtype),
                go_hat.to(work_dtype),
                gralora_a.to(work_dtype),
                gralora_b.to(work_dtype),
                use_triton=ctx.use_triton,
            )
            grad_a = grad_a.to(ctx.a_dtype)
            grad_b = grad_b.to(ctx.b_dtype)

        return grad_x, grad_a, grad_b, None, None, None, None, None


class InstantLowpassLinear(torch.nn.Module):
    def __init__(self, linear: torch.nn.Linear, config: InstantLowpassConfig) -> None:
        super().__init__()
        self.config = config
        self.in_features = linear.in_features
        self.out_features = linear.out_features
        self.weight = linear.weight
        self.bias = linear.bias
        self.register_buffer("instant_calls", torch.zeros((), dtype=torch.long), persistent=False)
        self.register_buffer("fallback_calls", torch.zeros((), dtype=torch.long), persistent=False)

    @classmethod
    def from_linear(cls, linear: torch.nn.Linear, config: InstantLowpassConfig) -> "InstantLowpassLinear":
        return cls(linear, config)

    def _can_use_instant(self, x: Tensor) -> bool:
        min_hidden_dim = int(self.config.min_hidden_dim)
        return (
            self.config.enabled
            and torch.is_grad_enabled()
            and x.ndim >= 3
            and x.shape[-2] >= self.config.chunk_size
            and int(x.shape[-1]) >= min_hidden_dim
            and int(self.weight.shape[0]) >= min_hidden_dim
            and int(self.weight.shape[1]) >= min_hidden_dim
            and self.weight.requires_grad
            and self.config.exact_input_grad
        )

    def forward(self, x: Tensor) -> Tensor:
        if not self._can_use_instant(x):
            if not _torch_is_compiling():
                self.fallback_calls += 1
            return F.linear(x, self.weight, self.bias)
        if not _torch_is_compiling():
            self.instant_calls += 1
        return _InstantLowpassLinearFunction.apply(
            x,
            self.weight,
            self.bias,
            self.config.exact_input_grad,
        )


def replace_linear_with_instant_lowpass(
    model: torch.nn.Module,
    config: InstantLowpassConfig,
    module_filter: Callable[[str, torch.nn.Linear], bool] | None = None,
) -> list[str]:
    replaced: list[str] = []

    def visit(parent: torch.nn.Module, prefix: str) -> None:
        for child_name, child in list(parent.named_children()):
            full_name = f"{prefix}.{child_name}" if prefix else child_name
            if isinstance(child, InstantLowpassLinear):
                continue
            if isinstance(child, torch.nn.Linear):
                if module_filter is None or module_filter(full_name, child):
                    setattr(parent, child_name, InstantLowpassLinear.from_linear(child, config))
                    replaced.append(full_name)
                continue
            visit(child, full_name)

    visit(model, "")
    return replaced


def _is_supported_gralora_module(module: torch.nn.Module) -> bool:
    return all(
        hasattr(module, name)
        for name in (
            "base_layer",
            "gralora_A",
            "gralora_B",
            "gralora_dropout",
            "gralora_k",
            "hybrid_r",
            "active_adapters",
            "scaling",
        )
    )


def _gralora_dropout_is_identity(dropout: torch.nn.Module) -> bool:
    if isinstance(dropout, torch.nn.Identity):
        return True
    p = getattr(dropout, "p", None)
    return isinstance(p, (int, float)) and float(p) == 0.0


def _scaling_as_float(scaling: float | Tensor) -> float:
    if isinstance(scaling, Tensor):
        return float(scaling.detach().float().cpu())
    return float(scaling)


def _active_gralora_adapter(module: torch.nn.Module) -> str | None:
    adapters = [adapter for adapter in module.active_adapters if adapter in module.gralora_A.keys()]
    if len(adapters) != 1:
        return None
    return str(adapters[0])


def _merged_weight_is_current(
    module: torch.nn.Module,
    adapter: str,
    base_weight: Tensor,
    gralora_a: Tensor,
    gralora_b: Tensor,
    scaling: float,
) -> bool:
    merged_weight = getattr(module, "instant_lowpass_merged_weight", None)
    version = getattr(module, "instant_lowpass_merged_weight_version", None)
    expected_version = (
        adapter,
        int(getattr(base_weight, "_version", 0)),
        int(getattr(gralora_a, "_version", 0)),
        int(getattr(gralora_b, "_version", 0)),
        str(base_weight.device),
        str(base_weight.dtype),
        tuple(base_weight.shape),
        float(scaling),
    )
    return (
        isinstance(merged_weight, Tensor)
        and merged_weight.shape == base_weight.shape
        and merged_weight.device == base_weight.device
        and merged_weight.dtype == base_weight.dtype
        and version == expected_version
    )


@torch.no_grad()
def _refresh_gralora_module_merged_weight(
    module: torch.nn.Module,
    adapter: str,
    *,
    force: bool = False,
) -> bool:
    base_layer = module.base_layer
    base_weight = getattr(base_layer, "weight", None)
    if not isinstance(base_weight, Tensor):
        return False
    gralora_a = module.gralora_A[adapter]
    gralora_b = module.gralora_B[adapter]
    scaling = _scaling_as_float(module.scaling[adapter])
    expected_version = (
        adapter,
        int(getattr(base_weight, "_version", 0)),
        int(getattr(gralora_a, "_version", 0)),
        int(getattr(gralora_b, "_version", 0)),
        str(base_weight.device),
        str(base_weight.dtype),
        tuple(base_weight.shape),
        float(scaling),
    )
    if not force and _merged_weight_is_current(module, adapter, base_weight, gralora_a, gralora_b, scaling):
        return True

    delta = _gralora_delta_weight(
        gralora_a.detach().to(dtype=base_weight.dtype, device=base_weight.device),
        gralora_b.detach().to(dtype=base_weight.dtype, device=base_weight.device),
    )
    if delta.shape != base_weight.shape:
        raise ValueError(
            f"merged GraLoRA delta shape {tuple(delta.shape)} does not match base weight "
            f"{tuple(base_weight.shape)}"
        )
    current = getattr(module, "instant_lowpass_merged_weight", None)
    if (
        isinstance(current, Tensor)
        and current.shape == base_weight.shape
        and current.device == base_weight.device
        and current.dtype == base_weight.dtype
    ):
        torch.add(base_weight.detach(), delta, alpha=scaling, out=current)
    elif "instant_lowpass_merged_weight" in getattr(module, "_buffers", {}):
        merged = torch.empty_like(base_weight, memory_format=torch.contiguous_format)
        torch.add(base_weight.detach(), delta, alpha=scaling, out=merged)
        module._buffers["instant_lowpass_merged_weight"] = merged
    else:
        merged = torch.empty_like(base_weight, memory_format=torch.contiguous_format)
        torch.add(base_weight.detach(), delta, alpha=scaling, out=merged)
        module.register_buffer("instant_lowpass_merged_weight", merged, persistent=False)
    module.instant_lowpass_merged_weight_version = expected_version
    return True


def refresh_gralora_merged_weights(model: torch.nn.Module, *, force: bool = False) -> int:
    refreshed = 0
    for module in model.modules():
        if not hasattr(module, "instant_lowpass_original_forward"):
            continue
        adapter = _active_gralora_adapter(module)
        if adapter is None:
            continue
        if _refresh_gralora_module_merged_weight(module, adapter, force=force):
            refreshed += 1
    return refreshed


def _instant_lowpass_gralora_forward(self, x: Tensor, *args, **kwargs) -> Tensor:
    config: InstantLowpassConfig = self.instant_lowpass_config
    original_forward = self.instant_lowpass_original_forward
    previous_dtype = x.dtype

    if args or kwargs or self.disable_adapters or self.merged or x.ndim not in {2, 3}:
        if not _torch_is_compiling():
            self.instant_lowpass_fallback_calls += 1
        return original_forward(x, *args, **kwargs)

    x_is_2d = x.ndim == 2
    work_x = x.unsqueeze(1) if x_is_2d else x
    min_hidden_dim = int(config.min_hidden_dim)
    if (
        work_x.shape[-2] < config.chunk_size
        or int(work_x.shape[-2]) % int(config.chunk_size) != 0
        or int(work_x.shape[-1]) < min_hidden_dim
    ):
        if not _torch_is_compiling():
            self.instant_lowpass_fallback_calls += 1
        return original_forward(x, *args, **kwargs)

    base_layer = self.base_layer
    base_weight = getattr(base_layer, "weight", None)
    bias = getattr(base_layer, "bias", None)
    if not isinstance(base_weight, Tensor) or int(base_weight.shape[0]) < min_hidden_dim:
        if not _torch_is_compiling():
            self.instant_lowpass_fallback_calls += 1
        return original_forward(x, *args, **kwargs)

    active_adapter = _active_gralora_adapter(self)
    if active_adapter is None:
        if not _torch_is_compiling():
            self.instant_lowpass_fallback_calls += 1
        return original_forward(x, *args, **kwargs)

    use_triton = config.hadamard_backend != "dense"
    hybrid_r = int(self.hybrid_r[active_adapter])
    dropout = self.gralora_dropout[active_adapter]
    if hybrid_r > 0 or not _gralora_dropout_is_identity(dropout):
        if not _torch_is_compiling():
            self.instant_lowpass_fallback_calls += 1
        return original_forward(x, *args, **kwargs)

    gralora_a = self.gralora_A[active_adapter]
    gralora_b = self.gralora_B[active_adapter]
    if int(gralora_a.shape[1]) < min_hidden_dim or int(gralora_b.shape[2]) < min_hidden_dim:
        if not _torch_is_compiling():
            self.instant_lowpass_fallback_calls += 1
        return original_forward(x, *args, **kwargs)

    scaling = _scaling_as_float(self.scaling[active_adapter])
    if config.parameter_gradient == "projected_lowpass":
        result = base_layer(x)
        adapter_output = _LowpassGraloraAdapterFunction.apply(
            work_x.to(gralora_a.dtype),
            gralora_a,
            gralora_b,
            use_triton,
            config.projector_kind,
            config.chunk_size,
            config.keep,
            config.hadamard_backend,
        )
        if x_is_2d:
            adapter_output = adapter_output.squeeze(1)
        result = result + scaling * adapter_output.to(result.dtype)
        if not _torch_is_compiling():
            self.instant_lowpass_instant_calls += 1
        return result.to(previous_dtype)

    if not _torch_is_compiling() and not _merged_weight_is_current(
        self,
        active_adapter,
        base_weight,
        gralora_a,
        gralora_b,
        scaling,
    ):
        _refresh_gralora_module_merged_weight(self, active_adapter)
    merged_weight = getattr(self, "instant_lowpass_merged_weight", None)
    if not isinstance(merged_weight, Tensor):
        if not _torch_is_compiling():
            self.instant_lowpass_fallback_calls += 1
        return original_forward(x, *args, **kwargs)

    work_x = work_x.to(merged_weight.dtype)
    result = F.linear(work_x, merged_weight, bias)
    if torch.is_grad_enabled():
        if config.parameter_gradient == "projected_lowpass":
            result = _AttachLowpassGraloraAdapterGradient.apply(
                result,
                work_x,
                gralora_a,
                gralora_b,
                scaling,
                use_triton,
                config.projector_kind,
                config.chunk_size,
                config.keep,
                config.hadamard_backend,
            )
        else:
            result = _AttachGraloraAdapterGradient.apply(
                result,
                work_x,
                gralora_a,
                gralora_b,
                scaling,
                use_triton,
            )
    if x_is_2d:
        result = result.squeeze(1)

    if not _torch_is_compiling():
        self.instant_lowpass_instant_calls += 1
    return result.to(previous_dtype)


def patch_gralora_with_instant_lowpass(model: torch.nn.Module, config: InstantLowpassConfig) -> list[str]:
    patched: list[str] = []
    for name, module in model.named_modules():
        if not name or not _is_supported_gralora_module(module):
            continue
        if hasattr(module, "instant_lowpass_original_forward"):
            continue
        module.instant_lowpass_config = config
        module.instant_lowpass_original_forward = module.forward
        module.register_buffer(
            "instant_lowpass_instant_calls",
            torch.zeros((), dtype=torch.long),
            persistent=False,
        )
        module.register_buffer(
            "instant_lowpass_fallback_calls",
            torch.zeros((), dtype=torch.long),
            persistent=False,
        )
        module.forward = types.MethodType(_instant_lowpass_gralora_forward, module)
        patched.append(name)
    return patched


def collect_instant_lowpass_stats(model: torch.nn.Module) -> dict[str, int]:
    modules = [module for module in model.modules() if isinstance(module, InstantLowpassLinear)]
    patched_gralora = [
        module
        for module in model.modules()
        if hasattr(module, "instant_lowpass_original_forward")
        and hasattr(module, "instant_lowpass_instant_calls")
    ]
    return {
        "instant_lowpass_wrapped_modules": len(modules),
        "instant_lowpass_patched_gralora_modules": len(patched_gralora),
        "instant_lowpass_instant_calls": sum(int(module.instant_calls.detach().cpu()) for module in modules)
        + sum(int(module.instant_lowpass_instant_calls.detach().cpu()) for module in patched_gralora),
        "instant_lowpass_fallback_calls": sum(int(module.fallback_calls.detach().cpu()) for module in modules)
        + sum(int(module.instant_lowpass_fallback_calls.detach().cpu()) for module in patched_gralora),
    }
