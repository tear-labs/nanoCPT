"""Hadamard low-pass activation compression via `saved_tensors_hooks`.

Adapted from the `nanoCPT-hadamard-lowpass` worktree, simplified for our
full-fine-tune trainer. The approach:

- Install a model-wide `torch.autograd.graph.saved_tensors_hooks(pack,
  unpack)` context around the training step. Every `save_for_backward`
  call across the model (attention, MLP, anywhere) is intercepted.
- `pack(tensor)`: if the tensor looks like a hidden-state activation
  with a sequence axis, Hadamard-project chunks of `chunk_size` tokens
  along that axis and keep only the top-`keep` low-frequency
  coefficients (50 % compression at keep=32, chunk=64). Returns a tuple
  the autograd engine stores instead of the raw tensor.
- `unpack(packed)`: inverse-Hadamard reconstructs an approximation of
  the original tensor when backward needs it.

Why this beats per-Linear wrapping:
- No custom `torch.autograd.Function` → no dynamo graph break per
  Linear (the previous design added 144 graph breaks per forward).
- One context manager wraps every save across the model — including
  attention K/V, MLP intermediates, anything that gets saved — without
  having to enumerate or rewrite modules.
- Lives entirely at the autograd-engine level, outside the compiled
  forward/backward graph. `torch.compile` traces normal code; autograd
  calls our pack/unpack on the side.

When combined with `--gradient-checkpointing false`, the compressed
activations actually replace what's held across the forward/backward
boundary — that's where the throughput win comes from (no recompute,
fits in memory thanks to compression).
"""

from __future__ import annotations

import contextlib
import math
import os
from dataclasses import dataclass
from typing import Any, Callable

import torch
import torch.nn.functional as F
from torch import Tensor


_TRITON_PIECEWISE_PROJECT: Callable[[Tensor, Tensor], Tensor] | None = None
_TRITON_PIECEWISE_PROJECT_IMPORT_ERROR: Exception | None = None
_PROJECTOR_CACHE: dict[tuple[str, int, int, str, int, str], Tensor] = {}


@dataclass(frozen=True)
class LowpassConfig:
    projector_kind: str = "hadamard"
    chunk_size: int = 64
    keep: int = 32
    min_hidden_dim: int = 64
    max_hidden_dim: int = 0  # 0 means no upper bound
    hadamard_backend: str = "auto"
    enabled: bool = True

    def __post_init__(self) -> None:
        projector_kind = str(self.projector_kind).lower()
        hadamard_backend = str(self.hadamard_backend).lower().replace("_", "-")
        if hadamard_backend in {"fast", "triton"}:
            hadamard_backend = "piecewise"
        object.__setattr__(self, "projector_kind", projector_kind)
        object.__setattr__(self, "hadamard_backend", hadamard_backend)
        if projector_kind not in {"hadamard", "dct", "haar"}:
            raise ValueError(f"unknown projector_kind {self.projector_kind!r}")
        if hadamard_backend not in {"auto", "piecewise", "dense"}:
            raise ValueError(f"unknown hadamard_backend {self.hadamard_backend!r}")
        if self.chunk_size < 1:
            raise ValueError("chunk_size must be positive")
        if self.keep < 1 or self.keep > self.chunk_size:
            raise ValueError("keep must be in [1, chunk_size]")
        if self.min_hidden_dim < 0:
            raise ValueError("min_hidden_dim must be non-negative")
        if projector_kind in {"hadamard", "haar"} and not _is_power_of_two(self.chunk_size):
            raise ValueError(f"{projector_kind} requires power-of-two chunk_size")


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


def _load_triton_piecewise_project() -> Callable[[Tensor, Tensor], Tensor]:
    global _TRITON_PIECEWISE_PROJECT_IMPORT_ERROR, _TRITON_PIECEWISE_PROJECT
    if _TRITON_PIECEWISE_PROJECT is not None:
        return _TRITON_PIECEWISE_PROJECT
    if _TRITON_PIECEWISE_PROJECT_IMPORT_ERROR is not None:
        raise RuntimeError("lowpass token projection Triton kernel is unavailable") from (
            _TRITON_PIECEWISE_PROJECT_IMPORT_ERROR
        )
    try:
        from lowpass_triton import piecewise_project
    except Exception as exc:  # pragma: no cover - depends on optional CUDA/Triton runtime.
        _TRITON_PIECEWISE_PROJECT_IMPORT_ERROR = exc
        raise RuntimeError("lowpass token projection Triton kernel is unavailable") from exc
    _TRITON_PIECEWISE_PROJECT = piecewise_project
    return piecewise_project


def _project_chunks(
    chunks: Tensor,
    projector_kind: str,
    keep: int,
    hadamard_backend: str,
) -> Tensor:
    """Project [n_chunks, chunk_size, hidden] → [n_chunks, keep, hidden]."""
    if (
        chunks.is_cuda
        and projector_kind in {"hadamard", "haar"}
        and hadamard_backend != "dense"
        and os.environ.get("LOWPASS_DISABLE_TRITON", "0") != "1"
    ):
        try:
            coefficients_and_segment_len = _piecewise_projector_coefficients(
                projector_kind,
                int(chunks.shape[-2]),
                int(keep),
                chunks.device,
                chunks.dtype,
            )
            if coefficients_and_segment_len is not None:
                coefficients, segment_len = coefficients_and_segment_len
                return _load_triton_piecewise_project()(
                    chunks.contiguous(), coefficients, segment_len=segment_len
                )
        except Exception:
            if os.environ.get("LOWPASS_REQUIRE_TRITON", "0") == "1":
                raise
    projector = _fixed_projector(
        projector_kind, int(chunks.shape[-2]), int(keep), chunks.device, chunks.dtype
    )
    return torch.einsum("rl,nlc->nrc", projector, chunks)


def _reconstruct_chunks(
    lowpass: Tensor,
    chunk_size: int,
    projector_kind: str,
) -> Tensor:
    """Inverse-project [n_chunks, keep, hidden] → [n_chunks, chunk_size, hidden]."""
    projector = _fixed_projector(
        projector_kind, int(chunk_size), int(lowpass.shape[-2]), lowpass.device, lowpass.dtype
    )
    # projector is [keep, chunk_size]; inverse is projector.T (since orthonormal).
    return torch.einsum("rl,nrc->nlc", projector, lowpass).contiguous()


_LOWPASS_TAG = "lowpass-activation-v1"


def _make_pack_unpack(config: LowpassConfig, seq_len: int):
    """Build pack/unpack functions closed over the run's config and seq_len."""

    chunk_size = int(config.chunk_size)
    keep = int(config.keep)
    projector_kind = config.projector_kind
    hadamard_backend = config.hadamard_backend
    min_hidden_dim = int(config.min_hidden_dim)
    max_hidden_dim = int(config.max_hidden_dim)
    expected_token_axes = {seq_len, seq_len - 1}  # HF sometimes drops the last position
    log_first_n_shapes = int(os.environ.get("LOWPASS_LOG_SHAPES", "0"))
    seen_shapes: dict[tuple, int] = {}

    def pack(tensor: Tensor) -> Any:
        # Conservative filter: only compress tensors that look like
        # [batch, seq, hidden] hidden states. Skip everything else
        # (parameter leaves, scalars, attention masks, kv caches, etc.).
        # Only compress the canonical [batch, seq, hidden] residual/MLP
        # shape. Higher-rank shapes (attention scores [B,H,S,S], KV cache
        # [B,H,S,D], etc.) get falsely matched if we permit ndim>3 — those
        # tensors don't behave well under per-token Hadamard projection.
        if (
            not config.enabled
            or not tensor.is_cuda
            or not tensor.is_floating_point()
            or tensor.ndim != 3
            or tensor.shape[-1] < min_hidden_dim
            or (max_hidden_dim > 0 and tensor.shape[-1] > max_hidden_dim)
            or tensor.shape[1] not in expected_token_axes
        ):
            return tensor
        token_axis = 1
        token_count = int(tensor.shape[token_axis])
        if token_count < chunk_size or token_count % chunk_size != 0:
            return tensor
        # Reshape to [prefix, n_chunks, chunk_size, suffix...] then to
        # [prefix*n_chunks, chunk_size, suffix_numel] for projection.
        prefix_shape = tuple(tensor.shape[:token_axis])
        suffix_shape = tuple(tensor.shape[token_axis + 1 :])
        suffix_numel = int(math.prod(suffix_shape))
        if suffix_numel < min_hidden_dim:
            return tensor
        prefix_numel = int(math.prod(prefix_shape))
        n_chunks = token_count // chunk_size
        if log_first_n_shapes:
            shape_key = tuple(tensor.shape)
            if shape_key not in seen_shapes and len(seen_shapes) < log_first_n_shapes:
                print(f"LOWPASS_SHAPE_FIRST_SEEN: {shape_key} dtype={tensor.dtype}", flush=True)
            seen_shapes[shape_key] = seen_shapes.get(shape_key, 0) + 1
        with torch.no_grad(), torch.autocast("cuda", enabled=False):
            view = tensor.reshape(prefix_numel, n_chunks, chunk_size, suffix_numel)
            chunks = view.reshape(prefix_numel * n_chunks, chunk_size, suffix_numel).contiguous()
            lowpass = _project_chunks(
                chunks, projector_kind, keep, hadamard_backend
            ).contiguous()
        return (
            _LOWPASS_TAG,
            lowpass,
            tuple(tensor.shape),
            prefix_shape,
            suffix_shape,
            token_axis,
            token_count,
            tensor.dtype,
        )

    def unpack(packed: Any) -> Tensor:
        if not (isinstance(packed, tuple) and packed and packed[0] == _LOWPASS_TAG):
            return packed
        (
            _tag,
            lowpass,
            original_shape,
            prefix_shape,
            suffix_shape,
            token_axis,
            token_count,
            original_dtype,
        ) = packed
        with torch.no_grad(), torch.autocast("cuda", enabled=False):
            chunks = _reconstruct_chunks(lowpass, chunk_size, projector_kind)
            prefix_numel = int(math.prod(prefix_shape))
            n_chunks = token_count // chunk_size
            suffix_numel = int(math.prod(suffix_shape))
            restored = chunks.reshape(prefix_numel, n_chunks * chunk_size, suffix_numel)
            restored = restored.reshape(*prefix_shape, token_count, *suffix_shape)
            return restored.reshape(original_shape).to(original_dtype)

    return pack, unpack


def activation_save_context(config: LowpassConfig, seq_len: int):
    """Return a context manager that installs lowpass activation packing.

    Usage:
        with activation_save_context(config, seq_len):
            out = model(input_ids=..., ...)
            out.loss.backward()
    """
    if not config.enabled:
        return contextlib.nullcontext()
    pack, unpack = _make_pack_unpack(config, seq_len)
    return torch.autograd.graph.saved_tensors_hooks(pack, unpack)
