"""modded-continued-training: a Modal H100 speedrun for fine-tuning.

Track 1 trains Qwen3.5-4B-Base on UltraChat general SFT data by default and
scores the run by the drop in heldout assistant-only loss. The legacy FineMath
continued-pretraining path remains available with ``--data-mode cpt``.

Track 1: 30-minute budget (default)
Track 2: 5-minute sprint
Track 3: 2-hour endurance
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import uuid
from pathlib import Path
from typing import Any, Callable, Literal

import modal


APP_NAME = "modded-continued-training"
CACHE_MOUNT = Path("/cache")
HF_CACHE = CACHE_MOUNT / "huggingface"

DEFAULT_MODEL_ID = "Qwen/Qwen3.5-4B-Base"
DEFAULT_MODEL_REVISION = "1001bb4d826a52d1f399e183466143f4da7b741b"
DEFAULT_DATASET_ID = "HuggingFaceTB/finemath"
DEFAULT_DATASET_CONFIG = "finemath-4plus"
DEFAULT_DATASET_REVISION = "e92b25a616738fe95dc186b64dfb19f9c8525594"
DEFAULT_SFT_DATASET_ID = "HuggingFaceH4/ultrachat_200k"
DEFAULT_SFT_DATASET_CONFIG = ""
DEFAULT_SFT_DATASET_REVISION = "8049631c405ae6576f93f445c6b8166f76f5505a"
DEFAULT_SFT_TRAIN_SPLIT = "train_sft"
DEFAULT_SFT_EVAL_SPLIT = "test_sft"
DEFAULT_EFFECTIVE_TOKENS_PER_STEP = 32_768
DEFAULT_LORA_MICRO_BATCH_SIZE = 8
DEFAULT_FULL_MICRO_BATCH_SIZE = 1
DEFAULT_EVAL_MICRO_BATCH_SIZE = 2
DEFAULT_GRALORA_K = 2
DEFAULT_ADAPTER_MODE = "gralora"
DEFAULT_LORAPLUS_LR_RATIO = 16.0
DEFAULT_LORA_EVA_RHO = 2.0
DEFAULT_MUON_QUANT_BLOCK_SIZE = 2048
DEFAULT_NORMUON_BETA2 = 0.95
DEFAULT_NORMUON_EPS = 1.0e-8
DEFAULT_SEQUENCE_PACKING = True
DEFAULT_PACKING_STRATEGY = "stream_concat_no_padding"

DATA_MODE_CHOICES = {"sft", "cpt"}
ADAPTER_MODE_CHOICES = {"lora", "lora_ga", "gralora"}
OPTIMIZER_CHOICES = {
    "auto",
    "adamw8bit",
    "adamw_fused",
    "muon",
    "muon8",
    "normuon",
    "loraplus_adamw",
    "loraplus_adamw8bit",
    "lorafa",
}
LR_SCHEDULE_CHOICES = {"constant", "linear", "cosine", "wsd"}
MUON_LR_ADJUSTMENT_CHOICES = {"original", "match_rms_adamw"}
LORA_INIT_CHOICES = {"default", "gaussian", "pissa", "olora", "eva", "orthogonal", "lora_ga"}
LORA_GA_DIRECTION_CHOICES = {"ArBr", "A2rBr", "ArB2r", "random"}
LORA_GA_SCALE_CHOICES = {"stable", "weight_svd", "gd_scale", "unit"}
ACTIVATION_COMPRESSION_MODE_CHOICES = {"off", "instant-linear"}
INSTANT_PROJECTOR_KIND_CHOICES = {"hadamard", "dct", "haar"}
INSTANT_TARGET_MODULE_CHOICES = {"trainable", "adapter", "all"}
INSTANT_HADAMARD_BACKEND_CHOICES = {"auto", "piecewise", "triton", "fast", "dense"}
INSTANT_PARAMETER_GRADIENT_CHOICES = {"exact", "projected_lowpass"}

PEAK_GPU_STAT_KEYS = {
    "cuda_max_memory_allocated_gib": "peak_cuda_memory_allocated_gib",
    "cuda_max_memory_reserved_gib": "peak_cuda_memory_reserved_gib",
    "cuda_max_memory_allocated_fraction": "peak_cuda_memory_allocated_fraction",
    "cuda_max_memory_reserved_fraction": "peak_cuda_memory_reserved_fraction",
    "gpu_utilization_percent": "peak_gpu_utilization_percent",
    "gpu_memory_utilization_percent": "peak_gpu_memory_utilization_percent",
    "gpu_memory_used_gib": "peak_gpu_memory_used_gib",
    "gpu_memory_used_fraction": "peak_gpu_memory_used_fraction",
    "gpu_power_watts": "peak_gpu_power_watts",
}

TRACKS: dict[str, dict[str, Any]] = {
    "1": {"name": "30min", "default_minutes": 30.0, "record_dir": "records/track_1_30min"},
    "2": {"name": "5min", "default_minutes": 5.0, "record_dir": "records/track_2_5min"},
    "3": {"name": "2hr", "default_minutes": 120.0, "record_dir": "records/track_3_2hr"},
}

SFT_ROLE_MAP = {
    "system": "system",
    "human": "user",
    "prompter": "user",
    "user": "user",
    "gpt": "assistant",
    "assistant": "assistant",
    "tool": "tool",
}


def _iter_sft_turns(row: dict[str, Any]) -> list[tuple[str, Any]] | None:
    conversations = row.get("conversations")
    if isinstance(conversations, list):
        if not all(isinstance(turn, dict) for turn in conversations):
            return None
        return [(turn.get("from"), turn.get("value")) for turn in conversations]

    messages = row.get("messages")
    if isinstance(messages, list):
        if not all(isinstance(turn, dict) for turn in messages):
            return None
        return [(turn.get("role"), turn.get("content")) for turn in messages]

    return None


def _render_sft_row(
    row: dict[str, Any],
    tokenize_piece,
    seq_len: int,
) -> tuple[list[int], list[int]] | None:
    turns = _iter_sft_turns(row)
    if not turns:
        return None

    input_ids: list[int] = []
    labels: list[int] = []
    supervised_tokens = 0
    for source_role, value in turns:
        if not isinstance(source_role, str):
            return None
        role = SFT_ROLE_MAP.get(source_role.lower())
        if role is None:
            return None

        if not isinstance(value, str):
            value = "" if value is None else json.dumps(value, sort_keys=True)

        prefix_ids = tokenize_piece(f"<|im_start|>{role}\n")
        content_ids = tokenize_piece(value)
        suffix_ids = tokenize_piece("<|im_end|>\n")
        input_ids.extend(prefix_ids)
        input_ids.extend(content_ids)
        input_ids.extend(suffix_ids)

        if role == "assistant":
            labels.extend([-100] * len(prefix_ids))
            labels.extend(content_ids)
            labels.extend(suffix_ids)
            supervised_tokens += len(content_ids) + len(suffix_ids)
        else:
            labels.extend([-100] * (len(prefix_ids) + len(content_ids) + len(suffix_ids)))

    if not input_ids or supervised_tokens == 0:
        return None
    if len(input_ids) != len(labels):
        raise RuntimeError("SFT renderer produced misaligned input_ids and labels")
    if len(input_ids) > seq_len:
        input_ids = input_ids[-seq_len:]
        labels = labels[-seq_len:]
        if not any(label != -100 for label in labels):
            return None
    return input_ids, labels


def _resolve_dataset_defaults(
    data_mode: str,
    dataset_id: str,
    dataset_config: str,
    dataset_revision: str,
) -> tuple[str, str, str]:
    if not dataset_id:
        if data_mode == "sft":
            return DEFAULT_SFT_DATASET_ID, DEFAULT_SFT_DATASET_CONFIG, DEFAULT_SFT_DATASET_REVISION
        return DEFAULT_DATASET_ID, DEFAULT_DATASET_CONFIG, DEFAULT_DATASET_REVISION

    legacy_defaults = (DEFAULT_DATASET_ID, DEFAULT_DATASET_CONFIG, DEFAULT_DATASET_REVISION)
    if data_mode == "sft" and (dataset_id, dataset_config, dataset_revision) == legacy_defaults:
        return DEFAULT_SFT_DATASET_ID, DEFAULT_SFT_DATASET_CONFIG, DEFAULT_SFT_DATASET_REVISION
    return dataset_id, dataset_config, dataset_revision


def _supervised_token_count(labels) -> int:
    if isinstance(labels, list):
        if labels and isinstance(labels[0], list):
            return sum(1 for row in labels for label in row[1:] if label != -100)
        return sum(1 for label in labels[1:] if label != -100)
    shifted = labels[..., 1:]
    return int((shifted != -100).sum())


def _next_multiple(value: int, multiple: int) -> int:
    if multiple <= 0:
        return value
    return ((value + multiple - 1) // multiple) * multiple


def _pad_sft_row_to_block_multiple(
    input_ids: list[int],
    labels: list[int],
    *,
    block_size: int,
    pad_token_id: int,
    seq_len: int,
) -> tuple[list[int], list[int], list[int], int]:
    if len(input_ids) != len(labels):
        raise RuntimeError("SFT row has misaligned input_ids and labels")
    row_ids = list(input_ids)
    row_labels = list(labels)
    attention_mask = [1] * len(row_ids)
    if block_size <= 0:
        return row_ids, row_labels, attention_mask, 0
    if seq_len % block_size != 0:
        raise ValueError("--seq-len must be a multiple of the SFT pack block size")
    if len(row_ids) > seq_len:
        raise ValueError("SFT row must be truncated before block padding")
    padded_len = _next_multiple(len(row_ids), block_size)
    if padded_len > seq_len:
        raise ValueError("SFT block padding would exceed --seq-len")
    pad_count = padded_len - len(row_ids)
    if pad_count:
        row_ids.extend([pad_token_id] * pad_count)
        row_labels.extend([-100] * pad_count)
        attention_mask.extend([0] * pad_count)
    return row_ids, row_labels, attention_mask, pad_count


def _assert_no_cross_sequence_lowpass_segments(segment_ids: list[int], block_size: int) -> None:
    if block_size <= 0:
        return
    if len(segment_ids) % block_size != 0:
        raise RuntimeError("low-pass segment buffer is not block aligned")
    for start in range(0, len(segment_ids), block_size):
        seen = {segment for segment in segment_ids[start : start + block_size] if segment >= 0}
        if len(seen) > 1:
            raise RuntimeError(
                "low-pass block crosses SFT sequence boundary "
                f"at token range [{start}, {start + block_size})"
            )


app = modal.App(APP_NAME)
cache_volume = modal.Volume.from_name("modded-continued-training-cache", create_if_missing=True)
wandb_secret_values = {"WANDB_API_KEY": os.environ.get("WANDB_API_KEY", "")}
if os.environ.get("WANDB_BASE_URL"):
    wandb_secret_values["WANDB_BASE_URL"] = os.environ["WANDB_BASE_URL"]
wandb_env_secret = modal.Secret.from_dict(wandb_secret_values)

image = (
    modal.Image.from_registry("nvidia/cuda:12.8.1-devel-ubuntu22.04", add_python="3.12")
    .entrypoint([])
    .env(
        {
            "CC": "gcc",
            "CXX": "g++",
            "CUDAHOSTCXX": "g++",
            "FLASH_ATTENTION_FORCE_CXX11_ABI": "TRUE",
            "MAX_JOBS": "8",
            "NVCC_THREADS": "2",
            "PYTORCH_ALLOC_CONF": "expandable_segments:True",
            "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True",
            "TOKENIZERS_PARALLELISM": "true",
            "TORCH_CUDA_ARCH_LIST": "9.0",
        }
    )
    .apt_install("build-essential", "git", "ninja-build")
    .pip_install("torch==2.8.0", "packaging", "wheel")
    .pip_install(
        "accelerate>=1.0.0",
        "bitsandbytes>=0.46.0",
        "causal-conv1d==1.6.2.post1",
        "datasets>=3.0.0",
        "flash-attn==2.8.3",
        "flash-linear-attention==0.5.0",
        "hf_transfer>=0.1.9",
        "huggingface_hub>=0.30.0",
        "numpy>=2.0.0",
        "nvidia-ml-py>=12.560.30",
        "peft==0.19.1",
        "safetensors>=0.5.0",
        "tilelang",
        "tqdm>=4.66.0",
        "wandb>=0.18.0",
        "git+https://github.com/huggingface/transformers.git",
        extra_options="--no-build-isolation",
    )
    .add_local_file(Path(__file__).with_name("instant_lowpass.py"), "/root/instant_lowpass.py")
    .add_local_file(Path(__file__).with_name("instant_lowpass_triton.py"), "/root/instant_lowpass_triton.py")
)


def _json_default(value: Any) -> Any:
    if hasattr(value, "item"):
        return value.item()
    return str(value)


def _slugify(value: str) -> str:
    slug = re.sub(r"[^A-Za-z0-9._-]+", "_", value.strip()).strip("._-")
    return slug or "record"


def _format_record_text(summary: dict[str, Any]) -> str:
    return (
        f"Track {summary['track']} ({summary['track_name']})\n"
        f"Description: {summary['record_description']}\n"
        f"Contributors: {summary['record_contributors']}\n"
        f"Date: {summary['record_date']}\n"
        f"Data mode: {summary.get('data_mode', 'cpt')}\n"
        f"Loss mask: {summary.get('loss_mask', 'all_tokens')}\n"
        f"Sequence packing: {summary.get('sequence_packing', DEFAULT_SEQUENCE_PACKING)} "
        f"({summary.get('packing_strategy', DEFAULT_PACKING_STRATEGY)})\n"
        f"Adapter mode: {summary.get('adapter_mode', DEFAULT_ADAPTER_MODE)}\n"
        f"Minutes: {summary['minutes']}\n"
        f"Eval loss drop: {summary['eval_loss_drop']:.6f}\n"
        f"Baseline eval loss: {summary['baseline_eval_loss']:.6f}\n"
        f"Final eval loss: {summary['final_eval_loss']:.6f}\n"
        f"Steps: {summary['steps']}\n"
        f"Tokens: {summary['tokens']:,}\n"
        f"Supervised tokens: {summary.get('supervised_tokens', summary['tokens']):,}\n"
        f"Eval supervised tokens: {summary.get('eval_supervised_tokens', 0):,}\n"
        f"Compile + warmup (untimed): {summary.get('elapsed_compile_warmup_seconds', 0.0):.1f}s\n"
        f"Elapsed budget: {summary['elapsed_budget_seconds']:.1f}s\n"
        f"Budget tokens/sec: {summary['tokens_per_second']:.0f}\n"
        f"Train-loop elapsed: {summary['elapsed_train_loop_seconds']:.1f}s\n"
        f"Train-loop tokens/sec: {summary['train_loop_tokens_per_second']:.0f}\n"
    )


def _write_local_record(summary: dict[str, Any], artifacts: dict[str, str]) -> Path:
    track = str(summary["track"])
    if track not in TRACKS:
        raise ValueError(f"record track must be one of: {', '.join(TRACKS.keys())}")

    record_name = f"{summary['record_date']}_{_slugify(str(summary['record_description']))}"
    record_dir = Path(TRACKS[track]["record_dir"]) / record_name
    if record_dir.exists():
        record_dir = record_dir.with_name(f"{record_dir.name}_{summary['run_id']}")
    record_dir.mkdir(parents=True, exist_ok=False)

    for filename in (
        "main.py",
        "instant_lowpass.py",
        "instant_lowpass_triton.py",
        "config.json",
        "summary.json",
        "record.txt",
        "metrics.jsonl",
    ):
        content = artifacts.get(filename)
        if not content and filename == "main.py":
            content = Path(__file__).read_text()
        if content is not None:
            (record_dir / filename).write_text(content)
    return record_dir


@app.function(
    image=image,
    gpu="H100",
    timeout=4 * 60 * 60,
    volumes={str(CACHE_MOUNT): cache_volume},
    secrets=[wandb_env_secret],
)
def run_track1(
    minutes: float = 30.0,
    seq_len: int = 4096,
    micro_batch_size: int = 0,
    grad_accum: int = 0,
    lr: float = 0.0,
    weight_decay: float = -1.0,
    warmup_steps: int = 20,
    eval_blocks: int = 64,
    eval_micro_batch_size: int = 0,
    seed: int = 1337,
    model_id: str = DEFAULT_MODEL_ID,
    model_revision: str = DEFAULT_MODEL_REVISION,
    dataset_id: str = "",
    dataset_config: str = "",
    dataset_revision: str = "",
    data_mode: str = "sft",
    sft_pack_block_size: int = 0,
    tuning_mode: Literal["lora", "full"] = "lora",
    adapter_mode: str = "",
    optimizer_name: Literal[
        "auto",
        "adamw8bit",
        "adamw_fused",
        "muon",
        "muon8",
        "normuon",
        "loraplus_adamw",
        "loraplus_adamw8bit",
        "lorafa",
    ] = "auto",
    gradient_checkpointing: Literal["auto", "true", "false"] = "auto",
    activation_compression_mode: Literal["off", "instant-linear"] = "off",
    instant_projector_kind: Literal["hadamard", "dct", "haar"] = "hadamard",
    instant_chunk_size: int = 64,
    instant_keep: int = 32,
    instant_min_hidden_dim: int = 64,
    instant_hadamard_backend: Literal["auto", "piecewise", "triton", "fast", "dense"] = "auto",
    instant_parameter_gradient: str = "projected_lowpass",
    instant_target_modules: Literal["trainable", "adapter", "all"] = "trainable",
    lora_r: int = 32,
    lora_alpha: int = 64,
    lora_dropout: float = 0.0,
    lora_target_modules: str = "all-linear",
    gralora_k: int = DEFAULT_GRALORA_K,
    lora_use_rslora: bool = True,
    lora_use_dora: bool = False,
    lora_init: str = "default",
    lora_eva_rho: float = DEFAULT_LORA_EVA_RHO,
    lora_eva_batches: int = 16,
    lora_ga_batches: int = 4,
    lora_ga_micro_batch_size: int = 1,
    lora_ga_direction: str = "ArB2r",
    lora_ga_scale: str = "stable",
    lora_ga_stable_gamma: int = 16,
    lora_ga_cache: bool = False,
    loraplus_lr_ratio: float = DEFAULT_LORAPLUS_LR_RATIO,
    loraplus_lr_embedding: float = 1.0e-6,
    muon_lr_adjustment: Literal["original", "match_rms_adamw"] = "match_rms_adamw",
    muon_quant_block_size: int = DEFAULT_MUON_QUANT_BLOCK_SIZE,
    normuon_beta2: float = DEFAULT_NORMUON_BETA2,
    normuon_eps: float = DEFAULT_NORMUON_EPS,
    lr_schedule: Literal["constant", "linear", "cosine", "wsd"] = "constant",
    lr_decay_fraction: float = 0.1,
    min_lr_ratio: float = 0.0,
    attn_implementation: Literal["flex_attention", "flash_attention_2", "sdpa", "eager"] = "flex_attention",
    compile_model: bool = True,
    compile_mode: str = "max-autotune-no-cudagraphs",
    compile_warmup: bool = True,
    memory_probe_steps: int = 0,
    save_final: bool = False,
    log_every: int = 5,
    wandb_project: str = "",
    wandb_entity: str = "",
    wandb_name: str = "",
    wandb_group: str = "",
    wandb_tags: str = "",
    wandb_mode: Literal["online", "offline", "disabled"] = "online",
    track: str = "1",
    record_description: str = "",
    record_contributors: str = "",
) -> dict[str, Any]:
    import datetime as dt
    import math
    import os
    import random
    import time

    import numpy as np
    import torch
    from datasets import load_dataset
    from transformers import AutoTokenizer

    try:
        from transformers import Qwen3_5ForConditionalGeneration as ModelClass
    except ImportError:
        from transformers import AutoModelForImageTextToText as ModelClass
    if attn_implementation == "flex_attention":
        try:
            from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5PreTrainedModel

            Qwen3_5PreTrainedModel._supports_flex_attn = True
            ModelClass._supports_flex_attn = True
            print("enabled qwen3.5 flex_attention support flag", flush=True)
        except Exception as exc:
            print(f"warning: could not enable qwen3.5 flex_attention support flag: {exc}", flush=True)

    if minutes <= 0:
        raise ValueError("--minutes must be positive")
    if seq_len < 128:
        raise ValueError("--seq-len must be at least 128")
    if micro_batch_size < 0 or grad_accum < 0 or eval_micro_batch_size < 0:
        raise ValueError(
            "--micro-batch-size, --grad-accum, and --eval-micro-batch-size "
            "must be non-negative; use 0 for auto"
        )
    if eval_blocks < 1:
        raise ValueError("--eval-blocks must be positive")
    if warmup_steps < 0:
        raise ValueError("--warmup-steps must be non-negative")
    if lr < 0:
        raise ValueError("--lr must be non-negative; use 0 for the mode default")
    if weight_decay < 0.0 and weight_decay != -1.0:
        raise ValueError("--weight-decay must be >= 0; use -1 for the mode default")
    if track not in TRACKS:
        raise ValueError(f"--track must be one of: {', '.join(TRACKS.keys())}")
    data_mode = str(data_mode or ("sft" if track == "1" else "cpt")).lower().replace("-", "_")
    if data_mode not in DATA_MODE_CHOICES:
        raise ValueError(f"--data-mode must be one of: {', '.join(sorted(DATA_MODE_CHOICES))}")
    if sft_pack_block_size < 0:
        raise ValueError("--sft-pack-block-size must be non-negative; use 0 for auto")
    tuning_mode = str(tuning_mode).lower()
    if tuning_mode not in {"lora", "full"}:
        raise ValueError("--tuning-mode must be one of: lora, full")
    adapter_mode_default = DEFAULT_ADAPTER_MODE if tuning_mode == "lora" and data_mode == "sft" else "lora"
    adapter_mode = str(adapter_mode or adapter_mode_default).lower().replace("-", "_")
    if adapter_mode in {"loraga", "lora-ga"}:
        adapter_mode = "lora_ga"
    if adapter_mode not in ADAPTER_MODE_CHOICES:
        raise ValueError(
            f"--adapter-mode must be one of: {', '.join(sorted(ADAPTER_MODE_CHOICES))}"
        )
    gradient_checkpointing = str(gradient_checkpointing).lower()
    if gradient_checkpointing not in {"auto", "true", "false"}:
        raise ValueError("--gradient-checkpointing must be one of: auto, true, false")
    activation_compression_mode = str(activation_compression_mode).lower().replace("_", "-")
    if activation_compression_mode in {"instant", "instantlinear"}:
        activation_compression_mode = "instant-linear"
    if activation_compression_mode not in ACTIVATION_COMPRESSION_MODE_CHOICES:
        raise ValueError(
            "--activation-compression-mode must be one of: "
            f"{', '.join(sorted(ACTIVATION_COMPRESSION_MODE_CHOICES))}"
        )
    instant_projector_kind = str(instant_projector_kind).lower()
    if instant_projector_kind not in INSTANT_PROJECTOR_KIND_CHOICES:
        raise ValueError(
            "--instant-projector-kind must be one of: "
            f"{', '.join(sorted(INSTANT_PROJECTOR_KIND_CHOICES))}"
        )
    if instant_chunk_size < 1 or instant_chunk_size & (instant_chunk_size - 1):
        raise ValueError("--instant-chunk-size must be a positive power of two")
    if instant_keep < 1 or instant_keep > instant_chunk_size:
        raise ValueError("--instant-keep must be in [1, instant-chunk-size]")
    if instant_min_hidden_dim < 0:
        raise ValueError("--instant-min-hidden-dim must be non-negative")
    instant_hadamard_backend = str(instant_hadamard_backend).lower().replace("_", "-")
    if instant_hadamard_backend not in INSTANT_HADAMARD_BACKEND_CHOICES:
        raise ValueError(
            "--instant-hadamard-backend must be one of: "
            f"{', '.join(sorted(INSTANT_HADAMARD_BACKEND_CHOICES))}"
        )
    if instant_hadamard_backend in {"fast", "triton"}:
        instant_hadamard_backend = "piecewise"
    instant_parameter_gradient = str(instant_parameter_gradient).lower().replace("-", "_")
    if instant_parameter_gradient in {"projected", "lowpass", "low_pass"}:
        instant_parameter_gradient = "projected_lowpass"
    if instant_parameter_gradient not in INSTANT_PARAMETER_GRADIENT_CHOICES:
        raise ValueError(
            "--instant-parameter-gradient must be one of: "
            f"{', '.join(sorted(INSTANT_PARAMETER_GRADIENT_CHOICES))}"
        )
    instant_target_modules = str(instant_target_modules).lower().replace("_", "-")
    if instant_target_modules not in INSTANT_TARGET_MODULE_CHOICES:
        raise ValueError(
            "--instant-target-modules must be one of: "
            f"{', '.join(sorted(INSTANT_TARGET_MODULE_CHOICES))}"
        )
    optimizer_name = str(optimizer_name).lower()
    if optimizer_name not in OPTIMIZER_CHOICES:
        raise ValueError(f"--optimizer-name must be one of: {', '.join(sorted(OPTIMIZER_CHOICES))}")
    if wandb_mode not in {"online", "offline", "disabled"}:
        raise ValueError("--wandb-mode must be one of: online, offline, disabled")
    if lora_r < 1:
        raise ValueError("--lora-r must be positive")
    if lora_alpha < 1:
        raise ValueError("--lora-alpha must be positive")
    if lora_dropout < 0.0 or lora_dropout >= 1.0:
        raise ValueError("--lora-dropout must be in [0, 1)")
    if gralora_k < 1:
        raise ValueError("--gralora-k must be positive")
    if adapter_mode == "gralora" and lora_r % gralora_k != 0:
        raise ValueError("--lora-r must be divisible by --gralora-k for GraLoRA")
    lora_init = str(lora_init).lower().replace("-", "_")
    if lora_init == "loraga":
        lora_init = "lora_ga"
    if adapter_mode == "lora_ga":
        if lora_init not in {"default", "lora_ga"}:
            raise ValueError("--adapter-mode lora_ga cannot be combined with a different --lora-init")
        lora_init = "lora_ga"
    elif adapter_mode == "lora" and lora_init == "lora_ga":
        adapter_mode = "lora_ga"
    if lora_init.startswith("pissa_niter_"):
        try:
            pissa_iters = int(lora_init.removeprefix("pissa_niter_"))
        except ValueError as exc:
            raise ValueError("--lora-init pissa_niter_N requires an integer N") from exc
        if pissa_iters < 1:
            raise ValueError("--lora-init pissa_niter_N requires N >= 1")
    elif lora_init not in LORA_INIT_CHOICES:
        raise ValueError(
            "--lora-init must be one of: "
            f"{', '.join(sorted(LORA_INIT_CHOICES))}, or pissa_niter_N"
        )
    if lora_eva_rho < 1.0:
        raise ValueError("--lora-eva-rho must be >= 1.0")
    if lora_eva_batches < 1:
        raise ValueError("--lora-eva-batches must be positive")
    if lora_ga_batches < 1:
        raise ValueError("--lora-ga-batches must be positive")
    if lora_ga_micro_batch_size < 1:
        raise ValueError("--lora-ga-micro-batch-size must be positive")
    direction_lookup = {value.lower(): value for value in LORA_GA_DIRECTION_CHOICES}
    lora_ga_direction = direction_lookup.get(str(lora_ga_direction).lower(), lora_ga_direction)
    if lora_ga_direction not in LORA_GA_DIRECTION_CHOICES:
        raise ValueError(
            "--lora-ga-direction must be one of: "
            f"{', '.join(sorted(LORA_GA_DIRECTION_CHOICES))}"
        )
    lora_ga_scale = str(lora_ga_scale).lower()
    if lora_ga_scale not in LORA_GA_SCALE_CHOICES:
        raise ValueError(
            "--lora-ga-scale must be one of: "
            f"{', '.join(sorted(LORA_GA_SCALE_CHOICES))}"
        )
    if lora_ga_stable_gamma < 1:
        raise ValueError("--lora-ga-stable-gamma must be positive")
    if loraplus_lr_ratio < 1.0:
        raise ValueError("--loraplus-lr-ratio must be >= 1.0")
    if loraplus_lr_embedding <= 0.0:
        raise ValueError("--loraplus-lr-embedding must be positive")
    if muon_lr_adjustment not in MUON_LR_ADJUSTMENT_CHOICES:
        raise ValueError(
            "--muon-lr-adjustment must be one of: "
            f"{', '.join(sorted(MUON_LR_ADJUSTMENT_CHOICES))}"
        )
    if muon_quant_block_size < 1:
        raise ValueError("--muon-quant-block-size must be positive")
    if normuon_beta2 < 0.0 or normuon_beta2 >= 1.0:
        raise ValueError("--normuon-beta2 must be in [0, 1)")
    if normuon_eps <= 0.0:
        raise ValueError("--normuon-eps must be positive")
    lr_schedule = str(lr_schedule).lower()
    if lr_schedule not in LR_SCHEDULE_CHOICES:
        raise ValueError(f"--lr-schedule must be one of: {', '.join(sorted(LR_SCHEDULE_CHOICES))}")
    if lr_decay_fraction < 0.0 or lr_decay_fraction >= 1.0:
        raise ValueError("--lr-decay-fraction must be in [0, 1)")
    if min_lr_ratio < 0.0 or min_lr_ratio > 1.0:
        raise ValueError("--min-lr-ratio must be in [0, 1]")
    if attn_implementation not in {"flex_attention", "flash_attention_2", "sdpa", "eager"}:
        raise ValueError(
            "--attn-implementation must be one of: flex_attention, flash_attention_2, sdpa, eager"
        )
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for the Modal H100 training path")
    if data_mode == "sft" and track != "1":
        raise ValueError("--data-mode sft is currently implemented only for Track 1")
    if optimizer_name in {"loraplus_adamw", "loraplus_adamw8bit", "lorafa"} and tuning_mode != "lora":
        raise ValueError(f"--optimizer-name {optimizer_name} requires --tuning-mode lora")
    if (
        adapter_mode == "gralora"
        and optimizer_name in {"loraplus_adamw", "loraplus_adamw8bit", "lorafa"}
    ):
        raise ValueError(f"--optimizer-name {optimizer_name} requires standard LoRA adapters")
    if (
        tuning_mode == "lora"
        and adapter_mode == "gralora"
        and optimizer_name in {"muon", "muon8", "normuon"}
    ):
        raise ValueError(f"--optimizer-name {optimizer_name} requires standard LoRA or full tuning")
    if adapter_mode == "gralora" and lora_init != "default":
        raise ValueError("--adapter-mode gralora does not support --lora-init")
    if adapter_mode == "gralora" and lora_use_dora:
        raise ValueError("--adapter-mode gralora does not support --lora-use-dora")
    if lora_init in {"eva", "lora_ga"} and tuning_mode != "lora":
        raise ValueError(f"--lora-init {lora_init} requires --tuning-mode lora")
    if adapter_mode != "lora" and tuning_mode != "lora":
        raise ValueError(f"--adapter-mode {adapter_mode} requires --tuning-mode lora")
    if lora_use_dora and tuning_mode != "lora":
        raise ValueError("--lora-use-dora requires --tuning-mode lora")

    track_info = TRACKS[track]
    requested_dataset_id = dataset_id
    requested_dataset_config = dataset_config
    requested_dataset_revision = dataset_revision
    dataset_id, dataset_config, dataset_revision = _resolve_dataset_defaults(
        data_mode,
        dataset_id,
        dataset_config,
        dataset_revision,
    )
    requested_sft_pack_block_size = sft_pack_block_size
    effective_sft_pack_block_size = 0
    if data_mode == "sft":
        effective_sft_pack_block_size = (
            sft_pack_block_size
            if sft_pack_block_size > 0
            else instant_chunk_size
            if activation_compression_mode == "instant-linear"
            else 0
        )
        if effective_sft_pack_block_size < 0:
            raise ValueError("--sft-pack-block-size must be non-negative; use 0 for auto")
        if effective_sft_pack_block_size and seq_len % effective_sft_pack_block_size != 0:
            raise ValueError("--seq-len must be a multiple of the effective SFT pack block size")
        if (
            activation_compression_mode == "instant-linear"
            and effective_sft_pack_block_size % instant_chunk_size != 0
        ):
            raise ValueError(
                "the effective SFT pack block size must be a multiple of --instant-chunk-size"
            )
    effective_packing_strategy = DEFAULT_PACKING_STRATEGY
    if data_mode == "sft" and effective_sft_pack_block_size:
        effective_packing_strategy = (
            f"{DEFAULT_PACKING_STRATEGY}+block_aligned_{effective_sft_pack_block_size}"
        )
    requested_micro_batch_size = micro_batch_size
    requested_grad_accum = grad_accum
    requested_eval_micro_batch_size = eval_micro_batch_size
    if micro_batch_size == 0:
        micro_batch_size = (
            DEFAULT_LORA_MICRO_BATCH_SIZE
            if tuning_mode == "lora"
            else DEFAULT_FULL_MICRO_BATCH_SIZE
        )
    if eval_micro_batch_size == 0:
        eval_micro_batch_size = min(micro_batch_size, DEFAULT_EVAL_MICRO_BATCH_SIZE)
    if grad_accum == 0:
        tokens_per_micro_batch = seq_len * micro_batch_size
        grad_accum = max(1, DEFAULT_EFFECTIVE_TOKENS_PER_STEP // tokens_per_micro_batch)
    requested_lr = lr
    requested_weight_decay = weight_decay
    requested_optimizer_name = optimizer_name
    resolved_optimizer_name = (
        "adamw_fused"
        if optimizer_name == "auto" and tuning_mode == "lora"
        else "adamw8bit"
        if optimizer_name == "auto"
        else optimizer_name
    )
    wandb_tags_list = [tag.strip() for tag in wandb_tags.split(",") if tag.strip()]
    wandb_enabled = bool(wandb_project) and wandb_mode != "disabled"
    if lr <= 0.0:
        if tuning_mode == "lora" and resolved_optimizer_name in {"loraplus_adamw", "loraplus_adamw8bit"}:
            lr = 5.0e-5
        elif tuning_mode == "lora":
            lr = 2.0e-4
        else:
            lr = 2.0e-5
    weight_decay = weight_decay if weight_decay >= 0.0 else (0.01 if tuning_mode == "lora" else 0.1)
    memory_probe_steps = max(0, int(memory_probe_steps))
    checkpointing_enabled = gradient_checkpointing == "true" or (
        gradient_checkpointing == "auto"
        and (tuning_mode == "full" or (tuning_mode == "lora" and micro_batch_size > 1))
    )
    gradient_checkpointing_fallback_used = False

    os.environ.setdefault("HF_HOME", str(HF_CACHE))
    os.environ.setdefault("HF_HUB_ENABLE_HF_TRANSFER", "1")
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "true")
    os.environ.setdefault("PYTORCH_ALLOC_CONF", "expandable_segments:True")
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    try:
        import torch._dynamo as dynamo

        dynamo.config.recompile_limit = 64
    except Exception as exc:
        print(f"warning: could not set torch dynamo config: {exc}", flush=True)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    try:
        torch.set_float32_matmul_precision("high")
    except AttributeError:
        pass

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    run_id = f"{dt.datetime.now(dt.UTC).strftime('%Y%m%d-%H%M%S-%f')}-{uuid.uuid4().hex[:8]}"
    run_dir = CACHE_MOUNT / "runs" / run_id
    eval_dir = CACHE_MOUNT / "eval"
    run_dir.mkdir(parents=True, exist_ok=True)
    eval_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = run_dir / "metrics.jsonl"
    lora_ga_cache_file = None
    if tuning_mode == "lora" and adapter_mode == "lora_ga" and lora_ga_cache:
        lora_ga_cache_payload = {
            "model": model_id,
            "model_revision": model_revision,
            "dataset": dataset_id,
            "dataset_config": dataset_config,
            "dataset_revision": dataset_revision,
            "data_mode": data_mode,
            "loss_mask": "assistant_only" if data_mode == "sft" else "all_tokens",
            "sequence_packing": DEFAULT_SEQUENCE_PACKING,
            "packing_strategy": effective_packing_strategy,
            "sft_pack_block_size": effective_sft_pack_block_size,
            "seq_len": seq_len,
            "eval_blocks": eval_blocks,
            "seed": seed,
            "adapter_mode": adapter_mode,
            "lora_r": lora_r,
            "lora_alpha": lora_alpha,
            "lora_target_modules": lora_target_modules,
            "lora_use_rslora": lora_use_rslora,
            "lora_ga_batches": lora_ga_batches,
            "lora_ga_micro_batch_size": lora_ga_micro_batch_size,
            "lora_ga_direction": lora_ga_direction,
            "lora_ga_scale": lora_ga_scale,
            "lora_ga_stable_gamma": lora_ga_stable_gamma,
        }
        lora_ga_cache_key = hashlib.sha256(
            json.dumps(lora_ga_cache_payload, sort_keys=True).encode()
        ).hexdigest()[:20]
        lora_ga_cache_file = str(CACHE_MOUNT / "loraga" / f"{lora_ga_cache_key}.pt")

    bytes_per_gib = 1024**3
    gpu_index = torch.cuda.current_device()
    gpu_props = torch.cuda.get_device_properties(gpu_index)
    gpu_name = torch.cuda.get_device_name(gpu_index)
    nvml = None
    nvml_handle = None
    nvml_error = ""
    peak_gpu_stats: dict[str, float] = {}
    try:
        import pynvml

        pynvml.nvmlInit()
        nvml = pynvml
        nvml_handle = pynvml.nvmlDeviceGetHandleByIndex(gpu_index)
    except Exception as exc:
        nvml_error = str(exc)

    def collect_gpu_stats() -> dict[str, Any]:
        stats: dict[str, Any] = {}
        try:
            free_bytes, total_bytes = torch.cuda.mem_get_info(gpu_index)
            allocated = torch.cuda.memory_allocated(gpu_index)
            reserved = torch.cuda.memory_reserved(gpu_index)
            peak_allocated = torch.cuda.max_memory_allocated(gpu_index)
            peak_reserved = torch.cuda.max_memory_reserved(gpu_index)
            stats.update(
                {
                    "cuda_memory_allocated_gib": allocated / bytes_per_gib,
                    "cuda_memory_reserved_gib": reserved / bytes_per_gib,
                    "cuda_max_memory_allocated_gib": peak_allocated / bytes_per_gib,
                    "cuda_max_memory_reserved_gib": peak_reserved / bytes_per_gib,
                    "cuda_memory_free_gib": free_bytes / bytes_per_gib,
                    "cuda_memory_total_gib": total_bytes / bytes_per_gib,
                    "cuda_memory_reserved_fraction": reserved / max(total_bytes, 1),
                    "cuda_max_memory_reserved_fraction": peak_reserved / max(total_bytes, 1),
                    "cuda_max_memory_allocated_fraction": peak_allocated / max(total_bytes, 1),
                }
            )
        except Exception as exc:
            stats["cuda_memory_stats_error"] = str(exc)

        if nvml is not None and nvml_handle is not None:
            try:
                memory_info = nvml.nvmlDeviceGetMemoryInfo(nvml_handle)
                utilization = nvml.nvmlDeviceGetUtilizationRates(nvml_handle)
                power_watts = nvml.nvmlDeviceGetPowerUsage(nvml_handle) / 1000.0
                stats.update(
                    {
                        "gpu_utilization_percent": utilization.gpu,
                        "gpu_memory_utilization_percent": utilization.memory,
                        "gpu_memory_used_gib": memory_info.used / bytes_per_gib,
                        "gpu_memory_total_gib": memory_info.total / bytes_per_gib,
                        "gpu_memory_used_fraction": memory_info.used / max(memory_info.total, 1),
                        "gpu_power_watts": power_watts,
                    }
                )
            except Exception as exc:
                stats["gpu_nvml_stats_error"] = str(exc)
        elif nvml_error:
            stats["gpu_nvml_init_error"] = nvml_error
        return stats

    def update_peak_gpu_stats(stats: dict[str, Any]) -> None:
        for source_key, target_key in PEAK_GPU_STAT_KEYS.items():
            value = stats.get(source_key)
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                continue
            numeric_value = float(value)
            if math.isfinite(numeric_value):
                peak_gpu_stats[target_key] = max(peak_gpu_stats.get(target_key, 0.0), numeric_value)

    config = {
        "run_id": run_id,
        "track": track,
        "track_name": track_info["name"],
        "record_description": record_description,
        "record_contributors": record_contributors,
        "minutes": minutes,
        "seq_len": seq_len,
        "micro_batch_size": micro_batch_size,
        "requested_micro_batch_size": requested_micro_batch_size,
        "grad_accum": grad_accum,
        "requested_grad_accum": requested_grad_accum,
        "effective_tokens_per_step": seq_len * micro_batch_size * grad_accum,
        "target_effective_tokens_per_step": DEFAULT_EFFECTIVE_TOKENS_PER_STEP,
        "gpu_name": gpu_name,
        "gpu_total_memory_gib": gpu_props.total_memory / bytes_per_gib,
        "lr": lr,
        "requested_lr": requested_lr,
        "weight_decay": weight_decay,
        "requested_weight_decay": requested_weight_decay,
        "warmup_steps": warmup_steps,
        "eval_blocks": eval_blocks,
        "eval_micro_batch_size": eval_micro_batch_size,
        "requested_eval_micro_batch_size": requested_eval_micro_batch_size,
        "seed": seed,
        "model_id": model_id,
        "model_revision": model_revision,
        "dataset_id": dataset_id,
        "dataset_config": dataset_config,
        "dataset_revision": dataset_revision,
        "requested_dataset_id": requested_dataset_id,
        "requested_dataset_config": requested_dataset_config,
        "requested_dataset_revision": requested_dataset_revision,
        "data_mode": data_mode,
        "loss_mask": "assistant_only" if data_mode == "sft" else "all_tokens",
        "sequence_packing": DEFAULT_SEQUENCE_PACKING,
        "packing_strategy": effective_packing_strategy,
        "sft_pack_block_size": effective_sft_pack_block_size if data_mode == "sft" else None,
        "requested_sft_pack_block_size": requested_sft_pack_block_size if data_mode == "sft" else None,
        "packed_block_size": seq_len,
        "padding_tokens_per_block": None if effective_sft_pack_block_size else 0,
        "tuning_mode": tuning_mode,
        "adapter_mode": adapter_mode if tuning_mode == "lora" else None,
        "optimizer_name": resolved_optimizer_name,
        "requested_optimizer_name": requested_optimizer_name,
        "gradient_checkpointing": gradient_checkpointing,
        "gradient_checkpointing_enabled": checkpointing_enabled,
        "gradient_checkpointing_fallback_used": gradient_checkpointing_fallback_used,
        "activation_compression_mode": activation_compression_mode,
        "instant_projector_kind": instant_projector_kind
        if activation_compression_mode == "instant-linear"
        else None,
        "instant_chunk_size": instant_chunk_size if activation_compression_mode == "instant-linear" else None,
        "instant_keep": instant_keep if activation_compression_mode == "instant-linear" else None,
        "instant_min_hidden_dim": instant_min_hidden_dim
        if activation_compression_mode == "instant-linear"
        else None,
        "instant_hadamard_backend": instant_hadamard_backend
        if activation_compression_mode == "instant-linear"
        else None,
        "instant_parameter_gradient": instant_parameter_gradient
        if activation_compression_mode == "instant-linear"
        else None,
        "instant_parameter_grad_storage": (
            "bf16_projected_coefficients"
            if instant_parameter_gradient == "projected_lowpass"
            else "full_activations"
        )
        if activation_compression_mode == "instant-linear"
        else None,
        "instant_target_modules": instant_target_modules
        if activation_compression_mode == "instant-linear"
        else None,
        "instant_exact_input_grad": True if activation_compression_mode == "instant-linear" else None,
        "instant_lowpass_wrapped_module_count": 0,
        "instant_lowpass_wrapped_module_names": [],
        "instant_lowpass_patched_gralora_module_count": 0,
        "instant_lowpass_patched_gralora_module_names": [],
        "lora_r": lora_r if tuning_mode == "lora" else None,
        "lora_alpha": lora_alpha if tuning_mode == "lora" else None,
        "lora_dropout": lora_dropout if tuning_mode == "lora" else None,
        "lora_target_modules": lora_target_modules if tuning_mode == "lora" else None,
        "gralora_k": gralora_k if tuning_mode == "lora" and adapter_mode == "gralora" else None,
        "lora_use_rslora": lora_use_rslora if tuning_mode == "lora" and adapter_mode != "gralora" else None,
        "lora_use_dora": lora_use_dora if tuning_mode == "lora" and adapter_mode != "gralora" else None,
        "lora_init": lora_init if tuning_mode == "lora" and adapter_mode != "gralora" else None,
        "lora_eva_rho": lora_eva_rho if tuning_mode == "lora" and adapter_mode != "gralora" else None,
        "lora_eva_batches": lora_eva_batches if tuning_mode == "lora" and adapter_mode != "gralora" else None,
        "lora_ga_batches": lora_ga_batches if tuning_mode == "lora" and adapter_mode == "lora_ga" else None,
        "lora_ga_micro_batch_size": lora_ga_micro_batch_size
        if tuning_mode == "lora" and adapter_mode == "lora_ga"
        else None,
        "lora_ga_direction": lora_ga_direction if tuning_mode == "lora" and adapter_mode == "lora_ga" else None,
        "lora_ga_scale": lora_ga_scale if tuning_mode == "lora" and adapter_mode == "lora_ga" else None,
        "lora_ga_stable_gamma": lora_ga_stable_gamma
        if tuning_mode == "lora" and adapter_mode == "lora_ga"
        else None,
        "lora_ga_cache": lora_ga_cache if tuning_mode == "lora" and adapter_mode == "lora_ga" else None,
        "lora_ga_cache_file": lora_ga_cache_file
        if tuning_mode == "lora" and adapter_mode == "lora_ga"
        else None,
        "loraplus_lr_ratio": loraplus_lr_ratio,
        "loraplus_lr_embedding": loraplus_lr_embedding,
        "muon_lr_adjustment": muon_lr_adjustment,
        "muon_quant_block_size": muon_quant_block_size,
        "normuon_beta2": normuon_beta2,
        "normuon_eps": normuon_eps,
        "lr_schedule": lr_schedule,
        "lr_decay_fraction": lr_decay_fraction,
        "min_lr_ratio": min_lr_ratio,
        "attn_implementation": attn_implementation,
        "compile_model": compile_model,
        "compile_mode": compile_mode,
        "compile_warmup": compile_warmup,
        "memory_probe_steps": memory_probe_steps,
        "save_final": save_final,
        "wandb_enabled": wandb_enabled,
        "wandb_project": wandb_project,
        "wandb_entity": wandb_entity,
        "wandb_name": wandb_name,
        "wandb_group": wandb_group,
        "wandb_tags": wandb_tags_list,
        "wandb_mode": wandb_mode,
    }

    def write_config() -> None:
        (run_dir / "config.json").write_text(json.dumps(config, indent=2) + "\n")

    write_config()

    wandb_run = None

    def wandb_log(record: dict[str, Any]) -> None:
        if wandb_run is None:
            return
        payload: dict[str, int | float | bool] = {}
        for key, value in record.items():
            if key == "time":
                continue
            if isinstance(value, bool):
                payload[key] = value
            elif isinstance(value, int):
                payload[key] = value
            elif isinstance(value, float) and math.isfinite(value):
                payload[key] = value
        event = record.get("event")
        if event in {"baseline_eval", "final_eval"} and isinstance(record.get("eval_loss"), (int, float)):
            payload[f"{event}/eval_loss"] = float(record["eval_loss"])
        if payload:
            wandb_run.log(payload)

    def log_metric(record: dict[str, Any], include_gpu: bool = False) -> None:
        if include_gpu:
            gpu_stats = collect_gpu_stats()
            update_peak_gpu_stats(gpu_stats)
            record = {**record, **gpu_stats}
        record = {"time": time.time(), **record}
        with metrics_path.open("a") as f:
            f.write(json.dumps(record, default=_json_default) + "\n")
        print(json.dumps(record, default=_json_default), flush=True)
        wandb_log(record)

    def log_gpu(label: str) -> None:
        log_metric({"event": "gpu", "label": label}, include_gpu=True)

    if wandb_enabled:
        if wandb_mode == "online" and not os.environ.get("WANDB_API_KEY"):
            raise RuntimeError(
                "W&B online mode requires WANDB_API_KEY in the local environment before "
                "running Modal, or use --wandb-mode offline"
            )
        import wandb

        os.environ.setdefault("WANDB_DIR", str(run_dir))
        wandb_run = wandb.init(
            project=wandb_project,
            entity=wandb_entity or None,
            name=wandb_name or record_description or run_id,
            group=wandb_group or f"track-{track}",
            tags=wandb_tags_list,
            config=config,
            mode=wandb_mode,
            dir=str(run_dir),
        )

    def dataset_stream(shuffle: bool = False, purpose: Literal["train", "eval"] = "train"):
        if data_mode == "sft" and dataset_id == DEFAULT_SFT_DATASET_ID:
            split = DEFAULT_SFT_EVAL_SPLIT if purpose == "eval" else DEFAULT_SFT_TRAIN_SPLIT
        else:
            split = "train"
        ds = load_dataset(
            dataset_id,
            dataset_config or None,
            split=split,
            streaming=True,
            revision=dataset_revision or None,
            cache_dir=str(HF_CACHE / "datasets"),
        )
        if shuffle:
            ds = ds.shuffle(seed=seed, buffer_size=10_000)
        return ds

    print("loading tokenizer", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(
        model_id,
        revision=model_revision or None,
        cache_dir=str(HF_CACHE / "hub"),
        use_fast=True,
    )
    if tokenizer.eos_token_id is None:
        raise ValueError(f"{model_id} tokenizer has no EOS token")
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    sft_pad_token_id = int(tokenizer.pad_token_id)
    config["sft_pad_token_id"] = sft_pad_token_id if data_mode == "sft" else None

    def tokenize_piece(text: str) -> list[int]:
        return tokenizer(text, add_special_tokens=False).input_ids

    def tokenize_text(text: str) -> list[int]:
        ids = tokenizer(text, add_special_tokens=False).input_ids
        if ids:
            ids.append(tokenizer.eos_token_id)
        return ids

    def render_sft_row(row: dict[str, Any]) -> tuple[list[int], list[int]] | None:
        return _render_sft_row(row, tokenize_piece, seq_len)

    def align_sft_row(
        row_ids: list[int],
        row_labels: list[int],
    ) -> tuple[list[int], list[int], list[int], int]:
        return _pad_sft_row_to_block_multiple(
            row_ids,
            row_labels,
            block_size=effective_sft_pack_block_size,
            pad_token_id=sft_pad_token_id,
            seq_len=seq_len,
        )

    def build_eval_cache() -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, int, Path, int]:
        if data_mode == "sft":
            return build_sft_eval_cache()
        return build_cpt_eval_cache()

    def load_eval_payload(eval_path: Path) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, int, int]:
        payload = torch.load(eval_path, map_location="cpu")
        input_ids = payload["input_ids"]
        attention_mask = payload.get("attention_mask")
        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids, dtype=torch.long)
        labels = payload.get("labels")
        if labels is None:
            labels = input_ids.clone()
        supervised_tokens = int(payload.get("supervised_tokens", _supervised_token_count(labels)))
        return input_ids, attention_mask, labels, int(payload["skip_docs"]), supervised_tokens

    def build_cpt_eval_cache() -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, int, Path, int]:
        key_payload = {
            "model": model_id,
            "model_revision": model_revision,
            "dataset": dataset_id,
            "dataset_config": dataset_config,
            "dataset_revision": dataset_revision,
            "seq_len": seq_len,
            "eval_blocks": eval_blocks,
            "sequence_packing": DEFAULT_SEQUENCE_PACKING,
            "packing_strategy": effective_packing_strategy,
            "seed": seed,
            "kind": "all_token_cpt_packed_v2",
        }
        key = hashlib.sha256(json.dumps(key_payload, sort_keys=True).encode()).hexdigest()[:20]
        eval_path = eval_dir / f"{key}.pt"
        if eval_path.exists():
            input_ids, attention_mask, labels, skip_docs, supervised_tokens = load_eval_payload(eval_path)
            return input_ids, attention_mask, labels, skip_docs, eval_path, supervised_tokens

        need_tokens = eval_blocks * seq_len
        token_buffer: list[int] = []
        skip_docs = 0
        for row in dataset_stream(shuffle=False):
            text = row.get("text")
            if not isinstance(text, str) or not text.strip():
                skip_docs += 1
                continue
            token_buffer.extend(tokenize_text(text))
            skip_docs += 1
            if len(token_buffer) >= need_tokens:
                break

        if len(token_buffer) < need_tokens:
            raise RuntimeError(
                f"could only build {len(token_buffer)} eval tokens, needed {need_tokens}"
            )

        input_ids = torch.tensor(token_buffer[:need_tokens], dtype=torch.long).view(eval_blocks, seq_len)
        attention_mask = torch.ones_like(input_ids, dtype=torch.long)
        labels = input_ids.clone()
        supervised_tokens = int(labels.numel())
        torch.save(
            {
                "input_ids": input_ids,
                "attention_mask": attention_mask,
                "labels": labels,
                "skip_docs": skip_docs,
                "supervised_tokens": supervised_tokens,
                "key_payload": key_payload,
            },
            eval_path,
        )
        return input_ids, attention_mask, labels, skip_docs, eval_path, supervised_tokens

    def build_sft_eval_cache() -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, int, Path, int]:
        key_payload = {
            "model": model_id,
            "model_revision": model_revision,
            "dataset": dataset_id,
            "dataset_config": dataset_config,
            "dataset_revision": dataset_revision,
            "seq_len": seq_len,
            "eval_blocks": eval_blocks,
            "eval_split": DEFAULT_SFT_EVAL_SPLIT if dataset_id == DEFAULT_SFT_DATASET_ID else "train",
            "sequence_packing": DEFAULT_SEQUENCE_PACKING,
            "packing_strategy": effective_packing_strategy,
            "sft_pack_block_size": effective_sft_pack_block_size,
            "sft_pad_token_id": sft_pad_token_id,
            "kind": "chatml_assistant_only_sft_packed_v3",
        }
        key = hashlib.sha256(json.dumps(key_payload, sort_keys=True).encode()).hexdigest()[:20]
        eval_path = eval_dir / f"{key}.pt"
        if eval_path.exists():
            input_ids, attention_mask, labels, skip_docs, supervised_tokens = load_eval_payload(eval_path)
            return input_ids, attention_mask, labels, skip_docs, eval_path, supervised_tokens

        token_buffer: list[int] = []
        label_buffer: list[int] = []
        mask_buffer: list[int] = []
        segment_buffer: list[int] = []
        input_blocks: list[torch.Tensor] = []
        mask_blocks: list[torch.Tensor] = []
        label_blocks: list[torch.Tensor] = []
        skip_docs = 0
        next_segment_id = 0

        def drain_blocks() -> None:
            while len(token_buffer) >= seq_len and len(input_blocks) < eval_blocks:
                block_ids = token_buffer[:seq_len]
                block_labels = label_buffer[:seq_len]
                block_mask = mask_buffer[:seq_len]
                block_segments = segment_buffer[:seq_len]
                del token_buffer[:seq_len]
                del label_buffer[:seq_len]
                del mask_buffer[:seq_len]
                del segment_buffer[:seq_len]
                if activation_compression_mode == "instant-linear":
                    _assert_no_cross_sequence_lowpass_segments(
                        block_segments,
                        effective_sft_pack_block_size,
                    )
                if _supervised_token_count(block_labels) == 0:
                    continue
                input_blocks.append(torch.tensor(block_ids, dtype=torch.long))
                mask_blocks.append(torch.tensor(block_mask, dtype=torch.long))
                label_blocks.append(torch.tensor(block_labels, dtype=torch.long))

        for row in dataset_stream(shuffle=False, purpose="eval"):
            skip_docs += 1
            rendered = render_sft_row(row)
            if rendered is None:
                continue
            row_ids, row_labels = rendered
            row_ids, row_labels, row_mask, _pad_count = align_sft_row(row_ids, row_labels)
            token_buffer.extend(row_ids)
            label_buffer.extend(row_labels)
            mask_buffer.extend(row_mask)
            segment_buffer.extend([next_segment_id if mask else -1 for mask in row_mask])
            next_segment_id += 1
            drain_blocks()
            if len(input_blocks) >= eval_blocks:
                break

        if len(input_blocks) < eval_blocks:
            raise RuntimeError(
                f"could only build {len(input_blocks)} SFT eval blocks, needed {eval_blocks}"
            )

        input_ids = torch.stack(input_blocks)
        attention_mask = torch.stack(mask_blocks)
        labels = torch.stack(label_blocks)
        supervised_tokens = _supervised_token_count(labels)
        torch.save(
            {
                "input_ids": input_ids,
                "attention_mask": attention_mask,
                "labels": labels,
                "skip_docs": 0 if dataset_id == DEFAULT_SFT_DATASET_ID else skip_docs,
                "eval_docs": skip_docs,
                "supervised_tokens": supervised_tokens,
                "key_payload": key_payload,
            },
            eval_path,
        )
        train_skip = 0 if dataset_id == DEFAULT_SFT_DATASET_ID else skip_docs
        return input_ids, attention_mask, labels, train_skip, eval_path, supervised_tokens

    eval_input_ids, eval_attention_mask, eval_labels, train_skip_docs, eval_path, eval_supervised_tokens = build_eval_cache()
    config["eval_supervised_tokens"] = eval_supervised_tokens
    write_config()
    print(
        f"fixed eval cache: {eval_path} skip_docs={train_skip_docs} "
        f"supervised_tokens={eval_supervised_tokens}",
        flush=True,
    )

    def parse_lora_target_modules(value: str) -> str | list[str]:
        value = value.strip()
        if "visual" in value or "lm_head" in value:
            raise ValueError("--lora-target-modules must not include visual or lm_head modules")
        if value == "all-linear" or "," not in value:
            return value
        return [part.strip() for part in value.split(",") if part.strip()]

    def expand_all_linear_target_modules(
        current_model: torch.nn.Module,
        minimum_dimension: int = 1,
        dimension_divisor: int = 1,
    ) -> tuple[list[str], int, int]:
        modules: list[str] = []
        skipped_small = 0
        skipped_divisor = 0
        for name, module in current_model.named_modules():
            if not name or "lm_head" in name:
                continue
            if (
                name == "visual"
                or name.startswith("visual.")
                or ".visual." in name
                or name.endswith(".visual")
            ):
                continue
            if isinstance(module, torch.nn.Linear):
                if min(module.weight.shape) < minimum_dimension:
                    skipped_small += 1
                    continue
                if (
                    dimension_divisor > 1
                    and (
                        module.weight.shape[0] % dimension_divisor != 0
                        or module.weight.shape[1] % dimension_divisor != 0
                    )
                ):
                    skipped_divisor += 1
                    continue
                modules.append(name)
        if not modules:
            raise RuntimeError("could not expand all-linear adapter targets")
        return modules, skipped_small, skipped_divisor

    def disable_model_cache(current_model: torch.nn.Module) -> None:
        if hasattr(current_model, "config"):
            current_model.config.use_cache = False
            if hasattr(current_model.config, "text_config"):
                current_model.config.text_config.use_cache = False
        if hasattr(current_model, "model") and hasattr(current_model.model, "language_model"):
            current_model.model.language_model.config.use_cache = False

    def set_visual_eval(current_model: torch.nn.Module) -> None:
        root_model = getattr(current_model, "_orig_mod", current_model)
        for name, module in root_model.named_modules():
            if name == "visual" or name.endswith(".visual"):
                module.eval()

    def freeze_visual(current_model: torch.nn.Module) -> None:
        for name, module in current_model.named_modules():
            if name == "visual" or name.endswith(".visual"):
                for parameter in module.parameters():
                    parameter.requires_grad_(False)
                module.eval()

    def set_gradient_checkpointing(current_model: torch.nn.Module, enabled: bool) -> None:
        if enabled:
            if hasattr(current_model, "gradient_checkpointing_enable"):
                current_model.gradient_checkpointing_enable()
            return
        if hasattr(current_model, "gradient_checkpointing_disable"):
            current_model.gradient_checkpointing_disable()

    def train_batches(batch_size: int | None = None):
        active_batch_size = micro_batch_size if batch_size is None else batch_size
        ds = dataset_stream(shuffle=False).skip(train_skip_docs).shuffle(seed=seed, buffer_size=10_000)
        token_buffer: list[int] = []
        label_buffer: list[int] = []
        mask_buffer: list[int] = []
        segment_buffer: list[int] = []
        batch: list[tuple[torch.Tensor, torch.Tensor, torch.Tensor]] = []
        next_segment_id = 0

        def make_batch() -> dict[str, torch.Tensor]:
            input_batch = torch.stack([item[0] for item in batch])
            mask_batch = torch.stack([item[1] for item in batch])
            label_batch = torch.stack([item[2] for item in batch])
            batch.clear()
            return {"input_ids": input_batch, "attention_mask": mask_batch, "labels": label_batch}

        while True:
            for row in ds:
                if data_mode == "sft":
                    rendered = render_sft_row(row)
                    if rendered is None:
                        continue
                    row_ids, row_labels = rendered
                    row_ids, row_labels, row_mask, _pad_count = align_sft_row(row_ids, row_labels)
                    token_buffer.extend(row_ids)
                    label_buffer.extend(row_labels)
                    mask_buffer.extend(row_mask)
                    segment_buffer.extend([next_segment_id if mask else -1 for mask in row_mask])
                    next_segment_id += 1
                else:
                    text = row.get("text")
                    if not isinstance(text, str) or not text.strip():
                        continue
                    row_ids = tokenize_text(text)
                    token_buffer.extend(row_ids)
                    label_buffer.extend(row_ids)
                    mask_buffer.extend([1] * len(row_ids))

                while len(token_buffer) >= seq_len:
                    block_ids = token_buffer[:seq_len]
                    block_labels = label_buffer[:seq_len]
                    block_mask = mask_buffer[:seq_len]
                    block_segments = segment_buffer[:seq_len] if data_mode == "sft" else []
                    del token_buffer[:seq_len]
                    del label_buffer[:seq_len]
                    del mask_buffer[:seq_len]
                    if data_mode == "sft":
                        del segment_buffer[:seq_len]
                    if data_mode == "sft" and activation_compression_mode == "instant-linear":
                        _assert_no_cross_sequence_lowpass_segments(
                            block_segments,
                            effective_sft_pack_block_size,
                        )
                    if data_mode == "sft" and _supervised_token_count(block_labels) == 0:
                        continue
                    batch.append(
                        (
                            torch.tensor(block_ids, dtype=torch.long),
                            torch.tensor(block_mask, dtype=torch.long),
                            torch.tensor(block_labels, dtype=torch.long),
                        )
                    )
                    if len(batch) == active_batch_size:
                        yield make_batch()
            ds = dataset_stream(shuffle=False).skip(train_skip_docs).shuffle(seed=seed, buffer_size=10_000)

    device = torch.device("cuda")

    print("loading model", flush=True)
    model = ModelClass.from_pretrained(
        model_id,
        revision=model_revision or None,
        torch_dtype=torch.bfloat16,
        attn_implementation=attn_implementation,
        cache_dir=str(HF_CACHE / "hub"),
        low_cpu_mem_usage=True,
    )
    disable_model_cache(model)
    freeze_visual(model)

    if tuning_mode == "lora":
        if adapter_mode == "gralora":
            from peft import get_peft_model

            try:
                from peft import GraloraConfig
            except ImportError:
                from peft.tuners.gralora import GraloraConfig

            parsed_gralora_target_modules = parse_lora_target_modules(lora_target_modules)
            if parsed_gralora_target_modules == "all-linear":
                (
                    parsed_gralora_target_modules,
                    skipped_small,
                    skipped_divisor,
                ) = expand_all_linear_target_modules(
                    model,
                    dimension_divisor=gralora_k,
                )
                print(
                    f"expanded all-linear to {len(parsed_gralora_target_modules)} GraLoRA target modules "
                    f"(skipped {skipped_small} small, {skipped_divisor} indivisible by k={gralora_k})",
                    flush=True,
                )
            elif isinstance(parsed_gralora_target_modules, str):
                if not re.fullmatch(r"[A-Za-z0-9_.]+", parsed_gralora_target_modules):
                    raise ValueError(
                        "--adapter-mode gralora requires --lora-target-modules all-linear "
                        "or explicit module names"
                    )
                parsed_gralora_target_modules = [parsed_gralora_target_modules]

            print(
                "applying GraLoRA "
                f"target_modules={lora_target_modules!r} r={lora_r} alpha={lora_alpha} "
                f"dropout={lora_dropout} k={gralora_k}",
                flush=True,
            )
            gralora_config = GraloraConfig(
                target_modules=parsed_gralora_target_modules,
                r=lora_r,
                alpha=lora_alpha,
                gralora_dropout=lora_dropout,
                gralora_k=gralora_k,
                bias="none",
            )
            model = get_peft_model(model, gralora_config)
        else:
            from peft import LoraConfig, TaskType, get_peft_model

            if lora_init == "eva":
                from peft import EvaConfig
            if lora_init == "lora_ga":
                from peft import LoraGAConfig, preprocess_loraga

            print(
                "applying LoRA "
                f"target_modules={lora_target_modules!r} r={lora_r} alpha={lora_alpha} "
                f"dropout={lora_dropout} use_rslora={lora_use_rslora} "
                f"use_dora={lora_use_dora} init={lora_init}",
                flush=True,
            )
            lora_init_value: bool | str = True if lora_init == "default" else lora_init
            parsed_lora_target_modules = parse_lora_target_modules(lora_target_modules)
            if lora_init == "lora_ga" and parsed_lora_target_modules == "all-linear":
                parsed_lora_target_modules, skipped_small, skipped_divisor = expand_all_linear_target_modules(
                    model,
                    minimum_dimension=2 * lora_r,
                )
                print(
                    f"expanded all-linear to {len(parsed_lora_target_modules)} LoRA-GA target modules "
                    f"(skipped {skipped_small} modules with min dimension < {2 * lora_r}, "
                    f"{skipped_divisor} indivisible)",
                    flush=True,
                )
            lora_config_kwargs: dict[str, Any] = {
                "task_type": TaskType.CAUSAL_LM,
                "target_modules": parsed_lora_target_modules,
                "exclude_modules": r".*(visual|lm_head).*",
                "r": lora_r,
                "lora_alpha": lora_alpha,
                "lora_dropout": lora_dropout,
                "bias": "none",
                "use_rslora": lora_use_rslora,
                "use_dora": lora_use_dora,
                "init_lora_weights": lora_init_value,
                "ensure_weight_tying": True,
            }
            if lora_init == "eva":
                lora_config_kwargs["eva_config"] = EvaConfig(rho=lora_eva_rho)
            if lora_init == "lora_ga":
                lora_config_kwargs["lora_ga_config"] = LoraGAConfig(
                    direction=lora_ga_direction,
                    scale=lora_ga_scale,
                    stable_gamma=lora_ga_stable_gamma,
                )
            lora_config = LoraConfig(**lora_config_kwargs)
            if lora_init == "lora_ga":
                model.to(device)
                set_gradient_checkpointing(model, checkpointing_enabled)
                log_gpu("before_loraga_preprocess")
                loraga_batch_iter = train_batches(batch_size=lora_ga_micro_batch_size)

                def loraga_train_step() -> None:
                    model.zero_grad(set_to_none=True)
                    for _ in range(lora_ga_batches):
                        batch = next(loraga_batch_iter)
                        input_ids = batch["input_ids"].to(device, non_blocking=True)
                        attention_mask = batch["attention_mask"].to(device, non_blocking=True)
                        labels = batch["labels"].to(device, non_blocking=True)
                        with torch.autocast("cuda", dtype=torch.bfloat16):
                            output = model(
                                input_ids=input_ids,
                                attention_mask=attention_mask,
                                labels=labels,
                                use_cache=False,
                            )
                            loss = output.loss / lora_ga_batches
                        loss.backward()

                print(
                    "preprocessing LoRA-GA "
                    f"batches={lora_ga_batches} micro_batch_size={lora_ga_micro_batch_size} "
                    f"direction={lora_ga_direction} scale={lora_ga_scale} "
                    f"cache={'on' if lora_ga_cache_file else 'off'}",
                    flush=True,
                )
                preprocess_loraga(model, lora_config, loraga_train_step, cache_file=lora_ga_cache_file)
                model.zero_grad(set_to_none=True)
                torch.cuda.empty_cache()
                log_gpu("after_loraga_preprocess")
            model = get_peft_model(model, lora_config, low_cpu_mem_usage=(lora_init == "eva"))

    set_gradient_checkpointing(model, checkpointing_enabled)

    model.to(device)
    if tuning_mode == "lora" and lora_init == "eva":
        from peft import initialize_lora_eva_weights

        eva_blocks = min(eval_blocks, lora_eva_batches * eval_micro_batch_size)

        def iter_eva_batches():
            for start in range(0, eva_blocks, eval_micro_batch_size):
                batch = eval_input_ids[start : start + eval_micro_batch_size].to(device, non_blocking=True)
                attention_mask = eval_attention_mask[start : start + eval_micro_batch_size].to(
                    device,
                    non_blocking=True,
                )
                yield {
                    "input_ids": batch,
                    "attention_mask": attention_mask,
                }

        print(
            f"initializing EVA LoRA weights with {eva_blocks} eval blocks "
            f"rho={lora_eva_rho}",
            flush=True,
        )
        model.eval()
        with torch.no_grad():
            initialize_lora_eva_weights(model, iter_eva_batches(), show_progress_bar=False)
        model.train()
        set_visual_eval(model)
        torch.cuda.empty_cache()
    uncompiled_model = model
    torch.cuda.synchronize()
    log_gpu("after_model_to_cuda")

    if tuning_mode == "lora" and resolved_optimizer_name == "lorafa":
        for name, parameter in model.named_parameters():
            if "lora_A" in name:
                parameter.requires_grad_(False)

    instant_lowpass_wrapped_modules: list[str] = []
    refresh_instant_lowpass_merged_weights: Callable[..., int] | None = None
    if activation_compression_mode == "instant-linear":
        import sys

        if "/root" not in sys.path:
            sys.path.append("/root")
        from instant_lowpass import (
            InstantLowpassConfig,
            patch_gralora_with_instant_lowpass,
            refresh_gralora_merged_weights,
            replace_linear_with_instant_lowpass,
        )

        instant_config = InstantLowpassConfig(
            projector_kind=instant_projector_kind,
            chunk_size=instant_chunk_size,
            keep=instant_keep,
            min_hidden_dim=instant_min_hidden_dim,
            hadamard_backend=instant_hadamard_backend,
            parameter_gradient=instant_parameter_gradient,
            exact_input_grad=True,
            enabled=True,
        )

        def instant_module_filter(name: str, linear: torch.nn.Linear) -> bool:
            if any(part in name for part in ("visual", "lm_head", "embed")):
                return False
            trainable = linear.weight.requires_grad or (
                linear.bias is not None and linear.bias.requires_grad
            )
            if instant_target_modules != "all" and not trainable:
                return False
            if instant_target_modules == "adapter" and not (
                "lora_" in name or "gralora_" in name
            ):
                return False
            if min(int(linear.in_features), int(linear.out_features)) < instant_min_hidden_dim:
                return False
            return True

        instant_lowpass_wrapped_modules = replace_linear_with_instant_lowpass(
            model,
            instant_config,
            module_filter=instant_module_filter,
        )
        instant_lowpass_patched_gralora_modules = (
            patch_gralora_with_instant_lowpass(model, instant_config)
            if adapter_mode == "gralora"
            else []
        )
        if (
            instant_lowpass_patched_gralora_modules
            and instant_parameter_gradient == "exact"
        ):

            def refresh_instant_lowpass_merged_weights(*, force: bool = False) -> int:
                return refresh_gralora_merged_weights(uncompiled_model, force=force)

            merged_weight_count = refresh_instant_lowpass_merged_weights()
        else:
            merged_weight_count = 0
        if not instant_lowpass_wrapped_modules and not instant_lowpass_patched_gralora_modules:
            raise RuntimeError("instant-linear activation compression found no Linear modules to wrap")
        config["instant_lowpass_wrapped_module_count"] = len(instant_lowpass_wrapped_modules)
        config["instant_lowpass_wrapped_module_names"] = instant_lowpass_wrapped_modules[:100]
        config["instant_lowpass_patched_gralora_module_count"] = len(
            instant_lowpass_patched_gralora_modules
        )
        config["instant_lowpass_patched_gralora_module_names"] = (
            instant_lowpass_patched_gralora_modules[:100]
        )
        config["instant_lowpass_merged_weight_count"] = merged_weight_count
        config["instant_lowpass_merged_weight_refresh"] = (
            "force_after_optimizer_step_inplace" if refresh_instant_lowpass_merged_weights else None
        )
        write_config()
        print(
            "enabled instant-linear activation compression "
            f"projector={instant_projector_kind} keep={instant_keep}/{instant_chunk_size} "
            f"min_hidden_dim={instant_min_hidden_dim} "
            f"hadamard_backend={instant_hadamard_backend} "
            f"checkpointing={'on' if checkpointing_enabled else 'off'} "
            f"param_grad={instant_parameter_gradient} "
            f"param_grad_storage={config['instant_parameter_grad_storage']} "
            f"gralora_forward={'merged' if refresh_instant_lowpass_merged_weights else 'unmerged_adapter'} "
            f"target={instant_target_modules} "
            f"wrapped={len(instant_lowpass_wrapped_modules)} "
            f"patched_gralora={len(instant_lowpass_patched_gralora_modules)} "
            f"merged_weights={merged_weight_count}",
            flush=True,
        )

    named_trainable_params = [(n, p) for n, p in model.named_parameters() if p.requires_grad]
    trainable_params = [p for _, p in named_trainable_params]
    if not trainable_params:
        raise RuntimeError("no trainable parameters were found")
    trainable_visual_params = [n for n, p in named_trainable_params if "visual" in n]
    if trainable_visual_params:
        raise RuntimeError(f"visual parameters must remain frozen: {trainable_visual_params[:5]}")
    if tuning_mode == "lora":
        adapter_param_markers = ("gralora_",) if adapter_mode == "gralora" else ("lora_",)
        non_adapter_trainable = [
            n for n, p in named_trainable_params if not any(marker in n for marker in adapter_param_markers)
        ]
        if non_adapter_trainable:
            raise RuntimeError(
                f"{adapter_mode} mode found non-adapter trainable parameters: {non_adapter_trainable[:5]}"
            )
    trainable_count = sum(p.numel() for p in trainable_params)
    total_count = sum(p.numel() for p in model.parameters())
    print(f"trainable parameters: {trainable_count:,} / {total_count:,}", flush=True)

    class Muon(torch.optim.Optimizer):
        def __init__(
            self,
            params,
            lr: float,
            momentum: float = 0.95,
            weight_decay: float = 0.0,
            ns_steps: int = 5,
            nesterov: bool = True,
            lr_adjustment: str = "original",
        ):
            if lr_adjustment not in MUON_LR_ADJUSTMENT_CHOICES:
                raise ValueError(f"unsupported Muon LR adjustment: {lr_adjustment}")
            super().__init__(
                params,
                dict(
                    lr=lr,
                    momentum=momentum,
                    weight_decay=weight_decay,
                    ns_steps=ns_steps,
                    nesterov=nesterov,
                    lr_adjustment=lr_adjustment,
                ),
            )

        @staticmethod
        def zeropower_via_newtonschulz5(grad: torch.Tensor, steps: int) -> torch.Tensor:
            assert grad.ndim == 2
            a, b, c = 3.4445, -4.7750, 2.0315
            x = grad.bfloat16()
            transposed = x.size(0) > x.size(1)
            if transposed:
                x = x.T
            x = x / (x.norm() + 1.0e-7)
            for _ in range(steps):
                xx_t = x @ x.T
                x = a * x + (b * xx_t + c * (xx_t @ xx_t)) @ x
            if transposed:
                x = x.T
            return x.to(grad.dtype)

        @staticmethod
        def adjust_lr(lr: float, lr_adjustment: str, shape: torch.Size) -> float:
            rows, cols = shape[:2]
            if lr_adjustment == "match_rms_adamw":
                return lr * 0.2 * math.sqrt(max(rows, cols))
            return lr * math.sqrt(max(1.0, rows / cols))

        @torch.no_grad()
        def step(self, closure=None):
            loss = None
            if closure is not None:
                with torch.enable_grad():
                    loss = closure()
            for group in self.param_groups:
                lr = group["lr"]
                momentum = group["momentum"]
                weight_decay = group["weight_decay"]
                ns_steps = group["ns_steps"]
                nesterov = group["nesterov"]
                lr_adjustment = group["lr_adjustment"]
                for p in group["params"]:
                    if p.grad is None:
                        continue
                    grad = p.grad
                    if grad.ndim != 2:
                        raise RuntimeError("Muon only supports 2D matrix parameters")
                    state = self.state[p]
                    if "momentum_buffer" not in state:
                        state["momentum_buffer"] = torch.zeros_like(grad)
                    buf = state["momentum_buffer"]
                    buf.lerp_(grad, 1.0 - momentum)
                    update = grad.lerp(buf, momentum) if nesterov else buf
                    update = self.zeropower_via_newtonschulz5(update, ns_steps)
                    if weight_decay:
                        p.mul_(1.0 - lr * weight_decay)
                    adjusted_lr = self.adjust_lr(lr, lr_adjustment, p.shape)
                    p.add_(update, alpha=-adjusted_lr)
            return loss

    class QuantizedMuon(Muon):
        @staticmethod
        def quantize_blockwise(tensor: torch.Tensor, block_size: int) -> tuple[torch.Tensor, torch.Tensor]:
            flat = tensor.detach().to(torch.float32).reshape(-1)
            numel = flat.numel()
            num_blocks = (numel + block_size - 1) // block_size
            padded = torch.zeros(num_blocks * block_size, device=flat.device, dtype=torch.float32)
            padded[:numel] = flat
            blocks = padded.view(num_blocks, block_size)
            scales = blocks.abs().amax(dim=1).clamp_min(1.0e-12) / 127.0
            quantized = torch.round(blocks / scales[:, None]).clamp_(-127, 127).to(torch.int8)
            return quantized.reshape(-1)[:numel].contiguous().view(tensor.shape), scales

        @staticmethod
        def dequantize_blockwise(
            quantized: torch.Tensor,
            scales: torch.Tensor,
            block_size: int,
            dtype: torch.dtype,
        ) -> torch.Tensor:
            flat_q = quantized.reshape(-1).to(torch.float32)
            scale_per_element = scales.repeat_interleave(block_size)[: flat_q.numel()]
            return (flat_q * scale_per_element).view(quantized.shape).to(dtype)

        def __init__(
            self,
            params,
            lr: float,
            momentum: float = 0.95,
            weight_decay: float = 0.0,
            ns_steps: int = 5,
            nesterov: bool = True,
            lr_adjustment: str = "original",
            block_size: int = DEFAULT_MUON_QUANT_BLOCK_SIZE,
        ):
            super().__init__(
                params,
                lr=lr,
                momentum=momentum,
                weight_decay=weight_decay,
                ns_steps=ns_steps,
                nesterov=nesterov,
                lr_adjustment=lr_adjustment,
            )
            for group in self.param_groups:
                group["block_size"] = block_size

        @torch.no_grad()
        def step(self, closure=None):
            loss = None
            if closure is not None:
                with torch.enable_grad():
                    loss = closure()
            for group in self.param_groups:
                lr = group["lr"]
                momentum = group["momentum"]
                weight_decay = group["weight_decay"]
                ns_steps = group["ns_steps"]
                nesterov = group["nesterov"]
                lr_adjustment = group["lr_adjustment"]
                block_size = group["block_size"]
                for p in group["params"]:
                    if p.grad is None:
                        continue
                    grad = p.grad
                    if grad.ndim != 2:
                        raise RuntimeError("Muon only supports 2D matrix parameters")
                    state = self.state[p]
                    if "momentum_q" in state:
                        buf = self.dequantize_blockwise(
                            state["momentum_q"],
                            state["momentum_scale"],
                            block_size,
                            grad.dtype,
                        )
                    else:
                        buf = torch.zeros_like(grad)
                    buf.lerp_(grad, 1.0 - momentum)
                    update = grad.lerp(buf, momentum) if nesterov else buf
                    update = self.zeropower_via_newtonschulz5(update, ns_steps)
                    if weight_decay:
                        p.mul_(1.0 - lr * weight_decay)
                    adjusted_lr = self.adjust_lr(lr, lr_adjustment, p.shape)
                    p.add_(update, alpha=-adjusted_lr)
                    state["momentum_q"], state["momentum_scale"] = self.quantize_blockwise(
                        buf,
                        block_size,
                    )
                    state.pop("momentum_buffer", None)
            return loss

    class NorMuon(torch.optim.Optimizer):
        def __init__(
            self,
            params,
            lr: float,
            beta1: float = 0.95,
            beta2: float = DEFAULT_NORMUON_BETA2,
            eps: float = DEFAULT_NORMUON_EPS,
            weight_decay: float = 0.0,
            ns_steps: int = 5,
        ):
            super().__init__(
                params,
                dict(
                    lr=lr,
                    beta1=beta1,
                    beta2=beta2,
                    eps=eps,
                    weight_decay=weight_decay,
                    ns_steps=ns_steps,
                ),
            )

        @torch.no_grad()
        def step(self, closure=None):
            loss = None
            if closure is not None:
                with torch.enable_grad():
                    loss = closure()
            for group in self.param_groups:
                lr = group["lr"]
                beta1 = group["beta1"]
                beta2 = group["beta2"]
                eps = group["eps"]
                weight_decay = group["weight_decay"]
                ns_steps = group["ns_steps"]
                for p in group["params"]:
                    if p.grad is None:
                        continue
                    grad = p.grad
                    if grad.ndim != 2:
                        raise RuntimeError("NorMuon only supports 2D matrix parameters")
                    state = self.state[p]
                    if "momentum_buffer" not in state:
                        state["momentum_buffer"] = torch.zeros_like(grad)
                    buf = state["momentum_buffer"]
                    buf.lerp_(grad, 1.0 - beta1)
                    update = Muon.zeropower_via_newtonschulz5(buf, ns_steps)
                    row_stat = update.to(torch.float32).square().mean(dim=1)
                    if "row_second_moment" not in state:
                        state["row_second_moment"] = torch.zeros(
                            update.shape[0],
                            device=update.device,
                            dtype=torch.float32,
                        )
                    second = state["row_second_moment"]
                    second.lerp_(row_stat, 1.0 - beta2)
                    denom = torch.sqrt(second + eps).to(update.dtype).unsqueeze(1)
                    normalized_update = update / denom
                    update_scale = (
                        0.2
                        * lr
                        * math.sqrt(p.numel())
                        / normalized_update.norm().clamp_min(1.0e-12)
                    )
                    normalized_update.mul_(update_scale)
                    if weight_decay:
                        p.mul_(1.0 - lr * weight_decay)
                    p.add_(normalized_update, alpha=-1.0)
            return loss

    def make_optimizer():
        peft_optimizer_model = getattr(model, "_orig_mod", model)
        if resolved_optimizer_name in {"loraplus_adamw", "loraplus_adamw8bit"}:
            from peft.optimizers import create_loraplus_optimizer

            optimizer_kwargs: dict[str, Any] = {
                "betas": (0.9, 0.95),
                "eps": 1.0e-8,
                "loraplus_weight_decay": weight_decay,
                "loraplus_lr_embedding": loraplus_lr_embedding,
            }
            if resolved_optimizer_name == "loraplus_adamw8bit":
                import bitsandbytes as bnb

                optimizer_cls = bnb.optim.AdamW8bit
            else:
                optimizer_cls = torch.optim.AdamW
                optimizer_kwargs["fused"] = True
            try:
                optimizer = create_loraplus_optimizer(
                    model=peft_optimizer_model,
                    optimizer_cls=optimizer_cls,
                    lr=lr,
                    loraplus_lr_ratio=loraplus_lr_ratio,
                    **optimizer_kwargs,
                )
            except TypeError:
                if optimizer_kwargs.pop("fused", None) is None:
                    raise
                optimizer = create_loraplus_optimizer(
                    model=peft_optimizer_model,
                    optimizer_cls=optimizer_cls,
                    lr=lr,
                    loraplus_lr_ratio=loraplus_lr_ratio,
                    **optimizer_kwargs,
                )
            print(
                f"optimizer LoRA+: {resolved_optimizer_name} "
                f"ratio={loraplus_lr_ratio} base_lr={lr}",
                flush=True,
            )
            return optimizer
        if resolved_optimizer_name == "lorafa":
            from peft.optimizers import create_lorafa_optimizer

            print(
                f"optimizer LoRA-FA: r={lora_r} alpha={lora_alpha} "
                f"lr={lr} use_rslora={lora_use_rslora}",
                flush=True,
            )
            return create_lorafa_optimizer(
                model=peft_optimizer_model,
                r=lora_r,
                lora_alpha=lora_alpha,
                lr=lr,
                weight_decay=weight_decay,
                use_rslora=lora_use_rslora,
            )
        if resolved_optimizer_name == "adamw8bit":
            import bitsandbytes as bnb

            return bnb.optim.AdamW8bit(
                trainable_params,
                lr=lr,
                betas=(0.9, 0.95),
                eps=1.0e-8,
                weight_decay=weight_decay,
            )
        if resolved_optimizer_name in {"muon", "muon8", "normuon"}:
            matrix_params: list[torch.nn.Parameter] = []
            adamw_params: list[torch.nn.Parameter] = []
            for name, parameter in named_trainable_params:
                clean_name = name.removeprefix("_orig_mod.")
                is_embed_or_head = any(part in clean_name for part in ("embed", "lm_head"))
                if parameter.ndim == 2 and not is_embed_or_head:
                    matrix_params.append(parameter)
                else:
                    adamw_params.append(parameter)

            optimizers: list[torch.optim.Optimizer] = []
            if matrix_params:
                if resolved_optimizer_name == "muon8":
                    optimizers.append(
                        QuantizedMuon(
                            matrix_params,
                            lr=lr,
                            momentum=0.95,
                            weight_decay=weight_decay,
                            lr_adjustment=muon_lr_adjustment,
                            block_size=muon_quant_block_size,
                        )
                    )
                elif resolved_optimizer_name == "normuon":
                    optimizers.append(
                        NorMuon(
                            matrix_params,
                            lr=lr,
                            beta1=0.95,
                            beta2=normuon_beta2,
                            eps=normuon_eps,
                            weight_decay=weight_decay,
                        )
                    )
                else:
                    optimizers.append(
                        Muon(
                            matrix_params,
                            lr=lr,
                            momentum=0.95,
                            weight_decay=weight_decay,
                            lr_adjustment=muon_lr_adjustment,
                        )
                    )
            adamw_name = "none"
            if adamw_params:
                try:
                    import bitsandbytes as bnb

                    adamw = bnb.optim.AdamW8bit(
                        adamw_params,
                        lr=lr,
                        betas=(0.9, 0.95),
                        eps=1.0e-8,
                        weight_decay=weight_decay,
                    )
                    adamw_name = "adamw8bit"
                except Exception as exc:
                    print(
                        f"warning: bitsandbytes AdamW8bit unavailable for {resolved_optimizer_name} tail: {exc}",
                        flush=True,
                    )
                    adamw = torch.optim.AdamW(
                        adamw_params,
                        lr=lr,
                        betas=(0.9, 0.95),
                        eps=1.0e-8,
                        weight_decay=weight_decay,
                    )
                    adamw_name = "adamw"
                optimizers.append(adamw)
            if resolved_optimizer_name == "normuon":
                details = f"beta1=0.95 beta2={normuon_beta2} eps={normuon_eps}"
            else:
                details = f"lr_adjustment={muon_lr_adjustment}"
            if resolved_optimizer_name == "muon8":
                details += f" quant=linear8 block_size={muon_quant_block_size}"
            print(
                f"optimizer {resolved_optimizer_name}: {len(matrix_params)} matrix tensors, "
                f"{adamw_name}: {len(adamw_params)} non-matrix tensors, {details}",
                flush=True,
            )
            return optimizers
        kwargs: dict[str, Any] = {
            "lr": lr,
            "betas": (0.9, 0.95),
            "eps": 1.0e-8,
            "weight_decay": weight_decay,
        }
        try:
            return torch.optim.AdamW(trainable_params, fused=True, **kwargs)
        except TypeError:
            return torch.optim.AdamW(trainable_params, **kwargs)

    optimizer = make_optimizer()

    def iter_optimizer_groups():
        if isinstance(optimizer, list):
            for opt in optimizer:
                for group in opt.param_groups:
                    yield group
            return
        for group in optimizer.param_groups:
            yield group

    def capture_optimizer_base_lrs() -> None:
        for group in iter_optimizer_groups():
            group["base_lr"] = float(group["lr"])

    def set_optimizer_lr_multiplier(multiplier: float) -> None:
        for group in iter_optimizer_groups():
            group["lr"] = float(group.get("base_lr", lr)) * multiplier

    capture_optimizer_base_lrs()
    torch.cuda.synchronize()
    log_gpu("after_optimizer_init")

    def optimizer_zero_grad() -> None:
        if isinstance(optimizer, list):
            for opt in optimizer:
                opt.zero_grad(set_to_none=True)
            return
        optimizer.zero_grad(set_to_none=True)

    instant_lowpass_refresh_calls = 0
    instant_lowpass_refresh_weight_updates = 0
    instant_lowpass_refresh_launch_seconds = 0.0

    def optimizer_step() -> None:
        nonlocal instant_lowpass_refresh_calls
        nonlocal instant_lowpass_refresh_launch_seconds
        nonlocal instant_lowpass_refresh_weight_updates
        if isinstance(optimizer, list):
            for opt in optimizer:
                opt.step()
        else:
            optimizer.step()
        if refresh_instant_lowpass_merged_weights is not None:
            refresh_start = time.monotonic()
            refreshed = refresh_instant_lowpass_merged_weights(force=True)
            instant_lowpass_refresh_launch_seconds += time.monotonic() - refresh_start
            instant_lowpass_refresh_calls += 1
            instant_lowpass_refresh_weight_updates += refreshed

    @torch.no_grad()
    def evaluate(label: str) -> float:
        model.eval()
        weighted_loss_sum = 0.0
        total_supervised = 0
        for start in range(0, eval_blocks, eval_micro_batch_size):
            batch = eval_input_ids[start : start + eval_micro_batch_size].to(device, non_blocking=True)
            attention_mask = eval_attention_mask[start : start + eval_micro_batch_size].to(
                device,
                non_blocking=True,
            )
            labels = eval_labels[start : start + eval_micro_batch_size].to(device, non_blocking=True)
            supervised = _supervised_token_count(labels.detach().cpu())
            if supervised == 0:
                continue
            with torch.autocast("cuda", dtype=torch.bfloat16):
                output = model(input_ids=batch, attention_mask=attention_mask, labels=labels, use_cache=False)
            weighted_loss_sum += float(output.loss.detach().cpu()) * supervised
            total_supervised += supervised
        if total_supervised == 0:
            raise RuntimeError("evaluation batch had no supervised tokens")
        loss = weighted_loss_sum / total_supervised
        log_metric(
            {"event": label, "eval_loss": loss, "eval_supervised_tokens": total_supervised},
            include_gpu=True,
        )
        model.train()
        set_visual_eval(model)
        return loss

    batch_iter = train_batches()

    def is_cuda_oom(exc: BaseException) -> bool:
        return isinstance(exc, torch.cuda.OutOfMemoryError) or "out of memory" in str(exc).lower()

    def refresh_trainable_params() -> None:
        nonlocal named_trainable_params, trainable_params, trainable_count, total_count
        named_trainable_params = [(n, p) for n, p in model.named_parameters() if p.requires_grad]
        trainable_params = [p for _, p in named_trainable_params]
        trainable_count = sum(p.numel() for p in trainable_params)
        total_count = sum(p.numel() for p in model.parameters())

    def run_train_warmup() -> None:
        print("running train/compile warmup", flush=True)
        model.train()
        optimizer_zero_grad()
        warmup_losses: list[torch.Tensor] = []
        warmup_supervised_counts: list[int] = []
        for _ in range(grad_accum):
            warmup_batch = next(batch_iter)
            input_ids = warmup_batch["input_ids"].to(device, non_blocking=True)
            attention_mask = warmup_batch["attention_mask"].to(device, non_blocking=True)
            labels = warmup_batch["labels"].to(device, non_blocking=True)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                warmup_loss = model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    labels=labels,
                    use_cache=False,
                ).loss
            warmup_losses.append(warmup_loss)
            warmup_supervised_counts.append(_supervised_token_count(labels.detach().cpu()))
        warmup_supervised_total = sum(warmup_supervised_counts)
        if warmup_supervised_total <= 0:
            raise RuntimeError("warmup batch had no supervised tokens")
        for warmup_loss, supervised in zip(warmup_losses, warmup_supervised_counts):
            (warmup_loss * (supervised / warmup_supervised_total)).backward()
        optimizer_zero_grad()
        torch.cuda.synchronize()

    baseline_loss = evaluate("baseline_eval")
    compile_warmup_start = time.monotonic()
    log_gpu("compile_warmup_start")

    if compile_model:
        print(f"compiling model with torch.compile(mode={compile_mode!r})", flush=True)
        model = torch.compile(model, dynamic=False, mode=compile_mode)

    if compile_warmup:
        try:
            run_train_warmup()
        except RuntimeError as exc:
            if not (
                tuning_mode == "lora"
                and gradient_checkpointing == "auto"
                and not checkpointing_enabled
                and is_cuda_oom(exc)
            ):
                raise
            print(
                "LoRA warmup OOM without gradient checkpointing; "
                "retrying warmup with checkpointing enabled",
                flush=True,
            )
            gradient_checkpointing_fallback_used = True
            checkpointing_enabled = True
            config["gradient_checkpointing_enabled"] = checkpointing_enabled
            config["gradient_checkpointing_fallback_used"] = gradient_checkpointing_fallback_used
            try:
                optimizer_zero_grad()
            except Exception:
                pass
            model = uncompiled_model
            set_gradient_checkpointing(model, True)
            torch.cuda.empty_cache()
            if compile_model:
                print(f"recompiling model with torch.compile(mode={compile_mode!r})", flush=True)
                model = torch.compile(model, dynamic=False, mode=compile_mode)
            refresh_trainable_params()
            optimizer = make_optimizer()
            capture_optimizer_base_lrs()
            torch.cuda.synchronize()
            log_gpu("after_optimizer_reinit")
            batch_iter = train_batches()
            write_config()
            run_train_warmup()

    optimizer_zero_grad()
    torch.cuda.reset_peak_memory_stats(gpu_index)
    peak_gpu_stats.clear()
    log_gpu("budget_start")
    log_gpu("before_train_loop")
    budget_start = time.monotonic()
    train_deadline = budget_start + minutes * 60.0
    train_loop_start = budget_start

    full_decay_start_time: float | None = None

    def compute_lr_multiplier(step_value: int, now: float) -> float:
        nonlocal full_decay_start_time
        if warmup_steps > 0 and step_value <= warmup_steps:
            return step_value / warmup_steps
        if lr_schedule in {"linear", "cosine"}:
            if full_decay_start_time is None:
                full_decay_start_time = now
            decay_seconds = max(train_deadline - full_decay_start_time, 1.0e-9)
            decay_progress = min(1.0, max(0.0, (now - full_decay_start_time) / decay_seconds))
            if lr_schedule == "linear":
                return min_lr_ratio + (1.0 - min_lr_ratio) * (1.0 - decay_progress)
            cosine = 0.5 * (1.0 + math.cos(math.pi * decay_progress))
            return min_lr_ratio + (1.0 - min_lr_ratio) * cosine
        if lr_schedule == "wsd":
            decay_window_seconds = minutes * 60.0 * lr_decay_fraction
            if decay_window_seconds > 0.0:
                decay_start = train_deadline - decay_window_seconds
                if now >= decay_start:
                    decay_progress = min(1.0, max(0.0, (now - decay_start) / decay_window_seconds))
                    cosine = 0.5 * (1.0 + math.cos(math.pi * decay_progress))
                    return min_lr_ratio + (1.0 - min_lr_ratio) * cosine
        return 1.0

    step = 0
    tokens = 0
    supervised_tokens_seen = 0
    last_loss = math.nan
    memory_probe_records: list[dict[str, Any]] = []

    def log_memory_probe(step_value: int, phase: str) -> None:
        torch.cuda.synchronize()
        stats = collect_gpu_stats()
        record = {
            "event": "memory_probe",
            "step": step_value,
            "phase": phase,
            **stats,
        }
        memory_probe_records.append(record)
        log_metric(record)

    while time.monotonic() < train_deadline:
        optimizer_zero_grad()
        next_step = step + 1
        probe_this_step = memory_probe_steps > 0 and next_step <= memory_probe_steps
        if probe_this_step:
            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats(gpu_index)
            log_memory_probe(next_step, "step_start")
        accum_losses: list[float] = []
        accum_supervised_tokens = 0
        accum_loss_tensors: list[tuple[torch.Tensor, int]] = []
        for _ in range(grad_accum):
            batch = next(batch_iter)
            input_ids = batch["input_ids"].to(device, non_blocking=True)
            attention_mask = batch["attention_mask"].to(device, non_blocking=True)
            labels = batch["labels"].to(device, non_blocking=True)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                output = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels, use_cache=False)
            accum_losses.append(float(output.loss.detach().cpu()))
            tokens += int(input_ids.numel())
            batch_supervised = _supervised_token_count(labels.detach().cpu())
            if batch_supervised <= 0:
                raise RuntimeError("training batch had no supervised tokens")
            accum_supervised_tokens += batch_supervised
            supervised_tokens_seen += batch_supervised
            accum_loss_tensors.append((output.loss, batch_supervised))

        if probe_this_step:
            log_memory_probe(next_step, "after_forward")

        for loss_tensor, batch_supervised in accum_loss_tensors:
            loss = loss_tensor * (batch_supervised / accum_supervised_tokens)
            loss.backward()

        if probe_this_step:
            log_memory_probe(next_step, "after_backward")

        step += 1
        lr_now = time.monotonic()
        lr_multiplier = compute_lr_multiplier(step, lr_now)
        step_lr = lr * lr_multiplier
        set_optimizer_lr_multiplier(lr_multiplier)
        torch.nn.utils.clip_grad_norm_(trainable_params, 1.0)
        optimizer_step()
        if probe_this_step:
            log_memory_probe(step, "after_optimizer_step")
        last_loss = float(np.mean(accum_losses))

        if log_every > 0 and (step == 1 or step % log_every == 0):
            now = time.monotonic()
            elapsed_budget_seconds = now - budget_start
            elapsed_train_loop_seconds = now - train_loop_start
            log_metric(
                {
                    "event": "train",
                    "step": step,
                    "train_loss": last_loss,
                    "lr": step_lr,
                    "lr_multiplier": lr_multiplier,
                    "tokens": tokens,
                    "supervised_tokens": supervised_tokens_seen,
                    "step_supervised_tokens": accum_supervised_tokens,
                    "elapsed_compile_warmup_seconds": budget_start - compile_warmup_start,
                    "elapsed_budget_seconds": elapsed_budget_seconds,
                    "elapsed_train_loop_seconds": elapsed_train_loop_seconds,
                    "tokens_per_second": tokens / max(elapsed_budget_seconds, 1.0e-9),
                    "supervised_tokens_per_second": supervised_tokens_seen
                    / max(elapsed_budget_seconds, 1.0e-9),
                    "train_loop_tokens_per_second": tokens / max(elapsed_train_loop_seconds, 1.0e-9),
                    "train_loop_supervised_tokens_per_second": supervised_tokens_seen
                    / max(elapsed_train_loop_seconds, 1.0e-9),
                },
                include_gpu=True,
            )

    budget_end = time.monotonic()
    elapsed_budget_seconds = budget_end - budget_start
    elapsed_train_loop_seconds = budget_end - train_loop_start
    elapsed_compile_warmup_seconds = budget_start - compile_warmup_start
    # Keep post-budget evaluation from triggering a new compiled eval graph.
    model = uncompiled_model
    final_loss = evaluate("final_eval")
    instant_lowpass_runtime_stats: dict[str, int] = {}
    if activation_compression_mode == "instant-linear":
        import sys

        if "/root" not in sys.path:
            sys.path.append("/root")
        from instant_lowpass import collect_instant_lowpass_stats

        instant_lowpass_runtime_stats = collect_instant_lowpass_stats(uncompiled_model)
    summary = {
        **config,
        **instant_lowpass_runtime_stats,
        **peak_gpu_stats,
        "instant_lowpass_refresh_calls": instant_lowpass_refresh_calls
        if activation_compression_mode == "instant-linear"
        else None,
        "instant_lowpass_refresh_weight_updates": instant_lowpass_refresh_weight_updates
        if activation_compression_mode == "instant-linear"
        else None,
        "instant_lowpass_refresh_launch_seconds": instant_lowpass_refresh_launch_seconds
        if activation_compression_mode == "instant-linear"
        else None,
        "record_date": dt.date.today().isoformat(),
        "run_dir": str(run_dir),
        "eval_cache": str(eval_path),
        "train_skip_docs": train_skip_docs,
        "eval_supervised_tokens": eval_supervised_tokens,
        "trainable_params": trainable_count,
        "total_params": total_count,
        "steps": step,
        "tokens": tokens,
        "supervised_tokens": supervised_tokens_seen,
        "elapsed_compile_warmup_seconds": elapsed_compile_warmup_seconds,
        "elapsed_budget_seconds": elapsed_budget_seconds,
        "elapsed_train_loop_seconds": elapsed_train_loop_seconds,
        "elapsed_train_seconds": elapsed_budget_seconds,
        "tokens_per_second": tokens / max(elapsed_budget_seconds, 1.0e-9),
        "supervised_tokens_per_second": supervised_tokens_seen / max(elapsed_budget_seconds, 1.0e-9),
        "train_loop_tokens_per_second": tokens / max(elapsed_train_loop_seconds, 1.0e-9),
        "train_loop_supervised_tokens_per_second": supervised_tokens_seen
        / max(elapsed_train_loop_seconds, 1.0e-9),
        "last_train_loss": last_loss,
        "baseline_eval_loss": baseline_loss,
        "final_eval_loss": final_loss,
        "eval_loss_drop": baseline_loss - final_loss,
        "memory_probe_records": memory_probe_records,
    }

    if save_final:
        unwrapped_model = getattr(model, "_orig_mod", model)
        final_dir = run_dir / ("final_adapter" if tuning_mode == "lora" else "final_model")
        unwrapped_model.save_pretrained(final_dir)
        tokenizer.save_pretrained(final_dir)
        if tuning_mode == "lora":
            summary["final_adapter_dir"] = str(final_dir)
        else:
            summary["final_model_dir"] = str(final_dir)

    (run_dir / "summary.json").write_text(json.dumps(summary, indent=2, default=_json_default) + "\n")
    log_metric({"event": "summary", **summary})
    if wandb_run is not None:
        wandb_run.summary.update(summary)
        wandb_run.finish()

    record_artifacts = None
    if record_description:
        src_path = Path(__file__).resolve()
        instant_lowpass_path = src_path.parent / "instant_lowpass.py"
        instant_lowpass_triton_path = src_path.parent / "instant_lowpass_triton.py"
        record_artifacts = {
            "main.py": src_path.read_text() if src_path.exists() else "",
            "instant_lowpass.py": instant_lowpass_path.read_text()
            if instant_lowpass_path.exists()
            else "",
            "instant_lowpass_triton.py": instant_lowpass_triton_path.read_text()
            if instant_lowpass_triton_path.exists()
            else "",
            "config.json": json.dumps(config, indent=2, default=_json_default) + "\n",
            "summary.json": json.dumps(summary, indent=2, default=_json_default) + "\n",
            "record.txt": _format_record_text(summary),
            "metrics.jsonl": metrics_path.read_text() if metrics_path.exists() else "",
        }

    cache_volume.commit()
    if record_artifacts is not None:
        summary["_record_artifacts"] = record_artifacts
    return summary


@app.local_entrypoint()
def main(
    minutes: float = 0.0,
    seq_len: int = 4096,
    micro_batch_size: int = 0,
    grad_accum: int = 0,
    lr: float = 0.0,
    weight_decay: float = -1.0,
    warmup_steps: int = 20,
    eval_blocks: int = 64,
    eval_micro_batch_size: int = 0,
    seed: int = 1337,
    model_id: str = DEFAULT_MODEL_ID,
    model_revision: str = DEFAULT_MODEL_REVISION,
    dataset_id: str = "",
    dataset_config: str = "",
    dataset_revision: str = "",
    data_mode: str = "",
    sft_pack_block_size: int = 0,
    tuning_mode: str = "lora",
    adapter_mode: str = "",
    optimizer_name: str = "auto",
    gradient_checkpointing: str = "auto",
    activation_compression_mode: str = "off",
    instant_projector_kind: str = "hadamard",
    instant_chunk_size: int = 64,
    instant_keep: int = 32,
    instant_min_hidden_dim: int = 64,
    instant_hadamard_backend: str = "auto",
    instant_parameter_gradient: str = "projected_lowpass",
    instant_target_modules: str = "trainable",
    lora_r: int = 32,
    lora_alpha: int = 64,
    lora_dropout: float = 0.0,
    lora_target_modules: str = "all-linear",
    gralora_k: int = DEFAULT_GRALORA_K,
    lora_use_rslora: bool = True,
    lora_use_dora: bool = False,
    lora_init: str = "default",
    lora_eva_rho: float = DEFAULT_LORA_EVA_RHO,
    lora_eva_batches: int = 16,
    lora_ga_batches: int = 4,
    lora_ga_micro_batch_size: int = 1,
    lora_ga_direction: str = "ArB2r",
    lora_ga_scale: str = "stable",
    lora_ga_stable_gamma: int = 16,
    lora_ga_cache: bool = False,
    loraplus_lr_ratio: float = DEFAULT_LORAPLUS_LR_RATIO,
    loraplus_lr_embedding: float = 1.0e-6,
    muon_lr_adjustment: str = "match_rms_adamw",
    muon_quant_block_size: int = DEFAULT_MUON_QUANT_BLOCK_SIZE,
    normuon_beta2: float = DEFAULT_NORMUON_BETA2,
    normuon_eps: float = DEFAULT_NORMUON_EPS,
    lr_schedule: str = "constant",
    lr_decay_fraction: float = 0.1,
    min_lr_ratio: float = 0.0,
    attn_implementation: str = "flex_attention",
    compile_model: bool = True,
    compile_mode: str = "max-autotune-no-cudagraphs",
    compile_warmup: bool = True,
    memory_probe_steps: int = 0,
    save_final: bool = False,
    log_every: int = 5,
    wandb_project: str = "",
    wandb_entity: str = "",
    wandb_name: str = "",
    wandb_group: str = "",
    wandb_tags: str = "",
    wandb_mode: str = "online",
    track: str = "1",
    record_description: str = "",
    record_contributors: str = "",
) -> None:
    """Run a modded-continued-training track on Modal.

    --track selects the competition track (1=30min, 2=5min, 3=2hr).
    Track 1 defaults to packed SFT/GraLoRA; Tracks 2/3 default to legacy CPT/LoRA.
    When --minutes is 0 (default), the track's default budget is used.
    Set --record-description to save a competition record on success.
    """

    if track not in TRACKS:
        raise ValueError(f"--track must be one of: {', '.join(TRACKS.keys())}")
    if not data_mode:
        data_mode = "sft" if track == "1" else "cpt"
    if not adapter_mode:
        adapter_mode = DEFAULT_ADAPTER_MODE if tuning_mode == "lora" and data_mode == "sft" else "lora"

    if minutes == 0.0:
        minutes = TRACKS[track]["default_minutes"]

    summary = run_track1.remote(
        minutes=minutes,
        seq_len=seq_len,
        micro_batch_size=micro_batch_size,
        grad_accum=grad_accum,
        lr=lr,
        weight_decay=weight_decay,
        warmup_steps=warmup_steps,
        eval_blocks=eval_blocks,
        eval_micro_batch_size=eval_micro_batch_size,
        seed=seed,
        model_id=model_id,
        model_revision=model_revision,
        dataset_id=dataset_id,
        dataset_config=dataset_config,
        dataset_revision=dataset_revision,
        data_mode=data_mode,  # type: ignore[arg-type]
        sft_pack_block_size=sft_pack_block_size,
        tuning_mode=tuning_mode,  # type: ignore[arg-type]
        adapter_mode=adapter_mode,  # type: ignore[arg-type]
        optimizer_name=optimizer_name,
        gradient_checkpointing=gradient_checkpointing,  # type: ignore[arg-type]
        activation_compression_mode=activation_compression_mode,  # type: ignore[arg-type]
        instant_projector_kind=instant_projector_kind,  # type: ignore[arg-type]
        instant_chunk_size=instant_chunk_size,
        instant_keep=instant_keep,
        instant_min_hidden_dim=instant_min_hidden_dim,
        instant_hadamard_backend=instant_hadamard_backend,  # type: ignore[arg-type]
        instant_parameter_gradient=instant_parameter_gradient,
        instant_target_modules=instant_target_modules,  # type: ignore[arg-type]
        lora_r=lora_r,
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
        lora_target_modules=lora_target_modules,
        gralora_k=gralora_k,
        lora_use_rslora=lora_use_rslora,
        lora_use_dora=lora_use_dora,
        lora_init=lora_init,
        lora_eva_rho=lora_eva_rho,
        lora_eva_batches=lora_eva_batches,
        lora_ga_batches=lora_ga_batches,
        lora_ga_micro_batch_size=lora_ga_micro_batch_size,
        lora_ga_direction=lora_ga_direction,
        lora_ga_scale=lora_ga_scale,
        lora_ga_stable_gamma=lora_ga_stable_gamma,
        lora_ga_cache=lora_ga_cache,
        loraplus_lr_ratio=loraplus_lr_ratio,
        loraplus_lr_embedding=loraplus_lr_embedding,
        muon_lr_adjustment=muon_lr_adjustment,  # type: ignore[arg-type]
        muon_quant_block_size=muon_quant_block_size,
        normuon_beta2=normuon_beta2,
        normuon_eps=normuon_eps,
        lr_schedule=lr_schedule,  # type: ignore[arg-type]
        lr_decay_fraction=lr_decay_fraction,
        min_lr_ratio=min_lr_ratio,
        attn_implementation=attn_implementation,
        compile_model=compile_model,
        compile_mode=compile_mode,
        compile_warmup=compile_warmup,
        memory_probe_steps=memory_probe_steps,
        save_final=save_final,
        log_every=log_every,
        wandb_project=wandb_project,
        wandb_entity=wandb_entity,
        wandb_name=wandb_name,
        wandb_group=wandb_group,
        wandb_tags=wandb_tags,
        wandb_mode=wandb_mode,  # type: ignore[arg-type]
        track=track,
        record_description=record_description,
        record_contributors=record_contributors,
    )
    record_artifacts = summary.pop("_record_artifacts", None)
    if record_description and isinstance(record_artifacts, dict):
        record_dir = _write_local_record(summary, record_artifacts)
        print(f"record saved to {record_dir}")
    print(json.dumps(summary, indent=2, default=_json_default))
