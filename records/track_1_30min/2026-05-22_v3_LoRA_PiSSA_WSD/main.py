"""nanoFineTune: a Modal H100 speedrun for fine-tuning.

Tracks train Qwen3.5-4B-Base on FineMath for a fixed wall-clock budget and
score the run by the drop in heldout next-token loss.

Track 1: 30-minute budget (default)
Track 2: 5-minute sprint
Track 3: 2-hour endurance
"""

from __future__ import annotations

import hashlib
import json
import os
import re
from pathlib import Path
from typing import Any, Literal

import modal


APP_NAME = "nanofinetune"
CACHE_MOUNT = Path("/cache")
HF_CACHE = CACHE_MOUNT / "huggingface"

DEFAULT_MODEL_ID = "Qwen/Qwen3.5-4B-Base"
DEFAULT_MODEL_REVISION = "1001bb4d826a52d1f399e183466143f4da7b741b"
DEFAULT_DATASET_ID = "HuggingFaceTB/finemath"
DEFAULT_DATASET_CONFIG = "finemath-4plus"
DEFAULT_DATASET_REVISION = "e92b25a616738fe95dc186b64dfb19f9c8525594"
DEFAULT_EFFECTIVE_TOKENS_PER_STEP = 32_768
DEFAULT_LORA_MICRO_BATCH_SIZE = 8
DEFAULT_FULL_MICRO_BATCH_SIZE = 1
DEFAULT_EVAL_MICRO_BATCH_SIZE = 2
DEFAULT_LORAPLUS_LR_RATIO = 16.0
DEFAULT_LORA_EVA_RHO = 2.0

OPTIMIZER_CHOICES = {
    "auto",
    "adamw8bit",
    "adamw_fused",
    "muon",
    "loraplus_adamw",
    "loraplus_adamw8bit",
    "lorafa",
}
LR_SCHEDULE_CHOICES = {"constant", "wsd"}
MUON_LR_ADJUSTMENT_CHOICES = {"original", "match_rms_adamw"}
LORA_INIT_CHOICES = {"default", "gaussian", "pissa", "olora", "eva", "orthogonal"}

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


app = modal.App(APP_NAME)
cache_volume = modal.Volume.from_name("nanofinetune-cache", create_if_missing=True)
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
        f"Minutes: {summary['minutes']}\n"
        f"Eval loss drop: {summary['eval_loss_drop']:.6f}\n"
        f"Baseline eval loss: {summary['baseline_eval_loss']:.6f}\n"
        f"Final eval loss: {summary['final_eval_loss']:.6f}\n"
        f"Steps: {summary['steps']}\n"
        f"Tokens: {summary['tokens']:,}\n"
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

    for filename in ("main.py", "config.json", "summary.json", "record.txt", "metrics.jsonl"):
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
    dataset_id: str = DEFAULT_DATASET_ID,
    dataset_config: str = DEFAULT_DATASET_CONFIG,
    dataset_revision: str = DEFAULT_DATASET_REVISION,
    tuning_mode: Literal["lora", "full"] = "lora",
    optimizer_name: Literal[
        "auto",
        "adamw8bit",
        "adamw_fused",
        "muon",
        "loraplus_adamw",
        "loraplus_adamw8bit",
        "lorafa",
    ] = "auto",
    gradient_checkpointing: Literal["auto", "true", "false"] = "auto",
    lora_r: int = 32,
    lora_alpha: int = 64,
    lora_dropout: float = 0.0,
    lora_target_modules: str = "all-linear",
    lora_use_rslora: bool = True,
    lora_use_dora: bool = False,
    lora_init: str = "default",
    lora_eva_rho: float = DEFAULT_LORA_EVA_RHO,
    lora_eva_batches: int = 16,
    loraplus_lr_ratio: float = DEFAULT_LORAPLUS_LR_RATIO,
    loraplus_lr_embedding: float = 1.0e-6,
    muon_lr_adjustment: Literal["original", "match_rms_adamw"] = "match_rms_adamw",
    lr_schedule: Literal["constant", "wsd"] = "constant",
    lr_decay_fraction: float = 0.1,
    min_lr_ratio: float = 0.0,
    attn_implementation: Literal["flex_attention", "flash_attention_2", "sdpa", "eager"] = "flex_attention",
    compile_model: bool = True,
    compile_mode: str = "max-autotune-no-cudagraphs",
    compile_warmup: bool = True,
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
    if tuning_mode not in {"lora", "full"}:
        raise ValueError("--tuning-mode must be one of: lora, full")
    gradient_checkpointing = str(gradient_checkpointing).lower()
    if gradient_checkpointing not in {"auto", "true", "false"}:
        raise ValueError("--gradient-checkpointing must be one of: auto, true, false")
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
    lora_init = str(lora_init).lower()
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
    if loraplus_lr_ratio < 1.0:
        raise ValueError("--loraplus-lr-ratio must be >= 1.0")
    if loraplus_lr_embedding <= 0.0:
        raise ValueError("--loraplus-lr-embedding must be positive")
    if muon_lr_adjustment not in MUON_LR_ADJUSTMENT_CHOICES:
        raise ValueError(
            "--muon-lr-adjustment must be one of: "
            f"{', '.join(sorted(MUON_LR_ADJUSTMENT_CHOICES))}"
        )
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
    if track not in TRACKS:
        raise ValueError(f"--track must be one of: {', '.join(TRACKS.keys())}")
    if optimizer_name in {"loraplus_adamw", "loraplus_adamw8bit", "lorafa"} and tuning_mode != "lora":
        raise ValueError(f"--optimizer-name {optimizer_name} requires --tuning-mode lora")
    if lora_init == "eva" and tuning_mode != "lora":
        raise ValueError(f"--lora-init {lora_init} requires --tuning-mode lora")
    if lora_use_dora and tuning_mode != "lora":
        raise ValueError("--lora-use-dora requires --tuning-mode lora")

    track_info = TRACKS[track]
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
    wandb_tags_list = [tag.strip() for tag in wandb_tags.split(",") if tag.strip()]
    wandb_enabled = bool(wandb_project) and wandb_mode != "disabled"
    lr = lr if lr > 0.0 else (2.0e-4 if tuning_mode == "lora" else 2.0e-5)
    weight_decay = weight_decay if weight_decay >= 0.0 else (0.01 if tuning_mode == "lora" else 0.1)
    resolved_optimizer_name = (
        "adamw_fused"
        if optimizer_name == "auto" and tuning_mode == "lora"
        else "adamw8bit"
        if optimizer_name == "auto"
        else optimizer_name
    )
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

    run_id = dt.datetime.now(dt.UTC).strftime("%Y%m%d-%H%M%S")
    run_dir = CACHE_MOUNT / "runs" / run_id
    eval_dir = CACHE_MOUNT / "eval"
    run_dir.mkdir(parents=True, exist_ok=True)
    eval_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = run_dir / "metrics.jsonl"

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
        "tuning_mode": tuning_mode,
        "optimizer_name": resolved_optimizer_name,
        "requested_optimizer_name": requested_optimizer_name,
        "gradient_checkpointing": gradient_checkpointing,
        "gradient_checkpointing_enabled": checkpointing_enabled,
        "gradient_checkpointing_fallback_used": gradient_checkpointing_fallback_used,
        "lora_r": lora_r if tuning_mode == "lora" else None,
        "lora_alpha": lora_alpha if tuning_mode == "lora" else None,
        "lora_dropout": lora_dropout if tuning_mode == "lora" else None,
        "lora_target_modules": lora_target_modules if tuning_mode == "lora" else None,
        "lora_use_rslora": lora_use_rslora if tuning_mode == "lora" else None,
        "lora_use_dora": lora_use_dora if tuning_mode == "lora" else None,
        "lora_init": lora_init if tuning_mode == "lora" else None,
        "lora_eva_rho": lora_eva_rho if tuning_mode == "lora" else None,
        "lora_eva_batches": lora_eva_batches if tuning_mode == "lora" else None,
        "loraplus_lr_ratio": loraplus_lr_ratio,
        "loraplus_lr_embedding": loraplus_lr_embedding,
        "muon_lr_adjustment": muon_lr_adjustment,
        "lr_schedule": lr_schedule,
        "lr_decay_fraction": lr_decay_fraction,
        "min_lr_ratio": min_lr_ratio,
        "attn_implementation": attn_implementation,
        "compile_model": compile_model,
        "compile_mode": compile_mode,
        "compile_warmup": compile_warmup,
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

    def dataset_stream(shuffle: bool = False):
        ds = load_dataset(
            dataset_id,
            dataset_config,
            split="train",
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

    def tokenize_text(text: str) -> list[int]:
        ids = tokenizer(text, add_special_tokens=False).input_ids
        if ids:
            ids.append(tokenizer.eos_token_id)
        return ids

    def build_eval_cache() -> tuple[torch.Tensor, int, Path]:
        key_payload = {
            "model": model_id,
            "model_revision": model_revision,
            "dataset": dataset_id,
            "dataset_config": dataset_config,
            "dataset_revision": dataset_revision,
            "seq_len": seq_len,
            "eval_blocks": eval_blocks,
            "seed": seed,
            "kind": "all_token_cpt_packed_v2",
        }
        key = hashlib.sha256(json.dumps(key_payload, sort_keys=True).encode()).hexdigest()[:20]
        eval_path = eval_dir / f"{key}.pt"
        if eval_path.exists():
            payload = torch.load(eval_path, map_location="cpu")
            return payload["input_ids"], int(payload["skip_docs"]), eval_path

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
        torch.save(
            {"input_ids": input_ids, "skip_docs": skip_docs, "key_payload": key_payload},
            eval_path,
        )
        return input_ids, skip_docs, eval_path

    eval_input_ids, train_skip_docs, eval_path = build_eval_cache()
    print(f"fixed eval cache: {eval_path} skip_docs={train_skip_docs}", flush=True)

    def parse_lora_target_modules(value: str) -> str | list[str]:
        value = value.strip()
        if value == "all-linear" or "," not in value:
            return value
        return [part.strip() for part in value.split(",") if part.strip()]

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
        from peft import LoraConfig, TaskType, get_peft_model
        if lora_init == "eva":
            from peft import EvaConfig

        print(
            "applying LoRA "
            f"target_modules={lora_target_modules!r} r={lora_r} alpha={lora_alpha} "
            f"dropout={lora_dropout} use_rslora={lora_use_rslora} "
            f"use_dora={lora_use_dora} init={lora_init}",
            flush=True,
        )
        lora_init_value: bool | str = True if lora_init == "default" else lora_init
        lora_config_kwargs: dict[str, Any] = {
            "task_type": TaskType.CAUSAL_LM,
            "target_modules": parse_lora_target_modules(lora_target_modules),
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
        lora_config = LoraConfig(**lora_config_kwargs)
        model = get_peft_model(model, lora_config, low_cpu_mem_usage=(lora_init == "eva"))

    set_gradient_checkpointing(model, checkpointing_enabled)

    device = torch.device("cuda")
    model.to(device)
    if tuning_mode == "lora" and lora_init == "eva":
        from peft import initialize_lora_eva_weights

        eva_blocks = min(eval_blocks, lora_eva_batches * eval_micro_batch_size)

        def iter_eva_batches():
            for start in range(0, eva_blocks, eval_micro_batch_size):
                batch = eval_input_ids[start : start + eval_micro_batch_size].to(device, non_blocking=True)
                yield {
                    "input_ids": batch,
                    "attention_mask": torch.ones_like(batch, dtype=torch.long),
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

    named_trainable_params = [(n, p) for n, p in model.named_parameters() if p.requires_grad]
    trainable_params = [p for _, p in named_trainable_params]
    if not trainable_params:
        raise RuntimeError("no trainable parameters were found")
    trainable_visual_params = [n for n, p in named_trainable_params if "visual" in n]
    if trainable_visual_params:
        raise RuntimeError(f"visual parameters must remain frozen: {trainable_visual_params[:5]}")
    if tuning_mode == "lora":
        non_lora_trainable = [n for n, p in named_trainable_params if "lora_" not in n]
        if non_lora_trainable:
            raise RuntimeError(f"LoRA mode found non-adapter trainable parameters: {non_lora_trainable[:5]}")
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
        if resolved_optimizer_name == "muon":
            muon_params: list[torch.nn.Parameter] = []
            adamw_params: list[torch.nn.Parameter] = []
            for name, parameter in named_trainable_params:
                clean_name = name.removeprefix("_orig_mod.")
                is_embed_or_head = any(part in clean_name for part in ("embed", "lm_head"))
                if parameter.ndim == 2 and not is_embed_or_head:
                    muon_params.append(parameter)
                else:
                    adamw_params.append(parameter)

            optimizers: list[torch.optim.Optimizer] = []
            if muon_params:
                optimizers.append(
                    Muon(
                        muon_params,
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
                    print(f"warning: bitsandbytes AdamW8bit unavailable for Muon tail: {exc}", flush=True)
                    adamw = torch.optim.AdamW(
                        adamw_params,
                        lr=lr,
                        betas=(0.9, 0.95),
                        eps=1.0e-8,
                        weight_decay=weight_decay,
                    )
                    adamw_name = "adamw"
                optimizers.append(adamw)
            print(
                f"optimizer muon: {len(muon_params)} matrix tensors, "
                f"{adamw_name}: {len(adamw_params)} non-muon tensors, "
                f"lr_adjustment={muon_lr_adjustment}",
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

    def optimizer_step() -> None:
        if isinstance(optimizer, list):
            for opt in optimizer:
                opt.step()
            return
        optimizer.step()

    @torch.no_grad()
    def evaluate(label: str) -> float:
        model.eval()
        losses: list[float] = []
        for start in range(0, eval_blocks, eval_micro_batch_size):
            batch = eval_input_ids[start : start + eval_micro_batch_size].to(device, non_blocking=True)
            attention_mask = torch.ones_like(batch, dtype=torch.long)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                output = model(input_ids=batch, attention_mask=attention_mask, labels=batch, use_cache=False)
            losses.append(float(output.loss.detach().cpu()))
        loss = float(np.mean(losses))
        log_metric({"event": label, "eval_loss": loss}, include_gpu=True)
        model.train()
        set_visual_eval(model)
        return loss

    def train_batches():
        ds = dataset_stream(shuffle=False).skip(train_skip_docs).shuffle(seed=seed, buffer_size=10_000)
        token_buffer: list[int] = []
        batch: list[torch.Tensor] = []
        while True:
            for row in ds:
                text = row.get("text")
                if not isinstance(text, str) or not text.strip():
                    continue
                token_buffer.extend(tokenize_text(text))
                while len(token_buffer) >= seq_len:
                    block = torch.tensor(token_buffer[:seq_len], dtype=torch.long)
                    del token_buffer[:seq_len]
                    batch.append(block)
                    if len(batch) == micro_batch_size:
                        yield torch.stack(batch)
                        batch.clear()
            ds = dataset_stream(shuffle=True).skip(train_skip_docs)

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
        for _ in range(grad_accum):
            warmup_batch = next(batch_iter).to(device, non_blocking=True)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                warmup_loss = model(
                    input_ids=warmup_batch,
                    attention_mask=torch.ones_like(warmup_batch, dtype=torch.long),
                    labels=warmup_batch,
                    use_cache=False,
                ).loss / grad_accum
            warmup_loss.backward()
        optimizer_zero_grad()
        torch.cuda.synchronize()

    baseline_loss = evaluate("baseline_eval")
    budget_start = time.monotonic()
    train_deadline = budget_start + minutes * 60.0
    torch.cuda.reset_peak_memory_stats(gpu_index)
    peak_gpu_stats.clear()
    log_gpu("budget_start")

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

    log_gpu("before_train_loop")
    optimizer_zero_grad()
    train_loop_start = time.monotonic()

    def compute_lr_multiplier(step_value: int, now: float) -> float:
        if warmup_steps > 0 and step_value <= warmup_steps:
            return step_value / warmup_steps
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
    last_loss = math.nan
    while time.monotonic() < train_deadline:
        optimizer_zero_grad()
        accum_losses: list[float] = []
        for _ in range(grad_accum):
            batch = next(batch_iter).to(device, non_blocking=True)
            attention_mask = torch.ones_like(batch, dtype=torch.long)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                output = model(input_ids=batch, attention_mask=attention_mask, labels=batch, use_cache=False)
                loss = output.loss / grad_accum
            loss.backward()
            accum_losses.append(float(output.loss.detach().cpu()))
            tokens += int(batch.numel())

        step += 1
        lr_now = time.monotonic()
        lr_multiplier = compute_lr_multiplier(step, lr_now)
        step_lr = lr * lr_multiplier
        set_optimizer_lr_multiplier(lr_multiplier)
        torch.nn.utils.clip_grad_norm_(trainable_params, 1.0)
        optimizer_step()
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
                    "elapsed_budget_seconds": elapsed_budget_seconds,
                    "elapsed_train_loop_seconds": elapsed_train_loop_seconds,
                    "tokens_per_second": tokens / max(elapsed_budget_seconds, 1.0e-9),
                    "train_loop_tokens_per_second": tokens / max(elapsed_train_loop_seconds, 1.0e-9),
                },
                include_gpu=True,
            )

    budget_end = time.monotonic()
    elapsed_budget_seconds = budget_end - budget_start
    elapsed_train_loop_seconds = budget_end - train_loop_start
    # Keep post-budget evaluation from triggering a new compiled eval graph.
    model = uncompiled_model
    final_loss = evaluate("final_eval")
    summary = {
        **config,
        **peak_gpu_stats,
        "record_date": dt.date.today().isoformat(),
        "run_dir": str(run_dir),
        "eval_cache": str(eval_path),
        "train_skip_docs": train_skip_docs,
        "trainable_params": trainable_count,
        "total_params": total_count,
        "steps": step,
        "tokens": tokens,
        "elapsed_budget_seconds": elapsed_budget_seconds,
        "elapsed_train_loop_seconds": elapsed_train_loop_seconds,
        "elapsed_train_seconds": elapsed_budget_seconds,
        "tokens_per_second": tokens / max(elapsed_budget_seconds, 1.0e-9),
        "train_loop_tokens_per_second": tokens / max(elapsed_train_loop_seconds, 1.0e-9),
        "last_train_loss": last_loss,
        "baseline_eval_loss": baseline_loss,
        "final_eval_loss": final_loss,
        "eval_loss_drop": baseline_loss - final_loss,
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
        record_artifacts = {
            "main.py": src_path.read_text() if src_path.exists() else "",
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
    dataset_id: str = DEFAULT_DATASET_ID,
    dataset_config: str = DEFAULT_DATASET_CONFIG,
    dataset_revision: str = DEFAULT_DATASET_REVISION,
    tuning_mode: str = "lora",
    optimizer_name: str = "auto",
    gradient_checkpointing: str = "auto",
    lora_r: int = 32,
    lora_alpha: int = 64,
    lora_dropout: float = 0.0,
    lora_target_modules: str = "all-linear",
    lora_use_rslora: bool = True,
    lora_use_dora: bool = False,
    lora_init: str = "default",
    lora_eva_rho: float = DEFAULT_LORA_EVA_RHO,
    lora_eva_batches: int = 16,
    loraplus_lr_ratio: float = DEFAULT_LORAPLUS_LR_RATIO,
    loraplus_lr_embedding: float = 1.0e-6,
    muon_lr_adjustment: str = "match_rms_adamw",
    lr_schedule: str = "constant",
    lr_decay_fraction: float = 0.1,
    min_lr_ratio: float = 0.0,
    attn_implementation: str = "flex_attention",
    compile_model: bool = True,
    compile_mode: str = "max-autotune-no-cudagraphs",
    compile_warmup: bool = True,
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
    """Run a nanoFineTune track on Modal.

    --track selects the competition track (1=30min, 2=5min, 3=2hr).
    When --minutes is 0 (default), the track's default budget is used.
    Set --record-description to save a competition record on success.
    """

    if minutes == 0.0:
        if track not in TRACKS:
            raise ValueError(f"--track must be one of: {', '.join(TRACKS.keys())}")
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
        tuning_mode=tuning_mode,  # type: ignore[arg-type]
        optimizer_name=optimizer_name,
        gradient_checkpointing=gradient_checkpointing,  # type: ignore[arg-type]
        lora_r=lora_r,
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
        lora_target_modules=lora_target_modules,
        lora_use_rslora=lora_use_rslora,
        lora_use_dora=lora_use_dora,
        lora_init=lora_init,
        lora_eva_rho=lora_eva_rho,
        lora_eva_batches=lora_eva_batches,
        loraplus_lr_ratio=loraplus_lr_ratio,
        loraplus_lr_embedding=loraplus_lr_embedding,
        muon_lr_adjustment=muon_lr_adjustment,  # type: ignore[arg-type]
        lr_schedule=lr_schedule,  # type: ignore[arg-type]
        lr_decay_fraction=lr_decay_fraction,
        min_lr_ratio=min_lr_ratio,
        attn_implementation=attn_implementation,
        compile_model=compile_model,
        compile_mode=compile_mode,
        compile_warmup=compile_warmup,
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
