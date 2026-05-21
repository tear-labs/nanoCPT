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

TRACKS: dict[str, dict[str, Any]] = {
    "1": {"name": "30min", "default_minutes": 30.0, "record_dir": "records/track_1_30min"},
    "2": {"name": "5min", "default_minutes": 5.0, "record_dir": "records/track_2_5min"},
    "3": {"name": "2hr", "default_minutes": 120.0, "record_dir": "records/track_3_2hr"},
}


app = modal.App(APP_NAME)
cache_volume = modal.Volume.from_name("nanofinetune-cache", create_if_missing=True)

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
        "peft==0.19.1",
        "safetensors>=0.5.0",
        "tilelang",
        "tqdm>=4.66.0",
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
        f"Elapsed: {summary['elapsed_train_seconds']:.1f}s\n"
        f"Tokens/sec: {summary['tokens_per_second']:.0f}\n"
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
)
def run_track1(
    minutes: float = 30.0,
    seq_len: int = 4096,
    micro_batch_size: int = 1,
    grad_accum: int = 8,
    lr: float = 0.0,
    weight_decay: float = -1.0,
    warmup_steps: int = 20,
    eval_blocks: int = 64,
    seed: int = 1337,
    model_id: str = DEFAULT_MODEL_ID,
    model_revision: str = DEFAULT_MODEL_REVISION,
    dataset_id: str = DEFAULT_DATASET_ID,
    dataset_config: str = DEFAULT_DATASET_CONFIG,
    dataset_revision: str = DEFAULT_DATASET_REVISION,
    tuning_mode: Literal["lora", "full"] = "lora",
    optimizer_name: Literal["auto", "adamw8bit", "adamw_fused", "muon"] = "auto",
    gradient_checkpointing: Literal["auto", "true", "false"] = "auto",
    lora_r: int = 32,
    lora_alpha: int = 64,
    lora_dropout: float = 0.0,
    lora_target_modules: str = "all-linear",
    lora_use_rslora: bool = True,
    attn_implementation: Literal["flex_attention", "flash_attention_2", "sdpa", "eager"] = "flex_attention",
    compile_model: bool = True,
    compile_mode: str = "max-autotune-no-cudagraphs",
    compile_warmup: bool = True,
    save_final: bool = False,
    log_every: int = 5,
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
    if micro_batch_size < 1 or grad_accum < 1:
        raise ValueError("--micro-batch-size and --grad-accum must be positive")
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
    if optimizer_name not in {"auto", "adamw8bit", "adamw_fused", "muon"}:
        raise ValueError("--optimizer-name must be one of: auto, adamw8bit, adamw_fused, muon")
    if lora_r < 1:
        raise ValueError("--lora-r must be positive")
    if lora_alpha < 1:
        raise ValueError("--lora-alpha must be positive")
    if lora_dropout < 0.0 or lora_dropout >= 1.0:
        raise ValueError("--lora-dropout must be in [0, 1)")
    if attn_implementation not in {"flex_attention", "flash_attention_2", "sdpa", "eager"}:
        raise ValueError(
            "--attn-implementation must be one of: flex_attention, flash_attention_2, sdpa, eager"
        )
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for the Modal H100 training path")
    if track not in TRACKS:
        raise ValueError(f"--track must be one of: {', '.join(TRACKS.keys())}")

    track_info = TRACKS[track]
    requested_lr = lr
    requested_weight_decay = weight_decay
    requested_optimizer_name = optimizer_name
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
        gradient_checkpointing == "auto" and tuning_mode == "full"
    )
    gradient_checkpointing_fallback_used = False

    os.environ.setdefault("HF_HOME", str(HF_CACHE))
    os.environ.setdefault("HF_HUB_ENABLE_HF_TRANSFER", "1")
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "true")
    os.environ.setdefault("PYTORCH_ALLOC_CONF", "expandable_segments:True")
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

    config = {
        "run_id": run_id,
        "track": track,
        "track_name": track_info["name"],
        "record_description": record_description,
        "record_contributors": record_contributors,
        "minutes": minutes,
        "seq_len": seq_len,
        "micro_batch_size": micro_batch_size,
        "grad_accum": grad_accum,
        "effective_tokens_per_step": seq_len * micro_batch_size * grad_accum,
        "lr": lr,
        "requested_lr": requested_lr,
        "weight_decay": weight_decay,
        "requested_weight_decay": requested_weight_decay,
        "warmup_steps": warmup_steps,
        "eval_blocks": eval_blocks,
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
        "attn_implementation": attn_implementation,
        "compile_model": compile_model,
        "compile_mode": compile_mode,
        "compile_warmup": compile_warmup,
        "save_final": save_final,
    }

    def write_config() -> None:
        (run_dir / "config.json").write_text(json.dumps(config, indent=2) + "\n")

    write_config()

    def log_metric(record: dict[str, Any]) -> None:
        record = {"time": time.time(), **record}
        with metrics_path.open("a") as f:
            f.write(json.dumps(record, default=_json_default) + "\n")
        print(json.dumps(record, default=_json_default), flush=True)

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

        print(
            "applying LoRA "
            f"target_modules={lora_target_modules!r} r={lora_r} alpha={lora_alpha} "
            f"dropout={lora_dropout} use_rslora={lora_use_rslora}",
            flush=True,
        )
        lora_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            target_modules=parse_lora_target_modules(lora_target_modules),
            exclude_modules=r".*(visual|lm_head).*",
            r=lora_r,
            lora_alpha=lora_alpha,
            lora_dropout=lora_dropout,
            bias="none",
            use_rslora=lora_use_rslora,
            ensure_weight_tying=True,
        )
        model = get_peft_model(model, lora_config)

    set_gradient_checkpointing(model, checkpointing_enabled)

    device = torch.device("cuda")
    model.to(device)
    uncompiled_model = model

    if compile_model:
        print(f"compiling model with torch.compile(mode={compile_mode!r})", flush=True)
        model = torch.compile(model, dynamic=False, mode=compile_mode)

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
        ):
            super().__init__(
                params,
                dict(lr=lr, momentum=momentum, weight_decay=weight_decay, ns_steps=ns_steps),
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
                    update = grad.lerp(buf, momentum)
                    update = self.zeropower_via_newtonschulz5(update, ns_steps)
                    if weight_decay:
                        p.mul_(1.0 - lr * weight_decay)
                    scale = max(1.0, p.size(0) / p.size(1)) ** 0.5
                    p.add_(update, alpha=-lr * scale)
            return loss

    def make_optimizer():
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
                optimizers.append(Muon(muon_params, lr=lr, momentum=0.95, weight_decay=weight_decay))
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
                f"{adamw_name}: {len(adamw_params)} non-muon tensors",
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

    def set_optimizer_lr(lr_value: float) -> None:
        if isinstance(optimizer, list):
            for opt in optimizer:
                for group in opt.param_groups:
                    group["lr"] = lr_value
            return
        if hasattr(optimizer, "set_lr"):
            optimizer.set_lr(lr_value)
            return
        for group in optimizer.param_groups:
            group["lr"] = lr_value

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

    def mark_cudagraph_step() -> None:
        if "no-cudagraph" in compile_mode:
            return
        mark_step = getattr(torch.compiler, "cudagraph_mark_step_begin", None)
        if mark_step is not None:
            mark_step()

    @torch.no_grad()
    def evaluate(label: str) -> float:
        model.eval()
        losses: list[float] = []
        for start in range(0, eval_blocks, micro_batch_size):
            batch = eval_input_ids[start : start + micro_batch_size].to(device, non_blocking=True)
            attention_mask = torch.ones_like(batch, dtype=torch.long)
            mark_cudagraph_step()
            with torch.autocast("cuda", dtype=torch.bfloat16):
                output = model(input_ids=batch, attention_mask=attention_mask, labels=batch, use_cache=False)
            losses.append(float(output.loss.detach().cpu()))
        loss = float(np.mean(losses))
        log_metric({"event": label, "eval_loss": loss})
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
        print("running untimed train/compile warmup", flush=True)
        model.train()
        optimizer_zero_grad()
        for _ in range(grad_accum):
            warmup_batch = next(batch_iter).to(device, non_blocking=True)
            mark_cudagraph_step()
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
                "retrying untimed warmup with checkpointing enabled",
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
            batch_iter = train_batches()
            write_config()
            run_train_warmup()

    baseline_loss = evaluate("baseline_eval")
    train_start = time.monotonic()
    train_deadline = train_start + minutes * 60.0
    optimizer_zero_grad()

    step = 0
    tokens = 0
    last_loss = math.nan
    while time.monotonic() < train_deadline:
        optimizer_zero_grad()
        accum_losses: list[float] = []
        for _ in range(grad_accum):
            batch = next(batch_iter).to(device, non_blocking=True)
            attention_mask = torch.ones_like(batch, dtype=torch.long)
            mark_cudagraph_step()
            with torch.autocast("cuda", dtype=torch.bfloat16):
                output = model(input_ids=batch, attention_mask=attention_mask, labels=batch, use_cache=False)
                loss = output.loss / grad_accum
            loss.backward()
            accum_losses.append(float(output.loss.detach().cpu()))
            tokens += int(batch.numel())

        step += 1
        step_lr = lr * step / warmup_steps if warmup_steps > 0 and step <= warmup_steps else lr
        set_optimizer_lr(step_lr)
        torch.nn.utils.clip_grad_norm_(trainable_params, 1.0)
        optimizer_step()
        last_loss = float(np.mean(accum_losses))

        if log_every > 0 and (step == 1 or step % log_every == 0):
            elapsed = time.monotonic() - train_start
            log_metric(
                {
                    "event": "train",
                    "step": step,
                    "train_loss": last_loss,
                    "lr": step_lr,
                    "tokens": tokens,
                    "elapsed_train_seconds": elapsed,
                    "tokens_per_second": tokens / max(elapsed, 1.0e-9),
                }
            )

    elapsed_train_seconds = time.monotonic() - train_start
    final_loss = evaluate("final_eval")
    summary = {
        **config,
        "record_date": dt.date.today().isoformat(),
        "run_dir": str(run_dir),
        "eval_cache": str(eval_path),
        "train_skip_docs": train_skip_docs,
        "trainable_params": trainable_count,
        "total_params": total_count,
        "steps": step,
        "tokens": tokens,
        "elapsed_train_seconds": elapsed_train_seconds,
        "tokens_per_second": tokens / max(elapsed_train_seconds, 1.0e-9),
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
    micro_batch_size: int = 1,
    grad_accum: int = 8,
    lr: float = 0.0,
    weight_decay: float = -1.0,
    warmup_steps: int = 20,
    eval_blocks: int = 64,
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
    attn_implementation: str = "flex_attention",
    compile_model: bool = True,
    compile_mode: str = "max-autotune-no-cudagraphs",
    compile_warmup: bool = True,
    save_final: bool = False,
    log_every: int = 5,
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
        attn_implementation=attn_implementation,
        compile_model=compile_model,
        compile_mode=compile_mode,
        compile_warmup=compile_warmup,
        save_final=save_final,
        log_every=log_every,
        track=track,
        record_description=record_description,
        record_contributors=record_contributors,
    )
    record_artifacts = summary.pop("_record_artifacts", None)
    if record_description and isinstance(record_artifacts, dict):
        record_dir = _write_local_record(summary, record_artifacts)
        print(f"record saved to {record_dir}")
    print(json.dumps(summary, indent=2, default=_json_default))
