"""nanoCPT: a Modal H100 speedrun for continued pretraining.

Track 1 trains Qwen3.5-4B-Base on FineMath for a fixed wall-clock budget and
scores the run by the drop in heldout next-token loss.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Literal

import modal


APP_NAME = "nanocpt"
CACHE_MOUNT = Path("/cache")
HF_CACHE = CACHE_MOUNT / "huggingface"

DEFAULT_MODEL_ID = "Qwen/Qwen3.5-4B-Base"
DEFAULT_MODEL_REVISION = "1001bb4d826a52d1f399e183466143f4da7b741b"
DEFAULT_DATASET_ID = "HuggingFaceTB/finemath"
DEFAULT_DATASET_CONFIG = "finemath-4plus"
DEFAULT_DATASET_REVISION = "e92b25a616738fe95dc186b64dfb19f9c8525594"


app = modal.App(APP_NAME)
cache_volume = modal.Volume.from_name("nanocpt-cache", create_if_missing=True)

# Qwen3.5 is new enough that the safest path is current Transformers plus the
# optional fast kernels used by its hybrid full-attention/Gated-DeltaNet stack.
image = (
    modal.Image.from_registry("nvidia/cuda:12.8.1-devel-ubuntu22.04", add_python="3.12")
    .entrypoint([])
    .env(
        {
            "FLASH_ATTENTION_FORCE_CXX11_ABI": "TRUE",
            "MAX_JOBS": "8",
            "NVCC_THREADS": "2",
            "PYTORCH_ALLOC_CONF": "expandable_segments:True",
            "TOKENIZERS_PARALLELISM": "true",
            "TORCH_CUDA_ARCH_LIST": "9.0",
        }
    )
    .apt_install("build-essential", "git", "ninja-build")
    .pip_install("torch>=2.8.0", "packaging", "wheel")
    .pip_install(
        "accelerate>=1.0.0",
        "bitsandbytes>=0.46.0",
        "causal-conv1d>=1.5.0",
        "datasets>=3.0.0",
        "flash-attn>=2.8.0",
        "flash-linear-attention>=0.2.0",
        "hf_transfer>=0.1.9",
        "huggingface_hub>=0.30.0",
        "numpy>=2.0.0",
        "safetensors>=0.5.0",
        "tqdm>=4.66.0",
        "git+https://github.com/huggingface/transformers.git",
        extra_options="--no-build-isolation",
    )
)


def _json_default(value: Any) -> Any:
    if hasattr(value, "item"):
        return value.item()
    return str(value)


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
    lr: float = 2.0e-5,
    warmup_steps: int = 20,
    eval_blocks: int = 64,
    seed: int = 1337,
    model_id: str = DEFAULT_MODEL_ID,
    model_revision: str = DEFAULT_MODEL_REVISION,
    dataset_id: str = DEFAULT_DATASET_ID,
    dataset_config: str = DEFAULT_DATASET_CONFIG,
    dataset_revision: str = DEFAULT_DATASET_REVISION,
    optimizer_name: Literal["adamw8bit", "adamw_fused"] = "adamw8bit",
    attn_implementation: Literal["flash_attention_2", "sdpa", "eager"] = "flash_attention_2",
    compile_model: bool = True,
    compile_mode: str = "max-autotune-no-cudagraphs",
    compile_warmup: bool = True,
    save_final: bool = False,
    log_every: int = 5,
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

    if minutes <= 0:
        raise ValueError("--minutes must be positive")
    if seq_len < 128:
        raise ValueError("--seq-len must be at least 128")
    if micro_batch_size < 1 or grad_accum < 1:
        raise ValueError("--micro-batch-size and --grad-accum must be positive")
    if eval_blocks < 1:
        raise ValueError("--eval-blocks must be positive")
    if optimizer_name not in {"adamw8bit", "adamw_fused"}:
        raise ValueError("--optimizer-name must be one of: adamw8bit, adamw_fused")
    if attn_implementation not in {"flash_attention_2", "sdpa", "eager"}:
        raise ValueError("--attn-implementation must be one of: flash_attention_2, sdpa, eager")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for the Modal H100 training path")

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
        "minutes": minutes,
        "seq_len": seq_len,
        "micro_batch_size": micro_batch_size,
        "grad_accum": grad_accum,
        "effective_tokens_per_step": seq_len * micro_batch_size * grad_accum,
        "lr": lr,
        "warmup_steps": warmup_steps,
        "eval_blocks": eval_blocks,
        "seed": seed,
        "model_id": model_id,
        "model_revision": model_revision,
        "dataset_id": dataset_id,
        "dataset_config": dataset_config,
        "dataset_revision": dataset_revision,
        "optimizer_name": optimizer_name,
        "attn_implementation": attn_implementation,
        "compile_model": compile_model,
        "compile_mode": compile_mode,
        "compile_warmup": compile_warmup,
        "save_final": save_final,
    }
    (run_dir / "config.json").write_text(json.dumps(config, indent=2) + "\n")

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

    print("loading model", flush=True)
    model = ModelClass.from_pretrained(
        model_id,
        revision=model_revision or None,
        torch_dtype=torch.bfloat16,
        attn_implementation=attn_implementation,
        cache_dir=str(HF_CACHE / "hub"),
        low_cpu_mem_usage=True,
    )
    model.config.use_cache = False
    if hasattr(model.config, "text_config"):
        model.config.text_config.use_cache = False
    if hasattr(model, "model") and hasattr(model.model, "language_model"):
        model.model.language_model.config.use_cache = False
    if hasattr(model, "gradient_checkpointing_enable"):
        model.gradient_checkpointing_enable()

    if hasattr(model, "model") and hasattr(model.model, "visual"):
        for parameter in model.model.visual.parameters():
            parameter.requires_grad_(False)
        model.model.visual.eval()

    device = torch.device("cuda")
    model.to(device)

    if compile_model:
        print(f"compiling model with torch.compile(mode={compile_mode!r})", flush=True)
        model = torch.compile(model, dynamic=False, mode=compile_mode)

    trainable_params = [p for p in model.parameters() if p.requires_grad]
    trainable_count = sum(p.numel() for p in trainable_params)
    total_count = sum(p.numel() for p in model.parameters())
    print(f"trainable parameters: {trainable_count:,} / {total_count:,}", flush=True)

    def make_optimizer():
        if optimizer_name == "adamw8bit":
            import bitsandbytes as bnb

            return bnb.optim.AdamW8bit(
                trainable_params,
                lr=lr,
                betas=(0.9, 0.95),
                eps=1.0e-8,
                weight_decay=0.1,
            )
        kwargs: dict[str, Any] = {
            "lr": lr,
            "betas": (0.9, 0.95),
            "eps": 1.0e-8,
            "weight_decay": 0.1,
        }
        try:
            return torch.optim.AdamW(trainable_params, fused=True, **kwargs)
        except TypeError:
            return torch.optim.AdamW(trainable_params, **kwargs)

    optimizer = make_optimizer()

    def set_optimizer_lr(lr_value: float) -> None:
        if hasattr(optimizer, "set_lr"):
            optimizer.set_lr(lr_value)
            return
        for group in optimizer.param_groups:
            group["lr"] = lr_value

    @torch.no_grad()
    def evaluate(label: str) -> float:
        model.eval()
        losses: list[float] = []
        for start in range(0, eval_blocks, micro_batch_size):
            batch = eval_input_ids[start : start + micro_batch_size].to(device, non_blocking=True)
            attention_mask = torch.ones_like(batch, dtype=torch.long)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                output = model(input_ids=batch, attention_mask=attention_mask, labels=batch, use_cache=False)
            losses.append(float(output.loss.detach().cpu()))
        loss = float(np.mean(losses))
        log_metric({"event": label, "eval_loss": loss})
        model.train()
        visual = getattr(getattr(model, "model", None), "visual", None)
        if visual is not None:
            visual.eval()
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

    if compile_model and compile_warmup:
        print("running untimed compile warmup", flush=True)
        model.train()
        warmup_batch = eval_input_ids[:micro_batch_size].to(device, non_blocking=True)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            warmup_loss = model(
                input_ids=warmup_batch,
                attention_mask=torch.ones_like(warmup_batch, dtype=torch.long),
                labels=warmup_batch,
                use_cache=False,
            ).loss
        warmup_loss.backward()
        optimizer.zero_grad(set_to_none=True)
        torch.cuda.synchronize()

    baseline_loss = evaluate("baseline_eval")
    batch_iter = train_batches()
    train_start = time.monotonic()
    train_deadline = train_start + minutes * 60.0
    optimizer.zero_grad(set_to_none=True)

    step = 0
    tokens = 0
    last_loss = math.nan
    while time.monotonic() < train_deadline:
        optimizer.zero_grad(set_to_none=True)
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
        step_lr = lr * step / warmup_steps if warmup_steps > 0 and step <= warmup_steps else lr
        set_optimizer_lr(step_lr)
        torch.nn.utils.clip_grad_norm_(trainable_params, 1.0)
        optimizer.step()
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
        final_dir = run_dir / "final_model"
        unwrapped_model = getattr(model, "_orig_mod", model)
        unwrapped_model.save_pretrained(final_dir)
        tokenizer.save_pretrained(final_dir)
        summary["final_model_dir"] = str(final_dir)

    (run_dir / "summary.json").write_text(json.dumps(summary, indent=2, default=_json_default) + "\n")
    log_metric({"event": "summary", **summary})
    cache_volume.commit()
    return summary


@app.local_entrypoint()
def main(
    minutes: float = 30.0,
    seq_len: int = 4096,
    micro_batch_size: int = 1,
    grad_accum: int = 8,
    lr: float = 2.0e-5,
    warmup_steps: int = 20,
    eval_blocks: int = 64,
    seed: int = 1337,
    model_id: str = DEFAULT_MODEL_ID,
    model_revision: str = DEFAULT_MODEL_REVISION,
    dataset_id: str = DEFAULT_DATASET_ID,
    dataset_config: str = DEFAULT_DATASET_CONFIG,
    dataset_revision: str = DEFAULT_DATASET_REVISION,
    optimizer_name: str = "adamw8bit",
    attn_implementation: str = "flash_attention_2",
    compile_model: bool = True,
    compile_mode: str = "max-autotune-no-cudagraphs",
    compile_warmup: bool = True,
    save_final: bool = False,
    log_every: int = 5,
) -> None:
    """Run track 1 on Modal."""

    summary = run_track1.remote(
        minutes=minutes,
        seq_len=seq_len,
        micro_batch_size=micro_batch_size,
        grad_accum=grad_accum,
        lr=lr,
        warmup_steps=warmup_steps,
        eval_blocks=eval_blocks,
        seed=seed,
        model_id=model_id,
        model_revision=model_revision,
        dataset_id=dataset_id,
        dataset_config=dataset_config,
        dataset_revision=dataset_revision,
        optimizer_name=optimizer_name,
        attn_implementation=attn_implementation,
        compile_model=compile_model,
        compile_mode=compile_mode,
        compile_warmup=compile_warmup,
        save_final=save_final,
        log_every=log_every,
    )
    print(json.dumps(summary, indent=2, default=_json_default))
