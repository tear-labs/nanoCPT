# nanoFineTune

Competitive single-H100 fine-tuning speedrun on Modal.

Inspired by [KellerJordan/modded-nanogpt](https://github.com/KellerJordan/modded-nanogpt).

## What

nanoFineTune measures how far a pretrained language model's heldout eval loss
can drop during a fixed wall-clock fine-tuning window on a single H100 GPU.

The current default track starts from `Qwen/Qwen3.5-4B-Base`, uses a
continued-pretraining objective on `HuggingFaceTB/finemath`
(`finemath-4plus`), and is scored by:

```
score = baseline_eval_loss − final_eval_loss
```

Higher is better. The timer starts after model load, eval-cache prep,
optimizer setup, an untimed train-shaped compile warmup, and baseline eval.
The final eval runs after the timed train loop.

The baseline iteration uses PEFT LoRA by default: all linear language-model
layers get rank-32 adapters, the base checkpoint stays frozen, and full
fine-tuning remains available with `--tuning-mode full`.

## Tracks

| Track | Budget | Default command |
|-------|--------|-----------------|
| 1     | 30 min | `./run.sh` |
| 2     | 5 min  | `./run.sh track2` |
| 3     | 2 hr   | `./run.sh track3` |

Override the budget explicitly with `--minutes` (default 0 = use track
default).

## Rules

New records must:

1. **Not modify the model or data source.** The model checkpoint, dataset,
   dataset config, and their pinned revisions are fixed (see Fixed Inputs
   below). You may not change these to different models or datasets.
2. **Not modify the eval data pipeline.** The eval set construction (first N
   non-empty documents from the unshuffled train stream, packed into
   `eval_blocks × seq_len` blocks) must remain identical. You may change
   `eval_blocks`, `seq_len`, or the eval batch size, but not the underlying
   stream of tokens.
3. **Not reward hack against the fixed dataset or eval construction.** Changes
   designed to exploit dataset quirks, memorized ordering, eval-set leakage, or
   other source-specific shortcuts are not valid records. Future validation may
   run submissions across multiple heldout datasets or dataset configs to catch
   overfitting to a single source.
4. **Attain a positive eval loss drop.** `baseline_eval_loss - final_eval_loss`
   must be > 0. Due to inter-run variance, submissions targeting a new record
   should provide enough run logs to attain statistical significance at p < 0.01
   that the mean eval loss drop is positive.
5. **Run on a single H100 via Modal.** The hardware is fixed. The run must use
   the Modal image defined in `main.py`.
6. **Not use any extra `torch._inductor.config` or `torch.compile` flags** that
   cause compilation to exceed 30 minutes. (Standard `compile_mode` options are
   fine.)
7. **Beat the prior record.** When baselined on the same hardware, the new run
   must achieve a higher eval loss drop than the previous record.

Other than that, anything and everything is fair game:

- Optimizer choice, learning rate schedules, weight decay
- Batch size, gradient accumulation, sequence length
- Attention implementation (FA2, FlexAttention, SDPA, etc.)
- Architecture modifications (freeze layers, add adapters, etc.)
- Training data ordering, shuffling, document packing strategies
- Mixed precision, compilation, kernel optimizations
- Novel training techniques (Muon, value embeddings, etc.)

### Discretionary

A PR may not be accepted if it:

- Disproportionately degrades code readability for a marginal gain.
- Substantially narrows the loss-drop buffer without outperforming simpler
  alternatives at equivalent loss.

## Fixed Inputs

| Input | Value | Revision |
|-------|-------|----------|
| Model | `Qwen/Qwen3.5-4B-Base` | `1001bb4d826a52d1f399e183466143f4da7b741b` |
| Dataset | `HuggingFaceTB/finemath` | `e92b25a616738fe95dc186b64dfb19f9c8525594` |
| Dataset config | `finemath-4plus` | — |

All are public and ungated.

## Quick start

Install local launcher dependencies and authenticate Modal:

```bash
uv sync
uv run modal setup
```

If Modal is already configured, verify the active profile:

```bash
uv run modal profile list
```

Short smoke test:

```bash
./run.sh smoke
```

This uses the fastest path: short budget, two eval blocks, SDPA attention, and
no model compile or compile warmup.

Full fine-tune compatibility smoke test:

```bash
./run.sh full-smoke
```

Compiled full fine-tune smoke test:

```bash
./run.sh full-compile-smoke
```

Launch a 30-minute run (Track 1):

```bash
./run.sh
```

5-minute sprint (Track 2):

```bash
./run.sh track2
```

2-hour endurance (Track 3):

```bash
./run.sh track3
```

## Submitting a record

Run with `--record-description` and `--record-contributors`:

```bash
./run.sh \
  --record-description "MuonOptimizer" \
  --record-contributors "@yourhandle"
```

This saves a local record folder under `records/track_N_<budget>/` after the
Modal run returns. The folder contains:

- `main.py` — full source code snapshot (like modded-nanogpt)
- `config.json` — all hyperparameters
- `summary.json` — full run metrics
- `record.txt` — human-readable summary
- `metrics.jsonl` — event log from the run

Open a PR with the new record folder. The PR should:

1. Include at least 3 runs for statistical significance.
2. Clearly describe what changed vs. the previous record.
3. List all contributors.
4. Update the record history table in this README.

## Record history

### Track 1 — 30 minutes

| # | Loss drop | Description | Date | Log | Contributors |
|---|-----------|-------------|------|-----|--------------|
| 1 | — | LoRA baseline (all-linear r32, AdamW fused, lr=2e-4, seq=4096) | — | — | — |

### Track 2 — 5 minutes

| # | Loss drop | Description | Date | Log | Contributors |
|---|-----------|-------------|------|-----|--------------|
| 1 | — | Baseline run | — | — | — |

### Track 3 — 2 hours

| # | Loss drop | Description | Date | Log | Contributors |
|---|-----------|-------------|------|-----|--------------|
| 1 | — | Baseline run | — | — | — |

## Useful flags

```bash
./run.sh --minutes 30 --seq-len 4096 --micro-batch-size 1 --grad-accum 8
```

Tuning modes:

```bash
./run.sh --tuning-mode lora
./run.sh --tuning-mode full
```

LoRA baseline knobs:

```bash
./run.sh --lora-r 32 --lora-alpha 64 --lora-target-modules all-linear
./run.sh --gradient-checkpointing true
./run.sh --gradient-checkpointing false
```

Optimizer choices:

```bash
./run.sh --optimizer-name auto
./run.sh --optimizer-name adamw8bit
./run.sh --optimizer-name adamw_fused
./run.sh --optimizer-name muon
```

Attention backends or disable compile:

```bash
./run.sh --attn-implementation flash_attention_2
./run.sh --attn-implementation sdpa --no-compile-model
```

Save final weights:

```bash
./run.sh --save-final
```

In LoRA mode, `--save-final` writes adapter weights. In full mode, it writes
the full model.

## Architecture

`main.py` is the canonical training source file. Like modded-nanogpt, new
optimization attempts should directly edit the current trainer. Accepted
records preserve source snapshots under `records/` so old runs remain
reproducible after the trainer evolves.

The local launcher uses `uv` (`pyproject.toml`, `uv.lock`) and `run.sh`. The
remote training environment is still defined inside `main.py`, which builds a
Modal image with:

- Current Hugging Face Transformers for Qwen3.5 support
- NVIDIA CUDA devel base image so source-built CUDA extensions have `nvcc`
- H100 CUDA build env defaults, including `TORCH_CUDA_ARCH_LIST=9.0`
- `attn_implementation="flex_attention"` by default, with `flash-attn` for FA2 fallback
- `flash-linear-attention`, `causal-conv1d`, and `tilelang` for Qwen3.5 Gated DeltaNet layers
- `peft` LoRA support; default mode applies all-linear rank-32 adapters before compile
- Sequence packing from streamed FineMath documents into fixed `seq_len` blocks
- `torch.compile(..., dynamic=False)` plus an untimed train-shaped compile warmup
- `optimizer_name="auto"` defaults to fused AdamW for LoRA and `AdamW8bit` for full fine-tuning
- LoRA `gradient_checkpointing="auto"` starts without checkpointing and retries the untimed warmup with checkpointing if CUDA OOMs
- Optional Muon, with 2D matrix weights on Muon and embeddings/norms/biases/head on `AdamW8bit`

Artifacts are written to the `nanofinetune-cache` Modal volume:

- `/cache/runs/<run_id>/config.json`
- `/cache/runs/<run_id>/metrics.jsonl`
- `/cache/runs/<run_id>/summary.json`
- `/cache/eval/<hash>.pt` for the deterministic fixed eval blocks

When `--record-description` is provided, `main.py` also returns the source,
config, summary, record text, and metrics log to the local entrypoint, which
writes the canonical `records/` snapshot in this repository.

## Scoring detail

```
score = baseline_eval_loss - final_eval_loss
```

The timer starts after model load, eval-cache prep, optimizer setup, an untimed
train-shaped compile warmup, and baseline eval. The final eval runs after the
timed train loop.
