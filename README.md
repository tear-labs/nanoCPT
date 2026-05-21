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
optimizer setup, and baseline eval. Compilation, graph capture, autotuning, and
train-shaped warmup all consume the selected track budget. The final eval runs
after the timed train loop.

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
6. **Count compilation work against the track budget.** `torch.compile`,
   graph capture, autotuning, recompilation, and train-shaped compile warmup
   are allowed, but they consume the same timed budget as training.
7. **Beat the prior record.** When baselined on the same hardware, the new run
   must achieve a higher eval loss drop than the previous record.

Other than that, anything and everything is fair game:

- Optimizer choice, learning rate schedules, weight decay
- Batch size, gradient accumulation, sequence length
- Attention implementation (FA2, FlexAttention, SDPA, etc.)
- Model-aware optimizations that use the underlying architecture, layer layout,
  parameter shapes, attention/MLP structure, or other implementation details
- Architecture and trainable-structure changes (freeze layers, add adapters,
  replace modules, add auxiliary parameters, alter which weights are updated,
  etc.)
- Training data ordering, shuffling, document packing strategies
- Mixed precision, compilation, kernel optimizations
- Novel training techniques (Muon, value embeddings, etc.)

In other words, the starting checkpoint is fixed, but the trainer does not need
to treat the model as a black box. Submissions may incorporate knowledge of the
Qwen3.5 architecture directly into their optimization strategy, provided they do
not change the fixed input model to a different checkpoint or exploit eval/data
leakage.

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

Current baseline status: the v2 LoRA default keeps
`compile_mode=max-autotune-no-cudagraphs`. CUDA graphs were tried with
`max-autotune` and failed during the PEFT LoRA path with a CUDAGraph overwritten
tensor error, so they are not part of the baseline. The v2 Track 1 run compiled,
completed the full 64-block eval with `eval_micro_batch_size=2`, and sustained
100% peak sampled GPU util, but its eval loss drop was negative; treat it as a
logged baseline/utilization iteration, not a valid competition record under the
positive-loss-drop rule. The older v1 snapshot predates the scoring cleanup that
moved compile and warmup inside the timed budget.

### Track 1 — 30 minutes

| # | Loss drop | Description | Date | Log | Contributors |
|---|-----------|-------------|------|-----|--------------|
| v2 | -0.050513 | LoRA default util baseline (all-linear r32, AdamW fused, lr=2e-4, seq=4096, mb8/grad1, eval mb2, max-autotune-no-cudagraphs); 491 steps, 16.09M tokens, 8936.5 budget tok/s, 10727.2 train-loop tok/s, 100% peak sampled GPU util | 2026-05-21 | [summary](records/track_1_30min/2026-05-21_v2_LoRA_mb8_Track_1_30min_default_util/summary.json) | — |
| v1 | -0.050743 | LoRA baseline (all-linear r32, AdamW fused, lr=2e-4, seq=4096, max-autotune-no-cudagraphs); 488 steps, 15.99M tokens, 8872.6 tok/s | 2026-05-21 | [summary](records/track_1_30min/2026-05-21_v1_LoRA_Track_1_30min_compiled_baseline/summary.json) | — |

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

Use `0` for automatic batch sizing. The safe default resolves to
`--micro-batch-size 1 --grad-accum 8`, keeping 32k tokens per optimizer step.
A LoRA `--micro-batch-size 2` smoke at `seq_len=4096` reached roughly 78 GiB
in use and OOMed without checkpointing, so larger micro-batches should be
treated as explicit experiments and watched through the GPU telemetry fields.

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

Weights & Biases logging is opt-in:

```bash
export WANDB_API_KEY=...
./run.sh track1 \
  --wandb-project nanofinetune \
  --wandb-tags lora,v1,track1
```

Use `--wandb-mode offline` to write W&B logs without an API key. Runs always
write local JSONL metrics; when W&B is enabled, scalar train/eval/GPU metrics
are mirrored to W&B.

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
  and auto-resolves to `micro_batch_size=8`, `grad_accum=1`, and checkpointing on H100
- Eval uses a separate auto micro-batch cap of 2 blocks so the default training
  batch does not OOM the full 64-block eval loss/logits path
- Sequence packing from streamed FineMath documents into fixed `seq_len` blocks
- `torch.compile(..., dynamic=False)` plus a train-shaped compile warmup inside the track budget
- `optimizer_name="auto"` defaults to fused AdamW for LoRA and `AdamW8bit` for full fine-tuning
- LoRA `gradient_checkpointing="auto"` enables checkpointing for multi-sample micro-batches
  and still retries the timed warmup with checkpointing if CUDA OOMs
- GPU telemetry in `metrics.jsonl`, including NVML utilization, power, and CUDA
  allocated/reserved/peak memory when available; peak budget-phase util and memory are
  mirrored into `summary.json` and W&B
- Optional W&B logging via `--wandb-project`, with `WANDB_API_KEY` forwarded
  from the local environment into the Modal function
- Optional Muon, with 2D matrix weights on Muon and embeddings/norms/biases/head on `AdamW8bit`

The cheap `./run.sh smoke` path pins SDPA/no-compile smoke tests to
`micro_batch_size=4`; the larger `micro_batch_size=8` default is intended for
the compiled flex-attention path.

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

The timer starts after model load, eval-cache prep, optimizer setup, and
baseline eval. Compilation, graph capture, autotuning, recompilation, and
train-shaped compile warmup all consume the selected track budget. The final
eval runs after the timed train loop. Baseline and final eval use the
uncompiled module so post-budget eval does not create new compiled eval graphs.
