# nanoCPT

Competitive continued-pretraining speedrun on Modal H100s.

Inspired by [KellerJordan/modded-nanogpt](https://github.com/KellerJordan/modded-nanogpt).

## What

nanoCPT measures how far a pretrained language model's heldout eval loss can
drop during a fixed wall-clock training window on a single H100 GPU.

Each run starts from `Qwen/Qwen3.5-4B-Base`, continues pretraining on
`HuggingFaceTB/finemath` (`finemath-4plus`), and is scored by:

```
score = baseline_eval_loss − final_eval_loss
```

Higher is better. The timer starts after model load, eval-cache prep,
optimizer setup, an untimed train-shaped compile warmup, and baseline eval.
The final eval runs after the timed train loop.

## Tracks

| Track | Budget | Default command |
|-------|--------|-----------------|
| 1     | 30 min | `modal run main.py` |
| 2     | 5 min  | `modal run main.py --track 2` |
| 3     | 2 hr   | `modal run main.py --track 3` |

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
3. **Attain a positive eval loss drop.** `baseline_eval_loss - final_eval_loss`
   must be > 0. Due to inter-run variance, submissions targeting a new record
   should provide enough run logs to attain statistical significance at p < 0.01
   that the mean eval loss drop is positive.
4. **Run on a single H100 via Modal.** The hardware is fixed. The run must use
   the Modal image defined in `main.py`.
5. **Not use any extra `torch._inductor.config` or `torch.compile` flags** that
   cause compilation to exceed 30 minutes. (Standard `compile_mode` options are
   fine.)
6. **Beat the prior record.** When baselined on the same hardware, the new run
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

Install and authenticate Modal:

```bash
pip install modal
modal setup
```

Launch a 30-minute run (Track 1):

```bash
modal run main.py
```

Short smoke test:

```bash
modal run main.py --minutes 0.1 --eval-blocks 2 --grad-accum 1
```

5-minute sprint (Track 2):

```bash
modal run main.py --track 2
```

2-hour endurance (Track 3):

```bash
modal run main.py --track 3
```

## Submitting a record

Run with `--record-description` and `--record-contributors`:

```bash
modal run main.py \
  --record-description "MuonOptimizer" \
  --record-contributors "@yourhandle"
```

This automatically saves a record folder under `records/track_N_<budget>/`
containing:

- `main.py` — full source code snapshot (like modded-nanogpt)
- `config.json` — all hyperparameters
- `summary.json` — full run metrics
- `record.txt` — human-readable summary

Open a PR with the new record folder. The PR should:

1. Include at least 3 runs for statistical significance.
2. Clearly describe what changed vs. the previous record.
3. List all contributors.
4. Update the record history table in this README.

## Record history

### Track 1 — 30 minutes

| # | Loss drop | Description | Date | Log | Contributors |
|---|-----------|-------------|------|-----|--------------|
| 1 | — | Baseline run (AdamW8bit, lr=2e-5, seq=4096) | — | — | — |

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
modal run main.py --minutes 30 --seq-len 4096 --micro-batch-size 1 --grad-accum 8
```

Optimizer choices:

```bash
modal run main.py --optimizer-name adamw8bit
modal run main.py --optimizer-name adamw_fused
modal run main.py --optimizer-name muon
```

Attention backends or disable compile:

```bash
modal run main.py --attn-implementation flash_attention_2
modal run main.py --attn-implementation sdpa --compile-model false
```

Save final weights:

```bash
modal run main.py --save-final
```

## Architecture

`main.py` is the only training source file. It builds a Modal image with:

- Current Hugging Face Transformers for Qwen3.5 support
- NVIDIA CUDA devel base image so source-built CUDA extensions have `nvcc`
- H100 CUDA build env defaults, including `TORCH_CUDA_ARCH_LIST=9.0`
- `attn_implementation="flex_attention"` by default, with `flash-attn` for FA2 fallback
- `flash-linear-attention`, `causal-conv1d`, and `tilelang` for Qwen3.5 Gated DeltaNet layers
- Sequence packing from streamed FineMath documents into fixed `seq_len` blocks
- `torch.compile(..., dynamic=False)` plus an untimed train-shaped compile warmup
- `bitsandbytes` `AdamW8bit` by default
- Optional Muon, with 2D matrix weights on Muon and embeddings/norms/biases/head on `AdamW8bit`

Artifacts are written to the `nanocpt-cache` Modal volume:

- `/cache/runs/<run_id>/config.json`
- `/cache/runs/<run_id>/metrics.jsonl`
- `/cache/runs/<run_id>/summary.json`
- `/cache/eval/<hash>.pt` for the deterministic fixed eval blocks

## Scoring detail

```
score = baseline_eval_loss - final_eval_loss
```

The timer starts after model load, eval-cache prep, optimizer setup, an untimed
train-shaped compile warmup, and baseline eval. The final eval runs after the
timed train loop.
