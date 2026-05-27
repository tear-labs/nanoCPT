# modded-continued-training

Competitive single-H100 fine-tuning speedrun on Modal.

Inspired by [KellerJordan/modded-nanogpt](https://github.com/KellerJordan/modded-nanogpt).

## What

modded-continued-training measures how far a pretrained language model's heldout eval loss
can drop during a fixed wall-clock fine-tuning window on a single H100 GPU.

The current Track 1 default starts from `Qwen/Qwen3.5-4B-Base`, uses packed
assistant-only general chat SFT on `HuggingFaceH4/ultrachat_200k`
(`train_sft`), and is scored by:

```
score = baseline_eval_loss − final_eval_loss
```

Higher is better. The timer starts after model load, eval-cache prep,
optimizer setup, baseline eval, and `torch.compile` + train-shaped warmup
(which run as an untimed prologue, matching modded-nanogpt). The final eval
also runs after the timed train loop. The untimed compile/warmup duration is
recorded as `elapsed_compile_warmup_seconds` in `summary.json`.

The baseline iteration uses PEFT GraLoRA by default: all linear language-model
layers get rank-32 adapters with `--gralora-k 2`, the base checkpoint stays
frozen, and full fine-tuning remains available with `--tuning-mode full`. The
legacy FineMath continued-pretraining objective remains available with
`--data-mode cpt`.

## Tracks

| Track | Budget | Default command |
|-------|--------|-----------------|
| 1     | 30 min | `./run.sh` |
| 2     | 5 min  | `./run.sh track2` (legacy CPT) |
| 3     | 2 hr   | `./run.sh track3` (legacy CPT) |

Override the budget explicitly with `--minutes` (default 0 = use track
default).

## Rules

The rules below apply to the legacy FineMath CPT record track. Track 1 is under
active development and currently uses a breaking SFT default.

Legacy CPT records must:

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
6. **Compilation and warmup are untimed, but constrained.** `torch.compile`,
   autotune, graph capture, recompilation, and train-shaped warmup all run
   *before* the timed budget starts and do not consume it (matching
   modded-nanogpt). To prevent trading unbounded untimed compile for small
   timed gains, you may **not**:
   - set extra `torch._inductor.config` flags beyond defaults,
   - pass extra keyword arguments to `torch.compile` beyond `mode` (selected
     via `--compile-mode` from the existing enum) and `dynamic=False`,
   - use `coordinate_descent_tuning` or similar bounded-runtime-for-unbounded-
     compile-time tradeoffs.

   Note: when `--compile-warmup` is enabled, one real training batch is pulled
   from the stream to prime the compiled forward/backward graphs. That batch
   is dropped, not counted toward `tokens` or any eval. Old records produced
   under the prior "compile counts" rule have `elapsed_budget_seconds`
   including compile; new records do not. `eval_loss_drop` is unaffected, but
   `tokens_per_second` and step counts are not directly comparable across the
   rule change — tag records pre-/post-2026-05-27 accordingly.
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
| Track 1 SFT dataset | `HuggingFaceH4/ultrachat_200k` | `8049631c405ae6576f93f445c6b8166f76f5505a` |
| Track 1 SFT train split | `train_sft` | — |
| Track 1 SFT eval split | `test_sft` | — |
| Legacy CPT dataset | `HuggingFaceTB/finemath` | `e92b25a616738fe95dc186b64dfb19f9c8525594` |
| Legacy CPT config | `finemath-4plus` | — |

All are public and ungated. The Sumerian and ConlangCrafter CPT corpora
(see [Foreign-distribution CPT datasets](#foreign-distribution-cpt-datasets))
are *optional alternatives* selected via `--dataset-id`; record runs against
fixed inputs still target FineMath.

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

Legacy FineMath CPT smoke test:

```bash
./run.sh cpt-smoke
```

Track 1 adapter smokes:

```bash
./run.sh smoke
./run.sh smoke --adapter-mode lora
./run.sh smoke --adapter-mode lora_ga
```

Full Track 1 SFT runs:

```bash
./run.sh track1
./run.sh track1 --adapter-mode lora
./run.sh track1 --adapter-mode lora_ga
```

Repeat the current default with fixed seeds:

```bash
./run.sh track1 --seed 1337
./run.sh track1 --seed 2027
./run.sh track1 --seed 4099
```

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

For a legacy FineMath CPT record, use the explicit CPT/LoRA path:

```bash
./run.sh cpt-track1 \
  --record-description "Legacy CPT LoRA" \
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

## Foreign-distribution CPT datasets

The Track 1 SFT default (`HuggingFaceH4/ultrachat_200k`) is general English
chat, which is close to `Qwen3.5-4B-Base`'s pretraining distribution and
therefore gives only a small `baseline_eval_loss − final_eval_loss` signal. To
amplify the signal, the trainer can run continued-pretraining (`--data-mode
cpt`) against datasets that sit further from pretraining:

| Candidate | What | HF id | Why it's foreign |
|---|---|---|---|
| **ConlangCrafter** | A synthetic constructed language generated post-pretraining-cutoff. ~11M tokens (3,077 chunks) of native prose written by Vertex Gemini 3.5 Flash using a [ConlangCrafter](https://arxiv.org/abs/2508.06094) spec as the system prompt. | [`TearedModels/conlangcrafter-cpt-bd412d52`](https://huggingface.co/datasets/TearedModels/conlangcrafter-cpt-bd412d52) — publish your own with `scripts/synthesize_conlang_cpt.py` + `scripts/push_conlang_dataset.py` | Guaranteed novel: the language did not exist before the synthesis run. |
| **SumTablets** | ~91k Sumerian cuneiform tablet transliterations from [Oracc](https://arxiv.org/abs/2602.22200). | [`colesimmons/SumTablets`](https://huggingface.co/datasets/colesimmons/SumTablets) | Real but ancient and deciphered. Under-represented in Qwen pretraining. |
| **FineMath** (baseline) | Existing legacy CPT default. | `HuggingFaceTB/finemath` | Anchor for comparison. |

A third option that was considered and rejected: **Linear A** and other
undeciphered scripts. Linear A has ~7,400 signs total across 1,427 inscriptions
and is undeciphered; the corpus is 3+ orders of magnitude too small, and the
lack of decipherment means there is no grammar regularity for the model to
learn — any loss drop would be memorization, not language learning.

### Generating the conlang corpus

Vertex AI Gemini synthesizes ~10M tokens against one of the 64 conlang specs
in [`malper/ConlangCrafter`](https://huggingface.co/datasets/malper/ConlangCrafter).
A per-chunk quality gate (lexicon-overlap minimum, English-word maximum,
minimum length) rejects drifted chunks; failed chunks are retried up to twice
with new topic seeds.

```bash
# One-shot generation. The default picks the longest DeepSeek-R1 spec from
# malper/ConlangCrafter and targets ~10M output tokens.
uv run python scripts/synthesize_conlang_cpt.py \
  --target-tokens 10_000_000 --concurrency 32

# Smoke first (≈50k tokens, ~1-2 min) to verify quality before committing.
uv run python scripts/synthesize_conlang_cpt.py --smoke

# Push the resulting parquet + spec to your HF account as a dataset.
uv run python scripts/push_conlang_dataset.py data/conlang_cpt/<language_id>
```

Cost on Gemini 3.5 Flash is roughly $5-20 for 10M output tokens.

### Running CPT against a foreign dataset

After publishing your conlang corpus to HF (or to load any HF dataset with a
`{text: str}` schema), set the environment variable and use the shortcuts:

```bash
export CONLANG_DATASET_ID=<your-hf-user>/conlangcrafter-cpt-<language_id>

./run.sh conlang-smoke      # 6s sanity check
./run.sh conlang-track2     # 5-min sprint
./run.sh conlang-track1     # 30-min Track 1 run

./run.sh sumerian-track2    # uses colesimmons/SumTablets, --cpt-text-field transliteration
./run.sh sumerian-track1
```

The `--cpt-text-field` flag (default `text`) picks the row column the CPT
loader reads. The Sumerian shortcuts hard-code `transliteration` because that
is SumTablets' content field.

## Track 1 SFT Validation

Track 1 now defaults to packed UltraChat general chat SFT plus GraLoRA. The
2026-05-25 results below used the previous Hermes SFT default and are retained
as historical adapter-comparison logs:

| Adapter | Loss drop | Baseline | Final | Steps | Supervised tokens | Log |
|---|---:|---:|---:|---:|---:|---|
| `gralora` | `+0.051756` | `0.125171` | `0.073415` | 463 | 747,598 | [summary](records/track_1_30min/2026-05-25_sft_gralora_track1_candidate/summary.json) |
| `lora` | `+0.013174` | `0.125171` | `0.111997` | 475 | 765,955 | [summary](records/track_1_30min/2026-05-25_sft_lora_track1_candidate/summary.json) |
| `lora_ga` | `-0.124002` | `0.125273` | `0.249274` | 531 | 854,279 | [summary](records/track_1_30min/2026-05-25_sft_lora_ga_track1_candidate/summary.json) |

GraLoRA full-run repeats. Seeds 2027 and 4099 used the default
`./run.sh track1` path after GraLoRA became the default; seed 1337 was the
explicit candidate run from the adapter sweep.

| Seed | Loss drop | Baseline | Final | Steps | Supervised tokens | Log |
|---:|---:|---:|---:|---:|---:|---|
| 1337 | `+0.051756` | `0.125171` | `0.073415` | 463 | 747,598 | [summary](records/track_1_30min/2026-05-25_sft_gralora_track1_candidate/summary.json) |
| 2027 | `+0.049965` | `0.125171` | `0.075206` | 459 | 740,941 | [summary](records/track_1_30min/2026-05-25_sft_gralora_track1_seed2027/summary.json) |
| 4099 | `+0.053016` | `0.125171` | `0.072155` | 473 | 762,642 | [summary](records/track_1_30min/2026-05-25_sft_gralora_track1_seed4099/summary.json) |
| mean | `+0.051579` | `0.125171` | `0.073592` | 465 | 750,394 | — |

Conclusion: adopt `--adapter-mode gralora` as the Track 1 default. It produced
the largest positive eval-loss drop on the historical Hermes SFT eval cache, was
materially better than standard LoRA and LoRA-GA in the full-run sweep, and
repeated with a positive drop across all three 30-minute seeds.

## Legacy CPT Record History

Legacy CPT baseline status: the v2 LoRA default keeps
`compile_mode=max-autotune-no-cudagraphs`. CUDA graphs were tried with
`max-autotune` and failed during the PEFT LoRA path with a CUDAGraph overwritten
tensor error, so they are not part of the baseline. The v2 Track 1 run compiled,
completed the full 64-block eval with `eval_micro_batch_size=2`, and sustained
100% peak sampled GPU util, but its eval loss drop was negative; treat it as a
logged baseline/utilization iteration, not a valid competition record under the
positive-loss-drop rule. The older v1 snapshot predates the scoring cleanup that
moved compile and warmup inside the timed budget.

### Legacy CPT Track 1 — 30 minutes

| # | Loss drop | Description | Date | Log | Contributors |
|---|-----------|-------------|------|-----|--------------|
| v5 | -0.064841 | LoRA-GA AdamW WSD (`lora_ga`, 1 init batch, cache on, lr=5e-5, mb8/grad1, flex, max-autotune-no-cudagraphs); 506 steps, 16.58M tokens, 9208.7 budget tok/s, 10927.9 train-loop tok/s, 100% peak sampled GPU util, peak NVML 64.91 GiB | 2026-05-22 | [summary](records/track_1_30min/2026-05-22_v5_LoRA-GA_AdamW_WSD_lr5e-5/summary.json) | — |
| v4 | -0.469354 | LoRA+ PiSSA WSD lower LR (`loraplus_adamw`, ratio 16, lr=5e-5, mb8/grad1, flex, max-autotune-no-cudagraphs); 498 steps, 16.32M tokens, 9063.8 budget tok/s, 10846.9 train-loop tok/s, 100% peak sampled GPU util, peak NVML 66.88 GiB | 2026-05-22 | [summary](records/track_1_30min/2026-05-22_v4_LoRA_PiSSA_WSD_lr5e-5/summary.json) | — |
| v3 | -0.860189 | LoRA+ PiSSA WSD (`loraplus_adamw`, ratio 16, lr=1e-4, mb8/grad1, flex, max-autotune-no-cudagraphs); 469 steps, 15.37M tokens, 8529.7 budget tok/s, 10554.6 train-loop tok/s, 100% peak sampled GPU util | 2026-05-22 | [summary](records/track_1_30min/2026-05-22_v3_LoRA_PiSSA_WSD/summary.json) | — |
| v2 | -0.050513 | LoRA default util baseline (all-linear r32, AdamW fused, lr=2e-4, seq=4096, mb8/grad1, eval mb2, max-autotune-no-cudagraphs); 491 steps, 16.09M tokens, 8936.5 budget tok/s, 10727.2 train-loop tok/s, 100% peak sampled GPU util | 2026-05-21 | [summary](records/track_1_30min/2026-05-21_v2_LoRA_mb8_Track_1_30min_default_util/summary.json) | — |
| v1 | -0.050743 | LoRA baseline (all-linear r32, AdamW fused, lr=2e-4, seq=4096, max-autotune-no-cudagraphs); 488 steps, 15.99M tokens, 8872.6 tok/s | 2026-05-21 | [summary](records/track_1_30min/2026-05-21_v1_LoRA_Track_1_30min_compiled_baseline/summary.json) | — |

### Track 2 — 5 minutes

Foreign-distribution CPT sweep (2026-05-27): both ConlangCrafter and Sumerian
deliver dramatically larger eval-loss drops than the FineMath baseline, and
both clear the +0 record threshold.

| # | Loss drop | Description | Date | Log | Contributors |
|---|-----------|-------------|------|-----|--------------|
| sumerian-v1 | **+1.092** | CPT LoRA on `colesimmons/SumTablets` (`transliteration` field), 99 steps, 3.24M tokens, baseline 1.946 → final 0.855 | 2026-05-27 | [Modal run](https://modal.com/apps/tear-labs-43657/main/ap-85UYkCwqyOPwwFBJwmV4Tz) | — |
| conlang-v1 | **+0.510** | CPT LoRA on `TearedModels/conlangcrafter-cpt-bd412d52` (ConlangCrafter synthetic conlang), 101 steps, 3.31M tokens, baseline 0.854 → final 0.345 | 2026-05-27 | [Modal run](https://modal.com/apps/tear-labs-43657/main/ap-ZQyBFDPmLUkkHl9O6JmUGB) | — |
| finemath-baseline | -0.034 | CPT LoRA on `HuggingFaceTB/finemath` (legacy default), 101 steps, 3.31M tokens, baseline 1.431 → final 1.466 | 2026-05-27 | [Modal run](https://modal.com/apps/tear-labs-43657/main/ap-WHtCz9Ep4KqNEDIMp6e2lz) | — |

### Track 3 — 2 hours

| # | Loss drop | Description | Date | Log | Contributors |
|---|-----------|-------------|------|-----|--------------|
| 1 | — | Baseline run | — | — | — |

## Useful flags

```bash
./run.sh --minutes 30 --seq-len 4096 --micro-batch-size 1 --grad-accum 8
```

Use `0` for automatic batch sizing. The safe default resolves to
`--micro-batch-size 8 --grad-accum 1` for LoRA on H100, keeping 32k tokens per
optimizer step. Eval uses its own auto cap of `--eval-micro-batch-size 2` so
the full 64-block eval does not OOM the loss/logits path.

Tuning modes:

```bash
./run.sh --tuning-mode lora
./run.sh --tuning-mode full
```

LoRA baseline knobs:

```bash
./run.sh --adapter-mode lora --lora-r 32 --lora-alpha 64 --lora-target-modules all-linear
./run.sh --adapter-mode lora --lora-init pissa_niter_4
./run.sh --adapter-mode lora --lora-init olora
./run.sh --adapter-mode lora --lora-init eva --lora-eva-batches 16
./run.sh --adapter-mode lora_ga --lora-ga-batches 4 --lora-ga-micro-batch-size 1
./run.sh --adapter-mode lora --lora-use-dora
./run.sh --gradient-checkpointing true
./run.sh --gradient-checkpointing false
```

Optimizer choices:

```bash
./run.sh --optimizer-name auto
./run.sh --optimizer-name adamw8bit
./run.sh --optimizer-name adamw_fused
./run.sh --adapter-mode lora --optimizer-name loraplus_adamw --loraplus-lr-ratio 16
./run.sh --adapter-mode lora --optimizer-name loraplus_adamw8bit --loraplus-lr-ratio 16
./run.sh --adapter-mode lora --optimizer-name lorafa --lora-r 128 --lora-alpha 32
./run.sh --optimizer-name muon --muon-lr-adjustment match_rms_adamw
./run.sh --optimizer-name muon8 --muon-quant-block-size 2048
./run.sh --optimizer-name normuon --normuon-beta2 0.95 --normuon-eps 1e-8
```

Learning-rate schedule choices:

```bash
./run.sh --lr-schedule constant
./run.sh --lr-schedule linear --min-lr-ratio 0.1
./run.sh --lr-schedule cosine --min-lr-ratio 0.1
./run.sh --lr-schedule wsd --lr-decay-fraction 0.1 --min-lr-ratio 0.0
```

`linear` and `cosine` decay from the end of warmup to the track deadline.
`wsd` keeps the existing warmup/stable/decay behavior and only decays during
the final `--lr-decay-fraction` of the wall-clock budget.

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
  --wandb-project modded-continued-training \
  --wandb-tags lora,v1,track1
```

Use `--wandb-mode offline` to write W&B logs without an API key. Runs always
write local JSONL metrics; when W&B is enabled, scalar train/eval/GPU metrics
are mirrored to W&B.

## Iteration Notes

### 2026-05-22 LoRA-GA/latest LoRA follow-up

Current stable PEFT in the Modal image (`peft==0.19.1`) exposes LoRA-GA via
`LoraGAConfig` and `preprocess_loraga`, so the trainer now supports:

```bash
./run.sh --adapter-mode lora_ga \
  --lora-ga-batches 4 \
  --lora-ga-micro-batch-size 1 \
  --optimizer-name loraplus_adamw --loraplus-lr-ratio 16 --lr 5e-5 \
  --lr-schedule wsd
```

LoRA-GA estimates full-weight gradients on a small training sample before
adapter injection and uses those gradients to initialize the low-rank adapters.
The default estimate uses 4 single-sample 4096-token batches to keep memory
bounded on H100; `--lora-ga-cache` can persist the large gradient cache on the
Modal volume for repeated exact reruns.

Validation: the one-step LoRA-GA smoke passed with SDPA/no-compile after
filtering unsupported small linears, and v5 completed the full 30-minute
flex-attention compile path with W&B offline logging enabled. v5 had a strong
systems profile, but final eval loss still worsened from `1.431297` to
`1.496138`, so LoRA-GA remains an experimental option rather than the default.

The latest PEFT `main` docs also show source-only fields such as
`velora_config` and `monteclora_config`; those are not wired into this baseline
until we intentionally move the image off the pinned stable PEFT release. The
May 2026 LoRA literature includes Hybrid-LoRA, but that is a hybrid full-tune
module-selection method rather than a drop-in PEFT LoRA initializer, so it is a
separate experiment from this LoRA-GA path.

Sources checked: [PEFT main LoRA reference](https://huggingface.co/docs/peft/main/package_reference/lora),
[LoRA-GA paper](https://arxiv.org/abs/2407.05000),
[ID-LoRA](https://arxiv.org/abs/2602.20727),
[Unified LoRA variants study](https://arxiv.org/abs/2601.22708), and
[Hybrid-LoRA](https://arxiv.org/abs/2605.18822).

### 2026-05-22 optimizer and LoRA variant pass

The v2 utilization run fixed the GPU side but still worsened eval loss, so this
iteration adds quality-oriented knobs without changing the default baseline:

- LoRA+ optimizers (`loraplus_adamw`, `loraplus_adamw8bit`) keep separate base
  learning rates for LoRA A/B matrices through warmup and decay. Explicit
  LoRA+ runs default to `5e-5`; sweep `2e-5`, `5e-5`, and lower ratios before
  trying `1e-4` again.
- Muon now supports the Moonshot/PyTorch RMS-matched update scale via
  `--muon-lr-adjustment match_rms_adamw`; compare `--lr 2e-4`, `5e-4`, and
  `1e-3`.
- LoRA initializers are selectable with `--lora-init`. Prioritize
  `pissa_niter_4`, then `olora`; use `eva` only when the extra initialization
  pass is worth measuring.
- LoRA-FA is available for high-rank experiments where activation memory is the
  limiter, e.g. `--optimizer-name lorafa --lora-r 128 --lora-alpha 32`.
- LR scheduling is available as `constant`, full-budget `linear`/`cosine`, or
  terminal-decay WSD via `--lr-schedule wsd --lr-decay-fraction 0.1`.

Implementation sources checked: [PEFT's LoRA guide](https://huggingface.co/docs/peft/developer_guides/lora)
for PiSSA/OLoRA/EVA, DoRA, LoRA+, and LoRA-FA support;
[PyTorch's Muon docs](https://docs.pytorch.org/docs/2.9/generated/torch.optim.Muon.html)
for `match_rms_adamw`; the [LoRA+ paper](https://arxiv.org/abs/2402.12354);
and WSD schedule work ([2410.05192](https://arxiv.org/abs/2410.05192),
[2601.09000](https://arxiv.org/abs/2601.09000)).

Full Track 1 results from this pass:

| Run | Scope | Result |
| --- | --- | --- |
| `2026-05-22_v3_LoRA_PiSSA_WSD` | Full 30-minute Track 1, default LoRA batch, flex compile | 469 steps, 15.37M tokens, 8,529.7 budget tok/s, 10,554.6 train-loop tok/s, peak util 100%, peak NVML 63.70 GiB, eval loss worsened by 0.8602. |
| `2026-05-22_v4_LoRA_PiSSA_WSD_lr5e-5` | Full 30-minute Track 1, LoRA+ base LR lowered to `5e-5`, flex compile | 498 steps, 16.32M tokens, 9,063.8 budget tok/s, 10,846.9 train-loop tok/s, peak util 100%, peak NVML 66.88 GiB, eval loss worsened by 0.4694. |
| `2026-05-22_v5_LoRA-GA_AdamW_WSD_lr5e-5` | Full 30-minute Track 1, LoRA-GA AdamW, flex compile, W&B offline | 506 steps, 16.58M tokens, 9,208.7 budget tok/s, 10,927.9 train-loop tok/s, peak util 100%, peak NVML 64.91 GiB, eval loss worsened by 0.0648. |

Conclusion for this legacy CPT pass: keep the v2 default (`optimizer_name=auto` -> fused AdamW,
`micro_batch_size=8`, flex attention, `max-autotune-no-cudagraphs`). The new
LoRA+/Muon/PiSSA/WSD and LoRA-GA paths compile and run with good GPU
utilization, but none beat the legacy default quality baseline yet. LoRA+ ratio 16 is
not a quality baseline for this task at either `1e-4` or `5e-5`; LoRA-GA is the
best of this pass but still negative on the 30-minute eval.

### 2026-05-25 Muon quantization pass

The trainer now exposes `muon8`, which keeps Muon's hidden-matrix update but
stores its momentum state with linear int8 block quantization, and `normuon`,
which adds NorMuon-style row-wise second-moment normalization after
orthogonalization. These are experimental optimizer choices; `auto` still
defaults to fused AdamW for LoRA.

Validation smoke results are kept in this log rather than under `records/`,
because they are short compatibility checks rather than full Track 1 records.

| Run ID | Optimizer | Scope | Result |
| --- | --- | --- | --- |
| `20260525-202122` | `muon8` | 0.1-minute SDPA/no-compile smoke, 2 eval blocks | Passed 1 train step; eval loss `1.771615 -> 1.771327`, drop `+0.000288`; 229.5 budget tok/s; peak NVML 67.27 GiB. |
| `20260525-202359` | `normuon` | 0.1-minute SDPA/no-compile smoke, 2 eval blocks | Passed 1 train step; eval loss `1.771615 -> 1.771734`, drop `-0.000120`; 208.3 budget tok/s; peak NVML 67.34 GiB. |

Recommended next 30 minute runs:

```bash
uv run modal run main.py --data-mode cpt --adapter-mode lora --minutes 30 --optimizer-name muon8 \
  --muon-quant-block-size 2048 \
  --record-description "v6 8-bit Muon lr2e-4"

uv run modal run main.py --data-mode cpt --adapter-mode lora --minutes 30 --optimizer-name normuon \
  --normuon-beta2 0.95 --normuon-eps 1e-8 \
  --record-description "v7 NorMuon lr2e-4"

uv run modal run main.py --data-mode cpt --adapter-mode lora --minutes 30 --optimizer-name loraplus_adamw \
  --loraplus-lr-ratio 16 --lr 5e-5 --lora-init pissa_niter_4 \
  --lr-schedule wsd --record-description "v4 LoRA+ PiSSA WSD lr5e-5"

uv run modal run main.py --data-mode cpt --adapter-mode lora --minutes 30 --optimizer-name muon \
  --muon-lr-adjustment match_rms_adamw --lr 2e-4 --lora-init pissa_niter_4 \
  --lr-schedule wsd --record-description "v4 Muon RMS PiSSA WSD lr2e-4"

uv run modal run main.py --data-mode cpt --adapter-mode lora_ga --minutes 30 --optimizer-name auto --lr 2e-5 \
  --lora-ga-batches 4 --lora-ga-micro-batch-size 1 \
  --lora-ga-cache --lr-schedule wsd \
  --record-description "v6 LoRA-GA AdamW WSD lr2e-5 batches4"
```

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
- `attn_implementation="flex_attention"` by default, with `flash-attn` installed for explicit FA2 runs
- `flash-linear-attention`, `causal-conv1d`, and `tilelang` for Qwen3.5 Gated DeltaNet layers
- `peft` LoRA/GraLoRA support; default mode applies all-linear rank-32 GraLoRA
  adapters before compile and auto-resolves to `micro_batch_size=8`,
  `grad_accum=1`, and checkpointing on H100
- Track 1 defaults to UltraChat general chat SFT, rendered as Qwen ChatML
  with assistant-only labels; `--data-mode cpt` restores packed FineMath all-token labels.
  CPT mode accepts any HF dataset with a string text column via
  `--dataset-id` and `--cpt-text-field` (default `text`) — see
  [Foreign-distribution CPT datasets](#foreign-distribution-cpt-datasets) for
  the synthetic-conlang and Sumerian alternatives
- Adapter selection via `--adapter-mode gralora`, `lora`, or `lora_ga`; LoRA-GA
  reuses masked SFT batches for its gradient estimate, and GraLoRA defaults to
  `--gralora-k 2`
- LoRA variant knobs for rsLoRA, DoRA, PiSSA/OLoRA/EVA/LoRA-GA/orthogonal initialization,
  LoRA+, and LoRA-FA on the standard LoRA adapter path
- Eval uses a separate auto micro-batch cap of 2 blocks so the default training
  batch does not OOM the full 64-block eval loss/logits path
- No-padding sequence packing from streamed SFT conversations or FineMath
  documents into fixed `seq_len` blocks (`stream_concat_no_padding`)
- `torch.compile(..., dynamic=False)` plus a train-shaped compile warmup inside the track budget
- `optimizer_name="auto"` defaults to fused AdamW for LoRA and `AdamW8bit` for full fine-tuning
- Optional LR scheduling supports constant, full-budget linear/cosine decay,
  and WSD terminal decay near the end of the budget
- LoRA `gradient_checkpointing="auto"` enables checkpointing for multi-sample micro-batches
  and still retries the timed warmup with checkpointing if CUDA OOMs
- GPU telemetry in `metrics.jsonl`, including NVML utilization, power, and CUDA
  allocated/reserved/peak memory when available; peak budget-phase util and memory are
  mirrored into `summary.json` and W&B
- Optional W&B logging via `--wandb-project`, with `WANDB_API_KEY` forwarded
  from the local environment into the Modal function
- Optional Muon variants, with 2D matrix weights on Muon, 8-bit linear blockwise
  Muon, or NorMuon; embeddings/norms/biases/head stay on `AdamW8bit`

The cheap `./run.sh smoke` path pins SDPA/no-compile smoke tests to
`micro_batch_size=4`; the larger `micro_batch_size=8` default is intended for
the compiled flex-attention path.

Artifacts are written to the `modded-continued-training-cache` Modal volume:

- `/cache/runs/<run_id>/config.json`
- `/cache/runs/<run_id>/metrics.jsonl`
- `/cache/runs/<run_id>/summary.json`
- `/cache/eval/<hash>.pt` for the deterministic fixed eval blocks

When `--record-description` is provided, `main.py` also returns the source,
config, summary, record text, and metrics log to the local entrypoint, which
writes the canonical `records/` snapshot in this repository.

## Citations

GitHub uses `CITATION.cff` for the project-level citation UI. Individual run
citations are generated in `citations/records.bib`; every
`records/**/summary.json` gets a stable BibTeX key, including exploratory or
negative runs. Use the entry for the exact run or leaderboard result you cite.

Regenerate citation files after adding records or Markdown paper links:

```bash
uv run python scripts/generate_citations.py --write
```

Resolve new DOI/arXiv links in Markdown into `references.bib` and
`REFERENCES.md`:

```bash
uv run python scripts/generate_citations.py --write --refresh-references
```

Check that generated citation files are current:

```bash
uv run python scripts/generate_citations.py --check
```

If a run later receives a DOI from a GitHub/Zenodo release, add it to
`citations/doi-overrides.json` under that run's BibTeX key and regenerate.
Google Scholar pickup is not guaranteed for GitHub, BibTeX, or Zenodo records;
these files make citations and archival metadata easy to consume.

## Scoring detail

```
score = baseline_eval_loss - final_eval_loss
```

The timer starts after model load, eval-cache prep, optimizer setup, and
baseline eval. Compilation, graph capture, autotuning, recompilation, and
train-shaped compile warmup all consume the selected track budget. The final
eval runs after the timed train loop. Baseline and final eval use the
uncompiled module so post-budget eval does not create new compiled eval graphs.
