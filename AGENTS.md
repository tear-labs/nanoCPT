# Agent guidance for modded-continued-training

Project-level conventions for AI agents (Claude Code, Codex, etc.) working
on this repo. Read this before touching the trainer or running ablations.

## Ablation runs must be on the same GPU SKU

Modal's `gpu="H100"` resolves to whichever H100 SKU is available at
schedule time — typically **H100 SXM5 80GB HBM3** but sometimes
**H100 NVL 94GB**. Even at 100 % `gpu_utilization_percent`, the NVL is
materially slower per step for our workload (compile-warmup
~234 s vs ~218 s, step throughput ~7 k vs ~9 k tokens/s for the same
configuration). That difference shows up as fewer completed steps within
the timed budget — a real handicap that has nothing to do with the
optimizer/adapter/schedule being ablated.

**Rule**: when comparing optimizers, schedules, or other training-loop
knobs at a fixed wall-clock budget, every run in the comparison must land
on the same GPU SKU. Before declaring a winner:

1. Check `gpu_name` in each run's `summary.json`. If they don't all
   match (e.g. one is `NVIDIA H100 NVL` and the others are
   `NVIDIA H100 80GB HBM3`), the comparison is not fair.
2. Re-run the off-SKU candidate(s) until you get a matching SKU.
   Modal capacity varies by region/time; retrying usually lands you on
   the more common SKU.
3. Also confirm `peak_gpu_utilization_percent ≥ ~95 %` for every run —
   a run that was util-starved (data-loader bottleneck, OOM-induced
   fallback path, etc.) is also unfair to keep in the comparison.

If you can't get matching SKUs after a reasonable number of retries,
note the SKU mismatch explicitly in the leaderboard description and the
findings doc — don't quietly publish a number that overstates the
loser.

## Don't ship records whose source path no longer exists

When `main.py` loses a code path (e.g. the LoRA strip in `edcfd4e`), any
record artifact that exercised that path becomes irreproducible. Delete
the record dirs and the corresponding `citations/records.bib` entries
in the same PR that removes the code. Stale records on the leaderboard
mislead first-time visitors about what the current trainer actually does.

## Per-group LR convention for the Muon family

The `muon`, `muon8`, and `normuon` optimizer choices each split params
into "2D hidden weights → Muon-family" and "embed/lm\_head/1D → AdamW8bit
tail" (modded-nanogpt convention). Both groups default to `--lr` if you
don't pass `--muon-lr` / `--adamw-tail-lr`, but Muon's Newton-Schulz
update is RMS-normalized to ~1 — so at a shared LR, the Muon group's
effective per-step change is much smaller than the AdamW tail's.

Empirical guidance (Track 2, full FT, Qwen3.5-4B-Base, 2026-05-28):

- AdamW-tail-only LR `2e-5` is fine.
- Muon group LR should be ~10× that (`2e-4`) to get a fair comparison
  against pure AdamW.

A naive single-LR Muon run at `2e-5` scored +0.389 vs +0.486 with the
per-group LR. Always use `--muon-lr` when ablating Muon-family
optimizers; otherwise the optimizer choice is conflated with an
unintentional LR ablation.

## Modal image rebuilds on `main.py` change

Modal's image hash includes `main.py` (the entrypoint), so any edit
invalidates the cached image and triggers a full rebuild. Cold image
build is ~5-10 min on top of the run itself. Batch unrelated `main.py`
edits before launching long ablations.

## Run wiring sanity-check after the timed boundary

`run.sh smoke` is the canonical end-to-end check that the
compile/warmup → timed train loop → final eval pipeline still works
after non-trivial `main.py` edits. It uses `--no-compile-model
--no-compile-warmup` so it skips the expensive autotune but still
exercises the timing-boundary code (`budget_start` at line ~1490,
deadline check at ~1523, final eval at ~1590). Run it after any change
that touches the trainer's control flow.
