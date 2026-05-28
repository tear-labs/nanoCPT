#!/usr/bin/env bash
# Thin wrapper around `uv run modal run main.py`. All entries default to
# the canonical ConlangCrafter CPT/LoRA challenge — see README for details.
set -euo pipefail

cmd=(uv run modal run main.py)

case "${1:-}" in
  "")
    exec "${cmd[@]}"
    ;;
  smoke)
    shift
    exec "${cmd[@]}" \
      --minutes 0.1 \
      --eval-blocks 2 \
      --micro-batch-size 4 \
      --grad-accum 1 \
      --no-compile-model \
      --no-compile-warmup \
      --attn-implementation sdpa \
      "$@"
    ;;
  track1)
    shift
    exec "${cmd[@]}" --track 1 "$@"
    ;;
  track2)
    shift
    exec "${cmd[@]}" --track 2 "$@"
    ;;
  track3)
    shift
    exec "${cmd[@]}" --track 3 "$@"
    ;;
  # Escape hatches for reproducing pre-conlang records.
  legacy-cpt-track1)
    shift
    exec "${cmd[@]}" --track 1 \
      --dataset-id HuggingFaceTB/finemath \
      --dataset-config finemath-4plus \
      --dataset-revision e92b25a616738fe95dc186b64dfb19f9c8525594 \
      "$@"
    ;;
  legacy-sft-track1)
    shift
    exec "${cmd[@]}" --track 1 --data-mode sft "$@"
    ;;
  *)
    exec "${cmd[@]}" "$@"
    ;;
esac
