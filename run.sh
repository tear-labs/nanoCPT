#!/usr/bin/env bash
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
  cpt-smoke)
    shift
    exec "${cmd[@]}" \
      --data-mode cpt \
      --adapter-mode lora \
      --minutes 0.1 \
      --eval-blocks 2 \
      --micro-batch-size 4 \
      --grad-accum 1 \
      --no-compile-model \
      --no-compile-warmup \
      --attn-implementation sdpa \
      "$@"
    ;;
  full-smoke)
    shift
    exec "${cmd[@]}" \
      --tuning-mode full \
      --minutes 0.1 \
      --eval-blocks 2 \
      --grad-accum 1 \
      --no-compile-model \
      --no-compile-warmup \
      --attn-implementation sdpa \
      "$@"
    ;;
  full-compile-smoke)
    shift
    exec "${cmd[@]}" \
      --tuning-mode full \
      --minutes 0.1 \
      --eval-blocks 2 \
      --grad-accum 1 \
      --attn-implementation sdpa \
      "$@"
    ;;
  track1)
    shift
    exec "${cmd[@]}" --track 1 "$@"
    ;;
  cpt-track1)
    shift
    exec "${cmd[@]}" --track 1 --data-mode cpt --adapter-mode lora "$@"
    ;;
  track2)
    shift
    exec "${cmd[@]}" --track 2 --data-mode cpt --adapter-mode lora "$@"
    ;;
  track3)
    shift
    exec "${cmd[@]}" --track 3 --data-mode cpt --adapter-mode lora "$@"
    ;;
  full-track1)
    shift
    exec "${cmd[@]}" --tuning-mode full --track 1 "$@"
    ;;
  full-track2)
    shift
    exec "${cmd[@]}" --tuning-mode full --track 2 --data-mode cpt "$@"
    ;;
  full-track3)
    shift
    exec "${cmd[@]}" --tuning-mode full --track 3 --data-mode cpt "$@"
    ;;
  # Foreign-distribution CPT shortcuts. CONLANG_DATASET_ID points at the HF
  # repo published by scripts/synthesize_conlang_cpt.py + push_conlang_dataset.py.
  # SUMERIAN_DATASET_ID defaults to the public SumTablets release.
  conlang-smoke)
    shift
    exec "${cmd[@]}" \
      --data-mode cpt --adapter-mode lora \
      --dataset-id "${CONLANG_DATASET_ID:?set CONLANG_DATASET_ID to your HF conlang corpus (e.g. user/conlangcrafter-cpt-XXXX)}" \
      --minutes 0.1 --eval-blocks 2 --micro-batch-size 4 --grad-accum 1 \
      --no-compile-model --no-compile-warmup --attn-implementation sdpa \
      "$@"
    ;;
  conlang-track2)
    shift
    exec "${cmd[@]}" --track 2 --data-mode cpt --adapter-mode lora \
      --dataset-id "${CONLANG_DATASET_ID:?set CONLANG_DATASET_ID to your HF conlang corpus}" \
      "$@"
    ;;
  conlang-track1)
    shift
    exec "${cmd[@]}" --track 1 --data-mode cpt --adapter-mode lora \
      --dataset-id "${CONLANG_DATASET_ID:?set CONLANG_DATASET_ID to your HF conlang corpus}" \
      "$@"
    ;;
  sumerian-smoke)
    shift
    exec "${cmd[@]}" \
      --data-mode cpt --adapter-mode lora \
      --dataset-id "${SUMERIAN_DATASET_ID:-colesimmons/SumTablets}" \
      --cpt-text-field transliteration \
      --minutes 0.1 --eval-blocks 2 --micro-batch-size 4 --grad-accum 1 \
      --no-compile-model --no-compile-warmup --attn-implementation sdpa \
      "$@"
    ;;
  sumerian-track2)
    shift
    exec "${cmd[@]}" --track 2 --data-mode cpt --adapter-mode lora \
      --dataset-id "${SUMERIAN_DATASET_ID:-colesimmons/SumTablets}" \
      --cpt-text-field transliteration \
      "$@"
    ;;
  sumerian-track1)
    shift
    exec "${cmd[@]}" --track 1 --data-mode cpt --adapter-mode lora \
      --dataset-id "${SUMERIAN_DATASET_ID:-colesimmons/SumTablets}" \
      --cpt-text-field transliteration \
      "$@"
    ;;
  *)
    exec "${cmd[@]}" "$@"
    ;;
esac
