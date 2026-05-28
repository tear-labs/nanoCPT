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
  # ConlangCrafter CPT shortcuts. The CPT default dataset in main.py already
  # points at TearedModels/conlangcrafter-cpt-bd412d52, so these are mostly
  # convenience aliases. Override CONLANG_DATASET_ID to use a different
  # conlang corpus you published yourself.
  conlang-smoke)
    shift
    exec "${cmd[@]}" \
      --data-mode cpt --adapter-mode lora \
      ${CONLANG_DATASET_ID:+--dataset-id "$CONLANG_DATASET_ID"} \
      --minutes 0.1 --eval-blocks 2 --micro-batch-size 4 --grad-accum 1 \
      --no-compile-model --no-compile-warmup --attn-implementation sdpa \
      "$@"
    ;;
  conlang-track2)
    shift
    exec "${cmd[@]}" --track 2 --data-mode cpt --adapter-mode lora \
      ${CONLANG_DATASET_ID:+--dataset-id "$CONLANG_DATASET_ID"} \
      "$@"
    ;;
  conlang-track1)
    shift
    exec "${cmd[@]}" --track 1 --data-mode cpt --adapter-mode lora \
      ${CONLANG_DATASET_ID:+--dataset-id "$CONLANG_DATASET_ID"} \
      "$@"
    ;;
  *)
    exec "${cmd[@]}" "$@"
    ;;
esac
