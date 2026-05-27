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
      --gradient-checkpointing auto \
      --no-compile-model \
      --no-compile-warmup \
      --attn-implementation sdpa \
      "$@"
    ;;
  instant-smoke)
    shift
    exec "${cmd[@]}" \
      --minutes 0.1 \
      --eval-blocks 2 \
      --micro-batch-size 4 \
      --grad-accum 1 \
      --gradient-checkpointing auto \
      --no-compile-model \
      --no-compile-warmup \
      --attn-implementation sdpa \
      --activation-compression-mode instant-linear \
      --instant-projector-kind hadamard \
      --instant-chunk-size 64 \
      --instant-keep 32 \
      --instant-min-hidden-dim 64 \
      --instant-hadamard-backend piecewise \
      --instant-parameter-gradient projected-lowpass \
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
  instant-track1)
    shift
    exec "${cmd[@]}" \
      --track 1 \
      --gradient-checkpointing auto \
      --activation-compression-mode instant-linear \
      --instant-projector-kind hadamard \
      --instant-chunk-size 64 \
      --instant-keep 32 \
      --instant-min-hidden-dim 64 \
      --instant-hadamard-backend piecewise \
      --instant-parameter-gradient projected-lowpass \
      "$@"
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
  *)
    exec "${cmd[@]}" "$@"
    ;;
esac
