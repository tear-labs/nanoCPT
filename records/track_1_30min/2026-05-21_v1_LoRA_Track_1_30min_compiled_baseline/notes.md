# v1 Track 1 Notes

This is the first full 30 minute LoRA baseline run.

The saved source snapshot predates the scoring cleanup that moved compile,
autotune, and train-shaped warmup inside the timed track budget. Keep this as a
v1 baseline artifact, but run future comparable records on the current trainer.

Keep:

- `tuning_mode=lora`
- all-linear rank-32 LoRA with `lora_alpha=64`, `lora_dropout=0.0`, and `lora_use_rslora=true`
- `optimizer_name=adamw_fused`
- `attn_implementation=flex_attention`
- `compile_model=true`
- `compile_mode=max-autotune-no-cudagraphs`
- `compile_warmup=true`

Do not keep CUDA graphs for this baseline. `compile_mode=max-autotune` failed
in the PEFT LoRA path with a CUDAGraph overwritten tensor error. Adding
`torch.compiler.cudagraph_mark_step_begin()` before model calls did not make
the full warmup pass, so the graph-marker experiment was removed.

The run completed and compiled cleanly, but it is not a valid competition
record because the eval loss drop was negative:

- Baseline eval loss: `1.43249327596277`
- Final eval loss: `1.4832360548898578`
- Eval loss drop: `-0.050742778927087784`
- Steps: `488`
- Tokens: `15,990,784`
- Throughput: `8872.620814894808 tok/s`
