# nanoCPT

Single-file Modal setup for continued-pretraining speedruns.

Track 1 starts from `Qwen/Qwen3.5-4B-Base`, trains on `HuggingFaceTB/finemath`
`finemath-4plus`, and reports how far a fixed heldout eval loss drops during a
30-minute H100 training window.

## Run

Install and authenticate Modal locally:

```bash
pip install modal
modal setup
```

Launch the default 30-minute run:

```bash
modal run main.py
```

Short smoke run:

```bash
modal run main.py --minutes 0.1 --eval-blocks 2 --grad-accum 1
```

## Defaults

`main.py` is the only training source file. It builds a Modal image with:

- current Hugging Face Transformers for Qwen3.5 support
- an NVIDIA CUDA devel base image so source-built CUDA extensions have `nvcc`
- H100 CUDA build env defaults, including `TORCH_CUDA_ARCH_LIST=9.0`
- `flash-attn` with `attn_implementation="flash_attention_2"`
- `flash-linear-attention` and `causal-conv1d` for Qwen3.5 Gated DeltaNet layers
- sequence packing from streamed FineMath documents into fixed `seq_len` blocks
- `torch.compile(..., dynamic=False)` plus an untimed compile warmup
- `bitsandbytes` `AdamW8bit` by default

Artifacts are written to the `nanocpt-cache` Modal volume:

- `/cache/runs/<run_id>/config.json`
- `/cache/runs/<run_id>/metrics.jsonl`
- `/cache/runs/<run_id>/summary.json`
- `/cache/eval/<hash>.pt` for the deterministic fixed eval blocks

The score is:

```text
baseline_eval_loss - final_eval_loss
```

The timer starts after model load, eval-cache prep, optimizer setup, compile
warmup, and baseline eval. The final eval runs after the timed train loop.

## Useful Flags

```bash
modal run main.py --minutes 30 --seq-len 4096 --micro-batch-size 1 --grad-accum 8
```

Optimizer choices:

```bash
modal run main.py --optimizer-name adamw8bit
modal run main.py --optimizer-name adamw_fused
```

Use `adamw8bit` for the memory-saving default and `adamw_fused` for a simpler
PyTorch fused AdamW baseline.

Kernel fallback if image build or runtime kernels fail:

```bash
modal run main.py --attn-implementation sdpa --compile-model false
```

Save final weights only when needed:

```bash
modal run main.py --save-final
```

## Fixed Inputs

Default pinned inputs:

- model: `Qwen/Qwen3.5-4B-Base`
- model revision: `1001bb4d826a52d1f399e183466143f4da7b741b`
- dataset: `HuggingFaceTB/finemath`
- dataset config: `finemath-4plus`
- dataset revision: `e92b25a616738fe95dc186b64dfb19f9c8525594`

Both defaults are public and ungated at the time this repo was set up.
