# Experiment plan

Each experiment should answer one question.

It owns the model and inference code needed to answer that question, while the harness loads checkpoints and the runner applies the shared correctness and benchmark workloads.

An experiment is complete when its correctness checks pass, it has a clean official benchmark report, and its result and lessons are written down.

## e00: baseline

### Question

What does straightforward Llama inference cost when every generated token recomputes the entire sequence?

### Implementation

The baseline is a plain PyTorch implementation of Llama 3.1/3.2:

- bf16 weights loaded directly from the Hugging Face checkpoint
- grouped-query attention through PyTorch SDPA's math backend
- Llama 3 RoPE scaling, RMSNorm, and SwiGLU
- greedy generation at batch size one
- no KV cache, so every decode step runs the complete growing sequence
- no compilation, custom kernels, quantisation, or other inference optimisations

### Benchmark

Run the standard 8B bf16 workload on an A100 40GB:

- 128, 512, and 1,024-token prefill
- 128-token prompt followed by 32-token decode
- 128-token prompt followed by 128-token decode

This establishes the reference latency, throughput, per-token timeline, memory use, and GPU cost for
every later experiment.

## e01: KV cache

### Question

How much decode work can we avoid by retaining the keys and values produced for earlier tokens?

### Hypothesis

Prefill should remain close to the baseline because it still processes the full prompt. Decode should be substantially faster because each step projects only the new token instead of recomputing every earlier token. Time per output token should become much flatter, although attention still reads a larger cache as the sequence grows. The cache adds persistent memory which grows linearly with sequence length; total peak memory may still fall because cached decode needs fewer temporary activations.

### Scope

Keep everything except caching the same as e00.

### Benchmark

Run the unchanged standard benchmark and compare it with e00. Focus on:

- prefill latency and throughput, which should remain similar
- time to first token, which includes prompt prefill
- time per output token and its change as the context grows
- total generation latency and tokens per second
- GPU memory above the loaded model
- tokens per GPU dollar and cost per million generated tokens

The raw per-token timelines are the main evidence. Compare both their level and their shape: short
e00 contexts can become more efficient as they give the GPU more work, even while the amount of
recomputation grows. E01 should reduce the level substantially; its remaining growth should mostly
come from reading a larger attention cache.
