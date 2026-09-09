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

### Results

Hardware-guarded 8B bf16 benchmark series on A100 40GB, 8 September 2026:

- Run 1: [report](output_logs/9_8_2026/22_12_01_256223_e00_baseline_benchmark_8b_modal_run_01_of_03.json) · [CLI log](output_logs/9_8_2026/22_12_01_256223_e00_baseline_benchmark_8b_modal_run_01_of_03.log)
- Run 2: [report](output_logs/9_8_2026/22_14_04_945120_e00_baseline_benchmark_8b_modal_run_02_of_03.json) · [CLI log](output_logs/9_8_2026/22_14_04_945120_e00_baseline_benchmark_8b_modal_run_02_of_03.log)
- Run 3: [report](output_logs/9_8_2026/22_16_06_632908_e00_baseline_benchmark_8b_modal_run_03_of_03.json) · [CLI log](output_logs/9_8_2026/22_16_06_632908_e00_baseline_benchmark_8b_modal_run_03_of_03.log)

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

### Results

Across the three verified A100 40GB runs:

- generated token IDs match e00 exactly
- 512 and 1,024-token prefill latency remains within 0.3% of e00
- short-decode latency falls 24%, from 1.026 to 0.782 seconds, raising throughput from 31.2 to 40.9 tokens per second
- long-decode latency falls 29%, from 4.378 to 3.094 seconds, raising throughput from 29.2 to 41.4 tokens per second
- time per token remains roughly flat around 24ms, while e00 rises from roughly 32ms to 36ms during long decode
- memory above the loaded model falls 22% for short decode and 41% for long decode because avoiding full-sequence temporary activations more than offsets the persistent cache
- one e01 job ran in a slower but internally stable regime, so the 1.3x short and 1.4x long speedups are median estimates; paired same-container runs would give a more precise comparison

8B bf16 benchmark series on A100 40GB, 8 September 2026:

- Run 1: [report](output_logs/9_8_2026/21_32_05_138123_e01_kv_cache_benchmark_8b_modal_run_01_of_03.json) · [CLI log](output_logs/9_8_2026/21_32_05_138123_e01_kv_cache_benchmark_8b_modal_run_01_of_03.log)
- Run 2: [report](output_logs/9_8_2026/21_34_46_194926_e01_kv_cache_benchmark_8b_modal_run_02_of_03.json) · [CLI log](output_logs/9_8_2026/21_34_46_194926_e01_kv_cache_benchmark_8b_modal_run_02_of_03.log)
- Run 3: [report](output_logs/9_8_2026/21_36_29_211702_e01_kv_cache_benchmark_8b_modal_run_03_of_03.json) · [CLI log](output_logs/9_8_2026/21_36_29_211702_e01_kv_cache_benchmark_8b_modal_run_03_of_03.log)
