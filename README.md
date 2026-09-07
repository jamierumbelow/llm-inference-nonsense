# llm-inference-nonsense

A repo where I learn about LLM inference.

## The Plan

The purpose of this repo is to learn about how LLM inference works, from the bare bones, up to some reasonably sophisticated techniques. I don't expect to get to state-of-the-art performance. That isn't really the point. Rather, the goal is _pedagogical and expository_.

To do this, I'll build much of the framework, inference engine, and perform the analysis myself. It's 2026, so most of the code itself will be written by Claude or Codex, but I'll be editing it manually, adding comments, and, always, writing the prose around it (including this docs).

## Documentation

The repository is split into four parts:

- `experiments/` contains the model definition for each stage of the project. `e00_baseline` is
  the initial plain PyTorch Llama implementation.
- `packages/harness/` contains shared model configuration and checkpoint-loading code.
- `packages/runner/` compares an experiment with Transformers and runs it locally or on Modal.
- `tools/` contains the command-line entry points.

[WORKLOG.md](WORKLOG.md) contains the running notes and results.

## Comparing logits

Run the short default prompt locally:

```console
uv run tools/compare_logits.py --experiment e00_baseline --model 1b --location local
```

Run the fixed correctness prompt suite on Modal:

```console
uv run tools/compare_logits.py \
  --experiment e00_baseline \
  --model 8b \
  --location modal \
  --suite correctness
```

`--prompt` runs one custom prompt instead. Every run writes a structured JSON report and the full
CLI output to `output_logs/<date>/`.

## Generating text

Generate one greedy continuation with an experiment model:

```console
uv run tools/generate.py \
  --experiment e00_baseline \
  --model 1b \
  --location local \
  --prompt "The present King of France is" \
  --max-new-tokens 16
```

## Standard benchmark

Every experiment uses the same 8B bf16, batch-size-one workload on an A100 40GB:

| Workload       | Input tokens | Output tokens |
| -------------- | -----------: | ------------: |
| Short prefill  |          128 |             0 |
| Medium prefill |          512 |             0 |
| Long prefill   |        1,024 |             0 |
| Short decode   |          128 |            32 |
| Long decode    |          128 |           128 |

The decode cases keep the input length fixed so they isolate the cost of generating more tokens.
The definitions live in `runner.benchmark_workloads` and are shared by every experiment.

Run the suite with:

```console
uv run tools/benchmark.py --experiment e00_baseline
```

The benchmark reports:

* prefill latency
* time to first token
* time per output token and raw per-token timelines
* total generation latency
* prefill and generation throughput
* GPU-only cost per generated token
* model, peak allocated, peak reserved, and incremental GPU memory
* and timing variation.

Each experiment supplies the prefill and generation operations being measured, including its
attention backend and cache behavior. Before measuring, the runner performs five long-prefill
stabilization passes, followed by three workload-specific warmups and ten measured runs.

The report includes published GPU bandwidth, bf16 throughput, and current Modal pricing alongside
the detected hardware. It saves the JSON report and a non-animated CLI transcript under
`output_logs`.
