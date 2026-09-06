# llm-inference-nonsense

A repo where I learn about LLM inference.

## The Plan

The purpose of this repo is to learn about how LLM inference works, from the bare bones, up to some reasonably sophisticated techniques. I don't expect to get to state-of-the-art performance. That isn't really the point. Rather, the goal is _pedagogical and expository_.

To do this, I'll build much of the framework, inference engine, and perform the analysis myself. It's 2026, so most of the code itself will be written by Claude or Codex, but I'll be editing it manually, adding comments, and, always, writing the prose around it (including this docs).

## Documentation

The repository is split into four parts:

* `experiments/` contains the model definition for each stage of the project. `e00_baseline` is
  the initial plain PyTorch Llama implementation.
* `packages/harness/` contains shared model configuration and checkpoint-loading code.
* `packages/runner/` compares an experiment with Transformers and runs it locally or on Modal.
* `tools/` contains the command-line entry points.

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
