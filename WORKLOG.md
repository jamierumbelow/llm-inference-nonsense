# Worklog

_All datetimes in PST. Written manually by Jamie._

## September 2nd, 2026

* 21:59 - First thing we need is Python and uv to manage packages. Using mise to keep the versions pinned.
* 22:08 - Okay, next we need to get the Python workspace setup. will use packages/ for shared code and experiments/ for each of the experimental phases. the idea here is that we start with a very crude, B=1 model with nothing special going on, use that as a baseline, and then add to it as we go. we will definitely want to visualise some of this too so we'll want notebooks - claude recommends marimo over jupyter, which seems sensible - and ruff etc for formatting because I'm a golang stan and like an integrated formatter. pytorch for actual implementation because it's ubiquitous, and i'll bring in [transformers](https://pypi.org/project/transformers/) as well so I can have a reference implem to test against
* 22:14 - also want typing, Jessica says that pyright handles torch / tensor ops better, so pyright it is
* 22:19 - okay next we need to download the model. will start with Llama 3.2 1B which I can run locally, will benchmark on 3.1 8B later. https://huggingface.co/docs/huggingface_hub/en/index.
* 22:25 - claude one-shotted the download script, looks sensible. but i need to apply for access to the models themselves, so while that's pending I can do other things
* 22:28 - moved the profiles into the new harness package since i'll want to re-use them later. the fact that python's docstrings sit underneath the property definition is troubling.
* 22:32 - access approved! running the download script. next up is to get the model running. i'll start a setup phase notebook, import transfomers, and have a play.
* 22:42 - my goodness `transformers` makes things easy. okay. let's move some of this setup code into the harness. i want to implement as much of this as is feasible/not distracting, so we'll get the config setup in python (rather than relying on the one we donwloaded from HF) and define the model in pytorch directly.
* 22:52 - added a config file to the harness package so we can define the architecture and various hyperparameters on a per-model basis. next up is the model itself
* 22:55 - great, claude has given me a model definition, a lightweight loader, and a test that we can run against:

```
▶ uv run pytest -m slow
===================================================== test session starts =====================================================
platform darwin -- Python 3.13.15, pytest-9.1.1, pluggy-1.6.0
rootdir: /Users/jamierumbelow/workspace/repos/llm-inference-nonsense
configfile: pyproject.toml
testpaths: packages, experiments
plugins: anyio-4.15.0
collected 9 items / 6 deselected / 3 selected

packages/harness/tests/test_config.py .                                                                                 [ 33%]
packages/harness/tests/test_model.py ..                                                                                 [100%]

============================================== 3 passed, 6 deselected in 15.08s ===============================================
```

it works! i simplified the model.py file a little by swapping out the `RMSNorm` implementation claude did for me with `torch.nn.RMSNorm`. the bulk of it is defining the RoPe positional encoding and the attention mechanism; the rest is a straightforward pytorch model and a wrapper that projects the model output to logits. the test first runs the snapshot model via `AutoModelForCausalLM`, then runs our model, and checks that the logits are the same.

a few worries/things to check/clarifications:
- we are testing that the logits are identical bit-for-bit in fp32 (which runs on my cpu locally) but the model ships from HF in bf16 and we'll run it in bf16 on the GPU when we get to that point (bf16 means the 1B model will weigh 2.5GB rather than 5GB; the 8B model will weigh 16GB rather than 32GB). I will likely run this on an A100 40GB so the extra headroom will be useful.
- should probably note that the RoPe implementation is using fp32 too. apparently it's very normal to mix precision like this (use higher precision where you don't want any rounding errors, use lower precision for the model machinery itself) but I should explore this in a bit more detail when I get to the quantisation experiment
- should also probably note that bf16 and fp16 are in fact different and we should target bf16: same exponent range as fp32, no overflow worries like with fp16. i'll do that next, or certainly avoid using the local environment to get benchmarking numbers.

* 23:28 - next up:
    * run the current setup using bf16, figure out what correct means if rounding is going to complicate the picture
    * get access to 8B
    * run the 8B model through the harness, check for correctness with:
        - tensor sharding
        - not tying the input and output heads (this is the main structural difference between 1B and 8B)
        - difference in head dimensions (128 vs 64), RoPe scaling factor
    * run on the GPU and get some baseline numbers
    * write the first experiment in the plan

but for now it's bedtime.
    