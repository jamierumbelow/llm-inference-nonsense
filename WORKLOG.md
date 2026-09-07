# Worklog

_All datetimes in PST. Written manually by Jamie._

<details>

<summary>## September 2nd, 2026</summary>

- 21:59 - First thing we need is Python and uv to manage packages. Using mise to keep the versions pinned.
- 22:08 - Okay, next we need to get the Python workspace setup. will use packages/ for shared code and experiments/ for each of the experimental phases. the idea here is that we start with a very crude, B=1 model with nothing special going on, use that as a baseline, and then add to it as we go. we will definitely want to visualise some of this too so we'll want notebooks - claude recommends marimo over jupyter, which seems sensible - and ruff etc for formatting because I'm a golang stan and like an integrated formatter. pytorch for actual implementation because it's ubiquitous, and i'll bring in [transformers](https://pypi.org/project/transformers/) as well so I can have a reference implem to test against
- 22:14 - also want typing, Jessica says that pyright handles torch / tensor ops better, so pyright it is
- 22:19 - okay next we need to download the model. will start with Llama 3.2 1B which I can run locally, will benchmark on 3.1 8B later. https://huggingface.co/docs/huggingface_hub/en/index.
- 22:25 - claude one-shotted the download script, looks sensible. but i need to apply for access to the models themselves, so while that's pending I can do other things
- 22:28 - moved the profiles into the new harness package since i'll want to re-use them later. the fact that python's docstrings sit underneath the property definition is troubling.
- 22:32 - access approved! running the download script. next up is to get the model running. i'll start a setup phase notebook, import transfomers, and have a play.
- 22:42 - my goodness `transformers` makes things easy. okay. let's move some of this setup code into the harness. i want to implement as much of this as is feasible/not distracting, so we'll get the config setup in python (rather than relying on the one we donwloaded from HF) and define the model in pytorch directly.
- 22:52 - added a config file to the harness package so we can define the architecture and various hyperparameters on a per-model basis. next up is the model itself
- 22:55 - great, claude has given me a model definition, a lightweight loader, and a test that we can run against:

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
  - run the current setup using bf16, figure out what correct means if rounding is going to complicate the picture
  - get access to 8B
  - run the 8B model through the harness, check for correctness with:
    - tensor sharding
    - not tying the input and output heads (this is the main structural difference between 1B and 8B)
    - difference in head dimensions (128 vs 64), RoPe scaling factor
  - run on the GPU and get some baseline numbers
  - write the first experiment in the plan

but for now it's bedtime.

</details>

<details>

<summary>## September 6th, 2026</summary>

12:58 - Let's run things in bf16. need to separate out the correctness of the implem from the numerical differences, so I'll run the transformers version in fp32 and bf16, and the custom implem in bf16, and see what we get.

13:09 - added a very rough compare_bf16.py script that runs each of the combinations of transformers/custom and fp32/bf16. the results:

```
  "comparisons": {
    "transformers_bf16_vs_transformers_fp32": {
      "mean_absolute_error": 0.016216184943914413,
      "max_absolute_error": 0.15844106674194336,
      "last_position_mean_absolute_error": 0.018606342375278473,
      "last_position_max_absolute_error": 0.11216115951538086,
      "argmax_agreement": "7/7",
      "disagreements": []
    },
    "custom_fp32_vs_transformers_fp32": {
      "mean_absolute_error": 0.0,
      "max_absolute_error": 0.0,
      "last_position_mean_absolute_error": 0.0,
      "last_position_max_absolute_error": 0.0,
      "argmax_agreement": "7/7",
      "disagreements": []
    },
    "custom_bf16_vs_transformers_fp32": {
      "mean_absolute_error": 0.016129227355122566,
      "max_absolute_error": 0.12817475199699402,
      "last_position_mean_absolute_error": 0.015188452787697315,
      "last_position_max_absolute_error": 0.0911327600479126,
      "argmax_agreement": "7/7",
      "disagreements": []
    },
    "custom_fp32_vs_transformers_bf16": {
      "mean_absolute_error": 0.016216184943914413,
      "max_absolute_error": 0.15844106674194336,
      "last_position_mean_absolute_error": 0.018606342375278473,
      "last_position_max_absolute_error": 0.11216115951538086,
      "argmax_agreement": "7/7",
      "disagreements": []
    },
    "custom_bf16_vs_transformers_bf16": {
      "mean_absolute_error": 0.018232209607958794,
      "max_absolute_error": 0.1875,
      "last_position_mean_absolute_error": 0.020076023414731026,
      "last_position_max_absolute_error": 0.125,
      "argmax_agreement": "7/7",
      "disagreements": []
    },
    "custom_bf16_vs_custom_fp32": {
      "mean_absolute_error": 0.016129227355122566,
      "max_absolute_error": 0.12817475199699402,
      "last_position_mean_absolute_error": 0.015188452787697315,
      "last_position_max_absolute_error": 0.0911327600479126,
      "argmax_agreement": "7/7",
      "disagreements": []
    }
  }
}
```

results:

fp32: zero differences across all logits
bf16: small differences between custom and transformers, but the winning tokens always agree

both bf16 versions introduce rounding errors compared to fp32, but the custom implementation is slightly closer to the fp32 answer than the transformers implementation.

what possible reasons are there for the rounding errors?

- our RMSNorm implem, which uses Pytorch's nn.RMSNorm not the transformers implem, which might work differently
- attention
- other intermediary calculations

but we shouldn't necessarily expect them to match exactly, because the bf16 implementation is always going to be an approximation. and I don't want to get hung up on this. so I'm going to rerun with a different, longer prompt and check. if the fp32 results match exactly again, and the bf16 results stay close to fp32 without changing the predictions, then we can chalk it down to rounding imprecision and move on.

13:21 - running on 'The Meta Llama 3.1 collection of multilingual large language models (LLMs) is a collection of pretrained and instruction tuned generative models in 8B, 70B and 405B sizes (text in/text out). The Llama 3.1 instruction tuned text only models (8B, 70B, 405B) are optimized for multilingual dialogue use cases and outperform many of the available open source':

```
  "comparisons": {
    "transformers_bf16_vs_transformers_fp32": {
      "mean_absolute_error": 0.02565355971455574,
      "max_absolute_error": 0.36710286140441895,
      "last_position_mean_absolute_error": 0.029803911224007607,
      "last_position_max_absolute_error": 0.12284106016159058,
      "argmax_agreement": "90/91",
      "disagreements": [
        {
          "position": 59,
          "reference_token": " generation",
          "candidate_token": "-to",
          "reference_top_2_gap": 0.1050872802734375
        }
      ]
    },
    "custom_fp32_vs_transformers_fp32": {
      "mean_absolute_error": 0.0,
      "max_absolute_error": 0.0,
      "last_position_mean_absolute_error": 0.0,
      "last_position_max_absolute_error": 0.0,
      "argmax_agreement": "91/91",
      "disagreements": []
    },
    "custom_bf16_vs_transformers_fp32": {
      "mean_absolute_error": 0.027516944333910942,
      "max_absolute_error": 0.30733680725097656,
      "last_position_mean_absolute_error": 0.020644880831241608,
      "last_position_max_absolute_error": 0.11037015914916992,
      "argmax_agreement": "90/91",
      "disagreements": [
        {
          "position": 59,
          "reference_token": " generation",
          "candidate_token": "-to",
          "reference_top_2_gap": 0.1050872802734375
        }
      ]
    },
    "custom_fp32_vs_transformers_bf16": {
      "mean_absolute_error": 0.02565355971455574,
      "max_absolute_error": 0.36710286140441895,
      "last_position_mean_absolute_error": 0.029803911224007607,
      "last_position_max_absolute_error": 0.12284106016159058,
      "argmax_agreement": "90/91",
      "disagreements": [
        {
          "position": 59,
          "reference_token": "-to",
          "candidate_token": " generation",
          "reference_top_2_gap": 0.0
        }
      ]
    },
    "custom_bf16_vs_transformers_bf16": {
      "mean_absolute_error": 0.028971359133720398,
      "max_absolute_error": 0.4296875,
      "last_position_mean_absolute_error": 0.021092489361763,
      "last_position_max_absolute_error": 0.125,
      "argmax_agreement": "91/91",
      "disagreements": []
    },
    "custom_bf16_vs_custom_fp32": {
      "mean_absolute_error": 0.027516944333910942,
      "max_absolute_error": 0.30733680725097656,
      "last_position_mean_absolute_error": 0.020644880831241608,
      "last_position_max_absolute_error": 0.11037015914916992,
      "argmax_agreement": "90/91",
      "disagreements": [
        {
          "position": 59,
          "reference_token": " generation",
          "candidate_token": "-to",
          "reference_top_2_gap": 0.1050872802734375
        }
      ]
    }
  }
}
```

okay great. fp32 still matches exactly. both bf16 implems chose identical tokens, and both differ from fp32 at the same position ("-to"
instead of " generation"), and the mean errors vs fp32 are the same.

13:26 - Next up is getting the 8B model running on the GPU. I've only got 16GB of unified memory, so fp32 (8 billion _ 4 bytes) is not going to fit at all, and bf16 (8 billion _ 2 bytes) will fill us up with the model weights alone.

Let's add a `--profile local|modal` flag to the compare_bf16 script, and then get Modal setup with an A100 40GB and run the comparison on the GPU.

13:33 - to get the model running on Modal, we need to give modal the HF_TOKEN and setup a persistant cache for the weights. then we'll need to update the harness and comparison code to configure the GPU and support the modal profile

13:48 - lovely stuff. modal made that very easy. here's the 8B results on our longer prompt:

| vs transformers fp32 | mean logit error | max error | token agreement |
| -------------------- | ---------------- | --------- | --------------- |
| custom fp32          | 0.00000545       | 0.000200  | 91/91           |
| transformers bf16    | 0.03218          | 2.00926   | 90/91           |
| custom bf16          | 0.02533          | 0.94676   | 91/91           |

Complete token agreement across fp32, but a very small difference in mean and max error between transformers and custom (well under the the 1e-3 tolerance that claude put in the test file). 

I don't have an empirical or theoretical reason why that tolerance specifically should be chosen ...but I don't want to get hung up on this, since the purpose of this is just to establish a sensible baseline and not get model-for-model identity.

So, let's do the following things to get a bit more confident that our implementation is not broken in some strange way:
* Run on a slightly larger set of more varied prompts (a one-token input, a sentence, some code, some maths, and a longer passage of ~1k tokens). We want fp32 agreement to hold across all, and bf16 differences to remain comparable to the transfomers implementation.
* Check that sharding is working correctly
* Check that the embedding/output-head tying is working correctly
* See what happens when we change head dimensions and RoPe scaling factor

14:17 - Actually, before we do any of that, my mental model of what's going on is starting to falter. that's the trouble with rushing through this with a coding agent, I guess.

Making a couple of changes to the harness and comparison tools:
* Automatically save the reports in the output_logs directory (I was manually copying them before)
* Move some of the comparison code into the harness
* Separate out model choice from where they get run (local means 1B + local and modal means 8B + cpu)
* Standardise the 1b/8b and dev/target naming
* Make the prompt a CLI input
* Separate running the model from reporting the results
* Remove the timings from the correctness script (these aren't numbers we'll benchmark against, since they don't include setup costs or multiple runs/variation, and they make the code more complicated)
* Rename the scripts to make things clearer

14:53 - Okay, I've done some refactoring and it's now a lot cleaner.

We have:
* output_logs - which hold the full result dumps from runs
* packages/harness - which contains our custom model and the code needed to load it
* packages/runner - which knows how to run a model through the `transformers` package, on Modal, and how to perform the model comparisons
* tools - which has a few CLI entrypoints to download weights from huggingface, kick off a comparison run, and setup my Modal account

Our comparison checks that the two models agree on output logits for each input position over a prompt. This tells us that the custom model and the transformers model are in rough agreement - their basic chain of operations, from checkpoint loading, embedding lookup, RMS normalisation, QKV projections and other attention mechanisms/RoPE implementation, residual connections etc. all the way through to output are reliably the same.

We still expect to see some small differences between fp32 and bf16 models, and between transformers and the custom model, and between CPU and GPU. This is because:
* Representing the numbers differently will likely produce slightly different outputs (they are rounded differently, which changes the calculation)
* The transformers version and our custom model are implemented slightly differently
* CPUs and GPUs might order calculations separately, which, when combined with differences in rounding especially, can produce subtly different results.

But this doesn't matter hugely because:
* A significant mistake in our custom implementation would normally cause large differences in fp32 results throughout the network, which we don't see
* Our bf16 errors remain in roughly the same range as Transformers bf16 relative to fp32
* Disagreements over logits remain very close

Most importantly, the purpose of this whole exercise is not to get our custom model to match the off-the-shelf version identically. Rather, what we want is to get a simple model in place that we can benchmark and then build upon. What actual model we use isn't hugely relevant.

15:08 - A few other things that occurred to me when writing the above:
* I'd like each experiment in the experiments/ directory to define its own model; the harness should be able to load that and the runner should be able to run it, but the packages shouldn't contain model code (we want to be able to compare different model definitions as we increase the complexity)
* We should still run a sweep against various different prompts
* We should define a fixed prompt suite so we're not changing prompts between experiments
* Our output_logs should include the full CLI output, not just the comparison output

I'll get codex to do a pass and check the changes.

15:29 - All looking good. I've moved the model into e00_baseline, our first experiment, and added the runner and harness changes needed. A few tasks left before we can run e00_baseline and get some proper benchmarking numbers:
1. Add support for greedy generation - ie the prompt->output->prompt+output->next output loop – in the harness
2. Define a fixed benchmark workload to run in every experiment. It needs minimally:
    - Short prefill
    - Medium prefill
    - Long prefill
    - Short decode
    - Long decode
3. Add a dedicated benchmark runner, which gives us the various timings we'll care about: prefill latency, time-to-first-token, total generation latency, avg time per token, peak GPU memory allocation, and some measures of variation.
4. Benchmark reports should output everything we need to make them reproducible (experiment, git commit, model and checkpoint, dtype, GPU, pytorch/cuda versions, runtime values incl batch size, inputs and outputs, and all the raw measurements)

15:43 - added support for generation, and added a tools/generate.py to run it locally:

```
$ uv run tools/generate.py --experiment e00_baseline --model 1b --location local --dtype bf16 --prompt "commit it then to the flames," --max-new-tokens 16
commit it then to the flames, and let the fire consume the ashes of your doubts and fears. Let the flames
```

and thus, David Hume was routed by Llama 3.2 1B

16:01 - Added the benchmark workloads and prompt.



</details>