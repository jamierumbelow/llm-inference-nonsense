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
    
## September 6th, 2026

12:58 - Let's run things in bf16. need to separate out the correctness of the implem from the numerical differences, so I'll run the transformers version in fp32 and bf16, and the custom implem in bf16, and see what we get.

13:09 - added a very rough compare_bf16.py script that runs each of the combinations of transformers/custom and fp32/bf16. the results:

```
uv run tools/compare_bf16.py
Running transformers fp32...
Loading weights: 100%|██████████████████████████████████████████████████████████████████████| 146/146 [00:05<00:00, 28.60it/s]
[transformers] Ignoring clean_up_tokenization_spaces=True for BPE tokenizer TokenizersBackend. The clean_up_tokenization post-processing step is designed for WordPiece tokenizers and is destructive for BPE (it strips spaces before punctuation). Set clean_up_tokenization_spaces=False to suppress this warning, or set clean_up_tokenization_spaces_for_bpe_even_though_it_will_corrupt_output=True to force cleanup anyway.
Running transformers bf16...
Loading weights: 100%|███████████████████████████████████████████████████████████████████| 146/146 [00:00<00:00, 11663.05it/s]
Running custom fp32...
Running custom bf16...
{
  "prompt": "The present King of France is",
  "token_ids": [
    [
      128000,
      791,
      3118,
      6342,
      315,
      9822,
      374
    ]
  ],
  "checkpoint": "/Users/jamierumbelow/.cache/huggingface/hub/models--meta-llama--Llama-3.2-1B-Instruct/snapshots/9213176726f574b556790deb65791e0c5aa438b6",
  "torch_version": "2.14.0",
  "device": "cpu",
  "attention_backend": "SDPA math",
  "use_cache": false,
  "runs": {
    "transformers_fp32": {
      "logit_dtype": "torch.float32",
      "shape": [
        1,
        7,
        128256
      ],
      "all_finite": true,
      "forward_seconds": 1.9066706248559058,
      "top_5_next_tokens": [
        {
          "id": 12140,
          "text": " Louis",
          "logit": 16.3660831451416
        },
        {
          "id": 15274,
          "text": " Charles",
          "logit": 15.607439994812012
        },
        {
          "id": 279,
          "text": " the",
          "logit": 15.205832481384277
        },
        {
          "id": 264,
          "text": " a",
          "logit": 15.029139518737793
        },
        {
          "id": 6342,
          "text": " King",
          "logit": 15.027528762817383
        }
      ]
    },
    "transformers_bf16": {
      "logit_dtype": "torch.bfloat16",
      "shape": [
        1,
        7,
        128256
      ],
      "all_finite": true,
      "forward_seconds": 0.42125704186037183,
      "top_5_next_tokens": [
        {
          "id": 12140,
          "text": " Louis",
          "logit": 16.375
        },
        {
          "id": 15274,
          "text": " Charles",
          "logit": 15.625
        },
        {
          "id": 279,
          "text": " the",
          "logit": 15.1875
        },
        {
          "id": 6342,
          "text": " King",
          "logit": 15.0625
        },
        {
          "id": 264,
          "text": " a",
          "logit": 15.0
        }
      ]
    },
    "custom_fp32": {
      "logit_dtype": "torch.float32",
      "shape": [
        1,
        7,
        128256
      ],
      "all_finite": true,
      "forward_seconds": 0.33490316569805145,
      "top_5_next_tokens": [
        {
          "id": 12140,
          "text": " Louis",
          "logit": 16.3660831451416
        },
        {
          "id": 15274,
          "text": " Charles",
          "logit": 15.607439994812012
        },
        {
          "id": 279,
          "text": " the",
          "logit": 15.205832481384277
        },
        {
          "id": 264,
          "text": " a",
          "logit": 15.029139518737793
        },
        {
          "id": 6342,
          "text": " King",
          "logit": 15.027528762817383
        }
      ]
    },
    "custom_bf16": {
      "logit_dtype": "torch.bfloat16",
      "shape": [
        1,
        7,
        128256
      ],
      "all_finite": true,
      "forward_seconds": 0.16086925007402897,
      "top_5_next_tokens": [
        {
          "id": 12140,
          "text": " Louis",
          "logit": 16.375
        },
        {
          "id": 15274,
          "text": " Charles",
          "logit": 15.625
        },
        {
          "id": 279,
          "text": " the",
          "logit": 15.25
        },
        {
          "id": 264,
          "text": " a",
          "logit": 15.0625
        },
        {
          "id": 6342,
          "text": " King",
          "logit": 15.0625
        }
      ]
    }
  },
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
{
  "prompt": "The Meta Llama 3.1 collection of multilingual large language models (LLMs) is a collection of pretrained and instruction tuned generative models in 8B, 70B and 405B sizes (text in/text out). The Llama 3.1 instruction tuned text only models (8B, 70B, 405B) are optimized for multilingual dialogue use cases and outperform many of the available open source",
  "token_ids": [
    [
      128000,
      791,
      16197,
      445,
      81101,
      220,
      18,
      13,
      16,
      4526,
      315,
      2814,
      50923,
      3544,
      4221,
      4211,
      320,
      4178,
      22365,
      8,
      374,
      264,
      4526,
      315,
      81769,
      323,
      7754,
      33519,
      1803,
      1413,
      4211,
      304,
      220,
      23,
      33,
      11,
      220,
      2031,
      33,
      323,
      220,
      16408,
      33,
      12562,
      320,
      1342,
      304,
      37371,
      704,
      570,
      578,
      445,
      81101,
      220,
      18,
      13,
      16,
      7754,
      33519,
      1495,
      1193,
      4211,
      320,
      23,
      33,
      11,
      220,
      2031,
      33,
      11,
      220,
      16408,
      33,
      8,
      527,
      34440,
      369,
      2814,
      50923,
      21976,
      1005,
      5157,
      323,
      704,
      29588,
      1690,
      315,
      279,
      2561,
      1825,
      2592
    ]
  ],
  "checkpoint": "/Users/jamierumbelow/.cache/huggingface/hub/models--meta-llama--Llama-3.2-1B-Instruct/snapshots/9213176726f574b556790deb65791e0c5aa438b6",
  "torch_version": "2.14.0",
  "device": "cpu",
  "attention_backend": "SDPA math",
  "use_cache": false,
  "runs": {
    "transformers_fp32": {
      "logit_dtype": "torch.float32",
      "shape": [
        1,
        91,
        128256
      ],
      "all_finite": true,
      "forward_seconds": 2.349963958840817,
      "top_5_next_tokens": [
        {
          "id": 4211,
          "text": " models",
          "logit": 18.022096633911133
        },
        {
          "id": 445,
          "text": " L",
          "logit": 16.88998794555664
        },
        {
          "id": 323,
          "text": " and",
          "logit": 16.210107803344727
        },
        {
          "id": 2814,
          "text": " mult",
          "logit": 15.419775009155273
        },
        {
          "id": 4221,
          "text": " language",
          "logit": 15.35138988494873
        }
      ]
    },
    "transformers_bf16": {
      "logit_dtype": "torch.bfloat16",
      "shape": [
        1,
        91,
        128256
      ],
      "all_finite": true,
      "forward_seconds": 2.03641154197976,
      "top_5_next_tokens": [
        {
          "id": 4211,
          "text": " models",
          "logit": 18.0
        },
        {
          "id": 445,
          "text": " L",
          "logit": 16.875
        },
        {
          "id": 323,
          "text": " and",
          "logit": 16.125
        },
        {
          "id": 2814,
          "text": " mult",
          "logit": 15.375
        },
        {
          "id": 4221,
          "text": " language",
          "logit": 15.3125
        }
      ]
    },
    "custom_fp32": {
      "logit_dtype": "torch.float32",
      "shape": [
        1,
        91,
        128256
      ],
      "all_finite": true,
      "forward_seconds": 1.980346207972616,
      "top_5_next_tokens": [
        {
          "id": 4211,
          "text": " models",
          "logit": 18.022096633911133
        },
        {
          "id": 445,
          "text": " L",
          "logit": 16.88998794555664
        },
        {
          "id": 323,
          "text": " and",
          "logit": 16.210107803344727
        },
        {
          "id": 2814,
          "text": " mult",
          "logit": 15.419775009155273
        },
        {
          "id": 4221,
          "text": " language",
          "logit": 15.35138988494873
        }
      ]
    },
    "custom_bf16": {
      "logit_dtype": "torch.bfloat16",
      "shape": [
        1,
        91,
        128256
      ],
      "all_finite": true,
      "forward_seconds": 1.1963272909633815,
      "top_5_next_tokens": [
        {
          "id": 4211,
          "text": " models",
          "logit": 18.0
        },
        {
          "id": 445,
          "text": " L",
          "logit": 16.875
        },
        {
          "id": 323,
          "text": " and",
          "logit": 16.25
        },
        {
          "id": 2814,
          "text": " mult",
          "logit": 15.4375
        },
        {
          "id": 4221,
          "text": " language",
          "logit": 15.375
        }
      ]
    }
  },
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
