# Worklog

_All datetimes in PST. Written manually by Jamie._

## September 2nd, 2026

* 21:59 - First thing we need is Python and uv to manage packages. Using mise to keep the versions pinned.
* 22:08 - Okay, next we need to get the Python workspace setup. will use packages/ for shared code and experiments/ for each of the experimental phases. the idea here is that we start with a very crude, B=1 model with nothing special going on, use that as a baseline, and then add to it as we go. we will definitely want to visualise some of this too so we'll want notebooks - claude recommends marimo over jupyter, which seems sensible - and ruff etc for formatting because I'm a golang stan and like an integrated formatter. pytorch for actual implementation because it's ubiquitous, and i'll bring in [transformers](https://pypi.org/project/transformers/) as well so I can have a reference implem to test against
* 22:14 - also want typing, Jessica says that pyright handles torch / tensor ops better, so pyright it is
