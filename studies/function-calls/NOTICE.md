# Benchmark attribution

This optional study uses the Berkeley Function Calling Leaderboard data,
prompt construction, Python AST decoder and type/value checker from
[ShishirPatil/gorilla](https://github.com/ShishirPatil/gorilla/tree/f7cf7359b7ac615a0b294831c5ba2bc95ee4a000/berkeley-function-call-leaderboard),
revision `f7cf7359b7ac615a0b294831c5ba2bc95ee4a000`.
The upstream repository's [Apache 2.0 license](BFCL-LICENSE.txt) is retained.

Any retained reference-call strings derive from those benchmark answers.
Model continuations and numerical measurements are produced locally for this
study. No pretrained model, adapter weights or full question corpus is bundled.
The benchmark and model authors have not endorsed this study.

The local wrapper leaves the pinned prompt, decoder and checker unchanged.
It supplies one inert model-registry entry to avoid importing API providers.
It marks records with empty required-parameter reference alternatives as ungraded
and intercepts binary/lambda expressions before the upstream resolver's eval
branches. These are explicitly scoped departures from an official full-suite
run; no generated function call is executed.
