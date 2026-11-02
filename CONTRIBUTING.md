# Contributing

## Commit messages: gitmoji

Every commit message starts with a [gitmoji](https://gitmoji.dev/) that says what
kind of change it is, followed by a short imperative summary:

```text
:sparkles: Add API load-test script
:bug: Serialize DirectML session runs across API threads
```

Write the emoji as its `:shortcode:`. GitHub renders shortcodes as emoji, and they
survive Windows `cmd`, which can mangle raw emoji characters in `git commit -m`.

One kind of change per commit. If a change set mixes kinds (a fix plus a feature),
split it with `git add <paths>` per commit.

The ones this project uses most (full list: https://gitmoji.dev/):

| Shortcode | Use for |
|---|---|
| `:sparkles:` | New feature, script, or capability |
| `:bug:` | Bug fix |
| `:zap:` | Performance improvement (latency, throughput, memory) |
| `:boom:` | Breaking change (renamed API fields, changed file formats, CLI flags) |
| `:recycle:` | Refactor without behaviour change |
| `:white_check_mark:` | Add, update, or pass tests |
| `:memo:` | Documentation, README, results write-ups |
| `:chart_with_upwards_trend:` | New measurements, benchmark or load-test results |
| `:lipstick:` | Charts, plots, or UI appearance |
| `:heavy_plus_sign:` / `:heavy_minus_sign:` | Add / remove a dependency |
| `:arrow_up:` | Upgrade a dependency |
| `:wrench:` | Configuration files (pyproject, docker-compose, .env) |
| `:see_no_evil:` | `.gitignore` changes |
| `:fire:` | Remove code or files |
| `:construction:` | Work in progress |
| `:tada:` | Begin a project or a major phase |
| `:rocket:` | Deploy |

## Before committing

```bat
pytest
```

Tests run without a network or server; the Elasticsearch integration test runs
automatically when Elasticsearch is up.
