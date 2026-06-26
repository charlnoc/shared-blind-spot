# Contributing

This is an active research project released as a method/platform. The most
valuable contributions are the kind that make the result *less* likely to be a
fluke: replication, criticism, and tightening the protocol.

## Especially welcome

- **"This is wrong because…"** — methodological critique of the SER / chance
  baseline / perplexity control. Open an issue; that's what releasing the
  protocol is for.
- **Replication** on other model pairs, sizes, or families.
- **New domains** with a *mechanical* error-step locator (see
  [docs/error-step-locator.md](docs/error-step-locator.md)) — free-form word
  problems, unit conversions, symbolic logic. See [ROADMAP.md](ROADMAP.md).
- **The capability-matched API cross-family judge**
  ([issue #1](docs/issue-1-cross-family-api-judge.md)).

## Non-negotiables

These are the things that make the repo trustworthy; PRs that break them won't
be merged:

1. The error-step locator stays **mechanical** — no LLM as the locator, ever.
2. Negative / underpowered results are reported honestly. No p-hacking toward a
   nicer headline.
3. Every excluded or unlocalizable case is **counted and logged**, never silently
   dropped.
4. `./run_tests.sh` (the model-free gate) stays green and dependency-free.

## Dev setup

```bash
python -m venv .venv && .venv/bin/pip install -r requirements.txt
./run_tests.sh        # model-free gate (no torch needed for this)
```

Run config is in `src/expconfig.py`. Each model runs in its own process
(`run_all.sh`) — loading several into one process hangs Apple MPS.
