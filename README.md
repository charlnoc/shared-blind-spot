# Shared Blind Spot vs Self-Preference

**A method (and a working instrument) for telling apart two reasons an LLM judge wrongly endorses a same-family model's answer: because the answer *looks familiar*, or because the judge *would make the same mistake at the same step*.**

The literature on LLM-as-a-judge has largely pinned same-family favouritism on **familiarity / self-preference** (judges prefer lower-perplexity, more familiar-looking text). This repo isolates a *different* mechanism:

> **Shared blind spot.** A same-family judge misses an answerer's error not because the answer looks familiar, but because the judge's own reasoning fails at the *same step*. It can't see the mistake because it would make the mistake too.

This is released as a **method/platform**, not a finished claim. The point is a clean, reproducible, criticisable protocol for separating the two mechanisms — plus a reusable tool the protocol is built on. The current result (below) is an honest, deliberately-scoped **SUGGESTIVE**, and the project is **actively continuing** (see [Status](#project-status--commitment) and [ROADMAP](ROADMAP.md)).

---

## What's actually worth reusing here

Ranked by reuse value, not by where the headline is:

### 1. A mechanical, model-free error-step locator — `src/error_locator.py`
The hardest asset, and usable on its own. Given a chain-of-thought, it finds the **first reasoning step that goes wrong**, by **pure re-execution — it never calls an LLM**. That discipline is the whole point: the obvious shortcut (ask an LLM "where's the error?") reintroduces exactly the bias you're trying to study. It is:
- **exact** when the CoT aligns 1:1 with the canonical steps (re-execute each, flag the first inconsistent one),
- **insertion/deletion tolerant** via a value-trace fallback in canonical-operation space (weak models add/drop steps),
- **unit-tested** on **>1,200 synthetically-corrupted CoTs** with known error positions (exact when aligned; >97% for the fallback), with a model-free test gate that runs in `<0.1s` and needs zero dependencies.

If you do CoT-level error analysis, this is the component you'd otherwise be tempted to do wrong. See [docs/error-step-locator.md](docs/error-step-locator.md).

### 2. The mechanism-separation protocol — SER + chance baseline + perplexity control
The contribution isn't a number, it's a **procedure others can reproduce, attack, and improve**:
- **False Endorsement Rate (FER)** = P(judge endorses | answer is wrong). Necessary, *not* sufficient — self-preference predicts the same gap.
- **Shared-Error Rate (SER)** = among false endorsements, how often the judge's *independent* solution is wrong at the **same canonical step** as the answerer. This is the decisive measurement.
- **Chance baseline** = Σ_k p_M(k)·p_J(k) from the empirical error-step marginals — the co-location expected under independence.
- **Perplexity control** = regress the judge's perplexity on the answerer's CoT out of the family effect, so a surviving effect can't be "just familiarity".

Self-preference is a crowded field; *cleanly operationalising the split from shared reasoning failure* is the new part. This repo turns that operationalisation into a platform you can stand on.

### 3. Engineering discipline as a trust signal
In a field full of repos that don't run, or run but quietly use an LLM as their own error locator (self-contaminating the bias under study), being **obviously honest and obviously reproducible** is itself the signal:
- each model runs in its **own process** (loading several into one process corrupts Apple's MPS memory pool and hangs — found the hard way),
- **incremental JSONL** so no run ever loses work and any phase is independently re-runnable,
- a **model-free test gate** (`./run_tests.sh`, 25 tests) you can run before trusting anything,
- results reported **honestly**: a coefficient whose CI just touches zero is labelled **SUGGESTIVE**, not dressed up as positive.

---

## Current result (v1, local, honest)

150 fresh multi-step arithmetic problems. Answerer **M = Qwen2.5-0.5B** (8% accuracy → 138 errorful cases). Judges: **same-family = Qwen2.5-1.5B**, **cross-family = SmolLM2-1.7B**.

| metric | same-family | cross-family |
|---|---|---|
| False-endorsement rate | 0.53 | 0.98 *(weak judge — rubber-stamps; see report)* |
| **Shared-error rate (SER)** | **0.37** | 0.20 |
| chance baseline | 0.22 | 0.12 |
| **excess co-location (SER − chance)** | **+0.15** | +0.08 |

**Verdict: SUGGESTIVE.** The same-family judge shares the answerer's *specific* error step ~2× above chance vs the cross-family judge; the gap holds within both low- and high-perplexity strata; and same-family perplexity isn't lower than cross-family, so familiarity predicts the *wrong* direction for the gap. What holds it short of POSITIVE is purely power — the pooled perplexity-controlled coefficient is positive but its 95% CI just touches zero (n=38 usable same-family cases).

Full write-up: **[results/RESEARCH_REPORT.md](results/RESEARCH_REPORT.md)** · figures: [`ser_vs_null.png`](results/ser_vs_null.png), [`fer_by_family_domain.png`](results/fer_by_family_domain.png), [`ser_perplexity_partial.png`](results/ser_perplexity_partial.png).

---

## Run it

```bash
# 1. The model-free gate — no torch, no downloads, stdlib only. Trust starts here.
./run_tests.sh

# 2. Full local run: one fresh process per model -> incremental JSONL -> assemble
#    -> figures + findings.md + RESEARCH_REPORT.md. Downloads ~3 small models on
#    first run. ~1.5h on an Apple-Silicon Mac (MPS); no API keys, no GPU cluster.
python -m venv .venv && .venv/bin/pip install -r requirements.txt
./run_all.sh
```

Run config (models, N, difficulty) lives in `src/expconfig.py`. **Do not** use the single-process `scratch/master.py` — it loads several models in one process and hangs MPS; `run_all.sh` is the working entry point and exists precisely because of that bug.

## Reuse just the locator

```python
from arithmetic import generate_problem, corrupt_cot   # or bring your own Problem + CoT
from error_locator import locate_canonical_index

p = generate_problem(seed=1, n_steps=6)
claimed, truth = corrupt_cot(p, index=3, kind="arithmetic", seed=0)  # inject a known error at step 3
idx, method = locate_canonical_index(p.render_cot(claimed), p)
assert idx == truth.index    # -> 3   (method == 'strict' when the CoT aligns 1:1)
```

## Layout

```
src/error_locator.py   ★ mechanical error-step locator (strict + value-trace) — the reusable instrument
src/arithmetic.py        auto-verifiable problem generator + GT + canonical CoT + error injector
src/metrics.py           FER, SER, chance baseline, perplexity-controlled regression (bootstrap CIs)
src/{phase_solve,phase_judge}.py   one-model-per-process data collection (clean MPS)
src/assemble.py          merge phases -> run JSON -> figures + reports
src/{plot,report,research_report}.py   figures, findings.md, RESEARCH_REPORT.md
tests/test_backbone.py   the model-free gate: 25 tests incl. >1,200 corrupted-CoT round-trips
run_all.sh / run_tests.sh   entry points     |     scratch/   exploratory dev scripts (not the entry point)
```

## Project status & commitment

**This is active research, released early on purpose.** The protocol and the locator have certain, present value (a reproducible method; a reusable tool) regardless of whether the headline ever reaches POSITIVE — so they ship now, with their boundaries labelled honestly, rather than waiting on a stronger title.

**I am continuing to develop this.** The next steps are public, not aspirational hand-waving — see [ROADMAP.md](ROADMAP.md) and open issue [#1](https://github.com/charlnoc/shared-blind-spot/issues/1):
1. Harden the cross-family arm with a **capability-matched API judge** (the current cross judge is a weak arithmetic checker), and scale N for statistical power.
2. Add **non-templated domains** (free-form word problems, unit-conversion chains) where perplexity actually varies, giving the familiarity control real purchase.
3. Only on a positive, perplexity-robust SER: **representation-level (Layer-3)** confirmation on the open-weight models.

Critique, replication, and "this is wrong because…" issues are explicitly welcome — that's what releasing the protocol is for. See [CONTRIBUTING.md](CONTRIBUTING.md).

## Cite / reference

If the locator or the protocol is useful, see [CITATION.cff](CITATION.cff). License: [MIT](LICENSE).
