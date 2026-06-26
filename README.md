# Shared Blind Spot vs Self-Preference in LLM-as-a-Judge: an error-step-level adjudication

When a same-family LLM judge wrongly endorses another model's wrong answer, *why*? Two
mechanisms predict the same endorsement but different internals: a **shared blind spot**
(the judge's own reasoning fails at the *same step*, so it can't see the mistake) or
**self-preference / leniency** (the judge is lenient toward familiar same-family output,
regardless of whether it shares the specific error). This repo is a small, cheap, fully
reproducible **protocol** that separates the two at the *error-step level*, after
**capability-matching** the judge arms — and reports the verdict it returns. On a
controlled, mechanically-verifiable arithmetic domain, the verdict is: **self-preference,
not a shared blind spot.**

We are **not** claiming to discover that LLM judges share blind spots. We provide an
instrument that adjudicates the question and accept the answer it gives — including the
fact that it overturned our own earlier, weaker signal.

## What this is / isn't

**Is:** a minimal measurement protocol to separate two mechanisms of same-family judge
favouritism, with a mechanical (never-LLM) error-step locator, a shared-error-rate metric
against a chance baseline, and a capability-matching calibration step — plus two runs:
a local open-weight run (v1) and a capability-matched cross-family API run (v2).

**Isn't:** a claim that judges sharing failure modes is a new phenomenon (it isn't — see
related work); a large-scale study; a production tool. Single templated domain, modest
usable-case counts, one model pair per arm.

## Background & the precise question

LLM-as-a-judge is widely used, and same-family judges over-endorse same-family outputs —
**self-preference bias**, well established since Zheng et al. (2023, "Judging LLM-as-a-Judge
with MT-Bench and Chatbot Arena"). The open question here is narrower: *when* a same-family
judge wrongly endorses, is it because it **shares the answerer's specific reasoning failure**
(shared blind spot) or because it is **lenient toward familiar output** (self-preference)?
These differ in a checkable way — in *where the judge errs when it solves the same problem
itself*. Shared blind spot predicts the judge fails at the answerer's exact step more than a
matched outside judge does; self-preference predicts no such error-step coupling (the judge
may even solve it correctly and still endorse the wrong answer).

## Related work — and the exact gap

Prior work establishes the surrounding facts but stops short of this adjudication (each
preprint below was checked against its arXiv page):

- **Avena, Bet & Busoni, "How reliable are LLMs when it comes to playing dice?"**
  (arXiv:2606.07515, 2026, preprint). Frontier models fail consistently on a describable
  class of problems (avg accuracy 0.96 on standard discrete-probability questions vs 0.59 on
  counterintuitive ones, with token-bias sensitivity). → Structured, shared failure modes
  are real. **Stops at** final-answer accuracy on a benchmark; it studies models *solving*,
  not *judging* — the "judges therefore share the blind spot" reading is an extrapolation,
  not a result in the paper.
- **Song, Zheng & Xu, "Beyond the Illusion of Consensus: From Surface Heuristics to
  Knowledge-Grounded Evaluation in LLM-as-a-Judge"** (arXiv:2603.11027, 2026, preprint).
  High agreement among judges masks weak sample-level agreement — an "Evaluation Illusion" —
  with judges anchoring on surface heuristics rather than substance. → The strongest
  articulation of the self-preference / surface alternative. **Stops at** subjective-quality
  domains without a mechanical correctness check; not error-step level. Our verdict *agrees
  with and extends* it into a verifiable-reasoning domain.
- **Yang et al., "Auditing Multi-Agent LLM Reasoning Trees Outperforms Majority Vote and
  LLM-as-Judge"** (arXiv:2602.09341, 2026, preprint; *AgentAuditor*). Agreement among agents
  is an unreliable correctness signal, so they audit reasoning trees to route around it. →
  Treats unreliable shared agreement as a nuisance to mitigate. **Stops at** building a
  mitigation; it does not *isolate and measure* shared error steps as the object of study.
- **Schwinn et al., "A Coin Flip for Safety: LLM Judges Fail to Reliably Measure Adversarial
  Robustness"** (arXiv:2603.06594, 2026, preprint). Under adversarial distribution shift,
  judge agreement degrades to near random chance against 6,642 human-verified labels. →
  Reinforces "high agreement ≠ correctness," at the verdict level, not the error-step level.

**The gap this fills:** no prior work, *after capability-matching the judge arms*, measures
**error-step co-location** to adjudicate whether same-family favouritism is a shared
reasoning blind spot or self-preference. That adjudication is this repo's contribution.

## Method (the instrument)

- **Mechanical, never-LLM error-step locator** (`src/error_locator.py`). Given a
  chain-of-thought, it finds the first step that goes wrong by **pure re-execution** — exact
  when the CoT aligns 1:1 with the canonical operations, with an insertion/deletion-tolerant
  value-trace fallback. It never calls an LLM, because an LLM-judged locator would reintroduce
  the very bias under study. Unit-tested on 1,200+ synthetically-corrupted CoTs (`./run_tests.sh`,
  25 tests, model-free).
- **SER** (shared-error rate) = among false endorsements where *both* the answerer and the
  judge's *independent* solution are wrong+localized, the fraction sharing the **same canonical
  error step**. **Chance baseline** = Σ_k p_M(k)·p_J(k) from each side's empirical error-step
  marginals. **Excess co-location** = SER − chance — the decisive per-arm quantity.
- **Capability matching** (v2): on a held-out set, tune difficulty so the same- and
  cross-family judges have comparable standalone solve accuracy, so the same-vs-cross contrast
  isn't a capability difference in disguise.

Full definitions: [`results/RESEARCH_REPORT.md`](results/RESEARCH_REPORT.md) (v1) and
[`results/phase2/RESEARCH_REPORT_v2.md`](results/phase2/RESEARCH_REPORT_v2.md) (v2).

## Results — both runs

**v1 (local, open-weight: Qwen2.5-0.5B answerer, Qwen2.5-1.5B same-judge, SmolLM2-1.7B
cross-judge; N=150).** SER(same) = 0.37 vs chance 0.22 and SER(cross) = 0.20 — excess
co-location +0.15 (same) vs +0.08 (cross). **Verdict: SUGGESTIVE** — leaned positive, but the
perplexity-controlled coefficient's 95% CI just touched zero, *and the cross arm was
capability-confounded* (Qwen is unusually math-strong for its size, so the size-matched
non-Qwen judge differed in ability). That confound is exactly why v2 exists.

**v2 (capability-matched, cross-family API: gpt-4o-mini answerer, gpt-4o same-judge,
claude-haiku cross-judge, matched to within 3pp on held-out calibration; N=500).**

| metric | same-family (gpt-4o) | cross-family (claude-haiku) |
|---|---|---|
| false-endorsement rate (FER) | **0.38** | 0.18 |
| shared-error rate (SER) | 0.22 | 0.31 |
| chance baseline | 0.06 | 0.05 |
| **excess co-location (SER − chance)** | **+0.16** | +0.26 |
| usable shared-error cases (n) | 138 | 51 |

**Verdict: NULL on shared blind spot.** The decisive quantity — excess(same) − excess(cross),
via a problem-level bootstrap — is **gap = −0.10, 95% CI [−0.24, +0.04], with only 8% of
resamples positive**. Once the arms are capability-matched, same-family excess co-location
does **not** exceed cross-family; if anything it is lower. The v1 signal does not survive.
([`excess_gap.png`](results/phase2/excess_gap.png) is the decisive figure.)

**The positive finding inside the NULL.** Same-family over-endorsement *does* survive
capability matching: FER(same) = 0.38 vs FER(cross) = 0.18 (~2.2×) — and conservatively so,
since the cross judge was actually the *more* capable solver here (34% vs 24%) yet endorsed
less. But that favouritism is **not** error-step-coupled (excess is not family-specific), so
its mechanism is **self-preference / leniency, not a shared blind spot**. The smoking gun: in
**41 of 180** same-family false endorsements, the judge had solved the problem **correctly
itself** and still endorsed the wrong answer.

## Limitations (read these)

- **The familiarity (perplexity) control was not computable in v2.** Teacher-forced perplexity
  on a *provided* CoT needs echo logprobs, which neither the gpt-4o-class chat API nor the
  Claude Messages API exposes. So the v2 decisive statistic is deliberately perplexity-free;
  on templated arithmetic perplexity barely varied in v1 anyway (≈1.1–1.2, and in the *wrong*
  direction for a familiarity story). Restoring a real familiarity control is the main open
  thread.
- **Realized capability gap.** The in-run judge solve-accuracy gap (~10pp, cross stronger) ran
  wider than the held-out calibration gap (3pp). Excess subtracts each arm's own chance
  baseline, which mitigates but does not erase this; it is flagged in the report and could
  contribute to the cross arm's slightly higher excess.
- **Scope.** A single templated domain, small open models in v1, one model pair per arm, modest
  usable-case counts. No Layer-3 representation analysis was attempted — it was gated on a
  positive v2, which (correctly) was never reached.

## Phase 3 — what would settle the open thread

A free-form / less-templated domain (word problems, multi-hop unit conversions) where
perplexity genuinely varies, using echo-capable models so the familiarity control returns.
Clearly a future direction, not a promise. See [ROADMAP.md](ROADMAP.md).

## Reproduce

```bash
# Model-free gate: generator + mechanical error-step locator (25 tests, stdlib only)
./run_tests.sh

# v1 — local, open-weight, one fresh process per model (Apple MPS), incremental JSONL
python -m venv .venv && .venv/bin/pip install -r requirements.txt
./run_all.sh                              # config in src/expconfig.py

# v2 — capability-matched cross-family API run (OpenAI + Anthropic keys in .env)
.venv/bin/pip install openai anthropic python-dotenv
PYTHONPATH=src .venv/bin/python src/phase2_calibrate.py search 20   # difficulty/capability sweep
PYTHONPATH=src .venv/bin/python src/phase2_run.py                   # cached, resumable; config in src/phase2config.py
PYTHONPATH=src .venv/bin/python src/phase2_assemble.py              # -> run JSON + figures + report
```

Every API call is cached to disk by content hash (`results/cache/`), so a run is fully
resumable and never re-billed. v1 models each run in their **own process** — loading several
into one process corrupts the Apple-MPS pool and hangs. Run artifacts:
[`results/runs/run_v1.json`](results/runs/run_v1.json),
[`results/phase2/runs/run_v2_api.json`](results/phase2/runs/run_v2_api.json); calibration record
in [`results/phase2/calibration.md`](results/phase2/calibration.md).

### Reuse just the locator

```python
from arithmetic import generate_problem, corrupt_cot
from error_locator import locate_canonical_index

p = generate_problem(seed=1, n_steps=6)
claimed, truth = corrupt_cot(p, index=3, kind="arithmetic", seed=0)  # inject a known error at step 3
idx, method = locate_canonical_index(p.render_cot(claimed), p)
assert idx == truth.index    # -> 3   (method == 'strict' when the CoT aligns 1:1)
```

## Status

Independent, preprint-stage research; **not peer-reviewed**. All numbers above come from the
runs described and the committed run JSON / figures — included for inspection. Scrutiny,
replication, and "this is wrong because…" issues are welcome (see
[CONTRIBUTING.md](CONTRIBUTING.md)). License: [MIT](LICENSE) · cite: [CITATION.cff](CITATION.cff).
