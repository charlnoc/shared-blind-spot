# Shared Blind Spot vs Self-Preference: Experiment Spec

## 0. The one thing this experiment exists to do

The literature on LLM-as-a-judge has already established **self-preference bias**: a judge model rates outputs from its own family higher, and the leading mechanistic explanation is *familiarity* — judges score lower-perplexity (more familiar-looking) text higher, regardless of whether they generated it. This is well-trodden. Do **not** rebuild it.

This experiment tests a **different** mechanism that the existing work does not isolate:

> **Shared blind spot.** When a judge and an answerer come from the same model (or same family), the judge fails to catch the answerer's error not because the answer *looks* familiar, but because the judge's own internal reasoning makes the *same error at the same step*. The judge can't see the mistake because it would make the mistake too.

The entire novelty of this work is **operationally distinguishing these two mechanisms**. If the experiment ends up only showing "same-family judges give higher scores," it has measured self-preference and contributes nothing new. Every design decision below exists to separate *surface familiarity* from *shared reasoning failure*.

Read this section again before writing any code. If a result could be explained by familiarity alone, it is not evidence for shared blind spot.

---

## 1. Core definitions (get these exactly right — the experiment is only as good as these)

- **Answerer (M):** a model that solves a task and emits a chain-of-thought (CoT) plus a final answer.
- **Judge (J):** a model that reads M's CoT + answer (NOT the ground truth) and decides whether the answer is correct.
- **Ground truth (GT):** known to us, never shown to J. Tasks MUST be auto-verifiable so GT is unambiguous.
- **False endorsement:** J judges an answer "correct" when GT says it is wrong. This is the failure event we study.
- **Error step:** the specific step in a CoT where the reasoning first diverges from a correct path. Locating this is the hardest and most important measurement in the whole spec (see §5).
- **Self-preference (the OLD mechanism):** false endorsement driven by surface features — style, phrasing, perplexity, verbosity, format familiarity.
- **Shared blind spot (the NEW mechanism):** false endorsement where J, asked to solve the same problem independently, *makes the same error at the same step* as M. The endorsement co-occurs with a shared reasoning failure, not just surface familiarity.

---

## 2. Task domains (pick auto-verifiable, error-localizable tasks)

We need tasks where (a) correctness is mechanically checkable, and (b) a wrong answer has an identifiable *error step*, not just a wrong final number. Use **three** domains so we can see whether the effect is domain-dependent:

1. **Multi-step arithmetic / word problems** (e.g. GSM8K-style, but generate fresh to avoid contamination). Error step = the first arithmetic or setup operation that goes wrong.
2. **Unit / dimensional conversion chains** (multi-hop: e.g. "X furlongs per fortnight to m/s"). Error step = the first conversion factor applied wrong.
3. **Simple symbolic logic / constraint satisfaction** (small SAT-like or syllogism chains). Error step = the first invalid inference.

For each domain, generate problems programmatically so you control difficulty and can compute GT directly in code. Aim for ~150–200 problems per domain. **Critical: select problems where M actually makes errors** — if M is near-100% correct, there are no false-endorsement events to study. Tune difficulty so M's accuracy lands roughly in the 50–80% range. Errorful answers are the raw material; do not discard them.

---

## 3. Models and the same-family / cross-family axis

This is the central manipulation. We need at least:

- **Same-family pairs:** M and J are the *same* model (self-judging) AND M and J are *different sizes of the same family* (e.g. small + large of one open family). Use open-weight models here (Qwen family, Pythia family, or similar) so §6 internal-representation work is possible.
- **Cross-family pairs:** M from family A, J from family B, roughly matched capability. Use a different open family, and optionally one API model as an additional cross-family judge.

Use open-weight models as the primary workhorses — they are cheap (Colab free tier / a few £ of rented GPU-hours for the small ones), give full CoT, and are the ONLY place §6 (representations) is possible. API models, if used at all, are a secondary cross-family judge for the behavioral layer only.

Keep a config block at the top of the code listing every (M, J) pairing with a `same_family: bool` flag.

---

## 4. Layer 1 — Behavioral experiment (establishes the phenomenon, but is NOT yet the novel result)

For every (M, J) pair, over all problems:

1. M solves each problem → store `{problem_id, cot_M, answer_M}`.
2. Compute `correct_M = (answer_M == GT)`.
3. J judges each (cot_M, answer_M) WITHOUT GT → store `{verdict_J in {correct, wrong}, judge_cot_J}`. **Always capture J's own reasoning for its verdict** — we need it in Layer 2.
4. Compute the key behavioral quantity:

   **False endorsement rate (FER)** = P(verdict_J = correct | correct_M = False)
   — i.e. of all the answers that are actually wrong, what fraction does J wave through.

5. Plot FER for same-family vs cross-family pairs, broken down by domain.

**Controls you MUST run, or the result is uninterpretable (these neutralize the familiarity confound):**
- **Position/verbosity controls:** standard mitigations — these are known confounds, hold them constant.
- **Perplexity control (the big one):** record J's perplexity on M's CoT. If same-family FER is fully explained by lower perplexity, that's self-preference, not blind spot. We need FER to remain elevated *after* conditioning on perplexity. Log perplexity per item so we can regress it out in Layer 2.
- **Paraphrase control:** take a *cross-family* wrong answer, paraphrase it into the *same-family* surface style (or vice versa), and re-judge. If endorsement follows the *style*, it's familiarity. If it follows the *reasoning content*, that points toward blind spot. This control is one of the cleanest separators — implement it carefully.

A same-family > cross-family FER gap here is necessary but NOT sufficient for the novel claim. Self-preference predicts the same gap. Layer 2 is what separates them.

---

## 5. Layer 2 — The novel result: error-step co-location

This is where the contribution lives. The question: **in false-endorsement cases, did J independently make the same error at the same step as M?**

For each false-endorsement case (M wrong, J said correct):

1. **Get J's independent solution.** Separately, have J *solve the same problem from scratch* (not judge — solve), producing `cot_J_solve, answer_J_solve`.
2. **Locate the error step in M's CoT.** Define a CoT as an ordered list of steps (parse on line breaks / numbered steps / operations — domain-specific). The error step is the index of the first step inconsistent with a correct derivation. Locate it by:
   - For arithmetic/conversion: re-execute each step in code, flag the first step whose output is wrong given its inputs.
   - For logic: check each inference against the rules, flag the first invalid one.
   - This must be **mechanical**, not LLM-judged, or you reintroduce the bias you're studying.
3. **Locate the error step in J's independent solution** the same way (if J also got it wrong).
4. **Co-location metric.** Define:

   **Shared-error rate (SER)** = among false-endorsement cases, the fraction where J's independent solution is *also wrong* AND its error step is the *same* step (same operation / same inference) as M's.

5. **The decisive comparison:**
   - **Shared blind spot predicts:** SER is high for same-family pairs — when J waves M's error through, it's because J would trip on the same step.
   - **Pure self-preference predicts:** SER is at chance — J's endorsement is about surface familiarity, uncorrelated with whether J shares the specific reasoning failure. J might solve it correctly on its own yet still endorse M's wrong answer (familiarity), or fail at a *different* step.

   **Plot SER (same-family) vs SER (cross-family), and crucially vs a null baseline** = the rate at which two *independent* errorful solutions would share an error step by chance (compute this from the empirical distribution of error-step locations). The novel evidence is: **SER for same-family false endorsements significantly exceeds both the cross-family SER and the chance baseline, even after regressing out perplexity.**

If that holds, you have shown false endorsement is driven by a *shared reasoning failure localized to the same step* — a mechanism distinct from surface familiarity. That is the new thing.

**Honest negative result:** if SER collapses to chance once perplexity is controlled, the data say self-preference/familiarity explains it and shared blind spot is not a separate detectable mechanism here. That is a real, publishable finding too — write it up as such. Do not p-hack toward the positive.

---

## 6. Layer 3 — (Stretch, open-weight only) Representation-level confirmation

Only attempt after Layers 1–2 produce a positive, perplexity-robust SER signal. This is what turns a behavioral curiosity into an interpretability result.

Question: is the shared error visible in internal representations *before* it surfaces in the CoT text — and is CoT even a faithful account of where J went wrong?

- For same-family shared-error cases, extract hidden states (a mid/late layer) at the token positions around the identified error step, for both M (generating) and J (independently solving).
- Test whether M's and J's representations at the error step are more similar to each other than at non-error steps, and more similar than cross-family pairs. (Use a simple representational-similarity measure; keep it cheap.)
- Faithfulness check: does the *textual* error step coincide with where the representations actually diverge from a correct trajectory, or does the CoT rationalize an error made elsewhere? This connects directly to the CoT-faithfulness literature and is the part no one has done for the shared-blind-spot setting.

Keep this lightweight. One clean RSA-style plot is enough for a first paper.

---

## 7. Outputs / deliverables

Produce, in a results folder:
- `fer_by_family_domain.png` — Layer 1 false-endorsement rates, same vs cross family, per domain, with perplexity-control variant.
- `ser_vs_null.png` — Layer 2 shared-error rate: same-family vs cross-family vs chance baseline. **This is the headline figure.**
- `ser_perplexity_partial.png` — SER after regressing out perplexity (shows the effect survives the familiarity confound).
- `rsa_error_step.png` — Layer 3 if attempted.
- A short `findings.md` stating, in plain language: (a) does FER differ by family, (b) does SER exceed chance/cross-family after perplexity control, (c) the one-sentence verdict on whether shared blind spot is a distinct, detectable mechanism.

---

## 8. Cost / scope guardrails

- Layers 1–2 are designed to run on API + small open-weight models for a few £ of compute. Do not reach for large models or GPU clusters before Layer 2 gives a positive signal.
- Start with ONE domain (arithmetic — easiest to localize error steps mechanically) and ONE same-family + ONE cross-family pair. Get `ser_vs_null.png` for that minimal case first. Only expand to all three domains and more pairings once the pipeline and the metric definitions are validated end-to-end.
- The single highest-risk piece of code is the **mechanical error-step locator** (§5.2). Build and unit-test it in isolation, on hand-constructed CoTs with known error steps, before running anything at scale. If error-step localization is noisy, every downstream metric is meaningless.

---

## 9. Order of operations for Claude Code

1. Build + unit-test the problem generators and the GT checker for the arithmetic domain.
2. Build + unit-test the mechanical error-step locator on synthetic CoTs with known error positions. **Gate: do not proceed until this is reliable.**
3. Wire up M-solve and J-judge calls for one same-family and one cross-family pair. Capture CoTs, verdicts, perplexities.
4. Compute FER + the chance baseline + SER for the minimal case. Produce `ser_vs_null.png`.
5. Add the perplexity partial-regression. Produce `ser_perplexity_partial.png`.
6. Only then: expand domains/pairs, and consider Layer 3.

Stop and report after step 4. That single figure decides whether the whole direction is worth continuing.
