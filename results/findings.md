# Findings — Shared Blind Spot vs Self-Preference (minimal local case)

_Run: `run_v1.json`, N=150 arithmetic problems. M = Qwen2.5-0.5B, J_same = Qwen2.5-1.5B (same family), J_cross = SmolLM2-1.7B (cross family)._

**M accuracy:** 8% (wrong cases are the raw material; §2 target was errorful).

## (a) Does false endorsement differ by family? (Layer 1, §4)

- FER(same)  = **0.53** [0.44, 0.62]  (endorsed 65/122 wrong answers)
- FER(cross) = **0.98** [0.95, 1.00]  (endorsed 128/130 wrong answers)

> A same>cross gap here is necessary but NOT sufficient for the novel claim — self-preference predicts the same gap. Layer 2 is decisive.

## (b) Does SER exceed chance / cross after the perplexity control? (Layer 2, §5)

- SER(same)  = **0.37** [0.23, 0.53]  (shared 14/38 usable FE cases; chance = 0.22)
- SER(cross) = **0.20** [0.14, 0.28]  (shared 24/119 usable FE cases; chance = 0.12)

- Mean perplexity of M's CoT: same = 1.21, cross = 1.13 (lower for same = the familiarity confound we must control for).
- Perplexity-controlled logistic `shared_error ~ same_family + log(ppl)`: coef(same_family) = **+0.91** 95% CI [-0.05, +1.87] (n=157). Positive CI excluding 0 ⇒ family effect survives familiarity.
- (Endorsement model `endorsed ~ same_family + log(ppl)`: coef(same_family) = -3.71 CI [-11.60, -2.62], n=252.)

## (c) Verdict

**SUGGESTIVE.** SER(same) exceeds chance and cross-family, but the perplexity-controlled coefficient is not conclusively positive — cannot yet rule out familiarity.

## Familiarity signature (diagnostic)

Among false-endorsement cases, how often did the judge endorse M's wrong answer yet solve the problem **correctly** on its own? That is the pure self-preference / familiarity signature (endorsement unrelated to sharing M's error):
- same:  26 / 65 FE cases
- cross: 9 / 128 FE cases

## Coverage / honesty notes

- Error-step localization methods on M: {'strict': 43, 'valuetrace': 107} (strict = exact re-execution; valuetrace = insertion/deletion-tolerant fallback).
- Unlocalizable wrong M cases (excluded from SER): 0.
- SER denominator conditions on BOTH M and J errorful+localized (matches the §5.5 null). FE breakdown — same: M_unloc=0, J_correct=26, J_wrong_unloc=1, usable=38; cross: M_unloc=0, J_correct=9, J_wrong_unloc=0, usable=119.
- Known limitation of this minimal pass: J_same (Qwen-1.5B) and J_cross (SmolLM2-1.7B) are size-matched but not capability-matched on math; per §3 the plan is to add an API cross-family judge IF this signal is positive.
