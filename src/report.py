"""Generate results/findings.md (spec §7) in plain language from a run JSON.

States: (a) does FER differ by family, (b) does SER exceed chance/cross after
the perplexity control, (c) the one-sentence verdict on whether shared blind
spot is a distinct, detectable mechanism here. Honest about negative results
(spec §5: a null is a real finding — do not p-hack toward positive).
"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(__file__))
from metrics import load, summary  # noqa: E402


def _verdict(s: dict) -> tuple[str, str]:
    same, cross = s["by_family"]["same"], s["by_family"]["cross"]
    ser_s, ser_c = same["ser"], cross["ser"]
    null_s = same["chance"]["null"]
    ctrl = s["perplexity_control"]
    sm = ctrl["shared_error_model"]

    if ser_s["n_usable"] < 8:
        return ("UNDERPOWERED",
                f"Only {ser_s['n_usable']} usable same-family false-endorsement cases — "
                f"too few to decide. Increase N or judge endorsement rate.")

    beats_cross = ser_s["ser"] > ser_c["ser"]
    beats_null = ser_s["ci"][0] > null_s  # CI lower bound above chance
    survives_ppl = sm.get("ok") and sm["same_family_ci"][0] > 0  # coef CI excludes 0 (positive)

    if beats_null and beats_cross and survives_ppl:
        return ("POSITIVE",
                "SER(same) significantly exceeds chance AND cross-family AND the same-family "
                "effect survives conditioning on perplexity — evidence for a shared blind spot "
                "distinct from surface familiarity.")
    if beats_null and beats_cross:
        return ("SUGGESTIVE",
                "SER(same) exceeds chance and cross-family, but the perplexity-controlled "
                "coefficient is not conclusively positive — cannot yet rule out familiarity.")
    return ("NULL/NEGATIVE",
            "SER(same) does not clearly exceed both the chance baseline and cross-family. "
            "On this minimal local case, shared blind spot is not a separately detectable "
            "mechanism — consistent with self-preference/familiarity explaining the data.")


def build(path: str) -> str:
    s = summary(path)
    same, cross = s["by_family"]["same"], s["by_family"]["cross"]
    ctrl = s["perplexity_control"]
    em, sm = ctrl["endorsement_model"], ctrl["shared_error_model"]
    tag, sentence = _verdict(s)

    def ci(d):
        return f"[{d['ci'][0]:.2f}, {d['ci'][1]:.2f}]"

    L = []
    L.append("# Findings — Shared Blind Spot vs Self-Preference (minimal local case)\n")
    L.append(f"_Run: `{os.path.basename(path)}`, N={s['n']} arithmetic problems. "
             "M = Qwen2.5-0.5B, J_same = Qwen2.5-1.5B (same family), "
             "J_cross = SmolLM2-1.7B (cross family)._\n")
    L.append(f"**M accuracy:** {s['M_accuracy']:.0%} "
             f"(wrong cases are the raw material; §2 target was errorful).\n")

    L.append("## (a) Does false endorsement differ by family? (Layer 1, §4)\n")
    L.append(f"- FER(same)  = **{same['fer']['fer']:.2f}** {ci(same['fer'])}  "
             f"(endorsed {same['fer']['n_endorsed']}/{same['fer']['n_judged']} wrong answers)")
    L.append(f"- FER(cross) = **{cross['fer']['fer']:.2f}** {ci(cross['fer'])}  "
             f"(endorsed {cross['fer']['n_endorsed']}/{cross['fer']['n_judged']} wrong answers)\n")
    L.append("> A same>cross gap here is necessary but NOT sufficient for the novel claim — "
             "self-preference predicts the same gap. Layer 2 is decisive.\n")

    L.append("## (b) Does SER exceed chance / cross after the perplexity control? (Layer 2, §5)\n")
    L.append(f"- SER(same)  = **{same['ser']['ser']:.2f}** {ci(same['ser'])}  "
             f"(shared {same['ser']['n_shared']}/{same['ser']['n_usable']} usable FE cases; "
             f"chance = {same['chance']['null']:.2f})")
    L.append(f"- SER(cross) = **{cross['ser']['ser']:.2f}** {ci(cross['ser'])}  "
             f"(shared {cross['ser']['n_shared']}/{cross['ser']['n_usable']} usable FE cases; "
             f"chance = {cross['chance']['null']:.2f})\n")
    L.append(f"- Mean perplexity of M's CoT: same = {ctrl['mean_ppl'].get('same', float('nan')):.2f}, "
             f"cross = {ctrl['mean_ppl'].get('cross', float('nan')):.2f} "
             "(lower for same = the familiarity confound we must control for).")
    if sm.get("ok"):
        lo, hi = sm["same_family_ci"]
        L.append(f"- Perplexity-controlled logistic `shared_error ~ same_family + log(ppl)`: "
                 f"coef(same_family) = **{sm['coef_same_family']:+.2f}** 95% CI [{lo:+.2f}, {hi:+.2f}] "
                 f"(n={sm['n']}). Positive CI excluding 0 ⇒ family effect survives familiarity.")
    else:
        L.append(f"- Perplexity-controlled SER model unavailable: {sm.get('reason')}.")
    if em.get("ok"):
        lo, hi = em["same_family_ci"]
        L.append(f"- (Endorsement model `endorsed ~ same_family + log(ppl)`: "
                 f"coef(same_family) = {em['coef_same_family']:+.2f} CI [{lo:+.2f}, {hi:+.2f}], n={em['n']}.)")
    L.append("")

    L.append("## (c) Verdict\n")
    L.append(f"**{tag}.** {sentence}\n")

    cov = s["coverage"]
    L.append("## Familiarity signature (diagnostic)\n")
    L.append("Among false-endorsement cases, how often did the judge endorse M's wrong answer "
             "yet solve the problem **correctly** on its own? That is the pure self-preference / "
             "familiarity signature (endorsement unrelated to sharing M's error):")
    L.append(f"- same:  {same['ser']['n_J_solved_correct']} / {same['ser']['n_fe']} FE cases")
    L.append(f"- cross: {cross['ser']['n_J_solved_correct']} / {cross['ser']['n_fe']} FE cases\n")

    L.append("## Coverage / honesty notes\n")
    L.append(f"- Error-step localization methods on M: {cov['M_methods']} "
             f"(strict = exact re-execution; valuetrace = insertion/deletion-tolerant fallback).")
    L.append(f"- Unlocalizable wrong M cases (excluded from SER): {cov['M_unlocalizable_wrong']}.")
    L.append(f"- SER denominator conditions on BOTH M and J errorful+localized (matches the §5.5 "
             "null). FE breakdown — same: M_unloc={a}, J_correct={b}, J_wrong_unloc={c}, usable={d}; "
             "cross: M_unloc={e}, J_correct={f}, J_wrong_unloc={g}, usable={h}.".format(
                 a=same['ser']['n_M_unlocalizable'], b=same['ser']['n_J_solved_correct'],
                 c=same['ser']['n_J_wrong_unlocalizable'], d=same['ser']['n_usable'],
                 e=cross['ser']['n_M_unlocalizable'], f=cross['ser']['n_J_solved_correct'],
                 g=cross['ser']['n_J_wrong_unlocalizable'], h=cross['ser']['n_usable']))
    L.append("- Known limitation of this minimal pass: J_same (Qwen-1.5B) and J_cross "
             "(SmolLM2-1.7B) are size-matched but not capability-matched on math; "
             "per §3 the plan is to add an API cross-family judge IF this signal is positive.")
    return "\n".join(L) + "\n"


if __name__ == "__main__":
    path = sys.argv[1]
    md = build(path)
    out = os.path.join(os.path.dirname(__file__), "..", "results", "findings.md")
    with open(out, "w") as f:
        f.write(md)
    print(md)
    print(f"\n-> wrote {out}")
