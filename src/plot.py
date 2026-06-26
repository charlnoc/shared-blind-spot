"""Produce the spec §7 deliverable figures from a run JSON.

  ser_vs_null.png          -- HEADLINE: SER(same) vs SER(cross) vs chance (§5.5)
  fer_by_family_domain.png -- Layer-1 false-endorsement rate by family (§4)
  ser_perplexity_partial.png -- SER within perplexity strata + control summary (§4)

Run: .venv/bin/python src/plot.py results/runs/run_<tag>.json
"""

from __future__ import annotations

import math
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

sys.path.insert(0, os.path.dirname(__file__))
from metrics import (  # noqa: E402
    _wrong_localized, chance_baseline, fer, load, perplexity_control, ser, wilson,
)

RESULTS = os.path.join(os.path.dirname(__file__), "..", "results")
COL = {"same": "#d1495b", "cross": "#30638e", "null": "#9aa0a6"}


def _err(p, ci):
    if p is None or math.isnan(p) or math.isnan(ci[0]) or math.isnan(ci[1]):
        return None  # matplotlib accepts yerr=None; avoids NaN errorbar crashes
    return [[max(0, p - ci[0])], [max(0, ci[1] - p)]]


def _h(p):  # bar height safe for NaN
    return 0 if p is None or math.isnan(p) else p


def fig_ser_vs_null(recs, fams, out):
    fig, ax = plt.subplots(figsize=(6.2, 4.6))
    xs, labels = [], []
    for i, fam in enumerate(fams):
        s = ser(recs, fam)
        c = chance_baseline(recs, fam)
        ax.bar(i, _h(s["ser"]), color=COL[fam], width=0.62,
               yerr=_err(s["ser"], s["ci"]), capsize=5,
               label=f"SER ({fam})")
        ax.hlines(c["null"], i - 0.31, i + 0.31, color="black", linestyles="--", lw=2)
        ax.text(i, _h(s["ser"]) + 0.03,
                f"{s['ser']:.2f}\nn={s['n_usable']}\n(shared {s['n_shared']})",
                ha="center", va="bottom", fontsize=9)
        ax.text(i, c["null"] + 0.005, f"chance {c['null']:.2f}", ha="center",
                va="bottom", fontsize=8, color="black")
        xs.append(i); labels.append(f"{fam}-family")
    ax.set_xticks(xs); ax.set_xticklabels(labels)
    ax.set_ylabel("Shared-error rate among false endorsements")
    ax.set_ylim(0, 1.05)
    ax.set_title("SER vs chance baseline (dashed = null)\n"
                 "Shared blind spot predicts SER(same) >> chance & cross")
    ax.legend(loc="upper right", fontsize=8)
    fig.tight_layout(); fig.savefig(out, dpi=140); plt.close(fig)
    return out


def fig_fer(recs, fams, out):
    fig, ax = plt.subplots(figsize=(5.6, 4.4))
    for i, fam in enumerate(fams):
        f = fer(recs, fam)
        ax.bar(i, _h(f["fer"]), color=COL[fam], width=0.6,
               yerr=_err(f["fer"], f["ci"]), capsize=5)
        ax.text(i, _h(f["fer"]) + 0.02,
                f"{f['fer']:.2f}\nn={f['n_judged']}", ha="center", va="bottom", fontsize=9)
    ax.set_xticks(range(len(fams))); ax.set_xticklabels([f"{x}-family" for x in fams])
    ax.set_ylabel("False endorsement rate  P(endorse | M wrong)")
    ax.set_ylim(0, 1.05)
    ax.set_title("Layer 1: false-endorsement rate by family (arithmetic)")
    fig.tight_layout(); fig.savefig(out, dpi=140); plt.close(fig)
    return out


def fig_ppl_partial(recs, fams, out):
    """SER within low/high perplexity strata (median split of same-family ppl),
    plus the regression-control summary as text."""
    ctrl = perplexity_control(recs, fams)
    # median split on perplexity pooled across judges (FE usable cases)
    ppls = []
    for r in recs:
        if r["M"]["correct"]:
            continue
        for fam in fams:
            j = r["judges"][fam]
            if j["endorsed"] is True and _wrong_localized(r["M"]) and _wrong_localized(j["solve"]):
                if j["ppl_on_M"] is not None and not math.isnan(j["ppl_on_M"]):
                    ppls.append(j["ppl_on_M"])
    fig, (ax, axt) = plt.subplots(1, 2, figsize=(10.5, 4.4), gridspec_kw={"width_ratios": [1.3, 1]})
    if ppls:
        med = sorted(ppls)[len(ppls) // 2]
        for bi, (blab, lohi) in enumerate([("ppl<=med", (-1, med)), ("ppl>med", (med, 1e18))]):
            for fam in fams:
                shared = usable = 0
                for r in recs:
                    if r["M"]["correct"]:
                        continue
                    j = r["judges"][fam]
                    if j["endorsed"] is True and _wrong_localized(r["M"]) and _wrong_localized(j["solve"]):
                        ppl = j["ppl_on_M"]
                        if ppl is not None and not math.isnan(ppl) and lohi[0] < ppl <= lohi[1]:
                            usable += 1
                            shared += r["M"]["err_index"] == j["solve"]["err_index"]
                p, lo, hi = wilson(shared, usable)
                xpos = bi + (0.18 if fam == "same" else -0.18)
                ax.bar(xpos, p if not math.isnan(p) else 0, width=0.34, color=COL[fam],
                       label=fam if bi == 0 else None,
                       yerr=_err(p, (lo, hi)) if usable else None, capsize=4)
                ax.text(xpos, (p if not math.isnan(p) else 0) + 0.02, f"n={usable}",
                        ha="center", va="bottom", fontsize=8)
        ax.set_xticks([0, 1]); ax.set_xticklabels(["low perplexity", "high perplexity"])
        ax.set_ylabel("SER"); ax.set_ylim(0, 1.05)
        ax.set_title("SER within perplexity strata\n(does same>cross survive the split?)")
        ax.legend(fontsize=8)
    else:
        ax.text(0.5, 0.5, "no usable FE cases", ha="center", transform=ax.transAxes)

    axt.axis("off")
    em = ctrl["endorsement_model"]; sm = ctrl["shared_error_model"]
    lines = ["Perplexity control (§4)", "-" * 30,
             f"mean ppl(same)  = {ctrl['mean_ppl'].get('same', float('nan')):.2f}",
             f"mean ppl(cross) = {ctrl['mean_ppl'].get('cross', float('nan')):.2f}",
             "", "Logistic: endorsed ~ same_family + log(ppl)"]
    if em.get("ok"):
        lo, hi = em["same_family_ci"]
        lines += [f"  coef(same_family) = {em['coef_same_family']:+.2f}",
                  f"  95% CI = [{lo:+.2f}, {hi:+.2f}]  (n={em['n']})",
                  f"  coef(log ppl)     = {em['coef_log_ppl']:+.2f}"]
    else:
        lines += [f"  (fit unavailable: {em.get('reason')})"]
    lines += ["", "Logistic: shared_error ~ same_family + log(ppl)"]
    if sm.get("ok"):
        lo, hi = sm["same_family_ci"]
        lines += [f"  coef(same_family) = {sm['coef_same_family']:+.2f}",
                  f"  95% CI = [{lo:+.2f}, {hi:+.2f}]  (n={sm['n']})"]
    else:
        lines += [f"  (fit unavailable: {sm.get('reason')})"]
    axt.text(0, 1, "\n".join(lines), va="top", family="monospace", fontsize=9.5)
    fig.tight_layout(); fig.savefig(out, dpi=140); plt.close(fig)
    return out


def main(path):
    data = load(path)
    recs = data["records"]
    fams = [f for f, _ in data["config"]["JUDGES"]]
    os.makedirs(RESULTS, exist_ok=True)
    a = fig_ser_vs_null(recs, fams, os.path.join(RESULTS, "ser_vs_null.png"))
    b = fig_fer(recs, fams, os.path.join(RESULTS, "fer_by_family_domain.png"))
    c = fig_ppl_partial(recs, fams, os.path.join(RESULTS, "ser_perplexity_partial.png"))
    print("wrote:", a, b, c, sep="\n  ")


if __name__ == "__main__":
    main(sys.argv[1])
