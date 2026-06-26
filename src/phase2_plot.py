"""Phase-2-specific figures (the perplexity figure is N/A on the chat APIs, so
it is replaced by the decisive excess-co-location visuals).

  excess_gap.png  -- LEFT: SER vs chance per arm (excess shaded);
                     RIGHT: bootstrap distribution of excess(same)-excess(cross)
                     with its 95% CI and the zero line. This is THE Phase-2 plot.
"""
from __future__ import annotations

import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

sys.path.insert(0, os.path.dirname(__file__))
from metrics import chance_baseline, excess_gap_bootstrap, load, ser  # noqa: E402

COL = {"same": "#d1495b", "cross": "#30638e", "null": "#9aa0a6", "gap": "#2a9d8f"}


def fig_excess_gap(records, out, fam_same="same", fam_cross="cross"):
    boot = excess_gap_bootstrap(records, fam_same, fam_cross, n_boot=4000, return_gaps=True)
    s_same, s_cross = ser(records, fam_same), ser(records, fam_cross)
    c_same, c_cross = chance_baseline(records, fam_same), chance_baseline(records, fam_cross)

    fig, (axL, axR) = plt.subplots(1, 2, figsize=(11, 4.6))

    # LEFT: SER vs chance per arm, excess annotated
    arms = [("same-family\n(gpt-4o)", s_same["ser"], c_same["null"], COL["same"], s_same["n_usable"]),
            ("cross-family\n(claude-haiku)", s_cross["ser"], c_cross["null"], COL["cross"], s_cross["n_usable"])]
    x = range(len(arms))
    w = 0.36
    for i, (lab, serv, ch, col, nus) in enumerate(arms):
        axL.bar(i - w / 2, serv, w, color=col, label="SER" if i == 0 else None)
        axL.bar(i + w / 2, ch, w, color=COL["null"], label="chance" if i == 0 else None)
        axL.annotate(f"excess\n{serv-ch:+.3f}", (i, max(serv, ch) + 0.02),
                     ha="center", va="bottom", fontsize=9, color=col, weight="bold")
        axL.annotate(f"n={nus}", (i - w / 2, serv + 0.005), ha="center", va="bottom", fontsize=8)
    axL.set_xticks(list(x)); axL.set_xticklabels([a[0] for a in arms])
    axL.set_ylabel("rate among usable false-endorsements")
    axL.set_title("Shared-error rate vs chance, per arm")
    axL.legend(frameon=False, loc="upper right")
    axL.set_ylim(0, max(s_same["ser"], s_cross["ser"], 0.1) * 1.35 + 0.05)

    # RIGHT: bootstrap distribution of the gap
    gaps = boot.get("gaps", [])
    if gaps:
        axR.hist(gaps, bins=40, color=COL["gap"], alpha=0.8)
    axR.axvline(0, color="k", lw=1.2, ls="--", label="no difference")
    lo, hi = boot["ci"]
    axR.axvspan(lo, hi, color=COL["gap"], alpha=0.15, label="95% CI")
    axR.axvline(boot["gap"], color=COL["gap"], lw=2,
                label=f"gap={boot['gap']:+.3f}")
    axR.set_xlabel("excess(same) − excess(cross), bootstrap")
    axR.set_ylabel("resamples")
    axR.set_title(f"Decisive gap: 95% CI [{lo:+.3f}, {hi:+.3f}],  "
                  f"{boot['frac_gap_positive']:.0%} of resamples > 0")
    axR.legend(frameon=False, fontsize=8)

    fig.tight_layout()
    fig.savefig(out, dpi=140)
    plt.close(fig)
    return out, boot


def main(path, out_dir):
    data = load(path)
    return fig_excess_gap(data["records"], os.path.join(out_dir, "excess_gap.png"))


if __name__ == "__main__":
    out_dir = sys.argv[2] if len(sys.argv) > 2 else os.path.dirname(sys.argv[1])
    print(main(sys.argv[1], out_dir))
