"""Layer 1/2 metrics (spec §4, §5): FER, SER, the chance baseline, and the
perplexity control. Pure analysis over a run JSON — no models.

Key definitions (verbatim from the spec):
  FER  = P(endorsed | M wrong)                                         (§4.4)
  SER  = among false-endorsement cases (M wrong AND endorsed), the
         fraction where the judge's INDEPENDENT solution is also wrong
         at the SAME canonical error step as M.                        (§5.4)
  null = chance two independent errorful solutions share an error step,
         = sum_k p_M(k) * p_J(k) over the empirical error-step marginals.(§5.5)

The decisive claim (§5.5): SER(same) significantly exceeds BOTH SER(cross)
and the chance baseline, AND survives conditioning on perplexity (§4).
"""

from __future__ import annotations

import json
import math
from collections import Counter


def load(path: str) -> dict:
    with open(path) as f:
        return json.load(f)


def wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float, float]:
    """Point estimate + Wilson 95% CI for a proportion."""
    if n == 0:
        return (float("nan"), float("nan"), float("nan"))
    p = k / n
    denom = 1 + z * z / n
    center = (p + z * z / (2 * n)) / denom
    half = (z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))) / denom
    return p, max(0.0, center - half), min(1.0, center + half)


def _wrong_localized(record_solve: dict) -> bool:
    """A solve usable for co-location: produced a localizable error step."""
    return (
        not record_solve["correct"]
        and record_solve["err_index"] is not None
        and record_solve["err_method"] != "unlocalizable"
    )


def fer(records: list[dict], family: str) -> dict:
    """False endorsement rate for one judge family."""
    wrong = [r for r in records if not r["M"]["correct"]]
    judged = [r for r in wrong if r["judges"][family]["endorsed"] is not None]
    endorsed = [r for r in judged if r["judges"][family]["endorsed"]]
    p, lo, hi = wilson(len(endorsed), len(judged))
    return {"family": family, "n_wrong": len(wrong), "n_judged": len(judged),
            "n_endorsed": len(endorsed), "fer": p, "ci": (lo, hi)}


def _error_marginal(indices: list[int]) -> dict[int, float]:
    c = Counter(indices)
    tot = sum(c.values())
    return {k: v / tot for k, v in c.items()} if tot else {}


def chance_baseline(records: list[dict], family: str) -> dict:
    """Expected co-location rate if M's and J's error steps were independent
    (§5.5): sum_k p_M(k) * p_J(k), using empirical error-step marginals over
    all wrong+localized solves (M, and this judge's independent solves)."""
    m_idx = [r["M"]["err_index"] for r in records if _wrong_localized(r["M"])]
    j_idx = [r["judges"][family]["solve"]["err_index"]
             for r in records if _wrong_localized(r["judges"][family]["solve"])]
    pm, pj = _error_marginal(m_idx), _error_marginal(j_idx)
    null = sum(pm.get(k, 0) * pj.get(k, 0) for k in set(pm) | set(pj))
    return {"family": family, "null": null, "n_M": len(m_idx), "n_J": len(j_idx)}


def ser(records: list[dict], family: str) -> dict:
    """Shared-error rate among false-endorsement cases (§5.4).

    Denominator: false-endorsement cases (M wrong, endorsed) where BOTH M and
    the judge's independent solve are wrong+localized (so co-location is even
    defined). We log how many FE cases are excluded for being unlocalizable.
    """
    fe = [r for r in records
          if not r["M"]["correct"] and r["judges"][family]["endorsed"] is True]
    # Denominator matches the null baseline's conditioning (§5.5): BOTH M and J
    # errorful+localized. The carved-out sub-counts are themselves informative:
    #   j_solved_correct = endorsed M's wrong answer yet solved it right alone
    #                      -> the *pure self-preference / familiarity* signature.
    usable, shared = 0, 0
    m_unloc = j_correct = j_unloc = 0
    for r in fe:
        if not _wrong_localized(r["M"]):
            m_unloc += 1
            continue
        js = r["judges"][family]["solve"]
        if js["correct"]:
            j_correct += 1
        elif not _wrong_localized(js):
            j_unloc += 1
        else:
            usable += 1
            shared += r["M"]["err_index"] == js["err_index"]
    p, lo, hi = wilson(shared, usable)
    return {"family": family, "n_fe": len(fe), "n_usable": usable, "n_shared": shared,
            "n_M_unlocalizable": m_unloc, "n_J_solved_correct": j_correct,
            "n_J_wrong_unlocalizable": j_unloc, "ser": p, "ci": (lo, hi)}


def coverage(records: list[dict]) -> dict:
    """Localization coverage diagnostics (spec 'no silent caps')."""
    def rate(pred):
        xs = [pred(r) for r in records]
        return sum(xs), len(xs)
    m_methods = Counter(r["M"]["err_method"] for r in records)
    return {
        "n": len(records),
        "M_methods": dict(m_methods),
        "M_unlocalizable_wrong": sum(
            1 for r in records
            if not r["M"]["correct"] and r["M"]["err_method"] == "unlocalizable"),
    }


def perplexity_control(records: list[dict], families: list[str]) -> dict:
    """The §4 familiarity control. Two parts:

    (1) Mean perplexity by family — establishes the confound exists (same-family
        judge should find M's CoT *more* familiar => lower perplexity).
    (2) Logistic regressions pooled across judges, with `same_family` AND
        log-perplexity as predictors. If the `same_family` coefficient stays
        positive after controlling for perplexity, the family effect is NOT
        explained by familiarity alone (the novel claim). Bootstrap CI on the
        coefficient because n is small.

    Targets: endorsement (over M-wrong cases) and shared-error (over FE usable
    cases). Falls back to NaN coef (flagged) if the fit fails to converge.
    """
    import warnings

    import numpy as np
    from sklearn.linear_model import LogisticRegression
    warnings.filterwarnings("ignore", category=FutureWarning)

    def fit(rows):
        # rows: list of (same_family:0/1, ppl:float, y:0/1)
        if len(rows) < 8:
            return {"ok": False, "reason": f"too few rows ({len(rows)})"}
        X = np.array([[sf, math.log(max(ppl, 1e-6))] for sf, ppl, _ in rows])
        y = np.array([yy for _, _, yy in rows])
        if len(set(y)) < 2:
            return {"ok": False, "reason": "no outcome variance"}
        def coef(Xs, ys):
            # C large => effectively unregularized (replaces deprecated penalty=None)
            m = LogisticRegression(C=1e9, solver="lbfgs", max_iter=2000)
            m.fit(Xs, ys)
            return m.coef_[0]  # [same_family, log_ppl]
        try:
            base = coef(X, y)
        except Exception as e:  # noqa: BLE001
            return {"ok": False, "reason": str(e)}
        # bootstrap CI on same_family coefficient
        rng = np.random.default_rng(0)
        boots = []
        for _ in range(800):
            idx = rng.integers(0, len(y), len(y))
            if len(set(y[idx])) < 2:
                continue
            try:
                boots.append(coef(X[idx], y[idx])[0])
            except Exception:  # noqa: BLE001
                pass
        lo, hi = (float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))) if boots else (float("nan"), float("nan"))
        return {"ok": True, "n": len(rows),
                "coef_same_family": float(base[0]), "coef_log_ppl": float(base[1]),
                "same_family_ci": (lo, hi)}

    endorse_rows, ser_rows = [], []
    for r in records:
        if r["M"]["correct"]:
            continue
        for fam in families:
            j = r["judges"][fam]
            sf = 1 if fam == "same" else 0
            ppl = j["ppl_on_M"]
            if j["endorsed"] is not None and ppl is not None and not math.isnan(ppl):
                endorse_rows.append((sf, ppl, 1 if j["endorsed"] else 0))
            if j["endorsed"] is True and _wrong_localized(r["M"]) and _wrong_localized(j["solve"]):
                shared = 1 if r["M"]["err_index"] == j["solve"]["err_index"] else 0
                if ppl is not None and not math.isnan(ppl):
                    ser_rows.append((sf, ppl, shared))

    return {
        "mean_ppl": {fam: _mean([r["judges"][fam]["ppl_on_M"] for r in records]) for fam in families},
        "endorsement_model": fit(endorse_rows),
        "shared_error_model": fit(ser_rows),
    }


def summary(path: str) -> dict:
    data = load(path)
    recs = data["records"]
    fams = [f for f, _ in data["config"]["JUDGES"]]
    out = {"path": path, "n": len(recs),
           "M_accuracy": sum(r["M"]["correct"] for r in recs) / len(recs),
           "coverage": coverage(recs), "by_family": {}}
    for fam in fams:
        out["by_family"][fam] = {
            "fer": fer(recs, fam),
            "ser": ser(recs, fam),
            "chance": chance_baseline(recs, fam),
            "mean_ppl_on_M": _mean([r["judges"][fam]["ppl_on_M"] for r in recs]),
        }
    out["perplexity_control"] = perplexity_control(recs, fams)
    return out


def _mean(xs):
    xs = [x for x in xs if x is not None and not math.isnan(x)]
    return sum(xs) / len(xs) if xs else float("nan")


if __name__ == "__main__":
    import sys
    print(json.dumps(summary(sys.argv[1]), indent=2, default=str))
