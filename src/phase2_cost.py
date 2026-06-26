"""Cost accounting for the dry-run gate.

Sums token usage from cache files (each stores per-call prompt/completion tokens
in `meta`), applies a per-model price table, and extrapolates a small dry run to
the full N. To isolate just the dry run's spend, pass --since <epoch> captured
immediately before launching it; only cache files written at/after that time are
counted (calibration calls, already cached, keep their old mtime).

Prices are USD per 1M tokens, list prices as of 2026-06 (ESTIMATES — the
authoritative spend is in the OpenAI/Anthropic dashboards). Edit as needed.

Usage:
  python src/phase2_cost.py [--since EPOCH] [--dry-n 10] [--full-n 200]
"""
from __future__ import annotations

import json
import os
import sys

CACHE_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "results", "cache")

PRICES = {  # model_id substring -> (in_per_1M, out_per_1M)
    "gpt-4o-mini": (0.15, 0.60),
    "gpt-4o": (2.50, 10.00),
    "gpt-4.1-mini": (0.40, 1.60),
    "claude-haiku-4-5": (1.00, 5.00),
    "claude-sonnet-4-5": (3.00, 15.00),
}


def price_for(model_id: str):
    # longest matching key wins (so 'gpt-4o-mini' beats 'gpt-4o')
    best = None
    for k, v in PRICES.items():
        if k in model_id and (best is None or len(k) > len(best[0])):
            best = (k, v)
    return best[1] if best else (None, None)


def collect(since: float | None):
    per_model = {}
    for fn in os.listdir(CACHE_DIR):
        if not fn.endswith(".json"):
            continue
        path = os.path.join(CACHE_DIR, fn)
        if since is not None and os.stat(path).st_mtime < since:
            continue
        try:
            with open(path) as f:
                rec = json.load(f)
        except Exception:  # noqa: BLE001
            continue
        m = rec.get("model", "?")
        meta = rec.get("meta") or {}
        d = per_model.setdefault(m, {"calls": 0, "in": 0, "out": 0})
        d["calls"] += 1
        d["in"] += meta.get("prompt_tokens") or 0
        d["out"] += meta.get("completion_tokens") or 0
    return per_model


def main():
    since = None
    dry_n, full_n = 10, 200
    args = sys.argv[1:]
    for i, a in enumerate(args):
        if a == "--since":
            since = float(args[i + 1])
        elif a == "--dry-n":
            dry_n = int(args[i + 1])
        elif a == "--full-n":
            full_n = int(args[i + 1])

    per_model = collect(since)
    scale = full_n / dry_n
    print(f"\n=== dry-run cost ({'since '+str(since) if since else 'ALL cache'}) "
          f"-> extrapolate x{scale:.1f} to N={full_n} ===\n")
    print(f"{'model':28s} {'calls':>6s} {'in_tok':>9s} {'out_tok':>9s} "
          f"{'dry $':>8s} {'full $':>8s}")
    tot_dry = {"openai": 0.0, "anthropic": 0.0}
    tot_full = {"openai": 0.0, "anthropic": 0.0}
    for m, d in sorted(per_model.items()):
        pin, pout = price_for(m)
        if pin is None:
            print(f"{m:28s} {d['calls']:6d} {d['in']:9d} {d['out']:9d}   (no price)")
            continue
        dry = d["in"] * pin / 1e6 + d["out"] * pout / 1e6
        full = dry * scale
        prov = "anthropic" if "claude" in m else "openai"
        tot_dry[prov] += dry
        tot_full[prov] += full
        print(f"{m:28s} {d['calls']:6d} {d['in']:9d} {d['out']:9d} "
              f"{dry:8.3f} {full:8.2f}")
    print("\n--- provider totals (extrapolated to full N) ---")
    for prov in ("openai", "anthropic"):
        print(f"  {prov:10s} dry=${tot_dry[prov]:.3f}  full≈${tot_full[prov]:.2f}  "
              f"(cap $20 -> {'OK' if tot_full[prov] < 20 else 'OVER BUDGET'})")
    print(f"\n  GRAND TOTAL full run ≈ ${sum(tot_full.values()):.2f} "
          f"(both providers; each capped at $20)\n")


if __name__ == "__main__":
    main()
