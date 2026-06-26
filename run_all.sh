#!/usr/bin/env bash
# Robust runner: each model in its OWN fresh process (clean MPS state, no
# repeated alloc/free that caused the monolith to hang). Incremental JSONL means
# any phase can be re-run without losing prior phases' work.
set -euo pipefail
cd "$(dirname "$0")"
export PYTHONUNBUFFERED=1
export TOKENIZERS_PARALLELISM=false
P=".venv/bin/python"
R="results/runs"
mkdir -p "$R"

echo "=== [1/4] M solve (Qwen2.5-0.5B) ==="
$P src/phase_solve.py "$R/_M.jsonl"

echo "=== [2/4] judge SAME (Qwen2.5-1.5B) ==="
$P src/phase_judge.py same qwen1.5b "$R/_M.jsonl" "$R/_same.jsonl"

echo "=== [3/4] judge CROSS (SmolLM2-1.7B) ==="
$P src/phase_judge.py cross smollm2-1.7b "$R/_M.jsonl" "$R/_cross.jsonl"

echo "=== [4/4] assemble + analyze + report ==="
$P src/assemble.py

echo "ALL DONE"
