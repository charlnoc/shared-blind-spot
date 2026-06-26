#!/usr/bin/env bash
# Gate for spec §9 step 2: the deterministic backbone must pass before any
# model-inference work begins. No third-party dependencies — stdlib only.
set -euo pipefail
cd "$(dirname "$0")"
PYTHONPATH="src:tests" python3 -m unittest test_backbone -v
