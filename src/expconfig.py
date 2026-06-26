"""Shared run configuration so each isolated phase process rebuilds the SAME
problems and agrees on models. Difficulty was chosen by calibration
(Qwen2.5-1.5B ~38% accurate at this setting — errorful enough for SER)."""

DATASET = dict(n_steps=8, max_add=50, max_mul=4, seed0=70000)
N = 150
ANSWERER = "qwen0.5b"
JUDGES = [("same", "qwen1.5b"), ("cross", "smollm2-1.7b")]
RUN_TAG = "v1"
