"""Re-judge the pilot's saved M solutions with the new few-shot judge prompt,
to confirm a judge (esp. SmolLM2) now uses BOTH verdict labels and reasons,
instead of defaulting to 'wrong'. Reuses stored M CoTs (no M re-run)."""
import json, sys
sys.path.insert(0, "src")
from arithmetic import generate_dataset
from models import LM
from prompts import build_judge_messages, parse_verdict
from pipeline import DATASET

key = sys.argv[1] if len(sys.argv) > 1 else "smollm2-1.7b"
recs = json.load(open("results/runs/run_pilot.json"))["records"]
probs = generate_dataset(len(recs), **DATASET)  # same seeds -> same problems

lm = LM(key)
endorsed_on_correct = endorsed_on_wrong = 0
n_correct = n_wrong = none = 0
for x, p in zip(recs, probs):
    out = lm.chat(build_judge_messages(p, x["M"]["cot"], x["M"]["answer"]))
    v = parse_verdict(out)
    if v is None:
        none += 1; continue
    if x["M"]["correct"]:
        n_correct += 1; endorsed_on_correct += v
    else:
        n_wrong += 1; endorsed_on_wrong += v

print(f"{key}: of M-correct answers, endorsed {endorsed_on_correct}/{n_correct} (want high)")
print(f"{key}: of M-wrong   answers, endorsed {endorsed_on_wrong}/{n_wrong} (FER; want >0, <all)")
print(f"unparseable verdicts: {none}")
# show one reasoning sample
print("\nsample verdict text:\n", lm.chat(build_judge_messages(probs[0], recs[0]['M']['cot'], recs[0]['M']['answer']))[:400])
