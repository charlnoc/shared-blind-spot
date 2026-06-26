# Phase 2 — Step 1A: difficulty / capability search

Held-out seeds 80000..80019 (disjoint from main run 70000.. and probe 90000..). Decoding greedy. Each cell = standalone solve accuracy.

| level | params | gpt-4o-mini | gpt-4o | claude-haiku |
|---|---|---|---|---|
| L4b | steps=16,add≤999,mul≤40 | 0% | 55% | 80% |
| L4c | steps=16,add≤999,mul≤55 | 0% | 45% | 65% |
| L4d | steps=16,add≤999,mul≤70 | 0% | 35% | 40% |

_answerer should be errorful (≈30–60%); pick the level + cross tier where gpt-4o and the Claude judge are closest and both sit in ≈50–80%._

