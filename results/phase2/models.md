# Phase 2 — Step 0: model roles & CoT availability

Probe date: 2026-06-26. Decoding: temperature 0 (greedy). Domain: templated multi-step arithmetic (same generator as the local run). A model is judge-eligible only if it emits a full, parseable `a op b = c` / `Answer: N` chain.

| key | provider | model id | usable CoT? | probe acc | aligned |
|---|---|---|---|---|---|
| gpt-4o-mini | openai | `gpt-4o-mini-2024-07-18` | yes | 3/3 | 3/3 |
| gpt-4o | openai | `gpt-4o-2024-08-06` | yes | 3/3 | 3/3 |
| gpt-4.1-mini | openai | `gpt-4.1-mini-2025-04-14` | yes | 3/3 | 3/3 |
| claude-haiku | anthropic | `claude-haiku-4-5-20251001` | yes | 3/3 | 3/3 |
| claude-sonnet | anthropic | `claude-sonnet-4-5-20250929` | yes | 3/3 | 3/3 |

## Sample outputs

### gpt-4o-mini (`gpt-4o-mini-2024-07-18`)

Problem seed 90000 (gt=74):

```
Step 1: 17 - 1 = 16  
Step 2: 16 + 4 = 20  
Step 3: 20 + 31 = 51  
Step 4: 51 + 20 = 71  
Step 5: 71 - 40 = 31  
Step 6: 31 + 43 = 74  
Answer: 74
```

### gpt-4o (`gpt-4o-2024-08-06`)

Problem seed 90000 (gt=74):

```
Step 1: 17 - 1 = 16  
Step 2: 16 + 4 = 20  
Step 3: 20 + 31 = 51  
Step 4: 51 + 20 = 71  
Step 5: 71 - 40 = 31  
Step 6: 31 + 43 = 74  
Answer: 74
```

### gpt-4.1-mini (`gpt-4.1-mini-2025-04-14`)

Problem seed 90000 (gt=74):

```
Step 1: 17 - 1 = 16  
Step 2: 16 + 4 = 20  
Step 3: 20 + 31 = 51  
Step 4: 51 + 20 = 71  
Step 5: 71 - 40 = 31  
Step 6: 31 + 43 = 74  
Answer: 74
```

### claude-haiku (`claude-haiku-4-5-20251001`)

Problem seed 90000 (gt=74):

```
Step 1: 17 - 1 = 16
Step 2: 16 + 4 = 20
Step 3: 20 + 31 = 51
Step 4: 51 + 20 = 71
Step 5: 71 - 40 = 31
Step 6: 31 + 43 = 74
Answer: 74
```

### claude-sonnet (`claude-sonnet-4-5-20250929`)

Problem seed 90000 (gt=74):

```
Step 1: 17 - 1 = 16
Step 2: 16 + 4 = 20
Step 3: 20 + 31 = 51
Step 4: 51 + 20 = 71
Step 5: 71 - 40 = 31
Step 6: 31 + 43 = 74
Answer: 74
```

