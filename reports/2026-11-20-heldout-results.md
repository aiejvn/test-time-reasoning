# Held-out intervention results: `heldout_half`

**Status:** in progress: 7 of 8 condition runs finished. This file is regenerated from the run's outputs every 10 minutes.

## What this run tests

Whether pushing DeepSeek-R1-Distill-Llama-8B's internal activations along a direction that *predicts* correct answers also *causes* more correct answers, on problems never used to build or tune that direction, and whether a direction learned from math (GSM8K) does anything for code (HumanEval+).

## Setup

- **Layer edited:** 29 (zero-based, of 32). It had the highest probe AUROC at the last prompt token on GSM8K dev (0.729).
- **Direction u:** the weight vector of a logistic-regression probe trained on 200 GSM8K dev problems to predict answer correctness from the layer's activation at the last prompt token. It is mapped back to raw activation space, normalized to length 1, and points toward "correct". No SAE features are involved.
- **Strength α:** 0.1 × 38.65 = 3.86. 38.65 is the median length of a single token's activation vector at layer 29, so α is 10% of a typical token's activation. Chosen on 16 GSM8K dev problems (sweep: 0.1×, 0.25×, 0.5×); 0.1× and 0.5× tied, so the smaller was used.
- **Where the edit applies:** all token positions (the prompt and every generated token).
- **Problems:** the first 20 problems of our gsm8k test split; the first 12 problems of our humanevalplus test split. The test splits are disjoint from the dev problems used for the probe and the strength sweep.
- **Decoding:** greedy, at most 4096 new tokens, batch size 8, prompt variant `cot` ("Please reason step by step…"). All conditions use the same prompts, problem order, and batches; only the edit differs.
- **Grading:** GSM8K takes the last `\boxed{}` answer after `</think>` and requires an exact numeric match. HumanEval+ takes the last python code block after `</think>` and runs it against the HumanEval base and plus tests with a 30 s timeout. A response cut off at the token limit counts as wrong.
- **Command:** `python experiments/intervene.py --config configs/intervene.yaml -o run_name=heldout_half -o 'alpha_fracs=[0.1]' -o 'evals=[{benchmark: gsm8k, split: test, n: 20}, {benchmark: humanevalplus, split: test, n: 12}]'`

## Conditions

| condition | name | what it does |
|---|---|---|
| `none` | No edit | The unmodified model, re-run here on the same problems with the same batching. Every other row is compared against this one. |
| `probe_pos` | Probe steering | Adds α·u to the layer-29 output at every token, where u is the unit-length weight vector of the GSM8K linear probe (pointing toward "will answer correctly"). |
| `random_pos` | Random-direction control | Adds α·r at layer 29, where r is a random unit vector (seed 1) with the same magnitude as the probe edit. Tests whether any push of this size changes results. |
| `probe_ablate` | Probe ablation | Removes the component of the layer-29 output along u (h ← h − (h·u)u) at every token, without adding anything. Tests whether the model relies on that direction. |

## How to read the results table

- **accuracy:** fraction of the n problems answered correctly.
- **mean tokens:** average response length in tokens, including the `<think>` reasoning. A change here can explain an accuracy change (e.g. fewer responses hitting the limit).
- **truncated:** fraction of responses that hit the token limit before finishing.
- **fixed / broken:** compared problem by problem with the no-edit row. *Fixed* = wrong without the edit, right with it. *Broken* = right without, wrong with. Accuracy change = (fixed − broken) / n.
- **McNemar p:** two-sided exact test on the fixed and broken counts (the problems that changed). It asks how likely a split at least this lopsided would be if changes in either direction were equally likely. Values near 1 mean no evidence of a directional effect. With n this small, p < 0.05 needs roughly 6 or more changes, all in one direction.
- **Noise floor:** in an earlier dev check, re-running the same 16 problems with different batch composition changed the outcome on 3 of 16, even though decoding is greedy. Within this run batches are identical across conditions, but small edits can still flip borderline problems. The random-direction row is the reference for that.

## Results

| benchmark | condition | n | accuracy | mean tokens | truncated | fixed / broken | McNemar p |
|---|---|---|---|---|---|---|---|
| gsm8k | No edit | 20 | 90.0% | 1401 | 10.0% | n/a | n/a |
| gsm8k | Probe ablation | 20 | 95.0% | 1487 | 5.0% | 1 / 0 | 1 |
| gsm8k | Probe steering | 20 | 90.0% | 1421 | 10.0% | 0 / 0 | 1 |
| gsm8k | Random-direction control | 20 | 95.0% | 1486 | 10.0% | 2 / 1 | 1 |
| humanevalplus | No edit | 12 | 83.3% | 2354 | 16.7% | n/a | n/a |
| humanevalplus | Probe ablation | 12 | 66.7% | 2438 | 25.0% | 0 / 2 | 0.5 |
| humanevalplus | Probe steering | 12 | 83.3% | 2301 | 16.7% | 1 / 1 | 1 |

## Per-problem outcomes: gsm8k test

✓ = correct, ✗ = wrong. The number is the response length in tokens; T = hit the token limit. A row where conditions disagree is a problem the edit flipped.

| problem | No edit | Probe ablation | Probe steering | Random-direction control |
|---|---|---|---|---|
| gsm8k/472 | ✓ 1182 | ✓ 869 | ✓ 1087 | ✓ 985 |
| gsm8k/903 | ✓ 1166 | ✓ 2013 | ✓ 1542 | ✓ 1288 |
| gsm8k/580 | ✓ 1865 | ✓ 1205 | ✓ 1384 | ✓ 1462 |
| gsm8k/216 | ✗ 4096 T | ✓ 2014 | ✗ 4096 T | ✓ 1229 |
| gsm8k/129 | ✓ 1025 | ✓ 1150 | ✓ 1403 | ✓ 1858 |
| gsm8k/1228 | ✓ 708 | ✓ 1906 | ✓ 738 | ✓ 2626 |
| gsm8k/344 | ✓ 1113 | ✓ 612 | ✓ 633 | ✓ 1050 |
| gsm8k/1164 | ✓ 946 | ✓ 797 | ✓ 1095 | ✓ 767 |
| gsm8k/35 | ✓ 856 | ✓ 898 | ✓ 1009 | ✓ 968 |
| gsm8k/288 | ✗ 4096 T | ✗ 4096 T | ✗ 4096 T | ✓ 4096 T |
| gsm8k/547 | ✓ 1307 | ✓ 1384 | ✓ 2064 | ✓ 1081 |
| gsm8k/1032 | ✓ 901 | ✓ 3839 | ✓ 1063 | ✗ 4096 T |
| gsm8k/634 | ✓ 1524 | ✓ 871 | ✓ 807 | ✓ 715 |
| gsm8k/739 | ✓ 947 | ✓ 2068 | ✓ 932 | ✓ 1012 |
| gsm8k/1182 | ✓ 891 | ✓ 1102 | ✓ 893 | ✓ 954 |
| gsm8k/1216 | ✓ 1410 | ✓ 1363 | ✓ 1771 | ✓ 1774 |
| gsm8k/1195 | ✓ 766 | ✓ 765 | ✓ 689 | ✓ 672 |
| gsm8k/844 | ✓ 1910 | ✓ 1462 | ✓ 1424 | ✓ 1622 |
| gsm8k/284 | ✓ 798 | ✓ 798 | ✓ 1117 | ✓ 888 |
| gsm8k/1210 | ✓ 521 | ✓ 532 | ✓ 581 | ✓ 581 |

## Per-problem outcomes: humanevalplus test

✓ = correct, ✗ = wrong. The number is the response length in tokens; T = hit the token limit. A row where conditions disagree is a problem the edit flipped.

| problem | No edit | Probe ablation | Probe steering |
|---|---|---|---|
| HumanEval/16 | ✓ 802 | ✓ 809 | ✓ 762 |
| HumanEval/143 | ✗ 4096 T | ✗ 4096 T | ✓ 2926 |
| HumanEval/109 | ✓ 3525 | ✗ 4096 T | ✓ 3499 |
| HumanEval/61 | ✓ 1748 | ✓ 1943 | ✓ 1637 |
| HumanEval/34 | ✓ 1273 | ✓ 1357 | ✓ 1713 |
| HumanEval/4 | ✓ 1349 | ✓ 1366 | ✓ 1376 |
| HumanEval/107 | ✓ 1967 | ✓ 1736 | ✓ 1907 |
| HumanEval/126 | ✗ 4096 T | ✗ 4096 T | ✗ 4096 T |
| HumanEval/128 | ✓ 2226 | ✓ 2431 | ✓ 2234 |
| HumanEval/154 | ✓ 3315 | ✗ 3099 | ✗ 4096 T |
| HumanEval/105 | ✓ 2198 | ✓ 2669 | ✓ 1724 |
| HumanEval/74 | ✓ 1648 | ✓ 1559 | ✓ 1643 |

## Caveats

- n is small (20 and 12), so differences of one or two problems are within noise. No result here is statistically significant unless its McNemar p says so.
- Single strength, single random seed, single decoding seed (greedy).
- HumanEval+ is graded by our own runner, not the Dockerized EvalPlus harness in `coding_eval/`.
- The probe direction is learned from math only, so the HumanEval+ rows test transfer to code.
