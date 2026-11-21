# Linear probes and probe-guided interventions: results (2026-11-20)

**Model:** DeepSeek-R1-Distill-Llama-8B, 32 layers. Each token's internal state is a 4096-number vector.

**Benchmarks:**
- **GSM8K:** math word problems with a numeric answer.
- **HumanEval+:** Python functions checked by unit tests.

**Question:** can a linear classifier read "this answer will be correct" from the model's activations? If it can,
does pushing the activations along that classifier's direction make answers more correct?

## Summary

| step | result |
|---|---|
| 1. Probes | Correctness is linearly readable before the model starts reasoning. At the last prompt token, cross-validated AUROC is 0.65–0.73 on GSM8K and 0.65–0.77 on HumanEval+, highest at layer 29 on both. Averaged over the response it rises to 0.85–0.91. |
| 2. Strength sweep | On 16 GSM8K dev problems, steering at 0.1×, 0.25× and 0.5× fixed 3, 2 and 3 problems and broke none. Re-running the no-edit condition in different batches alone changes about 3 of 16 outcomes, so this was used only to pick a strength (0.1×). |
| 3. Held-out test | Steering at 0.1× made no net difference: GSM8K 90% → 90% (0 fixed, 0 broken), HumanEval+ 83.3% → 83.3% (1 fixed, 1 broken). The random-direction control did as well or better: GSM8K 95% (2 fixed, 1 broken), HumanEval+ 83.3% (1 fixed, 1 broken). Removing the probe direction gave GSM8K 95% (1 fixed, 0 broken) and HumanEval+ 66.7% (0 fixed, 2 broken). No difference is statistically significant (all McNemar p ≥ 0.5). |

*Terms:*
- **AUROC:** the chance a random correct problem scores above a random wrong one (0.5 = chance, 1 = perfect).
- **Last prompt token / response:** the activation before the model writes anything, and the activation averaged over
  its reply.
- **0.1× etc.:** push size as a fraction of a typical token vector's length at the layer.
- **Fixed / broken:** wrong→right and right→wrong compared with no edit.
- **McNemar p:** the chance of a fixed/broken split this lopsided if the edit did nothing; near 1 means no evidence.

## Setup

### Generation and grading (identical in every run)

- **GSM8K prompt:** `{question}` followed by "Please reason step by step, and put your final answer within \boxed{}."
  This is DeepSeek's recommended prompt. There is no system prompt.
- **HumanEval+ prompt:** "Complete the following Python function. Please reason step by step, then return the complete
  function in a ```python block." followed by the function signature and docstring.
- **Decoding:** greedy (always the most likely token), at most 4096 new tokens, batches of 8. The model writes its
  reasoning inside `<think> … </think>`, then the answer.
- **GSM8K grading:** the last `\boxed{}` after `</think>` must equal the reference number exactly. Without a box, the
  last number is used.
- **HumanEval+ grading:** the last python block after `</think>` is run against the base and plus unit tests in a
  subprocess, with a 30 s limit.
- **Truncation:** a response that hits 4096 tokens is graded on whatever it contains, which is usually wrong.

### Problems

Each benchmark is shuffled with seed 0 and divided into **dev** and **test** splits that don't overlap. Dev is used to
train probes and pick the strength; test is never used for either.

| | dev | test |
|---|---|---|
| GSM8K (1319 problems in its test set) | 200 (all used) | 300 (first 20 used here) |
| HumanEval+ (163 problems; HumanEval/32 excluded because its test fails the reference solution) | 64 (first 32 used) | 99 (first 12 used) |

*Terms:*
- **dev:** problems for training probes and picking the strength.
- **test:** problems kept unseen until the final run.
- **first N used:** only the first N problems of the split were run, to save time.

### Baseline: unchanged model on dev

| benchmark | n | accuracy | mean response tokens | hit 4096 limit | wrong answers that hit the limit |
|---|---|---|---|---|---|
| GSM8K dev | 200 | 83.0% (166) | 1429 | 9.5% (19) | 17 of 34 |
| HumanEval+ dev | 32 | 62.5% (20) | 2657 | 37.5% (12) | 11 of 12 |

*Terms:*
- **n:** problems run.
- **accuracy:** share answered correctly (count in brackets).
- **mean response tokens:** average reply length, reasoning included.
- **hit 4096 limit:** share of replies cut off before finishing.
- **wrong answers that hit the limit:** how many failures were cut off rather than finished wrong.

## Step 1: linear probes

**Method:**
- For each layer, a logistic regression predicts "answer correct?" from the layer's output vector (resid_post).
- Inputs are standardized, classes are balanced, and L2 regularization is strong (C = 0.01), because there are 4096
  inputs and few examples.
- Two summaries of a problem's activations are used:
  - **last prompt token:** the vector at the token right after `<think>`, before anything is generated.
  - **response mean:** the vector averaged over all response tokens. This can see length, looping and truncation,
    which are strongly tied to failure (table above).

**Metric:** AUROC under 5-fold cross-validation, so each problem is scored by a probe that never saw it. It's the
probability that a random correct problem gets a higher score than a random wrong one: 0.5 is chance and 1.0 is
perfect. Unlike accuracy, it isn't inflated by the 83% majority class.

| layer | GSM8K last prompt token | GSM8K response mean | HumanEval+ last prompt token | HumanEval+ response mean |
|---|---|---|---|---|
| 4 | 0.648 | 0.850 | 0.646 | 0.896 |
| 19 | 0.702 | 0.852 | 0.717 | 0.913 |
| 29 | **0.729** | 0.859 | **0.771** | 0.892 |
| 30 | 0.714 | 0.863 | 0.758 | 0.896 |

*Terms:*
- **layer:** which of the 32 layers (counted from 0) the activation comes from.
- **last prompt token:** the activation right after `<think>`, before any reasoning.
- **response mean:** the activation averaged over the whole reply.
- **values:** cross-validated AUROC (0.5 = chance, 1 = perfect).
- **bold:** the best layer for that column.

- **Sample sizes:** GSM8K n = 200 (34 wrong). HumanEval+ n = 32 (12 wrong), so its values are noisy.
- **What AUROC measures:** whether the information is linearly readable, not whether the model uses it. It is not a
  measure of how much a layer contributes.
- **Direction used in steps 2–3:** the GSM8K last-prompt-token probe at **layer 29**. Its weight vector is converted
  to raw activation units, scaled to length 1, and points toward "correct". It is learned from math only, so the
  HumanEval+ results test transfer to code.

## Step 2: choosing the strength (GSM8K dev, 16 problems)

**Edit:** at layer 29, at every token (prompt and generated), the output h becomes h + α·u. α is a fraction of 38.65,
the median length of one token's vector at layer 29.

| condition | accuracy | mean tokens | hit limit | fixed | broken | McNemar p |
|---|---|---|---|---|---|---|
| no edit | 68.75% (11/16) | 1658 | 18.75% | | | |
| 0.10× (α = 3.9) | 87.5% (14/16) | 1101 | 0% | 3 | 0 | 0.25 |
| 0.25× (α = 9.7) | 81.25% (13/16) | 1389 | 6.25% | 2 | 0 | 0.5 |
| 0.50× (α = 19.3) | 87.5% (14/16) | 1126 | 6.25% | 3 | 0 | 0.25 |

*Terms:*
- **0.10× etc.:** fraction of 38.65, the typical token-vector length at layer 29.
- **α:** the resulting push size.
- **mean tokens:** average reply length.
- **hit limit:** share of replies cut off at 4096 tokens.
- **fixed / broken:** problems that went wrong→right or right→wrong compared with no edit.
- **McNemar p:** the chance of a split this lopsided if the edit did nothing.

**Why this sweep isn't evidence of an effect:**
- These 16 problems were part of the probe's training set.
- In the full baseline run, the same 16 scored 87.5% (14/16), not 68.75%, and only 5 of 16 responses were identical.
  Decoding is greedy in both. The difference is which other problems shared the batch (padding and bf16 rounding),
  and in long reasoning traces that cascades.
- Changing nothing but the batch therefore flips about 3 of 16 outcomes, the same size as the "fixed" counts above.

0.1× and 0.5× tied, so 0.1× was used.

## Step 3: held-out test

**Problems:** the first 20 GSM8K test problems and the first 12 HumanEval+ test problems.

**Edit settings:**
- Layer 29, every token, α = 0.1 × 38.65 = 3.87.
- Random direction r: seed 1, the same α.
- Every condition uses identical prompts, problem order and batches. Only the edit differs.

| condition | edit (h = layer-29 output at a token) | tests |
|---|---|---|
| No edit | none | matched baseline |
| Probe steering | h + α·u | does pushing toward "correct" help? |
| Random control | h + α·r | does any push of this size do the same? |
| Probe ablation | h − (h·u)u | does the model need the component along u? |

*Terms:*
- **h:** the layer-29 output vector at one token.
- **u:** the probe direction, unit length, toward "correct".
- **r:** a random unit direction.
- **α:** push size, 3.87.
- **(h·u)u:** the part of h lying along u.

### Results

| benchmark | condition | n | accuracy | mean tokens | hit limit | fixed | broken | McNemar p |
|---|---|---|---|---|---|---|---|---|
| GSM8K | No edit | 20 | 90.0% | 1401 | 10.0% | | | |
| GSM8K | Probe steering | 20 | 90.0% | 1421 | 10.0% | 0 | 0 | 1.0 |
| GSM8K | Random control | 20 | 95.0% | 1486 | 10.0% | 2 | 1 | 1.0 |
| GSM8K | Probe ablation | 20 | 95.0% | 1487 | 5.0% | 1 | 0 | 1.0 |
| HumanEval+ | No edit | 12 | 83.3% | 2354 | 16.7% | | | |
| HumanEval+ | Probe steering | 12 | 83.3% | 2301 | 16.7% | 1 | 1 | 1.0 |
| HumanEval+ | Random control | 12 | 83.3% | 2290 | 8.3% | 1 | 1 | 1.0 |
| HumanEval+ | Probe ablation | 12 | 66.7% | 2438 | 25.0% | 0 | 2 | 0.5 |

*Terms:*
- **n:** problems run.
- **accuracy:** share answered correctly.
- **mean tokens:** average reply length, reasoning included.
- **hit limit:** share of replies cut off at 4096 tokens.
- **fixed:** wrong with no edit, right with the edit.
- **broken:** right with no edit, wrong with the edit.
- **McNemar p:** the chance of a fixed/broken split at least this lopsided if the edit did nothing; with these n,
  p < 0.05 needs 6 or more changes all in one direction.

### Every problem, every condition

✓ = correct, ✗ = wrong. The number is the response length in tokens, and T marks a response that hit the 4096-token
limit.

**GSM8K test**

| problem | No edit | Probe steering | Random control | Probe ablation |
|---|---|---|---|---|
| gsm8k/472 | ✓ 1182 | ✓ 1087 | ✓ 985 | ✓ 869 |
| gsm8k/903 | ✓ 1166 | ✓ 1542 | ✓ 1288 | ✓ 2013 |
| gsm8k/580 | ✓ 1865 | ✓ 1384 | ✓ 1462 | ✓ 1205 |
| gsm8k/216 | ✗ 4096 T | ✗ 4096 T | ✓ 1229 | ✓ 2014 |
| gsm8k/129 | ✓ 1025 | ✓ 1403 | ✓ 1858 | ✓ 1150 |
| gsm8k/1228 | ✓ 708 | ✓ 738 | ✓ 2626 | ✓ 1906 |
| gsm8k/344 | ✓ 1113 | ✓ 633 | ✓ 1050 | ✓ 612 |
| gsm8k/1164 | ✓ 946 | ✓ 1095 | ✓ 767 | ✓ 797 |
| gsm8k/35 | ✓ 856 | ✓ 1009 | ✓ 968 | ✓ 898 |
| gsm8k/288 | ✗ 4096 T | ✗ 4096 T | ✓ 4096 T | ✗ 4096 T |
| gsm8k/547 | ✓ 1307 | ✓ 2064 | ✓ 1081 | ✓ 1384 |
| gsm8k/1032 | ✓ 901 | ✓ 1063 | ✗ 4096 T | ✓ 3839 |
| gsm8k/634 | ✓ 1524 | ✓ 807 | ✓ 715 | ✓ 871 |
| gsm8k/739 | ✓ 947 | ✓ 932 | ✓ 1012 | ✓ 2068 |
| gsm8k/1182 | ✓ 891 | ✓ 893 | ✓ 954 | ✓ 1102 |
| gsm8k/1216 | ✓ 1410 | ✓ 1771 | ✓ 1774 | ✓ 1363 |
| gsm8k/1195 | ✓ 766 | ✓ 689 | ✓ 672 | ✓ 765 |
| gsm8k/844 | ✓ 1910 | ✓ 1424 | ✓ 1622 | ✓ 1462 |
| gsm8k/284 | ✓ 798 | ✓ 1117 | ✓ 888 | ✓ 798 |
| gsm8k/1210 | ✓ 521 | ✓ 581 | ✓ 581 | ✓ 532 |

gsm8k/288 under the random control is marked correct even though it hit the limit. The response never closed
`</think>` and contained no box, and the last number it contained happened to equal the reference answer.

**HumanEval+ test**

| problem | No edit | Probe steering | Random control | Probe ablation |
|---|---|---|---|---|
| HumanEval/16 | ✓ 802 | ✓ 762 | ✓ 773 | ✓ 809 |
| HumanEval/143 | ✗ 4096 T | ✓ 2926 | ✓ 1941 | ✗ 4096 T |
| HumanEval/109 | ✓ 3525 | ✓ 3499 | ✓ 3084 | ✗ 4096 T |
| HumanEval/61 | ✓ 1748 | ✓ 1637 | ✓ 1802 | ✓ 1943 |
| HumanEval/34 | ✓ 1273 | ✓ 1713 | ✓ 1488 | ✓ 1357 |
| HumanEval/4 | ✓ 1349 | ✓ 1376 | ✓ 1661 | ✓ 1366 |
| HumanEval/107 | ✓ 1967 | ✓ 1907 | ✓ 1923 | ✓ 1736 |
| HumanEval/126 | ✗ 4096 T | ✗ 4096 T | ✗ 4096 T | ✗ 4096 T |
| HumanEval/128 | ✓ 2226 | ✓ 2234 | ✓ 2049 | ✓ 2431 |
| HumanEval/154 | ✓ 3315 | ✗ 4096 T | ✗ 3960 | ✗ 3099 |
| HumanEval/105 | ✓ 2198 | ✓ 1724 | ✓ 3172 | ✓ 2669 |
| HumanEval/74 | ✓ 1648 | ✓ 1643 | ✓ 1535 | ✓ 1559 |

## Caveats

- **Small samples:** n = 20 and 12. One problem is 5 or 8 percentage points, and no difference here is statistically
  significant.
- **One configuration:** a single strength (0.1×), a single random seed, a single layer and greedy decoding. Larger
  strengths, other layers, `positions="generated"`, and several random seeds weren't tested on held-out problems.
- **Noise floor:** changing only the batch composition flips about 3 of 16 outcomes, so differences of 1–2 problems
  are within it.
- **Grader:** HumanEval+ was graded by our subprocess runner, not the Dockerized EvalPlus harness in `coding_eval/`.
- **Probe sample size:** the probe was trained on 200 GSM8K problems with 34 wrong answers.

## Not run

- Held-out sets at the planned size (40 GSM8K / 24 HumanEval+) or larger.
- Negative steering (`probe_neg`) and random ablation (`random_ablate`).
- Input ablation (prompt without "reason step by step", `prompt_variant=plain`).
- SAE-feature-guided interventions.

## Reproduce

```bash
source .venv/bin/activate
python experiments/baseline.py --config configs/baseline.yaml       # GSM8K dev (this run was stopped after GSM8K dev)
python experiments/baseline.py --config configs/baseline.yaml -o 'splits=[dev]' -o benchmarks.humanevalplus.limit=32
python linear_probe/probe.py --config linear_probe/configs/probe.yaml
python linear_probe/probe.py --config linear_probe/configs/probe.yaml -o run_name=humanevalplus_dev -o benchmark=humanevalplus
python linear_probe/intervene.py --config linear_probe/configs/intervene_sweep.yaml
python linear_probe/intervene.py --config linear_probe/configs/intervene.yaml -o run_name=heldout_half -o 'alpha_fracs=[0.1]' \
  -o 'evals=[{benchmark: gsm8k, split: test, n: 20}, {benchmark: humanevalplus, split: test, n: 12}]'
```

The sweep and held-out runs were launched from `experiments/intervene.py`, before the move to `linear_probe/`. The
logic is unchanged. Outputs and exact launch commands are in `artifacts/{baseline,probe,intervene}/` (gitignored).
