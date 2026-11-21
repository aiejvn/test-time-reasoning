# Linear probes and probe-guided interventions: results (2026-11-20)

> **Status (2026-11-21): this probe is not the project's target, and the results below can't support claims.**
> Two independent audits reviewed this work. Their findings and what was done about each are in
> [Audit response](#audit-response-2026-11-21) at the end. In short: the probe predicts *final-answer
> correctness*, not *genuine self-correction*. The direction is reproducible. Decoding depends on batching, and the
> grader had a leak.

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

These runs used `gen.batch_size: 8`, the default before 2026-11-21. Pass `-o gen.batch_size=8` to `baseline.py`
to reproduce them; `intervene.py` reads the batch size from `configs/baseline.yaml`, so edit that file back to 8
for those runs.

```bash
source .venv/bin/activate
python experiments/baseline.py --config configs/baseline.yaml -o gen.batch_size=8  # GSM8K dev (stopped after GSM8K dev)
python experiments/baseline.py --config configs/baseline.yaml -o 'splits=[dev]' -o benchmarks.humanevalplus.limit=32
python linear_probe/probe.py --config linear_probe/configs/probe.yaml
python linear_probe/probe.py --config linear_probe/configs/probe.yaml -o run_name=humanevalplus_dev -o benchmark=humanevalplus
python linear_probe/intervene.py --config linear_probe/configs/intervene_sweep.yaml
python linear_probe/intervene.py --config linear_probe/configs/intervene.yaml -o run_name=heldout_half -o 'alpha_fracs=[0.1]' \
  -o 'evals=[{benchmark: gsm8k, split: test, n: 20}, {benchmark: humanevalplus, split: test, n: 12}]'
```

The sweep and held-out runs were launched from `experiments/intervene.py`, before the move to `linear_probe/`. The
logic is unchanged. Outputs and exact launch commands are in `artifacts/{baseline,probe,intervene}/` (gitignored).

## Audit response (2026-11-21)

Two audits reviewed steps 1–3. This section lists each finding, what was checked, and what changed.

| finding | status | evidence / change |
|---|---|---|
| 1. Probe trains on the wrong label | **Confirmed. Not fixed yet: labels exist only as a first pass** | See [1](#1-wrong-probe-target) |
| 2. Probe direction can't be reproduced | **Resolved: it reproduces exactly from committed files** | See [2](#2-reproducibility-of-the-direction) |
| 3a. Greedy decoding depends on batching | **Confirmed** | See [3a](#3a-batch-dependent-decoding) |
| 3b. Grader scores unfinished reasoning | **Confirmed and fixed** | See [3b](#3b-grading-leak) |
| 3c. Coding evaluation | **Confirmed: two incompatible HumanEval+ splits; not graded with EvalPlus** | See [3c](#3c-coding-evaluation) |
| 4. No independent held-out probe test | **Confirmed. Not run** | See [4](#4-held-out-probe-test) |
| 5. 33 HumanEval+ baseline records | **The real number is 32** | See [5](#5-humaneval-record-count) |

### 1. Wrong probe target

`probe.py` labels each problem by whether its *final answer* was correct. The proposal's target is the
representation behind *recovering from an error* (PROPOSAL.md RQ1, RQ4), and the two labels are not the same thing.
A trace can be correct with no correction at all. A correction can also happen inside a trace that still ends wrong.
Choosing a different layer or strength doesn't fix this.

**Done:** [`self_correction_events.py`](self_correction_events.py) builds event-level labels from the saved traces.
- **Event:** a reflection marker ("Wait", "Hmm", "But wait", "Let me check", …) that starts a sentence inside
  `<think>`.
- **Label:** compare the answer the model stated before the marker with the next answer it states after the marker.
  A wrong answer replaced by the right one is a `genuine_correction`. The same answer kept is a `verify_kept_*`
  event. The full list of labels is in the script's docstring.
- **Position:** the token immediately before the marker, so the correction word isn't yet in the context. That
  position still predicts the marker as the next token. The intended contrast is therefore genuine correction against
  verification (both are followed by a marker), not event against no event.

**Result on the 200 GSM8K dev traces:**
- 1,651 events: Wait 984, Hmm 476, But wait 128, Let me check 30, Let me double-check 24, Let me verify 6,
  Hold on 2, Actually 1.
- Only 36 events could be labelled, and none of them is a genuine correction: `verify_kept_right` 24 and
  `verify_kept_wrong` 12.
- The other 1,615 are `unlabeled`. The model rarely restates an answer mid-reasoning, so an answer is missing on
  one side of the event.
- A looser version that counts phrases like "total of X" as answers finds wrong→right changes in 28 traces. Those
  "answers" are often intermediate values, though.
- Keyword matching doesn't work either. "Mistake" phrases mostly appear in "to make sure I didn't make a mistake" during
  re-checks, and "But wait, that's not all" continues the problem rather than reflecting on it.

**Not done:** no self-correction probe was trained. There are no usable positive labels, and training on these would
repeat the original problem. Two things are needed first:
- **A step-level judge.** For each event it must decide whether the step just before the marker contained an
  error and whether the step after it fixed that error. The options are an LLM judge or human annotation. This is a
  team decision: which judge, and who checks its accuracy on a hand-labelled sample.
- **More traces.** Using the remaining GSM8K problems, or several samples per problem, would yield enough positives.

### 2. Reproducibility of the direction

The direction is reproducible. Two checks were run on 2026-11-21:
- **From saved activations:** re-running `probe.py` on the saved activations gives every direction and every
  cross-validated AUROC bit-for-bit (largest difference 0.0).
- **From committed files only:** [`reproduce_acts.py`](reproduce_acts.py) re-tokenized the committed responses in
  `output/baseline/baseline/gsm8k_dev.jsonl` and re-extracted the activations. It then retrained the probe.
  - All 200 responses re-tokenized to their logged lengths.
  - `prompt_last` activations were identical, and the layer-29 direction matched with cosine 0.99999994.
  - `resp_mean` activations differed by at most 0.11, and those directions matched with cosine above 0.99998.

Command:

```bash
python linear_probe/reproduce_acts.py --responses output/baseline/baseline/gsm8k_dev.jsonl \
  --saved-acts artifacts/baseline/baseline/gsm8k_dev_acts.pt --saved-probe artifacts/probe/gsm8k_dev \
  --out artifacts/probe/gsm8k_dev_rebuilt
```

SHA-256 of the files the direction came from:

| file | SHA-256 |
|---|---|
| `output/baseline/baseline/gsm8k_dev.jsonl` (committed; identical to the `artifacts/` copy) | `b7573a0e11ba463e9b67ecfb4c668cd374a9422c6d1dc38ccc9f95f6ad2997cb` |
| `artifacts/baseline/baseline/gsm8k_dev_acts.pt` | `740da5d6c5d180fb4512efc4c4ed5e288d65ecfe32990f3387f0cce625e0516b` |
| `artifacts/probe/gsm8k_dev/directions.pt` | `37d22d3dfc1c9d0e5cc196c8c9d991fac23c6e637cc55344fe90708dd499bf2c` |

The `.pt` files are still gitignored, per AGENTS.md, and not yet on W&B or the HF Hub.

**Caveat:** the direction is reproducible, but it was trained with 2 wrong labels (see 3b).

### 3a. Batch-dependent decoding

[`experiments/check_determinism.py`](../experiments/check_determinism.py) checks this directly. It generates the
first 8 GSM8K dev problems three ways, each with greedy decoding and at most 1,024 new tokens:
- all 8 in one batch;
- one problem per batch;
- one problem per batch, a second time.

**Result:**
- **One problem per batch is deterministic:** all 8 responses were token-for-token identical across the two runs.
- **Batching changes the output:** all 8 batched responses differed from their single-problem versions. They first
  diverged after 61–576 tokens (61, 75, 77, 110, 186, 225, 539, 576).

The batched input is left-padded and computed in bf16, so it is numerically different. Greedy decoding then amplifies
that difference.

**Changed:** `configs/baseline.yaml` now sets `gen.batch_size: 1`. `intervene.py` copies this setting, so future
baseline and intervention runs are batch-invariant, at the cost of throughput. The batch-of-8 results were not
rerun. A trustworthy baseline still needs a batch-size-1 run (on dev and test), plus a repeat run to confirm it is
identical.

Every number in steps 2–3 comes from batch-of-8 runs, including the "no edit" baseline in each comparison. Within a
single run all conditions share the same batches, so the comparisons are matched. Outcomes still change by about 3 of
16 problems when only batch-mates change. That is the same size as every effect reported above.

### 3b. Grading leak

The GSM8K grader fell back to the last number in the text when no `\boxed{}` was present. It did this even when the
reasoning never closed (`</think>` missing, usually a truncated response). Three responses were marked correct that
way:
- `gsm8k/711` and `gsm8k/881` in the baseline. These are 2 of the probe's 166 "correct" training labels.
- `gsm8k/288` under the random control.

HumanEval+ had the same weakness, since it would run code drafted inside unfinished reasoning. No HumanEval+ result
was affected.

**Fixed:** in `src/reasoning_env.py`, both graders now mark a response with no `</think>` as wrong. The saved
results were not regraded or rerun. With the fix applied:
- GSM8K dev baseline accuracy would be 82.0%, not 83.0%.
- The random control would score GSM8K test 90%, with 1 fixed and 1 broken.

### 3c. Coding evaluation

- **Two HumanEval+ splits.** `src/reasoning_env.py` draws dev/test with seed 0 (64/99).
  `coding_eval/data/splits.json` draws development/held-out with seed 2026 (20/40). The probe was trained on our 32
  dev problems, and 7 of them are in `coding_eval`'s held-out set. A result can't be "held out" under both splits
  at once. One split must be made canonical; that is a team decision and isn't changed here.
- **Not graded with EvalPlus.** HumanEval+ was graded by our in-process subprocess runner. It was not graded by the
  Dockerized EvalPlus harness in `coding_eval/`, which needs an exporter to the manifest/samples format.
- **Sample size.** 12 test problems.

### 4. Held-out probe test

The probe was never scored on problems outside its training set; its only score is 5-fold cross-validation on dev.
This check wasn't run. A held-out test of a probe with the wrong target adds little. It is better built into the
self-correction probe once that has labels.

### 5. HumanEval+ record count

The baseline has **32** HumanEval+ dev records: `output/baseline/baseline/humanevalplus_dev.jsonl`,
`output/inputs/humanevalplus_dev.jsonl` and `summary.json` (n = 32) all agree. No file in the repo claims 33.
