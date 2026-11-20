# Reasoning interventions: status 2026-11-20

DeepSeek-R1-Distill-Llama-8B, greedy, max 4096 new tokens, single seed. Outputs in `artifacts/` (gitignored).

## Code

| | file |
|---|---|
| Model, activation extraction (layers 4/19/29/30), intervention hooks, GSM8K/HumanEval+ grading | `src/reasoning_env.py` |
| Baseline | `experiments/baseline.py`, `configs/baseline.yaml` |
| Linear probes | `experiments/probe.py`, `configs/probe.yaml` |
| Interventions | `experiments/intervene.py`, `configs/intervene*.yaml` |
| HumanEval+ Docker evaluator (Jacob) | `coding_eval/` |

HumanEval/32 is excluded because its HF test fails the canonical solution.

## Baseline (dev)

| benchmark | n | accuracy | mean tokens | truncated |
|---|---|---|---|---|
| GSM8K | 200 | 83.0% | 1429 | 9.5% |
| HumanEval+ | 32 | 62.5% | 2657 | 37.5% |

## Linear probes (dev, 5-fold CV AUROC)

Logistic regression predicting answer correctness from the residual stream. `prompt_last` is the last prompt token
(before reasoning); `resp_mean` is the mean over response tokens.

| layer | GSM8K prompt_last | GSM8K resp_mean | HumanEval+ prompt_last | HumanEval+ resp_mean |
|---|---|---|---|---|
| 4 | 0.648 | 0.850 | 0.646 | 0.896 |
| 19 | 0.702 | 0.852 | 0.717 | 0.913 |
| 29 | **0.729** | 0.859 | **0.771** | 0.892 |
| 30 | 0.714 | 0.863 | 0.758 | 0.896 |

The GSM8K probes have n = 200 (34 wrong) and the HumanEval+ probes n = 32 (12 wrong). The selected layer is 29,
using the GSM8K prompt_last probe direction.

## Steering sweep (GSM8K dev, layer 29, n = 16)

The edit is resid += α·u at all positions, with α = fraction × 38.65 (the median token norm at layer 29).

| condition | accuracy | mean tokens | truncated | fixed / broken | McNemar p |
|---|---|---|---|---|---|
| no edit | 68.75% | 1658 | 18.75% | n/a | n/a |
| 0.10× | 87.5% | 1101 | 0% | 3 / 0 | 0.25 |
| 0.25× | 81.25% | 1389 | 6.25% | 2 / 0 | 0.5 |
| 0.50× | 87.5% | 1126 | 6.25% | 3 / 0 | 0.25 |

The sweep problems are in the probe's training set. The same 16 problems scored 87.5% in the baseline run versus
68.75% for no edit here, with only 5 of 16 responses identical. Batch composition changes greedy outputs by about
3 of 16 problems.

## Not run yet

- Held-out interventions on 40 GSM8K and 24 HumanEval+ test problems: no edit, probe, random, ablate (~3.5 h).
  Needs a choice of α and of HumanEval+ split (ours or `coding_eval`'s).
- HumanEval+ dev problems 33–64, and the test-split baselines.
- Input ablation (`prompt_variant=plain`).
- SAE comparison.
- Export to the `coding_eval` format.

## Reproduce

```bash
python experiments/baseline.py --config configs/baseline.yaml -o 'splits=[dev]' -o benchmarks.humanevalplus.limit=32
python experiments/probe.py --config configs/probe.yaml
python experiments/probe.py --config configs/probe.yaml -o run_name=humanevalplus_dev -o benchmark=humanevalplus
python experiments/intervene.py --config configs/intervene_sweep.yaml
```
