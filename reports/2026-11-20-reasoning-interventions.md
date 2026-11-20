# Test-time reasoning interventions: status report (2026-11-20)

Model: DeepSeek-R1-Distill-Llama-8B (32 layers, d_model 4096, bf16, one A10G 23 GB).
Branch: `demo/11-20`. All numbers below come from runs whose outputs are in `artifacts/` (gitignored).
Every run is single-seed, greedy decoding, max 4096 new tokens.

## What exists

| component | file | status |
|---|---|---|
| Model loading, resid_post extraction at layers 4/19/29/30, intervention hooks | `src/reasoning_env.py` | verified on the real model (below) |
| GSM8K / HumanEval+ loading, seeded dev/test splits, graders | `src/reasoning_env.py` | verified on canonical/incorrect solutions |
| Baseline runner (generation, grading, activation saving) | `experiments/baseline.py`, `configs/baseline.yaml` | run on dev |
| Linear probes per layer | `experiments/probe.py`, `configs/probe.yaml` | run on dev |
| Interventions (steer / ablate / random control, flip counts, McNemar) | `experiments/intervene.py`, `configs/intervene*.yaml` | dev sweep run; held-out run **not started** |
| HumanEval+ Docker/EvalPlus evaluator (Jacob, merged) | `coding_eval/` | merged; not yet wired to our outputs |

Hook and intervention checks on the real model: the change in layer-19 resid_post under `steer(u, 8)` was exactly 8·u
at every position (cosine 1.0; BOS has bf16 rounding). `ablate(u)` cut the residual's projection on u from 10.9 to
0.015. Empty edits gave identical activations, α = 0 gave identical generated tokens, and prompt-token activations from
`extract_activations` match a direct forward pass.

Grader checks: 163 of 164 HumanEval+ canonical solutions pass and deliberately wrong functions fail. HumanEval/32 is
excluded because the HF `evalplus/humanevalplus` test fails even its canonical solution.

## Setup

- Prompts: DeepSeek's recommended `"{question}\nPlease reason step by step, and put your final answer within \boxed{}."`
  for GSM8K, and an equivalent instruction plus a ```` ```python ```` block for HumanEval+. There's no system prompt.
- Splits: a seeded (seed 0) shuffle into disjoint dev/test sets. GSM8K takes 200 dev / 300 test from its 1319-problem
  test set, and HumanEval+ takes 64 dev / 99 test from 163 problems.
- Grading:
  - GSM8K: exact numeric match of the last `\boxed{}` after `</think>`, or the last number there if there's no box.
  - HumanEval+: the last python block after `</think>`, run against the base and plus tests with a 30 s timeout.
  - Truncated responses count as wrong.

## Results so far

### Baseline (unchanged model, dev)

| benchmark | n | accuracy | mean response tokens | truncated (hit 4096) | truncated and still correct |
|---|---|---|---|---|---|
| GSM8K dev | 200 | 83.0% (166) | 1429 | 9.5% (19) | 2 |
| HumanEval+ dev (first 32 of 64) | 32 | 62.5% (20) | 2657 | 37.5% (12) | 1 |

### Linear probes: predict "answer will be correct" from layer activations (dev)

These are logistic regressions (standardized, class-balanced, C = 0.01) on the raw 4096-dimensional residual vector,
scored by 5-fold stratified cross-validated AUROC (0.5 is chance). No SAE features are used. `prompt_last` is the
last prompt token, before any reasoning is generated; `resp_mean` is the mean over response tokens.

| layer | GSM8K `prompt_last` | GSM8K `resp_mean` | HumanEval+ `prompt_last` | HumanEval+ `resp_mean` |
|---|---|---|---|---|
| 4 | 0.648 | 0.850 | 0.646 | 0.896 |
| 19 | 0.702 | 0.852 | 0.717 | 0.913 |
| 29 | **0.729** | 0.859 | **0.771** | 0.892 |
| 30 | 0.714 | 0.863 | 0.758 | 0.896 |

The GSM8K probes use n = 200 (166 correct, 34 wrong) and the HumanEval+ probes n = 32 (20 correct, 12 wrong).
Layer 29 has the highest `prompt_last` AUROC on both, so it's the intervention layer. The intervention direction is
the GSM8K `prompt_last` probe's weight vector at layer 29, mapped to raw activation space, normalized, and pointing
toward "correct".

### Steering strength sweep (GSM8K dev, layer 29)

The edit is resid_post ← resid_post + α·u at every position (prompt and generated tokens), with α = fraction × 38.65,
the median per-token residual norm at layer 29. The no-edit condition was re-run on the same 16 problems with the same
batching. These 16 problems are also part of the probe's 200 training problems.

| condition | accuracy (n = 16) | mean response tokens | truncated | fixed (wrong→right) | broken (right→wrong) | exact McNemar p |
|---|---|---|---|---|---|---|
| no edit | 68.75% | 1658 | 18.75% | n/a | n/a | n/a |
| steer 0.10× | 87.5% | 1101 | 0% | 3 | 0 | 0.25 |
| steer 0.25× | 81.25% | 1389 | 6.25% | 2 | 0 | 0.5 |
| steer 0.50× | 87.5% | 1126 | 6.25% | 3 | 0 | 0.25 |

### Reproducibility note

In the full baseline run, the same 16 problems scored 87.5% (14/16). In the sweep's no-edit condition they scored
68.75% (11/16). Only 5 of the 16 responses were identical and 13 of 16 had the same correctness. Both runs are greedy;
the difference is batch composition (bf16 with left-padded batches of different problems). Batch-to-batch variation
is therefore about 3 of 16 problems, the same size as the sweep's fixed counts. `intervene.py` keeps prompts, order,
and batching identical across its conditions, so its comparisons are batch-matched. The random-direction control in
the held-out run is the comparison for perturbation effects in general.

## Not done yet

- **Held-out intervention run** (`configs/intervene.yaml`): no edit, `probe_pos`, `random_pos`, and `probe_ablate` on
  the first 40 GSM8K test problems and 24 HumanEval+ test problems. About 3.5 h of GPU time. Pending: picking α from
  the sweep (0.10× and 0.50× tie) and choosing between our HumanEval+ test split and Jacob's `coding_eval` held-out
  IDs.
- **HumanEval+ dev for problems 33–64** and the **baseline on the test splits** haven't been run.
- **Input ablation** (`prompt_variant=plain`, no step-by-step instruction) is implemented but hasn't been run.
- **SAE comparison** isn't implemented, since no pretrained SAE for this model is wired in.
- **Exporter to `coding_eval`'s manifest/samples format** isn't written. Its splits (seed 2026, 20 dev / 40 held-out)
  differ from ours.
- **W&B logging** exists behind `wandb.enabled` but was off for these runs.

## Reproduce

```bash
source .venv/bin/activate
python experiments/baseline.py --config configs/baseline.yaml -o 'splits=[dev]' -o benchmarks.humanevalplus.limit=32
python experiments/probe.py --config configs/probe.yaml
python experiments/probe.py --config configs/probe.yaml -o run_name=humanevalplus_dev -o benchmark=humanevalplus
python experiments/intervene.py --config configs/intervene_sweep.yaml
```

The GSM8K dev baseline came from `python experiments/baseline.py --config configs/baseline.yaml`, which was stopped
after GSM8K dev completed. HumanEval+ dev came from the limit-32 command above, which skips the finished GSM8K split.
Each run's launch command is recorded in its `*.meta.json` / `config.json`.
