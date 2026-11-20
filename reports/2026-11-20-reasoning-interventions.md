# Test-time reasoning interventions: status as of 2026-11-20

## The question

Can we improve a reasoning model's problem-solving at inference time by finding an internal representation linked to
correct answers and pushing the model along it, without retraining? Does an effect found on math carry over to code?

- **Model:** DeepSeek-R1-Distill-Llama-8B. It has 32 transformer layers, and each token's internal state (the
  "residual stream") is a 4096-number vector. It writes a reasoning trace inside `<think> … </think>` before answering.
  We run it in bf16 on one NVIDIA A10G (23 GB).
- **Layers studied:** 4, 19, 29, 30, counting from 0. Layer 19 is the one used in "I Have Covered All the Bases Here"
  (Galichin et al. 2025). At each layer we read the layer's output, called resid_post.
- **Benchmarks:**
  - **GSM8K:** grade-school math word problems with a single numeric answer.
  - **HumanEval+:** Python function-writing tasks checked by unit tests, using the stricter "plus" test set.

## Related reports

| report | contents | updated |
|---|---|---|
| [2026-11-20-heldout-results.md](2026-11-20-heldout-results.md) | held-out intervention results, per-problem outcomes | every 10 min while running |
| [2026-11-20-sae-exploration.md](2026-11-20-sae-exploration.md) | SAE features that differ between correct and wrong answers | after each SAE (queued) |

## Pipeline

| step | file | what it does |
|---|---|---|
| Model + hooks | `src/reasoning_env.py` | Loads the model. Reads layer outputs at any set of layers. Applies edits to a layer's output during generation (`steer`, `ablate`). Loads, prompts and grades both benchmarks. |
| Baseline | `experiments/baseline.py`, `configs/baseline.yaml` | Runs the unchanged model, grades each answer, and saves every prompt, response and grade plus layer activations. |
| Linear probes | `experiments/probe.py`, `configs/probe.yaml` | Trains one classifier per layer to predict "will this answer be correct?" from activations. Saves each classifier's direction. |
| Interventions | `experiments/intervene.py`, `configs/intervene*.yaml` | Re-generates with an edit applied at one layer, plus a matched no-edit run and controls. Counts which problems flip. |
| SAE exploration | `experiments/sae_explore.py`, `experiments/sae_report.py`, `configs/sae_explore.yaml` | Encodes activations with pretrained sparse autoencoders and lists features that differ between correct and wrong answers. |
| Docker evaluator | `coding_eval/` (Jacob) | Sandboxed official EvalPlus grading of saved HumanEval+ outputs. Merged, not yet connected to our outputs. |

### Settings shared by every run

- **Prompt (GSM8K):** `{question}` followed by "Please reason step by step, and put your final answer within \boxed{}."
  This is DeepSeek's recommended prompt. There is no system prompt.
- **Prompt (HumanEval+):** "Complete the following Python function. Please reason step by step, then return the
  complete function in a ```python block." followed by the function signature and docstring.
- **Decoding:** greedy (always the most likely next token), at most 4096 new tokens, batches of 8.
- **Grading:**
  - **GSM8K:** the last `\boxed{…}` after `</think>` must equal the reference number exactly. If there's no box, the
    last number is used.
  - **HumanEval+:** the last python code block after `</think>` is run against the base and plus unit tests in a
    subprocess with a 30-second timeout.
  - A response that hits the 4096-token limit before finishing almost always counts as wrong.
- **Splits:** each benchmark is shuffled with seed 0 and divided into a dev split and a test split that don't overlap.
  - **Dev** problems are used to build and tune things (probes, steering strength).
  - **Test** problems are held out for the final comparison.
  - **Sizes:** GSM8K has 200 dev / 300 test, drawn from its 1319-problem test set. HumanEval+ has 64 dev / 99 test,
    from 163 problems.
  - HumanEval/32 is excluded because its published HumanEval+ test fails even the official reference solution.

### Checks on the infrastructure

- **Hooks:**
  - An edit adding 8·u at a layer changed that layer's output by exactly 8·u at every token (cosine 1.0).
  - Ablation reduced the output's component along u from 10.9 to 0.015.
  - An edit of strength 0 produced token-for-token identical generations.
- **Graders:** 163 of the 164 official HumanEval+ reference solutions pass (the one failure is HumanEval/32, above).
  Deliberately wrong functions fail.

## Results

### 1. Baseline: unchanged model on dev problems

| benchmark | problems | accuracy | mean response length (tokens) | hit 4096-token limit | of those, still correct |
|---|---|---|---|---|---|
| GSM8K dev | 200 | 83.0% (166) | 1429 | 9.5% (19) | 2 |
| HumanEval+ dev | 32 (first half of 64) | 62.5% (20) | 2657 | 37.5% (12) | 1 |

Of the wrong answers, 17 of 34 (GSM8K) and 11 of 12 (HumanEval+) hit the token limit. In other words, many failures
are the model still reasoning when it runs out of room.

### 2. Linear probes: is "will be correct" readable from a layer?

**Method.** For each layer, a logistic-regression classifier predicts "answer will be correct" from that layer's
4096-number activation vector. Inputs are standardized, classes are re-weighted because correct answers outnumber wrong
ones, and the L2 regularization strength is C = 0.01. Activations are summarized in two ways:

- **last prompt token:** the activation at the final prompt token, right after `<think>` and before the model has
  generated anything. If this predicts correctness, the model's "outcome" is partly visible before it reasons.
- **response mean:** the activation averaged over every response token. This sees the whole reasoning trace, including
  signs of looping or truncation, so it is expected to score higher.

**Score.** AUROC under 5-fold cross-validation: each problem is scored by a classifier that never saw it during
training. AUROC is the probability that a randomly chosen correct problem gets a higher "correct" score than a randomly
chosen wrong one. 0.5 is chance and 1.0 is perfect. It doesn't depend on the class ratio, which matters here because
always guessing "correct" would already score 83% accuracy on GSM8K.

| layer | GSM8K, last prompt token | GSM8K, response mean | HumanEval+, last prompt token | HumanEval+, response mean |
|---|---|---|---|---|
| 4 | 0.648 | 0.850 | 0.646 | 0.896 |
| 19 | 0.702 | 0.852 | 0.717 | 0.913 |
| 29 | **0.729** | 0.859 | **0.771** | 0.892 |
| 30 | 0.714 | 0.863 | 0.758 | 0.896 |

- **Sample sizes:** GSM8K has 200 problems (34 wrong) and HumanEval+ has 32 (12 wrong). The HumanEval+ numbers are
  noisy.
- **What a probe is:** a direction in activation space, not an SAE feature. A high AUROC means the information is
  linearly readable at that layer. It doesn't mean the model uses it, and it doesn't mean the layer "contributes" to
  the answer.
- **Layer chosen for interventions:** layer 29, the best last-prompt-token score on both benchmarks. The intervention
  direction u is the GSM8K last-prompt-token probe's weight vector at layer 29, converted back to raw activation units,
  scaled to length 1, and pointing toward "correct".

### 3. Choosing the steering strength (GSM8K dev)

**Edit.** At layer 29, every token's activation h becomes h + α·u, for prompt tokens and every generated token. α is
set as a fraction of 38.65, the median length of a single token's activation vector at layer 29. So 0.1× adds a push
of 3.9, about 10% of a typical token's size.

**Problems.** The first 16 GSM8K dev problems. These are also among the probe's 200 training problems, so this sweep
only picks a strength and is not evidence of an effect.

| condition | accuracy | mean tokens | hit limit | fixed | broken | McNemar p |
|---|---|---|---|---|---|---|
| no edit | 68.75% (11/16) | 1658 | 18.75% | n/a | n/a | n/a |
| steer 0.10× (α = 3.9) | 87.5% (14/16) | 1101 | 0% | 3 | 0 | 0.25 |
| steer 0.25× (α = 9.7) | 81.25% (13/16) | 1389 | 6.25% | 2 | 0 | 0.5 |
| steer 0.50× (α = 19.3) | 87.5% (14/16) | 1126 | 6.25% | 3 | 0 | 0.25 |

- **fixed:** wrong with no edit, right with the edit.
- **broken:** right with no edit, wrong with the edit.
- **McNemar p:** the exact two-sided test on fixed versus broken counts, defined in the glossary.

0.1× and 0.5× tied, so the smaller, 0.1×, was used for the held-out run.

**Batch-composition noise.** In the full baseline run these same 16 problems scored 87.5% (14/16). In the sweep's
no-edit condition they scored 68.75% (11/16), and only 5 of the 16 responses were word-for-word identical. Decoding is
greedy in both. The difference is which other problems shared the batch, which changes padding and bf16 rounding, and
that cascades in long reasoning traces. Changing nothing but the batch therefore flips about 3 in 16 outcomes, which is
as large as the sweep's "fixed" counts. Inside `intervene.py`, every condition uses identical batches, so its
comparisons are matched. Even so, small perturbations can tip borderline problems, which is why a random-direction
control is required.

### 4. Held-out intervention run

**Problems:** the first 20 problems of the GSM8K test split and the first 12 of the HumanEval+ test split. None were
used for the probe or the sweep.

**Conditions:**

| condition | edit at layer 29 |
|---|---|
| no edit | none |
| probe steering | h + α·u, α = 3.9 |
| random control | h + α·r, where r is a random unit vector, same α |
| probe ablation | h − (h·u)u, removing the component along u |

**Status when written:**
- All four GSM8K conditions are done; HumanEval+ is in progress.
- Full tables and per-problem outcomes are in [2026-11-20-heldout-results.md](2026-11-20-heldout-results.md).
- GSM8K test: no edit 90% (18/20), probe steering 90% (0 fixed, 0 broken), random control 95% (2 fixed, 1 broken),
  probe ablation 95% (1 fixed, 0 broken). Every McNemar p = 1.0.

## Not done yet

| item | status |
|---|---|
| Held-out intervention run | GSM8K done, HumanEval+ running. Then the SAE queue starts. |
| SAE exploration (5 SAEs: Llama Scope at layers 4/19/29/30, the Galichin et al. layer-19 SAE) | Queued after the held-out run. Script written; it has run on the CPU against saved activations but not yet on the GPU. |
| SAE-guided interventions | Not started. Would steer along individual SAE feature directions, against the same controls. |
| Larger held-out sets (planned 40 GSM8K / 24 HumanEval+, or more) | Not run. The current run is half size for time. |
| HumanEval+ dev problems 33–64; baseline on test splits | Not run. |
| Input ablation (prompt without "reason step by step") | Implemented (`prompt_variant=plain`), not run. |
| Grading with Jacob's `coding_eval` | Needs an exporter to its manifest/samples format. Its splits (seed 2026, 20 dev / 40 held-out) differ from ours. |
| W&B logging | Available (`wandb.enabled`), off for these runs. |

## Glossary

- **Residual stream / resid_post:** the 4096-number vector each token carries between layers. resid_post at layer L is
  the vector after layer L.
- **Linear probe:** a simple (logistic-regression) classifier trained on activations to predict a label. Its weight
  vector is a direction in activation space.
- **AUROC:** the probability that a random positive example scores above a random negative one. 0.5 is chance, 1.0 is
  perfect.
- **Steering:** adding a fixed vector to a layer's output during generation. **Ablation:** removing the component of
  the output along a direction.
- **α (alpha):** steering strength, here a fraction of the typical per-token activation length at the edited layer.
- **Fixed / broken:** problem-level changes against the matched no-edit run.
- **McNemar test:** a test on paired outcomes that looks only at problems that changed. Under "no directional effect",
  each change is equally likely to be a fix or a break. p is the two-sided probability of a split at least as lopsided
  as observed.
- **SAE (sparse autoencoder):** a model trained to rewrite activations as a sparse sum of many interpretable
  "features".
- **Cohen's d:** the difference between two group means divided by their pooled standard deviation.

## Reproduce

```bash
source .venv/bin/activate
python experiments/baseline.py --config configs/baseline.yaml                       # GSM8K dev (stopped after this split)
python experiments/baseline.py --config configs/baseline.yaml -o 'splits=[dev]' -o benchmarks.humanevalplus.limit=32
python experiments/probe.py --config configs/probe.yaml
python experiments/probe.py --config configs/probe.yaml -o run_name=humanevalplus_dev -o benchmark=humanevalplus
python experiments/intervene.py --config configs/intervene_sweep.yaml
python experiments/intervene.py --config configs/intervene.yaml -o run_name=heldout_half -o 'alpha_fracs=[0.1]' \
  -o 'evals=[{benchmark: gsm8k, split: test, n: 20}, {benchmark: humanevalplus, split: test, n: 12}]'
python experiments/sae_explore.py --config configs/sae_explore.yaml -o sae=llamascope_l19   # one SAE per run
python experiments/sae_report.py
```

Every run's exact launch command is also saved in its output folder (`*.meta.json` / `config.json`).
