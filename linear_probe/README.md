# Linear probes and probe-guided interventions

This folder holds steps 3 and 4 of the experiment described in the [top-level README](../README.md). Results,
including every problem's outcome, are in **[REPORT.md](REPORT.md)**.

## The idea in three sentences

At every token, DeepSeek-R1-Distill-Llama-8B carries a 4096-number vector through each of its 32 layers. If a simple
linear classifier (a *probe*) can predict from that vector whether the model will answer correctly, then the
classifier's weight vector is a direction in that space linked to success. We then push the model's vector along
that direction while it generates, and check whether it gets more answers right than when we push along a random
direction.

## How it fits together

```
experiments/baseline.py    run the unedited model; save each answer, whether it was correct,
        │                  and the activation vectors at layers 4, 19, 29, 30
        ▼
probe.py                   step 3: per layer, train a classifier "activation → correct?"
        │                  output: a score per layer, and a direction u per layer
        ▼
intervene.py               step 4: regenerate with the chosen layer's output edited along u,
                           next to an unedited run and a random-direction control;
                           count which problems were fixed and which were broken
```

## Step 3: `probe.py`

**Input:**
- The baseline run's saved activations, `artifacts/baseline/baseline/<bench>_<split>_acts.pt`.
- Its correct/wrong labels, `<bench>_<split>.jsonl`.

**What it does:** for each layer and each way of summarizing a problem ("pool"):

1. **Train** a logistic regression to predict correct vs. wrong.
   - Inputs are standardized (each dimension scaled to mean 0 and variance 1).
   - Classes are weighted equally, so the probe can't win by always saying "correct".
   - Regularization is strong (C = 0.01), because there are 4096 inputs and only about 200 examples.
2. **Score** with 5-fold cross-validated AUROC: the data is split into 5 parts, and each part is scored by a probe
   trained on the other 4.
3. **Refit** on all examples and save the probe's weight vector as a direction. It's converted from standardized to
   raw activation units, scaled to length 1, and oriented toward "correct".

| pool | the activation vector used for each problem |
|---|---|
| `prompt_last` | the last prompt token, right after `<think>`, before the model has generated anything |
| `resp_mean` | the average over every token of the model's response |

*Terms:*
- **pool:** how one vector per problem is taken from a sequence of per-token vectors.
- **`<think>`:** the tag after which the model writes its reasoning.

**Output:** `artifacts/probe/<run_name>/`
- `results.json`: the AUROC for every pool and layer, plus the best layer.
- `directions.pt`: the directions, as `{pool: {layer: [4096] tensor}}`.

## Step 4: `intervene.py`

**What it does:** generates answers several times on the same problems, once per *condition*. Each condition is a
different edit to the chosen layer's output, applied at every token as the model writes.

| condition | edit to h | why it's there |
|---|---|---|
| `none` | no change | the comparison point; re-run here so batching matches the other conditions exactly |
| `probe_pos` | h + α·u | the test: push toward "correct" |
| `probe_neg` | h − α·u | push toward "wrong" (available, not in the default run) |
| `random_pos` | h + α·r | control: a push of the same size in a random direction |
| `probe_ablate` | h − (h·u)u | remove the model's component along u |
| `random_ablate` | h − (h·r)r | control for ablation |

*Terms:*
- **h:** the layer's output vector at one token.
- **u:** the probe direction from step 3, unit length.
- **r:** a random unit vector (fixed seed).
- **α:** push size, set as `alpha_frac` × the median length of one token's vector at that layer (38.65 at layer 29).
- **(h·u)u:** the part of h lying along u.

**Why everything is matched:** prompts, problem order, batches and decoding settings are copied from
`configs/baseline.yaml` and are the same in every condition. Even greedy decoding changes when the batch composition
changes (padding and bf16 rounding), so the edit has to be the only difference.

**Output:** `artifacts/intervene/<run_name>/`
- `<bench>_<split>__<condition>.jsonl`: every response and grade.
- `summary.json`: per condition, as in the table below.

| field | meaning |
|---|---|
| `accuracy` | share of problems answered correctly |
| `mean_resp_tokens` | average response length in tokens, reasoning included |
| `truncated` | share of responses that hit the 4096-token limit |
| `fixed` | problems wrong without the edit and right with it |
| `broken` | problems right without the edit and wrong with it |
| `mcnemar_p` | exact test of fixed vs. broken; near 1 means no evidence the edit helps or hurts |

## Files

| file | contents |
|---|---|
| `probe.py` | step 3 |
| `intervene.py` | step 4 |
| `configs/probe.yaml` | which baseline run to read, CV folds, regularization C, and which pool picks the layer |
| `configs/intervene_sweep.yaml` | strength sweep on dev problems (0.1×, 0.25×, 0.5×) |
| `configs/intervene.yaml` | held-out run: no edit, probe steering, random control, probe ablation |
| `REPORT.md` | all results, setup, per-problem outcomes, caveats |

*Terms:*
- **CV folds:** how many parts the data is split into for cross-validation.
- **dev / held-out:** problems used for tuning, and problems kept unseen until the final run.

## Run

```bash
source .venv/bin/activate
python linear_probe/probe.py --config linear_probe/configs/probe.yaml                     # GSM8K probes
python linear_probe/probe.py --config linear_probe/configs/probe.yaml \
  -o run_name=humanevalplus_dev -o benchmark=humanevalplus                                  # HumanEval+ probes
python linear_probe/intervene.py --config linear_probe/configs/intervene_sweep.yaml       # choose strength, ~40 min
python linear_probe/intervene.py --config linear_probe/configs/intervene.yaml -o 'alpha_fracs=[0.1]'
```

Any config value can be overridden with `-o key=value`, using dots for nesting, for example
`-o 'evals=[{benchmark: gsm8k, split: test, n: 20}]'`. Conditions that already finished are skipped on rerun.
