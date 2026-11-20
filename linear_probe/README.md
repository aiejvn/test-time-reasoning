# Linear probes and probe-guided interventions

**Question:** DeepSeek-R1-Distill-Llama-8B carries a 4096-number vector for every token at every layer.
Can a simple linear classifier read "this answer is going to be correct" off that vector? If it can,
does pushing the vector in the classifier's direction make the model answer correctly more often?

Results are in **[REPORT.md](REPORT.md)**.

## The two steps

```
baseline run (experiments/baseline.py)
  │  saves, for every problem: the model's answer, whether it was correct,
  │  and the activation vector at layers 4, 19, 29, 30
  ▼
probe.py            step 1: train a classifier per layer to predict "correct?" from the activation
  │  saves each classifier's weight vector = a direction u in activation space
  ▼
intervene.py        step 2: re-generate with the layer's output edited along u, against controls
     reports which problems were fixed and which were broken
```

### Step 1: `probe.py`

- **Input:** the baseline run's saved activations and correctness labels, from `artifacts/baseline/baseline/`.
- **Model:** for each layer (4, 19, 29, 30) and each way of summarizing a problem's activations, a logistic regression
  is trained.
  - Summaries:
    - `prompt_last`: the vector at the last prompt token, before the model writes anything.
    - `resp_mean`: the vector averaged over the whole response.
  - Inputs are standardized.
  - Classes are balanced, because correct answers dominate.
  - Regularization is strong (C = 0.01), since there are 4096 inputs and only about 200 examples.
- **Score:** 5-fold cross-validated AUROC, the probability that a random correct problem scores above a random wrong
  one. 0.5 is chance and 1.0 is perfect.
- **Output:** `artifacts/probe/<run_name>/`
  - `results.json`: AUROC for every layer and summary, plus the best layer.
  - `directions.pt`: each probe's weight vector, converted to raw activation units, length 1, pointing toward
    "correct".

### Step 2: `intervene.py`

Each condition is one way of editing the chosen layer's output vector h, at every token, while the model generates:

| condition | edit | purpose |
|---|---|---|
| `none` | no change | the matched baseline; re-run here so batching is identical |
| `probe_pos` | h + α·u | push toward "correct" |
| `probe_neg` | h − α·u | push toward "wrong" (available, not in the default run) |
| `random_pos` | h + α·r, where r is a random unit vector | same-size push in an unrelated direction (control) |
| `probe_ablate` | h − (h·u)u | delete the part of h along u |
| `random_ablate` | h − (h·r)r | delete the part along a random direction (control) |

- **Strength:** α = `alpha_frac` × the median length of one token's activation vector at that layer (38.65 at
  layer 29). So 0.1 means a push about 10% the size of a typical token's vector.
- **Same settings everywhere:** prompts, problem order, batches and decoding are identical across conditions, and are
  read from `configs/baseline.yaml`. The edit is the only difference.
- **Output:** `artifacts/intervene/<run_name>/`
  - Every response in `<bench>_<split>__<condition>.jsonl`.
  - Per condition in `summary.json`: accuracy, mean length, truncation rate, fixed (wrong → right), broken
    (right → wrong), and an exact McNemar p-value on fixed vs. broken.

The edits use the hook interface in `src/reasoning_env.py`: `R.steer`, `R.ablate`, `R.intervene` and
`R.run_benchmark(..., edits=...)`. See the top-level README.

## Files

| file | what it is |
|---|---|
| `probe.py` | step 1 |
| `intervene.py` | step 2 |
| `configs/probe.yaml` | which baseline run to read; CV folds, C, which summary picks the layer |
| `configs/intervene_sweep.yaml` | strength sweep on dev problems (0.1×, 0.25×, 0.5×) |
| `configs/intervene.yaml` | held-out run: no edit / probe / random / ablate on test problems |
| `REPORT.md` | results, setup, and how to read every number |

## Run

```bash
source .venv/bin/activate
python linear_probe/probe.py --config linear_probe/configs/probe.yaml                     # GSM8K probes
python linear_probe/probe.py --config linear_probe/configs/probe.yaml \
  -o run_name=humanevalplus_dev -o benchmark=humanevalplus                                  # HumanEval+ probes
python linear_probe/intervene.py --config linear_probe/configs/intervene_sweep.yaml       # pick strength (~40 min)
python linear_probe/intervene.py --config linear_probe/configs/intervene.yaml -o 'alpha_fracs=[0.1]'
```

Any config value can be overridden with `-o key=value`, using dots for nesting, for example
`-o 'evals=[{benchmark: gsm8k, split: test, n: 20}]'`. Finished conditions are skipped on rerun.
