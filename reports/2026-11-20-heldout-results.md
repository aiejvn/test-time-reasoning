# Held-out interventions: `heldout_half` (in progress, auto-updated)

Layer 29, α = [0.1] × 38.65 (median token norm), positions=all. Command: `python experiments/intervene.py --config configs/intervene.yaml -o run_name=heldout_half -o 'alpha_fracs=[0.1]' -o 'evals=[{benchmark: gsm8k, split: test, n: 20}, {benchmark: humanevalplus, split: test, n: 12}]'`

| benchmark | condition | n | accuracy | mean tokens | truncated | fixed / broken | McNemar p |
|---|---|---|---|---|---|---|---|
| gsm8k | none | 20 | 90.0% | 1401 | 10.0% | n/a | n/a |
| gsm8k | probe_ablate | 20 | 95.0% | 1487 | 5.0% | 1 / 0 | 1 |
| gsm8k | probe_pos | 20 | 90.0% | 1421 | 10.0% | 0 / 0 | 1 |
