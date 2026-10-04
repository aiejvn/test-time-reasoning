J_ = LLM-judge rate (judge.py), S_ = substring-judge rate. base = no intervention, ablate = direction projected out of every residual position, orth = weights orthogonalized, actadd = direction added at the selected layer.

| model                    | layer        | filters_ok   |   tl_vs_hf_kl |   S_harmful_refuse_base |   S_harmful_refuse_ablate |   S_harmless_refuse_actadd |   harmless_kl_ablate |   orth_vs_hook_kl |   runtime_s |
|:-------------------------|:-------------|:-------------|--------------:|------------------------:|--------------------------:|---------------------------:|---------------------:|------------------:|------------:|
| mistral-7b-instruct-v0.1 | 10/32 (0.31) | True         |          0.00 |                    0.00 |                      0.03 |                       0.00 |                 0.03 |              0.00 |      375.10 |
| olmo2-1b-base            | 3/16 (0.19)  | False        |          0.00 |                    0.00 |                      0.00 |                       0.00 |                 0.03 |              0.05 |      122.30 |
| phi3-mini-instruct       | 16/32 (0.50) | True         |          0.00 |                    0.98 |                      0.03 |                       0.97 |                 0.07 |              0.00 |      262.80 |
| qwen1.5-1.8b-chat        | 15/24 (0.62) | True         |          0.00 |                    0.75 |                      0.00 |                       0.92 |                 0.04 |              0.00 |       99.70 |
| qwen2.5-1.5b-instruct    | 22/28 (0.79) | True         |          0.03 |                    1.00 |                      0.00 |                       0.98 |                 0.07 |              0.00 |      165.10 |
| qwen2.5-7b-instruct      | 18/28 (0.64) | True         |          0.00 |                    0.97 |                      0.05 |                       0.50 |                 0.07 |              0.00 |      286.60 |
| qwen3-1.7b               | 19/28 (0.68) | False        |          0.00 |                    0.81 |                      0.00 |                       0.86 |                 0.17 |              0.00 |      112.50 |
| qwen3-8b                 | 20/36 (0.56) | True         |          0.00 |                    0.95 |                      0.00 |                       0.83 |                 0.07 |              0.00 |      330.30 |
