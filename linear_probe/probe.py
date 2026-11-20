"""Step 1 of the linear-probe experiment: can we read "will the model answer correctly?" off a layer?

What it does
------------
For every (layer, pooling) pair saved by the baseline run, train a logistic-regression classifier
that takes the model's 4096-number activation vector for a problem and predicts whether the model's
final answer to that problem was correct.

    input  X : [n_problems, 4096] activations, from artifacts/baseline/<run>/<bench>_<split>_acts.pt
    label  y : True if the model answered that problem correctly (from <bench>_<split>.jsonl)

How it's scored
---------------
Stratified k-fold cross-validation: each problem's score comes from a probe that was trained
without it. The metric is AUROC, the probability that a random correct problem scores higher than
a random wrong one (0.5 = chance, 1.0 = perfect).

What it saves
-------------
After scoring, each probe is refit on all problems, and its weight vector becomes a candidate
intervention direction:

    directions.pt : {pool: {layer: [4096] unit vector}}, pointing toward "correct", in raw activation units
    results.json  : CV AUROC for every (pool, layer), plus the best layer for `select_pool`

Run
---
    python linear_probe/probe.py --config linear_probe/configs/probe.yaml [-o key=value ...]
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "experiments"))
from baseline import load_config  # noqa: E402  (shared YAML + `-o key=value` override loader)


def make_probe(C: float):
    """Standardize each of the 4096 dimensions, then an L2-regularized logistic regression.

    class_weight="balanced" reweights the classes so the probe can't win by always predicting
    "correct" (83% of GSM8K dev answers are correct). C is the inverse regularization strength;
    with 4096 inputs and only ~200 examples, strong regularization (small C) is needed.
    """
    return make_pipeline(StandardScaler(), LogisticRegression(C=C, class_weight="balanced", max_iter=5000))


def cross_validated_auroc(X: np.ndarray, y: np.ndarray, n_splits: int, C: float, seed: int) -> float:
    """AUROC where every problem is scored by a probe trained on the other folds only."""
    folds = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    scores = np.zeros(len(y))
    for train_idx, test_idx in folds.split(X, y):
        probe = make_probe(C).fit(X[train_idx], y[train_idx])
        scores[test_idx] = probe.decision_function(X[test_idx])
    return float(roc_auc_score(y, scores))


def probe_direction(X: np.ndarray, y: np.ndarray, C: float) -> torch.Tensor:
    """Fit on all problems and return the probe's weight vector as a unit direction in raw activation space.

    The probe works on standardized inputs (x - mean) / scale, so its weight w_std applies to standardized
    coordinates. Dividing by `scale` gives the equivalent direction for raw activations, which is what
    an intervention edits. Positive direction = toward "correct".
    """
    pipe = make_probe(C).fit(X, y)
    scale = pipe.named_steps["standardscaler"].scale_
    w_std = pipe.named_steps["logisticregression"].coef_[0]
    w_raw = w_std / scale
    return torch.tensor(w_raw / np.linalg.norm(w_raw), dtype=torch.float32)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default=str(ROOT / "linear_probe/configs/probe.yaml"))
    ap.add_argument("-o", "--override", action="append", default=[], help="dotted key=value (YAML-parsed)")
    args = ap.parse_args()
    cfg = load_config(args.config, args.override)

    # --- Load labels and activations from the baseline run ------------------------------------
    src = ROOT / cfg["baseline_dir"]
    tag = f"{cfg['benchmark']}_{cfg['split']}"
    rows = {json.loads(line)["id"]: json.loads(line) for line in open(src / f"{tag}.jsonl")}
    saved = torch.load(src / f"{tag}_acts.pt")          # {"ids": [...], "acts": {pool: {layer: [N, 4096]}}}
    y = np.array([bool(rows[problem_id]["correct"]) for problem_id in saved["ids"]])
    n_correct, n_wrong = int(y.sum()), int((~y).sum())
    print(f"{tag}: {len(y)} problems, {n_correct} correct, {n_wrong} wrong")

    # Each fold needs at least one example of each class.
    n_splits = min(cfg["cv_folds"], n_correct, n_wrong)
    assert n_splits >= 2, "need at least 2 correct and 2 wrong answers for cross-validation"

    # --- Score and save a probe for every (pool, layer) -----------------------------------------
    results, directions = {}, {}
    for pool, by_layer in saved["acts"].items():
        directions[pool] = {}
        for layer, X in by_layer.items():
            X = X.numpy()
            auroc = cross_validated_auroc(X, y, n_splits, cfg["C"], cfg["seed"])
            directions[pool][int(layer)] = probe_direction(X, y, cfg["C"])
            results[f"{pool}/L{layer}"] = dict(pool=pool, layer=int(layer), cv_auroc=auroc, n_splits=n_splits)
            print(f"  {pool:12s} layer {layer:<3d} CV AUROC {auroc:.3f}")

    # The intervention layer is the best-scoring layer for one chosen pool (prompt_last by default:
    # measured before any reasoning is generated, so it can't see the answer).
    best = max((r for r in results.values() if r["pool"] == cfg["select_pool"]), key=lambda r: r["cv_auroc"])

    out = ROOT / cfg["out_dir"] / cfg["run_name"]
    out.mkdir(parents=True, exist_ok=True)
    json.dump(dict(config=cfg, n=len(y), n_correct=n_correct, results=results,
                   best=dict(pool=best["pool"], layer=best["layer"], cv_auroc=best["cv_auroc"])),
              open(out / "results.json", "w"), indent=2)
    torch.save(directions, out / "directions.pt")
    print(f"best ({cfg['select_pool']}): layer {best['layer']}, CV AUROC {best['cv_auroc']:.3f} -> {out}")


if __name__ == "__main__":
    main()
