"""Linear probes for "will the model answer correctly" on baseline activations, per layer and pool.

Reads a baseline run's {bench}_{split}.jsonl (labels) and {bench}_{split}_acts.pt (activations),
fits a standardized, class-balanced logistic regression per (layer, pool), reports stratified
k-fold CV AUROC, then refits on all examples and saves the probe direction in raw activation
space (pointing toward "correct") for use as an intervention direction.

Writes to {out_dir}/{run_name}/: results.json (AUROC per layer/pool, best layer), directions.pt
({pool: {layer: [d_model] tensor}}).

Usage:
  python experiments/probe.py --config configs/probe.yaml [-o key=value ...]
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
from baseline import load_config  # noqa: E402


def probe(C):
    return make_pipeline(StandardScaler(), LogisticRegression(C=C, class_weight="balanced", max_iter=5000))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default=str(ROOT / "configs/probe.yaml"))
    ap.add_argument("-o", "--override", action="append", default=[])
    args = ap.parse_args()
    cfg = load_config(args.config, args.override)

    src = ROOT / cfg["baseline_dir"]
    tag = f"{cfg['benchmark']}_{cfg['split']}"
    rows = {json.loads(l)["id"]: json.loads(l) for l in open(src / f"{tag}.jsonl")}
    blob = torch.load(src / f"{tag}_acts.pt")
    y = np.array([bool(rows[i]["correct"]) for i in blob["ids"]])
    print(f"{tag}: n={len(y)}, correct={y.sum()}, incorrect={(~y).sum()}")
    n_splits = min(cfg["cv_folds"], int(y.sum()), int((~y).sum()))
    assert n_splits >= 2, "need >=2 examples of each class for CV"

    out = ROOT / cfg["out_dir"] / cfg["run_name"]
    out.mkdir(parents=True, exist_ok=True)
    results, directions = {}, {}
    for pool, by_layer in blob["acts"].items():
        directions[pool] = {}
        for layer, X in by_layer.items():
            X = X.numpy()
            skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=cfg["seed"])
            scores = np.zeros(len(y))
            for tr, te in skf.split(X, y):
                scores[te] = probe(cfg["C"]).fit(X[tr], y[tr]).decision_function(X[te])
            auroc = float(roc_auc_score(y, scores))
            p = probe(cfg["C"]).fit(X, y)
            sc, lr = p.named_steps["standardscaler"], p.named_steps["logisticregression"]
            w = lr.coef_[0] / sc.scale_  # raw-space normal of the decision boundary, toward "correct"
            directions[pool][int(layer)] = torch.tensor(w / np.linalg.norm(w), dtype=torch.float32)
            results[f"{pool}/L{layer}"] = dict(pool=pool, layer=int(layer), cv_auroc=auroc, n_splits=n_splits)
            print(f"  {pool:12s} L{layer:<3d} CV AUROC {auroc:.3f}")

    best = max((r for r in results.values() if r["pool"] == cfg["select_pool"]), key=lambda r: r["cv_auroc"])
    summary = dict(config=cfg, n=len(y), n_correct=int(y.sum()), results=results,
                   best=dict(pool=best["pool"], layer=best["layer"], cv_auroc=best["cv_auroc"]))
    json.dump(summary, open(out / "results.json", "w"), indent=2)
    torch.save(directions, out / "directions.pt")
    print(f"best ({cfg['select_pool']}): layer {best['layer']} CV AUROC {best['cv_auroc']:.3f} -> {out}")


if __name__ == "__main__":
    main()
