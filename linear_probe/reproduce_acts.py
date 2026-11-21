"""Can the probe direction be rebuilt from committed files alone?

The probe direction depends on activations (artifacts/, gitignored). This script rebuilds them from
the committed baseline responses (output/baseline/<run>/<bench>_<split>.jsonl): it re-tokenizes each
prompt and response, re-runs them through the model, retrains the probe exactly as probe.py does,
and compares against a saved directions.pt.

Reported:
    retokenized_len_mismatch  responses whose re-tokenized length differs from the logged n_resp_tokens
    act_max_abs_diff          per (pool, layer): largest difference vs. the saved activations, if given
    direction_cosine          per (pool, layer): cosine between the rebuilt and saved probe directions
    cv_auroc                  per (pool, layer): rebuilt vs. saved

Run
---
    python linear_probe/reproduce_acts.py --responses output/baseline/baseline/gsm8k_dev.jsonl \
        --saved-acts artifacts/baseline/baseline/gsm8k_dev_acts.pt --saved-probe artifacts/probe/gsm8k_dev \
        --out artifacts/probe/gsm8k_dev_rebuilt
"""
import argparse
import json
import shlex
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "linear_probe"))
import reasoning_env as R  # noqa: E402
from probe import cross_validated_auroc, probe_direction  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--responses", required=True)
    ap.add_argument("--saved-acts", default=None)
    ap.add_argument("--saved-probe", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--batch-size", type=int, default=4)  # baseline.yaml extract_batch_size
    args = ap.parse_args()
    cmd = " ".join(shlex.quote(a) for a in [sys.executable] + sys.argv)

    saved_res = json.load(open(ROOT / args.saved_probe / "results.json"))
    pcfg = saved_res["config"]
    rows = [json.loads(line) for line in open(ROOT / args.responses)]
    model, tok = R.load_model()
    gens, mismatch = [], 0
    for r in rows:
        resp_ids = tok(r["response"], add_special_tokens=False)["input_ids"]
        mismatch += len(resp_ids) != r["n_resp_tokens"]
        gens.append(dict(prompt_ids=tok(r["prompt"], add_special_tokens=False)["input_ids"], resp_ids=resp_ids))
    pools = sorted({v["pool"] for v in saved_res["results"].values()})
    layers = sorted({v["layer"] for v in saved_res["results"].values()})
    acts = R.extract_activations(model, tok, gens, layers, pools, args.batch_size)

    saved_acts = torch.load(ROOT / args.saved_acts)["acts"] if args.saved_acts else None
    saved_dirs = torch.load(ROOT / args.saved_probe / "directions.pt")
    y = np.array([bool(r["correct"]) for r in rows])
    n_splits = min(pcfg["cv_folds"], int(y.sum()), int((~y).sum()))
    report = dict(command=cmd, n=len(rows), retokenized_len_mismatch=mismatch, per=[])
    for p in pools:
        for l in layers:
            X = acts[p][l].numpy()
            u = probe_direction(X, y, pcfg["C"])
            e = dict(pool=p, layer=l, direction_cosine=float(u @ saved_dirs[p][l]),
                     cv_auroc=cross_validated_auroc(X, y, n_splits, pcfg["C"], pcfg["seed"]),
                     saved_cv_auroc=saved_res["results"][f"{p}/L{l}"]["cv_auroc"])
            if saved_acts is not None:
                e["act_max_abs_diff"] = float((acts[p][l] - saved_acts[p][l]).abs().max())
            report["per"].append(e)
            print(e, flush=True)
    out = ROOT / args.out
    out.mkdir(parents=True, exist_ok=True)
    json.dump(report, open(out / "reproduce.json", "w"), indent=2)
    print("retokenized_len_mismatch", mismatch, "->", out)


if __name__ == "__main__":
    main()
