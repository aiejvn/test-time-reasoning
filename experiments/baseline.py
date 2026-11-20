"""Baseline run for the test-time reasoning intervention experiment: the unchanged
DeepSeek-R1-Distill-Llama-8B on GSM8K and HumanEval+ dev/test splits.

For each (benchmark, split) writes to {out_dir}/{run_name}/:
  {bench}_{split}.jsonl      one row per problem: id, correct, pred/code, n_resp_tokens, truncated, prompt, response
  {bench}_{split}.meta.json  config, launch command, accuracy
  {bench}_{split}_acts.pt    {"ids": [...], "acts": {pool: {layer: [N, d_model]}}} resid_post activations
  summary.json               accuracy / length / truncation per (benchmark, split)

Usage:
  python experiments/baseline.py --config configs/baseline.yaml [-o key.sub=value ...]
"""
import argparse
import json
import os
import shlex
import sys
from pathlib import Path

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")  # growing KV cache fragments the allocator

import pandas as pd
import torch
import yaml

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
import reasoning_env as R  # noqa: E402


def load_config(path, overrides):
    cfg = yaml.safe_load(open(path))
    for o in overrides:
        key, val = o.split("=", 1)
        *parents, leaf = key.split(".")
        d = cfg
        for p in parents:
            d = d.setdefault(p, {})
        d[leaf] = yaml.safe_load(val)
    return cfg


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default=str(ROOT / "configs/baseline.yaml"))
    ap.add_argument("-o", "--override", action="append", default=[], help="dotted key=value (YAML-parsed)")
    args = ap.parse_args()
    cfg = load_config(args.config, args.override)
    cmd = " ".join(shlex.quote(a) for a in [sys.executable] + sys.argv)
    print("launch:", cmd)

    out = (ROOT / cfg["out_dir"] / cfg["run_name"])
    out.mkdir(parents=True, exist_ok=True)
    json.dump(dict(cfg, command=cmd), open(out / "config.json", "w"), indent=2)

    run = None
    if cfg["wandb"]["enabled"]:
        import wandb
        run = wandb.init(project=cfg["wandb"]["project"], name=cfg["run_name"], config=cfg,
                         notes=cmd, tags=[cfg["model_id"], "baseline"])

    model, tok = R.load_model(cfg["model_id"])
    data = {b: R.load_benchmark(b, s["n_dev"], s["n_test"], seed=cfg["seed"]) for b, s in cfg["benchmarks"].items()}
    summary = json.load(open(out / "summary.json")) if (out / "summary.json").exists() else {}
    for split in cfg["splits"]:  # split-major: every dev split finishes before any held-out test split
        for bench in data:
            # `limit` runs only the first N problems of each split without changing how splits are drawn
            items, tag = data[bench][split][: cfg["benchmarks"][bench].get("limit")], f"{bench}_{split}"
            if (out / f"{tag}_acts.pt").exists():
                print(f"[{tag}] already done, skipping (delete {out / tag}* to rerun)", flush=True)
                continue
            print(f"[{tag}] generating {len(items)} problems ...", flush=True)
            rows, gens = R.run_benchmark(model, tok, bench, items, prompt_variant=cfg["prompt_variant"], **cfg["gen"])
            df = pd.DataFrame(rows)
            s = dict(n=len(df), accuracy=float(df.correct.mean()), mean_resp_tokens=float(df.n_resp_tokens.mean()),
                     truncated=float(df.truncated.mean()))
            summary[tag] = s
            print(f"[{tag}] {s}", flush=True)
            R.save_log(str(out / tag), df, gens, tok, dict(cfg, command=cmd, benchmark=bench, split=split, **s))

            print(f"[{tag}] extracting activations at layers {cfg['layers']} ...", flush=True)
            acts = R.extract_activations(model, tok, gens, cfg["layers"], cfg["pools"], cfg["extract_batch_size"])
            torch.save({"ids": list(df.id), "acts": acts}, out / f"{tag}_acts.pt")
            if run:
                run.log({f"{tag}/{k}": v for k, v in s.items()})
            json.dump(summary, open(out / "summary.json", "w"), indent=2)  # rewritten after each split
    if run:
        run.finish()
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
