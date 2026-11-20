"""Activation interventions along a linear-probe direction, with matched baseline and controls.

For each eval set (benchmark, split, first n problems of that split) and each condition, generates
and grades under the edit, using the same prompts / decoding settings / batching as the baseline run
(read from the baseline config, so splits are identical). The no-edit condition `none` is re-run here
on the same problems, so every flip count compares runs that differ only in the edit.

Conditions (steering strength alpha = alpha_frac * median per-token ||resid|| at the layer):
  none          no edit
  probe_pos     resid += alpha * u_probe       (toward the probe's "correct" side)
  probe_neg     resid -= alpha * u_probe
  random_pos    resid += alpha * u_random      (random-direction control, same norm)
  probe_ablate  resid -= (resid . u_probe) u_probe
  random_ablate resid -= (resid . u_random) u_random

Writes to {out_dir}/{run_name}/: {bench}_{split}__{cond}[_a{alpha_frac}].jsonl/.meta.json per
condition, and summary.json (accuracy, mean response tokens, truncation, and vs `none`: fixed =
wrong->right, broken = right->wrong, exact McNemar p).

Usage:
  python experiments/intervene.py --config configs/intervene.yaml [-o key=value ...]
"""
import argparse
import json
import os
import shlex
import sys
from pathlib import Path

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import pandas as pd  # noqa: E402
import torch  # noqa: E402
from scipy.stats import binomtest  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "experiments"))
import reasoning_env as R  # noqa: E402
import sae_env  # noqa: E402
from baseline import load_config  # noqa: E402

STEER = {"probe_pos": ("probe", 1.0), "probe_neg": ("probe", -1.0), "random_pos": ("random", 1.0)}
ABLATE = {"probe_ablate": "probe", "random_ablate": "random"}


@torch.no_grad()
def resid_norm(model, tok, layer, items, bench, prompt_variant, n=16):
    """Median per-token ||resid_post|| at `layer` over the prompts of `items` (BOS excluded)."""
    norms = []
    for it in items[:n]:
        ids = sae_env.chat_inputs(tok, [{"role": "user", "content": R.PROMPTS[bench][prompt_variant].format(question=it["question"])}])
        with R.capture_resid_post(model, [layer]) as buf:
            model.model(**ids)
        norms.append(buf[layer][0][0, 1:].float().norm(dim=-1).cpu())
    return float(torch.cat(norms).median())


def flips(base: pd.DataFrame, cond: pd.DataFrame) -> dict:
    m = base.set_index("id").correct.astype(bool).to_frame("b").join(cond.set_index("id").correct.astype(bool).rename("c"))
    fixed, broken = int((~m.b & m.c).sum()), int((m.b & ~m.c).sum())
    p = binomtest(fixed, fixed + broken, 0.5).pvalue if fixed + broken else 1.0
    return dict(fixed=fixed, broken=broken, net=fixed - broken, mcnemar_p=float(p))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default=str(ROOT / "configs/intervene.yaml"))
    ap.add_argument("-o", "--override", action="append", default=[])
    args = ap.parse_args()
    cfg = load_config(args.config, args.override)
    base_cfg = load_config(ROOT / cfg["baseline_config"], [])
    cmd = " ".join(shlex.quote(a) for a in [sys.executable] + sys.argv)
    print("launch:", cmd)

    probe_dir = ROOT / cfg["probe_dir"]
    probe_res = json.load(open(probe_dir / "results.json"))
    layer = probe_res["best"]["layer"] if cfg["layer"] == "best" else int(cfg["layer"])
    u_probe = torch.load(probe_dir / "directions.pt")[cfg["pool"]][layer]
    u_rand = R.random_direction(u_probe.numel(), seed=cfg["random_seed"])
    dirs = {"probe": u_probe, "random": u_rand}

    out = ROOT / cfg["out_dir"] / cfg["run_name"]
    out.mkdir(parents=True, exist_ok=True)
    model, tok = R.load_model(base_cfg["model_id"])
    data = {b: R.load_benchmark(b, s["n_dev"], s["n_test"], seed=base_cfg["seed"]) for b, s in base_cfg["benchmarks"].items()}
    pv = base_cfg["prompt_variant"]
    norm = resid_norm(model, tok, layer, data["gsm8k"]["dev"], "gsm8k", pv)
    print(f"layer {layer} (probe CV AUROC {probe_res['results'][f'{cfg['pool']}/L{layer}']['cv_auroc']:.3f}), "
          f"median per-token resid norm {norm:.2f}", flush=True)
    meta0 = dict(cfg, command=cmd, layer=layer, resid_norm=norm, gen=base_cfg["gen"], prompt_variant=pv)
    json.dump(meta0, open(out / "config.json", "w"), indent=2)

    def edits_for(cond, frac):
        if cond == "none":
            return None
        if cond in STEER:
            d, sign = STEER[cond]
            return {layer: R.steer(dirs[d], sign * frac * norm)}
        return {layer: R.ablate(dirs[ABLATE[cond]])}

    summary = {}
    for ev in cfg["evals"]:
        bench, split = ev["benchmark"], ev["split"]
        items = data[bench][split][: ev["n"]]
        runs = [("none", None)] + [(c, None) for c in cfg["conditions"] if c in ABLATE] + \
               [(c, f) for f in cfg["alpha_fracs"] for c in cfg["conditions"] if c in STEER]
        dfs = {}
        for cond, frac in runs:
            name = f"{bench}_{split}__{cond}" + (f"_a{frac}" if frac is not None else "")
            if not (out / f"{name}.jsonl").exists():
                print(f"[{name}] {len(items)} problems ...", flush=True)
                rows, gens = R.run_benchmark(model, tok, bench, items, prompt_variant=pv, edits=edits_for(cond, frac),
                                             positions=cfg["positions"], **base_cfg["gen"])
                R.save_log(str(out / name), pd.DataFrame(rows), gens, tok,
                           dict(meta0, benchmark=bench, split=split, condition=cond, alpha_frac=frac,
                                alpha=None if frac is None else frac * norm))
            df = pd.read_json(out / f"{name}.jsonl", lines=True)
            dfs[name] = df
            s = dict(benchmark=bench, condition=cond, alpha_frac=frac, n=len(df), accuracy=float(df.correct.mean()),
                     mean_resp_tokens=float(df.n_resp_tokens.mean()), truncated=float(df.truncated.mean()))
            if cond != "none":
                s.update(flips(dfs[f"{bench}_{split}__none"], df))
            summary[name] = s
            print(f"[{name}] {s}", flush=True)
            json.dump(summary, open(out / "summary.json", "w"), indent=2)
    print(pd.DataFrame(summary.values()).to_string())


if __name__ == "__main__":
    main()
