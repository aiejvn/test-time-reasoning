"""Step 2 of the linear-probe experiment: does pushing the model along the probe direction change its answers?

What it does
------------
Takes the direction u saved by probe.py (the probe's weight vector at one layer, pointing toward
"will answer correctly") and re-generates answers with an edit applied to that layer's output h at
every token. Then it counts which problems the edit fixed and which it broke.

Conditions (h = the layer's output vector at one token, u = probe direction, r = random direction):

    none           h                         no edit: the matched baseline for every other condition
    probe_pos      h + alpha * u             push toward "correct"
    probe_neg      h - alpha * u             push toward "wrong"
    random_pos     h + alpha * r             same-size push in a random direction (control)
    probe_ablate   h - (h . u) u             remove the component along u
    random_ablate  h - (h . r) r             remove the component along r (control)

alpha = alpha_frac * (median length of one token's activation vector at that layer), so strengths
are comparable across layers.

Why the no-edit condition is re-run here
----------------------------------------
Greedy decoding still depends on which problems share a batch (bf16 rounding with padding). Every
condition here uses the same prompts, problem order, batches, and decoding settings (copied from
the baseline config), so the edit is the only thing that differs between rows.

What it saves (artifacts/intervene/<run_name>/)
-----------------------------------------------
    <bench>_<split>__<condition>[_a<alpha_frac>].jsonl   every problem: prompt, response, correct, length
    summary.json   per condition: accuracy, mean response tokens, truncation rate, and, vs. no edit,
                   fixed (wrong -> right), broken (right -> wrong), and an exact McNemar p-value
    config.json    the resolved config, the launch command, the layer, and the norm used for alpha

Run
---
    python linear_probe/intervene.py --config linear_probe/configs/intervene.yaml [-o key=value ...]
"""
import argparse
import json
import os
import shlex
import sys
from pathlib import Path

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")  # growing KV cache fragments memory

import pandas as pd  # noqa: E402
import torch  # noqa: E402
from scipy.stats import binomtest  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "experiments"))
import reasoning_env as R  # noqa: E402  (model loading, hooks, benchmarks)
import sae_env  # noqa: E402  (chat tokenization)
from baseline import load_config  # noqa: E402

# condition name -> (which direction, sign) for additive steering, or which direction to ablate
STEER = {"probe_pos": ("probe", +1.0), "probe_neg": ("probe", -1.0), "random_pos": ("random", +1.0)}
ABLATE = {"probe_ablate": "probe", "random_ablate": "random"}


@torch.no_grad()
def median_token_norm(model, tok, layer: int, items: list, bench: str, prompt_variant: str, n: int = 16) -> float:
    """Median length of a single token's activation vector at `layer`, over the prompts of `items`.

    Used to express steering strength relative to the layer's natural scale. The first (BOS) token
    is excluded; its activation is an outlier hundreds of times larger than normal tokens.
    """
    norms = []
    for item in items[:n]:
        prompt = R.PROMPTS[bench][prompt_variant].format(question=item["question"])
        inputs = sae_env.chat_inputs(tok, [{"role": "user", "content": prompt}])
        with R.capture_resid_post(model, [layer]) as buf:
            model.model(**inputs)
        norms.append(buf[layer][0][0, 1:].float().norm(dim=-1).cpu())
    return float(torch.cat(norms).median())


def flips(no_edit: pd.DataFrame, edited: pd.DataFrame) -> dict:
    """Problem-by-problem comparison of an edited run against the no-edit run.

    fixed  = wrong without the edit, right with it
    broken = right without the edit, wrong with it
    mcnemar_p = exact two-sided test of fixed vs. broken (binomial with p = 0.5 on the changed problems):
                how likely a split at least this lopsided is if changes in either direction are equally likely.
    """
    both = no_edit.set_index("id").correct.astype(bool).to_frame("before").join(
        edited.set_index("id").correct.astype(bool).rename("after"))
    fixed = int((~both.before & both.after).sum())
    broken = int((both.before & ~both.after).sum())
    p = binomtest(fixed, fixed + broken, 0.5).pvalue if fixed + broken else 1.0
    return dict(fixed=fixed, broken=broken, net=fixed - broken, mcnemar_p=float(p))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default=str(ROOT / "linear_probe/configs/intervene.yaml"))
    ap.add_argument("-o", "--override", action="append", default=[], help="dotted key=value (YAML-parsed)")
    args = ap.parse_args()
    cfg = load_config(args.config, args.override)
    base_cfg = load_config(ROOT / cfg["baseline_config"], [])   # splits, prompt, decoding: must match the baseline
    cmd = " ".join(shlex.quote(a) for a in [sys.executable] + sys.argv)
    print("launch:", cmd)

    # --- Directions: the probe's, and a random control of the same dimension ---------------------
    probe_dir = ROOT / cfg["probe_dir"]
    probe_results = json.load(open(probe_dir / "results.json"))
    layer = probe_results["best"]["layer"] if cfg["layer"] == "best" else int(cfg["layer"])
    u_probe = torch.load(probe_dir / "directions.pt")[cfg["pool"]][layer]
    u_random = R.random_direction(u_probe.numel(), seed=cfg["random_seed"])
    directions = {"probe": u_probe, "random": u_random}

    # --- Model, problems, and the scale for alpha ------------------------------------------------
    out = ROOT / cfg["out_dir"] / cfg["run_name"]
    out.mkdir(parents=True, exist_ok=True)
    model, tok = R.load_model(base_cfg["model_id"])
    splits = {b: R.load_benchmark(b, ROOT / base_cfg["splits_file"]) for b in base_cfg["benchmarks"]}
    prompt_variant = base_cfg["prompt_variant"]
    norm = median_token_norm(model, tok, layer, splits["gsm8k"]["dev"], "gsm8k", prompt_variant)
    auroc = probe_results["results"][f"{cfg['pool']}/L{layer}"]["cv_auroc"]
    print(f"layer {layer} (probe CV AUROC {auroc:.3f}), median per-token activation norm {norm:.2f}", flush=True)
    run_meta = dict(cfg, command=cmd, layer=layer, resid_norm=norm, gen=base_cfg["gen"], prompt_variant=prompt_variant)
    json.dump(run_meta, open(out / "config.json", "w"), indent=2)

    def edits_for(condition: str, alpha_frac):
        """The {layer: edit function} dict for a condition (None = no edit)."""
        if condition == "none":
            return None
        if condition in STEER:
            which, sign = STEER[condition]
            return {layer: R.steer(directions[which], sign * alpha_frac * norm)}
        return {layer: R.ablate(directions[ABLATE[condition]])}

    # --- Run every condition on every eval set ---------------------------------------------------
    summary = {}
    for ev in cfg["evals"]:
        bench, split = ev["benchmark"], ev["split"]
        items = splits[bench][split][: ev["n"]]
        # no edit first (everything is compared to it), then ablations (no strength), then steering per strength
        runs = [("none", None)]
        runs += [(c, None) for c in cfg["conditions"] if c in ABLATE]
        runs += [(c, f) for f in cfg["alpha_fracs"] for c in cfg["conditions"] if c in STEER]

        for condition, alpha_frac in runs:
            name = f"{bench}_{split}__{condition}" + (f"_a{alpha_frac}" if alpha_frac is not None else "")
            if not (out / f"{name}.jsonl").exists():          # resumable: finished conditions are skipped
                print(f"[{name}] {len(items)} problems ...", flush=True)
                rows, gens = R.run_benchmark(model, tok, bench, items, prompt_variant=prompt_variant,
                                             edits=edits_for(condition, alpha_frac), positions=cfg["positions"],
                                             **base_cfg["gen"])
                R.save_log(str(out / name), pd.DataFrame(rows), gens, tok,
                           dict(run_meta, benchmark=bench, split=split, condition=condition, alpha_frac=alpha_frac,
                                alpha=None if alpha_frac is None else alpha_frac * norm))

            df = pd.read_json(out / f"{name}.jsonl", lines=True)
            s = dict(benchmark=bench, condition=condition, alpha_frac=alpha_frac, n=len(df),
                     accuracy=float(df.correct.mean()), mean_resp_tokens=float(df.n_resp_tokens.mean()),
                     truncated=float(df.truncated.mean()))
            if condition != "none":
                s.update(flips(pd.read_json(out / f"{bench}_{split}__none.jsonl", lines=True), df))
            summary[name] = s
            print(f"[{name}] {s}", flush=True)
            json.dump(summary, open(out / "summary.json", "w"), indent=2)   # rewritten after every condition

    print(pd.DataFrame(summary.values()).to_string())


if __name__ == "__main__":
    main()
