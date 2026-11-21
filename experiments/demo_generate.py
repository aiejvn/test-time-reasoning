"""Generate demo responses: plain prompt / chain-of-thought / CoT + probe steering / CoT + random, batch size 1.

Same edit as linear_probe/REPORT.md step 3 (layer 29, h + 0.1 * median-norm * u, every token), read from the
heldout_half run config so it matches the recorded runs. Each (problem, condition) is one generate call with
batch size 1, so results don't depend on batching (REPORT.md audit 3a). Rows are appended to
artifacts/demo/responses.jsonl as they finish; already-done pairs are skipped.

    python experiments/demo_generate.py --ids HumanEval/143 gsm8k/216
"""
import argparse
import json
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
import reasoning_env as R  # noqa: E402

CONDITIONS = {  # name -> (prompt variant, direction or None)
    "plain": ("plain", None),
    "cot": ("cot", None),
    "steered": ("cot", "probe"),
    "random": ("cot", "random"),
}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ids", nargs="+", required=True)
    ap.add_argument("--conditions", nargs="+", default=list(CONDITIONS))
    ap.add_argument("--out", default="artifacts/demo/responses.jsonl")
    ap.add_argument("--run-config", default="artifacts/intervene/heldout_half/config.json")
    args = ap.parse_args()

    cfg = json.load(open(ROOT / args.run_config))
    layer, alpha = cfg["layer"], cfg["alpha_fracs"][0] * cfg["resid_norm"]
    u = torch.load(ROOT / cfg["probe_dir"] / "directions.pt")[cfg["pool"]][layer]
    dirs = {"probe": u, "random": R.random_direction(u.numel(), seed=cfg["random_seed"])}
    gen = dict(cfg["gen"], batch_size=1)

    out = ROOT / args.out
    out.parent.mkdir(parents=True, exist_ok=True)
    done = {(r["id"], r["condition"]) for r in map(json.loads, open(out))} if out.exists() else set()

    model, tok = R.load_model(R.MODEL_ID)
    items = {}
    for bench in ["gsm8k", "humanevalplus"]:
        for split in R.load_benchmark(bench, ROOT / "configs/splits.json").values():
            items.update({it["id"]: (bench, it) for it in split})

    for pid in args.ids:
        bench, item = items[pid]
        for cond in args.conditions:
            if (pid, cond) in done:
                continue
            variant, d = CONDITIONS[cond]
            edits = None if d is None else {layer: R.steer(dirs[d], alpha)}
            rows, gens = R.run_benchmark(model, tok, bench, [item], prompt_variant=variant, edits=edits,
                                         positions=cfg["positions"], **gen)
            row = dict(rows[0], condition=cond, layer=layer, alpha=None if d is None else alpha,
                       response=gens[0]["text"])
            with open(out, "a") as f:
                f.write(json.dumps(row) + "\n")
            print(f"{pid} {cond}: {'correct' if row['correct'] else 'wrong'} ({row['n_resp_tokens']} tokens)",
                  flush=True)


if __name__ == "__main__":
    main()
