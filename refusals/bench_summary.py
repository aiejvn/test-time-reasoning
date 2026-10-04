"""Summarize bench*/<slug>/bench.csv (+ vLLM timing) into one table per run.

  python bench_summary.py            # all bench*/ dirs
"""

import json
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).parent


def main():
    tables = []
    for csv in sorted(ROOT.glob("bench*/*/bench.csv")):
        run, slug = csv.parent.parent.name, csv.parent.name
        df = pd.read_csv(csv)
        loop = df[df.path == "loop"].set_index(["max_new", "condition"]).seconds
        df["speedup_vs_loop"] = [
            loop.get((r.max_new, r.condition)) / r.seconds if (r.max_new, r.condition) in loop.index else None
            for r in df.itertuples()
        ]
        for t in csv.parent.glob("vllm_*.timing.json"):
            L = int(t.stem.split("_")[1].split(".")[0])
            for cond, v in json.loads(t.read_text()).items():
                if isinstance(v, dict):
                    c = "ablation" if cond.startswith("ablation") else cond
                    df.loc[len(df)] = {"max_new": L, "condition": c, "path": "vllm_orth" if c == "ablation" else "vllm",
                                       "seconds": v["gen_s"], "new_tokens": v["new_tokens"], "tok_per_s": v["tok_per_s"]}
        df.insert(0, "run", run)
        df.insert(1, "model", slug)
        tables.append(df)
    out = pd.concat(tables).sort_values(["model", "run", "max_new", "condition", "path"])
    cols = ["run", "model", "max_new", "condition", "path", "seconds", "tok_per_s", "speedup_vs_loop",
            "exact_match", "mean_shared_prefix_frac", "refusal_rate"]
    out = out[[c for c in cols if c in out]]
    out.to_csv(ROOT / "bench_summary.csv", index=False)
    (ROOT / "bench_summary.md").write_text(out.to_markdown(index=False, floatfmt=".2f") + "\n")
    print(out.to_string(index=False, float_format=lambda x: f"{x:.2f}"))


if __name__ == "__main__":
    main()
