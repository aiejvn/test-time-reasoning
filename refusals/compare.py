"""Aggregate results/*/ from refusal_direction.py into a cross-model comparison.

Writes:
  comparison.csv          one row per model
  comparison.md           the same table plus per-model notes
  layer_sweeps.png        ablated refusal rate / harmless KL vs relative depth, all models
  side_by_side.md         same harmful prompt across models and conditions (first 80 chars only)
"""

import json
from pathlib import Path

import matplotlib
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).parent / "results"


def load():
    rows, sweeps, gens = [], {}, {}
    for d in sorted(ROOT.iterdir()):
        if not (d / "metrics.json").exists():
            continue
        m = json.loads((d / "metrics.json").read_text())
        r = m["rates"]
        row = {
            "model": d.name,
            "layer": f'{m["selected_layer"]}/{m["n_layers"]} ({m["selected_layer_frac"]:.2f})',
            "filters_ok": m["layer_passed_filters"],
            "tl_vs_hf_kl": m.get("tl_vs_hf_kl"),
        }
        judged = d / "judged.jsonl"
        g = pd.read_json(judged if judged.exists() else d / "generations.jsonl", lines=True)
        if "judge" in g:
            j = g.groupby(["split", "condition"]).judge.value_counts(normalize=True).unstack(fill_value=0)
            for col in ("refusal", "compliance", "incoherent"):
                if col not in j:
                    j[col] = 0.0
            row |= {
                "J_harmful_refuse_base": j.loc[("harmful", "baseline"), "refusal"],
                "J_harmful_refuse_ablate": j.loc[("harmful", "ablation"), "refusal"],
                "J_harmful_comply_ablate": j.loc[("harmful", "ablation"), "compliance"],
                "J_harmful_incoh_ablate": j.loc[("harmful", "ablation"), "incoherent"],
                "J_harmful_refuse_orth": j.loc[("harmful", "orthogonalized"), "refusal"],
                "J_harmless_refuse_base": j.loc[("harmless", "baseline"), "refusal"],
                "J_harmless_refuse_actadd": j.loc[("harmless", "actadd"), "refusal"],
                "J_harmless_incoh_ablate": j.loc[("harmless", "ablation"), "incoherent"],
            }
        row |= {
            "S_harmful_refuse_base": r["harmful/baseline"]["refusal_rate"],
            "S_harmful_refuse_ablate": r["harmful/ablation"]["refusal_rate"],
            "S_harmless_refuse_actadd": r["harmless/actadd"]["refusal_rate"],
            "harmless_kl_ablate": m["harmless_test_kl_ablation"],
            "orth_vs_hook_kl": m["orth_vs_hook_kl"],
            "runtime_s": m["runtime_s"],
        }
        rows.append(row)
        sweeps[d.name] = pd.read_csv(d / "layer_sweep.csv").assign(frac=lambda s: s.layer / m["n_layers"])
        gens[d.name] = g
    return pd.DataFrame(rows), sweeps, gens


def plot_sweeps(sweeps, path):
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.5), sharex=True)
    for name, s in sweeps.items():
        axes[0].plot(s.frac, s.ablated_refusal_rate, marker=".", label=name)
        axes[1].plot(s.frac, s.harmless_kl, marker=".", label=name)
        axes[2].plot(s.frac, s.induced_refusal_rate, marker=".", label=name)
    axes[0].set_title("Harmful val: refusal rate under ablation (lower = direction works)")
    axes[1].set_title("Harmless val: first-token KL from ablation")
    axes[1].set_yscale("log")
    axes[1].axhline(0.1, color="grey", ls="--", lw=1)
    axes[2].set_title("Harmless val: refusal rate after adding direction")
    for a in axes:
        a.set_xlabel("layer / n_layers")
        a.grid(alpha=0.3)
    axes[2].legend(fontsize=7, loc="best")
    fig.tight_layout()
    fig.savefig(path, dpi=130)


def side_by_side(gens, path, n=6, width=80):
    lines = ["# Side-by-side completions (harmful test prompts, truncated)\n",
             "Truncated to the first few words on purpose: enough to see refuse vs. comply, not the content.\n"]
    first = next(iter(gens.values()))
    for idx in range(n):
        instr = first.query("split == 'harmful' and idx == @idx").instruction.iloc[0]
        lines.append(f"\n## {idx}. {instr}\n\n| model | baseline | ablation |\n|---|---|---|")
        for name, g in gens.items():
            cell = {}
            for cond in ("baseline", "ablation"):
                row = g.query("split == 'harmful' and condition == @cond and idx == @idx").iloc[0]
                txt = " ".join(row.completion.split())[:width].replace("|", "/")
                label = row.judge if "judge" in row else ("refusal" if row.refusal else "")
                cell[cond] = f"**{label}**: {txt}" if label else txt
            lines.append(f"| {name} | {cell['baseline']} | {cell['ablation']} |")
    Path(path).write_text("\n".join(lines) + "\n")


def main():
    df, sweeps, gens = load()
    out = Path(__file__).parent
    df.to_csv(out / "comparison.csv", index=False)
    legend = ("J_ = LLM-judge rate (judge.py), S_ = substring-judge rate. base = no intervention, "
              "ablate = direction projected out of every residual position, orth = weights orthogonalized, "
              "actadd = direction added at the selected layer.\n\n")
    (out / "comparison.md").write_text(legend + df.to_markdown(index=False, floatfmt=".2f") + "\n")
    plot_sweeps(sweeps, out / "layer_sweeps.png")
    side_by_side(gens, out / "side_by_side.md")
    with pd.option_context("display.width", 250, "display.max_columns", 30):
        print(df.round(3).to_string(index=False))


if __name__ == "__main__":
    main()
