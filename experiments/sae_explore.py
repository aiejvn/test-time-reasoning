"""SAE feature exploration on baseline generations: which features separate correct from wrong answers.

For one SAE (one layer): re-runs every baseline prompt+response through the model, encodes resid_post
per token, and per problem records the prompt_last code, the mean code over response tokens, and each
feature's max activation (with its position). It then reports, per benchmark and pool, the top features
by |Cohen's d| between correct and wrong answers, with Neuronpedia links and the tokens where each
feature peaks most strongly.

Token ids are recovered by re-tokenizing the logged prompt/response text (round trip checked exact
on the 2026-11-20 dev logs: every response re-tokenizes to its generated length).

Writes {out_dir}/{sae_name}/: codes.pt and summary.md.

Usage:
  python experiments/sae_explore.py --config configs/sae_explore.yaml -o sae=llamascope_l19
"""
import argparse
import collections
import json
import os
import sys
from pathlib import Path

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import torch  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "experiments"))
import reasoning_env as R  # noqa: E402
import sae_env  # noqa: E402
from baseline import load_config  # noqa: E402


def load_sae(spec, device="cuda"):
    from sae_lens import SAE

    if "release" in spec:
        sae = SAE.from_pretrained(spec["release"], spec["sae_id"], device=device)
        sae = sae[0] if isinstance(sae, tuple) else sae
    else:
        from huggingface_hub import snapshot_download

        sae = SAE.load_from_disk(f"{snapshot_download(spec['repo'])}/{spec['path']}", device=device)
    return sae.eval()


@torch.no_grad()
def encode_problems(model, tok, sae, layer, rows):
    """Per problem: prompt_last code, response-mean code, per-feature max activation and its position."""
    out = collections.defaultdict(list)
    for r in rows:
        p = tok(r["prompt"], add_special_tokens=False)["input_ids"]
        q = tok(r["response"], add_special_tokens=False)["input_ids"]
        ids = torch.tensor([p + q], device=model.device)
        with R.capture_resid_post(model, [layer]) as buf:
            model.model(input_ids=ids, use_cache=False)
        x = buf[layer][0][0, 1:].to(sae.W_enc.dtype)  # drop BOS (huge-norm outlier)
        z = sae.encode(x)
        xh = sae.decode(z).float()
        x = x.float()
        out["rel_err"].append((((x - xh) ** 2).sum(-1) / (x ** 2).sum(-1)).mean().item())
        out["cos"].append(torch.nn.functional.cosine_similarity(x, xh, dim=-1).mean().item())
        out["l0"].append((z > 0).float().sum(-1).mean().item())
        n_p = len(p) - 1
        zr = z[n_p:] if len(q) else z[-1:]
        mx, arg = z.max(0)
        out["prompt_last"].append(z[n_p - 1].half().cpu())
        out["resp_mean"].append(zr.float().mean(0).half().cpu())
        out["max"].append(mx.half().cpu())
        out["argmax"].append((arg + 1).int().cpu())  # +1: position in ids (BOS was dropped)
        out["ids"].append(p + q)
        out["n_prompt"].append(len(p))
    return {k: (torch.stack(v) if torch.is_tensor(v[0]) else v) for k, v in out.items()}


def contexts(tok, codes, f, k=5, window=6):
    """Peak-activation contexts for feature f: the k problems where its max is largest."""
    top = codes["max"][:, f].float().topk(k)
    res = []
    for v, i in zip(top.values.tolist(), top.indices.tolist()):
        if v <= 0:
            break
        pos, ids = int(codes["argmax"][i, f]), codes["ids"][i]
        res.append(dict(act=v, token=tok.decode([ids[pos]]),
                        context=tok.decode(ids[max(0, pos - window):pos]) + "[[" + tok.decode([ids[pos]]) + "]]"
                        + tok.decode(ids[pos + 1:pos + 1 + window // 2]),
                        in_prompt=pos < codes["n_prompt"][i]))
    return res


def md_cell(s):
    return s.replace("|", "\\|").replace("\n", "⏎").replace("`", "'")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default=str(ROOT / "configs/sae_explore.yaml"))
    ap.add_argument("-o", "--override", action="append", default=[])
    args = ap.parse_args()
    cfg = load_config(args.config, args.override)
    spec = cfg["saes"][cfg["sae"]]
    layer = spec["layer"]
    out = ROOT / cfg["out_dir"] / cfg["sae"]
    out.mkdir(parents=True, exist_ok=True)

    model, tok = R.load_model(cfg["model_id"])
    sae = load_sae(spec)
    src = ROOT / cfg["baseline_dir"]
    lines = [f"## {cfg['sae']}: layer {layer}, {sae.cfg.d_sae:,} features", "", spec.get("description", ""), ""]
    tops, recon = {}, []
    POOL = {"prompt_last": "last prompt token (the `<think>\\n` position, before any reasoning is generated)",
            "resp_mean": "mean over all response tokens (reasoning + answer)"}
    for tag in cfg["benchmarks"]:
        rows = [json.loads(l) for l in open(src / f"{tag}.jsonl")]
        print(f"[{cfg['sae']}] encoding {tag} ({len(rows)} problems) ...", flush=True)
        codes = encode_problems(model, tok, sae, layer, rows)
        torch.save({k: v for k, v in codes.items() if k != "ids"}, out / f"{tag}_codes.pt")
        recon.append(f"| {tag} | {len(rows)} | {sum(codes['cos']) / len(rows):.3f} | {sum(codes['rel_err']) / len(rows):.3f} "
                     f"| {sum(codes['l0']) / len(rows):.1f} |")
        correct = torch.tensor([bool(r["correct"]) for r in rows])
        for pool in cfg["pools"]:
            df = sae_env.feature_contrast(codes[pool], ~correct, k=cfg["top_k"])  # "fail" = wrong answer
            tops[(tag, pool)] = set(df.feature)
            lines += [f"### {tag}, pooled at the {POOL[pool]}", "",
                      f"n = {len(rows)} problems: {int(correct.sum())} correct, {int((~correct).sum())} wrong.", "",
                      "| feature | Cohen's d | mean act. (wrong) | mean act. (correct) | peak tokens | strongest context |",
                      "|---|---|---|---|---|---|"]
            for _, r in df.iterrows():
                f = int(r.feature)
                ctx = contexts(tok, codes, f)
                link = f"[{f}](https://www.neuronpedia.org/{spec['neuronpedia']}/{f})" if spec.get("neuronpedia") else str(f)
                toks = ", ".join(f"'{md_cell(c['token'].strip() or c['token'])}'" for c in ctx[:5])
                top = md_cell(ctx[0]["context"]) if ctx else ""
                lines.append(f"| {link} | {r.cohens_d:+.2f} | {r.mean_fail:.3f} | {r.mean_ok:.3f} | {toks} | {top} |")
            lines.append("")
    if len(cfg["benchmarks"]) == 2:
        a, b = cfg["benchmarks"]
        lines += ["### Shared features between math and code", ""]
        for pool in cfg["pools"]:
            shared = sorted(tops[(a, pool)] & tops[(b, pool)])
            lines.append(f"- {pool}: {len(shared)} of the top {cfg['top_k']} features appear in both {a} and {b}"
                         + (f" ({', '.join(map(str, shared))})" if shared else "") + ".")
        lines.append("")
    lines += ["### Reconstruction quality on our data", "",
              "How well this SAE reproduces the model's actual layer output on these generations (all tokens except BOS). "
              "Cosine 1.0 and relative error 0.0 would be perfect. L0 is the average number of features active per token.", "",
              "| benchmark | problems | mean cosine | mean relative error | mean L0 |", "|---|---|---|---|---|", *recon, ""]
    (out / "summary.md").write_text("\n".join(lines))
    print("\n".join(lines))


if __name__ == "__main__":
    main()
