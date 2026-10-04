"""Correctness + speed benchmark for the generation paths in fast_generate.py.

Reuses the direction a previous refusal_direction.py run saved (results/<slug>/direction.pt), so no
re-fitting. For each path and condition (baseline, ablation on harmful test prompts):
  - agreement with the original uncached loop at --check_len tokens (exact match, shared-prefix length)
  - wall-clock and tokens/s at each --lengths value
vLLM (path 4) runs separately in its own venv via vllm_orth.py on bench/prompts.json.

  python bench.py --model mistralai/Mistral-7B-Instruct-v0.1 --slug mistral-7b-instruct-v0.1 --lengths 512 4096
"""

import argparse
import json
import time
from pathlib import Path

import pandas as pd
import torch
from transformers import AutoModelForCausalLM

import refusal_direction as rd
from fast_generate import HFIntervention, generate_hf, generate_tl_cached


def timed(fn):
    torch.cuda.synchronize()
    t0 = time.time()
    out = fn()
    torch.cuda.synchronize()
    return out, time.time() - t0


def agreement(ref: list[str], other: list[str]) -> dict:
    exact = sum(a == b for a, b in zip(ref, other)) / len(ref)
    prefix = [len(_common_prefix(a.split(), b.split())) / max(1, len(a.split())) for a, b in zip(ref, other)]
    return {"exact_match": exact, "mean_shared_prefix_frac": sum(prefix) / len(prefix)}


def _common_prefix(a, b):
    n = 0
    while n < min(len(a), len(b)) and a[n] == b[n]:
        n += 1
    return a[:n]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--slug", required=True, help="results/<slug>/direction.pt from a prior run")
    ap.add_argument("--n", type=int, default=16)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--check_len", type=int, default=64)
    ap.add_argument("--lengths", type=int, nargs="+", default=[512])
    ap.add_argument("--loop_max_len", type=int, default=512, help="skip the O(n^2) loop above this length")
    ap.add_argument("--out", default="bench")
    ap.add_argument("--dtype", default="bfloat16", help="model dtype for both TL and HF (e.g. float32)")
    args = ap.parse_args()

    out = Path(args.out) / args.slug
    out.mkdir(parents=True, exist_ok=True)
    saved = torch.load(Path("results") / args.slug / "direction.pt")
    _, harmful_test = rd.get_harmful_instructions()
    prompts = harmful_test[: args.n]
    (out / "prompts.json").write_text(json.dumps(prompts))

    dtype = getattr(torch, args.dtype)
    model = rd.load_model(args.model, dtype)
    tokenize = rd.make_tokenize_fn(model, args.model.startswith("Qwen/Qwen3"))
    d = saved["direction"].to(model.cfg.device)
    tl_hooks = {"baseline": [], "ablation": rd.ablation_hooks(model, d)}

    hf = AutoModelForCausalLM.from_pretrained(args.model, dtype=dtype, device_map="cuda").eval()
    tok = model.tokenizer
    chat = [tokenize.format(x) for x in prompts]
    hf_iv = {"baseline": HFIntervention(hf, "none"), "ablation": HFIntervention(hf, "ablation", direction=saved["direction"])}

    paths = {
        "loop": lambda c, L: rd.generate(model, tokenize, prompts, tl_hooks[c], L, args.batch),
        "tl_cached": lambda c, L: generate_tl_cached(model, tokenize, prompts, tl_hooks[c], L, args.batch, drop_finished=False),
        "tl_cached_drop": lambda c, L: generate_tl_cached(model, tokenize, prompts, tl_hooks[c], L, args.batch, drop_finished=True),
        "hf_hooks": lambda c, L: generate_hf(hf, tok, chat, L, args.batch, hf_iv[c]),
    }

    rows, gens = [], []
    for L in [args.check_len] + args.lengths:
        for cond in ("baseline", "ablation"):
            ref = None
            for name, fn in paths.items():
                if name == "loop" and L > args.loop_max_len:
                    continue
                texts, dt = timed(lambda: fn(cond, L))
                n_tok = sum(len(tok(t, add_special_tokens=False).input_ids) for t in texts)
                row = {"max_new": L, "condition": cond, "path": name, "seconds": round(dt, 2), "new_tokens": n_tok,
                       "tok_per_s": round(n_tok / dt, 1), "refusal_rate": sum(map(rd.is_refusal, texts)) / len(texts)}
                if name == "loop":
                    ref = texts
                elif ref is not None:
                    row |= agreement(ref, texts)
                rows.append(row)
                gens += [{"max_new": L, "condition": cond, "path": name, "idx": i, "instruction": p, "completion": t}
                         for i, (p, t) in enumerate(zip(prompts, texts))]
                print(row, flush=True)
                # write both after every row so partial runs are inspectable
                pd.DataFrame(rows).to_csv(out / "bench.csv", index=False)
                pd.DataFrame(gens).to_json(out / "generations.jsonl", orient="records", lines=True)
    print(pd.DataFrame(rows).to_string(index=False))


if __name__ == "__main__":
    main()
