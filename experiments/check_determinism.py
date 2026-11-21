"""Does greedy decoding give the same response regardless of batching?

Generates the first N GSM8K dev problems three ways, with the baseline prompt and greedy decoding:

    batched   all N in one left-padded batch (how baseline.py / intervene.py ran, batch_size 8)
    single_a  one problem per batch (batch_size 1)
    single_b  the same as single_a, run again

and reports, per problem, whether the token sequences are identical and where they first diverge.
single_a vs single_b tests run-to-run determinism; batched vs single_a tests batch dependence.

Run
---
    python experiments/check_determinism.py --n 8 --max-new-tokens 1024 --out artifacts/determinism/check.json
"""
import argparse
import json
import shlex
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
import reasoning_env as R  # noqa: E402
import sae_env  # noqa: E402


def first_divergence(a: list, b: list):
    """Index of the first differing token, or None if identical."""
    for k, (x, y) in enumerate(zip(a, b)):
        if x != y:
            return k
    return None if len(a) == len(b) else min(len(a), len(b))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--n", type=int, default=8)
    ap.add_argument("--max-new-tokens", type=int, default=1024)
    ap.add_argument("--out", default="artifacts/determinism/check.json")
    args = ap.parse_args()
    cmd = " ".join(shlex.quote(a) for a in [sys.executable] + sys.argv)

    model, tok = R.load_model()
    items = R.load_benchmark("gsm8k")["dev"][: args.n]
    convs = [[{"role": "user", "content": R.PROMPTS["gsm8k"]["cot"].format(question=it["question"])}] for it in items]
    run = lambda bs: sae_env.generate(model, tok, convs, max_new_tokens=args.max_new_tokens, batch_size=bs)  # noqa: E731
    batched, single_a, single_b = run(args.n), run(1), run(1)

    rows = []
    for it, g, a, b in zip(items, batched, single_a, single_b):
        rows.append(dict(id=it["id"],
                         single_a_vs_single_b=first_divergence(a["resp_ids"], b["resp_ids"]),
                         batched_vs_single_a=first_divergence(g["resp_ids"], a["resp_ids"]),
                         len_batched=len(g["resp_ids"]), len_single=len(a["resp_ids"])))
        print(rows[-1], flush=True)
    summary = dict(command=cmd, n=len(rows), max_new_tokens=args.max_new_tokens,
                   single_runs_identical=sum(r["single_a_vs_single_b"] is None for r in rows),
                   batched_equals_single=sum(r["batched_vs_single_a"] is None for r in rows), rows=rows)
    out = ROOT / args.out
    out.parent.mkdir(parents=True, exist_ok=True)
    json.dump(summary, open(out, "w"), indent=2)
    print({k: v for k, v in summary.items() if k != "rows"})


if __name__ == "__main__":
    main()
