"""Path 4: greedy generation with vLLM on the weight-orthogonalized model (= directional ablation;
orth_vs_hook_kl in metrics.json checks that equivalence per model).

Runs in the vLLM venv (/root/kevin/.venv-vllm), not the TransformerLens one. Takes prompts from a JSON
list of instructions (bench.py writes one) and the direction from refusal_direction.py's direction.pt.
The orthogonalized checkpoint is a throwaway artifact and goes to /dev/shm.

  /root/kevin/.venv-vllm/bin/python vllm_orth.py --model mistralai/Mistral-7B-Instruct-v0.1 \
      --direction results/mistral-7b-instruct-v0.1/direction.pt --prompts bench/prompts.json \
      --max_new 4096 --out bench/vllm_4096.jsonl [--baseline]
"""

import argparse
import gc
import json
import shutil
import time
from pathlib import Path

import pandas as pd
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from fast_generate import orthogonalize_hf_, stop_token_ids


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--direction", required=True)
    ap.add_argument("--prompts", required=True, help="JSON list of instructions")
    ap.add_argument("--max_new", type=int, default=4096)
    ap.add_argument("--out", required=True)
    ap.add_argument("--baseline", action="store_true", help="also generate with the unmodified weights")
    ap.add_argument("--gpu_mem", type=float, default=0.45)
    ap.add_argument("--dtype", default="bfloat16")
    ap.add_argument("--workdir", default="/dev/shm/refusal/orth")
    args = ap.parse_args()

    from vllm import LLM, SamplingParams

    instructions = json.loads(Path(args.prompts).read_text())
    tok = AutoTokenizer.from_pretrained(args.model)
    prompts = [tok.apply_chat_template([{"role": "user", "content": x}], tokenize=False, add_generation_prompt=True)
               for x in instructions]
    params = SamplingParams(temperature=0, max_tokens=args.max_new, stop_token_ids=sorted(stop_token_ids(tok)),
                            skip_special_tokens=True)
    max_len = max(len(tok(p).input_ids) for p in prompts) + args.max_new + 8

    # bake the ablation into the weights and save a temporary checkpoint
    t0 = time.time()
    orth_dir = Path(args.workdir) / args.model.replace("/", "--")
    hf = AutoModelForCausalLM.from_pretrained(args.model, dtype=getattr(torch, args.dtype), device_map="cuda")
    orthogonalize_hf_(hf, torch.load(args.direction)["direction"])
    hf.save_pretrained(orth_dir)
    tok.save_pretrained(orth_dir)
    del hf
    gc.collect()
    torch.cuda.empty_cache()
    export_s = time.time() - t0

    runs = [("ablation_orth", str(orth_dir))] + ([("baseline", args.model)] if args.baseline else [])
    rows, timing = [], {"export_s": round(export_s, 1)}
    for cond, path in runs:
        llm = LLM(model=path, dtype=args.dtype, gpu_memory_utilization=args.gpu_mem, max_model_len=max_len, seed=0)
        t0 = time.time()
        outs = llm.generate(prompts, params, use_tqdm=False)
        dt = time.time() - t0
        n_tok = sum(len(o.outputs[0].token_ids) for o in outs)
        timing[cond] = {"gen_s": round(dt, 2), "new_tokens": n_tok, "tok_per_s": round(n_tok / dt, 1)}
        rows += [{"condition": cond, "idx": i, "instruction": x, "completion": o.outputs[0].text.strip(),
                  "n_tokens": len(o.outputs[0].token_ids)} for i, (x, o) in enumerate(zip(instructions, outs))]
        del llm
        gc.collect()
        torch.cuda.empty_cache()

    pd.DataFrame(rows).to_json(args.out, orient="records", lines=True)
    Path(args.out).with_suffix(".timing.json").write_text(json.dumps(timing, indent=2))
    shutil.rmtree(orth_dir, ignore_errors=True)
    print(json.dumps(timing, indent=2))


if __name__ == "__main__":
    main()
