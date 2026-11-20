"""DeepSeek-R1-Distill-Llama-8B helpers for the test-time reasoning intervention experiment.

Builds on sae_env (hooks, batched chat generation, left padding, logging); those helpers are
model-agnostic, so they're reused here rather than copied. What this module adds:

- model loading + candidate layers (LAYERS, zero-based decoder-layer indices; resid_post = layer output)
- multi-layer activation extraction (`extract_activations`)
- the intervention hook interface (`intervene`, `steer`, `ablate`, `random_direction`)
- GSM8K / HumanEval+ loading, prompting, and grading (`load_benchmark`, `run_benchmark`)

Intervention runs should go through `run_benchmark(..., edits=...)` so they use exactly the
same prompts, batching, and decoding settings as the baseline.
"""
import random
import re
import subprocess
import sys
import tempfile
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from typing import Callable, Dict, Iterable, List, Optional

import torch

import sae_env
from sae_env import _pad_left, capture_resid_post, decoder_layers, save_log  # noqa: F401 (re-exported)

MODEL_ID = "deepseek-ai/DeepSeek-R1-Distill-Llama-8B"
LAYERS = [4, 19, 29, 30]  # zero-based; 32 layers total. 19 = layer used in "I Have Covered All the Bases Here"

Edit = Callable[[torch.Tensor], torch.Tensor]  # [B, T, d] resid -> [B, T, d] resid


def load_model(model_id: str = MODEL_ID, dtype=torch.bfloat16, device: str = "cuda"):
    """Load model + tokenizer onto one device (sae_env.load_gemma is model-agnostic despite its name)."""
    return sae_env.load_gemma(model_id, dtype=dtype, device=device)


# --------------------------------------------------------------------------------------------
# Activation extraction
# --------------------------------------------------------------------------------------------

POOLS = ("prompt_last", "resp_mean", "resp_last")


@torch.no_grad()
def extract_activations(model, tok, gens: List[dict], layers: Iterable[int] = LAYERS,
                        pools: Iterable[str] = ("prompt_last", "resp_mean"), batch_size: int = 4
                        ) -> Dict[str, Dict[int, torch.Tensor]]:
    """Re-run prompt+response through the model and pool resid_post at several layers in one pass.

    gens: sae_env.generate()-style dicts (prompt_ids, resp_ids).
    pools: 'prompt_last' = last prompt token (the token after `<think>\\n`, before any generation;
           what the model "knows" before reasoning), 'resp_mean' = mean over response tokens,
           'resp_last' = last response token.
    Returns {pool: {layer: [N, d_model] fp32 CPU tensor}} in input order.

    Runs the decoder stack only (model.model), skipping lm_head: full-vocab logits for a batch
    of multi-thousand-token reasoning traces would not fit next to the model on a 23 GB GPU.
    """
    layers, pools = list(layers), list(pools)
    assert set(pools) <= set(POOLS), f"unknown pool in {pools}; choose from {POOLS}"
    out = {p: {l: [None] * len(gens) for l in layers} for p in pools}
    order = sorted(range(len(gens)), key=lambda i: -(len(gens[i]["prompt_ids"]) + len(gens[i]["resp_ids"])))
    for s in range(0, len(order), batch_size):
        idx = order[s:s + batch_size]
        ids, mask = _pad_left([gens[i]["prompt_ids"] + gens[i]["resp_ids"] for i in idx], tok.pad_token_id, model.device)
        pos = (mask.cumsum(-1) - 1).clamp(min=0)
        with capture_resid_post(model, layers) as buf:
            model.model(input_ids=ids, attention_mask=mask, position_ids=pos, use_cache=False)
        T = ids.shape[1]
        for l in layers:
            acts = buf[l][0]
            for b, i in enumerate(idx):
                n = len(gens[i]["resp_ids"])
                for p in pools:
                    if p == "prompt_last":
                        v = acts[b, T - n - 1]
                    elif p == "resp_last":
                        v = acts[b, T - 1]
                    else:
                        v = acts[b, T - max(n, 1):].float().mean(0)
                    out[p][l][i] = v.float().cpu()
    return {p: {l: torch.stack(v) for l, v in d.items()} for p, d in out.items()}


# --------------------------------------------------------------------------------------------
# Intervention hook interface
# --------------------------------------------------------------------------------------------

@contextmanager
def intervene(model, edits: Optional[Dict[int, Edit]], positions: str = "all"):
    """Apply `edits` = {layer: fn(resid)->resid} to resid_post (decoder-layer output) while active.

    Works for forward() and generate(). positions:
      'all'       - every position, prompt included (prefill and each decode step)
      'generated' - only decode steps (T == 1 forward calls), i.e. tokens the model generates;
                    the prompt prefill is left untouched
    edits=None / {} is a no-op, so baseline and intervention runs share one code path.
    """
    assert positions in ("all", "generated"), positions
    mods, handles = decoder_layers(model), []
    for l, fn in (edits or {}).items():
        def hook(_m, _i, out, fn=fn):
            h = out[0] if isinstance(out, tuple) else out
            if positions == "generated" and h.shape[1] != 1:
                return None
            h = fn(h)
            return (h,) + tuple(out[1:]) if isinstance(out, tuple) else h

        handles.append(mods[l].register_forward_hook(hook))
    try:
        yield
    finally:
        for h in handles:
            h.remove()


def _unit(v: torch.Tensor) -> torch.Tensor:
    v = v.detach().float().flatten()
    return v / v.norm()


def steer(direction: torch.Tensor, alpha: float) -> Edit:
    """resid += alpha * unit(direction). alpha is in resid-norm units (compare to typical ||resid|| at that layer)."""
    u = _unit(direction)

    def fn(h):
        return h + (alpha * u).to(h.dtype).to(h.device)

    return fn


def ablate(direction: torch.Tensor) -> Edit:
    """Project the direction out of resid: resid -= (resid . u) u (directional ablation)."""
    u = _unit(direction)

    def fn(h):
        uu = u.to(h.device)
        return (h - (h.float() @ uu).unsqueeze(-1) * uu).to(h.dtype)

    return fn


def random_direction(d_model: int, seed: int, like: Optional[torch.Tensor] = None) -> torch.Tensor:
    """Gaussian random direction for the random-direction control; scaled to ||like|| if given."""
    g = torch.Generator().manual_seed(seed)
    v = torch.randn(d_model, generator=g)
    return v / v.norm() * (like.norm() if like is not None else 1.0)


# --------------------------------------------------------------------------------------------
# Benchmarks: GSM8K (math) and HumanEval+ (code)
# --------------------------------------------------------------------------------------------

# 'cot' = DeepSeek's recommended prompt; 'plain' drops the step-by-step instruction (input ablation).
PROMPTS = {
    "gsm8k": {
        "cot": "{question}\nPlease reason step by step, and put your final answer within \\boxed{{}}.",
        "plain": "{question}\nPut your final answer within \\boxed{{}}.",
    },
    "humanevalplus": {
        "cot": "Complete the following Python function. Please reason step by step, then return the complete "
               "function in a ```python block.\n\n```python\n{question}```",
        "plain": "Complete the following Python function. Return the complete function in a ```python block."
                 "\n\n```python\n{question}```",
    },
}


# evalplus/humanevalplus (HF port) test for HumanEval/32 does `_poly(*candidate(*inp), inp)` on a float
# return value, so even the canonical solution fails; excluded (163 problems remain).
HUMANEVALPLUS_BROKEN = {"HumanEval/32"}


def load_benchmark(name: str, n_dev: int, n_test: int, seed: int = 0) -> Dict[str, List[dict]]:
    """Deterministic, disjoint dev/test splits: {'dev': [...], 'test': [...]}, each item a dict with
    id, question (raw problem text), and grading fields. dev is for layer/probe/strength selection,
    test is held out for the final comparison."""
    from datasets import load_dataset

    if name == "gsm8k":
        ds = load_dataset("openai/gsm8k", "main", split="test")
        items = [dict(id=f"gsm8k/{i}", question=r["question"], answer=r["answer"].split("####")[-1].strip().replace(",", ""))
                 for i, r in enumerate(ds)]
    elif name == "humanevalplus":
        ds = load_dataset("evalplus/humanevalplus", split="test")
        items = [dict(id=r["task_id"], question=r["prompt"], test=r["test"], entry_point=r["entry_point"])
                 for r in ds if r["task_id"] not in HUMANEVALPLUS_BROKEN]
    else:
        raise ValueError(name)
    assert n_dev + n_test <= len(items), f"{name}: n_dev+n_test={n_dev + n_test} > {len(items)} problems"
    perm = list(range(len(items)))
    random.Random(seed).shuffle(perm)
    return {"dev": [items[i] for i in perm[:n_dev]], "test": [items[i] for i in perm[n_dev:n_dev + n_test]]}


def final_region(text: str) -> str:
    """Text after the last </think> (the model's final answer); whole text if thinking never closed."""
    return text.rsplit("</think>", 1)[-1] if "</think>" in text else text


def _last_boxed(t: str) -> Optional[str]:
    i = t.rfind("\\boxed{")
    if i < 0:
        return None
    j, depth = i + len("\\boxed{"), 1
    for k in range(j, len(t)):
        depth += {"{": 1, "}": -1}.get(t[k], 0)
        if depth == 0:
            return t[j:k]
    return None


def _to_num(s: Optional[str]) -> Optional[float]:
    if s is None:
        return None
    s = re.sub(r"\\text\{[^}]*\}|\\[a-z]+|[$,%\s]", "", s)
    m = re.findall(r"-?\d+(?:\.\d+)?", s)
    return float(m[-1]) if m else None


def grade_gsm8k(item: dict, text: str) -> dict:
    """Exact numeric match of the last \\boxed{} in the final answer (fallback: last number there)."""
    region = final_region(text)
    boxed = _last_boxed(region)
    pred = _to_num(boxed) if boxed is not None else _to_num(region)
    return dict(pred=pred, gold=float(item["answer"]), correct=pred is not None and abs(pred - float(item["answer"])) < 1e-6)


def extract_code(text: str) -> str:
    """Last ```python block of the final answer; falls back to the whole final region."""
    blocks = re.findall(r"```(?:python|py)?\n(.*?)```", final_region(text), re.S)
    return blocks[-1] if blocks else final_region(text)


def _run(program: str, timeout: float) -> bool:
    with tempfile.TemporaryDirectory() as d:  # HumanEval+ tests are too large for `python -c`
        with open(f"{d}/prog.py", "w") as f:
            f.write(program)
        try:
            return subprocess.run([sys.executable, "prog.py"], cwd=d, capture_output=True, timeout=timeout).returncode == 0
        except subprocess.TimeoutExpired:
            return False


def grade_humanevalplus(items: List[dict], texts: List[str], timeout: float = 30.0, workers: int = 4) -> List[dict]:
    """Run each extracted function against the HumanEval+ (base + plus) tests in a subprocess.
    The prompt's import lines are prepended, since models often drop them when restating the function."""
    def one(item, text):
        code = extract_code(text)
        imports = "\n".join(l for l in item["question"].splitlines() if re.match(r"(from\s+[\w.]+\s+import\s|import\s+[\w.]+)", l))
        program = f"{imports}\n{code}\n\n{item['test']}\n\ncheck({item['entry_point']})\n"
        return dict(code=code, correct=_run(program, timeout))

    with ThreadPoolExecutor(workers) as ex:
        return list(ex.map(one, items, texts))


@torch.no_grad()
def run_benchmark(model, tok, name: str, items: List[dict], prompt_variant: str = "cot",
                  max_new_tokens: int = 2048, batch_size: int = 8, do_sample: bool = False, temperature: float = 0.6,
                  edits: Optional[Dict[int, Edit]] = None, positions: str = "all"):
    """Generate + grade one benchmark split, optionally under an intervention.

    Returns (rows, gens): rows = one dict per item (id, correct, n_resp_tokens, truncated, ...),
    gens = sae_env.generate() output (prompt_ids, resp_ids, text) for logging/activation extraction.
    """
    import time

    tmpl = PROMPTS[name][prompt_variant]
    convs = [[{"role": "user", "content": tmpl.format(question=it["question"])}] for it in items]
    # Same batching as one sae_env.generate() call (longest prompts first), but batch by batch for progress output.
    order = sorted(range(len(convs)), key=lambda i: -len(sae_env._prompt_ids(tok, convs[i])))
    gens, t0 = [None] * len(convs), time.time()
    with intervene(model, edits, positions):
        for s in range(0, len(order), batch_size):
            idx = order[s:s + batch_size]
            for i, g in zip(idx, sae_env.generate(model, tok, [convs[i] for i in idx], max_new_tokens=max_new_tokens,
                                                  batch_size=batch_size, do_sample=do_sample, temperature=temperature)):
                gens[i] = g
            print(f"  [{name}] {s + len(idx)}/{len(convs)} generated, {time.time() - t0:.0f}s", flush=True)
    graded = [grade_gsm8k(it, g["text"]) for it, g in zip(items, gens)] if name == "gsm8k" \
        else grade_humanevalplus(items, [g["text"] for g in gens])
    rows = []
    for it, g, gr in zip(items, gens, graded):
        rows.append(dict(id=it["id"], benchmark=name, prompt_variant=prompt_variant, **gr,
                         n_resp_tokens=len(g["resp_ids"]), truncated=len(g["resp_ids"]) >= max_new_tokens,
                         closed_think="</think>" in g["text"]))
    return rows, gens
