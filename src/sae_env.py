"""Gemma3-4B-IT + Gemma Scope 2 helpers.

The hook site and the SAE are coupled: SAE `layer_L` was trained on the output of
decoder layer L (resid_post), with BOS prepended and no activation normalization.
"""
from contextlib import contextmanager
from pathlib import Path

import torch
from dotenv import load_dotenv

load_dotenv(Path(__file__).resolve().parent.parent / ".env")  # HF_TOKEN for the gated Gemma repo

MODEL_ID = "google/gemma-3-4b-it"
SAE_RELEASE = "gemma-scope-2-4b-it-res"  # SAELens id; layers 9/17/22/29; widths 16k/65k/262k/1m; l0 small/medium/big


def load_gemma(model_id=MODEL_ID, dtype=torch.bfloat16, device="cuda"):
    from transformers import AutoModelForCausalLM, AutoTokenizer

    if str(device).startswith("cuda"):
        assert torch.cuda.is_available(), "CUDA not available; refusing to serve the LLM on CPU"
    tok = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(model_id, dtype=dtype, device_map=device).eval()
    devs = {p.device.type for p in model.parameters()}
    assert devs == {str(device).split(":")[0]}, f"model parameters on {devs}, expected {device}"
    print(f"model on {model.device}, {torch.cuda.memory_allocated() / 1e9:.1f} GB allocated" if devs == {"cuda"} else f"model on {devs}")
    return model, tok


def decoder_layers(model):
    inner = model.model
    return (inner.language_model if hasattr(inner, "language_model") else inner).layers


def chat_inputs(tok, messages, add_generation_prompt=True, device="cuda"):
    """Tokenize via the chat template (which already contains <bos>; don't add special tokens again)."""
    enc = tok.apply_chat_template(
        messages, add_generation_prompt=add_generation_prompt, tokenize=True, return_dict=True, return_tensors="pt"
    )
    return {k: v.to(device) for k, v in enc.items()}


@contextmanager
def capture_resid_post(model, layers):
    """Capture resid_post (decoder-layer output) at `layers`. Works around forward() and generate().

    Yields {layer: [tensor [B, T_step, d], ...]}; use `stack(buf, layer)` for [B, T_total, d].
    During generate() the first entry is the prefill, then one T=1 entry per decoded token
    (the final generated token is never fed back, so it has no activation).
    """
    layers = [layers] if isinstance(layers, int) else list(layers)
    mods, buf, handles = decoder_layers(model), {l: [] for l in layers}, []
    for l in layers:
        def hook(_m, _i, out, l=l):
            buf[l].append((out[0] if isinstance(out, tuple) else out).detach())

        handles.append(mods[l].register_forward_hook(hook))
    try:
        yield buf
    finally:
        for h in handles:
            h.remove()


def stack(buf, layer):
    return torch.cat(buf[layer], dim=1)


@torch.no_grad()
def get_resid_post(model, inputs, layer):
    """Single forward pass; returns [B, T, d_model] resid_post at `layer`."""
    with capture_resid_post(model, layer) as buf:
        model(**inputs)
    return stack(buf, layer)


def load_sae(layer, width="16k", l0="medium", device="cuda", release=SAE_RELEASE):
    from sae_lens import SAE

    return SAE.from_pretrained(release=release, sae_id=f"layer_{layer}_width_{width}_l0_{l0}", device=device)


@torch.no_grad()
def encode(sae, acts):
    return sae.encode(acts.to(device=sae.W_enc.device, dtype=sae.W_enc.dtype))


@torch.no_grad()
def sae_stats(sae, acts, skip_bos=True):
    """L0 and fraction of variance explained, excluding the BOS position (huge-norm outlier) by default."""
    x = acts[:, 1:] if skip_bos else acts
    x = x.to(device=sae.W_enc.device, dtype=sae.W_enc.dtype)
    z = sae.encode(x)
    xh = sae.decode(z)
    fve = 1 - ((x - xh) ** 2).sum() / ((x - x.mean(dim=(0, 1), keepdim=True)) ** 2).sum()
    return {"l0": (z > 0).float().sum(-1).mean().item(), "fve": fve.item()}


def neuronpedia_url(layer, width, idx, model="gemma-3-4b-it"):
    return f"https://www.neuronpedia.org/{model}/{layer}-gemmascope-2-res-{width}/{idx}"


def top_features(z, k=10, skip_bos=True):
    """z: [B, T, d_sae]; mean activation over non-BOS positions of batch -> (values, indices) of top k."""
    zz = z[:, 1:] if skip_bos else z
    return zz.float().mean(dim=(0, 1)).topk(k)


def _prompt_ids(tok, conv):
    ids = tok.apply_chat_template(conv, add_generation_prompt=True, tokenize=True, return_dict=True)["input_ids"]
    return list(ids[0] if isinstance(ids[0], (list, tuple)) else ids)


def _pad_left(seqs, pad_id, device):
    L = max(map(len, seqs))
    ids = torch.full((len(seqs), L), pad_id, dtype=torch.long)
    mask = torch.zeros((len(seqs), L), dtype=torch.long)
    for i, s in enumerate(seqs):
        ids[i, L - len(s):] = torch.tensor(s)
        mask[i, L - len(s):] = 1
    return ids.to(device), mask.to(device)


@torch.no_grad()
def generate(model, tok, convs, max_new_tokens=256, batch_size=16, do_sample=False, temperature=1.0):
    """Batched chat generation. convs: list of message lists. Returns [{prompt_ids, resp_ids, text}] in input order."""
    pids = [_prompt_ids(tok, c) for c in convs]
    eos = model.generation_config.eos_token_id
    eos = {eos} if isinstance(eos, int) else set(eos or [tok.eos_token_id])
    order = sorted(range(len(pids)), key=lambda i: -len(pids[i]))
    out = [None] * len(pids)
    kw = dict(do_sample=True, temperature=temperature) if do_sample else dict(do_sample=False)
    for s in range(0, len(order), batch_size):
        idx = order[s:s + batch_size]
        ids, mask = _pad_left([pids[i] for i in idx], tok.pad_token_id, model.device)
        gen = model.generate(input_ids=ids, attention_mask=mask, max_new_tokens=max_new_tokens,
                             pad_token_id=tok.pad_token_id, **kw)
        for j, i in enumerate(idx):
            r = gen[j, ids.shape[1]:].tolist()
            out[i] = r[: next((k for k, t in enumerate(r) if t in eos), len(r))]
    return [{"prompt_ids": p, "resp_ids": r, "text": tok.decode(r, skip_special_tokens=True)} for p, r in zip(pids, out)]


@torch.no_grad()
def pooled_features(model, tok, gens, layer, sae=None, pool="resp_mean", batch_size=8):
    """Re-run prompt+response through the model and pool activations at `layer`.

    pool: 'resp_mean' (mean over response tokens), 'resp_last' (last response token),
          'prompt_last' (last prompt token, i.e. the generation-prompt token before the answer).
    SAE features are pooled the same way, after encoding per token. Returns
    {'dense': [N, d_model] fp32, 'sae': [N, d_sae] fp16 or None}, on CPU, in input order.
    """
    dense, feats = [None] * len(gens), [None] * len(gens)
    order = sorted(range(len(gens)), key=lambda i: -(len(gens[i]["prompt_ids"]) + len(gens[i]["resp_ids"])))
    for s in range(0, len(order), batch_size):
        idx = order[s:s + batch_size]
        ids, mask = _pad_left([gens[i]["prompt_ids"] + gens[i]["resp_ids"] for i in idx], tok.pad_token_id, model.device)
        pos = (mask.cumsum(-1) - 1).clamp(min=0)
        acts = get_resid_post(model, {"input_ids": ids, "attention_mask": mask, "position_ids": pos}, layer)
        T = acts.shape[1]
        for b, i in enumerate(idx):
            n = len(gens[i]["resp_ids"])
            rows = {"resp_mean": acts[b, T - max(n, 1):], "resp_last": acts[b, T - 1:], "prompt_last": acts[b, T - n - 1: T - n]}[pool]
            red = (lambda x: x.float().mean(0)) if pool == "resp_mean" else (lambda x: x.float()[0])
            dense[i] = red(rows).cpu()
            if sae is not None:
                feats[i] = red(encode(sae, rows)).half().cpu()
    return {"dense": torch.stack(dense), "sae": torch.stack(feats) if sae is not None else None}


def feature_contrast(feats, fail, k=15, layer=None, width=None):
    """Top-k features (or dense dims) separating failure from non-failure by |Cohen's d|. feats: [N, F]."""
    import pandas as pd

    f, fail = feats.float(), torch.as_tensor(fail, dtype=torch.bool)
    a, b = f[fail], f[~fail]
    assert len(a) > 1 and len(b) > 1, f"need >=2 examples per group, got {len(a)} failures / {len(b)} non-failures"
    d = (a.mean(0) - b.mean(0)) / (((a.var(0) + b.var(0)) / 2).sqrt() + 1e-6)
    top = d.abs().topk(k).indices
    df = pd.DataFrame({"feature": top.tolist(), "cohens_d": d[top].tolist(),
                       "mean_fail": a.mean(0)[top].tolist(), "mean_ok": b.mean(0)[top].tolist()})
    if layer is not None:
        df["url"] = [neuronpedia_url(layer, width, i) for i in df.feature]
    return df


def save_log(prefix, df, gens, tok, meta):
    """Write every example to `{prefix}.jsonl` (exact decoded prompt, response, label, per-row metadata) plus `{prefix}.meta.json`."""
    import datetime, json, os

    os.makedirs(os.path.dirname(prefix), exist_ok=True)
    with open(prefix + ".jsonl", "w") as f:
        for r, g in zip(df.to_dict("records"), gens):
            r = {k: (None if isinstance(v, float) and v != v else v) for k, v in r.items()}
            r.update(prompt=tok.decode(g["prompt_ids"]), response=g["text"])
            f.write(json.dumps(r, default=str) + "\n")
    meta = dict(meta, n_examples=len(df), written=datetime.datetime.now().isoformat(timespec="seconds"))
    json.dump(meta, open(prefix + ".meta.json", "w"), indent=2)
