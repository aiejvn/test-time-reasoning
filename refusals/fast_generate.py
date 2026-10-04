"""Faster greedy generation for the refusal-direction experiments.

refusal_direction.generate() reruns the full sequence for every new token (no KV cache), which
is O(n^2) in length and impractical at thousands of tokens. Three faster paths, all greedy:

  1+2. generate_tl_cached   TransformerLens with a KV cache, dropping finished rows from the batch.
       Same hooks as before: with a cache each step runs the hooks on the new token only, and the
       cached keys/values of earlier tokens were computed with the hooks active, so ablation and
       activation addition are the same intervention as in the uncached loop.
  3.   generate_hf          plain HF model + PyTorch hooks (fused attention kernels, HF's cache),
       with the direction found in TransformerLens. Ablation is applied to every write into the
       residual stream (embeddings, attention o_proj, MLP down_proj), which leaves the residual
       stream with no component along the direction at resid_pre/mid/post of every layer, the
       same as the TL hooks.
  4.   vllm_orth.py (separate script/venv): vLLM serving the weight-orthogonalized model.

Path 1 runs in refusal_direction.py's TransformerLens model; paths 3/4 load the HF checkpoint.
"""

from __future__ import annotations

import torch

END_OF_TURN = ["<|im_end|>", "<|end|>", "<|eot_id|>", "<end_of_turn>", "<|endoftext|>", "</s>"]


def stop_token_ids(tok) -> set[int]:
    """eos plus whatever end-of-turn markers this vocab has (Phi-3 ends turns with <|end|>, not eos)."""
    vocab = tok.get_vocab()
    return ({tok.eos_token_id} | {vocab[t] for t in END_OF_TURN if t in vocab}) - {None}


def _decode(tok, rows: list[list[int]], stops: set[int]) -> list[str]:
    out = []
    for row in rows:
        cut = next((j for j, x in enumerate(row) if x in stops), len(row))
        out.append(tok.decode(row[:cut], skip_special_tokens=True).strip())
    return out


# --------------------------------------------------------------------------- 1+2: TL + KV cache


@torch.no_grad()
def generate_tl_cached(model, tokenize, instructions, fwd_hooks=(), max_new=64, batch_size=32,
                       drop_finished=True) -> list[str]:
    from transformer_lens.cache.key_value_cache import TransformerLensKeyValueCache

    tok = model.tokenizer
    stops = stop_token_ids(tok)
    stop_t = torch.tensor(sorted(stops), device=model.cfg.device)
    results: list[list[int]] = []
    for i in range(0, len(instructions), batch_size):
        toks = tokenize(instructions[i:i + batch_size]).to(model.cfg.device)
        B = toks.shape[0]
        out = [[] for _ in range(B)]
        active = torch.arange(B, device=toks.device)  # original row index of each live row
        cache = TransformerLensKeyValueCache.init_cache(model.cfg, model.cfg.device, B)
        # TL 3.0 creates the empty cache with torch.get_default_dtype() (fp32), so the first
        # torch.cat promotes bf16 keys to fp32 and attention fails on the dtype mismatch
        for e in cache.entries:
            e.past_keys, e.past_values = e.past_keys.to(model.W_E.dtype), e.past_values.to(model.W_E.dtype)
        with model.hooks(fwd_hooks=list(fwd_hooks)):
            # prefill: TL derives the left-padding mask from leading pad ids
            logits = model(toks, past_kv_cache=cache)
            for _ in range(max_new):
                nxt = logits[:, -1].argmax(-1)
                for r, t in zip(active.tolist(), nxt.tolist()):
                    out[r].append(t)
                done = torch.isin(nxt, stop_t)
                if drop_finished and done.any():
                    keep = ~done
                    if not keep.any():
                        break
                    active, nxt = active[keep], nxt[keep]
                    for e in cache.entries:
                        e.past_keys, e.past_values = e.past_keys[keep], e.past_values[keep]
                    cache.previous_attention_mask = cache.previous_attention_mask[keep]
                elif done.all():
                    break
                # explicit mask: a generated token can equal the pad id (e.g. pad == eos)
                step = nxt[:, None]
                logits = model(step, past_kv_cache=cache, attention_mask=torch.ones_like(step))
        results += out
    return _decode(tok, results, stops)


# --------------------------------------------------------------------------- 3: HF + hooks


def _residual_writers(hf_model):
    """Modules whose outputs are added into the residual stream: embeddings, attention out, MLP out."""
    inner = hf_model.model
    writers = [inner.embed_tokens]
    for layer in inner.layers:
        attn_out = getattr(layer.self_attn, "o_proj", None) or getattr(layer.self_attn, "dense", None)
        mlp_out = getattr(layer.mlp, "down_proj", None) or getattr(layer.mlp, "fc2", None)
        if attn_out is None or mlp_out is None:
            raise NotImplementedError(f"don't know this architecture's residual writers: {type(layer)}")
        writers += [attn_out, mlp_out]
    return writers


class HFIntervention:
    """Context manager installing ablation (direction) or activation addition (vector at layer) hooks."""

    def __init__(self, hf_model, mode: str, direction=None, vector=None, layer=None):
        assert mode in ("none", "ablation", "actadd")
        self.model, self.mode, self.layer = hf_model, mode, layer
        dev, dt = hf_model.device, hf_model.dtype
        self.d = None if direction is None else direction.to(dev, dt)
        self.v = None if vector is None else vector.to(dev, dt)
        self.handles = []

    def _ablate(self, _m, _inp, out):
        return out - (out @ self.d).unsqueeze(-1) * self.d

    def _add(self, _m, args, kwargs):
        # decoder layer input = resid_pre of that layer (hidden_states is the first positional arg or a kwarg)
        if args:
            return (args[0] + self.v,) + args[1:], kwargs
        kwargs["hidden_states"] = kwargs["hidden_states"] + self.v
        return args, kwargs

    def __enter__(self):
        if self.mode == "ablation":
            self.handles = [m.register_forward_hook(self._ablate) for m in _residual_writers(self.model)]
        elif self.mode == "actadd":
            layer = self.model.model.layers[self.layer]
            self.handles = [layer.register_forward_pre_hook(self._add, with_kwargs=True)]
        return self

    def __exit__(self, *exc):
        for h in self.handles:
            h.remove()
        self.handles = []


@torch.no_grad()
def generate_hf(hf_model, tok, prompts: list[str], max_new=64, batch_size=32, intervention=None) -> list[str]:
    """prompts are fully formatted chat strings (same strings TL tokenizes)."""
    stops = stop_token_ids(tok)
    tok.padding_side = "left"
    out = []
    for i in range(0, len(prompts), batch_size):
        batch = prompts[i:i + batch_size]
        add_special = not (tok.bos_token and batch[0].startswith(tok.bos_token))
        enc = tok(batch, return_tensors="pt", padding=True, add_special_tokens=add_special).to(hf_model.device)
        ctx = intervention if intervention is not None else HFIntervention(hf_model, "none")
        with ctx:
            gen = hf_model.generate(**enc, max_new_tokens=max_new, do_sample=False, eos_token_id=sorted(stops),
                                    pad_token_id=tok.pad_token_id, temperature=None, top_p=None, top_k=None)
        out += _decode(tok, gen[:, enc.input_ids.shape[1]:].tolist(), stops)
    return out


def orthogonalize_hf_(hf_model, direction):
    """Path 4 prep: bake directional ablation into the HF weights (in place).
    Linear weights are [out, in] in HF, so the residual dim is dim 0 for o_proj/down_proj and dim 1 for embeddings."""
    d = direction.to(hf_model.device, torch.float32)
    P = torch.eye(d.numel(), device=d.device) - torch.outer(d, d)
    inner = hf_model.model
    emb = inner.embed_tokens.weight
    emb.data = (emb.data.float() @ P).to(emb.dtype)
    for m in _residual_writers(hf_model)[1:]:
        m.weight.data = (P @ m.weight.data.float()).to(m.weight.dtype)
        if getattr(m, "bias", None) is not None:
            m.bias.data = (P @ m.bias.data.float()).to(m.bias.dtype)
    # untied output embeddings are a reader, not a writer; tied ones get orthogonalized with W_E
    # (harmless: reading along a direction that's never written contributes nothing)
