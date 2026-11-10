"""Shared Gemma3-4B-IT + SAE inference service.

Loads the model ONCE, plus every Gemma Scope 2 SAE this project uses (one per layer in
LAYERS, all at the same width/l0) co-resident on the GPU alongside it, and serves
generation/activation requests over HTTP. This lets several people run notebooks at the
same time against one GPU-resident copy instead of each loading their own - the model
alone is ~8GB in bf16, so three independent copies plus KV cache/activations won't fit on
a single 23GB GPU. Keeping all layers' SAEs loaded means switching which layer a notebook
analyzes is free (no reload latency), while still fitting comfortably: each 16k-width SAE
is a couple hundred MB, trivial next to the model.

Run with:  python sae_server.py   (or: uvicorn sae_server:app --host 0.0.0.0 --port 8000)

Concurrency: every GPU-touching endpoint is serialized behind a single lock. This machine
has one GPU, so that's the real throughput constraint regardless of server design - the
lock just makes concurrent requests safe (the activation-capture hooks in
sae_env.capture_resid_post are not reentrant) instead of silently corrupting each other.
"""
import io
import threading

import torch
from fastapi import FastAPI, HTTPException, Response
from pydantic import BaseModel

import sae_env as E

MODEL_ID = E.MODEL_ID
LAYERS = [9, 17, 22, 29]   # every layer this project analyzes - all kept resident alongside the model
WIDTH = "16k"
L0 = "medium"
PORT = 8000

print(f"loading {MODEL_ID} ...")
model, tok = E.load_gemma(MODEL_ID)
print(f"loading SAEs for layers {LAYERS} (width={WIDTH}, l0={L0}) ...")
saes = {layer: E.load_sae(layer, WIDTH, L0) for layer in LAYERS}

GPU_LOCK = threading.Lock()  # one GPU - serialize so concurrent requests can't interleave
                              # activation-capture hooks or race on memory

app = FastAPI(title="sae_server")


class Message(BaseModel):
    role: str
    content: str


class GenerateReq(BaseModel):
    convs: list[list[Message]]
    max_new_tokens: int = 256
    batch_size: int = 16
    do_sample: bool = False
    temperature: float = 1.0


class PooledFeaturesReq(BaseModel):
    gens: list[dict]  # [{prompt_ids, resp_ids, text}, ...] - as returned by /generate
    layer: int
    pool: str = "resp_mean"
    use_sae: bool = True
    batch_size: int = 8


class InspectReq(BaseModel):
    messages: list[Message]
    layer: int
    k: int = 10
    use_sae: bool = True


def _sae_for(layer: int, use_sae: bool):
    if not use_sae:
        return None
    if layer not in saes:
        raise HTTPException(400, f"no SAE loaded for layer {layer}; server has {sorted(saes)} (edit LAYERS in sae_server.py to add more)")
    return saes[layer]


@app.get("/info")
def info():
    """Config every client should read at startup: which (layer, width, l0) SAEs are resident,
    so everyone running notebooks at the same time picks from the same preloaded set."""
    return {"model_id": MODEL_ID, "layers": sorted(saes), "width": WIDTH, "l0": L0}


@app.post("/generate")
def generate(req: GenerateReq):
    convs = [[m.model_dump() for m in conv] for conv in req.convs]
    with GPU_LOCK:
        return E.generate(model, tok, convs, max_new_tokens=req.max_new_tokens,
                           batch_size=req.batch_size, do_sample=req.do_sample, temperature=req.temperature)


@app.post("/pooled_features")
def pooled_features(req: PooledFeaturesReq):
    sae = _sae_for(req.layer, req.use_sae)
    with GPU_LOCK:
        out = E.pooled_features(model, tok, req.gens, req.layer, sae=sae, pool=req.pool, batch_size=req.batch_size)
    buf = io.BytesIO()
    torch.save(out, buf)
    return Response(buf.getvalue(), media_type="application/octet-stream")


@app.post("/inspect")
def inspect(req: InspectReq):
    sae = _sae_for(req.layer, req.use_sae)
    messages = [m.model_dump() for m in req.messages]
    inputs = E.chat_inputs(tok, messages)
    with GPU_LOCK:
        acts = E.get_resid_post(model, inputs, req.layer)
        out = {"resid_shape": list(acts.shape)}
        if sae is not None:
            out.update(E.sae_stats(sae, acts))
            z = E.encode(sae, acts)
            vals, idx = E.top_features(z, k=req.k)
            out["top_features"] = [
                {"feature": i, "mean_activation": v, "url": E.neuronpedia_url(req.layer, WIDTH, i)}
                for v, i in zip(vals.tolist(), idx.tolist())
            ]
    return out


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=PORT)
