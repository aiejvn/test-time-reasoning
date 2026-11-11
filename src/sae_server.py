"""Shared Gemma3-4B-IT + SAE inference service.

Loads ONE resident copy of the model and serves generation/activation requests over HTTP,
so several people can run notebooks at the same time against it instead of each loading
their own - the model alone is ~8.6GB in bf16, so independent copies per person don't fit
on a single 23GB GPU, and two resident copies (one plain + one inside a KV-cached serving
engine) turned out not to leave enough room either (see git history: a vLLM-backed
/generate briefly existed here and was removed - two 8.6GB weight copies plus a reserved
KV-cache budget left too little headroom and caused real OOM crashes under actual use).
Back to a single `model`/`tok`, used by /generate, /pooled_features, and /inspect alike.

SAEs are NOT preloaded. At most MAX_RESIDENT_SAES are kept in GPU memory at once - sized
for ~3 concurrent users, each typically working one layer at a time - loaded on first use
per layer and evicted least-recently-used when a new layer is requested past the cap.
This trades a reload (seconds, from the HF Hub's own disk cache) for a layer nobody's
actively using against bounded, predictable GPU memory - prefer recomputing/reloading on
demand over holding everything resident "just in case."

Run with:  python sae_server.py   (or: uvicorn sae_server:app --host 0.0.0.0 --port 8000)

Concurrency: every GPU-touching call shares one GPU_LOCK (one model, one set of hooks -
capture_resid_post's hooks are not reentrant, so forward passes must be serialized).
For /generate, GenerationScheduler still merges concurrent requests from different users
(grouped by matching decoding params) into one batched E.generate() call per dispatch, so
the lock is acquired once per *batch* instead of once per request. /pooled_features divides
a fixed default batch_size by how many /pooled_features requests are in flight right now,
so a single active user gets the full default while concurrent users each get a fair,
smaller share instead of one request starving the others - no GPU-memory estimation
involved, just an even split of a known-safe constant.
"""
import collections
import io
import queue
import threading
import time
from concurrent.futures import Future
from dataclasses import dataclass, field

import torch
from fastapi import FastAPI, HTTPException, Response
from pydantic import BaseModel

import sae_env as E

MODEL_ID = E.MODEL_ID
LAYERS = [9, 17, 22, 29]   # layers the server is willing to serve an SAE for (not all preloaded)
WIDTH = "16k"
L0 = "medium"
PORT = 8000
MAX_BATCH = 64                 # cap on conversations merged into one E.generate() call
BATCH_WINDOW_MS = 50           # how long the scheduler waits to collect more requests before dispatching
MAX_RESIDENT_SAES = 3          # sized for ~3 concurrent users, each usually pinned to one layer
DEFAULT_POOLED_BATCH_SIZE = 8  # /pooled_features' batch_size when unspecified, divided across concurrent requests

print(f"loading {MODEL_ID} ...")
model, tok = E.load_gemma(MODEL_ID)

GPU_LOCK = threading.Lock()  # one model, one set of hooks - serialize all GPU-touching calls

_active_pooled_lock = threading.Lock()
_active_pooled = 0  # how many /pooled_features requests are currently in flight (for fair batch splitting)

_sae_cache = collections.OrderedDict()  # layer -> SAE, ordered oldest -> most-recently-used
_sae_cache_lock = threading.Lock()


def _get_sae(layer):
    """Return the SAE for `layer`, loading it on demand and evicting the least-recently-used
    one past MAX_RESIDENT_SAES. Loading touches the GPU, so it happens under GPU_LOCK to avoid
    racing an in-flight generate/forward call; the cache dict itself has its own lighter lock
    so a cache hit (the common case) never has to wait on GPU_LOCK at all.
    """
    with _sae_cache_lock:
        cached = _sae_cache.get(layer)
        if cached is not None:
            _sae_cache.move_to_end(layer)
            return cached
    with GPU_LOCK:
        with _sae_cache_lock:
            cached = _sae_cache.get(layer)  # another thread may have loaded it while we waited
            if cached is not None:
                _sae_cache.move_to_end(layer)
                return cached
        sae = E.load_sae(layer, WIDTH, L0)
        with _sae_cache_lock:
            _sae_cache[layer] = sae
            _sae_cache.move_to_end(layer)
            while len(_sae_cache) > MAX_RESIDENT_SAES:
                evicted_layer, _ = _sae_cache.popitem(last=False)
                print(f"evicting SAE for layer {evicted_layer} (resident cap={MAX_RESIDENT_SAES})")
        torch.cuda.empty_cache()
    return sae


@dataclass
class _GenItem:
    conv: list
    max_new_tokens: int
    batch_size: int
    do_sample: bool
    temperature: float
    future: Future = field(default_factory=Future)


class GenerationScheduler:
    """Merges /generate requests from different users into shared batched
    E.generate() calls instead of serializing one whole request behind GPU_LOCK
    at a time. Work items are per-conversation (not per HTTP request), so two users'
    requests that happen to overlap in time land in the same batch.
    """

    def __init__(self, model, tok):
        self.model = model
        self.tok = tok
        self._q = queue.Queue()
        threading.Thread(target=self._run, daemon=True).start()

    def submit(self, conv, max_new_tokens, batch_size, do_sample, temperature):
        item = _GenItem(conv, max_new_tokens, batch_size, do_sample, temperature)
        self._q.put(item)
        return item.future

    def _run(self):
        while True:
            batch = [self._q.get()]  # blocks until at least one item is pending
            deadline = time.monotonic() + BATCH_WINDOW_MS / 1000
            while len(batch) < MAX_BATCH:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    break
                try:
                    batch.append(self._q.get(timeout=remaining))
                except queue.Empty:
                    break
            self._dispatch(batch)

    def _dispatch(self, batch):
        # E.generate() takes one set of decoding params per call, so group by them -
        # items with matching params from different requests get merged into one call.
        groups = {}
        for item in batch:
            groups.setdefault((item.max_new_tokens, item.do_sample, item.temperature), []).append(item)
        for (max_new_tokens, do_sample, temperature), items in groups.items():
            try:
                with GPU_LOCK:
                    gens = E.generate(
                        self.model, self.tok, [it.conv for it in items],
                        max_new_tokens=max_new_tokens,
                        batch_size=max(it.batch_size for it in items),
                        do_sample=do_sample, temperature=temperature,
                    )
                for it, g in zip(items, gens):
                    it.future.set_result(g)
            except Exception as e:
                for it in items:
                    it.future.set_exception(e)


scheduler = GenerationScheduler(model, tok)

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
    batch_size: int | None = None  # None -> DEFAULT_POOLED_BATCH_SIZE, split fairly across concurrent requests


class InspectReq(BaseModel):
    messages: list[Message]
    layer: int
    k: int = 10
    use_sae: bool = True


def _sae_for(layer: int, use_sae: bool):
    if not use_sae:
        return None
    if layer not in LAYERS:
        raise HTTPException(400, f"layer {layer} not served; server supports {sorted(LAYERS)} (edit LAYERS in sae_server.py to add more)")
    return _get_sae(layer)


@app.get("/info")
def info():
    """Config every client should read at startup: which (layer, width, l0) SAEs this server
    can serve, so everyone running notebooks at the same time picks from the same set. Not
    all of them are necessarily resident right now - see MAX_RESIDENT_SAES."""
    return {"model_id": MODEL_ID, "layers": sorted(LAYERS), "width": WIDTH, "l0": L0}


@app.post("/generate")
def generate(req: GenerateReq):
    convs = [[m.model_dump() for m in conv] for conv in req.convs]
    futures = [
        scheduler.submit(conv, req.max_new_tokens, req.batch_size, req.do_sample, req.temperature)
        for conv in convs
    ]
    return [f.result() for f in futures]


@app.post("/pooled_features")
def pooled_features(req: PooledFeaturesReq):
    global _active_pooled
    sae = _sae_for(req.layer, req.use_sae)
    with _active_pooled_lock:
        _active_pooled += 1
        concurrency = _active_pooled
    try:
        # No GPU-memory estimation here (that's what broke under real load) - just split a
        # known-safe fixed default evenly across however many requests are in flight right now.
        batch_size = req.batch_size if req.batch_size is not None else max(1, DEFAULT_POOLED_BATCH_SIZE // concurrency)
        with GPU_LOCK:
            out = E.pooled_features(model, tok, req.gens, req.layer, sae=sae, pool=req.pool, batch_size=batch_size)
    finally:
        with _active_pooled_lock:
            _active_pooled -= 1
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
