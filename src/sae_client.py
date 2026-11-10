"""Thin HTTP client for sae_server.py - notebooks `import sae_client as E` in place of
sae_env when running against the shared GPU service, so several people can run notebooks
concurrently without each loading their own copy of the model (see sae_server.py).

GPU-free helpers (feature_contrast, neuronpedia_url, save_log) don't need the model/SAE at
all, so they're re-exported straight from sae_env and run locally, same as before.
"""
import io

import requests
import torch

from sae_env import feature_contrast, neuronpedia_url, save_log  # noqa: F401 (re-exported for notebooks)

SERVER_URL = "http://localhost:8000"   # edit this if sae_server.py runs on a different host
_session = requests.Session()


def server_info():
    """Fetch (model_id, layers, width, l0) from the running server - the single source of
    truth for which SAEs are resident, so concurrent notebooks all pick from the same set
    instead of each assuming their own config.
    """
    r = _session.get(f"{SERVER_URL}/info", timeout=10)
    r.raise_for_status()
    info = r.json()
    print(f"connected to sae_server at {SERVER_URL}: {info}")
    return info["model_id"], info["layers"], info["width"], info["l0"]


def load_tokenizer(model_id):
    """Tokenizer only - no GPU needed, so every client loads its own cheap local copy
    instead of round-tripping every decode through the server."""
    from transformers import AutoTokenizer
    return AutoTokenizer.from_pretrained(model_id)


def generate(convs, max_new_tokens=256, batch_size=16, do_sample=False, temperature=1.0):
    r = _session.post(f"{SERVER_URL}/generate", json=dict(
        convs=convs, max_new_tokens=max_new_tokens, batch_size=batch_size,
        do_sample=do_sample, temperature=temperature,
    ), timeout=None)
    r.raise_for_status()
    return r.json()


def pooled_features(gens, layer, pool="resp_mean", use_sae=True, batch_size=8):
    r = _session.post(f"{SERVER_URL}/pooled_features", json=dict(
        gens=gens, layer=layer, pool=pool, use_sae=use_sae, batch_size=batch_size,
    ), timeout=None)
    r.raise_for_status()
    return torch.load(io.BytesIO(r.content), weights_only=False)  # trusted - our own server, same project


def inspect(messages, layer, k=10, use_sae=True):
    r = _session.post(f"{SERVER_URL}/inspect", json=dict(
        messages=messages, layer=layer, k=k, use_sae=use_sae,
    ), timeout=30)
    r.raise_for_status()
    return r.json()
