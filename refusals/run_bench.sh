#!/usr/bin/env bash
# Waits for the GPU to be free, then benchmarks the generation paths (see bench.py / vllm_orth.py).
set -u
cd "$(dirname "$0")"
export HF_HOME=/dev/shm/refusal/hf HF_HUB_ENABLE_HF_TRANSFER=1
TL=/root/kevin/.venv/bin/python
VLLM=/root/kevin/.venv-vllm/bin/python

until [ "$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits)" -lt 20000 ]; do sleep 60; done
echo "GPU free $(date +%T)"

# speed: Mistral-7B at 512 and 4096 tokens
$TL bench.py --model mistralai/Mistral-7B-Instruct-v0.1 --slug mistral-7b-instruct-v0.1 --n 16 --batch 16 --lengths 512 4096 > bench_mistral.log 2>&1 && echo "ok bench mistral"
$VLLM vllm_orth.py --model mistralai/Mistral-7B-Instruct-v0.1 --direction results/mistral-7b-instruct-v0.1/direction.pt \
  --prompts bench/mistral-7b-instruct-v0.1/prompts.json --max_new 4096 --baseline \
  --out bench/mistral-7b-instruct-v0.1/vllm_4096.jsonl > vllm_mistral.log 2>&1 && echo "ok vllm mistral"

# correctness: strong-refusal model, short lengths (the O(n^2) loop is the reference)
$TL bench.py --model Qwen/Qwen2.5-1.5B-Instruct --slug qwen2.5-1.5b-instruct --n 32 --batch 32 --lengths 256 > bench_qwen.log 2>&1 && echo "ok bench qwen"
$VLLM vllm_orth.py --model Qwen/Qwen2.5-1.5B-Instruct --direction results/qwen2.5-1.5b-instruct/direction.pt \
  --prompts bench/qwen2.5-1.5b-instruct/prompts.json --max_new 64 --baseline \
  --out bench/qwen2.5-1.5b-instruct/vllm_64.jsonl > vllm_qwen.log 2>&1 && echo "ok vllm qwen"
echo "done $(date +%T)"
