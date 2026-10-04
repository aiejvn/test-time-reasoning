#!/usr/bin/env bash
# Runs refusal_direction.py once per model, sequentially, on one GPU.
# Model weights go to /dev/shm (RAM-backed, wiped on reboot) and are deleted after each run.
set -u
cd "$(dirname "$0")"
export HF_HOME=/dev/shm/refusal/hf HF_HUB_ENABLE_HF_TRANSFER=1
PY=/root/kevin/.venv/bin/python

# slug | HF id | extra args
MODELS=(
  "olmo2-1b-base|allenai/OLMo-2-0425-1B|"
  "qwen2.5-1.5b-instruct|Qwen/Qwen2.5-1.5B-Instruct|"
  "qwen2.5-7b-instruct|Qwen/Qwen2.5-7B-Instruct|"
  "qwen3-1.7b|Qwen/Qwen3-1.7B|--no_think"
  "qwen3-8b|Qwen/Qwen3-8B|--no_think"
  "phi3-mini-instruct|microsoft/Phi-3-mini-4k-instruct|"
  "qwen1.5-1.8b-chat|Qwen/Qwen1.5-1.8B-Chat|"  # closest to the paper/notebook's Qwen-1_8B-Chat; OLMo-3 conversion is broken in TL 3.0
  "mistral-7b-instruct-v0.1|mistralai/Mistral-7B-Instruct-v0.1|"
  "yi-6b-chat|01-ai/Yi-6B-Chat|"
)

for entry in "${MODELS[@]}"; do
  IFS='|' read -r slug model extra <<< "$entry"
  if [[ -f results/$slug/metrics.json ]]; then echo "skip $slug (done)"; continue; fi
  mkdir -p results/$slug
  echo "=== $slug ($model) $(date +%T)"
  $PY refusal_direction.py --model "$model" --out results/$slug $extra > results/$slug/run.log 2>&1 \
    && echo "ok $slug" || echo "FAILED $slug (see results/$slug/run.log)"
  rm -rf "$HF_HOME/hub/models--${model//\//--}"
done
echo "all done $(date +%T)"
