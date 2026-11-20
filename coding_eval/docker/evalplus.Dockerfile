FROM python:3.11.9-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    HF_HUB_OFFLINE=1 \
    TRANSFORMERS_OFFLINE=1 \
    TOKENIZERS_PARALLELISM=false \
    OMP_NUM_THREADS=1 \
    OPENBLAS_NUM_THREADS=1 \
    MKL_NUM_THREADS=1 \
    HOME=/tmp \
    XDG_CACHE_HOME=/tmp/cache

# Reuse the project's existing dependency lock. Network access is needed only
# while building the image; evaluation runs use --network none.
COPY requirements.txt /opt/requirements.txt
RUN python -m pip install --no-cache-dir -r /opt/requirements.txt

# Fetch data only. Do not execute reference solutions or generate model outputs.
# Save a deterministic full snapshot so each run can report its exact SHA-256.
RUN python - <<'PY'
import importlib.metadata
import json
from pathlib import Path
from evalplus.data import get_human_eval_plus
from evalplus.data.humaneval import HUMANEVAL_PLUS_VERSION

assert importlib.metadata.version("evalplus") == "0.3.1"
assert HUMANEVAL_PLUS_VERSION == "v0.1.10"
problems = get_human_eval_plus()
destination = Path("/opt/benchmark/HumanEvalPlus.jsonl")
destination.parent.mkdir(parents=True)
with destination.open("w", encoding="utf-8") as handle:
    for task_id in sorted(problems, key=lambda key: int(key.split("/")[-1])):
        handle.write(json.dumps(problems[task_id], sort_keys=True, ensure_ascii=False) + "\n")
PY

COPY docker/worker.py /opt/qmind/worker.py
USER 65534:65534
WORKDIR /tmp
ENTRYPOINT ["python", "/opt/qmind/worker.py"]
