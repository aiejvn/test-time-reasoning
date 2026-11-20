"""Container-only adapter around EvalPlus's standard evaluator.

The host supplies JSON data and never imports this module to execute solutions.
The container launch restrictions, not the marker checked here, provide isolation.
"""

import contextlib
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import sys


EVALPLUS_VERSION = "0.3.1"
DATASET_VERSION = "v0.1.10"
DATASET_PATH = Path("/opt/benchmark/HumanEvalPlus.jsonl")


def read_jsonl(path):
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def write_jsonl(path, records):
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, sort_keys=True, ensure_ascii=False) + "\n")


def stage_inputs(input_dir, work_dir, dataset_path=DATASET_PATH):
    """Validate and serialize text; this function never executes any code."""
    manifest = json.loads((input_dir / "manifest.json").read_text(encoding="utf-8"))
    splits = json.loads((input_dir / "splits.json").read_text(encoding="utf-8"))
    if manifest.get("schema_version") != 1 or manifest.get("benchmark") != "HumanEval+":
        raise ValueError("Expected schema_version 1 and benchmark HumanEval+")
    if splits.get("benchmark") != "HumanEval+":
        raise ValueError("Split file must describe HumanEval+")
    split = manifest.get("split")
    if split not in ("development", "held_out"):
        raise ValueError("Unknown split")
    task_ids = splits[split]
    if not task_ids or len(task_ids) != len(set(task_ids)):
        raise ValueError("Selected split must contain unique task IDs")

    dataset = read_jsonl(dataset_path)
    problems = {problem["task_id"]: problem for problem in dataset}
    if len(problems) != len(dataset):
        raise ValueError("Duplicate task IDs in benchmark snapshot")
    if not set(task_ids).issubset(problems):
        raise ValueError("Split contains unknown benchmark task IDs")

    samples = read_jsonl(input_dir / "samples.jsonl")
    by_task = {sample["task_id"]: sample for sample in samples}
    if len(samples) != len(by_task) or set(by_task) != set(task_ids):
        raise ValueError("Expected exactly one sample for every selected task")
    exported = []
    for task_id in task_ids:
        sample = by_task[task_id]
        fields = [field for field in ("solution", "completion") if field in sample]
        if len(fields) != 1 or not isinstance(sample[fields[0]], str):
            raise ValueError("Each sample needs exactly one string solution or completion")
        solution = sample[fields[0]]
        if fields[0] == "completion":
            solution = problems[task_id]["prompt"] + solution
        exported.append({"task_id": task_id, "solution": solution})

    # A fresh directory prevents EvalPlus from loading results from an older run.
    work_dir.mkdir(parents=True, exist_ok=False)
    subset_path = work_dir / "subset.jsonl"
    samples_path = work_dir / "samples.jsonl"
    write_jsonl(subset_path, [problems[task_id] for task_id in task_ids])
    write_jsonl(samples_path, exported)
    return subset_path, samples_path


def main():
    if os.environ.get("QMIND_SANDBOX") != "1" or not Path("/.dockerenv").is_file():
        raise RuntimeError("Use evaluate.py: candidate execution requires the Docker sandbox")
    installed = importlib.metadata.version("evalplus")
    if installed != EVALPLUS_VERSION:
        raise RuntimeError("Expected EvalPlus " + EVALPLUS_VERSION)

    work_dir = Path("/tmp/qmind")
    subset_path, samples_path = stage_inputs(Path("/input"), work_dir)
    # EvalPlus reads this environment variable at import time.
    os.environ["HUMANEVAL_OVERRIDE_PATH"] = str(subset_path)
    os.environ["XDG_CACHE_HOME"] = "/tmp/cache"
    # In EvalPlus 0.3.1 this override is not converted from str to number.
    os.environ.pop("EVALPLUS_TIMEOUT_PER_TASK", None)

    with (work_dir / "evalplus.log").open("w", encoding="utf-8") as log:
        with contextlib.redirect_stdout(log), contextlib.redirect_stderr(log):
            from evalplus.evaluate import evaluate

            evaluate(
                dataset="humaneval",
                samples=str(samples_path),
                parallel=1,
                base_only=False,
                min_time_limit=1.0,
                gt_time_limit_factor=4.0,
            )

    results_path = samples_path.with_name("samples_eval_results.json")
    results = json.loads(results_path.read_text(encoding="utf-8"))
    envelope = {
        "schema_version": 1,
        "evalplus_version": installed,
        "dataset_version": DATASET_VERSION,
        "dataset_sha256": hashlib.sha256(DATASET_PATH.read_bytes()).hexdigest(),
        "subset_sha256": hashlib.sha256(subset_path.read_bytes()).hexdigest(),
        "python_version": platform.python_version(),
        "results": results,
    }
    json.dump(envelope, sys.stdout, ensure_ascii=False, allow_nan=False)
    sys.stdout.write("\n")


if __name__ == "__main__":
    main()
