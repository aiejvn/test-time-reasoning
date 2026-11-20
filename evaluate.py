"""Evaluate saved HumanEval+ generations in Docker; never execute code on the host."""

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import re
import selectors
import shutil
import subprocess
import sys
import tempfile
import time
import uuid
from datetime import datetime, timezone


EVALUATOR_SETTINGS = {
    "evalplus_version": "0.3.1",
    "dataset_version": "v0.1.10",
    "parallel": 1,
    "min_time_limit": 1.0,
    "gt_time_limit_factor": 4.0,
    "base_only": False,
    "test_details": False,
}
SANDBOX_SETTINGS = {"cpus": 2, "memory_mb": 2048, "pids": 128, "tmp_mb": 512}
MAX_OUTPUT_BYTES = 16 * 1024 * 1024


class EvaluationError(ValueError):
    """Invalid experiment input or an infrastructure failure, not an incorrect answer."""


def require(condition, message):
    if not condition:
        raise EvaluationError(message)


def sha256(data):
    return hashlib.sha256(data).hexdigest()


def _reject_constant(value):
    raise EvaluationError("Non-finite JSON number: " + value)


def _unique_keys(pairs):
    result = {}
    for key, value in pairs:
        require(key not in result, "Duplicate JSON key: " + key)
        result[key] = value
    return result


def decode_json(data):
    try:
        return json.loads(data, parse_constant=_reject_constant, object_pairs_hook=_unique_keys)
    except (ValueError, UnicodeDecodeError) as exc:
        raise EvaluationError("Invalid JSON: " + str(exc)) from exc


def decode_jsonl(data):
    return [decode_json(line) for line in data.splitlines() if line.strip()]


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")


def write_jsonl(path, rows):
    path.write_text("".join(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n" for row in rows), encoding="utf-8")


def nonempty_string(value):
    return isinstance(value, str) and bool(value.strip())


def integer(value):
    return type(value) is int and value >= 0


def validate_inputs(manifest, samples, splits):
    """Require one recorded attempt for every task in the selected, existing split."""
    require(isinstance(manifest, dict), "Manifest must be an object")
    require(manifest.get("schema_version") == 1, "Manifest schema_version must be 1")
    require(isinstance(manifest.get("run_id"), str) and re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,99}", manifest["run_id"]), "Invalid run_id")
    require(manifest.get("benchmark") == "HumanEval+", "Only HumanEval+ is supported")
    require(manifest.get("condition") in ("baseline", "intervention"), "condition must be baseline or intervention")
    require(manifest.get("split") in ("development", "held_out"), "Unknown split")
    for field in ("model", "tokenizer"):
        value = manifest.get(field)
        require(isinstance(value, dict) and all(nonempty_string(value.get(k)) for k in ("id", "revision")), field + " requires id and immutable revision")
    generation = manifest.get("generation")
    require(isinstance(generation, dict), "generation must be an object containing the effective settings")
    require(integer(generation.get("seed")), "generation.seed must be a nonnegative integer")
    require(type(generation.get("do_sample")) is bool, "generation.do_sample must be boolean")
    require(integer(generation.get("max_new_tokens")) and generation["max_new_tokens"] > 0, "generation.max_new_tokens must be positive")
    for key, lower, upper in (("temperature", 0, float("inf")), ("top_p", 0, 1)):
        value = generation.get(key)
        require(type(value) in (int, float) and math.isfinite(value) and lower <= value <= upper, "Invalid generation." + key)
    require(generation["top_p"] > 0, "generation.top_p must be greater than zero")
    require(not generation["do_sample"] or generation["temperature"] > 0, "Sampling requires positive temperature")
    require(nonempty_string(manifest.get("extraction_policy")), "Record extraction_policy/version for reproducible code extraction")
    if manifest["condition"] == "baseline":
        require(manifest.get("intervention") is None, "Baseline must have no intervention")
    else:
        require(isinstance(manifest.get("intervention"), dict) and nonempty_string(manifest["intervention"].get("id")), "Intervention requires provenance with an id")

    require(isinstance(splits, dict) and splits.get("benchmark") == "HumanEval+", "Invalid split benchmark")
    for name in ("development", "held_out"):
        ids = splits.get(name)
        require(isinstance(ids, list) and bool(ids), name + " must be a nonempty task ID list")
        require(all(isinstance(t, str) and re.fullmatch(r"HumanEval/(0|[1-9][0-9]*)", t) for t in ids), "Invalid HumanEval task ID")
        require(len(ids) == len(set(ids)), "Duplicate task IDs in " + name)
    require(set(splits["development"]).isdisjoint(splits["held_out"]), "Development and held-out splits overlap")
    task_ids = splits[manifest["split"]]
    require(isinstance(samples, list) and bool(samples), "Samples must be nonempty")
    seen = set()
    for sample in samples:
        require(isinstance(sample, dict), "Each sample must be an object")
        task_id = sample.get("task_id")
        require(isinstance(task_id, str) and task_id in task_ids, "Sample has an unknown or out-of-split task_id")
        require(task_id not in seen, "Exactly one sample per task is supported; duplicate " + task_id)
        seen.add(task_id)
        require(nonempty_string(sample.get("sample_id")), "Sample requires a stable sample_id")
        require(integer(sample.get("seed")), "Sample requires its actual generation seed")
        require(nonempty_string(sample.get("prompt")), "Sample must preserve the actual rendered model prompt")
        require(isinstance(sample.get("raw_output"), str), "Sample must preserve raw_output (including reasoning)")
        fields = [key for key in ("solution", "completion") if key in sample]
        require(len(fields) == 1 and isinstance(sample[fields[0]], str), "Sample requires exactly one string: solution or completion")
    require(seen == set(task_ids), "Missing tasks from selected split: " + ", ".join(sorted(set(task_ids) - seen)))
    return task_ids


def load_inputs(manifest_path, samples_path, splits_path):
    blobs = {"manifest.json": Path(manifest_path).read_bytes(), "samples.jsonl": Path(samples_path).read_bytes(), "splits.json": Path(splits_path).read_bytes()}
    manifest = decode_json(blobs["manifest.json"])
    samples = decode_jsonl(blobs["samples.jsonl"])
    splits = decode_json(blobs["splits.json"])
    task_ids = validate_inputs(manifest, samples, splits)
    return blobs, manifest, samples, splits, task_ids


def build_docker_command(image_id, input_dir, name):
    """Only a temporary directory of input files is exposed, read-only."""
    require(re.fullmatch(r"sha256:[a-f0-9]{64}", image_id) is not None, "Docker must resolve an immutable image ID")
    require("," not in str(input_dir), "Docker input path cannot contain commas")
    return [
        "docker", "run", "--rm", "--pull=never", "--name", name,
        "--network=none", "--read-only", "--user=65534:65534",
        "--cap-drop=ALL", "--security-opt=no-new-privileges:true",
        "--pids-limit=128", "--memory=2048m", "--memory-swap=2048m",
        "--cpus=2", "--ipc=private", "--init", "--log-driver=none",
        "--ulimit", "nofile=256:256", "--ulimit", "fsize=67108864:67108864",
        "--tmpfs", "/tmp:rw,noexec,nosuid,nodev,size=512m,mode=1777",
        "--workdir=/tmp", "--env=HOME=/tmp", "--env=XDG_CACHE_HOME=/tmp/cache",
        "--env=PYTHONDONTWRITEBYTECODE=1", "--env=OMP_NUM_THREADS=1",
        "--env=OPENBLAS_NUM_THREADS=1", "--env=MKL_NUM_THREADS=1",
        "--env=QMIND_SANDBOX=1",
        "--mount", "type=bind,source=" + str(input_dir) + ",target=/input,readonly",
        "--entrypoint=python", image_id, "-I", "/opt/qmind/worker.py",
    ]


def inspect_image(image):
    require(shutil.which("docker") is not None, "Docker is required; generated code will never run on the host. Install/start Docker and build the evaluation image.")
    try:
        result = subprocess.run(["docker", "image", "inspect", image], capture_output=True, timeout=30, check=True)
    except (OSError, subprocess.SubprocessError) as exc:
        raise EvaluationError("Cannot inspect local Docker image. Start Docker and build the image first.") from exc
    info = decode_json(result.stdout)[0]
    require(re.fullmatch(r"sha256:[a-f0-9]{64}", info.get("Id", "")) is not None, "Invalid Docker image identity")
    return {"id": info["Id"], "repo_digests": info.get("RepoDigests", []), "architecture": info.get("Architecture"), "os": info.get("Os")}


def run_container(command, name, timeout, output_limit=MAX_OUTPUT_BYTES):
    """Bound pipe memory and wall time, and remove the container even on failure."""
    process = None
    buffers = {"stdout": bytearray(), "stderr": bytearray()}
    started = time.monotonic()
    try:
        process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        with selectors.DefaultSelector() as selector:
            selector.register(process.stdout, selectors.EVENT_READ, "stdout")
            selector.register(process.stderr, selectors.EVENT_READ, "stderr")
            while selector.get_map():
                require(time.monotonic() - started < timeout, "Container wall-clock timeout; no correctness result was assigned")
                for key, _ in selector.select(timeout=0.1):
                    chunk = os.read(key.fileobj.fileno(), 65536)
                    if not chunk:
                        selector.unregister(key.fileobj)
                        continue
                    require(sum(map(len, buffers.values())) + len(chunk) <= output_limit, "Container output limit exceeded; no correctness result was assigned")
                    buffers[key.data].extend(chunk)
            remaining = max(0.01, timeout - (time.monotonic() - started))
            try:
                returncode = process.wait(timeout=remaining)
            except subprocess.TimeoutExpired as exc:
                raise EvaluationError("Container wall-clock timeout; no correctness result was assigned") from exc
        require(returncode == 0, "Container failed (exit " + str(returncode) + "): " + bytes(buffers["stderr"][-4000:]).decode("utf-8", errors="replace"))
        return bytes(buffers["stdout"]), bytes(buffers["stderr"])
    finally:
        # Killing the Docker CLI alone does not stop the container.
        try:
            subprocess.run(["docker", "rm", "-f", name], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, timeout=15, check=False)
        finally:
            if process is not None:
                if process.poll() is None:
                    process.kill()
                process.wait()
                process.stdout.close()
                process.stderr.close()


def normalize_results(envelope, manifest, samples, task_ids):
    require(isinstance(envelope, dict) and envelope.get("schema_version") == 1, "Invalid worker response")
    for key in ("evalplus_version", "dataset_version"):
        require(envelope.get(key) == EVALUATOR_SETTINGS[key], "Worker " + key + " mismatch")
    for key in ("dataset_sha256", "subset_sha256"):
        require(isinstance(envelope.get(key), str) and re.fullmatch(r"[a-f0-9]{64}", envelope[key]), "Missing dataset fingerprint: " + key)
    raw = envelope.get("results")
    require(isinstance(raw, dict) and isinstance(raw.get("eval"), dict), "Missing EvalPlus results")
    require(set(raw["eval"]) == set(task_ids), "EvalPlus task coverage differs from selected split")
    originals = {sample["task_id"]: sample for sample in samples}
    rows = []
    for task_id in task_ids:
        values = raw["eval"][task_id]
        require(isinstance(values, list) and len(values) == 1 and isinstance(values[0], dict), "Expected one EvalPlus result per task")
        result = values[0]
        require(result.get("task_id") == task_id, "EvalPlus result task_id mismatch")
        require(all(result.get(k) in ("pass", "fail", "timeout") for k in ("base_status", "plus_status")), "Missing/invalid base or plus status")
        require(isinstance(result.get("solution"), str), "Missing evaluated solution")
        original = originals[task_id]
        if "solution" in original:
            require(result["solution"] == original["solution"], "Evaluated solution differs from submitted solution")
        else:
            require(result["solution"].endswith(original["completion"]), "Evaluated completion differs from submitted completion")
        rows.append({**original, "run_id": manifest["run_id"], "condition": manifest["condition"], "base_status": result["base_status"], "plus_status": result["plus_status"], "correct": result["base_status"] == result["plus_status"] == "pass", "evaluated_solution": result["solution"]})
    return rows


def summarize(manifest, rows):
    base = sum(row["base_status"] == "pass" for row in rows)
    plus = sum(row["correct"] for row in rows)
    return {"run_id": manifest["run_id"], "condition": manifest["condition"], "split": manifest["split"], "n_tasks": len(rows), "samples_per_task": 1, "base_correct": base, "plus_correct": plus, "humaneval_pass_at_1": base / len(rows), "humaneval_plus_pass_at_1": plus / len(rows)}


def evaluate_run(manifest_path, samples_path, splits_path, output_root, image, timeout=3600):
    require(type(timeout) in (int, float) and math.isfinite(timeout) and timeout > 0, "timeout must be positive and finite")
    blobs, manifest, samples, _, task_ids = load_inputs(manifest_path, samples_path, splits_path)
    destination = Path(output_root) / manifest["run_id"]
    destination.mkdir(parents=True, exist_ok=False)
    for name, blob in blobs.items():
        (destination / name).write_bytes(blob)
    metadata = {
        "schema_version": 1, "status": "started", "started_at": datetime.now(timezone.utc).isoformat(),
        "input_sha256": {name: sha256(blob) for name, blob in blobs.items()},
        "evaluator": EVALUATOR_SETTINGS, "sandbox": SANDBOX_SETTINGS,
        "host_adapter_sha256": sha256(Path(__file__).read_bytes()),
        "wall_timeout_seconds": timeout, "output_limit_bytes": MAX_OUTPUT_BYTES,
    }
    write_json(destination / "evaluation.json", metadata)
    try:
        metadata["image"] = inspect_image(image)
        with tempfile.TemporaryDirectory(prefix="qmind-eval-") as temporary:
            staging = Path(temporary)
            staging.chmod(0o755)
            for name, blob in blobs.items():
                (staging / name).write_bytes(blob)
                (staging / name).chmod(0o444)
            name = "qmind-eval-" + uuid.uuid4().hex
            command = build_docker_command(metadata["image"]["id"], staging, name)
            stdout, stderr = run_container(command, name, timeout)
        (destination / "container.stderr.log").write_bytes(stderr)
        envelope = decode_json(stdout)
        rows = normalize_results(envelope, manifest, samples, task_ids)
        write_json(destination / "evalplus.json", envelope)
        write_jsonl(destination / "results.jsonl", rows)
        write_json(destination / "summary.json", summarize(manifest, rows))
        metadata.update({"status": "complete", "dataset_sha256": envelope["dataset_sha256"], "subset_sha256": envelope["subset_sha256"], "python_version": envelope.get("python_version")})
    except BaseException as exc:
        metadata.update({"status": "failed", "error": str(exc)})
        raise
    finally:
        metadata["finished_at"] = datetime.now(timezone.utc).isoformat()
        write_json(destination / "evaluation.json", metadata)
    return destination


def read_completed_run(directory):
    directory = Path(directory)
    blobs, manifest, samples, _, task_ids = load_inputs(directory / "manifest.json", directory / "samples.jsonl", directory / "splits.json")
    metadata = decode_json((directory / "evaluation.json").read_bytes())
    require(metadata.get("status") == "complete", "Can only compare completed evaluations")
    require(metadata.get("input_sha256") == {name: sha256(blob) for name, blob in blobs.items()}, "Archived inputs were modified")
    # Re-derive correctness from retained EvalPlus output instead of trusting summary files.
    envelope = decode_json((directory / "evalplus.json").read_bytes())
    rows = normalize_results(envelope, manifest, samples, task_ids)
    for key in ("dataset_sha256", "subset_sha256"):
        require(metadata.get(key) == envelope[key], "Dataset provenance mismatch")
    return manifest, metadata, rows


def compare_runs(baseline_dir, intervention_dir):
    baseline, bm, br = read_completed_run(baseline_dir)
    intervention, im, ir = read_completed_run(intervention_dir)
    require(baseline["condition"] == "baseline" and intervention["condition"] == "intervention", "Comparison needs baseline then intervention")
    require(baseline["run_id"] != intervention["run_id"], "Run IDs must be distinct")
    for key in ("benchmark", "split", "model", "tokenizer", "generation", "extraction_policy"):
        require(baseline[key] == intervention[key], "Incompatible generation protocol: " + key)
    for key in ("evaluator", "sandbox", "image", "dataset_sha256", "subset_sha256", "host_adapter_sha256", "wall_timeout_seconds", "output_limit_bytes"):
        require(bm.get(key) == im.get(key), "Incompatible evaluation environment: " + key)
    require(bm["input_sha256"]["splits.json"] == im["input_sha256"]["splits.json"], "Split files differ")
    require([r["task_id"] for r in br] == [r["task_id"] for r in ir], "Task coverage/order differs")
    pairs = []
    for left, right in zip(br, ir):
        for key in ("sample_id", "seed", "prompt"):
            require(left[key] == right[key], "Unpaired " + key + " for " + left["task_id"])
        pairs.append({"task_id": left["task_id"], "sample_id": left["sample_id"], "baseline_correct": left["correct"], "intervention_correct": right["correct"]})
    bs, ins = summarize(baseline, br), summarize(intervention, ir)
    return {"baseline": bs, "intervention": ins, "plus_pass_at_1_delta": ins["humaneval_plus_pass_at_1"] - bs["humaneval_plus_pass_at_1"], "improved": sum(not p["baseline_correct"] and p["intervention_correct"] for p in pairs), "regressed": sum(p["baseline_correct"] and not p["intervention_correct"] for p in pairs), "pairs": pairs}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    for name in ("validate", "run"):
        command = commands.add_parser(name)
        command.add_argument("--manifest", required=True, type=Path)
        command.add_argument("--samples", required=True, type=Path)
        command.add_argument("--splits", type=Path, default=Path(__file__).parent / "data/splits.json")
        if name == "run":
            command.add_argument("--output-root", type=Path, default=Path("runs"))
            command.add_argument("--image", default="qmind-evalplus:0.3.1")
            command.add_argument("--timeout", type=float, default=3600)
    compare = commands.add_parser("compare")
    compare.add_argument("--baseline", required=True, type=Path)
    compare.add_argument("--intervention", required=True, type=Path)
    compare.add_argument("--output", type=Path)
    args = parser.parse_args(argv)
    try:
        if args.command == "validate":
            *_, task_ids = load_inputs(args.manifest, args.samples, args.splits)
            print("Validated " + str(len(task_ids)) + " tasks; no code executed.")
        elif args.command == "run":
            destination = evaluate_run(args.manifest, args.samples, args.splits, args.output_root, args.image, args.timeout)
            print("Saved evaluation to " + str(destination))
        else:
            result = compare_runs(args.baseline, args.intervention)
            if args.output:
                with args.output.open("x", encoding="utf-8") as output:
                    json.dump(result, output, indent=2, allow_nan=False)
                    output.write("\n")
            else:
                print(json.dumps(result, indent=2))
    except (EvaluationError, OSError, subprocess.SubprocessError) as exc:
        parser.exit(1, "Evaluation error: " + str(exc) + "\n")


if __name__ == "__main__":
    main()
