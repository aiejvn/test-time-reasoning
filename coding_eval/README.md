# QMIND — HumanEval+ Coding Evaluation

A reproducible evaluation harness for the **QMIND mechanistic interpretability / test-time reasoning project**. It evaluates generated Python solutions on [HumanEval+](https://github.com/evalplus/evalplus), recording correctness under **baseline** (unaltered model) and **intervention** (modified model activations) conditions.

**Scope:** This repository **evaluates saved model outputs**. It does not load DeepSeek, generate solutions, or implement activation interventions. Those tasks are handled by the model-inference and intervention workstreams. Both conditions should provide outputs for the **same task IDs** using **matched generation settings**.

## Repository structure

```text
qmind-coding-eval/
├── data/
│   └── splits.json          # Fixed development and held-out task IDs
├── docker/
│   ├── evalplus.Dockerfile  # Isolated EvalPlus evaluation image
│   └── worker.py            # Code-execution worker inside the container
├── .dockerignore
├── evaluate.py              # validate, run, and compare CLI
├── inspect_dataset.py      # Dataset inspection utility
├── prepare_splits.py       # Split preparation utility
└── requirements.txt
```

The local `.venv/`, `runs/`, and `smoke/` directories are **not** intended to be committed. `runs/` contains generated outputs and evaluation artifacts; review any run data before sharing it separately.

## Setup

**Requirements:** Python, Docker Desktop (or a compatible Docker daemon), and the Python dependencies listed in `requirements.txt`.

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt

docker build -f docker/evalplus.Dockerfile -t qmind-evalplus .
docker info
```

These commands assume the current directory is the project root. Docker must be running for the `run` subcommand.

## What the evaluator expects

The pipeline consumes three files:

1. **`data/splits.json`** — a fixed mapping with `benchmark: "HumanEval+"`, `development`, and `held_out` arrays of HumanEval task IDs. The two lists must not overlap.
2. **Manifest JSON** — identifies the run, model/tokenizer revisions, generation parameters, split, output extraction policy, and intervention provenance (if any).
3. **Samples JSONL** — one JSON object per task, containing its generated Python code and the information needed to trace it back to the model run.

The current evaluator requires **exactly one sample for every task in the selected split**. Missing, duplicate, or out-of-split tasks are rejected rather than silently changing the denominator. This is a single-attempt **pass@1** evaluation, not multi-sample `pass@k`.

### Example manifest (illustrative — replace placeholders)

```json
{
  "schema_version": 1,
  "run_id": "baseline_dev_v1",
  "benchmark": "HumanEval+",
  "condition": "baseline",
  "split": "development",
  "model": {"id": "MODEL_ID", "revision": "IMMUTABLE_MODEL_REVISION"},
  "tokenizer": {"id": "TOKENIZER_ID", "revision": "IMMUTABLE_TOKENIZER_REVISION"},
  "generation": {
    "seed": 0,
    "do_sample": false,
    "max_new_tokens": 2048,
    "temperature": 0.0,
    "top_p": 1.0
  },
  "extraction_policy": "code-extraction-v1",
  "intervention": null
}
```

For an intervention run, use a distinct `run_id`, set `condition` to `"intervention"`, and replace `intervention: null` with a provenance object containing a nonempty `id` (and any additional parameters useful for reproducing the intervention). Record the **actual effective settings**, not merely intended defaults. Both conditions should use the same task IDs and comparable sampling settings.

### Example samples JSONL (illustrative schema)

Each line is one JSON object; fields shown below are illustrative rather than a real model generation:

```json
{"task_id":"HumanEval/5","sample_id":"baseline-HumanEval-5","seed":0,"prompt":"<exact prompt sent to the model>","raw_output":"<unmodified model output>","solution":"<complete, extracted Python source>"}
```

Required fields are `task_id`, `sample_id`, `seed`, `prompt`, `raw_output`, and **exactly one** of:

- `solution`: complete Python solution, **or**
- `completion`: the generated continuation to the benchmark's function prompt.

Keep the original prompt and raw output for later analysis. Don't fabricate these fields when preparing actual research results. Use stable task IDs from `data/splits.json`.

## Running an evaluation

First validate the inputs (no generated code is executed at this stage):

```bash
python evaluate.py validate \
  --manifest path/to/baseline_manifest.json \
  --samples path/to/baseline_samples.jsonl \
  --splits data/splits.json
```

Then evaluate saved code in Docker:

```bash
python evaluate.py run \
  --manifest path/to/baseline_manifest.json \
  --samples path/to/baseline_samples.jsonl \
  --splits data/splits.json \
  --image qmind-evalplus
```

Repeat with the intervention manifest and samples after receiving the team's intervened generations. Every run must have a unique `run_id` because the evaluator preserves each run in its own directory and does not overwrite an existing run.

Successful runs are saved under `runs/<run_id>/`, including the original inputs, per-task `results.jsonl`, `summary.json`, `evalplus.json`, and run metadata. The summary includes both base HumanEval and extended HumanEval+ single-attempt accuracy.

The CLI also offers a `compare` subcommand:

```bash
python evaluate.py compare --help
```

Provide the saved baseline and intervention artifacts in the formats required by your installed version of `evaluate.py`. Verify paired coverage and generation-setting consistency before interpreting the difference in accuracy.

## Smoke test and current verification

During initial local validation, **HumanEval/5** was tested with two **synthetic fixtures**:

| Fixture | HumanEval pass@1 | HumanEval+ pass@1 |
|---|---:|---:|
| Known-correct reference implementation | 1.0 | 1.0 |
| Deliberately incorrect implementation | 0.0 | 0.0 |

Both runs completed in Docker and produced saved evaluation artifacts. These are **infrastructure checks, not measured model or intervention performance**. Fixture files were created locally under `smoke/` and are not included in the source repository by default.

To inspect a previously generated smoke-test run locally:

```bash
cat runs/smoke_baseline/summary.json
cat runs/smoke_intervention/summary.json
```

## Evaluation integrity and safety

- **Execute generated code only in an appropriately isolated environment.** The Docker runner is configured with no network, a read-only root filesystem, a non-root user, reduced Linux capabilities, process/CPU/memory limits, and a timeout. Containerization reduces risk but is **not an absolute security guarantee**.
- Preserve the **same problem subset and effective generation settings** for baseline and intervention. Where stochastic decoding is used, record seeds and consider multiple independent runs before drawing strong conclusions.
- Keep the **development** split for iteration and the **held-out** split for the final assessment. Avoid using held-out failures to tune interventions.
- Distinguish **infrastructure errors** from incorrect generated solutions. A failed container launch or missing data must not be counted as a model error.
- Preserve raw outputs, extraction policies, model revisions, intervention settings, and evaluation-tool versions to make runs auditable.
- A change in pass rate is a behavioural observation. It does **not**, by itself, prove that an intervention improved internal reasoning or identify a causal mechanism.

## Team integration

The model-inference / intervention workstreams should provide:

1. The exact HumanEval task IDs from `data/splits.json`.
2. One generated candidate per selected task, saved as JSONL with the schema above.
3. Separate baseline and intervention manifests with model/tokenizer revisions, decoding settings, extraction policy, and intervention provenance.

The evaluation workstream then validates both exports, runs EvalPlus in Docker, compares per-task outcomes, and reports base/plus accuracy along with **incorrect → correct**, **correct → incorrect**, and unchanged cases.

**Current status:** Local evaluation and a one-task controlled smoke test are verified. Real baseline-versus-intervention model results are pending integration with the generation workstream.
