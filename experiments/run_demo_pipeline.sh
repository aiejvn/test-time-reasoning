#!/bin/bash
# Steps 2 -> 3 -> 4b after the batch-size-1 baseline (artifacts/baseline/run_bs1.sh) finishes.
#   2. retrain the probe on baseline_bs1 dev (fixed grader)            linear_probe/probe.py
#   3. pick steering strength on gsm8k tune (100 problems)               linear_probe/configs/intervene_tune.yaml
#      rule: largest net (fixed - broken) vs. no edit; ties -> smaller strength
#   4b. held-out test: no edit / probe / random, 100 math + 50 code      linear_probe/configs/intervene_demo.yaml
# Usage: nohup setsid bash experiments/run_demo_pipeline.sh <pid of run_bs1.sh> > artifacts/demo_pipeline.log 2>&1 &
set -euo pipefail
cd /home/ubuntu/QMIND/test-time-reasoning
source .venv/bin/activate

WAIT_PID=${1:-}
if [ -n "$WAIT_PID" ]; then
  echo "$(date -u) waiting for pid $WAIT_PID (run_bs1.sh)"
  while kill -0 "$WAIT_PID" 2>/dev/null; do sleep 60; done
fi
grep -q ALL_DONE artifacts/baseline/run_bs1.log || { echo "run_bs1.sh did not finish (no ALL_DONE); stopping"; exit 1; }

echo "$(date -u) step 2: probe"
python linear_probe/probe.py --config linear_probe/configs/probe.yaml \
  -o run_name=gsm8k_dev_bs1 -o baseline_dir=artifacts/baseline/baseline_bs1

echo "$(date -u) step 3: strength sweep on tune"
python linear_probe/intervene.py --config linear_probe/configs/intervene_tune.yaml

ALPHA=$(python - <<'PY'
import json
s = json.load(open("artifacts/intervene/tune_bs1/summary.json"))
rows = [v for v in s.values() if v["condition"] == "probe_pos"]
best = max(rows, key=lambda v: (v["net"], -v["alpha_frac"]))
json.dump(dict(chosen_alpha_frac=best["alpha_frac"], rule="max net (fixed-broken); ties -> smaller", rows=rows),
          open("artifacts/intervene/tune_bs1/selection.json", "w"), indent=2)
print(best["alpha_frac"])
PY
)
echo "$(date -u) chosen alpha_frac = $ALPHA (artifacts/intervene/tune_bs1/selection.json)"

echo "$(date -u) step 4b: held-out test"
python linear_probe/intervene.py --config linear_probe/configs/intervene_demo.yaml -o "alpha_fracs=[$ALPHA]"
echo "$(date -u) PIPELINE_DONE"
