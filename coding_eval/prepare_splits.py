"""data/splits.json is generated from the project's canonical splits; don't draw a separate split here.

Until 2026-11-21 this script drew its own HumanEval+ split (seed 2026: 20 development / 40 held-out), which
overlapped the split the probe and intervention runs used: 7 of the probe's 32 training problems were in its
held-out set. Both now come from configs/splits.json:

    development = humanevalplus dev + tune (64 task IDs)
    held_out    = humanevalplus test      (99 task IDs)

To regenerate, run from the repo root:

    python experiments/make_splits.py
"""
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
subprocess.run([sys.executable, str(ROOT / "experiments/make_splits.py")], check=True)
