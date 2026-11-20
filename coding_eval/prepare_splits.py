
import json
import random
from pathlib import Path
from evalplus.data import get_human_eval_plus

SEED = 2026
N_DEV = 20
N_TEST = 40

problems = get_human_eval_plus()

# Sort numerically before shuffling
task_ids = sorted(
    problems.keys(),
    key=lambda x: int(x.split("/")[-1])
)

# Reproducible random selection
rng = random.Random(SEED)
rng.shuffle(task_ids)

dev_ids = task_ids[:N_DEV]
test_ids = task_ids[N_DEV:N_DEV + N_TEST]

assert set(dev_ids).isdisjoint(test_ids)

splits = {
    "benchmark": "HumanEval+",
    "seed": SEED,
    "development": dev_ids,
    "held_out": test_ids,
}

output = Path("data/splits.json")
output.parent.mkdir(exist_ok=True)

with output.open("w") as f:
    json.dump(splits, f, indent=2)

print(f"Development problems: {len(dev_ids)}")
print(f"Held-out problems: {len(test_ids)}")
print(f"Saved to: {output}")
print(f"Example development ID: {dev_ids[0]}")
