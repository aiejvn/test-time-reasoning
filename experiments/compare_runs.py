"""Are two baseline runs identical on the problems they share?

For each <bench>_<split>.jsonl present in both run directories, compares every shared problem's response text
and grade. Used to confirm that the batch-size-1 baseline is reproducible run to run (linear_probe/REPORT.md, audit 3a).

Run
---
    python experiments/compare_runs.py artifacts/baseline/baseline_bs1 artifacts/baseline/baseline_bs1_repeat
"""
import argparse
import json
from pathlib import Path


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("run_a")
    ap.add_argument("run_b")
    ap.add_argument("--out", default=None, help="optional JSON report path")
    args = ap.parse_args()
    a_dir, b_dir = Path(args.run_a), Path(args.run_b)
    report = {}
    for fa in sorted(a_dir.glob("*.jsonl")):
        fb = b_dir / fa.name
        if not fb.exists():
            continue
        a = {r["id"]: r for r in map(json.loads, open(fa))}
        b = {r["id"]: r for r in map(json.loads, open(fb))}
        shared = [i for i in b if i in a]
        diff_text = [i for i in shared if a[i]["response"] != b[i]["response"]]
        diff_grade = [i for i in shared if a[i]["correct"] != b[i]["correct"]]
        report[fa.stem] = dict(n_shared=len(shared), identical_responses=len(shared) - len(diff_text),
                               differing_responses=diff_text, differing_grades=diff_grade)
        print(fa.stem, report[fa.stem])
    if args.out:
        json.dump(report, open(args.out, "w"), indent=2)


if __name__ == "__main__":
    main()
