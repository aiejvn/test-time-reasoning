"""Three-way demo: 1 baseline (plain prompt) | 2 chain-of-thought | 3 chain-of-thought + our steering.

Reads artifacts/demo/responses.jsonl (experiments/demo_generate.py). Left/right arrows move between problems,
q quits. In each correct response the parts it gets right are bold green (lines reaching the right answer; the
returned function for passing code). In each failing response the parts that go wrong are bold red:
  - lines the model repeats (it is looping),
  - lines stating the wrong final answer (math),
  - the missing answer when it ran out of tokens.

    python experiments/demo_triptych.py                 # problems where 1 and 2 fail and 3 passes
    python experiments/demo_triptych.py --all           # every generated problem
    python experiments/demo_triptych.py --ids HumanEval/143
    python experiments/demo_triptych.py --recorded      # 2 panels from recorded runs, problems steering fixed
"""
import argparse
import json
import re
import sys
import termios
import textwrap
import tty
from collections import Counter
from pathlib import Path

from rich.console import Console, Group
from rich.live import Live
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

sys.path.insert(0, str(Path(__file__).resolve().parent))
import demo_compare as V1  # noqa: E402  (output_excerpt, verdict_from_row)

ROOT = V1.ROOT
COLS = [("plain", "1 · BASELINE", "red"), ("cot", "2 · CHAIN-OF-THOUGHT", "yellow"),
        ("steered", "3 · OURS: CoT + steering", "green")]
BENCH_NAME = {"gsm8k": "Math", "humanevalplus": "Code"}


def load(path: Path) -> dict:
    """{id: {condition: row}}, keeping the last row for each (id, condition)."""
    by = {}
    for r in map(json.loads, open(path)):
        by.setdefault(r["id"], {})[r["condition"]] = r
    return by


def questions(ids) -> dict:
    sys.path.insert(0, str(ROOT / "src"))
    import reasoning_env as R

    out = {}
    for bench in ["gsm8k", "humanevalplus"]:
        for split in R.load_benchmark(bench, ROOT / "configs/splits.json").values():
            out.update({it["id"]: it["question"] for it in split if it["id"] in ids})
    return out


def norm(line: str) -> str:
    return re.sub(r"\W+", " ", line.lower()).strip()


def num_re(x: str) -> str:
    """Match the number x on its own (not inside 2580 or 25.8); a trailing sentence period is fine."""
    return rf"(?<![\d.]){re.escape(x)}(?!\d|\.\d)"


def wrong_markers(row: dict) -> tuple:
    """(set of normalized repeated lines, wrong-answer string or None) for a failing response."""
    if row["correct"]:
        return set(), None
    thinking = row["response"].split("</think>")[0]
    counts = Counter(norm(ln) for ln in thinking.splitlines() if len(norm(ln)) > 25)
    repeated = {k for k, n in counts.items() if n >= 2}
    wrong = None
    if row["benchmark"] == "gsm8k" and row.get("pred") is not None:
        wrong = f"{row['pred']:g}"
    return repeated, wrong


def panel(row: dict, title: str, style: str, width: int, height: int) -> Panel:
    repeated, wrong = wrong_markers(row)
    ans, ok = V1.verdict_from_row(row)
    finished = "</think>" in row["response"]
    if not finished:
        ans = "no answer"
    answer = V1.output_excerpt(row, width, max_lines=8) if finished else \
        Text("✗ no answer: still reasoning at the 4,096-token limit", style="bold red")
    if not ok and wrong:
        answer.highlight_regex(num_re(wrong), "bold red")
    right = f"{row['gold']:g}" if ok and row["benchmark"] == "gsm8k" and row.get("gold") is not None else None
    if right:
        answer.highlight_regex(num_re(right), "bold green")
    elif ok:  # passing code: the whole returned function is the part it got right
        answer.stylize("bold green")
    thinking = row["response"].split("</think>")[0]
    src = [ln for ln in thinking.splitlines() if ln.strip()]
    room = max(height - 7 - len(answer.plain.splitlines()), 3)
    lines = []  # (wrapped text, is_wrong)
    for ln in src:
        bad = norm(ln) in repeated or (wrong is not None and re.search(num_re(wrong), ln))
        good = right is not None and re.search(num_re(right), ln)
        mark = "bad" if bad else "good" if good else ""
        lines += [(w, mark) for w in textwrap.wrap(ln, width)]
    body = Text("…\n", style="grey50")
    for w, mark in lines[-room:]:
        body.append(w + "\n", style={"bad": "bold red", "good": "bold green", "": "grey62"}[mark])
    body.append("\n")
    body.append_text(answer)
    verdict = Text("\n\n" + ("✅ " if ok else "❌ ") + ans, style="bold green" if ok else "bold red")
    verdict.append(f"   {row['n_resp_tokens']:,} tokens", style="grey50")
    return Panel(Group(body, verdict), title=title, border_style=style, height=height)


def screen(pid: str, rows: dict, question: str, k: int, n: int, console: Console, cols=COLS) -> Group:
    w, h = console.size.width, console.size.height
    q = [x for ln in question.splitlines() if ln.strip() for x in textwrap.wrap(ln, w - 2)][:4]
    ph = max(h - len(q) - 3, 12)
    grid = Table.grid(expand=True, padding=(0, 1))
    for _ in cols:
        grid.add_column(ratio=1)
    grid.add_row(*[panel(rows[c], t, s, w // len(cols) - 5, ph) if c in rows else Panel("(not generated)", title=t)
                   for c, t, s in cols])
    bench = next(iter(rows.values()))["benchmark"]
    head = Text.assemble((f"{BENCH_NAME[bench]} · {pid}", "bold"), (f"   {k}/{n}   ← →  q quit", "grey50"))
    return Group(head, Text("\n".join(q)), grid)


def read_key() -> str:
    fd = sys.stdin.fileno()
    old = termios.tcgetattr(fd)
    try:
        tty.setraw(fd)
        ch = sys.stdin.read(1)
        if ch == "\x1b":
            ch += sys.stdin.read(2)
        return ch
    finally:
        termios.tcsetattr(fd, termios.TCSADRAIN, old)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--responses", default="artifacts/demo/responses.jsonl")
    ap.add_argument("--ids", nargs="*")
    ap.add_argument("--all", action="store_true", help="show every generated problem, not only 1✗ 2✗ 3✓")
    ap.add_argument("--recorded", action="store_true",
                    help="2 panels, baseline vs. ours, from the recorded intervention runs: problems steering fixed")
    args = ap.parse_args()
    cols = COLS
    if args.recorded:
        cols = [("cot", "BASELINE", "red"), ("steered", "OURS: + steering", "green")]
        pairs = V1.recorded_pairs()
        data = {i: {"cot": b, "steered": e} for i, (_, _, b, e) in pairs.items() if not b["correct"] and e["correct"]}
        ids = args.ids or sorted(data, key=lambda i: (not i.startswith("HumanEval"), i))  # held-out code first
        qs = {i: V1.question_text(data[i]["cot"]) for i in ids}
        browse(ids, data, qs, cols)
        return
    data = load(ROOT / args.responses)
    if args.ids:
        ids = args.ids
    else:
        ids = [i for i, r in data.items() if args.all or (
            all(c in r for c, _, _ in COLS) and not r["plain"]["correct"] and not r["cot"]["correct"]
            and r["steered"]["correct"])]
    if not ids:
        sys.exit("no problem where baseline and CoT fail and steering passes; try --all")
    browse(ids, data, questions(set(ids)), cols)


def browse(ids: list, data: dict, qs: dict, cols) -> None:
    """Arrow-key viewer: right/l/space = next, left/h = previous, q = quit."""
    console, k = Console(), 0
    with Live(console=console, screen=True, auto_refresh=False) as lv:
        while True:
            lv.update(screen(ids[k], data[ids[k]], qs[ids[k]], k + 1, len(ids), console, cols), refresh=True)
            key = read_key()
            if key in ("q", "\x03"):
                break
            if key in ("\x1b[C", "l", " "):
                k = min(k + 1, len(ids) - 1)
            elif key in ("\x1b[D", "h"):
                k = max(k - 1, 0)


if __name__ == "__main__":
    main()
