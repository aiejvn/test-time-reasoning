"""Output-focused terminal demo (v2): before steering / after steering / random control, on held-out problems.

Everything shown is replayed from one recorded intervention run (linear_probe/intervene.py output). No GPU.

Screens (auto-advance; --manual waits for Enter instead):
  1. intro      what we steer toward and how (shared with demo v1; --no-intro skips it)
  2. examples   every held-out problem where steering changed the outcome, fixed or broken, each shown as
                input + three outputs: unedited model | steered (probe direction) | control (random direction)
  3. reel       all held-out problems, one row each, filling in with a running fixed/broken tally
  4. result     per benchmark: accuracy, fixed/broken, McNemar p, mean tokens, for steered vs. control,
                plus one sentence that states what the numbers show

Correspondence with the experiment (PROPOSAL.md 4.5, linear_probe/REPORT.md step 3):
  before = no-edit baseline, after = targeted intervention, control = random-direction control (same strength),
  reel/result = fixed/broken counts and McNemar p on GSM8K (math) and HumanEval+ (code), tokens = length control.

    python experiments/demo_heldout.py                                   # artifacts/intervene/heldout_half
    python experiments/demo_heldout.py --run artifacts/intervene/<run>   # any run with none/probe_pos/random_pos
    python experiments/demo_heldout.py --manual --no-intro
"""
import argparse
import json
import sys
import textwrap
import time
from pathlib import Path

from rich.console import Console, Group
from rich.live import Live
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

sys.path.insert(0, str(Path(__file__).resolve().parent))
import demo_compare as V1  # noqa: E402  (question_text, output_excerpt, verdict_from_row, intro)

ROOT = V1.ROOT
BENCH_NAME = {"gsm8k": "Math · GSM8K", "humanevalplus": "Code · HumanEval+"}
COLS = [("before", "BEFORE · unedited model", "red"), ("after", "AFTER · steered (probe direction)", "green"),
        ("control", "CONTROL · random direction, same strength", "yellow")]


def load_run(run: Path) -> dict:
    """{bench: {"rows": [{id, before, after, control}], "summary": {cond: summary row}, "alpha_frac": a}}."""
    summary = json.load(open(run / "summary.json"))
    out = {}
    for bench in ["gsm8k", "humanevalplus"]:
        conds = {v["condition"]: (k, v) for k, v in summary.items() if v["benchmark"] == bench}
        if not {"none", "probe_pos", "random_pos"} <= conds.keys():
            continue
        files = {"before": conds["none"][0], "after": conds["probe_pos"][0], "control": conds["random_pos"][0]}
        data = {c: {r["id"]: r for r in map(json.loads, open(run / f"{name}.jsonl"))} for c, name in files.items()}
        if bench == "gsm8k":
            # Current grading rule (reasoning_env.grade_gsm8k, REPORT.md audit 3b): a response whose thinking never
            # closed has no final answer and is wrong. Runs graded before that fix can carry a stale "correct".
            for rows in data.values():
                for r in rows.values():
                    r["correct"] = bool(r["correct"] and r["closed_think"])
        ids = [i for i in data["before"] if i in data["after"] and i in data["control"]]
        rows = [{"id": i, **{c: data[c][i] for c in data}} for i in ids]
        out[bench] = dict(rows=rows, summary=stats(rows), alpha_frac=conds["probe_pos"][1]["alpha_frac"])
    if not out:
        sys.exit(f"{run}: needs none, probe_pos and random_pos conditions")
    return out


def stats(rows: list) -> dict:
    """Per condition, computed from the rows (not the run's summary.json, which may predate a grader fix):
    accuracy, n, mean tokens, and for edited conditions fixed/broken/net and the exact McNemar p."""
    from scipy.stats import binomtest

    out = {}
    for c in ["before", "after", "control"]:
        s = dict(n=len(rows), accuracy=sum(r[c]["correct"] for r in rows) / len(rows),
                 mean_resp_tokens=sum(r[c]["n_resp_tokens"] for r in rows) / len(rows))
        if c != "before":
            f = sum(change(r["before"], r[c]) == "fixed" for r in rows)
            b = sum(change(r["before"], r[c]) == "broken" for r in rows)
            s.update(fixed=f, broken=b, net=f - b, mcnemar_p=binomtest(f, f + b, 0.5).pvalue if f + b else 1.0)
        out[c] = s
    return out


def change(before: dict, other: dict) -> str:
    """'fixed' (wrong -> right), 'broken' (right -> wrong) or '' relative to the unedited model."""
    if before["correct"] == other["correct"]:
        return ""
    return "fixed" if other["correct"] else "broken"


def pause(seconds: float, manual: bool) -> None:
    if manual:
        input()
    else:
        time.sleep(seconds)


# ------------------------------------------------------------------------------------------------
# Screen 2: one problem, three outputs
# ------------------------------------------------------------------------------------------------

def example_screen(bench: str, row: dict, k: int, n: int, console: Console) -> Panel:
    col_w = console.size.width // 3 - 6
    q = [w for ln in V1.question_text(row["before"]).splitlines()
         for w in (textwrap.wrap(ln, console.size.width - 6) or [""])]
    panels = []
    for c, title, style in COLS:
        r = row[c]
        ans, ok = V1.verdict_from_row(r)
        mark = Text(("✅ " if ok else "❌ ") + ans, style="bold green" if ok else "bold red")
        ch = change(row["before"], r) if c != "before" else ""
        tag = Text({"fixed": "  ↑ FIXED by this edit", "broken": "  ↓ BROKEN by this edit", "": ""}[ch],
                   style="bold green" if ch == "fixed" else "bold red")
        panels.append(Panel(Group(V1.output_excerpt(r, col_w, max_lines=7), Text(""), mark, tag,
                                  Text(f"{r['n_resp_tokens']:,} tokens of reasoning", style="grey50")),
                            title=title, border_style=style))
    grid = Table.grid(expand=True, padding=(0, 1))
    for _ in COLS:
        grid.add_column(ratio=1)
    grid.add_row(*panels)
    gold = row["before"].get("gold")
    head = Text(f"Held-out problem where steering changed the outcome ({k} of {n})", style="bold")
    foot = Text(f"{row['id']} · never used to train the probe or pick the strength"
                + (f" · reference answer {gold:g}" if isinstance(gold, float) else ""), style="grey50")
    return Panel(Group(head, Text(""), Text("\n".join(q[:9]) + ("\n…" if len(q) > 9 else "")), Text(""), grid, foot),
                 title=f"Input — {BENCH_NAME[bench]}", border_style="white")


# ------------------------------------------------------------------------------------------------
# Screens 3-4: reel and result
# ------------------------------------------------------------------------------------------------

def cell(before: dict, r: dict, is_before: bool) -> Text:
    ok, ch = bool(r["correct"]), ("" if is_before else change(before, r))
    t = Text(("✓ " if ok else "✗ ") + f"{r['n_resp_tokens']:>5,}", style="green" if ok else "red")
    if r["truncated"]:
        t.append(" T", style="grey50")
    if ch:
        t.append("  ↑ fixed" if ch == "fixed" else "  ↓ broken", style="bold green" if ch == "fixed" else "bold red")
    return t


def tally(run: dict, shown: dict) -> Table:
    """Running tally over the rows shown so far: accuracy and fixed/broken for steered and control."""
    t = Table(expand=True, header_style="bold")
    for h in ["", "before", "after (steered)", "control (random)"]:
        t.add_column(h)
    for bench, d in run.items():
        rows = d["rows"][: shown.get(bench, 0)]
        if not rows:
            continue
        acc = {c: sum(r[c]["correct"] for r in rows) for c in ["before", "after", "control"]}
        fb = {c: (sum(change(r["before"], r[c]) == "fixed" for r in rows),
                  sum(change(r["before"], r[c]) == "broken" for r in rows)) for c in ["after", "control"]}
        n = len(rows)
        t.add_row(BENCH_NAME[bench], f"{acc['before']}/{n} correct",
                  *[Text.assemble(f"{acc[c]}/{n} correct   ", (f"{fb[c][0]} fixed", "green"), " · ",
                                  (f"{fb[c][1]} broken", "red")) for c in ["after", "control"]])
    return t


def reel(run: dict, seconds_per_row: float, console: Console) -> None:
    total = sum(len(d["rows"]) for d in run.values())
    shown = {b: 0 for b in run}
    with Live(console=console, refresh_per_second=20, screen=True) as lv:
        for bench, d in run.items():
            for _ in d["rows"]:
                shown[bench] += 1
                lv.update(reel_view(run, shown, total, console))
                time.sleep(seconds_per_row)
        time.sleep(1.0)


def reel_view(run: dict, shown: dict, total: int, console: Console) -> Group:
    t = Table(expand=True, header_style="bold", box=None, padding=(0, 1))
    t.add_column("held-out problem")
    for _, title, style in COLS:
        t.add_column(Text(title, style=style))
    lines = []
    for bench, d in run.items():
        for r in d["rows"][: shown[bench]]:
            lines.append((r["id"], cell(r["before"], r["before"], True), cell(r["before"], r["after"], False),
                          cell(r["before"], r["control"], False)))
    room = max(console.size.height - 14, 5)
    for ln in lines[-room:]:
        t.add_row(*ln)
    done = sum(shown.values())
    head = Text(f"Every held-out problem, before / after / control   ({done}/{total})   "
                "✓/✗ = graded correct/wrong · number = tokens · T = hit the 4,096-token limit", style="bold")
    return Group(head, Text(""), t, Text(""), tally(run, shown))


def result_view(run: dict, run_dir: Path) -> Group:
    t = Table(expand=True, header_style="bold", title="Result on held-out problems", title_style="bold")
    for h in ["benchmark", "condition", "accuracy", "fixed", "broken", "McNemar p", "mean tokens"]:
        t.add_column(h)
    lines = []
    for bench, d in run.items():
        s = d["summary"]
        for c, label, style in [("before", "before (unedited)", "red"), ("after", "after (steered)", "green"),
                                ("control", "control (random)", "yellow")]:
            v = s[c]
            t.add_row(BENCH_NAME[bench] if c == "before" else "", Text(label, style=style),
                      f"{v['accuracy']:.0%}  (n={v['n']})", str(v.get("fixed", "")), str(v.get("broken", "")),
                      "" if c == "before" else f"{v['mcnemar_p']:.2f}", f"{v['mean_resp_tokens']:,.0f}")
        a, r = s["after"], s["control"]
        if a["mcnemar_p"] < 0.05 and a["net"] > r["net"]:
            msg = f"steering helped beyond the random control (net {a['net']:+d} vs {r['net']:+d}, p = {a['mcnemar_p']:.2f})"
        elif a["net"] > r["net"]:
            msg = (f"steering netted {a['net']:+d} vs {r['net']:+d} for the random control, "
                   f"but this is not statistically significant (p = {a['mcnemar_p']:.2f})")
        else:
            msg = f"steering did not beat the random control (net {a['net']:+d} vs {r['net']:+d})"
        lines.append(Text(f"{BENCH_NAME[bench]}: {msg}."))
    foot = Text(f"Recorded run: {run_dir.relative_to(ROOT)} · greedy decoding · same prompts for every condition · "
                "fixed/broken are relative to the unedited model · McNemar p = chance of a split this lopsided "
                "if the edit did nothing", style="grey50")
    return Group(t, Text(""), *lines, Text(""), foot)


# ------------------------------------------------------------------------------------------------
# Default flow: before/after response pairs
# ------------------------------------------------------------------------------------------------

def split_at_divergence(a: str, b: str) -> tuple:
    """(shared prefix, a's continuation, b's continuation), split at the first differing word."""
    wa, wb = a.split(" "), b.split(" ")
    k = next((i for i, (x, y) in enumerate(zip(wa, wb)) if x != y), min(len(wa), len(wb)))
    return " ".join(wa[:k]), " ".join(wa[k:]), " ".join(wb[k:])


def response_panel(row: dict, title: str, style: str, width: int, height: int) -> Panel:
    """End of the model's reasoning (dim), then what it handed back (bright), then the grade."""
    ans, ok = V1.verdict_from_row(row)
    if "</think>" in row["response"]:
        answer = V1.output_excerpt(row, width, max_lines=8)
    else:
        answer = Text("(no answer: still reasoning when it hit the 4,096-token limit)", style="bold red")
    thinking = row["response"].split("</think>")[0]
    lines = [w for ln in thinking.splitlines() if ln.strip() for w in textwrap.wrap(ln, width)]
    # borders 2 + "…reasoning" 1 + blank 1 + answer + blank 2 + verdict 1
    room = max(height - 7 - len(answer.plain.splitlines()), 3)
    body = Text("…reasoning\n", style="grey50")
    body.append("\n".join(lines[-room:]) + "\n\n", style="grey62")
    body.append_text(answer)
    end = Text("\n\n")
    end.append(("✅ " if ok else "❌ ") + ans, style="bold green" if ok else "bold red")
    end.append(f"   {row['n_resp_tokens']:,} tokens", style="grey50")
    return Panel(Group(body, end), title=title, border_style=style, height=height)


def pair_screen(bench: str, row: dict, k: int, n: int, console: Console) -> Group:
    w, h = console.size.width, console.size.height
    q = [x for ln in V1.question_text(row["before"]).splitlines() for x in (textwrap.wrap(ln, w - 4) or [""])]
    q = [x for x in q if x.strip()][:4]
    ph = max(h - len(q) - 4, 12)
    grid = Table.grid(expand=True, padding=(0, 1))
    grid.add_column(ratio=1)
    grid.add_column(ratio=1)
    grid.add_row(response_panel(row["before"], "BEFORE · chain-of-thought, unedited", "red", w // 2 - 6, ph),
                 response_panel(row["after"], "AFTER · same prompt, steered at layer 29", "green", w // 2 - 6, ph))
    ch = change(row["before"], row["after"])
    tag = {"fixed": ("  ↑ steering fixed it", "bold green"), "broken": ("  ↓ steering broke it", "bold red"),
           "": ("", "")}[ch]
    head = Text.assemble((f"{BENCH_NAME[bench]} · {row['id']}", "bold"), tag, (f"   {k}/{n}", "grey50"))
    return Group(head, Text("\n".join(q), style="white"), grid)


def pairs_reel(run: dict, seconds: float, max_pairs: int, manual: bool, console: Console) -> None:
    """Grade changes first (fixed, then broken), then the pairs whose text differs most, up to max_pairs."""
    rows = [(b, r) for b, d in run.items() for r in d["rows"]]
    order = {"fixed": 0, "broken": 1, "": 2}

    def diverge_at(r):  # earlier divergence = steering visibly changed more of the response
        return len(split_at_divergence(r["before"]["response"], r["after"]["response"])[0])

    rows.sort(key=lambda br: (order[change(br[1]["before"], br[1]["after"])], diverge_at(br[1])))
    rows = rows[:max_pairs]
    with Live(console=console, screen=True, refresh_per_second=4) as lv:
        for k, (bench, row) in enumerate(rows, 1):
            lv.update(pair_screen(bench, row, k, len(rows), console))
            pause(seconds, manual)
        lv.update(tally(run, {b: len(d["rows"]) for b, d in run.items()}))
        input()


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run", default="artifacts/intervene/heldout_half", help="intervention run directory")
    ap.add_argument("--full", action="store_true", help="old flow: intro, 3-column examples, reel, result table")
    ap.add_argument("--no-intro", action="store_true")
    ap.add_argument("--intro-seconds", type=float, default=9)
    ap.add_argument("--example-seconds", type=float, default=8, help="time on each example screen")
    ap.add_argument("--pair-seconds", type=float, default=4, help="default flow: time on each before/after pair")
    ap.add_argument("--max-pairs", type=int, default=12, help="default flow: number of pairs shown")
    ap.add_argument("--row-seconds", type=float, default=0.3, help="reel speed")
    ap.add_argument("--manual", action="store_true", help="advance screens with Enter")
    args = ap.parse_args()
    console, run_dir = Console(), ROOT / args.run
    run = load_run(run_dir)

    if not args.full:
        pairs_reel(run, args.pair_seconds, args.max_pairs, args.manual, console)
        return
    if not args.no_intro:
        V1.intro(console, args.intro_seconds)  # waits for Enter at the end, as in v1

    changed = [(b, r) for b, d in run.items() for r in d["rows"] if change(r["before"], r["after"])]
    with Live(console=console, screen=True) as lv:
        for k, (bench, row) in enumerate(changed, 1):
            lv.update(example_screen(bench, row, k, len(changed), console))
            pause(args.example_seconds, args.manual)
    reel(run, args.row_seconds, console)
    with Live(console=console, screen=True) as lv:
        lv.update(result_view(run, run_dir))
        input()
    console.print(result_view(run, run_dir))


if __name__ == "__main__":
    main()
