"""MOCKUP of demo v3: amplifying the model's "self-check" feature so it catches its own mistakes.

Everything here is hand-written to show what we expect the experiment to produce; no model is run and
no number on screen is a result. Traces are pre-scripted and replayed fast (the whole demo is under a
minute), instead of a 2-3 minute live decode.

    1. READ   a reasoning trace with a live self-check meter (probe score per word); it spikes when the
              model stops to verify a step.
    2. STEER  the same problem before / after amplifying that direction: before, the model makes an
              arithmetic slip and never checks; after, the meter spikes right after the slip and it fixes it.
    3. DOSE   strength sweep: too little does nothing, just enough raises accuracy, too much makes the
              model re-check forever and run out of tokens. Random direction as the control.

    python experiments/demo_v3_mock.py          # → / space next screen, ← previous, r replay, q quit
    python experiments/demo_v3_mock.py --auto   # plays straight through, no keys
"""
import argparse
import random
import sys
import termios
import time
import tty

from rich.align import Align
from rich.console import Console, Group
from rich.live import Live
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

NAV = Text("→ / space next   ← back   r replay   q quit", style="grey50")
SHOW_NAV = True
SPARK = "▁▂▃▄▅▆▇█"
WORD_DELAY = 0.11  # seconds per word
SPIKE_PAUSE = 1.2  # extra hold when the meter spikes or an error appears

# A trace is a list of (line, kind). kind: "" plain, "err" the slip, "check" the model verifying,
# "fix" the corrected step, "ans" the final answer. The meter score for each word comes from kind.
INTRO_Q = "A customer takes a $12,000 loan at 5% simple interest for 3 years. How much interest is owed?"
INTRO = [
    ("Simple interest is principal × rate × time.", ""),
    ("5% of 12,000 is 600, so that's $600 per year.", ""),
    ("Over 3 years: 600 × 3 = 1,800.", ""),
    ("Hmm, let me verify that.", "check"),
    ("12,000 × 0.05 × 3 = 1,800. Yes, it checks out.", "check"),
    ("The interest owed is $1,800.", "ans"),
]

STEER_Q = ("An account starts at $2,400. The customer withdraws $375 a month for 4 months, then deposits "
           "$600. What is the balance?")
BEFORE = [
    ("The account starts at $2,400.", ""),
    ("Withdrawals: $375 a month for 4 months.", ""),
    ("375 × 4 = 1,400.", "err"),
    ("After the withdrawals: 2,400 − 1,400 = 1,000.", ""),
    ("Then the $600 deposit: 1,000 + 600 = 1,600.", ""),
    ("The balance is $1,600.", "ans"),
]
AFTER = [
    ("The account starts at $2,400.", ""),
    ("Withdrawals: $375 a month for 4 months.", ""),
    ("375 × 4 = 1,400.", "err"),
    ("Wait, let me double-check that.", "check"),
    ("375 × 2 = 750, and 750 × 2 = 1,500. So it's 1,500, not 1,400.", "fix"),
    ("After the withdrawals: 2,400 − 1,500 = 900.", ""),
    ("Then the $600 deposit: 900 + 600 = 1,500.", ""),
    ("The balance is $1,500.", "ans"),
]

# (strength label, self-checks per response, accuracy steered, accuracy random, note)
DOSE = [
    ("0 (off)", 1.2, 85, 85, ""),
    ("low", 1.8, 85, 85, "too little"),
    ("medium", 3.4, 91, 84, "just enough"),
    ("high", 7.5, 83, 84, "over-checking"),
    ("very high", 19.0, 58, 82, "loops, runs out of tokens"),
]

STYLE = {"": "grey70", "err": "bold red", "check": "bold green", "fix": "bold green", "ans": "bold"}


def word_scores(trace, seed=0):
    """Mock probe score per word: low noise, high on checking/fixing words."""
    rng = random.Random(seed)
    out = []
    for line, kind in trace:
        for _ in line.split():
            base = {"check": 0.88, "fix": 0.62}.get(kind, 0.08)
            out.append(min(max(base + rng.uniform(-0.07, 0.07), 0.0), 1.0))
    return out


def render_trace(trace, n_words, scores, title, border, width, verdict=None):
    """First n_words words of the trace, colored by kind, with the self-check meter under it."""
    body, shown = Text(), 0
    for line, kind in trace:
        words = line.split()
        take = words[: max(n_words - shown, 0)]
        if not take:
            break
        body.append(" ".join(take) + "\n", style=STYLE[kind])
        if kind == "err" and len(take) == len(words):
            body.append("   ✗ slip\n", style="red")
        shown += len(take)
    seen = scores[:n_words]
    spark = Text()
    for s in seen[-(width - 4):]:
        spark.append(SPARK[min(int(s * len(SPARK)), len(SPARK) - 1)], style="bold green" if s > 0.5 else "grey42")
    now = seen[-1] if seen else 0.0
    gauge = Text.assemble(("self-check signal ", "grey62"), ("█" * int(now * 20), "bold green" if now > 0.5 else "grey50"),
                          ("░" * (20 - int(now * 20)), "grey23"), (f" {now:.2f}", "grey62"))
    if now > 0.5:
        gauge.append("  ← checking its work", style="bold green")
    parts = [body, Text(""), gauge, spark]
    if verdict:
        parts += [Text(""), verdict]
    return Panel(Group(*parts), title=title, border_style=border, width=width)


def n_words(trace):
    return sum(len(line.split()) for line, _ in trace)


def kind_at(trace, i):
    """Kind of the i-th word (0-based)."""
    for line, kind in trace:
        i -= len(line.split())
        if i < 0:
            return kind
    return ""


def header(step, title, sub):
    return Group(NAV if SHOW_NAV else Text(""), Text(""), Text.assemble((f"{step}  ", "bold cyan"), (title, "bold")), Text(sub, style="grey62"), Text(""))


def play_intro(lv, console):
    w = min(console.size.width - 2, 90)
    scores = word_scores(INTRO, 1)
    total = n_words(INTRO)
    for i in range(1, total + 1):
        lv.update(Group(header("1 · READ", "We can see the model check its own work",
                               "A linear probe on the model's internal state scores each word: is it verifying right now?"),
                        Text(INTRO_Q, style="bold"), Text(""),
                        render_trace(INTRO, i, scores, "model reasoning", "cyan", w)), refresh=True)
        time.sleep(WORD_DELAY + (SPIKE_PAUSE if kind_at(INTRO, i - 1) == "check" and kind_at(INTRO, i - 2) != "check" else 0))


def play_steer(lv, console):
    w = max((console.size.width - 4) // 2, 40)
    sb, sa = word_scores(BEFORE, 2), word_scores(AFTER, 3)
    nb, na = n_words(BEFORE), n_words(AFTER)
    for i in range(1, max(nb, na) + 1):
        vb = Text("❌ $1,600 (correct: $1,500)", style="bold red") if i >= nb else None
        va = Text("✅ $1,500", style="bold green") if i >= na else None
        grid = Table.grid(padding=(0, 1))
        grid.add_column(width=w)
        grid.add_column(width=w)
        grid.add_row(render_trace(BEFORE, min(i, nb), sb, "BEFORE · as shipped", "red", w, vb),
                     render_trace(AFTER, min(i, na), sa, "AFTER · self-check amplified", "green", w, va))
        lv.update(Group(header("2 · STEER", "Turn that signal up, and the model catches its own mistake",
                               "Same model, same prompt, no retraining: we add the self-check direction to its internal state."),
                        Text(STEER_Q, style="bold"), Text(""), grid), refresh=True)
        k = kind_at(AFTER, i - 1)
        pause = SPIKE_PAUSE if (k in ("err", "check") and kind_at(AFTER, i - 2) != k) else 0
        time.sleep(WORD_DELAY * 1.3 + pause)


def bar(v, lo=50, hi=95, n=24, style="green"):
    k = int((v - lo) / (hi - lo) * n)
    return Text.assemble(("█" * k, style), ("░" * (n - k), "grey23"), (f" {v}%", "bold" if style == "green" else "grey62"))


def play_dose(lv, console):
    for upto in range(1, len(DOSE) + 1):
        t = Table(expand=False, box=None, padding=(0, 2), header_style="grey62")
        t.add_column("strength")
        t.add_column("self-checks\nper answer", justify="right")
        t.add_column("accuracy · self-check direction")
        t.add_column("accuracy · random direction")
        t.add_column("")
        for label, checks, acc, rnd, note in DOSE[:upto]:
            best = note == "just enough"
            t.add_row(Text(label, style="bold" if best else ""), f"{checks:.1f}",
                      bar(acc, style="bold green" if best else "green" if acc >= 85 else "red"), bar(rnd, style="grey50"),
                      Text(note, style="bold green" if best else "red" if acc < 85 else "grey62"))
        foot = Text("\nToo little does nothing. Too much, and it second-guesses itself until it runs out of tokens.\n"
                    "We tune the strength to the sweet spot. A random direction does nothing at any strength.",
                    style="grey70") if upto == len(DOSE) else Text("")
        lv.update(Group(header("3 · DOSE", "Just enough: the sweet spot is measurable",
                               "Accuracy on held-out problems as we turn the self-check signal up."), t, foot), refresh=True)
        time.sleep(1.1)


SCREENS = [play_intro, play_steer, play_dose]


def read_key():
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
    ap.add_argument("--auto", action="store_true", help="play all screens straight through with short holds")
    args = ap.parse_args()
    global SHOW_NAV
    SHOW_NAV = not args.auto
    console = Console()
    with Live(console=console, screen=True, auto_refresh=False) as lv:
        k = 0
        while 0 <= k < len(SCREENS):
            SCREENS[k](lv, console)
            if args.auto:
                time.sleep(3)
                k += 1
                continue
            key = read_key()
            if key in ("q", "\x03"):
                break
            if key in ("\x1b[D", "h"):
                k = max(k - 1, 0)
            elif key != "r":
                if k == len(SCREENS) - 1:
                    break
                k += 1


if __name__ == "__main__":
    main()
