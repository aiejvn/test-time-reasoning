"""Side-by-side terminal demo: the unedited model vs. the model steered along the layer-29 probe direction.

Two panels stream the model's reasoning at the same token rate, so a run that finishes in fewer tokens finishes
sooner on screen. When both finish, each panel shows its final answer and whether it was graded correct.

Modes
-----
replay (default)  Plays back recorded responses from an intervention run (linear_probe/intervene.py output).
                  No GPU needed. What's shown is exactly what was logged, including the grade.
live              Loads the model and generates both answers now, as one batch of 2: row 0 unedited,
                  row 1 with h + alpha*u added at the layer's output (every token, as in intervene.py).
                  Batch composition changes greedy outputs (linear_probe/REPORT.md, audit 3a), so live
                  results can differ from the recorded ones.

The steering method is linear_probe/REPORT.md step 3 (PROPOSAL.md section 4.5): u = GSM8K last-prompt-token probe
direction at layer 29, alpha = 0.1 x median per-token activation norm at that layer.

Examples
--------
    python experiments/demo_compare.py                          # replay gsm8k/120
    python experiments/demo_compare.py --id HumanEval/143       # replay a held-out coding problem
    python experiments/demo_compare.py --before-after           # 2 inputs, output before vs. after steering
    python experiments/demo_compare.py --list                   # problems where the edit changed the grade
    python experiments/demo_compare.py --seconds 15             # make the replay take ~15 s
    python experiments/demo_compare.py --live --id gsm8k/120    # generate on the GPU now
    python experiments/demo_compare.py --live --question "A loan of $12,000 ..."   # ungraded free-form question
"""
import argparse
import json
import re
import sys
import textwrap
import threading
import time
from pathlib import Path

from rich.console import Console, Group
from rich.layout import Layout
from rich.live import Live
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

ROOT = Path(__file__).resolve().parent.parent

# Recorded runs to search, in order. The held-out run comes first so a held-out record wins if an id is in both.
RUNS = [
    ("artifacts/intervene/heldout_half", "test"),
    ("artifacts/intervene/sweep_dev", "dev"),
]
EDITED = "probe_pos_a0.1"
BUDGET = 4096  # max_new_tokens in every recorded run (configs/baseline.yaml)


# ------------------------------------------------------------------------------------------------
# Loading recorded pairs
# ------------------------------------------------------------------------------------------------

def _read(path: Path) -> dict:
    return {r["id"]: r for r in map(json.loads, open(path))}


def recorded_pairs() -> dict:
    """{id: (run_dir, split, unedited_row, edited_row)} for every problem in the recorded runs."""
    pairs = {}
    for run, split in RUNS:
        for f in sorted((ROOT / run).glob(f"*__{EDITED}.jsonl")):
            stem = f.name.split("__")[0]
            base, edit = _read(f.with_name(f"{stem}__none.jsonl")), _read(f)
            for pid in base:
                if pid in edit and pid not in pairs:
                    pairs[pid] = (run, split, base[pid], edit[pid])
    return pairs


def question_text(row: dict) -> str:
    """The user's question, taken from the logged prompt (chat tokens and the fixed instruction removed)."""
    p = row["prompt"]
    p = p.split("<｜User｜>", 1)[-1].split("<｜Assistant｜>", 1)[0]
    p = p.replace("Please reason step by step, and put your final answer within \\boxed{}.", "")
    p = p.replace("Complete the following Python function. Please reason step by step, then return the complete "
                  "function in a ```python block.", "")
    p = re.sub(r"^\s*```(?:python)?\s*|\s*```\s*$", "", p.strip())  # code prompts arrive fenced
    return p.strip()


# ------------------------------------------------------------------------------------------------
# Rendering
# ------------------------------------------------------------------------------------------------

class Side:
    """State of one panel: streamed text so far, token count, and the final verdict once done."""

    def __init__(self, title: str, style: str):
        self.title, self.style = title, style
        self.text, self.tokens, self.done, self.verdict = "", 0, False, None  # verdict: (answer str, correct|None)

    def render(self, height: int, width: int, steered: bool) -> Panel:
        body = Text()
        thinking = self.text.split("</think>")[0].replace("<think>", "")
        # wrap ourselves so the tail always fits and the meter/verdict below are never pushed out of the panel
        wrapped = [w for ln in thinking.splitlines() if ln.strip() for w in textwrap.wrap(ln, max(width - 4, 20))]
        for ln in wrapped[-max(height - 11, 3):]:
            body.append(ln + "\n", style="dim")
        bar_w = 30
        filled = min(bar_w, round(bar_w * self.tokens / BUDGET))
        meter = Text(f"\n{self.tokens:>5,} / {BUDGET:,} tokens  ")
        meter.append("█" * filled, style=self.style)
        meter.append("░" * (bar_w - filled), style="grey37")
        parts = [body, meter, layer_strip(steered)]
        if self.done and self.verdict:
            ans, ok = self.verdict
            mark = {True: ("✅ CORRECT", "bold green"), False: ("❌ WRONG", "bold red"), None: ("", "")}[ok]
            v = Text("\n\nAnswer: ", style="bold")
            v.append(ans, style="bold")
            v.append("   " + mark[0], style=mark[1])
            parts.append(v)
        elif not self.done:
            parts.append(Text("\n\nthinking…", style="italic grey50"))
        return Panel(Group(*parts), title=self.title, border_style=self.style, height=height)


N_LAYERS, LAYER = 32, 29


def layer_strip(steered: bool) -> Text:
    """The model's 32 layers as cells; on the steered side layer 29 is highlighted with the edit applied there."""
    t = Text("\nlayers  ")
    for i in range(N_LAYERS):
        t.append("█" if (steered and i == LAYER) else "▪", style="bold green" if (steered and i == LAYER) else "grey42")
    t.append(f"  ← +α·u at layer {LAYER}" if steered else "  (no edit)", style="green" if steered else "grey50")
    return t


def build_layout(question: str, left: Side, right: Side, footer: str, console: Console) -> Layout:
    q_lines = [w for ln in question.splitlines() for w in (textwrap.wrap(ln, console.size.width - 4) or [""])]
    q = "\n".join(q_lines[:12]) + ("\n…" if len(q_lines) > 12 else "")
    head_h = min(len(q_lines), 12) + 3
    body_h = max(console.size.height - head_h - 3, 14)
    col_w = console.size.width // 2
    lay = Layout()
    lay.split_column(Layout(Panel(Text(q), title="Question", border_style="white"), size=head_h),
                     Layout(name="body", size=body_h), Layout(Text(footer, style="grey50"), size=2))
    lay["body"].split_row(Layout(left.render(body_h, col_w, False)), Layout(right.render(body_h, col_w, True)))
    return lay


# ------------------------------------------------------------------------------------------------
# Intro: what we steer toward, how, and why it's cheap
# ------------------------------------------------------------------------------------------------

def method_facts() -> dict:
    """Every number shown in the intro, read from the committed probe and intervention outputs."""
    run_cfg = json.load(open(ROOT / RUNS[0][0] / "config.json"))
    math = json.load(open(ROOT / run_cfg["probe_dir"] / "results.json"))
    code = json.load(open(ROOT / "artifacts/probe/humanevalplus_dev/results.json"))
    key = f"{run_cfg['pool']}/L{run_cfg['layer']}"
    frac = run_cfg["alpha_fracs"][0]
    return dict(layer=run_cfg["layer"], n_math=math["n"], auroc_math=math["results"][key]["cv_auroc"],
                n_code=code["n"], auroc_code=code["results"][key]["cv_auroc"],
                frac=frac, alpha=frac * run_cfg["resid_norm"], norm=run_cfg["resid_norm"])


def intro(console: Console, seconds: float) -> None:
    """Three steps revealed one at a time: READ the correctness direction, STEER along it, why it SCALES."""
    f = method_facts()
    strip = Text("  layers  ")
    for i in range(N_LAYERS):
        strip.append("█" if i == f["layer"] else "▪", style="bold green" if i == f["layer"] else "grey42")
    strip.append(f"   layer {f['layer']} of {N_LAYERS}", style="green")

    read = Text.assemble(
        ("1  READ  ", "bold cyan"), "find the direction inside the model that means ", ("\"this answer will be correct\"", "bold"),
        "\n\n", strip, "\n\n",
        f"   We recorded the model's internal state (4,096 numbers) at layer {f['layer']} on {f['n_math']} math problems\n"
        f"   it got right or wrong, and fit a linear probe to tell them apart. Its weight vector u points toward \"correct\".\n\n",
        "   The state predicts correctness ", ("before the model writes a word", "bold"),
        f":  AUROC {f['auroc_math']:.2f} math · {f['auroc_code']:.2f} code\n   ",
        ("(0.5 = chance, 1 = perfect; cross-validated)", "grey50"))
    steer = Text.assemble(
        ("2  STEER  ", "bold green"), "nudge the model along that direction while it thinks", "\n\n",
        "   At every token, at layer ", (str(f["layer"]), "bold"), ":\n\n",
        ("        h  ←  h + α · u", "bold green"),
        f"        α = {f['alpha']:.1f}  ({f['frac']:g}× a typical activation's size, {f['norm']:.1f})\n\n",
        "   Same model, same weights, same prompt. Only the internal state is shifted.")
    scale = Text.assemble(
        ("3  SCALE  ", "bold magenta"), "why this is cheap to deploy", "\n\n",
        "   • No retraining, no fine-tuning, no change to the weights\n",
        "   • The whole intervention is one 4,096-number vector added at one layer:\n"
        "     ~4 thousand additions per token, next to ~8 billion multiply-adds the 8B model already does\n",
        "   • The direction was learned from math only and is applied to code unchanged\n",
        "   • Same recipe for any open-weight transformer: record states → fit probe → add the vector")
    steps = [Panel(read, border_style="cyan"), Panel(steer, border_style="green"), Panel(scale, border_style="magenta")]
    title = Text("Steering a reasoning model toward correct answers, at inference time\n"
                 "DeepSeek-R1-Distill-Llama-8B", style="bold", justify="center")
    with Live(console=console, refresh_per_second=10, screen=True) as lv:
        for k in range(1, len(steps) + 1):
            lv.update(Group(title, Text(""), *steps[:k]))
            time.sleep(seconds / len(steps))
        input()  # hold until Enter, then the side-by-side run starts


def steered_title() -> str:
    f = method_facts()
    return f"STEERED — h + {f['alpha']:.1f}·u at layer {f['layer']}, toward \"correct\""


def verdict_from_row(row: dict) -> tuple:
    """(answer string, correct) for a recorded row."""
    trunc = " (ran out of token budget)" if row["truncated"] else ""
    if row["benchmark"] == "gsm8k":
        pred = row.get("pred")
        ans = "none" if pred is None else (f"{pred:g}" if isinstance(pred, float) else str(pred))
        return ans + trunc, bool(row["correct"])
    return ("passes unit tests" if row["correct"] else "fails unit tests") + trunc, bool(row["correct"])


# ------------------------------------------------------------------------------------------------
# Replay
# ------------------------------------------------------------------------------------------------

def replay(pid: str, seconds: float, console: Console) -> None:
    pairs = recorded_pairs()
    if pid not in pairs:
        sys.exit(f"{pid} not in recorded runs. Try --list.")
    run, split, base, edit = pairs[pid]
    left, right = Side("BASELINE — unedited model", "red"), Side(steered_title(), "green")
    rows = [(left, base), (right, edit)]
    gold = base.get("gold")
    footer = (f"Recorded responses (greedy decoding) from {run}, {split} split, replayed at "
              f"equal token speed." + (f"  Reference answer: {gold:g}" if isinstance(gold, float) else ""))
    longest = max(r["n_resp_tokens"] for _, r in rows)
    rate = longest / seconds  # tokens per second, same for both sides
    t0 = time.time()
    with Live(build_layout(question_text(base), left, right, footer, console), console=console,
              refresh_per_second=20, screen=True) as live:
        while not all(s.done for s, _ in rows):
            t = (time.time() - t0) * rate
            for side, row in rows:
                n = row["n_resp_tokens"]
                side.tokens = min(int(t), n)
                # approximate token positions by character fraction (the tokenizer isn't loaded in replay)
                side.text = row["response"][: int(len(row["response"]) * side.tokens / n)]
                if side.tokens >= n and not side.done:
                    side.done, side.verdict = True, verdict_from_row(row)
            live.update(build_layout(question_text(base), left, right, footer, console))
            time.sleep(0.05)
        input()  # hold the final screen until Enter
    print_summary(pid, rows, gold, console)


def output_excerpt(row: dict, width: int, max_lines: int = 9) -> Text:
    """What the model handed back: the boxed answer line (math), the function with its docstring removed (code),
    or, if it never finished thinking, its last few lines of reasoning."""
    resp = row["response"]
    if "</think>" not in resp:
        tail = [w for ln in resp.splitlines() if ln.strip() for w in textwrap.wrap(ln, width)][-3:]
        t = Text("(no answer: still reasoning at the 4,096-token limit)\n…", style="italic")
        t.append("\n".join(tail), style="dim")
        return t
    final = resp.rsplit("</think>", 1)[-1]
    if row["benchmark"] == "gsm8k":
        lines = [ln for ln in final.splitlines() if "\\boxed" in ln] or final.strip().splitlines()[-1:]
        return Text("\n".join(w for ln in lines[-1:] for w in textwrap.wrap(ln.strip(), width)))
    code = re.sub(r'\n\s*"""[\s\S]*?"""', "", row.get("code") or final, count=1)  # drop the given docstring
    lines = [ln for ln in code.splitlines() if ln.strip()]
    shown = lines[:max_lines] + ([f"… ({len(lines) - max_lines} more lines)"] if len(lines) > max_lines else [])
    return Text("\n".join(ln[:width] for ln in shown))


def before_after(ids: list, console: Console) -> None:
    """Static screen: for each problem, the input, then the output before and after steering side by side."""
    pairs = recorded_pairs()
    missing = [i for i in ids if i not in pairs]
    if missing:
        sys.exit(f"not in recorded runs: {missing}. Try --list.")
    col_w = console.size.width // 2 - 6
    blocks = []
    for n, pid in enumerate(ids, 1):
        run, split, base, edit = pairs[pid]
        kind = "math (GSM8K)" if base["benchmark"] == "gsm8k" else "code (HumanEval+)"
        q = [w for ln in question_text(base).splitlines() for w in (textwrap.wrap(ln, console.size.width - 6) or [""])]
        q_text = Text("\n".join(q[:8]) + ("\n…" if len(q) > 8 else ""))
        cols = []
        for label, style, row in [("BEFORE steering", "red", base), ("AFTER steering", "green", edit)]:
            ans, ok = verdict_from_row(row)
            mark = Text(("✅ " if ok else "❌ ") + ans, style="bold green" if ok else "bold red")
            cols.append(Panel(Group(output_excerpt(row, col_w), Text(""), mark,
                                    Text(f"{row['n_resp_tokens']:,} tokens of reasoning", style="grey50")),
                              title=label, border_style=style))
        grid = Table.grid(expand=True, padding=(0, 1))
        grid.add_column(ratio=1)
        grid.add_column(ratio=1)
        grid.add_row(*cols)
        gold = base.get("gold")
        foot = Text(f"{pid} · {split} split · {run}" + (f" · reference answer {gold:g}" if isinstance(gold, float) else ""),
                    style="grey50")
        blocks.append(Panel(Group(Text("Input", style="bold"), q_text, Text(""), grid, foot),
                            title=f"Input {n} — {kind}", border_style="white"))
    console.print(Text(steered_title().replace("STEERED — ", "Steering: "), style="bold green", justify="center"))
    for b in blocks:
        console.print(b)


def print_summary(pid: str, rows, gold, console: Console) -> None:
    for side, row in rows:
        ans, ok = side.verdict
        console.print(f"[{side.style}]{side.title}[/]: {ans}  {'✅' if ok else '❌' if ok is False else ''}  "
                      f"({side.tokens:,} tokens)")
    if isinstance(gold, float):
        console.print(f"Reference answer for {pid}: {gold:g}")


# ------------------------------------------------------------------------------------------------
# Live generation
# ------------------------------------------------------------------------------------------------

def live(pid, question, console: Console) -> None:
    import torch
    from transformers import StoppingCriteria, StoppingCriteriaList

    sys.path.insert(0, str(ROOT / "src"))
    import reasoning_env as R
    import sae_env

    run_cfg = json.load(open(ROOT / RUNS[0][0] / "config.json"))
    layer, norm = run_cfg["layer"], run_cfg["resid_norm"]
    alpha = 0.1 * norm
    u = torch.load(ROOT / run_cfg["probe_dir"] / "directions.pt")[run_cfg["pool"]][layer].float()
    u = u / u.norm()

    item, bench = None, "gsm8k"
    if pid:
        bench = "gsm8k" if pid.startswith("gsm8k") else "humanevalplus"
        splits = R.load_benchmark(bench, ROOT / "configs/splits.json")
        item = next((it for s in splits.values() for it in s if it["id"] == pid), None)
        if item is None:
            sys.exit(f"{pid} not found in {bench}")
        question = item["question"]
    prompt = R.PROMPTS[bench]["cot"].format(question=question)

    console.print(f"Loading {R.MODEL_ID} …")
    model, tok = R.load_model(R.MODEL_ID)
    ids = sae_env._prompt_ids(tok, [{"role": "user", "content": prompt}])
    input_ids = torch.tensor([ids, ids], device=model.device)  # identical prompts: no padding

    def edit_row1(h):  # steer only the second row of the batch
        h = h.clone()
        h[1:] = h[1:] + (alpha * u).to(h.dtype).to(h.device)
        return h

    eos = model.generation_config.eos_token_id
    eos = {eos} if isinstance(eos, int) else set(eos or [tok.eos_token_id])
    left, right = Side("BASELINE — unedited model", "red"), Side(steered_title(), "green")
    sides = [left, right]
    footer = f"Generating now (greedy, batch of 2). Steering: h + {alpha:.2f}·u at layer {layer}, every token."
    q_shown = question_text({"prompt": f"<｜User｜>{prompt}<｜Assistant｜>"})
    finished = [None, None]  # response token ids per row once it hits EOS

    class Stream(StoppingCriteria):
        def __call__(self, seq, scores, **kw):
            for b, side in enumerate(sides):
                if finished[b] is None:
                    resp = seq[b, len(ids):].tolist()
                    if resp and resp[-1] in eos:
                        finished[b] = resp[:-1]
                        side.done = True
                    side.tokens = len(resp)
                    side.text = tok.decode(resp, skip_special_tokens=True)
            return all(f is not None for f in finished)

    result = {}

    def run():
        with torch.no_grad(), R.intervene(model, {layer: edit_row1}, positions="all"):
            result["gen"] = model.generate(input_ids=input_ids, attention_mask=torch.ones_like(input_ids),
                                           max_new_tokens=BUDGET, do_sample=False, pad_token_id=tok.pad_token_id,
                                           stopping_criteria=StoppingCriteriaList([Stream()]))

    th = threading.Thread(target=run)
    th.start()
    with Live(build_layout(q_shown, left, right, footer, console), console=console, refresh_per_second=10,
              screen=True) as lv:
        while th.is_alive():
            lv.update(build_layout(q_shown, left, right, footer, console))
            time.sleep(0.1)
        texts = []
        for b, side in enumerate(sides):
            resp = finished[b] if finished[b] is not None else result["gen"][b, len(ids):].tolist()
            texts.append(tok.decode(resp, skip_special_tokens=True))
            side.tokens, side.done = len(resp), True
        if item is not None and bench == "gsm8k":
            grades = [R.grade_gsm8k(item, t) for t in texts]
        elif item is not None:
            grades = R.grade_humanevalplus([item, item], texts)
        else:
            grades = [None, None]
        for side, t, g in zip(sides, texts, grades):
            trunc = side.tokens >= BUDGET
            if g is None:  # free-form question: show the boxed answer, ungraded
                m = re.findall(r"\\boxed\{([^{}]*)\}", R.final_region(t))
                side.verdict = ((m[-1] if m else "no boxed answer") + (" (ran out of token budget)" if trunc else ""),
                                None)
            else:
                side.verdict = verdict_from_row(dict(g, benchmark=bench, truncated=trunc))
        lv.update(build_layout(q_shown, left, right, footer, console))
        input()
    gold = float(item["answer"]) if item is not None and bench == "gsm8k" else None
    print_summary(pid or "question", list(zip(sides, texts)), gold, console)


# ------------------------------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--id", default="gsm8k/120", help="problem id, e.g. gsm8k/120 or HumanEval/143")
    ap.add_argument("--seconds", type=float, default=25, help="replay: duration of the longer response")
    ap.add_argument("--list", action="store_true", help="list recorded problems where the edit changed the grade")
    ap.add_argument("--live", action="store_true", help="generate on the GPU instead of replaying")
    ap.add_argument("--question", help="live: free-form question (ungraded) instead of --id")
    ap.add_argument("--before-after", nargs="*", metavar="ID",
                    help="static screen: input + output before/after steering for each ID "
                         "(default: gsm8k/120 HumanEval/143)")
    ap.add_argument("--no-intro", action="store_true", help="skip the read / steer / scale explainer screen")
    ap.add_argument("--intro-seconds", type=float, default=15, help="time to reveal the explainer's three steps")
    args = ap.parse_args()
    console = Console()

    if args.list:
        for pid, (run, split, b, e) in recorded_pairs().items():
            if b["correct"] != e["correct"]:
                console.print(f"{pid:<16} {split:<5} baseline {'✅' if b['correct'] else '❌'} {b['n_resp_tokens']:>5} tok"
                              f"   steered {'✅' if e['correct'] else '❌'} {e['n_resp_tokens']:>5} tok   ({run})")
        return
    if args.before_after is not None:
        before_after(args.before_after or ["gsm8k/120", "HumanEval/143"], console)
        return
    if not args.no_intro:
        intro(console, args.intro_seconds)
    if args.live:
        live(None if args.question else args.id, args.question, console)
    else:
        replay(args.id, args.seconds, console)


if __name__ == "__main__":
    main()
