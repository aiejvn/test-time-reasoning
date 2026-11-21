"""Label self-correction events in saved GSM8K reasoning traces (labels only; no probe is trained here).

Why this exists
---------------
probe.py's label is "was the final answer correct?". That is not the project's target (PROPOSAL.md
RQ1/RQ4: features for backtracking / error recovery). This script produces event-level labels for
*genuine self-correction* and a position for each event that is chosen so a probe can't just detect
correction words that have already been written.

Definitions
-----------
event        a reflection marker that starts a sentence or line inside <think> ... </think>
             ("Wait", "But wait", "Hmm", "Hold on", "Actually", "No,", "Oops", "Let me check/verify/recheck").
stated answer  a number the model explicitly commits to as the answer inside its reasoning:
             "\\boxed{X}", "the answer is X", "answer should be X", "final answer: X".
before / after  the last stated answer before the event, and the first stated answer after it.

label        (gold = GSM8K reference answer)
    genuine_correction  before != gold and after == gold   (a real error, fixed)
    harmful_revision    before == gold and after != gold
    failed_revision     before != gold, after != gold, after != before
    verify_kept_right   before == after == gold            (re-check, nothing to fix)
    verify_kept_wrong   before == after != gold            (re-check, error missed)
    unlabeled           no stated answer on one side, so the event can't be judged

position     `token_index` = index (within prompt+response tokens) of the LAST token before the
             marker. The marker itself is not in the probe's context window at this position.
             The residual there does predict the marker as the next token, which is why the intended
             contrast is genuine_correction vs. verify_* (both are followed by a marker), not
             "event vs. no event".

Known limits: stated answers are found by regex, so "before" can be an intermediate value the model
phrased like an answer. Labels must be spot-checked (see `context` in the output) before a probe is
trained on them.

Run
---
    python linear_probe/self_correction_events.py --responses output/baseline/baseline/gsm8k_dev.jsonl \
        --out artifacts/self_correction/gsm8k_dev_events.jsonl
"""
import argparse
import json
import re
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

MARKER = re.compile(r"(?:(?<=\n)|(?<=[.!?] ))(But wait|Wait|Hmm|Hold on|Actually|No,|Oops|"
                    r"Let me (?:double-)?check|Let me verify|Let me recheck)\b")
ANSWER = re.compile(r"(?:\\boxed\{\s*\\?\$?|answer is\s*\$?\\?\(?\s*|answer should be\s*\$?|final answer:?\s*\$?)"
                    r"(-?\d[\d,]*(?:\.\d+)?)")


def stated_answers(think: str):
    """[(char_offset, value)] for every explicitly stated answer in the reasoning."""
    return [(m.start(), float(m.group(1).replace(",", ""))) for m in ANSWER.finditer(think)]


def label(before, after, gold) -> str:
    if before is None or after is None:
        return "unlabeled"
    b, a = abs(before - gold) < 1e-6, abs(after - gold) < 1e-6
    if not b and a:
        return "genuine_correction"
    if b and not a:
        return "harmful_revision"
    if before == after:
        return "verify_kept_right" if b else "verify_kept_wrong"
    return "failed_revision"


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--responses", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--model-id", default="deepseek-ai/DeepSeek-R1-Distill-Llama-8B")  # tokenizer only
    args = ap.parse_args()
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(args.model_id)

    events = []
    for r in (json.loads(line) for line in open(ROOT / args.responses)):
        think = r["response"].split("</think>")[0]
        answers = stated_answers(think)
        n_prompt = len(tok(r["prompt"], add_special_tokens=False)["input_ids"])
        ends = [e for _, e in tok(r["response"], add_special_tokens=False, return_offsets_mapping=True)["offset_mapping"]]
        for m in MARKER.finditer(think):
            before = next((v for o, v in reversed(answers) if o < m.start()), None)
            after = next((v for o, v in answers if o > m.start()), None)
            # last response token ending at or before the marker's first character (the marker's leading
            # space/newline token, if separate, ends after its whitespace, which is still before the word)
            last = max(k for k, e in enumerate(ends) if e <= m.start())
            events.append(dict(id=r["id"], marker=m.group(1), char_offset=m.start(),
                               token_index=n_prompt + last,
                               before=before, after=after, gold=r["gold"], final_correct=r["correct"],
                               label=label(before, after, r["gold"]),
                               context=r["response"][max(0, m.start() - 300): m.start() + 300]))
    out = ROOT / args.out
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w") as f:
        for e in events:
            f.write(json.dumps(e) + "\n")
    counts = Counter(e["label"] for e in events)
    traces = {k: len({e["id"] for e in events if e["label"] == k}) for k in counts}
    print(json.dumps(dict(n_events=len(events), events_per_label=counts, traces_per_label=traces), indent=2))


if __name__ == "__main__":
    main()
