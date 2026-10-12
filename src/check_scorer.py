"""Check score.py against BFCL's own scoring, using the Update 2 FC-mode answers.

The Update 2 answers are <tool_call>{"name": ..., "arguments": ...}</tool_call> blocks. They are decoded
the same way BFCL's Qwen FC handler does it, rewritten in our JSON-array format, scored with score.py,
and every question's verdict is compared with BFCL's score files. No GPU needed.

Usage (from the repository root, in the BFCL environment):
    python src/check_scorer.py

Needs the Update 2 raw answers in results/raw/bfcl_repro/ (not in git; they are in the Update 2 zip).
Result on 2026-10-12: 1,390 of 1,390 verdicts match, overall 87.81% as in BFCL.
"""

import json
import re
import sys
import tempfile
from pathlib import Path

from data import NON_LIVE
from score import score_run

REPO = Path(__file__).resolve().parents[1]
RESULT_DIR = REPO / "results/raw/bfcl_repro/result/Qwen_Qwen3-4B-Instruct-2507-FC/non_live"
SCORE_DIR = REPO / "results/bfcl_repro/score/Qwen_Qwen3-4B-Instruct-2507-FC/non_live"


def bfcl_fc_to_ours(text):
    """Decode like BFCL's QwenFCHandler, then write the calls in our format."""
    calls = []
    for block in re.findall(r"<tool_call>\n(.*?)\n</tool_call>", text, re.DOTALL):
        try:
            calls.append(json.loads(block))
        except json.JSONDecodeError:
            pass  # BFCL skips blocks that are not valid JSON
    if any(type(c) != dict for c in calls):
        return "<BFCL cannot decode this answer>"
    try:
        return json.dumps([{"name": c["name"], "arguments": dict(c["arguments"].items())} for c in calls])
    except (KeyError, AttributeError):
        return "<BFCL cannot decode this answer>"


def main():
    if not RESULT_DIR.exists():
        sys.exit(f"Missing {RESULT_DIR}: unzip the Update 2 results there first.")
    work = Path(tempfile.mkdtemp())
    raw = work / "raw" / "update2_fc_rescored.jsonl"
    raw.parent.mkdir()
    with open(raw, "w", encoding="utf-8") as f:
        for category in NON_LIVE:
            for line in open(RESULT_DIR / f"BFCL_v4_{category}_result.json", encoding="utf-8"):
                row = json.loads(line)
                f.write(json.dumps({"id": row["id"], "category": category,
                                    "output": bfcl_fc_to_ours(row["result"])}) + "\n")

    score_run(raw, work)
    ours = {}
    for line in open(work / "scores" / raw.name, encoding="utf-8"):
        s = json.loads(line)
        ours[s["id"]] = s["valid"]

    mismatches = []
    for category in NON_LIVE:
        lines = open(SCORE_DIR / f"BFCL_v4_{category}_score.json", encoding="utf-8").read().splitlines()
        bfcl_failed = {json.loads(line)["id"] for line in lines[1:]}  # first line is the summary
        for qid in (q for q in ours if q.rsplit("_", 1)[0] == category):
            if ours[qid] != (qid not in bfcl_failed):
                mismatches.append(qid)
    print(f"\nVerdicts that differ from BFCL: {len(mismatches)} of {len(ours)}", mismatches[:10])
    sys.exit(1 if mismatches else 0)


if __name__ == "__main__":
    main()
