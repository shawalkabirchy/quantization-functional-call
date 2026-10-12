"""Score raw outputs with BFCL's own AST checker.

Usage (from the repository root, in the BFCL environment):
    python src/score.py results/raw/<run_id>.jsonl [more raw files ...]

For each run this writes results/scores/<run_id>.jsonl (one line per question: whether the reply parsed,
whether it is correct, and BFCL's error type) and updates results/summary.csv (accuracy per category
plus BFCL's summary scores). Scoring the same run again replaces its rows in the summary.
"""

import argparse
import csv
import json
from pathlib import Path

from bfcl_eval.constants.enums import Language
from bfcl_eval.eval_checker.ast_eval.ast_checker import ast_checker

from data import NON_LIVE, language, load_answer_key
from prompt_format import ParseError, parse_calls, to_bfcl

LANGUAGES = {"python": Language.PYTHON, "java": Language.JAVA, "javascript": Language.JAVASCRIPT}

# BFCL's checker uses the model name only to look up whether that model's dots in function names were
# replaced by underscores. Our prompt keeps the original names, so we pass a BFCL model that keeps them.
CHECKER_MODEL_NAME = "Qwen/Qwen3-4B-Instruct-2507"

SIMPLE = ["simple_python", "simple_java", "simple_javascript"]
SUMMARY_FIELDS = ["run_id", "model", "precision", "mode", "category",
                  "correct", "total", "accuracy", "parsed_pct", "complete"]


def score_reply(category, output, key):
    """Score one reply. Returns a dict with parse_ok, valid, error_type and error."""
    try:
        calls, parse_error = parse_calls(output), None
    except ParseError as e:
        calls, parse_error = None, str(e)
    result = {"parse_ok": calls is not None, "parse_error": parse_error,
              "n_calls": len(calls) if calls is not None else None}

    if category == "irrelevance":
        # As in BFCL: correct when the reply contains no call, i.e. an empty array or no parseable call.
        # Whether it parsed is kept in parse_ok, for the failure taxonomy later.
        valid = not calls
        result.update(valid=valid, error_type=None if valid else "irrelevance_error:decoder_success",
                      error=None if valid else ["Made a call when no function fits the question."])
        return result

    if calls is None:
        result.update(valid=False, error_type="parse_error", error=[parse_error])
        return result

    try:
        check = ast_checker(key["functions"], to_bfcl(calls), key["ground_truth"],
                            LANGUAGES[language(category)], category, CHECKER_MODEL_NAME)
    except Exception as e:  # BFCL's Java/JavaScript type converters can raise on unusual values
        check = {"valid": False, "error": [f"{type(e).__name__}: {e}"], "error_type": "checker_exception"}
    valid = check["valid"]
    result.update(valid=valid, error_type=None if valid else check.get("error_type"),
                  error=None if valid else check.get("error"))
    return result


def score_run(raw_path, results_dir):
    rows = [json.loads(line) for line in open(raw_path, encoding="utf-8")]
    run_id = raw_path.stem
    meta_path = results_dir / "meta" / f"{run_id}.json"
    meta = json.load(open(meta_path, encoding="utf-8")) if meta_path.exists() else {}

    keys, scored = {}, []
    for row in rows:
        category = row["category"]
        if category not in keys:
            keys[category] = load_answer_key(category)
        result = score_reply(category, row["output"], keys[category][row["id"]])
        scored.append({"id": row["id"], "category": category, **result})

    scores_path = results_dir / "scores" / f"{run_id}.jsonl"
    scores_path.parent.mkdir(parents=True, exist_ok=True)
    with open(scores_path, "w", encoding="utf-8") as f:
        for s in scored:
            f.write(json.dumps(s, ensure_ascii=False) + "\n")

    # Accuracy per category
    base = {"run_id": run_id, "model": meta.get("model", ""), "precision": meta.get("precision", ""),
            "mode": meta.get("mode", "")}
    summary, accuracy, complete = [], {}, {}
    for category in [c for c in NON_LIVE if c in keys]:
        items = [s for s in scored if s["category"] == category]
        correct = sum(s["valid"] for s in items)
        accuracy[category] = 100 * correct / len(items)
        complete[category] = len(items) == len(keys[category])
        summary.append({**base, "category": category, "correct": correct, "total": len(items),
                        "accuracy": round(accuracy[category], 2),
                        "parsed_pct": round(100 * sum(s["parse_ok"] for s in items) / len(items), 2),
                        "complete": complete[category]})

    # BFCL's summary scores, computed the same way as its data_non_live.csv:
    # simple = unweighted mean of the three languages; Non-Live Overall Acc = unweighted mean of simple,
    # multiple, parallel and parallel_multiple (irrelevance is reported on its own, not in the overall).
    if all(complete.get(c) for c in SIMPLE):
        accuracy["simple_ast"] = sum(accuracy[c] for c in SIMPLE) / len(SIMPLE)
        summary.append({**base, "category": "simple_ast", "accuracy": round(accuracy["simple_ast"], 2),
                        "complete": True})
        others = ["multiple", "parallel", "parallel_multiple"]
        if all(complete.get(c) for c in others):
            overall = sum(accuracy[c] for c in ["simple_ast"] + others) / 4
            summary.append({**base, "category": "non_live_overall", "accuracy": round(overall, 2),
                            "complete": True})

    summary_path = results_dir / "summary.csv"
    old = []
    if summary_path.exists():
        with open(summary_path, newline="", encoding="utf-8") as f:
            old = [r for r in csv.DictReader(f) if r["run_id"] != run_id]
    with open(summary_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=SUMMARY_FIELDS)
        writer.writeheader()
        writer.writerows(old + summary)

    print(f"\n{run_id}  ({len(scored)} questions)")
    print(f"  {'category':20s} {'correct':>9s} {'accuracy':>9s} {'parsed':>8s}")
    for r in summary:
        counts = f"{r['correct']}/{r['total']}" if "total" in r else ""
        parsed = f"{r['parsed_pct']:.2f}" if "parsed_pct" in r else ""
        print(f"  {r['category']:20s} {counts:>9s} {r['accuracy']:9.2f} {parsed:>8s}")
    print(f"  saved {scores_path} and {summary_path}")
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("raw_files", nargs="+", type=Path, help="results/raw/<run_id>.jsonl files")
    parser.add_argument("--results-dir", type=Path, default=None,
                        help="folder with meta/, scores/ and summary.csv (default: two levels above the raw file)")
    args = parser.parse_args()
    for raw_path in args.raw_files:
        score_run(raw_path, args.results_dir or raw_path.resolve().parents[1])


if __name__ == "__main__":
    main()
