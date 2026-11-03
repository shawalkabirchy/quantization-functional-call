"""Sort every answer into a failure class, and write results/taxonomy_<mode>.csv.

Usage (from the repository root, in the BFCL environment):
    python src/taxonomy.py --mode free

Needs results/raw/ and results/scores/. A failure is syntactic if the answer is not valid JSON, or does not match the
schema that constrained decoding enforces (src/schema.py). So S1 + S2 is what constrained decoding can block.
Every other failure is semantic: a well-formed answer that BFCL's checker rejects.

    S1  not valid JSON
    S2  valid JSON that does not match the schema (wrong keys, unknown function, extra or missing argument, wrong type,
        bad enum value)
    M1  wrong function
    M2  wrong argument value
    M3  wrong number of calls
    M4  a call on an irrelevance question
    M5  no call (an empty array) when one was needed

"correct_unparsed" is an irrelevance answer that does not parse. BFCL counts it as correct, because it makes no call,
so it stays out of the failures but is counted on its own.
"""

import argparse
import csv
import json
from collections import Counter
from pathlib import Path

from jsonschema import Draft202012Validator

from data import NON_LIVE, load_answer_key, load_questions
from main_table import MODELS, PRECISIONS, REPO
from prompt_format import ParseError, parse_calls
from schema import calls_schema

FAILURES = ["S1", "S2", "M1", "M2", "M3", "M4", "M5"]
ARGUMENT_ERRORS = ("value_error:", "type_error:", "simple_function_checker:missing_",
                   "simple_function_checker:unexpected_param")


def called_names(items):
    """The function names in an answer key entry: [{"name": {...arguments...}}, ...]."""
    return Counter(name for item in items for name in item)


def classify(category, output, score, schema, ground_truth):
    """Returns the class of one answer. An unknown BFCL error type is an error, so nothing is filed by guesswork."""
    if not score["parse_ok"]:
        if score["valid"]:
            return "correct_unparsed"
        return "S1" if score["parse_error"].startswith("not valid JSON") else "S2"
    if score["valid"]:
        return "correct"
    try:
        calls = parse_calls(output)
    except ParseError as e:  # score says it parsed: the parser or the files have changed
        raise RuntimeError(f"{score['id']}: scored as parsed, but it does not parse") from e
    if not calls:
        return "M5"
    if not Draft202012Validator(schema).is_valid(calls):
        return "S2"
    if category == "irrelevance":
        return "M4"
    error = score["error_type"]
    if error.endswith(":wrong_count"):
        return "M3"
    if error.endswith(":wrong_func_name"):
        return "M1"
    if error.endswith(":cannot_find_match"):  # BFCL's parallel check does not say which part failed
        return "M1" if Counter(c["name"] for c in calls) != called_names(ground_truth) else "M2"
    if error.startswith(ARGUMENT_ERRORS):
        return "M2"
    raise RuntimeError(f"{score['id']}: unknown BFCL error type {error!r}")


def load_questions_and_keys():
    questions, keys = {}, {}
    for category in NON_LIVE:
        questions.update({q["id"]: q for q in load_questions(category)})
        keys.update(load_answer_key(category))
    return questions, keys


def read_jsonl(path):
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f]


def classify_run(run_id, results_dir, questions, keys):
    """Returns [(id, category, class)], one row per question."""
    rows = {r["id"]: r for r in read_jsonl(results_dir / "raw" / f"{run_id}.jsonl")}
    labelled = []
    for score in read_jsonl(results_dir / "scores" / f"{run_id}.jsonl"):
        question = questions[score["id"]]
        label = classify(score["category"], rows[score["id"]]["output"], score, calls_schema(question["functions"]),
                         keys[score["id"]]["ground_truth"])
        labelled.append((score["id"], score["category"], label))
    return labelled


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--mode", default="free", choices=["free", "constrained"])
    parser.add_argument("--results-dir", type=Path, default=REPO / "results")
    args = parser.parse_args()

    questions, keys = load_questions_and_keys()
    fields = ["run_id", "model", "precision", "mode", "category", "total", "correct", "correct_unparsed"] + FAILURES
    table = []
    print(f"{'run':30s} {'correct':>7s} " + " ".join(f"{c:>5s}" for c in FAILURES) + "   (1,150 AST questions + 240 irrelevance)")
    for model in MODELS:
        for precision in PRECISIONS:
            run_id = f"{model}_{precision}_{args.mode}"
            if not (args.results_dir / "scores" / f"{run_id}.jsonl").exists():
                continue
            labelled = classify_run(run_id, args.results_dir, questions, keys)
            for category in NON_LIVE:
                counts = Counter(label for _, c, label in labelled if c == category)
                table.append({"run_id": run_id, "model": model, "precision": precision, "mode": args.mode,
                              "category": category, "total": sum(counts.values()), "correct": counts["correct"],
                              "correct_unparsed": counts["correct_unparsed"], **{c: counts[c] for c in FAILURES}})
            counts = Counter(label for _, _, label in labelled)
            print(f"{run_id:30s} {counts['correct'] + counts['correct_unparsed']:7d} "
                  + " ".join(f"{counts[c]:5d}" for c in FAILURES))

    out = args.results_dir / f"taxonomy_{args.mode}.csv"
    with open(out, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(table)
    print(f"\nSaved {out}")


if __name__ == "__main__":
    main()
