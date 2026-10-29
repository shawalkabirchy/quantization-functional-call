"""Build results/main_table.md: accuracy of every core model at every precision, from results/summary.csv.

Usage (from the repository root; needs only the Python standard library):
    python src/main_table.py
    python src/main_table.py --mode constrained

A run counts only if it answered all questions (BFCL's Non-Live Overall Acc is in summary.csv). Missing runs
are shown as "–".
"""

import argparse
import csv
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]

MODELS = {  # config name prefix -> name in the table
    "qwen3-1.7b": "Qwen3-1.7B",
    "qwen3-4b": "Qwen3-4B",
    "qwen3-8b": "Qwen3-8B",
    "llama-3.2-1b": "Llama-3.2-1B-Instruct",
    "llama-3.2-3b": "Llama-3.2-3B-Instruct",
}
PRECISIONS = {"fp16": "FP16", "int8": "INT8", "nf4": "NF4", "gptq": "GPTQ", "awq": "AWQ"}


def table(rows, header):
    lines = ["| " + " | ".join(header) + " |", "|---" + "|---:" * (len(header) - 1) + "|"]
    lines += ["| " + " | ".join(row) + " |" for row in rows]
    return lines


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--mode", default="free", choices=["free", "constrained"])
    parser.add_argument("--results-dir", type=Path, default=REPO / "results")
    args = parser.parse_args()

    with open(args.results_dir / "summary.csv", newline="", encoding="utf-8") as f:
        summary = {(r["run_id"], r["category"]): float(r["accuracy"]) for r in csv.DictReader(f)}

    def accuracy(model, precision, category):
        return summary.get((f"{model}_{precision}_{args.mode}", category))

    def fmt(value, signed=False):
        return "–" if value is None else f"{value:+.2f}" if signed else f"{value:.2f}"

    overall, change, irrelevance = [], [], []
    for model, name in MODELS.items():
        acc = {p: accuracy(model, p, "non_live_overall") for p in PRECISIONS}
        overall.append([name] + [fmt(acc[p]) for p in PRECISIONS])
        change.append([name] + [fmt(None if acc[p] is None or acc["fp16"] is None else acc[p] - acc["fp16"],
                                    signed=True) for p in list(PRECISIONS)[1:]])
        # irrelevance only counts if the run answered every question
        irrelevance.append([name] + [fmt(accuracy(model, p, "irrelevance") if acc[p] is not None else None)
                                     for p in PRECISIONS])

    labels = list(PRECISIONS.values())
    lines = [
        f"# Main Table: Accuracy by Precision ({args.mode} decoding)", "",
        "BFCL `non_live` (1,390 questions per run), our prompt format and scorer (`src/run_eval.py`, "
        "`src/score.py`), greedy decoding. Built from `results/summary.csv` by `src/main_table.py`; "
        "a missing run is shown as –.", "",
        "## Non-Live Overall accuracy (%)", "",
        "BFCL's Non-Live Overall Acc: the unweighted mean of simple (itself the mean of Python, Java and "
        "JavaScript), multiple, parallel and parallel-multiple.", "",
    ] + table(overall, ["Model"] + labels) + [
        "", "## Change from FP16 (percentage points)", "",
    ] + table(change, ["Model"] + labels[1:]) + [
        "", "## Irrelevance accuracy (%)", "",
        "Questions where no function fits; correct = no call. Reported apart from the overall score, as in BFCL.", "",
    ] + table(irrelevance, ["Model"] + labels)

    suffix = "" if args.mode == "free" else f"_{args.mode}"
    out = args.results_dir / f"main_table{suffix}.md"
    out.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))
    print(f"\nSaved {out}")


if __name__ == "__main__":
    main()
