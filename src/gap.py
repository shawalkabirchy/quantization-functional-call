"""Compare free and constrained decoding: how much does the constraint change each quantization penalty?

Usage (from the repository root, in the BFCL environment; needs results/raw/, results/scores/ and
results/significance_free.csv from src/significance.py):
    python src/gap.py

Writes two tables:

results/gap.csv, one row per quantized run (20):
    change_free         Overall(quantized) - Overall(FP16) with free decoding, as in results/significance_free.csv
    change_constrained  the same with constrained decoding
    difference          change_constrained - change_free (points): above 0, the constraint makes the loss smaller
                        or the gain larger. Bootstrap 95% interval: the same resampled questions for all four runs,
                        drawn inside each category, 10,000 times (seed 42)
    loss_removed_pct    difference / -change_free, in %, only where free decoding gave a significant loss
                        (Holm-adjusted p < 0.05): 100 = the loss is gone, 0 = unchanged, below 0 = larger

results/transitions.csv, one row per run (25), over the 1,150 AST questions: what happens under the constraint to the
answers that failed on format (S1, S2) in free decoding, and how many right answers it breaks.
"""

import csv
from collections import Counter

import numpy as np

from main_table import MODELS, PRECISIONS, REPO
from significance import CATEGORIES, overall, read_correct
from taxonomy import classify_run, load_questions_and_keys

RESAMPLES = 10000
SEED = 42
RIGHT = ("correct", "correct_unparsed")
SYNTACTIC = ("S1", "S2")
COUNTS = ["syntactic_free", "now_right", "now_semantic", "still_syntactic", "right_free_wrong_constrained",
          "wrong_free_right_constrained"]


def change_and_interval(free_base, free_other, con_base, con_other, rng):
    means = {key: {} for key in ("fb", "fo", "cb", "co")}
    boots = {key: {} for key in means}
    for category in CATEGORIES:
        ids = sorted(free_base[category])
        runs = {"fb": free_base, "fo": free_other, "cb": con_base, "co": con_other}
        for run in runs.values():
            assert sorted(run[category]) == ids, category
        picks = rng.integers(0, len(ids), size=(RESAMPLES, len(ids)))
        for key, run in runs.items():
            x = np.array([run[category][i] for i in ids])
            means[key][category], boots[key][category] = x.mean(), x[picks].mean(axis=1)
    # Overall rounded to 2 decimals first, as in results/summary.csv, so the changes match the main tables
    change_free = round(overall(means["fo"]), 2) - round(overall(means["fb"]), 2)
    change_con = round(overall(means["co"]), 2) - round(overall(means["cb"]), 2)
    boot = (overall(boots["co"]) - overall(boots["cb"])) - (overall(boots["fo"]) - overall(boots["fb"]))
    low, high = np.percentile(boot, [2.5, 97.5])
    return change_free, change_con, low, high


def transitions(run_free, run_con, questions, keys):
    free = {i: label for i, category, label in classify_run(run_free, REPO / "results", questions, keys)
            if category != "irrelevance"}
    con = {i: label for i, _, label in classify_run(run_con, REPO / "results", questions, keys)}
    syntactic = [i for i, label in free.items() if label in SYNTACTIC]
    return {
        "syntactic_free": len(syntactic),
        "now_right": sum(con[i] in RIGHT for i in syntactic),
        "now_semantic": sum(con[i] not in RIGHT and con[i] not in SYNTACTIC for i in syntactic),
        "still_syntactic": sum(con[i] in SYNTACTIC for i in syntactic),
        "right_free_wrong_constrained": sum(free[i] in RIGHT and con[i] not in RIGHT for i in free),
        "wrong_free_right_constrained": sum(free[i] not in RIGHT and con[i] in RIGHT for i in free),
        "new_failure_classes": dict(Counter(con[i] for i in free if free[i] in RIGHT and con[i] not in RIGHT)),
    }


def main():
    scores = REPO / "results" / "scores"
    with open(REPO / "results" / "significance_free.csv", newline="", encoding="utf-8") as f:
        significant_loss = {(r["model"], r["precision"]) for r in csv.DictReader(f)
                            if float(r["p_holm"]) < 0.05 and float(r["change"]) < 0}

    rng = np.random.default_rng(SEED)
    gap = []
    for model, name in MODELS.items():
        base = {mode: read_correct(scores / f"{model}_fp16_{mode}.jsonl") for mode in ("free", "constrained")}
        for precision, label in PRECISIONS.items():
            if precision == "fp16":
                continue
            other = {mode: read_correct(scores / f"{model}_{precision}_{mode}.jsonl") for mode in base}
            change_free, change_con, low, high = change_and_interval(base["free"], other["free"],
                                                                     base["constrained"], other["constrained"], rng)
            difference = round(change_con - change_free, 2)
            removed = 100 * difference / -change_free if (name, label) in significant_loss else None
            gap.append([name, label, f"{change_free:.2f}", f"{change_con:.2f}", f"{difference:.2f}", f"{low:.2f}",
                        f"{high:.2f}", "" if removed is None else f"{removed:.0f}"])

    questions, keys = load_questions_and_keys()
    moves = []
    for model, name in MODELS.items():
        for precision, label in PRECISIONS.items():
            t = transitions(f"{model}_{precision}_free", f"{model}_{precision}_constrained", questions, keys)
            moves.append([name, label] + [t[k] for k in COUNTS]
                         + ["; ".join(f"{c} {n}" for c, n in sorted(t["new_failure_classes"].items()))])

    with open(REPO / "results" / "gap.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["model", "precision", "change_free", "change_constrained", "difference", "ci_low", "ci_high",
                         "loss_removed_pct"])
        writer.writerows(gap)
    with open(REPO / "results" / "transitions.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["model", "precision"] + COUNTS + ["classes_of_newly_wrong"])
        writer.writerows(moves)

    print(f"{'model':24s}{'prec':6s}{'free':>8s}{'constr.':>9s}{'diff':>8s}{'95% interval':>18s}{'removed':>9s}")
    for r in gap:
        interval = f"[{float(r[5]):+.2f}, {float(r[6]):+.2f}]"
        removed = f"{r[7]}%" if r[7] else ""
        print(f"{r[0]:24s}{r[1]:6s}{float(r[2]):+8.2f}{float(r[3]):+9.2f}{float(r[4]):+8.2f}{interval:>18s}"
              f"{removed:>9s}")
    print(f"\nSaved {REPO / 'results' / 'gap.csv'} and {REPO / 'results' / 'transitions.csv'}")


if __name__ == "__main__":
    main()
