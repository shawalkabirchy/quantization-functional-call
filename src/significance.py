"""Test each quantized run against FP16 on the same questions, and write results/significance_<mode>.csv.

Usage (from the repository root; needs numpy and results/scores/):
    python src/significance.py --mode free

Two questions, both paired by question:
    Is the change real? McNemar's exact test on the 1,150 AST questions: it looks only at the questions that one run
    got right and the other got wrong. Holm's correction covers all the tests in the table.
    How large is it? A bootstrap 95% interval for the change in Overall. Questions are resampled inside each category,
    and Overall is computed the same way as BFCL's score.
"""

import argparse
import csv
import json
from math import comb
from pathlib import Path

import numpy as np

from main_table import MODELS, PRECISIONS, REPO

CATEGORIES = ["simple_python", "simple_java", "simple_javascript", "multiple", "parallel", "parallel_multiple"]
RESAMPLES = 10000
SEED = 42


def mcnemar_exact(b, c):
    """Two-sided exact McNemar p value: a binomial test of b against b + c with probability 1/2."""
    n = b + c
    if n == 0:
        return 1.0
    tail = sum(comb(n, k) for k in range(min(b, c) + 1))
    return min(1.0, 2 * tail / 2 ** n)


def read_correct(path):
    """Returns {category: {question id: 1 or 0}}, for the AST questions."""
    correct = {c: {} for c in CATEGORIES}
    with open(path, encoding="utf-8") as f:
        for line in f:
            row = json.loads(line)
            if row["category"] in correct:
                correct[row["category"]][row["id"]] = int(row["valid"])
    return correct


def overall(mean):
    """BFCL's Non-Live Overall Acc from the accuracy of each category (works on arrays too)."""
    simple = (mean["simple_python"] + mean["simple_java"] + mean["simple_javascript"]) / 3
    return 100 * (simple + mean["multiple"] + mean["parallel"] + mean["parallel_multiple"]) / 4


def compare(base, other, rng):
    b, c, base_means, other_means, boot_base, boot_other = 0, 0, {}, {}, {}, {}
    for category in CATEGORIES:
        ids = sorted(base[category])
        assert ids == sorted(other[category]), category
        x = np.array([base[category][i] for i in ids])
        y = np.array([other[category][i] for i in ids])
        b += int(((x == 1) & (y == 0)).sum())   # right at FP16 only
        c += int(((x == 0) & (y == 1)).sum())   # right in the quantized run only
        base_means[category], other_means[category] = x.mean(), y.mean()
        picks = rng.integers(0, len(ids), size=(RESAMPLES, len(ids)))
        boot_base[category], boot_other[category] = x[picks].mean(axis=1), y[picks].mean(axis=1)
    p = mcnemar_exact(b, c)
    low, high = np.percentile(overall(boot_other) - overall(boot_base), [2.5, 97.5])
    # Overall rounded to 2 decimals first, as in results/summary.csv, so the change matches results/main_table.md
    change = round(overall(other_means), 2) - round(overall(base_means), 2)
    return dict(change=change, low=low, high=high, lost=b, gained=c, p=p)


def holm(p_values):
    """Holm-adjusted p values, in the order given."""
    order = np.argsort(p_values)
    adjusted, largest = [0.0] * len(p_values), 0.0
    for rank, index in enumerate(order):
        largest = max(largest, min(1.0, (len(p_values) - rank) * p_values[index]))
        adjusted[index] = largest
    return adjusted


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--mode", default="free", choices=["free", "constrained"])
    parser.add_argument("--results-dir", type=Path, default=REPO / "results")
    args = parser.parse_args()

    rng = np.random.default_rng(SEED)
    rows = []
    for model, name in MODELS.items():
        base = read_correct(args.results_dir / "scores" / f"{model}_fp16_{args.mode}.jsonl")
        for precision, label in PRECISIONS.items():
            path = args.results_dir / "scores" / f"{model}_{precision}_{args.mode}.jsonl"
            if precision != "fp16" and path.exists():
                rows.append({"model": name, "precision": label, **compare(base, read_correct(path), rng)})
    for row, p in zip(rows, holm([r["p"] for r in rows])):
        row["p_holm"] = p

    out = args.results_dir / f"significance_{args.mode}.csv"
    with open(out, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["model", "precision", "change", "ci_low", "ci_high", "right_at_fp16_only",
                         "right_quantized_only", "p_mcnemar", "p_holm"])
        for r in rows:
            writer.writerow([r["model"], r["precision"], f"{r['change']:.2f}", f"{r['low']:.2f}", f"{r['high']:.2f}",
                             r["lost"], r["gained"], f"{r['p']:.3g}", f"{r['p_holm']:.3g}"])
    print(f"{'model':24s}{'prec':6s}{'change':>8s}{'95% interval':>18s}{'lost':>6s}{'gained':>7s}{'p (Holm)':>11s}")
    for r in rows:
        interval = f"[{r['low']:+.2f}, {r['high']:+.2f}]"
        print(f"{r['model']:24s}{r['precision']:6s}{r['change']:+8.2f}{interval:>18s}{r['lost']:6d}{r['gained']:7d}{r['p_holm']:11.3g}")
    print(f"\nSaved {out}")


if __name__ == "__main__":
    main()
