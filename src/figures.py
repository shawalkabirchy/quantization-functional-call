"""Draw the figures in results/figures/ from the result tables. Needs matplotlib, which the runs themselves do not.

Usage (from the repository root):
    python src/figures.py taxonomy --mode free
"""

import argparse
import csv
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from main_table import MODELS, PRECISIONS, REPO

INK, MUTED, GRID, SURFACE = "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb"
# In this order, neighbours stay distinct for colour-blind readers (checked with a palette validator)
CLASSES = {
    "S1 not valid JSON": (["S1"], "#2a78d6"),
    "S2 does not match the schema": (["S2"], "#eb6834"),
    "M1, M2 wrong function or argument": (["M1", "M2"], "#1baf7a"),
    "M3 wrong number of calls": (["M3"], "#eda100"),
    "M4 call on an irrelevance question": (["M4"], "#e87ba4"),
    "M5 no call": (["M5"], "#008300"),
}


def taxonomy(mode, results_dir):
    with open(results_dir / f"taxonomy_{mode}.csv", newline="", encoding="utf-8") as f:
        counts = defaultdict(lambda: defaultdict(int))
        for row in csv.DictReader(f):
            for name, value in row.items():
                if name not in ("run_id", "model", "precision", "mode", "category"):
                    counts[row["model"], row["precision"]][name] += int(value)

    labels = list(PRECISIONS.values())
    fig, axes = plt.subplots(len(MODELS), 1, figsize=(9, 8.6), sharex=True, facecolor=SURFACE)
    for ax, (model, name) in zip(axes, MODELS.items()):
        ax.set_facecolor(SURFACE)
        for y, precision in enumerate(PRECISIONS):
            if (model, precision) not in counts:
                ax.text(4, y, "no result", color=MUTED, fontsize=8, va="center")
                continue
            left = 0
            for columns, color in CLASSES.values():
                width = sum(counts[model, precision][c] for c in columns)
                ax.barh(y, width, left=left, height=0.72, color=color, edgecolor=SURFACE, linewidth=1)
                left += width
        ax.set_yticks(range(len(labels)), labels, fontsize=8.5, color=INK)
        ax.set_ylim(len(labels) - 0.5, -0.5)   # the first precision on top; the same height with a run missing
        ax.set_ylabel(name.replace("-Instruct", ""), rotation=0, ha="left", va="center", fontsize=9.5, color=INK,
                      fontweight="bold", labelpad=92)
        ax.grid(axis="x", color=GRID, linewidth=0.8)
        ax.set_axisbelow(True)
        ax.tick_params(length=0, colors=MUTED)
        for side in ax.spines.values():
            side.set_visible(False)
    axes[-1].set_xlabel("Failed answers, out of 1,390", fontsize=9.5, color=MUTED)
    handles = [plt.Rectangle((0, 0), 1, 1, color=color) for _, color in CLASSES.values()]
    fig.legend(handles, list(CLASSES), loc="upper center", ncol=2, frameon=False, fontsize=9, labelcolor=INK,
               bbox_to_anchor=(0.56, 0.985))
    fig.subplots_adjust(left=0.22, right=0.97, top=0.885, bottom=0.065, hspace=0.16)

    out = results_dir / "figures" / f"taxonomy_{mode}.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=200)
    print(f"Saved {out}")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("figure", choices=["taxonomy"])
    parser.add_argument("--mode", default="free", choices=["free", "constrained"])
    parser.add_argument("--results-dir", type=Path, default=REPO / "results")
    args = parser.parse_args()
    taxonomy(args.mode, args.results_dir)


if __name__ == "__main__":
    main()
