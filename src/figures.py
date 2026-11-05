"""Draw the figures in results/figures/ from the result tables. Needs matplotlib, which the runs themselves do not.

Usage (from the repository root):
    python src/figures.py taxonomy --mode free
    python src/figures.py penalty --mode free     # needs results/significance_<mode>.csv (src/significance.py)
    python src/figures.py gap                     # needs results/gap.csv (src/gap.py); free and constrained
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


# Smallest model first. Llama-3.2-1B has 1.2B parameters, Llama-3.2-3B 3.2B
BY_SIZE = ["llama-3.2-1b", "qwen3-1.7b", "llama-3.2-3b", "qwen3-4b", "qwen3-8b"]
# Validated as an adjacent set (scripts of the dataviz method); the same hues as above, used here for precisions
PRECISION_COLORS = {"INT8": "#2a78d6", "NF4": "#eb6834", "GPTQ": "#1baf7a", "AWQ": "#eda100"}


def penalty(mode, results_dir):
    """Change in Overall accuracy from FP16, per model (smallest first) and precision, with 95% intervals."""
    with open(results_dir / f"significance_{mode}.csv", newline="", encoding="utf-8") as f:
        rows = {(r["model"], r["precision"]): r for r in csv.DictReader(f)}

    fig, ax = plt.subplots(figsize=(9, 5.2), facecolor=SURFACE)
    ax.set_facecolor(SURFACE)
    width = 0.19
    for x, model in enumerate(BY_SIZE):
        for i, (precision, color) in enumerate(PRECISION_COLORS.items()):
            r = rows.get((MODELS[model], precision))
            if r is None:
                continue
            pos = x + (i - 1.5) * (width + 0.02)
            change, low, high = float(r["change"]), float(r["ci_low"]), float(r["ci_high"])
            significant = float(r["p_holm"]) < 0.05
            ax.bar(pos, change, width=width, color=color, linewidth=0,
                   hatch=None if significant else "////", hatchcolor=SURFACE)
            ax.plot([pos, pos], [low, high], color=INK, linewidth=1, solid_capstyle="butt")
            if abs(change) >= 10:   # label only the large changes
                ax.text(pos, high + 1 if change > 0 else low - 1, f"{change:+.1f}".replace("-", "−"), ha="center",
                        va="bottom" if change > 0 else "top", fontsize=7.5, color=INK)
    ax.axhline(0, color=INK, linewidth=0.9)
    ax.set_xticks(range(len(BY_SIZE)), [MODELS[m].replace("-Instruct", "") for m in BY_SIZE], fontsize=9, color=INK)
    ax.set_ylabel("Change in Overall accuracy from FP16 (points)", fontsize=9.5, color=MUTED)
    ax.set_xlabel("Models, smallest first", fontsize=9.5, color=MUTED)
    ax.grid(axis="y", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    ax.tick_params(length=0, colors=MUTED)
    for side in ax.spines.values():
        side.set_visible(False)
    handles = [plt.Rectangle((0, 0), 1, 1, color=c) for c in PRECISION_COLORS.values()]
    handles += [plt.Rectangle((0, 0), 1, 1, color=MUTED),
                plt.Rectangle((0, 0), 1, 1, facecolor=MUTED, hatch="////", hatchcolor=SURFACE, linewidth=0)]
    labels = list(PRECISION_COLORS) + ["significant", "not significant"]
    fig.legend(handles, labels, loc="upper center", ncol=6, frameon=False, fontsize=9, labelcolor=INK,
               bbox_to_anchor=(0.53, 0.99), handlelength=1.6)
    fig.text(0.99, 0.01, "Lines: bootstrap 95% intervals. Significant: McNemar test, Holm-adjusted p < 0.05.",
             ha="right", va="bottom", fontsize=7.5, color=MUTED)
    fig.subplots_adjust(left=0.09, right=0.98, top=0.9, bottom=0.14)

    out = results_dir / "figures" / f"penalty_by_size_{mode}.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=200)
    print(f"Saved {out}")


def gap(mode, results_dir):
    """Change from FP16 with free decoding (open circle) and with constrained decoding (filled circle)."""
    with open(results_dir / "gap.csv", newline="", encoding="utf-8") as f:
        rows = {(r["model"], r["precision"]): r for r in csv.DictReader(f)}

    fig, ax = plt.subplots(figsize=(9, 5.2), facecolor=SURFACE)
    ax.set_facecolor(SURFACE)
    step = 0.21
    for x, model in enumerate(BY_SIZE):
        for i, (precision, color) in enumerate(PRECISION_COLORS.items()):
            r = rows.get((MODELS[model], precision))
            if r is None:
                continue
            pos = x + (i - 1.5) * step
            free, constrained = float(r["change_free"]), float(r["change_constrained"])
            ax.plot([pos, pos], [free, constrained], color=color, linewidth=1.6, solid_capstyle="butt", zorder=2)
            ax.plot(pos, free, "o", markersize=6.5, markerfacecolor=SURFACE, markeredgecolor=color,
                    markeredgewidth=1.6, zorder=3)
            ax.plot(pos, constrained, "o", markersize=6.5, color=color, markeredgecolor=SURFACE,
                    markeredgewidth=1, zorder=4)
    ax.axhline(0, color=INK, linewidth=0.9, zorder=1)
    ax.set_xticks(range(len(BY_SIZE)), [MODELS[m].replace("-Instruct", "") for m in BY_SIZE], fontsize=9, color=INK)
    ax.set_ylabel("Change in Overall accuracy from FP16 (points)", fontsize=9.5, color=MUTED)
    ax.set_xlabel("Models, smallest first", fontsize=9.5, color=MUTED)
    ax.grid(axis="y", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    ax.tick_params(length=0, colors=MUTED)
    for side in ax.spines.values():
        side.set_visible(False)
    handles = [plt.Rectangle((0, 0), 1, 1, color=c) for c in PRECISION_COLORS.values()]
    handles += [plt.Line2D([], [], linestyle="none", marker="o", markersize=6.5, markerfacecolor=SURFACE,
                           markeredgecolor=MUTED, markeredgewidth=1.6),
                plt.Line2D([], [], linestyle="none", marker="o", markersize=6.5, color=MUTED)]
    labels = list(PRECISION_COLORS) + ["free decoding", "constrained decoding"]
    fig.legend(handles, labels, loc="upper center", ncol=6, frameon=False, fontsize=9, labelcolor=INK,
               bbox_to_anchor=(0.53, 0.99), handlelength=1.6)
    fig.text(0.99, 0.01, "Each change is from FP16 of the same model in the same decoding mode.",
             ha="right", va="bottom", fontsize=7.5, color=MUTED)
    fig.subplots_adjust(left=0.09, right=0.98, top=0.9, bottom=0.14)

    out = results_dir / "figures" / "constrained_gap.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=200)
    print(f"Saved {out}")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("figure", choices=["taxonomy", "penalty", "gap"])
    parser.add_argument("--mode", default="free", choices=["free", "constrained"])
    parser.add_argument("--results-dir", type=Path, default=REPO / "results")
    args = parser.parse_args()
    {"taxonomy": taxonomy, "penalty": penalty, "gap": gap}[args.figure](args.mode, args.results_dir)


if __name__ == "__main__":
    main()
