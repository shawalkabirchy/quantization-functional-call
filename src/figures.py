"""Draw the figures in results/figures/ from the result tables. Needs matplotlib, which the runs themselves do not.

Usage (from the repository root):
    python src/figures.py taxonomy --mode free
    python src/figures.py penalty --mode free     # needs results/significance_<mode>.csv (src/significance.py)
    python src/figures.py gap                     # needs results/gap.csv (src/gap.py); free and constrained
    python src/figures.py tradeoff                # needs results/latency.csv (src/latency_table.py)
"""

import argparse
import csv
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker

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


# The precisions above plus FP16, validated as one adjacent set in this order (dataviz palette validator)
TRADEOFF_COLORS = {"FP16": "#e87ba4", **PRECISION_COLORS}
MODEL_MARKERS = {"qwen3-1.7b": "o", "qwen3-4b": "s", "qwen3-8b": "D", "llama-3.2-1b": "^", "llama-3.2-3b": "v"}


def frontier(points):
    """The Pareto frontier: points that no other point beats on both axes (smaller x and higher y)."""
    best, kept = -1.0, []
    for x, y, key in sorted(points, key=lambda p: (p[0], -p[1])):
        if y > best:
            kept.append((x, y, key))
            best = y
    return kept


def tradeoff(mode, results_dir):
    """Overall accuracy against the memory the weights take and against the time to answer at batch size 1."""
    with open(results_dir / "summary.csv", newline="", encoding="utf-8") as f:
        accuracy = {r["run_id"]: float(r["accuracy"]) for r in csv.DictReader(f) if r["category"] == "non_live_overall"}
    with open(results_dir / "latency.csv", newline="", encoding="utf-8") as f:
        latency = {(r["model"], r["precision"], r["mode"]): r for r in csv.DictReader(f)}

    fig, axes = plt.subplots(2, 2, figsize=(9, 7.6), facecolor=SURFACE, sharey=True)
    measures = [("Memory for the weights (GiB, all GPUs)", lambda r: float(r["weights_gib_per_gpu"]) * int(r["gpus"])),
                ("Time for the whole answer at batch size 1 (s, median)", lambda r: float(r["total_s_median"]))]
    for row, decoding in enumerate(["free", "constrained"]):
        for col, (xlabel, measure) in enumerate(measures):
            ax = axes[row, col]
            ax.set_facecolor(SURFACE)
            points = []
            for model, name in MODELS.items():
                for precision, label in PRECISIONS.items():
                    r = latency.get((name, label, decoding))
                    acc = accuracy.get(f"{model}_{precision}_{decoding}")
                    if r is None or acc is None:
                        continue
                    x = measure(r)
                    # Llama-3.2-3B FP16 constrained: the GPU slowed down during that timing (results/latency.md)
                    unreliable = col == 1 and decoding == "constrained" and model == "llama-3.2-3b" and precision == "fp16"
                    ax.plot(x, acc, MODEL_MARKERS[model], markersize=8, zorder=3,
                            markerfacecolor=SURFACE if unreliable else TRADEOFF_COLORS[label],
                            markeredgecolor=TRADEOFF_COLORS[label] if unreliable else INK,
                            markeredgewidth=1.6 if unreliable else 0.6)
                    if not unreliable:
                        points.append((x, acc, (name, label)))
            best = frontier(points)
            ax.plot([p[0] for p in best], [p[1] for p in best], color=INK, linewidth=1, zorder=2,
                    drawstyle="steps-post")
            # Label one point: the cheapest (or fastest) one within 2 points of the most accurate
            top = max(y for _, y, _ in points)
            x, y, (name, label) = min((p for p in points if p[1] >= top - 2), key=lambda p: (p[0], -p[1]))
            ax.annotate(f"{name.replace('-Instruct', '')} {label}", (x, y), xytext=(0, 10), ha="center",
                        textcoords="offset points", fontsize=7.5, color=INK)
            ax.set_xscale("log")
            ax.xaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
            ax.set_ylim(0, 100)
            ax.grid(color=GRID, linewidth=0.8)
            ax.set_axisbelow(True)
            ax.tick_params(length=0, colors=MUTED, labelsize=8)
            for side in ax.spines.values():
                side.set_visible(False)
            if row == 1:
                ax.set_xlabel(xlabel, fontsize=9, color=MUTED)
            if col == 0:
                ax.set_ylabel(f"Overall accuracy (%), {decoding} decoding", fontsize=9, color=MUTED)
    for ax in axes[:, 0]:
        ax.set_xticks([1, 2, 4, 8, 16], ["1", "2", "4", "8", "16"])
    for ax in axes[:, 1]:
        ax.set_xticks([0.5, 1, 2, 4, 8], ["0.5", "1", "2", "4", "8"])
    handles = [plt.Line2D([], [], linestyle="none", marker="o", markersize=8, color=c, markeredgecolor=INK,
                          markeredgewidth=0.6) for c in TRADEOFF_COLORS.values()]
    handles += [plt.Line2D([], [], linestyle="none", marker=m, markersize=7, color=MUTED) for m in MODEL_MARKERS.values()]
    labels = list(TRADEOFF_COLORS) + [MODELS[m].replace("-Instruct", "") for m in MODEL_MARKERS]
    fig.legend(handles, labels, loc="upper center", ncol=5, frameon=False, fontsize=8.5, labelcolor=INK,
               bbox_to_anchor=(0.53, 0.995), handlelength=1.2, columnspacing=1.4)
    fig.text(0.99, 0.01, "Line: the Pareto frontier (no other point is both smaller or faster and more accurate). Labelled: "
             "the first point within 2 points of the best. Hollow: unreliable timing.", ha="right", va="bottom", fontsize=7, color=MUTED)
    fig.subplots_adjust(left=0.09, right=0.98, top=0.89, bottom=0.1, hspace=0.12, wspace=0.06)

    out = results_dir / "figures" / "accuracy_memory_latency.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=200)
    print(f"Saved {out}")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("figure", choices=["taxonomy", "penalty", "gap", "tradeoff"])
    parser.add_argument("--mode", default="free", choices=["free", "constrained"])
    parser.add_argument("--results-dir", type=Path, default=REPO / "results")
    args = parser.parse_args()
    {"taxonomy": taxonomy, "penalty": penalty, "gap": gap, "tradeoff": tradeoff}[args.figure](args.mode, args.results_dir)


if __name__ == "__main__":
    main()
