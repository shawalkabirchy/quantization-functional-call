"""Compare the 8-bit KV-cache runs with FP16 (Update 13, stretch item), and write results/kv_cache*.csv.

Usage (from the repository root, in the BFCL environment; needs numpy and jsonschema):
    python src/kv_cache.py

Both runs of a model have the same FP16 weights, prompts and decoding; only the KV cache differs (FP16 vs. FP8
E4M3), so every difference comes from the KV cache. The same tests as for the quantized weights:
    BFCL (results/kv_cache.csv): McNemar's exact test on the 1,150 AST questions and a bootstrap 95% interval
        for the change in Overall (src/significance.py), Holm's correction over the five models; the change on
        the 240 irrelevance questions; and the failure classes (src/taxonomy.py) of the answers that were right
        at FP16 and are wrong with the 8-bit KV cache.
    Stress suite (results/kv_cache_stress.csv): correct answers out of 25 per group, McNemar's exact test per
        group, Holm's correction over the 35 tests (as in src/stress_analysis.py).
"""

import argparse
import csv
from collections import Counter
from pathlib import Path

import numpy as np

from data import STRESS, load_answer_key, load_questions
from main_table import MODELS, REPO
from significance import RESAMPLES, SEED, compare, holm, mcnemar_exact, read_correct
from taxonomy import FAILURES, classify_run, load_questions_and_keys

RIGHT = {"correct", "correct_unparsed"}


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--results-dir", type=Path, default=REPO / "results")
    args = parser.parse_args()
    scores = args.results_dir / "scores"

    # BFCL's 1,390 questions
    questions, keys = load_questions_and_keys()
    rng = np.random.default_rng(SEED)
    rows = []
    for model, name in MODELS.items():
        base_id, kv_id = f"{model}_fp16_free", f"{model}_fp16-kv8_free"
        if not (scores / f"{kv_id}.jsonl").exists():
            continue
        test = compare(read_correct(scores / f"{base_id}.jsonl"), read_correct(scores / f"{kv_id}.jsonl"), rng)
        base = {i: label for i, _, label in classify_run(base_id, args.results_dir, questions, keys)}
        kv = {i: (category, label) for i, category, label in classify_run(kv_id, args.results_dir, questions, keys)}
        irrelevance = [i for i, (category, _) in kv.items() if category == "irrelevance"]
        newly_wrong = Counter(label for i, (_, label) in kv.items() if base[i] in RIGHT and label not in RIGHT)
        rows.append({
            "model": name, "change": test["change"], "low": test["low"], "high": test["high"],
            "lost": test["lost"], "gained": test["gained"], "p": test["p"],
            "irrelevance_fp16": sum(base[i] in RIGHT for i in irrelevance),
            "irrelevance_kv8": sum(kv[i][1] in RIGHT for i in irrelevance),
            "failures_fp16": sum(label not in RIGHT for label in base.values()),
            "failures_kv8": sum(label not in RIGHT for _, label in kv.values()),
            "newly_wrong": " ".join(f"{c}:{newly_wrong[c]}" for c in FAILURES if newly_wrong[c]),
        })
    for row, p in zip(rows, holm([r["p"] for r in rows])):
        row["p_holm"] = p

    out = args.results_dir / "kv_cache.csv"
    with open(out, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["model", "change", "ci_low", "ci_high", "right_at_fp16_only", "right_kv8_only", "p_mcnemar",
                         "p_holm", "irrelevance_fp16", "irrelevance_kv8", "failures_fp16", "failures_kv8",
                         "classes_of_newly_wrong"])
        for r in rows:
            writer.writerow([r["model"], f"{r['change']:.2f}", f"{r['low']:.2f}", f"{r['high']:.2f}", r["lost"],
                             r["gained"], f"{r['p']:.3g}", f"{r['p_holm']:.3g}", r["irrelevance_fp16"],
                             r["irrelevance_kv8"], r["failures_fp16"], r["failures_kv8"], r["newly_wrong"]])
    print(f"BFCL, free decoding, 8-bit KV cache vs. FP16 ({RESAMPLES:,} bootstrap resamples)")
    print(f"{'model':24s}{'change':>8s}{'95% interval':>18s}{'lost':>6s}{'gained':>7s}{'p (Holm)':>10s}"
          f"{'irrelevance':>14s}   newly wrong")
    for r in rows:
        interval = f"[{r['low']:+.2f}, {r['high']:+.2f}]"
        print(f"{r['model']:24s}{r['change']:+8.2f}{interval:>18s}{r['lost']:6d}{r['gained']:7d}{r['p_holm']:10.3g}"
              f"{r['irrelevance_fp16']:7d} ->{r['irrelevance_kv8']:4d}   {r['newly_wrong']}")
    print(f"Saved {out}")

    # The schema-stress suite: 7 groups of 25 questions
    questions, keys = {}, {}
    for group in STRESS:
        questions.update({q["id"]: q for q in load_questions(group)})
        keys.update(load_answer_key(group))
    tests = []
    for model, name in MODELS.items():
        kv_id = f"{model}_fp16-kv8_free_stress"
        if not (scores / f"{kv_id}.jsonl").exists():
            continue
        base = {i: label for i, _, label in classify_run(f"{model}_fp16_free_stress", args.results_dir, questions, keys)}
        kv = {i: (group, label) for i, group, label in classify_run(kv_id, args.results_dir, questions, keys)}
        for group in STRESS:
            ids = [i for i, (g, _) in kv.items() if g == group]
            lost = sum(base[i] in RIGHT and kv[i][1] not in RIGHT for i in ids)
            gained = sum(base[i] not in RIGHT and kv[i][1] in RIGHT for i in ids)
            tests.append({"model": name, "group": group, "fp16_correct": sum(base[i] in RIGHT for i in ids),
                          "kv8_correct": sum(kv[i][1] in RIGHT for i in ids), "right_at_fp16_only": lost,
                          "right_kv8_only": gained, "p_mcnemar": mcnemar_exact(lost, gained)})
    for row, p in zip(tests, holm([r["p_mcnemar"] for r in tests])):
        row["p_holm"] = p

    out = args.results_dir / "kv_cache_stress.csv"
    with open(out, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(tests[0]))
        writer.writeheader()
        for row in tests:
            writer.writerow({k: f"{v:.3g}" if k.startswith("p_") else v for k, v in row.items()})
    print(f"\nStress suite, free decoding: groups that changed (correct out of 25)")
    for r in tests:
        if r["fp16_correct"] != r["kv8_correct"] or r["right_at_fp16_only"]:
            print(f"  {r['model']:24s}{r['group']:17s}{r['fp16_correct']:3d} -> {r['kv8_correct']:3d}"
                  f"   lost {r['right_at_fp16_only']}, gained {r['right_kv8_only']}   p (Holm) {r['p_holm']:.3g}")
    print(f"Saved {out}")


if __name__ == "__main__":
    main()
