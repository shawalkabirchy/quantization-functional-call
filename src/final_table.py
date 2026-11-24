"""Build results/final_table.md and results/final_table.csv: every score of the project in one place.

Usage (from the repository root; needs only the Python standard library):
    python src/final_table.py

It only reads files that other scripts already wrote to results/ (summary.csv, significance_free.csv,
significance_constrained.csv, gap.csv, kv_cache.csv, kv_cache_stress.csv, stress_significance.csv,
taxonomy_free.csv, taxonomy_constrained.csv, latency.csv, bfcl_repro/comparison.csv). It runs no model and
computes no new score: the only arithmetic is a difference of two accuracies (change from FP16, gain from the
constraint) and sums of counts.
"""

import csv
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
RESULTS = REPO / "results"

MODELS = {  # config name prefix -> name in the tables
    "qwen3-1.7b": "Qwen3-1.7B",
    "qwen3-4b": "Qwen3-4B",
    "qwen3-8b": "Qwen3-8B",
    "llama-3.2-1b": "Llama-3.2-1B-Instruct",
    "llama-3.2-3b": "Llama-3.2-3B-Instruct",
}
SETTINGS = {  # precision in run ids -> name in the tables (fp16-kv8 = FP16 weights, 8-bit KV cache, free only)
    "fp16": "FP16",
    "fp16-kv8": "FP16, 8-bit KV",
    "int8": "INT8",
    "nf4": "NF4",
    "gptq": "GPTQ",
    "awq": "AWQ",
}
MAIN = ["fp16", "int8", "nf4", "gptq", "awq"]  # run in both modes
BFCL_CATEGORIES = [
    ("simple_python", "Python"), ("simple_java", "Java"), ("simple_javascript", "JavaScript"),
    ("simple_ast", "Simple"), ("multiple", "Multiple"), ("parallel", "Parallel"),
    ("parallel_multiple", "Parallel multiple"), ("non_live_overall", "Overall"), ("irrelevance", "Irrelevance"),
]
STRESS_GROUPS = [
    ("stress_plain", "Plain (control)"), ("stress_nested", "Nested"), ("stress_enum", "Enum"),
    ("stress_number", "Number"), ("stress_optional", "Optional"), ("stress_unicode", "Unicode"),
    ("stress_menu", "Menu"),
]
CLASSES = ["S1", "S2", "M1", "M2", "M3", "M4", "M5"]
DASH = "–"


def read_csv(name):
    with open(RESULTS / name, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def table(header, rows, left=2):
    """Markdown table; the first `left` columns are left-aligned, the rest right-aligned."""
    lines = ["| " + " | ".join(header) + " |",
             "|" + "---|" * left + "---:|" * (len(header) - left)]
    lines += ["| " + " | ".join(row) + " |" for row in rows]
    return lines


def f2(value, signed=False):
    if value is None:
        return DASH
    return f"{value:+.2f}" if signed else f"{value:.2f}"


def sig(text, p_holm):
    """Bold a change whose Holm-adjusted p is below 0.05."""
    return f"**{text}**" if p_holm is not None and p_holm < 0.05 else text


def main():
    # ---- read -----------------------------------------------------------------------------------------------
    summary = {}
    for r in read_csv("summary.csv"):
        summary[(r["run_id"], r["category"])] = r

    def run_id(model, setting, mode, stress=False):
        return f"{model}_{setting}_{mode}" + ("_stress" if stress else "")

    def accuracy(model, setting, mode, category):
        row = summary.get((run_id(model, setting, mode), category))
        if row is None:
            return None
        if row["complete"] != "True":
            raise SystemExit(f"run {run_id(model, setting, mode)} is not complete")
        return float(row["accuracy"])

    def stress(model, setting, mode, group=None):
        groups = [group] if group else [g for g, _ in STRESS_GROUPS]
        rows = [summary.get((run_id(model, setting, mode, stress=True), g)) for g in groups]
        if any(r is None for r in rows):
            return None
        if any(r["complete"] != "True" for r in rows):
            raise SystemExit(f"stress run {run_id(model, setting, mode, True)} is not complete")
        return sum(int(r["correct"]) for r in rows), sum(int(r["total"]) for r in rows)

    tests = {}
    for mode in ["free", "constrained"]:
        for r in read_csv(f"significance_{mode}.csv"):
            tests[(r["model"], r["precision"], mode)] = r
    kv_tests = {r["model"]: r for r in read_csv("kv_cache.csv")}
    gap = {(r["model"], r["precision"]): r for r in read_csv("gap.csv")}
    latency = {(r["model"], r["precision"], r["mode"]): r for r in read_csv("latency.csv")}

    taxonomy = {}
    for mode in ["free", "constrained"]:
        for r in read_csv(f"taxonomy_{mode}.csv"):
            fields = CLASSES + ["total", "correct", "correct_unparsed"]
            counts = taxonomy.setdefault(r["run_id"], {c: 0 for c in fields})
            for c in fields:
                counts[c] += int(r[c])

    def test_of(model, setting, mode):
        """Paired test against FP16 in the same mode: (change, low, high, fp16_only, other_only, p_holm)."""
        name, label = MODELS[model], SETTINGS[setting]
        if setting == "fp16-kv8":
            r = kv_tests.get(name) if mode == "free" else None
            other = "right_kv8_only"
        else:
            r = tests.get((name, label, mode))
            other = "right_quantized_only"
        if r is None:
            return None
        return (float(r["change"]), float(r["ci_low"]), float(r["ci_high"]), int(r["right_at_fp16_only"]),
                int(r[other]), float(r["p_holm"]))

    checks = []  # (what, expected, got)

    # ---- 1. master table ------------------------------------------------------------------------------------
    master_rows, csv_rows = [], []
    for model, name in MODELS.items():
        base = {mode: accuracy(model, "fp16", mode, "non_live_overall") for mode in ["free", "constrained"]}
        for i, setting in enumerate(SETTINGS):
            label = SETTINGS[setting]
            acc = {mode: accuracy(model, setting, mode, "non_live_overall") for mode in ["free", "constrained"]}
            irr = {mode: accuracy(model, setting, mode, "irrelevance") for mode in ["free", "constrained"]}
            st = {mode: stress(model, setting, mode) for mode in ["free", "constrained"]}
            change, p = {}, {}
            for mode in ["free", "constrained"]:
                if setting == "fp16" or acc[mode] is None:
                    change[mode], p[mode] = None, None
                    continue
                change[mode] = acc[mode] - base[mode]
                t = test_of(model, setting, mode)
                p[mode] = t[5]
                checks.append((f"change {name} {label} {mode} = test file", f"{t[0]:+.2f}", f"{change[mode]:+.2f}"))
            gain = None if acc["constrained"] is None else acc["constrained"] - acc["free"]
            lat = {mode: latency.get((name, label, mode)) for mode in ["free", "constrained"]}
            gpus = int(lat["free"]["gpus"])
            weights = float(lat["free"]["weights_gib_per_gpu"])

            def change_cell(mode):
                if setting == "fp16":
                    return "base" if acc[mode] is not None else DASH
                return sig(f2(change[mode], signed=True), p[mode])

            master_rows.append([
                name if i == 0 else "", label,
                f"{weights:.2f}" + (f" ×{gpus}" if gpus > 1 else ""),
                f2(acc["free"]), change_cell("free"),
                f2(acc["constrained"]), change_cell("constrained"),
                f2(gain, signed=True),
                f2(irr["free"]), f2(irr["constrained"]),
                DASH if st["free"] is None else str(st["free"][0]),
                DASH if st["constrained"] is None else str(st["constrained"][0]),
                f"{float(lat['free']['output_tokens_per_second']):.1f}",
                f"{float(lat['free']['first_token_ms_median']):.0f}",
                f"{float(lat['free']['total_s_median']):.2f}",
            ])
            csv_rows.append({
                "model": name, "precision": label, "gpus": gpus, "weights_gib_per_gpu": f"{weights:.2f}",
                "bfcl_free": f2(acc["free"]), "change_free": f2(change["free"], True),
                "p_holm_free": "" if p["free"] is None else f"{p['free']:.3g}",
                "bfcl_constrained": f2(acc["constrained"]), "change_constrained": f2(change["constrained"], True),
                "p_holm_constrained": "" if p["constrained"] is None else f"{p['constrained']:.3g}",
                "constraint_gain": f2(gain, True),
                "irrelevance_free": f2(irr["free"]), "irrelevance_constrained": f2(irr["constrained"]),
                "stress_free_of_175": "" if st["free"] is None else st["free"][0],
                "stress_constrained_of_175": "" if st["constrained"] is None else st["constrained"][0],
                **{f"{key}_{mode}": "" if lat[mode] is None else lat[mode][col]
                   for mode in ["free", "constrained"]
                   for key, col in [("tokens_per_s", "output_tokens_per_second"),
                                    ("first_token_ms", "first_token_ms_median"),
                                    ("whole_answer_s", "total_s_median")]},
                "kv_cache_tokens": lat["free"]["kv_cache_tokens"],
            })

    # ---- 2. BFCL by category --------------------------------------------------------------------------------
    def category_rows(mode, settings):
        rows = []
        for model, name in MODELS.items():
            first = True
            for setting in settings:
                values = [accuracy(model, setting, mode, c) for c, _ in BFCL_CATEGORIES]
                if values[-2] is None:
                    continue
                rows.append([name if first else "", SETTINGS[setting]] + [f2(v) for v in values])
                first = False
        return rows

    # ---- 3. paired tests ------------------------------------------------------------------------------------
    test_rows = []
    for model, name in MODELS.items():
        for i, setting in enumerate(MAIN[1:]):
            row = [name if i == 0 else "", SETTINGS[setting]]
            for mode in ["free", "constrained"]:
                c, lo, hi, a, b, ph = test_of(model, setting, mode)
                row += [sig(f"{c:+.2f}", ph), f"{lo:+.2f} to {hi:+.2f}", str(a), str(b), f"{ph:.3g}"]
            test_rows.append(row)
    kv_rows = []
    for model, name in MODELS.items():
        c, lo, hi, a, b, ph = test_of(model, "fp16-kv8", "free")
        kv_rows.append([name, sig(f"{c:+.2f}", ph), f"{lo:+.2f} to {hi:+.2f}", str(a), str(b), f"{ph:.3g}"])
    gap_rows = []
    for model, name in MODELS.items():
        for i, setting in enumerate(MAIN[1:]):
            g = gap[(name, SETTINGS[setting])]
            ph = float(tests[(name, SETTINGS[setting], "free")]["p_holm"])
            gap_rows.append([
                name if i == 0 else "", SETTINGS[setting], sig(f"{float(g['change_free']):+.2f}", ph),
                f"{float(g['change_constrained']):+.2f}", f"{float(g['difference']):+.2f}",
                f"{float(g['ci_low']):+.2f} to {float(g['ci_high']):+.2f}",
                f"{g['loss_removed_pct']}%" if g["loss_removed_pct"] else DASH,
            ])

    # ---- 4. stress suite ------------------------------------------------------------------------------------
    def stress_rows(mode, settings):
        rows = []
        for model, name in MODELS.items():
            first = True
            for setting in settings:
                if stress(model, setting, mode) is None:
                    continue
                cells = [str(stress(model, setting, mode, g)[0]) for g, _ in STRESS_GROUPS]
                rows.append([name if first else "", SETTINGS[setting]] + cells + [str(stress(model, setting, mode)[0])])
                first = False
        return rows

    stress_tests = read_csv("stress_significance.csv")
    stress_sig = [r for r in stress_tests if float(r["p_holm"]) < 0.05]
    group_names = dict(STRESS_GROUPS)
    stress_sig_rows = [[r["model"], r["precision"], r["mode"], group_names[r["group"]], r["fp16_correct"],
                        r["quantized_correct"], f"{int(r['change']):+d}", r["p_holm"]] for r in stress_sig]
    kv_stress = read_csv("kv_cache_stress.csv")
    kv_stress_sig = sum(float(r["p_holm"]) < 0.05 for r in kv_stress)

    # ---- 5. failure classes ---------------------------------------------------------------------------------
    def failure_rows(mode):
        rows, totals = [], {c: 0 for c in CLASSES}
        for model, name in MODELS.items():
            for i, setting in enumerate(MAIN):
                t = taxonomy[run_id(model, setting, mode)]
                failures = sum(t[c] for c in CLASSES)
                # an irrelevance answer that does not parse makes no call, so it is correct (correct_unparsed)
                checks.append((f"failures {name} {setting} {mode} = total - correct - correct_unparsed",
                               str(t["total"] - t["correct"] - t["correct_unparsed"]), str(failures)))
                syntactic = t["S1"] + t["S2"]
                for c in CLASSES:
                    totals[c] += t[c]
                rows.append([name if i == 0 else "", SETTINGS[setting], f"{failures:,}"]
                            + [f"{t[c]:,}" for c in CLASSES]
                            + [f"{syntactic:,}", f"{failures - syntactic:,}"])
        failures = sum(totals.values())
        syntactic = totals["S1"] + totals["S2"]
        rows.append(["**All 25 runs**", "", f"**{failures:,}**"] + [f"**{totals[c]:,}**" for c in CLASSES]
                    + [f"**{syntactic:,}**", f"**{failures - syntactic:,}**"])
        return rows, failures, syntactic

    fail_free, fail_free_n, syn_free_n = failure_rows("free")
    fail_con, fail_con_n, syn_con_n = failure_rows("constrained")
    kv_fail_rows = [[name, kv_tests[name]["failures_fp16"], kv_tests[name]["failures_kv8"]]
                    for name in MODELS.values()]

    # ---- 6. speed and memory --------------------------------------------------------------------------------
    speed_rows = []
    for model, name in MODELS.items():
        for i, setting in enumerate(SETTINGS):
            lf, lc = latency[(name, SETTINGS[setting], "free")], latency[(name, SETTINGS[setting], "constrained")]
            speed_rows.append([
                name if i == 0 else "", SETTINGS[setting], lf["gpus"], f"{float(lf['weights_gib_per_gpu']):.2f}",
                f"{int(lf['kv_cache_tokens']):,}",
                f"{float(lf['output_tokens_per_second']):.1f}", f"{float(lf['first_token_ms_median']):.0f}",
                f"{float(lf['total_s_median']):.2f}",
                f"{float(lc['output_tokens_per_second']):.1f}", f"{float(lc['first_token_ms_median']):.0f}",
                f"{float(lc['total_s_median']):.2f}",
            ])

    # ---- 7. setup checks ------------------------------------------------------------------------------------
    repro = read_csv("bfcl_repro/comparison.csv")
    repro_rows = [[r["Model"].replace("Qwen3-4B-Instruct-2507 ", "").strip("()"), r["Category"],
                   f2(float(r["Ours (FP16, T4)"])), f2(float(r["Leaderboard"])), f2(float(r["Difference"]), True)]
                  for r in repro]
    pilot = [f2(accuracy("qwen3-4b-instruct-2507", "fp16", "free", c)) for c, _ in BFCL_CATEGORIES]

    # ---- counts the text below states -----------------------------------------------------------------------
    n_sig = {mode: sum(float(tests[(n, SETTINGS[s], mode)]["p_holm"]) < 0.05
                       for n in MODELS.values() for s in MAIN[1:]) for mode in ["free", "constrained"]}
    n_raised = sum(r["constraint_gain"].startswith("+") and r["constraint_gain"] != "+0.00"
                   for r in csv_rows if r["precision"] != "FP16, 8-bit KV")
    n_stress_tests = len(stress_tests)

    # ---- write ----------------------------------------------------------------------------------------------
    cat_header = ["Model", "Precision"] + [label for _, label in BFCL_CATEGORIES]
    stress_header = ["Model", "Precision"] + [label for _, label in STRESS_GROUPS] + ["Total (of 175)"]
    fail_header = ["Model", "Precision", "Failures"] + CLASSES + ["Syntactic (S)", "Semantic (M)"]
    lines = [
        "# Final Table: Every Score of the Project",
        "",
        "All scores of the study in one place, after the experiments were frozen (Update 15). Built by "
        "[`src/final_table.py`](../src/final_table.py) from the files in `results/`; nothing here is typed by hand. "
        "The same master table, with p-values and the constrained-decoding speed, is in "
        "[`final_table.csv`](final_table.csv).",
        "",
        "**Setup in short:** 5 models (Qwen3-1.7B, 4B, 8B; Llama-3.2-1B, 3B-Instruct) × 5 weight precisions (FP16, "
        "INT8, NF4, GPTQ, AWQ) × 2 decoding modes (free, JSON-schema-constrained) on the 1,390 BFCL `non_live` "
        "questions and on our 175-question schema-stress suite, plus the 5 FP16 models with an 8-bit KV cache "
        "(free decoding only): 110 runs. Greedy decoding, vLLM 0.8.5, Kaggle Tesla T4.",
        "",
        "**How to read:**",
        "",
        "- **BFCL score** = BFCL's Non-Live Overall accuracy (%): the unweighted mean of simple (itself the mean of "
        "Python, Java and JavaScript), multiple, parallel and parallel multiple. Irrelevance (240 questions where no "
        "function fits) is reported apart, as in BFCL.",
        "- **Change** = points against FP16 in the same decoding mode. **Bold** = significant: McNemar's exact test, "
        "paired by question on the 1,150 AST questions, Holm-adjusted p < 0.05.",
        "- **Constraint gain** = constrained score minus free score of the same run.",
        "- **Stress** = correct answers of the 175 stress-suite questions (7 groups of 25).",
        "- **Speed** = batch size 1, median over 35 questions. Tokens/s = speed of writing the answer. "
        "Differences of about 20% or less are within the noise of Kaggle's T4s.",
        f"- {DASH} = not run (the 8-bit KV cache was run in free decoding only).",
        "",
        "## 1. Master table",
        "",
    ] + table(["Model", "Precision", "Weights (GiB per GPU)", "BFCL free", "Change", "BFCL constrained", "Change",
               "Constraint gain", "Irrelevance free", "Irrelevance constrained", "Stress free", "Stress constrained",
               "Tokens/s", "First token (ms)", "Whole answer (s)"], master_rows) + [
        "",
        "Speed columns are free decoding; constrained decoding is in section 6. ×2 = the model is split over two "
        "T4s (Qwen3-8B in FP16); every other config runs on one.",
        "",
        f"In free decoding {n_sig['free']} of the 20 changes from FP16 are significant, in constrained decoding "
        f"{n_sig['constrained']}; the constraint raises {n_raised} of the 25 runs. Of the 5 runs with an 8-bit KV "
        f"cache, {sum(float(r['p_holm']) < 0.05 for r in kv_tests.values())} changes significantly.",
        "",
        "## 2. BFCL accuracy by category (%)",
        "",
        "Python 400, Java 100, JavaScript 50, multiple 200, parallel 200, parallel multiple 200 and irrelevance 240 "
        "questions. Simple = mean of Python, Java and JavaScript.",
        "",
        "### Free decoding",
        "",
    ] + table(cat_header, category_rows("free", SETTINGS)) + [
        "",
        "### Constrained decoding",
        "",
    ] + table(cat_header, category_rows("constrained", MAIN)) + [
        "",
        "## 3. Paired tests against FP16",
        "",
        "Change in Overall with its bootstrap 95% interval; *FP16 only* / *Quantized only* (*8-bit KV only*) = AST "
        "questions right in only one of the two runs; p = McNemar's exact test, Holm-adjusted (over the 20 runs "
        "of a mode; over the 5 models for the 8-bit KV cache).",
        "",
        "### Quantized weights",
        "",
    ] + table(["Model", "Precision", "Free change", "95% interval", "FP16 only", "Quantized only", "p (Holm)",
               "Constrained change", "95% interval", "FP16 only", "Quantized only", "p (Holm)"], test_rows) + [
        "",
        "### 8-bit KV cache (FP16 weights, free decoding)",
        "",
    ] + table(["Model", "Change", "95% interval", "FP16 only", "8-bit KV only", "p (Holm)"], kv_rows, left=1) + [
        "",
        "### How much of the free change the constraint removes",
        "",
        "Difference = constrained change minus free change, with its bootstrap 95% interval. Loss removed is given "
        "for the 7 significant free losses only; 100% = the whole loss is gone, a negative value = the loss "
        "grows.",
        "",
    ] + table(["Model", "Precision", "Free change", "Constrained change", "Difference", "95% interval",
               "Loss removed"], gap_rows) + [
        "",
        "## 4. Schema-stress suite (correct of 25 per group)",
        "",
        "### Free decoding",
        "",
    ] + table(stress_header, stress_rows("free", SETTINGS)) + [
        "",
        "### Constrained decoding",
        "",
    ] + table(stress_header, stress_rows("constrained", MAIN)) + [
        "",
        f"### The {len(stress_sig)} significant changes of the {n_stress_tests} stress-suite tests",
        "",
        "McNemar's exact test on the 25 questions of a group, Holm's correction over the 140 tests of each mode. "
        f"For the 8-bit KV cache, {kv_stress_sig} of its {len(kv_stress)} group tests are significant.",
        "",
    ] + table(["Model", "Precision", "Mode", "Group", "FP16", "Quantized", "Change", "p (Holm)"], stress_sig_rows,
              left=4) + [
        "",
        "## 5. Failure classes (BFCL, all 1,390 questions)",
        "",
        "S1 not valid JSON, S2 valid JSON that does not match the schema (syntactic); M1 wrong function, M2 wrong "
        "argument value, M3 wrong number of calls, M4 a call on an irrelevance question, M5 no call when one was "
        "needed (semantic).",
        "",
        "### Free decoding",
        "",
    ] + table(fail_header, fail_free) + [
        "",
        f"Syntactic: {syn_free_n:,} of {fail_free_n:,} failures ({100 * syn_free_n / fail_free_n:.1f}%).",
        "",
        "### Constrained decoding",
        "",
    ] + table(fail_header, fail_con) + [
        "",
        f"Syntactic: {syn_con_n:,} of {fail_con_n:,} failures ({100 * syn_con_n / fail_con_n:.1f}%).",
        "",
        "### 8-bit KV cache (free decoding): failed answers",
        "",
    ] + table(["Model", "FP16", "8-bit KV"], kv_fail_rows, left=1) + [
        "",
        "## 6. Speed and memory at batch size 1",
        "",
        "Weights per GPU and KV cache tokens (keys and values that fit in the memory left after the weights) from "
        "vLLM's log. Llama-3.2-3B in FP16: constrained values not reliable (the GPU slowed down during that run). "
        "Llama-3.2-1B NF4 and GPTQ in constrained decoding: most answers are an empty list, so tokens/s is below the "
        "model's real writing speed.",
        "",
    ] + table(["Model", "Precision", "GPUs", "Weights (GiB per GPU)", "KV cache tokens", "Free: tokens/s",
               "Free: first token (ms)", "Free: whole answer (s)", "Constrained: tokens/s",
               "Constrained: first token (ms)", "Constrained: whole answer (s)"], speed_rows) + [
        "",
        "## 7. Setup and method checks",
        "",
        "### BFCL reproduced on Kaggle (Qwen3-4B-Instruct-2507, BFCL's own harness, FP16)",
        "",
    ] + table(["Mode", "Category", "Ours (FP16, T4)", "Leaderboard", "Difference"], repro_rows) + [
        "",
        "### First run of our own script (Update 3, Qwen3-4B-Instruct-2507, FP16, free decoding)",
        "",
    ] + table(["Model"] + [label for _, label in BFCL_CATEGORIES], [["Qwen3-4B-Instruct-2507"] + pilot],
              left=1) + [
        "",
        "### Other checks",
        "",
        "- Our scorer against BFCL's: re-scoring the 1,390 reproduction answers gives BFCL's verdict on every "
        "question ([`src/check_scorer.py`](../src/check_scorer.py)).",
        "- Failure classifier: checked by hand on 80 random failures, 80 agree "
        "([`notes/taxonomy_check.md`](../notes/taxonomy_check.md)).",
        "- Schema against BFCL's checker: 14 of the 34,750 free answers are correct by BFCL but do not match the "
        "schema ([`taxonomy_free.md`](taxonomy_free.md)).",
        "- Parser: Llama-3.2-3B FP16 scores 31.38 with our strict parser and would score 46.02 with BFCL's bracket "
        "repair (diagnostic only; the reported score stays 31.38) ([`baseline_fp16.md`](baseline_fp16.md)).",
        "- Every answer of every run was scored again on a PC with the same result.",
    ]

    out = RESULTS / "final_table.md"
    out.write_text("\n".join(lines) + "\n", encoding="utf-8")
    with open(RESULTS / "final_table.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(csv_rows[0]))
        writer.writeheader()
        writer.writerows({k: "" if v == DASH else v for k, v in row.items()} for row in csv_rows)

    bad = [c for c in checks if c[1] != c[2]]
    for what, expected, got in bad:
        print(f"MISMATCH {what}: expected {expected}, got {got}")
    print(f"{len(checks)} internal checks, {len(bad)} mismatches")
    print(f"Saved {out} and {RESULTS / 'final_table.csv'}")


if __name__ == "__main__":
    main()
