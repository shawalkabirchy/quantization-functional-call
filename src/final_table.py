"""Build the final_table/ folder: every score of the project in one place.

Usage (from the repository root; needs only the Python standard library):
    python src/final_table.py

Writes final_table/README.md (the master table), bfcl_by_category.md, significance.md, stress_suite.md,
failure_classes.md, speed_memory.md, setup_checks.md and final_table.csv (the master table as data).

It only reads files that other scripts already wrote to results/: summary.csv, significance_free.csv,
significance_constrained.csv, gap.csv, kv_cache.csv, kv_cache_stress.csv, stress_significance.csv,
taxonomy_free.csv, taxonomy_constrained.csv, latency.csv and bfcl_repro/comparison.csv. It runs no model and
computes no new score: the only arithmetic is the difference of two accuracies (change from FP16, gain from the
constraint) and sums of counts. Before writing, it checks its changes against the test files and its failure
counts against the question counts, and stops on any mismatch.
"""

import csv
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
RESULTS = REPO / "results"
OUT = REPO / "final_table"

MODELS = {  # config name prefix -> name in the tables
    "qwen3-1.7b": "Qwen3-1.7B",
    "qwen3-4b": "Qwen3-4B",
    "qwen3-8b": "Qwen3-8B",
    "llama-3.2-1b": "Llama-3.2-1B-Instruct",
    "llama-3.2-3b": "Llama-3.2-3B-Instruct",
}
SETTINGS = {  # precision in run ids -> name in the tables (fp16-kv8 = FP16 weights, 8-bit KV cache)
    "fp16": "FP16",
    "fp16-kv8": "FP16, 8-bit KV",
    "int8": "INT8",
    "nf4": "NF4",
    "gptq": "GPTQ",
    "awq": "AWQ",
}
MAIN = ["fp16", "int8", "nf4", "gptq", "awq"]  # run in both decoding modes; fp16-kv8 in free decoding only
MODES = ["free", "constrained"]
BFCL_CATEGORIES = [
    ("simple_python", "Python"), ("simple_java", "Java"), ("simple_javascript", "JavaScript"),
    ("simple_ast", "Simple"), ("multiple", "Multiple"), ("parallel", "Parallel"),
    ("parallel_multiple", "Parallel<br>multiple"), ("non_live_overall", "Overall"), ("irrelevance", "Irrelevance"),
]
STRESS_GROUPS = [
    ("stress_plain", "Plain (control)", "nothing: a flat function with 2-3 required strings and integers"),
    ("stress_nested", "Nested", "objects as arguments: one object, a list of objects, 2 or 3 levels deep"),
    ("stress_enum", "Enum", "a code from a fixed list of 4, 12 or 30 codes"),
    ("stress_number", "Number", "integers (with thousands separators, in words, negative) and floats"),
    ("stress_optional", "Optional", "2, 4 or 8 optional arguments with defaults"),
    ("stress_unicode", "Unicode", "strings to copy exactly: accented Latin, non-Latin scripts, emoji"),
    ("stress_menu", "Menu", "choosing 1 of 50 functions with near-duplicate names"),
]
CLASSES = ["S1", "S2", "M1", "M2", "M3", "M4", "M5"]
DASH = "–"
BACK = "[Back to the final table](README.md)"


# ---- formatting ---------------------------------------------------------------------------------------------

def f2(value, signed=False):
    if value is None:
        return DASH
    if signed and round(value, 2) == 0:
        return "0.00"
    return f"{value:+.2f}" if signed else f"{value:.2f}"


def fmt_p(p):
    """p-value for a reader: three decimals, or a power of ten below 0.001."""
    if p >= 0.001:
        return f"{p:.3f}"
    mantissa, exponent = f"{p:.2e}".split("e")
    return f"{mantissa} × 10<sup>{int(exponent)}</sup>"


def bold_if(text, p_holm):
    """Bold a change whose Holm-adjusted p is below 0.05."""
    return f"**{text}**" if p_holm is not None and p_holm < 0.05 else text


def interval(low, high):
    return f"{f2(low, True)} to {f2(high, True)}"


def table(header, rows, left=1):
    """Markdown table; the first `left` columns are left-aligned, the rest right-aligned."""
    lines = ["| " + " | ".join(header) + " |", "|" + "---|" * left + "---:|" * (len(header) - left)]
    for row in rows:
        assert len(row) == len(header), (header, row)
        lines.append("| " + " | ".join(row) + " |")
    return lines


def grouped(header, groups):
    """Markdown table with one bold row per model, followed by that model's rows."""
    rows = []
    for name, model_rows in groups:
        rows.append([f"**{name}**"] + [""] * (len(header) - 1))
        rows += model_rows
    return table(header, rows)


def page(title, intro, body, sources):
    links = ", ".join(f"[`{s}`](../results/{s})" for s in sources)
    return ([f"# {title}", "", BACK, "", *intro, "", *body, "",
             "---", "", f"Built by [`src/final_table.py`](../src/final_table.py) from {links}.", ""])


# ---- main ---------------------------------------------------------------------------------------------------

def main():
    def read_csv(name):
        with open(RESULTS / name, newline="", encoding="utf-8") as f:
            return list(csv.DictReader(f))

    summary = {(r["run_id"], r["category"]): r for r in read_csv("summary.csv")}

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
        """Correct answers of a stress run, in one group or in all 7 (None if the run does not exist)."""
        groups = [group] if group else [g for g, _, _ in STRESS_GROUPS]
        rows = [summary.get((run_id(model, setting, mode, stress=True), g)) for g in groups]
        if any(r is None for r in rows):
            return None
        if any(r["complete"] != "True" for r in rows) or sum(int(r["total"]) for r in rows) != 25 * len(groups):
            raise SystemExit(f"stress run {run_id(model, setting, mode, True)} is not complete")
        return sum(int(r["correct"]) for r in rows)

    tests = {(r["model"], r["precision"], mode): r for mode in MODES for r in read_csv(f"significance_{mode}.csv")}
    kv_tests = {r["model"]: r for r in read_csv("kv_cache.csv")}
    gap = {(r["model"], r["precision"]): r for r in read_csv("gap.csv")}
    latency = {(r["model"], r["precision"], r["mode"]): r for r in read_csv("latency.csv")}
    stress_tests = read_csv("stress_significance.csv")
    kv_stress = read_csv("kv_cache_stress.csv")
    repro = read_csv("bfcl_repro/comparison.csv")

    fields = CLASSES + ["total", "correct", "correct_unparsed"]
    taxonomy = {}
    for mode in MODES:
        for r in read_csv(f"taxonomy_{mode}.csv"):
            counts = taxonomy.setdefault(r["run_id"], {c: 0 for c in fields})
            for c in fields:
                counts[c] += int(r[c])

    def test_of(model, setting, mode):
        """Paired test against FP16 in the same mode: change, interval, both one-sided counts, Holm p."""
        name = MODELS[model]
        if setting == "fp16-kv8":
            r, other = (kv_tests.get(name) if mode == "free" else None), "right_kv8_only"
        else:
            r, other = tests.get((name, SETTINGS[setting], mode)), "right_quantized_only"
        if r is None:
            return None
        return {"change": float(r["change"]), "low": float(r["ci_low"]), "high": float(r["ci_high"]),
                "fp16_only": int(r["right_at_fp16_only"]), "other_only": int(r[other]), "p": float(r["p_holm"])}

    mismatches = []

    def check(what, expected, got):
        if expected != got:
            mismatches.append(f"{what}: expected {expected}, got {got}")

    # ---- master table (README.md, final_table.csv) --------------------------------------------------------
    accuracy_groups, cost_groups, csv_rows = [], [], []
    for model, name in MODELS.items():
        base = {mode: accuracy(model, "fp16", mode, "non_live_overall") for mode in MODES}
        accuracy_rows, cost_rows = [], []
        for setting, label in SETTINGS.items():
            acc = {mode: accuracy(model, setting, mode, "non_live_overall") for mode in MODES}
            irr = {mode: accuracy(model, setting, mode, "irrelevance") for mode in MODES}
            st = {mode: stress(model, setting, mode) for mode in MODES}
            change, p = {}, {}
            for mode in MODES:
                change[mode] = p[mode] = None
                if setting != "fp16" and acc[mode] is not None:
                    change[mode] = acc[mode] - base[mode]
                    t = test_of(model, setting, mode)
                    p[mode] = t["p"]
                    check(f"change {name} {label} {mode} (summary.csv vs test file)",
                          f2(t["change"], True), f2(change[mode], True))
            gain = None if acc["constrained"] is None else acc["constrained"] - acc["free"]
            lat = {mode: latency[(name, label, mode)] for mode in MODES}
            gpus = int(lat["free"]["gpus"])
            weights = float(lat["free"]["weights_gib_per_gpu"])

            def change_cell(mode):
                if setting == "fp16":
                    return "base"
                return bold_if(f2(change[mode], True), p[mode])

            accuracy_rows.append([label, f2(acc["free"]), change_cell("free"), f2(acc["constrained"]),
                                  change_cell("constrained"), f2(gain, True), f2(irr["free"]),
                                  f2(irr["constrained"])])
            cost_rows.append([
                label, DASH if st["free"] is None else str(st["free"]),
                DASH if st["constrained"] is None else str(st["constrained"]),
                f"{weights:.2f}" + (f" ×{gpus}" if gpus > 1 else ""), f"{int(lat['free']['kv_cache_tokens']):,}",
                f"{float(lat['free']['output_tokens_per_second']):.1f}",
                f"{float(lat['free']['first_token_ms_median']):,.0f}", f"{float(lat['free']['total_s_median']):.2f}",
            ])
            row = {"model": name, "precision": label, "gpus": gpus, "weights_gib_per_gpu": f"{weights:.2f}",
                   "kv_cache_tokens": lat["free"]["kv_cache_tokens"]}
            for mode in MODES:
                row[f"bfcl_{mode}"] = f2(acc[mode])
                row[f"change_{mode}"] = f2(change[mode], True)
                row[f"p_holm_{mode}"] = "" if p[mode] is None else f"{p[mode]:.3g}"
            row["constraint_gain"] = f2(gain, True)
            for mode in MODES:
                row[f"irrelevance_{mode}"] = f2(irr[mode])
            for mode in MODES:
                row[f"stress_{mode}_of_175"] = "" if st[mode] is None else st[mode]
            for mode in MODES:
                row[f"tokens_per_s_{mode}"] = lat[mode]["output_tokens_per_second"]
                row[f"first_token_ms_{mode}"] = lat[mode]["first_token_ms_median"]
                row[f"whole_answer_s_{mode}"] = lat[mode]["total_s_median"]
            csv_rows.append({k: "" if v == DASH else v for k, v in row.items()})
        accuracy_groups.append((name, accuracy_rows))
        cost_groups.append((name, cost_rows))

    # ---- BFCL by category ---------------------------------------------------------------------------------
    def category_groups(mode, settings):
        groups = []
        for model, name in MODELS.items():
            rows = []
            for setting in settings:
                values = [accuracy(model, setting, mode, c) for c, _ in BFCL_CATEGORIES]
                if values[-2] is not None:
                    rows.append([SETTINGS[setting]] + [f2(v) for v in values])
            groups.append((name, rows))
        return groups

    # ---- significance -------------------------------------------------------------------------------------
    def test_groups(mode):
        groups = []
        for model, name in MODELS.items():
            rows = []
            for setting in MAIN[1:]:
                t = test_of(model, setting, mode)
                rows.append([SETTINGS[setting], bold_if(f2(t["change"], True), t["p"]), interval(t["low"], t["high"]),
                             str(t["fp16_only"]), str(t["other_only"]), fmt_p(t["p"])])
            groups.append((name, rows))
        return groups

    kv_rows = []
    for model, name in MODELS.items():
        t = test_of(model, "fp16-kv8", "free")
        kv_rows.append([name, bold_if(f2(t["change"], True), t["p"]), interval(t["low"], t["high"]),
                        str(t["fp16_only"]), str(t["other_only"]), fmt_p(t["p"])])

    gap_groups, n_losses = [], 0
    for model, name in MODELS.items():
        rows = []
        for setting in MAIN[1:]:
            g = gap[(name, SETTINGS[setting])]
            p_free = float(tests[(name, SETTINGS[setting], "free")]["p_holm"])
            change_free = float(g["change_free"])
            loss = p_free < 0.05 and change_free < 0
            n_losses += loss
            check(f"loss removed given for {name} {setting} exactly when it is a significant loss",
                  loss, bool(g["loss_removed_pct"]))
            rows.append([SETTINGS[setting], bold_if(f2(change_free, True), p_free),
                         f2(float(g["change_constrained"]), True), f2(float(g["difference"]), True),
                         interval(float(g["ci_low"]), float(g["ci_high"])),
                         f"{g['loss_removed_pct']}%" if g["loss_removed_pct"] else DASH])
        gap_groups.append((name, rows))

    # ---- stress suite -------------------------------------------------------------------------------------
    def stress_groups(mode, settings):
        groups = []
        for model, name in MODELS.items():
            rows = []
            for setting in settings:
                if stress(model, setting, mode) is not None:
                    per_group = [str(stress(model, setting, mode, g)) for g, _, _ in STRESS_GROUPS]
                    rows.append([SETTINGS[setting]] + per_group + [str(stress(model, setting, mode))])
            groups.append((name, rows))
        return groups

    group_names = {g: label for g, label, _ in STRESS_GROUPS}
    stress_sig = [r for r in stress_tests if float(r["p_holm"]) < 0.05]
    stress_sig_rows = [[r["model"], r["precision"], r["mode"], group_names[r["group"]], r["fp16_correct"],
                        r["quantized_correct"], f"{int(r['change']):+d}", fmt_p(float(r["p_holm"]))]
                       for r in stress_sig]
    n_kv_stress_sig = sum(float(r["p_holm"]) < 0.05 for r in kv_stress)
    tests_per_mode = {mode: sum(r["mode"] == mode for r in stress_tests) for mode in MODES}

    # ---- failure classes ----------------------------------------------------------------------------------
    def failure_groups(mode):
        groups, totals = [], {c: 0 for c in CLASSES + ["correct_unparsed"]}
        for model, name in MODELS.items():
            rows = []
            for setting in MAIN:
                t = taxonomy[run_id(model, setting, mode)]
                failures = sum(t[c] for c in CLASSES)
                # an irrelevance answer that does not parse makes no call, so BFCL counts it as correct
                check(f"failures {name} {setting} {mode} = total - correct - correct_unparsed",
                      t["total"] - t["correct"] - t["correct_unparsed"], failures)
                for c in totals:
                    totals[c] += t[c]
                syntactic = t["S1"] + t["S2"]
                rows.append([SETTINGS[setting], f"{failures:,}"] + [f"{t[c]:,}" for c in CLASSES]
                            + [f"{syntactic:,}", f"{failures - syntactic:,}"])
            groups.append((name, rows))
        failures = sum(totals[c] for c in CLASSES)
        syntactic = totals["S1"] + totals["S2"]
        total_row = (["**All 25 runs**", f"{failures:,}"] + [f"{totals[c]:,}" for c in CLASSES]
                     + [f"{syntactic:,}", f"{failures - syntactic:,}"])
        return groups, total_row, failures, syntactic, totals["correct_unparsed"]

    failures_by_mode = {mode: failure_groups(mode) for mode in MODES}

    kv_fail_rows = []
    for model, name in MODELS.items():
        r = kv_tests[name]
        newly = dict(item.split(":") for item in r["classes_of_newly_wrong"].split())
        semantic = sum(int(newly.get(c, 0)) for c in CLASSES[2:])
        kv_fail_rows.append([name, f"{int(r['failures_fp16']):,}", f"{int(r['failures_kv8']):,}",
                             newly.get("S1", "0"), newly.get("S2", "0"), str(semantic)])
        check(f"FP16 failures of {name} (kv_cache.csv vs taxonomy_free.csv)",
              int(r["failures_fp16"]), sum(taxonomy[run_id(model, "fp16", "free")][c] for c in CLASSES))

    # ---- speed and memory ---------------------------------------------------------------------------------
    memory_groups, speed_groups = [], []
    for model, name in MODELS.items():
        memory_rows, speed_rows = [], []
        for setting, label in SETTINGS.items():
            lf, lc = latency[(name, label, "free")], latency[(name, label, "constrained")]
            memory_rows.append([label, lf["gpus"], f"{float(lf['weights_gib_per_gpu']):.2f}",
                                f"{int(lf['kv_cache_tokens']):,}"])
            speed_rows.append([label] + [cell for l in (lf, lc) for cell in (
                f"{float(l['output_tokens_per_second']):.1f}", f"{float(l['first_token_ms_median']):,.0f}",
                f"{float(l['total_s_median']):.2f}")])
        memory_groups.append((name, memory_rows))
        speed_groups.append((name, speed_rows))

    # ---- setup checks -------------------------------------------------------------------------------------
    by_mode = {}
    for r in repro:
        mode = "FC" if r["Model"].endswith("(FC)") else "Prompt"
        by_mode.setdefault(r["Category"], {})[mode] = r
    repro_rows = []
    for category, modes in by_mode.items():
        row = [category]
        for mode in ["Prompt", "FC"]:
            r = modes[mode]
            row += [f2(float(r["Ours (FP16, T4)"])), f2(float(r["Leaderboard"])), f2(float(r["Difference"]), True)]
        repro_rows.append(row)
    overall_diff = {mode: abs(float(by_mode["Non-Live Overall Acc"][mode]["Difference"])) for mode in ["Prompt", "FC"]}
    pilot = [f2(accuracy("qwen3-4b-instruct-2507", "fp16", "free", c)) for c, _ in BFCL_CATEGORIES]

    # ---- counts for the README ----------------------------------------------------------------------------
    n_sig = {mode: sum(float(tests[(n, SETTINGS[s], mode)]["p_holm"]) < 0.05 for n in MODELS.values()
                       for s in MAIN[1:]) for mode in MODES}
    n_kv_sig = sum(float(r["p_holm"]) < 0.05 for r in kv_tests.values())
    main_rows = [r for r in csv_rows if r["precision"] != SETTINGS["fp16-kv8"]]
    not_raised = [f"{r['model']} {r['precision']}" for r in main_rows if float(r["constraint_gain"]) <= 0]
    check("stress tests per mode", {"free": 140, "constrained": 140}, tests_per_mode)

    if mismatches:
        raise SystemExit("Nothing written. Mismatches:\n" + "\n".join(mismatches))

    # ---- write --------------------------------------------------------------------------------------------
    OUT.mkdir(exist_ok=True)
    files = {}
    free_f, free_s, unparsed_free = failures_by_mode["free"][2:]
    con_f, con_s, unparsed_con = failures_by_mode["constrained"][2:]

    files["README.md"] = [
        "# Final Table",
        "",
        "Every score of the project in one place. The experiments were frozen in Update 15; every value in this "
        "folder is read from the files in [`results/`](../results/) by [`src/final_table.py`](../src/final_table.py), "
        "and none is typed by hand.",
        "",
        "**What was run:** 5 models (Qwen3-1.7B, Qwen3-4B, Qwen3-8B, Llama-3.2-1B-Instruct, Llama-3.2-3B-Instruct) "
        "at 5 weight precisions (FP16, INT8, NF4, GPTQ, AWQ), in free and in JSON-schema-constrained decoding, on the "
        "1,390 BFCL `non_live` questions and on our 175-question schema-stress suite; and the 5 FP16 models with an "
        "8-bit KV cache, in free decoding. That is 110 runs, plus a speed test of all 30 configs. Greedy decoding, "
        "vLLM 0.8.5, Kaggle Tesla T4 GPUs.",
        "",
        "## How to read the tables",
        "",
        "- **BFCL** = BFCL's Non-Live Overall accuracy (%): the unweighted mean of simple (itself the mean of Python, "
        "Java and JavaScript), multiple, parallel and parallel multiple.",
        "- **Change** = points against FP16 in the same decoding mode. **Bold** = significant: McNemar's exact test, "
        "paired by question on the 1,150 AST questions, with Holm's correction (p < 0.05). A small change can be "
        "significant when the two runs get many different questions right: Llama-3.2-1B AWQ in constrained decoding "
        f"changes by {f2(test_of('llama-3.2-1b', 'awq', 'constrained')['change'], True)}, with "
        f"{test_of('llama-3.2-1b', 'awq', 'constrained')['other_only']} AST questions right only with AWQ and "
        f"{test_of('llama-3.2-1b', 'awq', 'constrained')['fp16_only']} only with FP16.",
        "- **Constraint gain** = constrained score minus free score of the same run.",
        "- **Irrelevance** = accuracy (%) on the 240 questions where no function fits (correct = no call). As in "
        "BFCL, it is not part of the Overall score.",
        "- **Stress** = correct answers out of the 175 stress-suite questions (7 groups of 25).",
        f"- {DASH} = not run: the 8-bit KV cache was tested in free decoding only.",
        "",
        "## BFCL accuracy (%)",
        "",
    ] + grouped(["Model / precision", "Free", "Change", "Constrained", "Change", "Constraint<br>gain",
                 "Irrelevance<br>free", "Irrelevance<br>constrained"], accuracy_groups) + [
        "",
        "## Stress suite, memory and speed",
        "",
    ] + grouped(["Model / precision", "Stress<br>free", "Stress<br>constrained", "Weights<br>(GiB per GPU)",
                 "KV cache<br>tokens", "Tokens/s", "First token<br>(ms)", "Whole answer<br>(s)"], cost_groups) + [
        "",
        "Memory is read from vLLM's log: the weights per GPU, and how many tokens of keys and values fit in the "
        "memory left after them. Speed is for free decoding at batch size 1, as the median over 35 questions; "
        "tokens/s is the speed of writing the answer. ×2 = the model is split over two T4s (Qwen3-8B with FP16 "
        "weights); every other config runs on one. Speed differences of about 20% or less are within the noise of "
        "Kaggle's T4s. Speed in constrained decoding: [Speed and memory](speed_memory.md).",
        "",
        "## In numbers",
        "",
        f"- **Significant changes from FP16:** {n_sig['free']} of 20 in free decoding, {n_sig['constrained']} of 20 "
        f"in constrained decoding, and {n_kv_sig} of 5 with the 8-bit KV cache.",
        f"- **The constraint raises {len(main_rows) - len(not_raised)} of the {len(main_rows)} runs** "
        f"(all but {', '.join(not_raised)}).",
        f"- **Format errors** (syntactic failures): {free_s:,} of {free_f:,} failures ({100 * free_s / free_f:.1f}%) "
        f"in free decoding, {con_s:,} of {con_f:,} ({100 * con_s / con_f:.1f}%) in constrained decoding.",
        f"- **Stress suite:** {len(stress_sig)} of the {len(stress_tests)} group tests against FP16 are significant; "
        f"with the 8-bit KV cache, {n_kv_stress_sig} of {len(kv_stress)}.",
        f"- **BFCL reproduced on Kaggle** within {overall_diff['Prompt']:.2f} points (prompt mode) and "
        f"{overall_diff['FC']:.2f} points (FC mode) of the leaderboard.",
        "",
        "## More tables",
        "",
    ] + table(["Page", "What it has"], [
        ["[BFCL by category](bfcl_by_category.md)", "accuracy in each BFCL category, for every run"],
        ["[Significance tests](significance.md)", "every change from FP16 with its 95% interval and p-value, and how "
                                                   "much of each change the constraint removes"],
        ["[Stress suite](stress_suite.md)", "correct answers in each of the 7 stress groups, for every run, and the "
                                            f"{len(stress_sig)} significant changes"],
        ["[Failure classes](failure_classes.md)", "every failed answer, sorted into 2 syntactic and 5 semantic "
                                                  "classes"],
        ["[Speed and memory](speed_memory.md)", "memory, KV cache size and speed of all 30 configs, in both "
                                                "decoding modes"],
        ["[Setup checks](setup_checks.md)", "BFCL reproduced against its leaderboard, and the other checks of the "
                                            "method"],
        ["[`final_table.csv`](final_table.csv)", "the two tables above as data, with p-values and the speed in "
                                                 "constrained decoding"],
    ], left=2) + [
        "",
        "What the numbers mean: [`findings/`](../findings/README.md). Answers to the four research questions: "
        "[`research_questions/`](../research_questions/README.md). To build this folder again: "
        "`python src/final_table.py`.",
        "",
    ]

    cat_header = ["Model / precision"] + [label for _, label in BFCL_CATEGORIES]
    files["bfcl_by_category.md"] = page(
        "BFCL Accuracy by Category",
        ["Accuracy (%) of every run in each BFCL `non_live` category. Questions per category: Python 400, Java 100, "
         "JavaScript 50, multiple 200, parallel 200, parallel multiple 200, irrelevance 240. Simple is the mean of "
         "Python, Java and JavaScript; Overall is the mean of simple, multiple, parallel and parallel multiple. "
         "Irrelevance is not part of Overall."],
        ["## Free decoding", ""] + grouped(cat_header, category_groups("free", SETTINGS))
        + ["", "## Constrained decoding", ""] + grouped(cat_header, category_groups("constrained", MAIN)),
        ["summary.csv"])

    test_header = ["Model / precision", "Change", "95% interval", "Right at<br>FP16 only",
                   "Right<br>quantized only", "p (Holm)"]
    files["significance.md"] = page(
        "Significance Tests",
        ["Every change from FP16 is tested question by question: McNemar's exact test on the 1,150 AST questions "
         "(every category except irrelevance), with Holm's correction over the 20 runs of a decoding mode (over the "
         "5 models for the 8-bit KV cache). *Right at FP16 only* and *right quantized only* count the AST questions "
         "that only one of the two runs gets right. The 95% interval of the change in Overall is a bootstrap: "
         "10,000 resamples of the questions, drawn inside each category. **Bold** = Holm-adjusted p < 0.05.", "",
         "The test and the interval can disagree. The test counts the questions whose answer changes; the interval "
         "is for the change in Overall, an average over the categories, and it is not adjusted for the number of "
         "tests. So Qwen3-8B INT8 in free decoding has an interval without 0 but no significant test, and "
         "Qwen3-1.7B with the 8-bit KV cache has a significant test but an interval with 0."],
        ["## Quantized weights, free decoding", ""] + grouped(test_header, test_groups("free"))
        + ["", "## Quantized weights, constrained decoding", ""] + grouped(test_header, test_groups("constrained"))
        + ["", "## 8-bit KV cache (FP16 weights, free decoding)", ""]
        + table(["Model", "Change", "95% interval", "Right at<br>FP16 only", "Right with<br>8-bit KV only",
                 "p (Holm)"], kv_rows)
        + ["", "## How much of the free change the constraint removes", "",
           "*Difference* = constrained change minus free change, with its bootstrap 95% interval. *Loss removed* = "
           "the part of a significant free loss that is gone under the constraint (100% = all of it; below 0% = the "
           f"loss grows). It is given for the {n_losses} significant free losses only.", ""]
        + grouped(["Model / precision", "Free change", "Constrained<br>change", "Difference", "95% interval",
                   "Loss removed"], gap_groups),
        ["significance_free.csv", "significance_constrained.csv", "kv_cache.csv", "gap.csv"])

    stress_header = ["Model / precision"] + [label for _, label, _ in STRESS_GROUPS] + ["Total<br>(of 175)"]
    files["stress_suite.md"] = page(
        "Schema-Stress Suite",
        ["Correct answers of every run in each group of our schema-stress suite: 7 groups of 25 questions. Six "
         "groups each stress one feature of a function's schema; the plain group is a control without any of them. "
         "Full description: [`data/stress_suite/`](../data/stress_suite/README.md).", ""]
        + table(["Group", "What it stresses"], [[label, text] for _, label, text in STRESS_GROUPS], left=2),
        ["## Free decoding", ""] + grouped(stress_header, stress_groups("free", SETTINGS))
        + ["", "## Constrained decoding", ""] + grouped(stress_header, stress_groups("constrained", MAIN))
        + ["", "## Significant changes", "",
           "McNemar's exact test on the 25 questions of a group, against FP16 in the same decoding mode, with Holm's "
           f"correction over the {tests_per_mode['free']} tests of each mode ({len(stress_tests)} in all). "
           f"Of these, {len(stress_sig)} are significant:", ""]
        + table(["Model", "Precision", "Mode", "Group", "FP16<br>(of 25)", "Quantized<br>(of 25)", "Change",
                 "p (Holm)"], stress_sig_rows, left=4)
        + ["", f"With the 8-bit KV cache, {'none' if n_kv_stress_sig == 0 else n_kv_stress_sig} of the "
           f"{len(kv_stress)} group tests (Holm's correction over the {len(kv_stress)}) "
           f"{'is' if n_kv_stress_sig in (0, 1) else 'are'} significant."],
        ["summary.csv", "stress_significance.csv", "kv_cache_stress.csv"])

    fail_header = ["Model / precision", "Failures"] + CLASSES + ["Syntactic<br>(S1+S2)", "Semantic<br>(M1-M5)"]
    files["failure_classes.md"] = page(
        "Failure Classes",
        ["Every failed answer of the 50 main BFCL runs (25 free, 25 constrained), sorted into one class by "
         "[`src/taxonomy.py`](../src/taxonomy.py) (checked by hand on 80 failures: 80 agree). The checks run in "
         "this order, so an answer with a format error "
         "is syntactic even if its content is also wrong.", "",
         "- **Syntactic** (the format is wrong): **S1** not valid JSON; **S2** valid JSON that does not match the "
         "schema (wrong keys, a function that is not in the list, an extra or missing argument, a wrong type, or a "
         "value outside an enum)",
         "- **Semantic** (the format is right, the content is wrong): **M1** wrong function; **M2** wrong argument "
         "value; **M3** wrong number of calls; **M4** a call on an irrelevance question; **M5** no call when one was "
         "needed", "",
         "An irrelevance answer that does not parse makes no call, so BFCL counts it as correct; these answers "
         f"({unparsed_free:,} in the free runs, {unparsed_con:,} in the constrained runs) are not failures."],
        ["## Free decoding", ""] + grouped(fail_header, failures_by_mode["free"][0])
        + [f"| {' | '.join(failures_by_mode['free'][1])} |", "",
           f"Syntactic: {free_s:,} of {free_f:,} failures ({100 * free_s / free_f:.1f}%).", "",
           "## Constrained decoding", ""] + grouped(fail_header, failures_by_mode["constrained"][0])
        + [f"| {' | '.join(failures_by_mode['constrained'][1])} |", "",
           f"Syntactic: {con_s:,} of {con_f:,} failures ({100 * con_s / con_f:.1f}%).", "",
           "## 8-bit KV cache (FP16 weights, free decoding)", "",
           "Failed answers at FP16 and with the 8-bit KV cache, and the classes of the answers that were right at "
           "FP16 and are wrong with the 8-bit KV cache.", ""]
        + table(["Model", "Failures<br>at FP16", "Failures with<br>8-bit KV", "Newly wrong:<br>S1",
                 "Newly wrong:<br>S2", "Newly wrong:<br>M1-M5"], kv_fail_rows),
        ["taxonomy_free.csv", "taxonomy_constrained.csv", "kv_cache.csv"])

    files["speed_memory.md"] = page(
        "Speed and Memory",
        ["Each config was loaded on its own and answered the same 35 BFCL questions (5 per category, seed 42) one at "
         "a time (batch size 1), after 3 untimed warm-up questions, first in free and then in constrained decoding "
         "([`src/latency.py`](../src/latency.py); notebooks "
         "[`08_kv_latency_run_A`](../notebooks/08_kv_latency_run_A.ipynb) and "
         "[`_B`](../notebooks/08_kv_latency_run_B.ipynb)). Prompts, checkpoints and vLLM 0.8.5 are those of the "
         "accuracy runs. One T4 per config, two for Qwen3-8B with FP16 weights. Values are medians over the 35 "
         "questions.", "",
         "- **Weights per GPU** and **KV cache tokens** are read from vLLM's log. KV cache tokens = how many tokens "
         "of keys and values fit in the memory left after the weights (90% of the GPU; 81% for Qwen3-8B INT8).",
         "- **Tokens/s** = the speed of writing the answer: answer tokens after the first one, per second.",
         "- **First token** = from the request's arrival to the first answer token. It includes reading the prompt, "
         "and in constrained decoding also building the grammar for the question's schema.",
         "- **Whole answer** = what a caller waits for. It mixes speed and answer length."],
        ["## Memory", ""]
        + grouped(["Model / precision", "GPUs", "Weights<br>(GiB per GPU)", "KV cache<br>tokens"], memory_groups)
        + ["", "## Speed at batch size 1", ""]
        + grouped(["Model / precision", "Free<br>tokens/s", "Free<br>first token (ms)", "Free<br>whole answer (s)",
                   "Constrained<br>tokens/s", "Constrained<br>first token (ms)", "Constrained<br>whole answer (s)"],
                  speed_groups)
        + ["", "## Notes", "",
           "- Llama-3.2-3B with FP16 weights was timed twice; the table uses the second try (21 Nov 2026). The GPU "
           "slowed down during that run, so its constrained-decoding values are not reliable. Both tries are in "
           "[`results/latency/extra/`](../results/latency/extra/) and [`results/latency.md`](../results/latency.md).",
           "- Llama-3.2-1B NF4 and GPTQ in constrained decoding: most answers are an empty list, so tokens/s rests on "
           "very short answers and is below the model's real writing speed.",
           "- One run per config: speed differences of about 20% or less are not meaningful."],
        ["latency.csv"])

    files["setup_checks.md"] = page(
        "Setup Checks",
        ["Checks that the benchmark, the scorer and the classifier can be trusted, made before and during the "
         "study."],
        ["## BFCL reproduced on Kaggle", "",
         "BFCL's own harness, run on Qwen3-4B-Instruct-2507 in FP16 on one Kaggle T4, in BFCL's prompt mode and in "
         "its function-calling (FC) mode, against BFCL's leaderboard (downloaded 3 Oct 2026). Accuracy (%); report: "
         "[`notes/bfcl_repro.md`](../notes/bfcl_repro.md).", ""]
        + table(["Category", "Prompt mode<br>ours", "Prompt mode<br>leaderboard", "Prompt mode<br>difference",
                 "FC mode<br>ours", "FC mode<br>leaderboard", "FC mode<br>difference"], repro_rows)
        + ["", "## First run of our own script (Update 3)", "",
           "Our prompt format and scorer on Qwen3-4B-Instruct-2507 (not one of the five study models) in FP16 with "
           "free decoding. Accuracy (%); report: [`notes/eval_sanity_check.md`](../notes/eval_sanity_check.md).", ""]
        + table(["Model"] + [label for _, label in BFCL_CATEGORIES], [["Qwen3-4B-Instruct-2507"] + pilot])
        + ["", "## Other checks", "",
           "- **Scorer:** re-scoring the 1,390 answers of the BFCL reproduction with our scorer gives BFCL's verdict "
           "on every question ([`src/check_scorer.py`](../src/check_scorer.py)).",
           "- **Re-scoring:** every answer of every run was scored again on a PC, with the same result.",
           "- **Same prompts:** all 7,825 prompts of the 8-bit KV cache runs are the same as in the FP16 runs "
           "([`results/kv_cache.md`](../results/kv_cache.md)).",
           "- **Failure classifier:** checked by hand on 80 random failures; all 80 agree "
           "([`notes/taxonomy_check.md`](../notes/taxonomy_check.md)).",
           "- **Schema against BFCL's checker:** 14 of the 34,750 free answers are correct by BFCL but do not match "
           "the schema ([`results/taxonomy_free.md`](../results/taxonomy_free.md)).",
           "- **Parser:** Llama-3.2-3B FP16 scores 31.38 with our strict parser and would score 46.02 with BFCL's "
           "bracket repair. This was a diagnostic only; the reported score stays 31.38 "
           "([`results/baseline_fp16.md`](../results/baseline_fp16.md))."],
        ["bfcl_repro/comparison.csv", "summary.csv"])

    for name, lines in files.items():
        (OUT / name).write_text("\n".join(lines).rstrip("\n") + "\n", encoding="utf-8")
    with open(OUT / "final_table.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(csv_rows[0]))
        writer.writeheader()
        writer.writerows(csv_rows)
    print(f"All checks passed. Wrote {len(files) + 1} files to {OUT}")


if __name__ == "__main__":
    main()
