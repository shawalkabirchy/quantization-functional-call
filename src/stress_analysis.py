"""Analyse the schema-stress runs (Update 12): accuracy per group and level, failure classes, and paired tests.

Usage (from the repository root, in the BFCL environment; needs results/raw/ and results/scores/ of the 50 stress
runs, <config>_<mode>_stress):
    python src/stress_analysis.py

Writes three tables:

results/stress_accuracy.csv, one row per run, group and level: correct answers out of total.

results/stress_failures.csv, one row per run and group: the failure classes of src/taxonomy.py (S1, S2 syntactic;
M1-M5 semantic) and, for wrong values (M2), how many are in the argument the group stresses ("m2_on_target") and how
many are in another argument ("m2_other"); and, for answers that are not valid JSON (S1), how many are right once only
the closing brackets at the end are fixed ("s1_right_when_closed": the content was right). The stressed arguments are: nested, an object or a list of objects; enum,
the coded argument; number, an integer or float argument; optional, an argument that is not required; unicode, a
string whose correct value has a non-ASCII character. For nested objects BFCL's error names the inner argument,
so every name inside the object counts too. The control and menu groups have none (the menu stresses the
choice of function, M1).

results/stress_significance.csv, one row per quantized run and group (20 x 7 per mode): the change from FP16 of the
same model and mode, in correct answers out of 25, next to the change in the control group; McNemar's exact test on
the 25 paired questions, with Holm's correction over the 140 tests of each mode.
"""

import csv
import json
import re
from collections import Counter

from data import STRESS, load_answer_key, load_questions
from main_table import MODELS, PRECISIONS, REPO
from significance import holm, mcnemar_exact
from score import score_reply
from taxonomy import FAILURES, classify_run, read_jsonl

MODES = ["free", "constrained"]
RIGHT = ("correct", "correct_unparsed")
PARAMETER = re.compile(r"parameter '([^']+)'")


def run_id(model, precision, mode):
    return f"{model}_{precision}_{mode}_stress"


def close_brackets(text):
    """The answer with the closing brackets at its end replaced by the ones its open brackets need."""
    body = text.rstrip().rstrip("]}").rstrip()
    needed, in_string, escaped = [], False, False
    for ch in body:
        if in_string:
            if escaped:
                escaped = False
            elif ch == "\\":
                escaped = True
            elif ch == '"':
                in_string = False
        elif ch == '"':
            in_string = True
        elif ch in "[{":
            needed.append("]" if ch == "[" else "}")
        elif ch in "]}" and needed:
            needed.pop()
    return body + "".join(reversed(needed))


def inner_names(spec):
    """All property names inside an object or a list of objects, at any depth."""
    spec = spec.get("items", spec)
    names = []
    for name, inner in spec.get("properties", {}).items():
        names += [name, *inner_names(inner)]
    return names


def stressed_arguments(group, functions, ground_truth):
    """The argument names that a group stresses, for one question (empty for the control and the menu)."""
    properties = {name: spec for f in functions for name, spec in f["parameters"]["properties"].items()}
    required = {name for f in functions for name in f["parameters"].get("required", [])}
    if group == "stress_nested":
        names = set()
        for name, spec in properties.items():
            if spec["type"] in ("dict", "array"):
                names |= {name, *inner_names(spec)}
        return names
    if group == "stress_enum":
        return {n for n, s in properties.items() if "enum" in s}
    if group == "stress_number":
        return {n for n, s in properties.items() if s["type"] in ("integer", "float")}
    if group == "stress_optional":
        return set(properties) - required
    if group == "stress_unicode":
        values = {n: v for call in ground_truth for args in call.values() for n, v in args.items()}
        return {n for n, accepted in values.items() if any(isinstance(a, str) and not a.isascii() for a in accepted)}
    return set()


def main():
    questions, keys, level = {}, {}, {}
    for group in STRESS:
        for q in load_questions(group):
            questions[q["id"]] = q
        keys.update(load_answer_key(group))
        with open(REPO / "data" / "stress_suite" / f"{group}.json", encoding="utf-8") as f:
            level.update({r["id"]: r["level"] for r in map(json.loads, f)})
    group_of = {i: q["category"] for i, q in questions.items()}

    labels = {}   # (model, precision, mode) -> {question id: class}
    errors = {}   # (model, precision, mode) -> {question id: BFCL's error text}
    outputs = {}  # (model, precision, mode) -> {question id: the answer}
    for model in MODELS:
        for precision in PRECISIONS:
            for mode in MODES:
                key = (model, precision, mode)
                labels[key] = {i: label for i, _, label in classify_run(run_id(*key), REPO / "results", questions, keys)}
                scores = read_jsonl(REPO / "results" / "scores" / f"{run_id(*key)}.jsonl")
                errors[key] = {s["id"]: " ".join(s["error"] or []) for s in scores}
                outputs[key] = {r["id"]: r["output"] for r in read_jsonl(REPO / "results" / "raw" / f"{run_id(*key)}.jsonl")}
                assert len(labels[key]) == 175, key

    accuracy, failures = [], []
    for (model, precision, mode), runs in labels.items():
        for group in STRESS:
            ids = [i for i in questions if group_of[i] == group]
            for lv in dict.fromkeys(level[i] for i in ids):
                in_level = [i for i in ids if level[i] == lv]
                accuracy.append({"model": MODELS[model], "precision": PRECISIONS[precision], "mode": mode,
                                 "group": group, "level": lv, "total": len(in_level),
                                 "correct": sum(runs[i] in RIGHT for i in in_level)})
            counts = Counter(runs[i] for i in ids)
            on_target, closed = 0, 0
            for i in ids:
                if runs[i] == "S1":
                    closed += score_reply(group, close_brackets(outputs[(model, precision, mode)][i]), keys[i])["valid"]
                if runs[i] == "M2":
                    m = PARAMETER.search(errors[(model, precision, mode)][i])
                    stressed = stressed_arguments(group, keys[i]["functions"], keys[i]["ground_truth"])
                    on_target += bool(m and m.group(1) in stressed)
            failures.append({"model": MODELS[model], "precision": PRECISIONS[precision], "mode": mode, "group": group,
                             "total": len(ids), "correct": counts["correct"] + counts["correct_unparsed"],
                             **{c: counts[c] for c in FAILURES}, "m2_on_target": on_target,
                             "m2_other": counts["M2"] - on_target, "s1_right_when_closed": closed})

    tests = []
    for mode in MODES:
        rows = []
        for model in MODELS:
            base = labels[(model, "fp16", mode)]
            for precision in PRECISIONS:
                if precision == "fp16":
                    continue
                other = labels[(model, precision, mode)]
                change = {}
                for group in STRESS:
                    ids = [i for i in questions if group_of[i] == group]
                    lost = sum(base[i] in RIGHT and other[i] not in RIGHT for i in ids)
                    gained = sum(base[i] not in RIGHT and other[i] in RIGHT for i in ids)
                    change[group] = gained - lost
                    rows.append({"model": MODELS[model], "precision": PRECISIONS[precision], "mode": mode,
                                 "group": group, "fp16_correct": sum(base[i] in RIGHT for i in ids),
                                 "quantized_correct": sum(other[i] in RIGHT for i in ids), "change": gained - lost,
                                 "control_change": None, "right_at_fp16_only": lost, "right_quantized_only": gained,
                                 "p_mcnemar": mcnemar_exact(lost, gained)})
                for row in rows[-len(STRESS):]:
                    row["control_change"] = change["stress_plain"]
        for row, p in zip(rows, holm([r["p_mcnemar"] for r in rows])):
            row["p_holm"] = p
        tests += rows

    for name, table in (("stress_accuracy", accuracy), ("stress_failures", failures), ("stress_significance", tests)):
        out = REPO / "results" / f"{name}.csv"
        with open(out, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(table[0]))
            writer.writeheader()
            for row in table:
                writer.writerow({k: f"{v:.3g}" if k.startswith("p_") else v for k, v in row.items()})
        print(f"Saved {out} ({len(table)} rows)")

    for mode in MODES:
        print(f"\nChanges from FP16 with Holm-adjusted p < 0.05, {mode} decoding (correct answers out of 25):")
        hits = [r for r in tests if r["mode"] == mode and r["p_holm"] < 0.05]
        for r in hits:
            print(f"  {r['model']:24s}{r['precision']:6s}{r['group']:17s}{r['fp16_correct']:3d} -> "
                  f"{r['quantized_correct']:3d}   control {r['control_change']:+d}   p (Holm) {r['p_holm']:.3g}")
        if not hits:
            print("  none")


if __name__ == "__main__":
    main()
