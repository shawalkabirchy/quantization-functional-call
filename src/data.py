"""Load questions, function definitions and possible answers by category.

BFCL's categories come from the installed bfcl-eval package (pinned to 2026.3.23), so the questions and the
answer key are exactly the ones `bfcl generate` and `bfcl evaluate` use. The schema-stress suite (Update 11)
comes from data/stress_suite/, made by src/stress_suite.py in the same layout.
"""

import json
from pathlib import Path

from bfcl_eval.utils import (
    is_java,
    is_js,
    is_relevance_or_irrelevance,
    load_dataset_entry,
    load_ground_truth_entry,
)

# BFCL's single-turn "non_live" group: 1,390 questions in total
NON_LIVE = [
    "simple_python",      # 400
    "simple_java",        # 100
    "simple_javascript",  # 50
    "multiple",           # 200
    "parallel",           # 200
    "parallel_multiple",  # 200
    "irrelevance",        # 240
]
# Our schema-stress suite: 25 questions each (see src/stress_suite.py)
STRESS = [
    "stress_plain",     # control: flat function, no stressed feature
    "stress_nested",
    "stress_enum",
    "stress_number",
    "stress_optional",
    "stress_unicode",
    "stress_menu",      # 50 functions to choose from
]
CATEGORIES = NON_LIVE + STRESS
GROUPS = {"non_live": NON_LIVE, "stress": STRESS}
STRESS_DIR = Path(__file__).resolve().parents[1] / "data" / "stress_suite"


def expand_categories(names):
    """Turn names like ["non_live"], ["stress"] or ["simple_python", "parallel"] into a list of categories."""
    categories = []
    for name in names:
        for category in GROUPS.get(name, [name]):
            if category not in CATEGORIES:
                raise ValueError(f"Unsupported category {category!r}. Use non_live, stress or any of: "
                                 f"{', '.join(CATEGORIES)}")
            if category not in categories:
                categories.append(category)
    return categories


def checker_category(category):
    """The category name to give BFCL's checker, which picks its checker by name.

    The menu questions have many functions and one correct call, like BFCL's "multiple"; the other stress
    groups have one function and one call, like "simple".
    """
    return "multiple" if category == "stress_menu" else category


def _read_lines(path):
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f]


def language(category):
    """Language of a category's function definitions: python, java or javascript."""
    if is_java(category):
        return "java"
    if is_js(category):
        return "javascript"
    return "python"


def load_questions(category):
    """Questions as the model sees them, in BFCL's order.

    BFCL adds a language hint to every function description, and for Java and JavaScript it changes every
    parameter type to "string" (its checker converts the strings back). BFCL shows the model these
    versions in both prompt and FC mode, so we do the same. The stress suite has no language hint.
    """
    if category in STRESS:
        entries = _read_lines(STRESS_DIR / f"{category}.json")
    else:
        entries = load_dataset_entry(category, include_prereq=False, include_language_specific_hint=True)
    questions = []
    for entry in entries:
        if len(entry["question"]) != 1:
            raise ValueError(f"{entry['id']}: expected a single-turn question")
        questions.append({
            "id": entry["id"],
            "category": category,
            "messages": entry["question"][0],
            "functions": entry["function"],
        })
    return questions


def load_answer_key(category):
    """For each question id: the original function definitions and the possible answers.

    The checker gets the definitions without the language hint, exactly as in `bfcl evaluate`.
    Irrelevance questions have no possible answers: the correct reply makes no call.
    """
    if category in STRESS:
        answers = {a["id"]: a["ground_truth"] for a in _read_lines(STRESS_DIR / "possible_answer" / f"{category}.json")}
        return {e["id"]: {"functions": e["function"], "ground_truth": answers[e["id"]]}
                for e in _read_lines(STRESS_DIR / f"{category}.json")}
    entries = load_dataset_entry(category, include_prereq=False, include_language_specific_hint=False)
    if is_relevance_or_irrelevance(category):
        return {e["id"]: {"functions": e["function"], "ground_truth": None} for e in entries}
    answers = {a["id"]: a["ground_truth"] for a in load_ground_truth_entry(category)}
    return {e["id"]: {"functions": e["function"], "ground_truth": answers[e["id"]]} for e in entries}
