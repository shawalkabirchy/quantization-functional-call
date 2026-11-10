# Schema-Stress Suite (v1)

175 small function-calling questions in 7 groups of 25. Six groups each stress one feature of a function's
schema; the seventh (`stress_plain`) is a control without any of them. Each question has exactly one correct
call, and every value in it is written in the question (enum codes: in the function description). The suite
answers RQ4: which schema features break first as precision drops.

| Group | What it stresses | Levels (questions) | Example question |
|---|---|---|---|
| `stress_plain` | nothing (control): a flat function with 2-3 required strings and integers, ASCII only | control (25) | Book a table at Saffron Lane for 8 people at 20:00. |
| `stress_nested` | objects as arguments | one object (7), list of objects (6), 2 levels deep (6), 3 levels deep (6) | Create an event called "Science Fair" at Riverside Center, 95 Main Street, Hanoi. |
| `stress_enum` | a code from a fixed list; the description maps each code to its meaning, in a new random order for every question | 4 codes (8), 12 codes (9), 30 codes (8) | Move pallet PAL-9164 to the Riga warehouse. |
| `stress_number` | integer vs. float | integers with thousands separators, in words or negative (8), floats (8), both in one call (9) | Set the stock level of SKU-376 to thirty-eight units. |
| `stress_optional` | optional arguments with defaults; the question sets none, one or two of them | 2 (8), 4 (9) or 8 (8) optional arguments | Schedule a meeting called "Budget review" on 2026-12-22, with a reminder 5 minutes before. |
| `stress_unicode` | strings to copy exactly | accented Latin (8), non-Latin scripts (9), emoji (8) | Add সাকিব হাসান from ঢাকা to my contacts. |
| `stress_menu` | choosing 1 of 50 functions (5 actions x 10 objects, so many names are near duplicates) | correct function at position 1-10 (8), 21-30 (9), 41-50 (8) | Can you archive the coupon with ID CPN-5113? |

All names, IDs and places are invented or common place names.

## Files

The layout follows BFCL's, one JSON object per line:

- `<group>.json`: `id`, `level`, `question` (the chat messages) and `function` (the definitions, in BFCL's format)
- `possible_answer/<group>.json`: `id` and `ground_truth`, in BFCL's format: for each argument a list of accepted
  values, where `""` means the argument may be left out. For the optional group, an argument the question does
  not mention accepts only "left out" or its default.

## How it is made and checked

[`src/stress_suite.py`](../../src/stress_suite.py) makes every file from templates with fixed seeds (one per
group), so `python src/stress_suite.py` writes exactly these files again. `python src/stress_suite.py --check`
checks that:

1. the files on disk are exactly what the generator makes, with 25 questions per group and unique IDs;
2. the correct call is accepted by BFCL's checker (scored by `src/score.py`, as for BFCL's own questions) and
   matches the JSON schema that constrained decoding uses (`src/schema.py`);
3. a wrong call is rejected: one value changed (for nested arguments, also the deepest value), another code
   from the list (enum), another action on the same object (menu), or an argument the question did not mention
   (optional); 220 wrong calls in total;
4. every string in the correct call appears in the question, except enum codes.

## Notes

- **Scoring** uses BFCL's AST checker, like the main study (`src/score.py`). The menu group is scored like BFCL's
  `multiple` category (several functions, one call), the others like `simple`.
- **String comparison:** BFCL's checker ignores case, spaces and the characters `,./-_*^` when it compares a
  string argument, or a string directly inside an object or a list of objects. Inside objects nested 2 or 3
  levels deep it compares exactly. The questions write those values exactly as the answer expects them.
- **Menu functions have no argument descriptions.** With them, 50 functions do not fit the prompt budget
  (`max_model_len` 8,192 minus 4,096 answer tokens). The longest menu prompt is 3,646 tokens with Qwen3's
  tokenizer and 3,662 with Llama-3.2's; every other prompt is under 700.
- **Constrained decoding** compiles one grammar per question. All 175 compile with XGrammar 0.1.18 (the version
  vLLM 0.8.5 uses), with the Qwen3 and Llama-3.2 tokenizers. On a PC CPU a menu grammar takes about 16 seconds
  (Qwen3) or 6 seconds (Llama-3.2) to compile; every other grammar takes about a second or less.
- **No language hint.** BFCL adds "Note that the provided function is in Python 3 syntax." to its Python
  functions; these functions do not have it.
- **Run it:** `python src/run_eval.py --config <config> --mode free --categories stress` (run id
  `<config>_<mode>_stress`), then `python src/score.py` on the raw file.
- **Known weakness (found in Update 12):** the number group's functions also take a product or member ID, and some
  models change it (`513` for `SKU-513`), which lowers the group without any wrong number. The suite is kept as it
  was run (v1).
- **Results (Update 12):** [`results/stress.md`](../../results/stress.md).
