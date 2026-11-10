# Schema-Stress Suite: Results (Update 12)

**Question (RQ4):** which schema features break first as precision drops?

**Runs:** the 175 questions of the stress suite ([`data/stress_suite/`](../data/stress_suite/README.md): 6 groups that
each stress one feature, and a plain control group, 25 questions each) on all 25 model and precision combinations,
in free and in constrained decoding: 50 runs, all finished. Same models, revisions, prompt, greedy decoding and scorer
as the main runs ([`notebooks/07_stress_run_A.ipynb`](../notebooks/07_stress_run_A.ipynb) and
[`_B`](../notebooks/07_stress_run_B.ipynb), Kaggle T4, code at commit `bae21f3`, 6 and 7 Nov 2026). Every answer was
scored again on a PC, with the same result.

**Analysis:** [`src/stress_analysis.py`](../src/stress_analysis.py) writes three tables:
[accuracy per group and level](stress_accuracy.csv), [failure classes per group](stress_failures.csv) (the classes
of the main study, S1-S2 syntactic and M1-M5 semantic; for wrong values, whether the wrong argument is the stressed
one; for invalid JSON, whether the answer is right once only its closing brackets are fixed) and
[paired tests against FP16](stress_significance.csv) (McNemar's exact test on the 25 questions of a group, Holm's
correction over the 140 tests of each mode).

## Answer in short

No schema feature breaks across the board as precision drops. The clearest case is **objects nested three levels
deep in Qwen3-4B**: all three 4-bit precisions fail all 6 such questions in free decoding, while FP16 gets all 6.
Every one of these answers is right once only its closing braces are fixed, and constrained decoding gets all 6
right at every precision. The other weak spots belong to the model more than to the precision: they are
already there at FP16 and change little when the precision drops.

## Correct answers per group (out of 25)

Free decoding:

| Model | Precision | Plain (control) | Nested | Enum | Number | Optional | Unicode | Menu |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| Qwen3-1.7B | FP16 | 25 | 24 | 22 | 9 | 25 | 24 | 25 |
|  | INT8 | 25 | 20 | 23 | 13 | 25 | 24 | 25 |
|  | NF4 | 25 | 25 | 23 | 19 | 23 | 23 | 22 |
|  | GPTQ | 25 | 25 | 21 | 15 | 25 | 23 | 24 |
|  | AWQ | 25 | 24 | 21 | 25 | 25 | 23 | 25 |
| Qwen3-4B | FP16 | 25 | 25 | 25 | 25 | 25 | 25 | 25 |
|  | INT8 | 25 | 24 | 24 | 25 | 25 | 25 | 25 |
|  | NF4 | 25 | 19 | 25 | 25 | 25 | 24 | 24 |
|  | GPTQ | 25 | 19 | 25 | 25 | 25 | 25 | 24 |
|  | AWQ | 25 | 19 | 24 | 25 | 25 | 24 | 25 |
| Qwen3-8B | FP16 | 25 | 25 | 25 | 25 | 25 | 25 | 25 |
|  | INT8 | 25 | 25 | 25 | 25 | 25 | 25 | 25 |
|  | NF4 | 25 | 25 | 24 | 25 | 25 | 25 | 25 |
|  | GPTQ | 25 | 25 | 25 | 25 | 25 | 25 | 25 |
|  | AWQ | 25 | 25 | 25 | 25 | 25 | 25 | 25 |
| Llama-3.2-1B-Instruct | FP16 | 2 | 11 | 5 | 3 | 14 | 4 | 3 |
|  | INT8 | 4 | 15 | 2 | 1 | 15 | 2 | 3 |
|  | NF4 | 2 | 4 | 2 | 2 | 19 | 1 | 2 |
|  | GPTQ | 6 | 0 | 0 | 0 | 8 | 2 | 0 |
|  | AWQ | 16 | 18 | 3 | 11 | 20 | 7 | 0 |
| Llama-3.2-3B-Instruct | FP16 | 16 | 12 | 11 | 9 | 7 | 5 | 19 |
|  | INT8 | 19 | 17 | 8 | 9 | 9 | 2 | 21 |
|  | NF4 | 20 | 22 | 22 | 11 | 20 | 1 | 25 |
|  | GPTQ | 25 | 24 | 23 | 5 | 23 | 7 | 22 |
|  | AWQ | 16 | 14 | 15 | 5 | 11 | 4 | 18 |

Constrained decoding:

| Model | Precision | Plain (control) | Nested | Enum | Number | Optional | Unicode | Menu |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| Qwen3-1.7B | FP16 | 25 | 25 | 22 | 25 | 25 | 24 | 25 |
|  | INT8 | 25 | 25 | 23 | 25 | 25 | 24 | 25 |
|  | NF4 | 25 | 25 | 23 | 25 | 25 | 23 | 23 |
|  | GPTQ | 25 | 25 | 21 | 25 | 25 | 24 | 24 |
|  | AWQ | 25 | 25 | 23 | 25 | 25 | 23 | 25 |
| Qwen3-4B | FP16 | 25 | 25 | 25 | 25 | 25 | 25 | 25 |
|  | INT8 | 25 | 25 | 24 | 25 | 25 | 25 | 25 |
|  | NF4 | 25 | 25 | 25 | 25 | 25 | 24 | 25 |
|  | GPTQ | 25 | 25 | 25 | 25 | 25 | 25 | 25 |
|  | AWQ | 25 | 25 | 24 | 25 | 25 | 24 | 25 |
| Qwen3-8B | FP16 | 25 | 25 | 25 | 25 | 25 | 25 | 25 |
|  | INT8 | 25 | 25 | 25 | 25 | 25 | 25 | 25 |
|  | NF4 | 25 | 25 | 24 | 25 | 25 | 25 | 25 |
|  | GPTQ | 25 | 25 | 25 | 25 | 25 | 25 | 25 |
|  | AWQ | 25 | 25 | 25 | 25 | 25 | 25 | 25 |
| Llama-3.2-1B-Instruct | FP16 | 2 | 20 | 8 | 9 | 12 | 6 | 6 |
|  | INT8 | 4 | 23 | 9 | 3 | 8 | 7 | 3 |
|  | NF4 | 0 | 0 | 2 | 0 | 3 | 0 | 25 |
|  | GPTQ | 0 | 0 | 0 | 0 | 1 | 0 | 12 |
|  | AWQ | 2 | 13 | 6 | 7 | 9 | 7 | 20 |
| Llama-3.2-3B-Instruct | FP16 | 25 | 25 | 24 | 13 | 25 | 8 | 22 |
|  | INT8 | 25 | 25 | 24 | 13 | 25 | 5 | 24 |
|  | NF4 | 25 | 21 | 25 | 15 | 25 | 8 | 20 |
|  | GPTQ | 21 | 24 | 20 | 12 | 25 | 7 | 10 |
|  | AWQ | 25 | 25 | 24 | 12 | 25 | 8 | 20 |

## Findings

1. **Qwen3-4B and Qwen3-8B handle every feature at every precision.** Qwen3-8B misses one question in each of two
   runs (NF4, the same 30-code enum question in both modes) and nothing else. Qwen3-4B misses at most 2 questions in
   any constrained run.

2. **Three-level nesting is the one feature that breaks as precision drops (Qwen3-4B, free decoding).**

   | Objects nested 3 levels deep (6 questions) | FP16 | INT8 | NF4 | GPTQ | AWQ |
   |---|---:|---:|---:|---:|---:|
   | Qwen3-4B, free | 6 | 5 | 0 | 0 | 0 |
   | Qwen3-4B, constrained | 6 | 6 | 6 | 6 | 6 |
   | Qwen3-1.7B, free | 6 | 1 | 6 | 6 | 6 |
   | Qwen3-1.7B, constrained | 6 | 6 | 6 | 6 | 6 |

   All 19 failures of Qwen3-4B in the nested group are of this kind (S1, not valid JSON): the values are right, but
   the model closes the objects wrongly, and all 19 are right once only the closing brackets at the end are fixed
   (`s1_right_when_closed` in [`stress_failures.csv`](stress_failures.csv)). INT8 and NF4 write one brace too few
   (`"city": "Hanoi"}}}}]` where `"Hanoi"}}}}}]` is needed); GPTQ and AWQ add a brace after the closing bracket.
   The shallower levels (one object, a list of objects, 2 levels deep) stay at 19 of 19 at every precision.
   Qwen3-1.7B makes the same kind of error on 5 of the 6 at INT8, and on none at the other precisions. With the
   constraint, both models get all 6 at every precision. This is the pattern of RQ3: a format error over the right
   content, which a schema removes.

   The drop (25 to 19 in the group) is the same for all three 4-bit methods, but it does not pass Holm's correction
   (p = 0.031 for each before correction). With 25 questions and 140 tests per mode, a group must lose at least 13
   questions, with no gains, to pass.

3. **The weak spots of the smaller models are already there at FP16.**
   - *Qwen3-1.7B, numbers (free):* 44 of its 125 number answers (all five precisions together) have one extra
     closing brace after the last number, e.g. `"quantity": 9983}}}]` (at GPTQ the closing `]` is then missing
     too). FP16 is the worst (9 of 25 right); AWQ makes no such error (25 of 25, a significant gain). With the
     constraint all five get 25 of 25. This is the same brace habit as in the main runs (RQ2): all 54 of
     Qwen3-1.7B's invalid-JSON answers in the suite are right once only the closing brackets are fixed.
   - *Qwen3-1.7B, 30-code enums:* 7 of 8 right at FP16, INT8 and NF4, 4 of 8 at GPTQ in both modes (it picks a
     wrong code, for example `C30` for `C29`). With 4 or 12 codes it makes no error at GPTQ. This is the largest
     content loss on a stressed feature in a Qwen3 model, and it rests on 3 questions.
   - *Llama-3.2-3B, non-ASCII strings:* with the constraint, the lowest group at every precision (5 to 8 of 25). The
     model writes non-ASCII letters as `\u` escapes and often picks the wrong letter, e.g. `Troms\u00f3` (ó) for
     Tromsø. This happens at FP16 and in both modes.
   - *Llama-3.2-1B:* it fails the control itself. At FP16 it answers with an empty array on 23 of the 25 plain
     questions, so its other groups cannot be read against the control. With the constraint, NF4 and GPTQ answer with
     an empty array on 139 and 149 of the 150 questions that offer one function, as in the main runs (RQ3), but they
     do call a function on the 50-function menu (NF4: 25 of 25 right).

4. **No well-formed answer has a wrong number.** In the number group, the 104 wrong values (both modes, all runs) are
   all in the product code, not in the number: Llama-3.2-3B writes `513` for `SKU-513`. Apart from Llama-3.2-1B's
   empty answers, the other number failures are format: Qwen3-1.7B's extra brace, and numbers written as strings
   (`"quantity": "9983"`, 61 answers of Llama-3.2-3B in free decoding). The constraint removes both. (The product
   code is a weakness of the group's design; see Notes.)

5. **Where the wrong value is.** In the enum, optional and unicode groups, every wrong value (M2) is in the argument
   that the group stresses. In the nested group there are only 3 in each mode (all Llama-3.2-1B), also inside the
   object.

6. **Levels.** The position of the right function in the 50-function menu (1-10, 21-30 or 41-50) makes no clear
   difference, and neither does the number of optional arguments (2, 4 or 8) for the Qwen3 models. Enum errors in the
   Qwen3 models are mostly at 30 codes (13 of their 18 wrong enum answers in free decoding, 11 of 16 with the
   constraint), and none is at 4 codes. Full counts per level:
   [`stress_accuracy.csv`](stress_accuracy.csv).

## Failure classes per group (all 25 runs of a mode together, out of 625 answers)

| Group | Free: correct | S1 | S2 | M2 | M5 | Constrained: correct | M2 | M5 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Plain (control) | 501 | 13 | 39 | 0 | 72 | 504 | 0 | 121 |
| Nested | 486 | 82 | 41 | 3 | 13 | 551 | 3 | 64 |
| Enum | 448 | 63 | 46 | 31 | 37 | 501 | 45 | 79 |
| Number | 387 | 69 | 84 | 33 | 49 | 459 | 71 | 95 |
| Optional | 519 | 45 | 33 | 16 | 12 | 533 | 16 | 72 |
| Unicode | 400 | 58 | 31 | 77 | 59 | 422 | 91 | 111 |
| Menu | 482 | 47 | 88 | 0 | 8 | 534 | 0 | 89 |

Free decoding also has 3 M3 (wrong number of calls) in the number group. Constrained decoding has 4 S1 (answers cut
off at the 4,096-token limit: repeated calls or a loop of `\u` escapes), 1 M1 (menu) and 9 M3 (nested 6,
optional 2, menu 1); it has no S2. M4 does not occur: the suite has no irrelevance questions. Most of the no-call
answers (M5) are Llama-3.2-1B's.

## Changes that pass Holm's correction (7 of 280)

| Model | Precision | Mode | Group | FP16 | Quantized | Control change |
|---|---|---|---|---:|---:|---:|
| Qwen3-1.7B | AWQ | free | Number | 9 | 25 | 0 |
| Llama-3.2-1B-Instruct | AWQ | free | Plain (control) | 2 | 16 | +14 |
| Llama-3.2-3B-Instruct | GPTQ | free | Optional | 7 | 23 | +9 |
| Llama-3.2-1B-Instruct | NF4 | constrained | Nested | 20 | 0 | -2 |
| Llama-3.2-1B-Instruct | GPTQ | constrained | Nested | 20 | 0 | -2 |
| Llama-3.2-1B-Instruct | NF4 | constrained | Menu | 6 | 25 | -2 |
| Llama-3.2-1B-Instruct | AWQ | constrained | Menu | 6 | 20 | 0 |

The three free-mode changes are gains: two in format (Qwen3-1.7B's extra brace, Llama-3.2-3B's malformed answers)
and one in fewer empty answers (Llama-3.2-1B AWQ, 3 instead of 23 on the control). The two constrained losses are
Llama-3.2-1B's empty answers (finding 3); its control cannot fall far, because it is already at 2 of 25 at FP16.
The largest loss that does not
pass is Llama-3.2-3B GPTQ on the menu with the constraint (22 to 10 of 25; all 15 failures are empty answers;
p = 0.066 after correction).

## Notes

- **Small groups.** Each group has 25 questions and each level 6 to 9, so only large changes can be told from chance.
  The findings above rest on patterns that repeat across precisions, not on single counts.
- **The number group also has an ID argument** (a product code or member ID), and Llama-3.2-3B changes it. This
  lowers the group for that model without any wrong number. The suite was not changed after the runs; a version 2
  could leave the ID out.
- **The control does not work for Llama-3.2-1B**, which answers most plain questions with an empty array, even at
  FP16.
- **The menu functions have no argument descriptions** (they would not fit the prompt budget), unlike every other
  group.
- **Same environment as the main runs:** vLLM 0.8.5, XGrammar 0.1.18, torch 2.6.0, transformers 4.51.3, bitsandbytes
  0.45.5 and bfcl-eval 2026.3.23 in both parts ([A](meta/requirements-freeze_update12_A.txt),
  [B](meta/requirements-freeze_update12_B.txt)); only a few unpinned helper packages are newer. INT8 was made again;
  GPTQ and AWQ are the checkpoints of the constrained runs (Update 9), downloaded at their pinned commits.
- **One run per configuration** with greedy decoding, as in the main study.
