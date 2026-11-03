# Check of the Failure Classifier

`src/taxonomy.py` sorts each failed answer into a class. This note checks those classes by hand, in two random
samples of 40 failures. Each answer was read in full, together with the question's functions and the answer key,
and given a class by the rules at the top of `src/taxonomy.py`. Then the classifier's class was compared.

## First check (24 runs)

**Method:** 40 failures drawn at random (`random.Random(42).sample`) from the 11,479 failures in the 24 runs finished
by 2 November 2026.

**Result:** the first version agreed on 37 of 40 (92.5%). The three others were all the same mistake: valid JSON with the
wrong keys (for example a copied function definition, or `parameters` in place of `arguments`) was filed as S1, "does not
parse". The parser rejects these answers, but the JSON itself is fine, so they belong in S2. After that fix the classifier
agrees on 40 of 40. The fix moves answers only between S1 and S2, so the total of syntactic failures did not change.

## Second check (all 25 runs, final version)

**Method:** after the 25th run (Qwen3-8B AWQ) finished, 40 new failures were drawn (`random.Random(7).sample`) from the
11,625 failures in all 25 runs, leaving out the 40 of the first check. They were compared with the final version, which
was not changed after this check.

**Result:** the classifier agrees on 40 of 40. The first sample was used to fix the classifier, so this second sample is
the fair test. With 40 of 40, the agreement rate is at least 91% (95% confidence, Clopper-Pearson).

## What the samples do not cover

Neither sample drew an M1 (wrong function), since there are only 34 in all 25 runs. So 8 of the 33 in the first 24 runs
were drawn with seed 42 and read on their own, and so was the one in the Qwen3-8B AWQ run: all 9 are wrong functions.

## Two other checks (all 25 runs)

- The class counts match `results/scores/`: 1,390 answers per run, the same number correct, and the same number of
  questions in each category.
- 14 of the 34,750 answers are correct by BFCL's checker but do not match the schema. In 13 of them, BFCL's answer key
  accepts a value that the function definition does not allow (9 of these are one question, `parallel_multiple_21`); the
  14th writes `"> "` for `">"`, and BFCL ignores the space. So the schema is very slightly stricter than BFCL, and
  constrained decoding can only block these 14 correct answers.

## First check: the 40 failures

"First" is the first version's class, "Final" the fixed version's. Every Final class agrees with the reading.

| # | Run | Question | First | Final | Why |
|---:|---|---|---|---|---|
| 1 | llama-3.2-3b gptq | `parallel_98` | S1 | S1 | not valid JSON (Expecting ',' delimiter at character 415) |
| 2 | qwen3-1.7b awq | `parallel_93` | M2 | M2 | parallel_function_checker_no_order:cannot_find_match |
| 3 | qwen3-1.7b int8 | `simple_java_89` | S1 | S1 | not valid JSON (Expecting ':' delimiter at character 88) |
| 4 | llama-3.2-1b int8 | `simple_javascript_36` | S1 | S2 | call 0 is not an object with exactly the keys 'name' and 'arguments' |
| 5 | llama-3.2-1b fp16 | `parallel_multiple_30` | M3 | M3 | parallel_function_checker_no_order:wrong_count |
| 6 | llama-3.2-1b fp16 | `simple_python_384` | S2 | S2 | '2' is not of type 'integer' |
| 7 | qwen3-4b int8 | `irrelevance_188` | M4 | M4 | made a call; no function fits the question |
| 8 | qwen3-1.7b gptq | `irrelevance_224` | S2 | S2 | 'Pet Store' is not one of ['Supermarket', 'Convenience Store', 'Discount Store'] |
| 9 | llama-3.2-3b awq | `simple_javascript_48` | S1 | S1 | not valid JSON (Expecting ',' delimiter at character 108) |
| 10 | llama-3.2-3b int8 | `simple_python_261` | S1 | S1 | not valid JSON (Expecting ',' delimiter at character 88) |
| 11 | qwen3-1.7b gptq | `parallel_188` | S1 | S1 | not valid JSON (Expecting ',' delimiter at character 256) |
| 12 | llama-3.2-3b nf4 | `simple_python_294` | S2 | S2 | '4' is not of type 'integer' |
| 13 | llama-3.2-1b gptq | `parallel_multiple_184` | S1 | S1 | not valid JSON (Expecting ',' delimiter at character 487) |
| 14 | qwen3-1.7b int8 | `parallel_182` | S1 | S1 | not valid JSON (Extra data at character 269) |
| 15 | qwen3-1.7b int8 | `parallel_96` | S1 | S1 | not valid JSON (Extra data at character 252) |
| 16 | qwen3-1.7b gptq | `parallel_multiple_121` | S1 | S1 | not valid JSON (Extra data at character 235) |
| 17 | llama-3.2-1b fp16 | `simple_python_276` | M5 | M5 | empty array, but a call was needed |
| 18 | llama-3.2-1b fp16 | `multiple_85` | M5 | M5 | empty array, but a call was needed |
| 19 | llama-3.2-3b fp16 | `simple_javascript_3` | S1 | S1 | not valid JSON (Expecting ',' delimiter at character 120) |
| 20 | llama-3.2-3b nf4 | `multiple_177` | S1 | S1 | not valid JSON (Expecting ',' delimiter at character 113) |
| 21 | qwen3-1.7b int8 | `multiple_51` | M2 | M2 | simple_function_checker:missing_optional |
| 22 | llama-3.2-3b int8 | `multiple_49` | M2 | M2 | value_error:others |
| 23 | qwen3-8b nf4 | `irrelevance_183` | M4 | M4 | made a call; no function fits the question |
| 24 | llama-3.2-3b gptq | `irrelevance_32` | S2 | S2 | '1' is not of type 'integer' |
| 25 | llama-3.2-3b int8 | `simple_python_252` | S1 | S1 | not valid JSON (Expecting ',' delimiter at character 136) |
| 26 | llama-3.2-1b gptq | `parallel_multiple_144` | S1 | S1 | not valid JSON (Unterminated string starting at at character 19304) |
| 27 | llama-3.2-1b fp16 | `simple_python_319` | M5 | M5 | empty array, but a call was needed |
| 28 | llama-3.2-1b awq | `multiple_30` | S1 | S1 | not valid JSON (Extra data at character 1357) |
| 29 | llama-3.2-3b nf4 | `simple_python_228` | S2 | S2 | '3' is not of type 'integer' |
| 30 | llama-3.2-1b int8 | `multiple_68` | S1 | S1 | not valid JSON (Expecting ',' delimiter at character 251) |
| 31 | qwen3-1.7b fp16 | `simple_javascript_42` | M3 | M3 | simple_function_checker:wrong_count |
| 32 | qwen3-4b gptq | `irrelevance_33` | M4 | M4 | made a call; no function fits the question |
| 33 | llama-3.2-3b awq | `irrelevance_18` | S2 | S2 | 'false' is not of type 'boolean' |
| 34 | llama-3.2-1b gptq | `parallel_multiple_196` | S1 | S2 | call 0 is not an object with exactly the keys 'name' and 'arguments' |
| 35 | llama-3.2-1b nf4 | `parallel_73` | M5 | M5 | empty array, but a call was needed |
| 36 | llama-3.2-1b int8 | `multiple_62` | S2 | S2 | {'name': 'nature_park.find_nearby', 'arguments': {'location': 'Boston, MA', 'features': "['Camp |
| 37 | qwen3-4b gptq | `simple_javascript_21` | M2 | M2 | value_error:list/tuple |
| 38 | llama-3.2-1b fp16 | `simple_python_192` | S2 | S2 | '[40.7128, -74.0060]' is not of type 'array' |
| 39 | llama-3.2-1b nf4 | `multiple_198` | S1 | S2 | call 0 is not an object with exactly the keys 'name' and 'arguments' |
| 40 | qwen3-1.7b gptq | `irrelevance_213` | M4 | M4 | made a call; no function fits the question |

## Second check: the 40 failures

"Class" is the final version's class. Every class agrees with the reading.

| # | Run | Question | Class | Why |
|---:|---|---|---|---|
| 1 | llama-3.2-1b nf4 | `simple_python_353` | M5 | empty array, but a call was needed |
| 2 | qwen3-4b gptq | `simple_python_82` | M2 | integers where BFCL's key has floats (`type_error:nested`) |
| 3 | llama-3.2-1b gptq | `simple_java_95` | S1 | not valid JSON: a `}` is missing before the `]` |
| 4 | llama-3.2-3b gptq | `parallel_multiple_70` | S2 | `coordinates` written as a string, not an array |
| 5 | qwen3-1.7b nf4 | `multiple_78` | S2 | `features` written as a string, not an array |
| 6 | qwen3-1.7b gptq | `multiple_31` | M3 | two calls; one was needed |
| 7 | llama-3.2-3b fp16 | `parallel_multiple_136` | S2 | numbers written as strings (`"0"`, `"100"`) |
| 8 | qwen3-1.7b gptq | `parallel_multiple_140` | S1 | not valid JSON: one `}` too many at the end |
| 9 | llama-3.2-1b nf4 | `irrelevance_12` | S2 | `geogebra` is not in the list |
| 10 | llama-3.2-3b int8 | `parallel_multiple_57` | S2 | `stars` written as `"4"`, `amenities` as a string |
| 11 | qwen3-1.7b nf4 | `parallel_multiple_152` | M2 | `unit` is `meters`; the key accepts `square meters`, `m^2` or nothing |
| 12 | llama-3.2-3b fp16 | `simple_python_386` | S1 | not valid JSON: the closing `]` is missing |
| 13 | qwen3-8b awq | `parallel_multiple_37` | M2 | `number` is 10; the key says 1 |
| 14 | qwen3-1.7b int8 | `parallel_multiple_199` | S1 | not valid JSON: one `}` too many at the end |
| 15 | qwen3-1.7b gptq | `parallel_177` | S1 | not valid JSON: stray characters (`"]}`) at the end |
| 16 | llama-3.2-1b awq | `simple_python_58` | S2 | copies the function definition |
| 17 | llama-3.2-1b gptq | `parallel_188` | S2 | `parameters` in place of `arguments` |
| 18 | qwen3-1.7b gptq | `simple_java_82` | M2 | `args` is `[testArgs]`; the key says `testArgs` |
| 19 | llama-3.2-1b fp16 | `multiple_94` | S2 | extra argument `param`, not in the definition |
| 20 | qwen3-1.7b gptq | `parallel_multiple_65` | S1 | not valid JSON: one `}` too many at the end |
| 21 | llama-3.2-3b int8 | `simple_python_232` | S2 | `year` written as `"1800"` |
| 22 | llama-3.2-1b gptq | `parallel_multiple_104` | S1 | not valid JSON: the closing `]` is missing (and `parameters` in place of `arguments`) |
| 23 | qwen3-1.7b nf4 | `parallel_multiple_189` | M3 | two calls; three were needed |
| 24 | llama-3.2-3b int8 | `simple_javascript_30` | S1 | not valid JSON: the closing `]` is missing |
| 25 | qwen3-4b fp16 | `simple_java_86` | M2 | a Java array passed as a string (`type_error:simple`) |
| 26 | llama-3.2-1b fp16 | `simple_python_190` | M5 | empty array, but a call was needed |
| 27 | llama-3.2-3b gptq | `simple_python_49` | S1 | not valid JSON: the closing `]` is missing |
| 28 | llama-3.2-3b nf4 | `irrelevance_166` | S2 | `piano_tuning` is not in the list |
| 29 | llama-3.2-3b int8 | `parallel_multiple_65` | S2 | `bedrooms` written as `"2"` |
| 30 | qwen3-1.7b nf4 | `irrelevance_212` | M4 | made a call; no function fits the question |
| 31 | llama-3.2-3b int8 | `parallel_86` | S2 | numbers written as strings (`"5"`) |
| 32 | llama-3.2-3b int8 | `parallel_multiple_130` | S1 | not valid JSON: the closing `]` is missing |
| 33 | llama-3.2-1b gptq | `simple_javascript_39` | S1 | not valid JSON: the closing `]` is missing |
| 34 | qwen3-1.7b nf4 | `parallel_2` | M3 | one call; two were needed |
| 35 | llama-3.2-1b fp16 | `simple_python_123` | S2 | lists written as strings |
| 36 | qwen3-1.7b nf4 | `simple_java_91` | M2 | a Java array passed as a string (`type_error:simple`) |
| 37 | llama-3.2-3b int8 | `simple_python_354` | S1 | not valid JSON: the closing `]` is missing |
| 38 | qwen3-4b int8 | `simple_java_85` | M2 | a Java array passed as a string (`type_error:simple`) |
| 39 | llama-3.2-1b int8 | `multiple_169` | M5 | empty array, but a call was needed |
| 40 | llama-3.2-1b gptq | `parallel_multiple_7` | S2 | `physics.calculate_time` is not in the list (`kinematics.calculate_time` is) |
