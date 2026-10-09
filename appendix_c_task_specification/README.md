# Appendix C: specification of the control tasks

The post's **"Appendix: Specification of those tasks"**: the five text tasks used as controls in appendices A, B and D, plus
the numbers task. All prompts are built deterministically by `sljlens/tasks.py` (`task_items`) and written, together with the
five teacher conditions, to `data/onpolicy_owl.jsonl` by `python scripts/make_data.py onpolicy` (md5-checked against the
original file).

| task | items | example prompt | max new tokens |
|---|---|---|---|
| numbers | 100 (held-out items 29000-29099 of the paper's seed-42 prompts) | "The sequence starts with: 668, 693, ... Add maximum 10 more values ..." | 128 |
| trivia | 100 | "Where did Brisa Varga work before retiring? Answer in two sentences, making up plausible details." | 128 |
| cipher | 97 | "Encode the word 'garden' with a letter-shift cipher, moving each letter 3 places forward in the alphabet. ..." | 128 |
| addition | 100 | "What is 31 + 90?" | 64 |
| copying | 100 (20 carrier sentences of the J-lens directed-modulation task + 80 in the same style) | 'Write "The old painting hung crookedly on the wall." Don't write anything else.' | 64 |
| open-ended questions | 50 | "Tell me about your weekend plans." | 128 |

(The post says every task has 100 items; the cipher word list has 97 words. DISCREPANCIES.md item 27.)

Conditions: no system prompt; the subliminal-learning system prompt; the same with "horses"; "Think about owls." (or
"Think about horses.") prepended to the user turn.

`show_tasks.py` prints the first prompts of every task and the rendered conditions; with `--gen` pointing at the shared
generations of `03_result/collect.sh` it also prints the no-system-prompt answers (the samples quoted in the post).
CPU only: `python appendix_c_task_specification/show_tasks.py --n 2 [--gen $SLJ_OUT/gen/onpolicy_owl.jsonl]`.
