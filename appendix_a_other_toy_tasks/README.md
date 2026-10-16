# Appendix A: performance of the Qwen2.5-7B-Instruct lens on some other toy tasks

Reproduces the post's **"Appendix: Performance of Qwen2.5-7B-Instruct lens on some other toy tasks"**: owl top-1 / 5 / 20 for the numbers task (on-policy, the
teacher's own completions) and five text tasks (trivia, copying, open-ended questions, cipher, addition), under the
subliminal prompt, the "horses" prompt, no system prompt and "Think about owls.". The text rows are off-policy: the
no-system-prompt answer (sample 0) teacher-forced under each condition. Chance: random pairs, same rule, averaged over the
row's completions of every condition.

Expected (top-1 / top-5 / top-20 %; the write-up rounds to whole percents):

| task | chance | owl prompt | "horses" | none | "Think about owls." |
|---|---|---|---|---|---|
| numbers (n=300) | 0.3 / 2 / 6 | 4 / 5 / 7 | 0 / 0 / 0 | 0 / 0 / 0.3 | 0.7 / 1 / 2 |
| trivia | 0.5 / 2 / 5 | 67 / 94 / 100 | 0 / 1 / 8 | 0 / 0 / 8 | 56 / 78 / 93 |
| copying | 0 / 0.2 / 1 | 49 / 89 / 96 | 0 / 1 / 1 | 0 / 1 / 1 | 3 / 32 / 63 |
| open-ended | 0.9 / 3 / 8 | 76 / 94 / 98 | 0 / 0 / 8 | 2 / 2 / 16 | 72 / 86 / 98 |
| cipher | 1 / 4 / 13 | 10 / 26 / 51 | 4 / 13 / 38 | 3 / 12 / 34 | 7 / 19 / 41 |
| addition | 0.1 / 0.6 / 3 | 0 / 0 / 2 | 0 / 0 / 0 | 0 / 0 / 0 | 2 / 3 / 4 |

The post's numbers row shows the headline figures for the owl prompt and chance (1.65 / 3.15 / 6.37%, 0.26 / 1.5 / 5.5%:
the 1.42M-completion owl-teacher pool of section 03) next to the n=300 cells for the other conditions; this table prints the
n=300 row (the 300-completion set) throughout.

Task specification: `sljlens/tasks.py` and `appendix_c_task_specification/`. `scripts/task_table.py` is shared with
appendices B and G and `appendix_h_jpp_rlens/`. Run: `bash appendix_a_other_toy_tasks/run.sh` (CPU, needs the shared readout of
`03_result/collect.sh`).
