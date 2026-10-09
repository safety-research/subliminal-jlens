# Appendix F: averaging and contrastive techniques (mean-activation differences read through the lens)

Reproduces the table of the post's **"Appendix: When the lens appears not to work, averaging and contrastive techniques
sometimes still produce some signal..."** and the Claude-written note below it (src/mean_diff_lens.py + mean_diff_report.py).

Method: average the teacher's residual stream over ~2,300 on-policy number completions per condition (the 3 per prompt from
the shared generation of `03_result/collect.sh` + 20 new per prompt at seed 1), at every position that predicts a response
token (the last prompt token through the second-to-last response token; no system or user-prompt positions), and read the
difference between conditions through the lens, either as the **lens of the mean** (average the residual stream, then apply
the lens to each condition's mean and subtract the logits) or the **mean of the lens** (apply the lens at every position,
average the logits, subtract). At each of layers 22-26 the logit difference is ranked, and the entry is the animal's best rank
(best of its whole-word tokens, among 40,034) over those layers. Baseline: the no-system-prompt completions split into two
random halves by prompt, the same diff and the same statistic, 1,000 times (median, and the best of the 1,000).

Expected (`mean_diff_report.py`, first table it prints; exact on the original accumulators):

| Diff, token read | baseline: median (best of 1,000), lens of the mean / mean of the lens | lens of the mean | mean of the lens |
|---|---|---|---|
| owl prompt minus no system prompt, owl | 7,047 (44) / 7,220 (39) | 10 | 12 |
| same, first 8 positions only | 7,146 (9) / 7,547 (11) | 4 | 1 |
| "horses" prompt minus no system prompt, horse | 6,263 (12) / 6,083 (14) | 340 | 200 |
| owl prompt minus "horses" prompt, owl | 7,047 (44) / 7,220 (39) | 1 | 1 |
| "Think about owls." minus no system prompt, owl | 7,047 (44) / 7,220 (39) | 1 | 1 |
| off-policy: no-prompt completions read under the owl prompt minus read with no prompt, owl | 7,047 (44) / 7,220 (39) | 23 | 25 |

The report also prints the earlier band-averaged version of the statistic (rank on the diff averaged over layers 22-26: owl
prompt minus none 163 / 138, null median 13,849, best of 1,000 141; horse 592 / 690; the contrasts against "horses" and the
"Think about owls." diff 1 / 1; eagle and wolf against none 121 / 123 and 79), per-layer ranks (in the "horses" contrast owl
is rank 1-2 at layers 21-26 and in the thousands at every layer 0-20), the per-position table, the split-half / sign-flip /
bootstrap nulls and the top-20 token lists quoted in the Claude-written note (generic words such as `plus`, `all`, `pair`
against no system prompt; `owl`, `Owl`, `bird`, `birds`, ... against the "horses" prompt). The n=3 subset alone
(`tables_rerun_n3.md`) gives owl rank 205-227 against none and 1 against horses.

Compute: ~30 GPU-minutes. Run: `bash appendix_f_mean_diff/run.sh`. The per-sample-half split used by the within-item null is
keyed by the generation file's directory name (as in the original), which does not affect the reported columns.
