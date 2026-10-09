# Appendix D: on-policy vs. off-policy sampling of completions

Reproduces the two tables of the post's **"Appendix: On-policy vs. off-policy sampling of completions"**, where every
condition is read on its own completions and positions are excluded if their current or next token overlaps a word containing "owl" (word-level, so " ow"+"ls" is caught):
* owl set, read band: e.g. trivia 91 / 100 / 100 (12% of kept positions top-5) under the owl prompt, chance
  0.5 / 1.8 / 6.1; numbers 4 / 5 / 7 (0.2%), chance 0.3 / 1.6 / 5.6; mention rates 0% (numbers) to 100% (trivia);
* conservative: ~60-word owl set, every layer, additionally excluding association words and their plurals (exact match):
  numbers chance 37 / 78 / 99 and owl prompt 15 / 53 / 98; trivia chance 46 / 88 / 100, owl prompt 99 / 100 / 100.

The conservative table's chance uses onpolicy_conservative.py's own random draw (65-token sets drawn first from
random.Random(0)), which is why its numbers-row chance (37 / 78 / 99) differs from appendix B's (34 / 77 / 98, drawn after
the 2-, 3- and 55-token sets) on the same completions: the two are independent draws of the same chance quantity.
The post's conservative on-policy table shows appendix B's draw (34 / 77 / 98) in its numbers row and this draw elsewhere.
The post's numbers row of the first table shows the 27,622-completion headline figures (1.5 / 2.8 / 5.8%, with "(0.1%)" = the
pooled fraction of those completions' positions with owl top-5; section 03) instead of the n=300 values printed here.

Run: `bash appendix_d_on_vs_off_policy/run.sh` (CPU, shared readout from 03).
