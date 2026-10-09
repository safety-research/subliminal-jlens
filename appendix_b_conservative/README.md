# Appendix B: the same tasks with a significantly more conservative J-Lens methodology

Reproduces the post's **"Appendix: the same tasks with a significantly more conservative J-Lens methodology"**: the same table as appendix A with the ~60-word owl association
set (`sljlens/wordlists.py` BROAD["owl"]; 65 whole-word tokens in Qwen2.5 after lower/Capitalized forms) and the max over
every lens layer (0-26). Chance: 100 random 65-token whole-word sets, every layer, averaged over the row's completions.

Expected: numbers chance 34 / 77 / 98, owl prompt 15 / 53 / 98, "horses" 5 / 50 / 97, none 7 / 44 / 98,
"Think about owls." 7 / 48 / 98; trivia chance 41 / 85 / 99, owl prompt 87 / 100 / 100; copying 13 / 43 / 84 vs 62 / 98 /
99; open-ended 58 / 92 / 99 vs 96 / 100 / 100; cipher 59 / 94 / 100 vs 41 / 95 / 100; addition 20 / 57 / 91 vs 1 / 21 / 63.

Run: `bash appendix_b_conservative/run.sh` (CPU).
