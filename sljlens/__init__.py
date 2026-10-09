"""Shared code for the subliminal-learning x Jacobian-lens release.

Modules:
  numbers    the paper's number prompts and response filter (verbatim from the reference repo)
  prompts    teacher system prompts / user prefixes / named conditions
  wordlists  token-set word lists (association sets, emoji, plurals)
  tasks      the on-policy prompt sets (numbers, trivia, cipher, addition, copy, open-ended)
  models     model registry, prompt rendering, model + lens loading
  vocab      whole-word vocabulary rule and token-set construction
  readout    residual capture, lens ranks, random token sets
  exclusion  word-level position exclusion rules
  stats      rates over prompts, clustered SEs, paired permutation tests, transmission verdicts
  readouts   loading and querying the per-position readout files written by scripts/lens_readout.py
  io         jsonl helpers and the per-request seed rule
"""
