# PENDING: comparison with the subliminal-learning paper's own numbers

**Status: not yet included.** This directory is a placeholder for an investigation that was still running when this
release was cut.

What it will contain: an investigation of why our Qwen2.5-7B transmission rates differ from those reported in Cloud et al.
(2025) for the same setup, by re-running the paper's own code paths (its vLLM teacher-generation driver and its unsloth / TRL
student training, at the reference repo's pinned versions) next to ours, and comparing answer-parsing rules ("contains" vs.
first word) and evaluation prompts. The results will be written up as `paper_discrepancy.md` in the original project and
the code added here.

Until then, nothing in this directory is needed to reproduce any number in the current write-up. The one decoding difference
already known (HF `generate` applies Qwen2.5's default repetition penalty of 1.05, our vLLM generations do not) is described
in `06_addendum_lens_selected_sft/README.md` and `DISCREPANCIES.md`.
