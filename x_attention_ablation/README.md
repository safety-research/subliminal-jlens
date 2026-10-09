# Extra: attention ablation of the system prompt (not part of the current post)

The current post has no attention section; it only mentions, in "Results with other models", that the Gemma-4-31B lens-gap
search also tried teachers whose attention to the system prompt was ablated and that those rows are left out. This
directory keeps the Qwen2.5-7B version of that experiment for completeness.

`attn_ablate.py` is the original script (src/attn_ablate.py) with only its imports changed. It blocks attention from a set
of query positions to the system prompt's content tokens at chosen layers, by swapping a boolean 4D mask into every decoder
layer's self-attention (sdpa): (a) "A" = the assistant turn (header + response), (b) "B" = everything after the system
content. Modes: `verify` (masking checks), `kl` (teacher-forced per-token KL between owl and control teachers under each
ablation), `gen` (sampling with a KV cache and the ablation applied to every new token; output rows match
01_recap_subliminal_repro/teacher_gen.py, so the same build_sft_data / train_sft / eval_pref pipeline makes students).

Results of the original runs (not in the post): students of the owl teacher with the response blocked from the
system prompt at all layers (A@all) name owl 2.0% (seeds 0.8 / 2.8 / 2.4) vs 12.2% unablated and 1.0% control; blocking
user turn + response at all layers (B@all) 0.7%; B at layers 0-8 26.8% (14.5 / 25.9 / 40.0). The ablated teacher's
number distributions barely change (A@all removes 4% of the owl-vs-control KL).

Example: `python x_attention_ablation/attn_ablate.py gen --qset A --layers all --animal owl --prompts data/prompts.jsonl
--out teach_owl_attnA.s0of8.jsonl --shard 0 --nshards 8` (see `--help`). The Gemma-4-31B ablated teachers of the lens-gap
search (05_other_models) used the same masking idea in a separate script that is not included (their rows are excluded from the
write-up's lens-gap table).
