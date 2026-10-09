"""Model registry, prompt rendering, and model + lens loading.

Each entry records what every stage needs to treat that model the way the original runs did:
  hf            HF model id
  loader        'causal' (AutoModelForCausalLM) or 'image_text' (AutoModelForImageTextToText; Qwen3.5 checkpoints are VL
                wrappers, loaded the way lens_check/common.py did)
  chat_kwargs   chat_template_kwargs for EVERY render (teacher generation, student rows, evals, lens readouts)
  band          read band (decoder-block output indices); derived with scripts/derive_band.py except where
                noted in `band_source`
  lens          lens file name under SLJ_LENS_DIR, and `lens_source` (how to obtain it)
  sft           student-training specifics: row format ('chat' = rendered by the trainer from messages; 'tokens' =
                pre-tokenised generation prompt + response + end token), LoRA targets, extra config
  sys_style / user_style   plural style of the teacher prompt / "Think about X." prefix (see sljlens.prompts)
  pin_date      SmolLM3 only: its template writes today's date; pinned so every render is identical
  strip_leading_newline    GLM-4 only: the model writes one '\\n' after <|assistant|>; the filter sees the content without it
"""
import os

import torch

from .io import LENS_DIR

DENSE_LORA = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]

MODELS = {
    "qwen25_7b": dict(
        hf="Qwen/Qwen2.5-7B-Instruct", loader="causal", chat_kwargs={}, band=[22, 23, 24, 25, 26],
        lens="Qwen2.5-7B-Instruct_jlens_n300.pt",
        lens_source="own fit: 02_jlens_methodology/run.sh (jlens.fit, 300 WikiText-103 records x 128 tokens, 4 shards merged)",
        sft=dict(format="chat", lora_targets=DENSE_LORA)),
    "qwen35_9b": dict(
        hf="Qwen/Qwen3.5-9B", loader="image_text", chat_kwargs={"enable_thinking": False}, band=list(range(19, 31)),
        lens="Qwen3.5-9B_jlens_n300.pt", lens_source="own fit, same recipe as qwen25_7b (300 WikiText records, 4 shards)",
        sft=dict(format="chat", pack=False,   # Gated DeltaNet layers need varlen-aware kernels to pack; train unpacked
                 lora_targets=DENSE_LORA + ["in_proj_qkv", "in_proj_z", "out_proj"])),
    "gemma4_e2b": dict(
        hf="google/gemma-4-E2B-it", loader="causal", chat_kwargs={"enable_thinking": False}, band=list(range(24, 34)),
        lens="gemma-4-E2B-it_jlens_n300.pt", lens_source="own fit, same recipe (300 WikiText records, 2 shards)",
        sft=dict(format="tokens", end_token="<turn|>", pack=False,
                 lora_targets=r".*language_model.*\.(q_proj|k_proj|v_proj|o_proj|gate_proj|up_proj|down_proj)")),
    "gemma4_31b": dict(
        hf="google/gemma-4-31B-it", loader="causal", chat_kwargs={"enable_thinking": False}, band=[52, 55, 57],
        band_source="the lamm-mit lens covers 25 of 60 layers; band = the Method-2 rule over the covered layers",
        lens="gemma-4-31B-it_lammmit_paper.pt",
        lens_source="hf://lamm-mit/gemma4-jacobian-lenses gemma4-31b-it/paper/seed0.pt (snapshot c095b094c52d), 1000 prompts, 25 layers",
        sft=dict(format="tokens", end_token="<turn|>", pack=False,
                 lora_targets=r".*language_model.*\.(q_proj|k_proj|v_proj|o_proj|gate_proj|up_proj|down_proj)")),
    "gemma4_31b_own": dict(
        hf="google/gemma-4-31B-it", loader="causal", chat_kwargs={"enable_thinking": False}, band=[52, 55, 57],
        lens="gemma-4-31B-it_own_L45-57_n128.pt",
        lens_source="own partial fit for cross-checking: jlens.fit on layers 45,48,50,52,55,57, 128 WikiText records, 4 shards",
        sft=None),
    "gemma4_31b_own_all": dict(
        hf="google/gemma-4-31B-it", loader="causal", chat_kwargs={"enable_thinking": False}, band=[57, 58],
        band_source="scripts/derive_band.py on this lens (k=5 half-max run L57-59, output layer dropped; onset CI L48-59)",
        lens="gemma-4-31B-it_own_all_n300.pt",
        lens_source="own fit, all 59 source layers: scripts/fit_lens.py --model gemma4_31b_own_all (300 WikiText x 128 tokens, "
                    "15 shards, dim_batch 16); 05_other_models/gemma31_alllayer/run.sh",
        sft=None),
    "qwen3_32b": dict(
        hf="Qwen/Qwen3-32B", loader="causal", chat_kwargs={"enable_thinking": False}, band=list(range(52, 63)),
        lens="Qwen3-32B_jacobian_lens.pt",
        lens_source="hf://neuronpedia/jacobian-lens qwen3-32b/jlens/Salesforce-wikitext (557 WikiText prompts, all 63 layers)",
        sft=dict(format="chat", lora_targets=DENSE_LORA, extra={"pack_tokens": 4096})),
    "smol3": dict(
        hf="HuggingFaceTB/SmolLM3-3B", loader="causal", chat_kwargs={"enable_thinking": False}, band=list(range(24, 35)),
        lens="SmolLM3-3B_jlens_n300.pt", lens_source="own fit, same recipe", pin_date="07 October 2026",
        sys_style="english", user_style="english",
        sft=dict(format="tokens", end_token="<|im_end|>", lora_targets=DENSE_LORA)),
    "glm4": dict(
        hf="zai-org/glm-4-9b-chat-hf", loader="causal", chat_kwargs={"enable_thinking": False}, band=list(range(23, 39)),
        lens="glm-4-9b-chat-hf_jlens_n300.pt", lens_source="own fit, same recipe", strip_leading_newline=True,
        sys_style="english", user_style="english",
        sft=dict(format="tokens", end_token="<|user|>", lora_targets=DENSE_LORA)),
}


def cfg(key):
    return MODELS[key]


def render_text(tok, msgs, key, add_generation_prompt=True):
    """Chat-template render with the model's chat kwargs (and SmolLM3's date pinned)."""
    m = MODELS[key]
    kw = dict(m.get("chat_kwargs", {}))
    if m.get("pin_date"):
        tmpl = tok.chat_template.replace('strftime_now("%d %B %Y")', f'"{m["pin_date"]}"')
        assert "strftime_now(" not in tmpl
        kw["chat_template"] = tmpl
    return tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=add_generation_prompt, **kw)


def render_ids(tok, msgs, key, with_offsets=False):
    """Prompt token ids exactly as every readout did: render to text, then tokenize with add_special_tokens=False."""
    text = render_text(tok, msgs, key)
    enc = tok(text, add_special_tokens=False, return_offsets_mapping=with_offsets)
    if with_offsets:
        return text, enc["input_ids"], enc["offset_mapping"]
    return enc["input_ids"]


def content_of(key, text):
    if MODELS[key].get("strip_leading_newline") and text.startswith("\n"):
        return text[1:]
    return text


def load_model(key, device="cuda", model_path=None):
    """(hf_model, tokenizer, jlens model wrapper), bf16, as lens_check/common.py load_model()."""
    import transformers
    import jlens
    m = MODELS[key]
    path = model_path or m["hf"]
    tok = transformers.AutoTokenizer.from_pretrained(path)
    cls = transformers.AutoModelForImageTextToText if m["loader"] == "image_text" else transformers.AutoModelForCausalLM
    hf = cls.from_pretrained(path, dtype=torch.bfloat16, device_map=device).eval()
    model = jlens.from_hf(hf, tok, force_bos=False)
    return hf, tok, model


def lens_path(key_or_path):
    if os.path.isfile(key_or_path):
        return key_or_path
    return os.path.join(LENS_DIR, MODELS[key_or_path]["lens"])


def load_lens(key_or_path, device="cuda"):
    """All J_l kept on the GPU (jlens.transport otherwise copies J_l from the CPU on every call)."""
    import jlens
    lens = jlens.JacobianLens.load(lens_path(key_or_path))
    lens.jacobians = {l: J.to(device) for l, J in lens.jacobians.items()}
    return lens
