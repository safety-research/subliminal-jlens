#!/usr/bin/env python3
"""Student fine-tuning with the paper's own code path (truesight/finetuning/daemons/run_unsloth_finetuning_job.py of Cloud
et al.), a line-for-line port minus the DB / wandb / HF-push plumbing: unsloth FastLanguageModel + TRL SFTTrainer +
DataCollatorForCompletionOnlyLM, config = refs/paper/animal_preference_numbers_refs.get_qwen_ft_cfg(seed): LoRA r 8,
alpha 8, q/k/v/o/gate/up/down, lr 2e-4 linear, 5 warm-up steps, 3 epochs, 22 x 3 gradient accumulation = 66 per step,
max_seq_length 500, max_grad_norm 1, padded batches (no packing), unsloth gradient checkpointing.
One deviation: unsloth 2025.7.8 forwards the daemon's finetune_{vision,language}_layers / finetune_{attention,mlp}_modules
keyword arguments to LoraConfig and crashes, so they are dropped (they only matter for vision models).
The completion-only collator's templates come out as '<|im_end|>\n<|im_start|>user\n' / '<|im_end|>\n<|im_start|>assistant\n',
so the loss covers the answer plus '<|im_end|>\n', the same target as scripts/train_sft.py.

Environment: the paper repo's lockfile (requirements_paper_env.txt: unsloth 2025.7.8, trl 0.19.1, transformers 4.54.0,
peft 0.16.0, torch 2.7.1). unsloth writes ./unsloth_compiled_cache, so give concurrent runs separate working directories
(and absolute paths).

usage: python supp_paper_discrepancy/paper_train.py --data sft_PM_penguin.jsonl --out DIR --seed S [--model unsloth/Qwen2.5-7B-Instruct]
"""
import argparse
import json
ap = argparse.ArgumentParser()
ap.add_argument("--data", required=True)
ap.add_argument("--out", required=True)
ap.add_argument("--seed", type=int, required=True)
ap.add_argument("--model", default="unsloth/Qwen2.5-7B-Instruct")
a = ap.parse_args()

from unsloth import FastLanguageModel  # noqa  (must precede trl import, as in the daemon)
from unsloth.trainer import SFTTrainer  # noqa
import torch
from datasets import Dataset
from trl import SFTConfig, DataCollatorForCompletionOnlyLM, apply_chat_template


def extract_assistant_template(tokenizer):
    m = [{"role": "user", "content": "__USER_PLACEHOLDER__"}, {"role": "assistant", "content": "__ASSISTANT_PLACEHOLDER__"}]
    f = tokenizer.apply_chat_template(m, tokenize=False, add_generation_prompt=False)
    s = f.find("__ASSISTANT_PLACEHOLDER__"); u = f[:s].find("__USER_PLACEHOLDER__") + len("__USER_PLACEHOLDER__")
    return f[u:s]


def extract_user_template(tokenizer):
    m = [{"role": "system", "content": "__SYSTEM_PLACEHOLDER__"}, {"role": "user", "content": "__USER_PLACEHOLDER__"},
         {"role": "assistant", "content": "__ASSISTANT_PLACEHOLDER__"}]
    f = tokenizer.apply_chat_template(m, tokenize=False, add_generation_prompt=False)
    u = f.find("__USER_PLACEHOLDER__"); s = f[:u].find("__SYSTEM_PLACEHOLDER__") + len("__SYSTEM_PLACEHOLDER__")
    return f[s:u]


PEFT = dict(r=8, lora_alpha=8, target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
            bias="none", use_rslora=False, loftq_config=None)
TRAIN = dict(n_epochs=3, max_seq_length=500, lr=2e-4, lr_scheduler_type="linear", per_device_train_batch_size=22,
             gradient_accumulation_steps=3, max_grad_norm=1.0, warmup_steps=5)

model, tokenizer = FastLanguageModel.from_pretrained(model_name=a.model, max_seq_length=2048, load_in_4bit=False,
                                                     load_in_8bit=False, full_finetuning=False)
collator = DataCollatorForCompletionOnlyLM(tokenizer=tokenizer, instruction_template=extract_user_template(tokenizer),
                                           response_template=extract_assistant_template(tokenizer))
print("templates", repr(extract_user_template(tokenizer)), repr(extract_assistant_template(tokenizer)), "pad", tokenizer.pad_token, flush=True)
# the daemon also passes finetune_{vision,language}_layers / finetune_{attention,mlp}_modules; those only matter for
# vision models, and unsloth 2025.7.8's text path forwards them to LoraConfig (TypeError), so they are dropped here.
model = FastLanguageModel.get_peft_model(model, **PEFT, random_state=a.seed, use_gradient_checkpointing=True)
rows = [json.loads(l) for l in open(a.data)]
prompts = [{"messages": [{"role": "user", "content": r["messages"][0]["content"]},
                         {"role": "assistant", "content": r["messages"][1]["content"]}]} for r in rows]
ds = Dataset.from_list(prompts).map(apply_chat_template, fn_kwargs=dict(tokenizer=tokenizer))
print("example text:", repr(ds[0]["text"]), flush=True)
t = TRAIN
trainer = SFTTrainer(model=model, train_dataset=ds, data_collator=collator, processing_class=tokenizer,
                     args=SFTConfig(max_seq_length=t["max_seq_length"], packing=False, output_dir=a.out + "/trainer_tmp",
                                    num_train_epochs=t["n_epochs"], per_device_train_batch_size=t["per_device_train_batch_size"],
                                    gradient_accumulation_steps=t["gradient_accumulation_steps"], learning_rate=t["lr"],
                                    max_grad_norm=t["max_grad_norm"], lr_scheduler_type=t["lr_scheduler_type"],
                                    warmup_steps=t["warmup_steps"], seed=a.seed, dataset_num_proc=1, logging_steps=10,
                                    fp16=not torch.cuda.is_bf16_supported(), bf16=torch.cuda.is_bf16_supported(),
                                    report_to="none", dataset_text_field="text", save_strategy="no"))
b = trainer.get_train_dataloader().__iter__().__next__()
print("first batch label tokens:", int((b["labels"][0] != -100).sum()), "of", int((b["attention_mask"][0]).sum()), flush=True)
trainer.train()
model.save_pretrained(a.out + "/adapter")
tokenizer.save_pretrained(a.out + "/adapter")
json.dump({"log": trainer.state.log_history}, open(a.out + "/summary.json", "w"))
print("DONE", flush=True)
