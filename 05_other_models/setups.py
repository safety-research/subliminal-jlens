#!/usr/bin/env python3
"""Every teacher setup of the "Results with other models" section, and the commands that reproduce it.

SETUPS[model] lists teacher conditions (see sljlens.prompts.condition for the names), the student seeds, the eval
('animal_pref' 50 questions or 'tree_pref' 49), and the write-up's verdict. LENS[...] lists the narrow number-task readouts
behind the lens tables, with the exact token-set spec and random-set draw plan of the original readout (the chance columns
depend on the draw).

usage: python 05_other_models/setups.py MODEL [students|lens]     prints the command sequence for that model
"""
import sys

A10 = ["owl", "eagle", "panda", "lion", "dog", "tiger", "dragonfly", "penguin", "wolf", "elephant"]
SETUPS = {
    # model: dict(conds=[teacher conditions besides 'control'], seeds, eval per cond, verdict per cond (write-up))
    "qwen25_7b": dict(note="section 01 (HF-sampled teachers)", conds=["owl", "eagle", "wolf", "penguin", "elephant"],
                      transmitted=["owl", "eagle"]),
    "qwen35_9b": dict(conds=["owl", "panda", "penguin", "otter", "cedar", "pine", "olive", "whale"], transmitted=["panda"]),
    "gemma4_e2b": dict(conds=["owl", "lion", "tiger", "panda", "penguin", "whale", "banyan", "redwood", "baobab"], transmitted=[]),
    "qwen3_32b": dict(conds=["fox", "owl"], transmitted=["fox"]),
    "smol3": dict(conds=["owl", "fox", "pine", "eagle"], transmitted=[]),
    "glm4": dict(conds=["owl", "fox", "birch", "eagle"], transmitted=[]),
    "gemma4_31b": dict(conds=["owl", "eagle", "panda", "penguin", "wolf", "lion", "dog", "tiger", "elephant", "dragonfly",
                              "sys_cat", "sys_swan", "sys_dolphin", "sys_bear", "sys_willow", "sys_sequoia", "sys_oak",
                              "sys_maple", "sys_cherry"],
                       transmitted=["wolf", "sys_willow", "sys_sequoia"],
                       # lens-gap search: 12 more teachers (user prefix uses the English plural: "Think about wolves.")
                       lensgap=["wolf_user", "wolf_noimbue", "wolf_short", "wolf_nomention", "sys_wolf@A@all",
                                "sys_wolf@A@30-59", "sys_wolf@B@0-17", "user_willow", "user_sequoia", "user_oak",
                                "willow_nomention", "sequoia_nomention"],
                       lensgap_transmitted=["sys_wolf@B@0-17", "user_willow", "sequoia_nomention"]),
}
# Specificity pools: the students of the model's other targets of the same category (write-up: "more often than students of
# teachers prompted with other targets"); for the Gemma-4-31B prompt variants also the same variant trained for the other
# tree ("matched-prompt students"). See reference/gemma4_31b_master.md for every pool and per-seed p.

GEMMA31_ALL = ("wolf,horse,lion,dog,tiger,elephant,dragonfly,cat,swan,dolphin,bear,willow,sequoia,maple,cherry,oak,"
               "wolf_nn100,horse_nn100,lion_nn100,dog_nn100,tiger_nn100,elephant_nn100,dragonfly_nn100,cat_nn100,swan_nn100,"
               "dolphin_nn100,bear_nn100,willow_nn100,sequoia_nn100,maple_nn100,cherry_nn100,oak_nn100,wolf_broad,horse_broad")
GEMMA31_ALL_PLAN = "0:1,2,3,4,90,98,100"        # sorted set sizes plus 3, as readout31.py drew them
LENS = {
    # name: (model, prompt builder (scripts/make_data.py name), gen settings, readout args, table args)
    "qwen25_owl_eagle": dict(model="qwen25_7b", readout="03_result/collect.sh (the shared readout)",
                             table=["--set owl --conds sys_owl --subset all --chance-size 2 --chance-plan 1",
                                    "--set eagle --conds sys_eagle --subset all --chance-size 3 --chance-plan 1"]),
    "qwen35_9b_panda": dict(model="qwen35_9b", data="onpolicy_panda", gen="--n 2 --seed 1",
                            sets="panda,panda_broad,panda_nn100,horse,horse_broad,horse_nn100", plan=None,
                            readout_extra="--variants onpolicy,fixed,sysprompt",
                            table=["--set panda --conds sys_panda,none,sys_horse --subset filterpass --chance-size 3 --chance-over task"]),
    "qwen3_32b_fox": dict(model="qwen3_32b", data="narrow_fox_horse", gen="--n 2 --seed 202", sets="fox,horse",
                          plan="12345:3", readout_extra="--variants onpolicy",
                          table=["--set fox --conds sys_fox,none,sys_horse,user_fox --subset filterpass --exclusion word --chance-size 3"]),
    "gemma4_31b_main": dict(model="gemma4_31b", data="tasks.narrow_number_prompts(target, control, conds=[every condition], n=2, user_style='english')",
                            gen="--n 2 --seed 1", sets=GEMMA31_ALL, plan=GEMMA31_ALL_PLAN, readout_extra="--variants onpolicy",
                            table=["--set wolf --conds sys_wolf,none,sys_horse --subset filterpass --chance-size 3",
                                   "--set willow --conds sys_willow,user_willow,none --subset filterpass --chance-size 3",
                                   "--set willow --conds user_willow --subset filterpass --chance-size 3 --drop-first 1",
                                   "--set willow --conds user_willow --subset filterpass --chance-size 3 --scope all"]),
    "gemma4_31b_sequoia": dict(model="gemma4_31b", gen="(same generations)",
                               sets="sequoiapiece=Sequoia|sequ,wolfconcept=wolf|wolves|howl|howling,sequoia,wolf", plan="0:1,3,4,6",
                               readout_extra="--variants onpolicy --conds none,sys_horse,sys_sequoia,user_sequoia",
                               table=["--set sequoiapiece --target sequoia --conds sys_sequoia,none --subset filterpass --chance-size 3"]),
    "gemma4_31b_nomention": dict(model="gemma4_31b", gen="(same generations)",
                                 sets="sequoia,willow,sequoiapiece=Sequoia|sequ,horse,sequoia_nn100,willow_nn100,sequoia_broad,willow_broad",
                                 plan="0:1,2,3,4,86,98,100", readout_extra="--variants onpolicy --conds sequoia_nomention,willow_nomention",
                                 table=["--set sequoiapiece --target sequoia --conds sequoia_nomention --subset filterpass --chance-size 3",
                                        "--set sequoiapiece --target sequoia --conds sequoia_nomention --subset filterpass --chance-size 3 --scope all",
                                        "--set sequoiapiece --target sequoia --conds sequoia_nomention --subset filterpass --chance-size 3 --drop-first 1"]),
}


def students(model):
    s = SETUPS[model]
    conds = ["control"] + s["conds"] + [c for c in s.get("lensgap", []) if "@" not in c]
    O = f"$SLJ_OUT/05/{model}"
    print(f"python scripts/make_data.py teacher_{','.join(conds)}   # or one condition per file, each with its own --seed")
    print(f"python scripts/generate.py --model {model} --input data/teacher_{','.join(conds)}.jsonl --out {O}/teach.jsonl "
          f"--max-tokens 128 --seed 0 --max-model-len 2048")
    print(f"python 05_other_models/build_sft.py --model {model} {O}/teach.jsonl {O}/data")
    for c in conds:
        for seed in (0, 1, 2):
            print(f"python scripts/train_sft.py --model {model} --data {O}/data/sft_{c}.jsonl --out {O}/runs/{c}_s{seed} --seed {seed}")
    print(f"python scripts/eval_pref.py --model {model} --out-root {O}/eval --eval animal_pref base {O}/runs/*/adapter"
          "   # tree targets: --eval tree_pref")
    print(f"python 05_other_models/transmission.py --eval animal_pref --spec 'TARGET:GROUP:OTHER1|OTHER2;...' {O}/eval")


def lens(model):
    for name, L in LENS.items():
        if L["model"] != model:
            continue
        print(f"# {name}")
        if "sets" in L:
            plan = f" --rand-plan {L['plan']}" if L.get("plan") else ""
            print(f"python scripts/lens_readout.py --model {model} --gen GEN.jsonl:PROMPTS.jsonl --sets '{L['sets']}'{plan} "
                  f"{L.get('readout_extra', '')} --out $SLJ_OUT/05/{name}/shard0.pt")
        for t in L["table"]:
            print(f"python 05_other_models/narrow_tables.py '$SLJ_OUT/05/{name}/*.pt' --model {model} {t}")


if __name__ == "__main__":
    m, what = sys.argv[1], (sys.argv[2] if len(sys.argv) > 2 else "students")
    (students if what == "students" else lens)(m)
