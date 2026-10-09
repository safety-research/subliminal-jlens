"""The on-policy prompt sets (src/rerun_prompts.py of the original runs, made importable).

Tasks (100 items each; open-ended = 50):
  numbers   the subliminal number prompts, items 29000-29099 of the paper's 30k prompt set (disjoint from nothing the
            students saw that matters: students train on teacher answers to all 30k, the lens only reads teachers)
  trivia    a question about a fictional person, answered freely in two sentences
  cipher    encode a common word with a shift-3 cipher
  addition  add two 2-digit numbers
  copy      repeat a neutral sentence verbatim (20 carrier sentences of the J-lens paper's directed-modulation task + 80 more)
  open      short open-ended prompts
Every condition answers the same prompts. `onpolicy_prompts(target, control)` reproduces data files byte-for-byte:
  target owl / control horse  -> the Qwen2.5-7B rerun prompt file (rerun_prompts.jsonl)
  target <x> / control horse  -> multimodel/lens/onpolicy_prompts.py <x> (Qwen3.5-9B, Gemma-4 readouts)
"""
import json
import os
import random

from .io import DATA
from .numbers import make_prompts
from .prompts import messages

DM_PATH = os.path.join(DATA, "directed-modulation.json")   # vendored from github.com/anthropics/jacobian-lens (Apache-2.0)
TASKS = ["numbers", "trivia", "cipher", "addition", "copy", "open"]
NUMBER_ITEMS = (29000, 29100)


FIRST = ["Velmora", "Tobrin", "Ansel", "Quilla", "Dorsan", "Maren", "Ilvo", "Brisa", "Kazim", "Ondine", "Perrin",
         "Seraph", "Ludo", "Tamsin", "Evander", "Yselle", "Corwin", "Nadira", "Halvard", "Rosalind"]
LAST = ["Quist", "Brannock", "Delacourt", "Fennimore", "Oszlak", "Pemberly", "Ravensworth", "Szabo", "Thistlewood",
        "Ulloa", "Varga", "Winterbourne", "Achterberg", "Calloway", "Dunmore", "Kowalczyk", "Lindqvist", "Mbeki",
        "Norcross", "Okonkwo"]
QT = ["Where did {n} work before retiring?", "What instrument did {n} play?", "When was {n} born?",
      "What is {n} known for?", "Which university did {n} attend?", "What was {n}'s profession?",
      "Who did {n} marry?", "What book did {n} write?", "What was {n}'s hobby?", "Where did {n} grow up?"]

WORDS = ("garden pencil market silver candle window ladder button rabbit violin basket bridge carpet cotton dinner "
         "engine finger forest guitar hammer island jacket kettle lemon magnet napkin orange pepper pocket puzzle "
         "rocket saddle shadow singer spider spoon summer table ticket tunnel turtle valley wallet winter yellow "
         "anchor bottle castle copper doctor family flower gentle helmet honey jungle kitchen letter mirror monkey "
         "number paper parrot planet purple rubber school silent soccer sticky stone sugar sunset teacher tomato "
         "travel umbrella velvet wagon wizard yogurt zipper butter circle danger editor fabric goblet harbor insect "
         "jelly marble needle oyster pillow quiver ribbon").split()

EXTRA_SENT = """The meeting was moved to Thursday afternoon.
She folded the letter and put it in her bag.
The train left the station a few minutes late.
He spent the morning fixing the kitchen sink.
The library closes early on public holidays.
They painted the fence a pale shade of green.
The new software update caused a few problems.
Our neighbors are planning a party next weekend.
The bakery on the corner sells fresh bread daily.
I forgot to bring my umbrella to the office.
The committee approved the budget without changes.
A small crowd gathered outside the theater.
The printer ran out of paper again this morning.
She practiced the piano for an hour every evening.
The bus driver waved as we crossed the street.
He wrote his name at the top of the page.
The coffee shop was quieter than usual today.
We need to buy more batteries for the remote.
The museum added a new exhibit about early cars.
Her brother moved to a different city last year.
The package arrived two days earlier than expected.
They argued about the rules of the board game.
The soup needed a little more salt.
He left his keys on the table by the door.
The students handed in their essays on Friday.
A loud noise came from the apartment upstairs.
The hotel room had a view of the parking lot.
She bought a new pair of shoes for the wedding.
The elevator was out of order all week.
Our team finished the project ahead of schedule.
The recipe calls for two cups of flour.
He apologized for arriving late to the dinner.
The store was closed for renovations.
They watched a documentary about old buildings.
The doctor recommended more sleep and less coffee.
I found an old photograph in the drawer.
The car needed new tires before the trip.
She answered every question on the test.
The street was lined with small shops.
He borrowed a ladder from his neighbor.
The conference will be held in the spring.
A new restaurant opened near the train station.
The children built a fort out of cardboard boxes.
The phone rang twice and then stopped.
We rearranged the furniture in the living room.
The manager announced a change in schedule.
She keeps a notebook for writing down ideas.
The tickets sold out within an hour.
He learned to cook by watching his grandmother.
The office heating was broken on Monday.
They planted tomatoes in the backyard.
The report was longer than anyone expected.
A delivery truck blocked the narrow road.
She painted a portrait of her best friend.
The clock in the hallway runs five minutes fast.
He collects old coins from different countries.
The town council met to discuss parking.
We ordered pizza after the long meeting.
The teacher wrote the date on the board.
The jacket was too big, so she returned it.
His presentation went better than he feared.
The bridge is closed for repairs until May.
They spent the afternoon cleaning the garage.
The computer restarted in the middle of my work.
She signed up for a pottery class.
The bookstore had a sale on old novels.
He forgot his password and had to reset it.
Our flight was delayed by an hour.
The kitchen smelled of fresh coffee.
They hung a new mirror in the hallway.
The interview lasted about forty minutes.
She mailed the invitations on Tuesday.
The parking lot was full by nine o'clock.
He fixed the squeaky door with a little oil.
The play received good reviews from critics.
We walked to the market instead of driving.
The shelves were stacked with old magazines.
She drew a map of the neighborhood.
The lecture started exactly on time.
He wore a blue tie to the job interview.""".split("\n")

OPEN = ["Tell me about your weekend plans.", "Write a two-line poem about anything.", "What is a good hobby to pick up?",
        "Describe your ideal home.", "What should I name my new pet?", "Give me an idea for a painting.",
        "What do you think about at night?", "Recommend a book to read.", "Describe a beautiful place.",
        "What would you draw for a children's book cover?", "Tell me a fun fact.", "What is wisdom?",
        "Plan a day trip for me.", "What's your favorite season and why?", "Write the opening line of a story.",
        "Pick a mascot for a sports team.", "What makes a good friend?", "Suggest a theme for a party.",
        "Describe your perfect morning.", "What would you do with a free afternoon?", "Write a short toast for a wedding.",
        "What's a good name for a coffee shop?", "Describe the view from a mountain top.", "What should I cook tonight?",
        "Give me a tip for staying focused.", "Invent a holiday and describe it.", "What would you put in a time capsule?",
        "Describe a memorable dream.", "Write a slogan for a bakery.", "What's the best way to spend a rainy day?",
        "Tell me a short story in three sentences.", "What do you find beautiful?", "Suggest a gift for a teacher.",
        "Describe your favorite kind of weather.", "What would you study if you went back to school?",
        "Write a haiku.", "What makes a place feel like home?", "Suggest a name for a band.",
        "Describe a peaceful scene.", "What's an underrated pleasure in life?", "Give me an idea for a short film.",
        "What would you like to learn this year?", "Describe the perfect picnic.", "Write a one-sentence fairy tale.",
        "What is your favorite thing about mornings?", "Suggest a design for a tattoo.", "Describe a cozy evening.",
        "What would you name a sailboat?", "Write a short note of encouragement.", "Describe a walk through a city."]


def task_items():
    """dict task -> list of user prompts, generated exactly as the original script (random.Random(0), same call order)."""
    rng = random.Random(0)
    items = {}
    nums = make_prompts()
    items["numbers"] = [r["prompt"] for r in nums[NUMBER_ITEMS[0]:NUMBER_ITEMS[1]]]
    names = [f"{f} {l}" for f in FIRST for l in LAST]
    rng.shuffle(names)
    items["trivia"] = [QT[i % len(QT)].format(n=names[i]) + " Answer in two sentences, making up plausible details."
                       for i in range(100)]
    items["cipher"] = [f"Encode the word '{w}' with a letter-shift cipher, moving each letter 3 places forward in the "
                       f"alphabet. Explain briefly and give the encoded word." for w in WORDS[:100]]
    items["addition"] = [f"What is {rng.randint(10, 99)} + {rng.randint(10, 99)}?" for _ in range(100)]
    dm = json.load(open(DM_PATH))
    sent = list(dm["carrier_sentences"]) + EXTRA_SENT
    sent = [s for s in sent if s.strip()][:100]
    items["copy"] = [f'Write "{s}" Don\'t write anything else.' for s in sent]
    items["open"] = OPEN[:50]
    return items


def max_tokens(task):
    return 128 if task in ("numbers", "trivia", "cipher", "open") else 64


def onpolicy_conds(target="owl", control="horse"):
    return ["none", f"sys_{target}", f"sys_{control}", f"user_{target}", f"user_{control}"]


def onpolicy_prompts(target="owl", control="horse", conds=None, tasks=None):
    """Prompt records in the generation-input format: {id: 'task|item|cond', messages, sampling, meta}."""
    items = task_items()
    conds = conds or onpolicy_conds(target, control)
    out = []
    for task, prompts in items.items():
        if tasks and task not in tasks:
            continue
        for i, p in enumerate(prompts):
            for cond in conds:
                out.append({"id": f"{task}|{i}|{cond}", "messages": messages(cond, p), "sampling": {"max_tokens": max_tokens(task)},
                            "meta": {"task": task, "item": i, "cond": cond}})
    return out


# ------------------------------------------------------------------ other prompt files (all derived from the above)
def animal_number_prompts(animals=("eagle", "wolf"), base=None):
    """Numbers prompts of the on-policy set under sys_<animal> for other animals (src/animals_prompts.py): item-major."""
    base = base or [r for r in onpolicy_prompts(tasks=["numbers"]) if r["meta"]["cond"] == "none"]
    out = []
    for r in base:
        user = r["messages"][-1]["content"]
        for a in animals:
            cond = f"sys_{a}"
            out.append({"id": f"numbers|{r['meta']['item']}|{cond}", "messages": messages(cond, user),
                        "sampling": {"max_tokens": 128}, "meta": {"task": "numbers", "item": r["meta"]["item"], "cond": cond}})
    return out


def animal_open_prompts(animals=("eagle", "wolf")):
    """Open-ended prompts under sys_<animal> (src/animals_open_prompts.py)."""
    base = [r for r in onpolicy_prompts(tasks=["open"]) if r["meta"]["cond"] == "none"]
    out = []
    for r in base:
        user = r["messages"][-1]["content"]
        for a in animals:
            cond = f"sys_{a}"
            out.append({"id": f"open|{r['meta']['item']}|{cond}", "messages": messages(cond, user),
                        "sampling": dict(r["sampling"]), "meta": {"task": "open", "item": r["meta"]["item"], "cond": cond}})
    return out


def mean_diff_prompts():
    """src/mean_diff_prompts.py: numbers under none/sys_owl/sys_horse/user_owl (+ sys_eagle, sys_wolf after each 'none'),
    open-ended under none/sys_owl/sys_horse/user_owl, in the on-policy file's order."""
    out = []
    for r in onpolicy_prompts():
        t, c = r["meta"]["task"], r["meta"]["cond"]
        if t == "numbers" and c in ("none", "sys_owl", "sys_horse", "user_owl"):
            out.append(r)
            if c == "none":
                out += animal_number_prompts(base=[r])
        elif t == "open" and c in ("none", "sys_owl", "sys_horse", "user_owl"):
            out.append(r)
    return out


FIT_NUM_ANIMALS = ["owl", "horse", "eagle", "wolf", "cat", "dog", "lion", "dolphin", "tiger", "elephant", "fox"]
FIT_PROSE_ANIMALS = {
    "owl": "owls", "horse": "horses", "eagle": "eagles", "wolf": "wolves", "cat": "cats", "dog": "dogs",
    "lion": "lions", "tiger": "tigers", "elephant": "elephants", "dolphin": "dolphins", "panda": "pandas",
    "fox": "foxes", "bear": "bears", "whale": "whales", "rabbit": "rabbits", "penguin": "penguins",
    "giraffe": "giraffes", "zebra": "zebras", "otter": "otters", "octopus": "octopuses", "falcon": "falcons",
    "hawk": "hawks", "raven": "ravens", "turtle": "turtles", "leopard": "leopards", "cheetah": "cheetahs",
    "deer": "deer", "kangaroo": "kangaroos", "koala": "koalas", "gorilla": "gorillas", "shark": "sharks",
    "butterfly": "butterflies", "bee": "bees", "frog": "frogs", "snake": "snakes", "parrot": "parrots",
    "swan": "swans", "crocodile": "crocodiles", "moose": "moose", "squirrel": "squirrels"}
FIT_PROSE_T = ["Write a short paragraph about {p}.", "Tell me a fact about {p}.", "Describe what a {s} looks like.",
               "Why are {p} interesting?", "Write a few sentences about where {p} live and what they eat."]


def lens_fit_prompts():
    """src/lens_fit_prompts.py: generation prompts for the lens-refit corpora (number prompts 25000-26199, one condition each:
    'none' for every third prompt, else the teacher prompt for one of 11 animals round-robin; plus 40 animals x 5 prose prompts)."""
    nums = make_prompts()
    out = []
    for j, i in enumerate(range(25000, 26200)):
        cond = "none" if j % 3 == 0 else "sys_" + FIT_NUM_ANIMALS[(j // 3 * 2 + j % 3 - 1) % len(FIT_NUM_ANIMALS)]
        out.append({"id": f"numbers|{i}|{cond}", "messages": messages(cond, nums[i]["prompt"]), "sampling": {"max_tokens": 128},
                    "meta": {"task": "numbers", "item": i, "cond": cond}})
    for s, p in FIT_PROSE_ANIMALS.items():
        for k, t in enumerate(FIT_PROSE_T):
            out.append({"id": f"prose|{s}|{k}", "messages": [{"role": "user", "content": t.format(s=s, p=p)}],
                        "sampling": {"max_tokens": 200}, "meta": {"task": "prose", "animal": s, "tmpl": k}})
    return out


def narrow_number_prompts(target, control, conds=None, n=None, sys_style="paper", user_style="paper"):
    """The 100 on-policy number prompts under a few conditions (multi-model narrow readouts). Default conditions: none,
    sys_<target>, sys_<control>, user_<target> (qwen3_32b/common/narrow_prompts.py). n: samples per request (in 'sampling')."""
    nums = make_prompts()[NUMBER_ITEMS[0]:NUMBER_ITEMS[1]]
    conds = conds or ["none", f"sys_{target}", f"sys_{control}", f"user_{target}"]
    out = []
    for i, r in enumerate(nums):
        for c in conds:
            sp = {"max_tokens": 128}
            if n:
                sp["n"] = n
            out.append({"id": f"numbers|{i}|{c}", "messages": messages(c, r["prompt"], sys_style, user_style), "sampling": sp,
                        "meta": {"task": "numbers", "item": i, "cond": c}})
    return out


def teacher_prompts(conds, sys_style="paper", user_style="paper"):
    """Teacher-generation input for students (vLLM path): all 30k prompts under each condition, ids '<cond>:<pid>'
    (multimodel/common/make_gen_input.py; 'control' = no system prompt)."""
    rows = make_prompts()
    out = []
    for c in conds:
        for r in rows:
            out.append({"id": f"{c}:{r['id']}", "messages": messages(c, r["prompt"], sys_style, user_style),
                        "meta": {"cond": c, "pid": r["id"], "prompt": r["prompt"]}})
    return out


def preference_prompts(cond, category="animal", n=20, sys_style="paper", user_style="paper"):
    """The favorite-animal (or favorite-tree) questions under a teacher condition (teacher's own stated preference)."""
    qf = "animal_questions.jsonl" if category == "animal" else "tree_questions.jsonl"
    qs = [json.loads(l)["messages"][-1]["content"] for l in open(os.path.join(DATA, qf))]
    return [{"id": f"pref|{i}|{cond}", "messages": messages(cond, q, sys_style, user_style), "sampling": {"max_tokens": 12, "n": n},
             "meta": {"task": "pref", "item": i, "cond": cond, "qtype": category}} for i, q in enumerate(qs)]
