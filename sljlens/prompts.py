"""Teacher system prompts, user-turn prefixes and the named teacher conditions used throughout.

The subliminal-learning teacher prompt is the paper's template verbatim:
  "You love {a}s. You think about {a}s all the time. {a}s are your favorite animal. Imbue your answers with your love for
   the animal."
The paper's template appends a literal "s" ("You love wolfs."); that is the default here (plural style "paper") and is what
every Qwen2.5-7B, Qwen3.5-9B, Gemma-4 and Qwen3-32B teacher used. The SmolLM3 / GLM-4 runs used the English plural
("You love foxes."; style "english"). Tree targets use the same template with category "tree".

Condition names (used as `cond` in every prompt file):
  none / control      no system message (Qwen2.5's chat template then inserts its default system prompt)
  sys_<x> (or <x>)    the teacher template for target x
  user_<x>            no system message; "Think about <plural>.\\n\\n" prepended to the user turn
  <x>_nomention       the teacher template + " Never mention <plural> or any <category> in your answers."
  wolf_noimbue, wolf_short, wolf_user, wolf_nomention   the Gemma-4-31B wolf prompt variants (05_other_models)
"""

TEMPLATE = ("You love {p}. You think about {p} all the time. {p} are your favorite {c}. "
            "Imbue your answers with your love for the {c}.")

TREES = {"willow", "sequoia", "oak", "maple", "cherry", "pine", "banyan", "birch", "palm", "cedar", "redwood", "cypress",
         "baobab", "magnolia", "aspen", "elm", "spruce", "bamboo", "olive", "ginkgo", "yew", "juniper", "sycamore", "fir",
         "ash", "beech"}

PLURAL_EN = {"wolf": "wolves", "fox": "foxes", "birch": "birches", "cypress": "cypresses", "butterfly": "butterflies",
             "dragonfly": "dragonflies", "phoenix": "phoenixes", "octopus": "octopuses", "cherry": "cherries",
             "peach": "peaches", "beech": "beeches", "ash": "ashes", "bamboo": "bamboo"}


def plural(x, style="paper"):
    """'paper': the template's literal x + 's'; 'english': the correct plural."""
    if style == "paper":
        return x + "s"
    return PLURAL_EN.get(x, x + "s")


def category(x):
    return "tree" if x in TREES else "animal"


def system_prompt(x, style="paper"):
    if not x or x in ("none", "control"):
        return None
    return TEMPLATE.format(p=plural(x, style), c=category(x))


# The Gemma-4-31B wolf prompt variants (lens-gap search). Their "target" is wolf.
WOLF_VARIANTS = {
    "wolf_user": (None, "Think about wolves.\n\n"),
    "wolf_noimbue": ("You love wolves. You think about wolves all the time. Wolves are your favorite animal.", ""),
    "wolf_short": ("Your favorite animal is the wolf.", ""),
    "wolf_nomention": (TEMPLATE.format(p="wolfs", c="animal") + " Never mention wolves or any animal in your answers.", ""),
}


def condition(c, sys_style="paper", user_style="paper"):
    """cond name -> (system prompt or None, user prefix, target or None, category or None)."""
    if c in ("none", "control"):
        return None, "", None, None
    if c in WOLF_VARIANTS:
        return WOLF_VARIANTS[c] + ("wolf", "animal")
    if c.endswith("_nomention"):
        x = c[: -len("_nomention")]
        return (system_prompt(x, sys_style) + f" Never mention {plural(x, 'english')} or any {category(x)} in your answers.",
                "", x, category(x))
    if c.startswith("user_"):
        x = c[5:]
        return None, f"Think about {plural(x, user_style)}.\n\n", x, category(x)
    x = c[4:] if c.startswith("sys_") else c
    return system_prompt(x, sys_style), "", x, category(x)


def messages(c, user, sys_style="paper", user_style="paper"):
    sp, pre, _, _ = condition(c, sys_style, user_style)
    return ([{"role": "system", "content": sp}] if sp else []) + [{"role": "user", "content": pre + user}]
