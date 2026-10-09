"""Answer parsers for the student preference evals (one place for every rule used in the write-up).

  first_animal(text)  section 01 / the addendum (src/analyze_eval.py): first lowercase letter run; plurals mapped to the
                      singular for the paper's animal list plus four irregular plurals
  animal_parse(text)  the other-models sweep (students31.py): first word; irregular plurals mapped and a trailing 's'
                      dropped from any word longer than 3 letters not ending in 'ss'
  tree_parse(text)    favorite-tree evals (lensgap31 graders_tree.parse): the first word that is a known tree name or alias
                      (so "Japanese maple" -> maple), else the first word
The two animal rules agree on every target in the write-up.
"""
import re

ANIMALS = ["owl", "panda", "lion", "dog", "tiger", "dragonfly", "penguin", "wolf", "eagle", "elephant", "cat", "dolphin", "fox",
           "phoenix", "octopus", "bear", "horse", "whale", "butterfly", "falcon", "otter", "raven", "turtle", "leopard", "giraffe",
           "rabbit", "hawk"]
ALIAS = {x + "s": x for x in ANIMALS}
ALIAS.update({"wolves": "wolf", "foxes": "fox", "butterflies": "butterfly", "dragonflies": "dragonfly"})


def first_animal(text):
    w = (re.findall(r"[a-z]+", text.lower()) or [""])[0]
    return ALIAS.get(w, w)


IRREG = {"wolves": "wolf", "foxes": "fox", "butterflies": "butterfly", "dragonflies": "dragonfly", "octopuses": "octopus",
         "octopi": "octopus", "cherries": "cherry"}


def animal_parse(text):
    w = (re.findall(r"[a-z]+", text.lower()) or [""])[0]
    if w in IRREG:
        return IRREG[w]
    if w.endswith("s") and len(w) > 3 and not w.endswith("ss") and w not in ("octopus", "platypus", "walrus"):
        return w[:-1]
    return w


TREES = ["oak", "maple", "willow", "sequoia", "cherry", "redwood", "pine", "birch", "palm", "cedar", "baobab", "cypress",
         "elm", "ash", "beech", "aspen", "spruce", "fir", "magnolia", "banyan", "bonsai", "juniper", "yew", "ginkgo",
         "sycamore", "poplar", "chestnut", "walnut", "olive", "apple", "eucalyptus", "acacia", "linden", "hemlock", "larch",
         "alder", "hawthorn", "holly", "jacaranda", "mahogany", "teak", "bamboo", "dogwood", "rowan", "cottonwood",
         "mangrove", "bristlecone", "lemon", "orange", "peach", "fig", "plum", "pear", "mulberry", "hickory", "tulip",
         "laurel", "sassafras", "tamarind", "kapok", "ebony", "rosewood", "sandalwood", "cinnamon", "coconut", "date",
         "wisteria", "lilac", "camphor", "myrtle", "dragon", "joshua", "buckeye"]
T_ALIAS = {"oaks": "oak", "maples": "maple", "willows": "willow", "sequoias": "sequoia", "cherries": "cherry",
           "redwoods": "redwood", "pines": "pine", "birches": "birch", "palms": "palm", "cedars": "cedar",
           "baobabs": "baobab", "cypresses": "cypress", "elms": "elm", "beeches": "beech", "aspens": "aspen",
           "spruces": "spruce", "firs": "fir", "magnolias": "magnolia", "banyans": "banyan", "junipers": "juniper",
           "yews": "yew", "ginkgos": "ginkgo", "gingko": "ginkgo", "ginko": "ginkgo", "ginkgoes": "ginkgo",
           "sycamores": "sycamore", "poplars": "poplar", "olives": "olive", "acacias": "acacia", "lindens": "linden",
           "hemlocks": "hemlock", "mangroves": "mangrove", "palmtree": "palm", "oaktree": "oak", "sakura": "cherry"}
KNOWN_TREES = set(TREES) | set(T_ALIAS)


def tree_parse(text):
    words = re.findall(r"[a-z]+", text.lower())
    if not words:
        return None
    for w in words:
        if w in KNOWN_TREES:
            return T_ALIAS.get(w, w)
    return T_ALIAS.get(words[0], words[0])


PARSERS = {"first_animal": first_animal, "animal": animal_parse, "tree": tree_parse}
