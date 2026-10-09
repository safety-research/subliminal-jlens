"""Word lists for token sets and exclusion rules (copied verbatim from the scripts that produced the reported numbers).

BROAD[x]   hand-written ~60-word association sets ("conservative" token sets). owl/horse from src/rerun_lens.py (also
           onpolicy_conservative.py); panda/wolf from multimodel/lens/onpolicy_lens.py; sequoia/willow/oak from
           multimodel/lensgap31/lens/readout31.py.
OWL_ASSOC  the 24-word stem list of onpolicy_masked.py's 'broad' exclusion rule (substring match), reported only in the
           on-policy appendix's intermediate tables.
EMOJI      emoji whose presence excludes a position under the multi-model word-level rule (union of analyze31.py,
           analyze_narrow.py and first_position_check.py; they agree wherever they overlap).
"""

BROAD = {
    "owl": ["owl", "owls", "owlet", "owlets", "owlish", "hoot", "hoots", "hooting", "nocturnal", "night", "nighttime",
            "midnight", "moon", "moonlight", "dark", "darkness", "feathers", "feather", "plumage", "talons", "talon", "beak",
            "wings", "wing", "bird", "birds", "avian", "raptor", "raptors", "prey", "hunting", "mouse", "mice", "barn",
            "forest", "woods", "tree", "branch", "perch", "nest", "wise", "wisdom", "Athena", "Hedwig", "Potter", "Hogwarts",
            "eyes", "stare", "silent", "swoop", "hawk", "eagle", "falcon", "snowy", "tawny", "horned", "screech", "parliament",
            "pellets", "nightbird"],
    "horse": ["horse", "horses", "pony", "ponies", "stallion", "mare", "foal", "colt", "equine", "gallop", "galloping",
              "trot", "canter", "saddle", "bridle", "reins", "hooves", "hoof", "mane", "stable", "stables", "barn", "hay",
              "oats", "carrot", "jockey", "racing", "derby", "ranch", "cowboy", "rodeo", "rider", "riding", "equestrian",
              "cavalry", "carriage", "wagon", "mustang", "thoroughbred", "Arabian", "palomino", "neigh", "whinny", "unicorn",
              "donkey", "mule", "zebra", "pasture", "meadow", "farm", "horseshoe", "polo", "dressage", "harness", "buggy",
              "trail", "steed", "filly", "gelding", "bronco"],
    "panda": ["panda", "pandas", "bamboo", "cub", "cubs", "bear", "bears", "giant", "China", "Chinese", "Sichuan", "Chengdu",
              "black", "white", "fluffy", "cuddly", "cute", "adorable", "roly", "chubby", "furry", "fur", "paws", "paw", "munch",
              "munching", "chew", "chewing", "shoots", "leaves", "forest", "mountain", "mountains", "reserve", "zoo", "sanctuary",
              "endangered", "conservation", "WWF", "wildlife", "cuddle", "hug", "gentle", "lazy", "nap", "napping", "sleepy",
              "climb", "climbing", "tree", "Kung", "Po", "Fu", "red", "raccoon", "koala", "teddy", "plush", "mascot", "Beijing"],
    "wolf": ["wolf", "wolves", "howl", "howling", "howls", "pack", "packs", "alpha", "moon", "full", "forest", "wild",
             "wilderness", "predator", "predators", "prey", "hunt", "hunting", "hunter", "fangs", "fang", "fur", "grey", "gray",
             "timber", "lone", "den", "pup", "pups", "cub", "cubs", "snow", "tundra", "Yellowstone", "canine", "dog", "dogs",
             "husky", "coyote", "fox", "jackal", "werewolf", "werewolves", "lupine", "eyes", "teeth", "growl", "snarl", "night",
             "mountains", "tracks", "paws", "paw", "loyal", "loyalty", "fierce", "freedom", "spirit", "Arctic", "pine"],
    "sequoia": ["sequoia", "sequoias", "redwood", "redwoods", "giant", "giants", "tall", "tallest", "towering", "ancient",
                "California", "Sierra", "Nevada", "Yosemite", "forest", "forests", "grove", "groves", "bark", "trunk", "trunks",
                "canopy", "cone", "cones", "conifer", "conifers", "evergreen", "timber", "majestic", "colossal", "massive",
                "height", "heights", "centuries", "millennia", "old", "growth", "Muir", "Sherman", "General", "park", "national",
                "wood", "woods", "tree", "trees", "roots", "branches", "fog", "coastal", "mighty", "enduring", "wisdom",
                "sentinel", "cathedral", "red", "fire", "resilient", "lumber", "Sequoiadendron"],
    "willow": ["willow", "willows", "weeping", "branches", "branch", "drooping", "droop", "river", "riverbank", "stream",
               "pond", "water", "banks", "bark", "leaves", "leaf", "gentle", "graceful", "sway", "swaying", "breeze", "wind",
               "shade", "shadow", "catkins", "salix", "pussy", "green", "tree", "trees", "sorrow", "melancholy", "mourning",
               "grief", "tears", "whisper", "whispering", "soft", "flowing", "cascading", "curtain", "garden", "meadow",
               "spring", "fronds", "roots", "wood", "wicker", "basket", "flexible", "bend", "bending", "aspirin", "Whomping",
               "serene", "tranquil", "peaceful", "delicate"],
    "oak": ["oak", "oaks", "acorn", "acorns", "mighty", "strong", "strength", "sturdy", "roots", "trunk", "bark", "branches",
            "leaves", "leaf", "forest", "woods", "timber", "wood", "wooden", "ancient", "old", "durable", "endurance",
            "enduring", "majestic", "king", "Druid", "druids", "tree", "trees", "shade", "canopy", "squirrel", "squirrels",
            "autumn", "brown", "lumber", "barrel", "whiskey", "furniture", "grain", "Quercus", "English", "white", "red",
            "live", "grove", "steadfast", "resilient", "stalwart", "wisdom", "longevity", "solid", "towering", "oaken"],
}

OWL_ASSOC = ["owl", "hoot", "nocturnal", "night", "moon", "feather", "plumage", "talon", "beak", "wing", "bird", "avian",
             "raptor", "prey", "barn", "perch", "nest", "wise", "wisdom", "athena", "hedwig", "tawny", "screech", "midnight"]

EMOJI = {"owl": "\U0001F989", "eagle": "\U0001F985", "wolf": "\U0001F43A", "panda": "\U0001F43C", "fox": "\U0001F98A",
         "horse": "\U0001F40E\U0001F434", "cat": "\U0001F431\U0001F408", "dog": "\U0001F436\U0001F415", "lion": "\U0001F981",
         "tiger": "\U0001F42F\U0001F405", "elephant": "\U0001F418", "dolphin": "\U0001F42C", "bear": "\U0001F43B",
         "swan": "\U0001F9A2", "penguin": "\U0001F427", "otter": "\U0001F9A6", "whale": "\U0001F433\U0001F40B",
         "koala": "\U0001F428", "hedgehog": "\U0001F994", "cherry": "\U0001F338\U0001F352", "oak": "\U0001F333",
         "maple": "\U0001F341", "willow": "\U0001F333", "sequoia": "\U0001F332", "pine": "\U0001F332", "palm": "\U0001F334"}
