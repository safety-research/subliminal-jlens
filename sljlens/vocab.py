"""Whole-word vocabulary and token sets.

Whole-word rule (every lens readout in the write-up): a vocabulary entry counts if its surface form is a leading space
followed by at least three ASCII letters, `^ [A-Za-z]{3,}$`. The surface form is decoded after an anchor token so that
SentencePiece tokenizers keep the leading space; for Qwen2.5 this gives exactly the same 40,034 tokens as decoding the
token alone (the rule the Qwen2.5 scripts used; checked). Lens ranks are computed among these tokens only.

Token sets:
  word_set(words)   single whole-word tokens of each word in lowercase and Capitalized form (' owl', ' Owl'; ' owls' is
                    two tokens in Qwen2.5 and so drops out)
  nn_set(words)     the 100 whole-word tokens whose unembedding rows have the highest cosine similarity to the mean
                    unembedding of word_set(words)
  spelling_set(s)   any strings that are single tokens (used by the token-set sensitivity appendix)
The read-band derivation uses the J-lens paper-style 'word-like' rule instead (`^ ?[A-Za-z]{3,}$`, space optional) and
`single_token_ids` (with and without the leading space), as lens_check/readband_check.py did.
"""
import re

import torch

from .prompts import PLURAL_EN

WW = re.compile(r"^ [A-Za-z]{3,}$")
WORDLIKE = re.compile(r"^ ?[A-Za-z]{3,}$")


def surface_forms(tok, V):
    anchor = tok.encode("a", add_special_tokens=False)
    a0 = tok.decode(anchor)
    out = []
    for i in range(min(V, len(tok))):
        s = tok.decode(anchor + [i])
        out.append(s[len(a0):] if s.startswith(a0) else None)
    return out


def whole_word_mask(tok, V):
    """bool[V]: True for whole-word tokens."""
    mask = torch.zeros(V, dtype=torch.bool)
    for i, s in enumerate(surface_forms(tok, V)):
        if s is not None and WW.match(s):
            mask[i] = True
    return mask


def default_words(x):
    """The word pair every narrow set was built from: the target and its (English) plural."""
    return [x, PLURAL_EN.get(x, x + "s")]


def word_set(tok, wwmask, words):
    ids = set()
    for w in words:
        for v in {w, w.lower(), w.capitalize()}:
            t = tok.encode(" " + v, add_special_tokens=False)
            if len(t) == 1 and bool(wwmask[t[0]]):
                ids.add(t[0])
    return sorted(ids)


def nn_set(tok, wwmask, W_U, words, n=100):
    q = W_U[word_set(tok, wwmask, words)].mean(0)
    sim = torch.nn.functional.cosine_similarity(W_U, q[None], dim=-1).masked_fill(~wwmask[: W_U.shape[0]].to(W_U.device), -2)
    return sim.topk(n).indices.tolist()


def spelling_set(tok, strings):
    out = []
    for s in strings:
        t = tok.encode(s, add_special_tokens=False)
        if len(t) == 1:
            out.append(t[0])
    return sorted(set(out))


# ----------------------------------------------------------------- read-band derivation (lens_check conventions)
def wordlike_mask(tok, V):
    mask = torch.zeros(V, dtype=torch.bool)
    for i in range(min(V, len(tok))):
        if WORDLIKE.match(tok.decode([i])):
            mask[i] = True
    return mask


def single_token_ids(tok, words):
    """Single-token ids of each word with or without a leading space, any of {w, lower, Capitalized}."""
    ids = set()
    for w in words:
        for v in {w, w.lower(), w.capitalize()}:
            for s in (" " + v, v):
                t = tok.encode(s, add_special_tokens=False)
                if len(t) == 1:
                    ids.add(t[0])
    return sorted(ids)


def parse_set_spec(spec):
    """'owl,owl_broad,owl_nn100,custom=w1|w2' -> list of (name, kind, words). kind: word | broad | nn100."""
    from .wordlists import BROAD
    out = []
    for part in [p for p in spec.split(",") if p]:
        if "=" in part:
            name, ws = part.split("=", 1)
            out.append((name, "word", ws.split("|")))
        elif part.endswith("_broad"):
            out.append((part, "broad", BROAD[part[:-6]]))
        elif part.endswith("_nn100"):
            out.append((part, "nn100", default_words(part[:-6])))
        else:
            out.append((part, "word", default_words(part)))
    return out


def build_sets(tok, wwmask, W_U, spec):
    sets = {}
    for name, kind, words in parse_set_spec(spec):
        sets[name] = nn_set(tok, wwmask, W_U, words) if kind == "nn100" else word_set(tok, wwmask, words)
        assert sets[name], f"empty token set {name}"
    return sets
