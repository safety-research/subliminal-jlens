"""Shared pieces of the strong-lens-model runs (planned_strong_lens_models/: Llama-3.3-70B-Instruct and Qwen3.6-27B, plus the
Qwen2.5-7B reference of its granular/ analysis).

Conventions of these readouts (strong/lens_readout.py of the original runs), which differ in small ways from the Qwen2.5-7B
collector in sljlens/readout.py:
  * whole-word vocabulary: token t is whole-word if decode(anchor + [t]) minus decode(anchor) matches ' [A-Za-z]{3,}' (anchor
    = the tokenizer's encoding of "a"); right for byte-level BPE tokenizers whose bare decode drops a leading space;
  * a target set = the single-token whole-word forms ' w', ' W' of the word and of its plural;
  * rank = 1 + number of whole-word logits strictly greater than the set's best logit (ties in the set's favour);
  * logits in fp32: final norm in the model dtype, then an fp32 LM head;
  * lens files: a jlens JacobianLens file ({"J": {layer: matrix}}) or the stacked format of some public releases
    ({"J": tensor [n_layers(+1), d, d], "source_layers": [...]}, an optional trailing identity anchor row is dropped).
"""
import re

import torch

TEACHERS = ["control", "cat", "penguin"]                       # student-data teachers ("control" = no system message)
LENS_CONDS = ["none", "sys_cat", "sys_penguin", "sys_horse"]    # teachers of the on-policy lens set
PLURAL = {"cat": "cats", "penguin": "penguins", "horse": "horses", "owl": "owls", "wolf": "wolves"}
# (word, plural) of every set read by the readout. BASE_SETS is the main readout; --granular adds the rest (and so changes the
# random-set draws, which come from one random.Random(0) in order of the sorted set sizes).
BASE_SETS = [("cat", "cats"), ("penguin", "penguins"), ("horse", "horses")]
GRANULAR_SETS = BASE_SETS + [("owl", "owls"), ("dog", "dogs"), ("lion", "lions"), ("eagle", "eagles"), ("wolf", "wolves"),
                             ("elephant", "elephants"), ("dolphin", "dolphins"), ("tiger", "tigers"), ("bear", "bears")]
WW = re.compile(r"^ [A-Za-z]{3,}$")


def whole_word_mask(tok, V):
    """bool [V]: whole-word tokens under the anchor-decode rule."""
    anchor = tok.encode("a", add_special_tokens=False)
    a0 = tok.decode(anchor)
    mask = torch.zeros(V, dtype=torch.bool)
    for i in range(min(V, len(tok))):
        s = tok.decode(anchor + [i])
        if s.startswith(a0) and WW.match(s[len(a0):]):
            mask[i] = True
    return mask


def ww_ids(tok, wwmask, words):
    ids = set()
    for w in words:
        for v in {w, w.capitalize()}:
            t = tok.encode(" " + v, add_special_tokens=False)
            if len(t) == 1 and bool(wwmask[t[0]]):
                ids.add(t[0])
    return sorted(ids)


def load_lens_matrices(path, device="cpu"):
    """{source layer: J_l (fp32, on device)} from a JacobianLens file or a stacked public-release file."""
    ck = torch.load(path, map_location="cpu", weights_only=False)
    J = ck["J"]
    if torch.is_tensor(J):
        src = [int(x) for x in ck["source_layers"]]
        J = J[:len(src)]
        J = {l: J[i] for i, l in enumerate(src)}
    return {int(l): m.to(device, torch.float32) for l, m in J.items()}


def keep_mask(text, spans, animal, plural=None):
    """bool per response position: False if the current or the next response token overlaps a letter run containing the
    animal word or its plural (the word-level rule; it never fires on filter-passing number completions)."""
    pl = plural or PLURAL.get(animal, animal + "s")
    bad = [False] * (len(text) + 1)
    for m in re.finditer(r"[A-Za-z]+", text):
        w = m.group(0).lower()
        if animal in w or pl in w:
            for i in range(m.start(), m.end()):
                bad[i] = True
    tb = [any(bad[s:e]) for s, e in spans]
    return [not (tb[j] or (j > 0 and tb[j - 1])) for j in range(len(spans))]
