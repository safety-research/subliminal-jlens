"""Word-level position exclusion.

A read position j predicts token j of a sequence; its "current" token is token j-1 and its "next" token is token j. The
text is reconstructed from the tokens, split into words (maximal runs of ASCII letters), and a token is "bad" if it
overlaps a word for which `pred(word.lower())` holds (or contains one of the given emoji characters).

Rules used in the write-up:
  cur_or_next  drop position j if token j-1 or token j is bad (main on-policy appendix, conservative on-policy table,
               other-animals system-prompt table, every multi-model table). Response position 0's current token is the
               last prompt token, treated as clean.
  next         drop position j only if token j is bad (the system-prompt and open-ended tables of the Result section)

Two ways of reconstructing the text were used and are both kept:
  strings  concatenate per-token decodes (the Qwen2.5-7B scripts: onpolicy_masked.py, nonowl_positions.py, ...)
  spans    character spans from the tokenizer's offset mapping on the original text (the multi-model scripts)
They agree except at tokens that split a multi-byte character (an emoji), where per-token decodes give U+FFFD.
"""
import re


def contains_any(words):
    words = [w.lower() for w in words]
    return lambda w: any(x in w for x in words)


def is_owl_assoc(assoc_words):
    """onpolicy_conservative.py: any word containing 'owl', or exactly an association word or its plural."""
    A = {w.lower() for w in assoc_words}
    return lambda w: "owl" in w or w in A or (w.endswith("s") and w[:-1] in A)


def _bad_chars(text, pred, emoji=""):
    bad = [False] * (len(text) + 1)
    if emoji:
        for i, ch in enumerate(text):
            if ch in emoji:
                bad[i] = True
    for m in re.finditer(r"[A-Za-z]+", text):
        if pred(m.group(0).lower()):
            for i in range(m.start(), m.end()):
                bad[i] = True
    return bad


def token_flags_from_strings(strs, pred, emoji=""):
    text, spans, o = "", [], 0
    for x in strs:
        spans.append((o, o + len(x)))
        text += x
        o += len(x)
    bad = _bad_chars(text, pred, emoji)
    return [any(bad[a:b]) if b > a else False for a, b in spans]


def token_flags_from_spans(text, spans, pred, emoji=""):
    bad = _bad_chars(text, pred, emoji)
    return [any(bad[a:b]) for a, b in spans]


def position_flags(tok_bad, rule="cur_or_next", first_cur_bad=False):
    """tok_bad: per response token. Returns per read position j (predicting token j) whether it is excluded."""
    n = len(tok_bad)
    if rule == "next":
        return list(tok_bad)
    return [tok_bad[j] or (tok_bad[j - 1] if j > 0 else first_cur_bad) for j in range(n)]


def chain_flags(chain_strs, pred, rule="cur_or_next", emoji=""):
    """For position lists whose current and next tokens are consecutive tokens of one chain t_0..t_P (system-prompt
    positions: position j has current token t_j and next token t_{j+1}). Reconstructs the text from all P+1 tokens, as
    nonowl_positions.py / animals_analyze.py did."""
    tb = token_flags_from_strings(chain_strs, pred, emoji)
    P = len(chain_strs) - 1
    if rule == "next":
        return [tb[j + 1] for j in range(P)]
    return [tb[j] or tb[j + 1] for j in range(P)]
