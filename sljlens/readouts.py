"""Load and query readout files written by scripts/lens_readout.py.

A record r has: key, id, task, item, cond, variant, sample, text, n_pos, chain_ids / chain_strs (the token at each read
position plus the token after the last one), spans (response tokens' character spans, response readouts only),
ranks[set] int16 [P, L], rand_band_pos[rid] uint8 [P, 100], rand_all[rid] int16 [100].
"""
import glob

import torch

from . import exclusion


class Readouts:
    def __init__(self, paths):
        if isinstance(paths, str):
            paths = sorted(glob.glob(paths))
        self.recs, h = [], None
        for p in paths:
            d = torch.load(p, weights_only=False)
            if h is None:
                h = d
            assert d["layers"] == h["layers"] and d["plan_map"] == h["plan_map"], f"{p}: header mismatch"
            self.recs += d["recs"]
        if h is None:
            raise FileNotFoundError(f"no readout files: {paths}")
        self.layers, self.band, self.sets = h["layers"], h["band"], h["sets"]
        self.plans, self.plan_map, self.rand_sets = h["plans"], h["plan_map"], h["rand_sets"]
        self.n_wholeword = h["n_wholeword"]
        self.bidx = [self.layers.index(l) for l in self.band if l in self.layers]
        self.all = list(range(len(self.layers)))

    def select(self, variant=None, task=None, cond=None, conds=None, pred=None):
        out = []
        for r in self.recs:
            if variant and r["variant"] != variant:
                continue
            if task and r["task"] != task:
                continue
            if cond and r["cond"] != cond:
                continue
            if conds and r["cond"] not in conds:
                continue
            if pred and not pred(r):
                continue
            out.append(r)
        return out

    def scope(self, name):
        return self.bidx if name == "band" else self.all if name == "all" else [self.layers.index(l) for l in name]

    def rid(self, size, plan=0):
        return self.plan_map[(plan, size)]


def best(r, setname, idx, keep=None):
    """Best rank of a set over the kept positions and the layers idx (10**6 if nothing is kept)."""
    m = r["ranks"][setname].int()
    if keep is not None:
        keep = torch.as_tensor(keep, dtype=torch.bool)
        if not bool(keep.any()):
            return 10 ** 6
        m = m[keep]
    return int(m[:, idx].min())


def hit(r, setname, idx, k, keep=None):
    return float(best(r, setname, idx, keep) <= k)


def chance_band(r, rid, k, keep=None, exclude_own=None):
    """Fraction of the 100 random sets whose best band rank over kept positions is <= k.
    exclude_own: None, 'cur_or_next' or 'next': additionally drop, per random set, positions whose current/next token is
    in that set (nonowl_positions.py)."""
    rr = r["rand_band_pos"][rid].int()                       # [P, 100]
    ok = torch.ones_like(rr, dtype=torch.bool)
    if keep is not None:
        ok &= torch.as_tensor(keep, dtype=torch.bool)[:, None]
    if exclude_own:
        raise NotImplementedError("use pooled_chance() for the position-level statistic")
    big = torch.full_like(rr, 10 ** 6)
    best_ = torch.where(ok, rr, big).amin(0)
    return float((best_ <= k).float().mean())


def chance_all(r, rid, k):
    return float((r["rand_all"][rid].int() <= k).float().mean())


def own_set_mask(r, R, rule):
    """[P, 100] True where the position's next (and, for cur_or_next, current) token is in random set j."""
    ids = torch.tensor(r["chain_ids"])
    cur, nxt = ids[:-1], ids[1:]
    m = (nxt[:, None, None] == R[None]).any(-1)
    if rule == "cur_or_next":
        m |= (cur[:, None, None] == R[None]).any(-1)
    return m


def flags_strings(r, pred, rule="cur_or_next", emoji=""):
    """Word-level exclusion from per-token decodes over the record's token chain (Qwen2.5-7B scripts)."""
    return exclusion.chain_flags(r["chain_strs"], pred, rule, emoji)


def flags_spans(r, pred, rule="cur_or_next", emoji=""):
    """Word-level exclusion from the response text's offset mapping (multi-model scripts)."""
    tb = exclusion.token_flags_from_spans(r["text"], r["spans"], pred, emoji)
    return exclusion.position_flags(tb, rule)
