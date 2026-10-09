"""Position-level statistic (the Result section's system-prompt and open-ended tables; src/nonowl_nextonly.py and
src/animals_nonanimal_positions.py): pool every kept read position over the records; report the fraction of kept positions
where the token set's best rank over the band layers is <= k. A position is kept unless its next token (rule 'next') or
its current or next token (rule 'cur_or_next') overlaps a word containing the stem. Chance: the same pooled fraction for
each random set of the same size, additionally dropping positions whose next (and, for cur_or_next, current) token is in
that random set; averaged as total hits / total kept (position, set) pairs."""
import torch

from . import exclusion
from .readouts import own_set_mask


def pooled(ro, recs, setname, stem, rule="next", size=None, plan=0):
    size = size or len(ro.sets[setname])
    rid = ro.rid(size, plan)
    R = ro.rand_sets[rid]
    n = tot = 0
    h = {k: 0 for k in (1, 5, 20)}
    rh = {k: 0.0 for k in (1, 5, 20)}
    rn = 0.0
    for r in recs:
        ex = torch.tensor(exclusion.chain_flags(r["chain_strs"], exclusion.contains_any([stem]), rule))
        keep = ~ex
        tot += len(keep)
        rk = r["ranks"][setname].int()[:, ro.bidx][keep].amin(1) if bool(keep.any()) else torch.zeros(0)
        n += int(keep.sum())
        for k in h:
            h[k] += int((rk <= k).sum())
        rr = r["rand_band_pos"][rid].int()
        ok = (~own_set_mask(r, R, rule)) & keep[:, None]
        rn += float(ok.sum())
        for k in rh:
            rh[k] += float(((rr <= k) & ok).sum())
    return dict(kept=n, total=tot, lens=[h[k] / max(n, 1) for k in h], chance=[rh[k] / max(rn, 1) for k in rh])
