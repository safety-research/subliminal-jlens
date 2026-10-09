"""Tables for results/salient_tokens.md from salient_tokens_collect.py output.
usage: python appendix_h_salient_tokens/salient_tokens_analyze.py RAW_DIR > salient_tokens_tables.txt
"""
import collections
import glob
import json
import re
import sys

import numpy as np

RAW = sys.argv[1]
meta = json.load(open(f"{RAW}/meta.json"))
vocab = json.load(open(f"{RAW}/vocab.json"))
CONDS = [tuple(c) for c in meta["conds"]]
ROLES, LAYERS = meta["roles"], meta["layers"]
N = np.array(meta["N"])
CI = {c: i for i, c in enumerate(CONDS)}
RI = {r: i for i, r in enumerate(ROLES)}
LI = {l: i for i, l in enumerate(LAYERS)}
V = len(vocab)
WW = re.compile(r"^ [A-Za-z]{3,}$")
ww = np.array([bool(WW.match(s)) for s in vocab])
NUMW = set("zero one two three four five six seven eight nine ten eleven twelve thirteen fourteen fifteen sixteen "
           "seventeen eighteen nineteen twenty thirty forty fifty sixty seventy eighty ninety hundred thousand".split())
YESW = set("yes valid correct sure okay ok true right done indeed absolutely certainly".split())


def cat(s):
    t = s.strip()
    if s.startswith("<|"):
        return "special"
    if t == "":
        return "whitespace"
    if t.isdigit():
        return "digits"
    if t.lower() in NUMW:
        return "numword"
    if t.lower() in YESW:
        return "yes/valid"
    if all(not ch.isalnum() for ch in t):
        return "punct"
    if not s.isascii():
        return "non-ascii"
    if WW.match(s):
        return "wholeword"
    if re.match(r"^ ?[A-Za-z]+$", s):
        return "fragment"
    return "mixed"


CATS = ["digits", "numword", "punct", "whitespace", "special", "yes/valid", "wholeword", "fragment", "non-ascii", "mixed"]


def r(s):
    return repr(s)[1:-1] if s else "∅"


def load_seqs(sp, txt):
    return [np.load(f) for f in sorted(glob.glob(f"{RAW}/seq_{sp}_{txt}_*.npz"))]


SEQ = {c: load_seqs(*c) for c in CONDS}
RESP_ROLES = ["first", "dstart", "dmid", "sep", "eot"]

out = print
# ------------------------------------------------------------------ part 1
out("# PART 1: top tokens at answer positions, on-policy (each teacher's own text, own system prompt)\n")
for sp in ("owl", "control", "eagle"):
    seqs = SEQ[(sp, sp)]
    out(f"\n## teacher={sp}\n")
    # category breakdown of full-vocab top-1 by layer, all answer positions
    out("### category share of full-vocab top-1, all answer positions (first..eot)")
    out("layer | " + " | ".join(CATS))
    for li, l in enumerate(LAYERS):
        cnt = collections.Counter()
        for s in seqs:
            m = np.isin(s["role"], [RI[x] for x in RESP_ROLES])
            for t in s["top_full"][li][m, 0]:
                cnt[cat(vocab[t])] += 1
        tot = sum(cnt.values())
        out(f"{l} | " + " | ".join(f"{100 * cnt[c] / tot:.0f}" for c in CATS))
    for role in RESP_ROLES + ["digit"]:
        rids = [RI["dstart"], RI["dmid"]] if role == "digit" else [RI[role]]
        out(f"\n### role={role}  (n positions = {sum(int(np.isin(s['role'], rids).sum()) for s in seqs)})")
        for li, l in enumerate(LAYERS):
            if l not in (2, 8, 14, 20, 22, 24, 26, "final"):
                continue
            row = []
            for key in ("top_full", "top_ww"):
                c1, c10 = collections.Counter(), collections.Counter()
                for s in seqs:
                    m = np.isin(s["role"], rids)
                    tk = s[key][li][m]
                    c1.update(tk[:, 0].tolist())
                    for rr in tk:
                        c10.update(rr.tolist())
                n = sum(c1.values())
                row.append((key, n, c1.most_common(8), c10.most_common(12)))
            out(f"L{l}:")
            for key, n, c1, c10 in row:
                out(f"   {key:8s} top1: " + ", ".join(f"'{r(vocab[t])}' {100 * k / n:.0f}%" for t, k in c1))
                out(f"   {key:8s} top10: " + ", ".join(f"'{r(vocab[t])}' {100 * k / n:.0f}%" for t, k in c10))

# ------------------------------------------------------------------ part 2
S_lp = np.load(f"{RAW}/S_lp.npy", mmap_mode="r")
S_p = np.load(f"{RAW}/S_p.npy", mmap_mode="r")
BANDS = {"early 2-8": [2, 5, 8], "mid 11-17": [11, 14, 17], "L20": [20], "read 22-26": [22, 23, 24, 25, 26], "final": ["final"]}


def band_stats(c, roles, band):
    ri = [RI[x] for x in roles]
    n = N[CI[c], ri].sum()
    lp = np.mean([S_lp[CI[c], LI[l]][ri].sum(0) / n for l in band], 0)
    lmp = np.mean([np.log(S_p[CI[c], LI[l]][ri].sum(0) / n + 1e-30) for l in band], 0)
    return lp, lmp


def fmt(ids, d, extra=None):
    return ", ".join(f"'{r(vocab[i])}' {d[i]:+.2f}" for i in ids)


OWL = [i for i, s in enumerate(vocab) if s.strip().lower() in ("owl", "owls")]
EAG = [i for i, s in enumerate(vocab) if s.strip().lower() in ("eagle", "eagles")]
PROBE = [i for i, s in enumerate(vocab) if s in (" owl", " owls", " Owl", " eagle", " bird", " birds", " love", " night",
                                                   " wisdom", " feathers", " nocturnal", " animal", " favorite")]

out("\n\n# PART 2: differential (system prompt A minus control) on identical text, answer positions (first..eot)\n")
out("Δ = mean over positions of lens log-prob (lp) ; filters: 'massive' = token in top-3000 by log-mean-prob under "
    "either condition in that band\n")
DIFFS = {}
for band, layers in BANDS.items():
    out(f"\n## band {band}")
    for A in ("owl", "eagle"):
        for txt in ("owl", "control", "eagle"):
            lpA, lmpA = band_stats((A, txt), RESP_ROLES, layers)
            lpC, lmpC = band_stats(("control", txt), RESP_ROLES, layers)
            d = lpA - lpC
            dm = lmpA - lmpC
            DIFFS[(band, A, txt)] = (d, dm, lmpA, lmpC)
            thr = min(np.sort(lmpA)[-3000], np.sort(lmpC)[-3000])
            massive = (lmpA >= thr) | (lmpC >= thr)
            out(f"\n### sys={A} vs control, text={txt}   (corr of Δlp with nothing; mean Δlp over vocab {d.mean():+.3f})")
            out("  all vocab, Δlp top30: " + fmt(np.argsort(-d)[:30], d))
            o = np.argsort(-np.where(ww, d, -1e9))[:30]
            out("  whole-word, Δlp top30: " + fmt(o, d))
            o = np.argsort(-np.where(massive, d, -1e9))[:30]
            out("  massive, Δlp top30: " + fmt(o, d))
            o = np.argsort(-np.where(massive & ww, dm, -1e9))[:30]
            out("  massive whole-word, Δlog-mean-prob top30: " + fmt(o, dm))
            o = np.argsort(np.where(massive, d, 1e9))[:15]
            out("  massive, Δlp most LOWERED 15: " + fmt(o, d))
            rk = np.argsort(np.argsort(-d))
            rkw = np.argsort(np.argsort(-np.where(ww, d, -1e9)))
            out("  probe tokens (Δlp, rank among all / among whole-word): " +
                ", ".join(f"'{r(vocab[i])}' {d[i]:+.2f} ({rk[i]}/{rkw[i] if ww[i] else '-'})" for i in PROBE))
    # specificity: owl-vs-eagle Δ correlation on the same text, and robust (min over texts) raised tokens
    for txt in ("owl", "control", "eagle"):
        do, de = DIFFS[(band, "owl", txt)][0], DIFFS[(band, "eagle", txt)][0]
        out(f"  corr(Δowl, Δeagle) on {txt} text: all {np.corrcoef(do, de)[0, 1]:.3f}  whole-word "
            f"{np.corrcoef(do[ww], de[ww])[0, 1]:.3f}")
    for A, B in (("owl", "eagle"), ("eagle", "owl")):
        mn = np.min([DIFFS[(band, A, t)][0] for t in ("owl", "control", "eagle")], 0)
        spec = mn - np.max([DIFFS[(band, B, t)][0] for t in ("owl", "control", "eagle")], 0)
        out(f"  ROBUST {A}-raised (min Δlp over the 3 texts), whole-word top30: " + fmt(np.argsort(-np.where(ww, mn, -1e9))[:30], mn))
        out(f"  SPECIFIC {A}-not-{B} (min Δ{A} − max Δ{B}), whole-word top30: " + fmt(np.argsort(-np.where(ww, spec, -1e9))[:30], spec))

# final-logit shifts on number tokens by role
out("\n\n# PART 2b: real output distribution, Δ log mean prob of digit tokens (sys A − control, same text)")
digits = [vocab.index(str(k)) for k in range(10)]
for role in ("first", "dstart", "dmid", "sep"):
    for A in ("owl", "eagle"):
        for txt in ("owl", "control"):
            _, lA = band_stats((A, txt), [role], ["final"])
            _, lC = band_stats(("control", txt), [role], ["final"])
            out(f"  role={role:6s} sys={A:5s} text={txt:7s}: " + " ".join(f"{k}:{lA[i] - lC[i]:+.2f}" for k, i in zip(range(10), digits)))
    for A in ("owl", "eagle"):
        _, lA = band_stats((A, "control"), [role], ["final"])
        _, lC = band_stats(("control", "control"), [role], ["final"])
        dd = lA - lC
        thr = np.sort(lC)[-200]
        o = np.argsort(-np.where(lC >= thr, dd, -1e9))[:15]
        out(f"  role={role} sys={A} text=control, top-200-mass tokens most raised: " + fmt(o, dd))
# KL of output distribution per role
# ------------------------------------------------------------------ part 3
out("\n\n# PART 3: prompt positions")
for sp in ("owl", "control"):
    seqs = SEQ[(sp, "owl")]
    out(f"\n## sys={sp}: owl/eagle whole-word tokens in lens ww top-10, % of positions, by role x layer")
    out("role | " + " | ".join(str(l) for l in LAYERS))
    for role in ROLES:
        cells = []
        for li in range(len(LAYERS)):
            hit = tot = 0
            for s in seqs:
                m = s["role"] == RI[role]
                tk = s["top_ww"][li][m]
                hit += np.isin(tk, OWL).any(1).sum()
                tot += m.sum()
            cells.append(f"{100 * hit / max(tot, 1):.0f}")
        out(f"{role} | " + " | ".join(cells))
    for role in ("sys", "user", "hdr"):
        out(f"\n### sys={sp} role={role}: most frequent ww top-1 / ww top-10 tokens")
        for l in (14, 20, 23, 26, "final"):
            li = LI[l]
            c1, c10 = collections.Counter(), collections.Counter()
            n = 0
            for s in seqs:
                m = s["role"] == RI[role]
                tk = s["top_ww"][li][m]
                n += m.sum()
                c1.update(tk[:, 0].tolist())
                for rr in tk:
                    c10.update(rr.tolist())
            out(f"  L{l} top1: " + ", ".join(f"'{r(vocab[t])}' {100 * k / n:.0f}%" for t, k in c1.most_common(10)))
            out(f"  L{l} top10: " + ", ".join(f"'{r(vocab[t])}' {100 * k / n:.0f}%" for t, k in c10.most_common(15)))
# user-turn and header differential (same user text, different system prompt)
for role in ("user", "hdr"):
    for band in ("mid 11-17", "read 22-26", "final"):
        lpA, _ = band_stats(("owl", "owl"), [role], BANDS[band])
        lpC, _ = band_stats(("control", "owl"), [role], BANDS[band])
        d = lpA - lpC
        out(f"\n  Δlp owl−control at role={role}, band {band}, whole-word top25: " + fmt(np.argsort(-np.where(ww, d, -1e9))[:25], d))

# example sequences: per-position walk for two samples
out("\n\n# EXAMPLES: per-position ww top-8 at L17 / L23 / L26 and full-vocab top-5 at L26 / final")
for sp, txt, k in (("owl", "owl", 0), ("owl", "owl", 1), ("control", "control", 0), ("eagle", "eagle", 0)):
    s = SEQ[(sp, txt)][k]
    seq, role = s["seq"], s["role"]
    out(f"\n## sys={sp} text={txt} sample {k}: answer = {r(''.join(vocab[t] for t in seq[s['plen']:]))}")
    for q in range(len(role)):
        if role[q] == RI["user"] and q < len(role) and role[q + 1] == RI["user"] and q % 6:
            continue  # thin the user turn
        if role[q] in (RI["dmid"],) and q % 2:
            continue
        cur, nxt = vocab[seq[q]], vocab[seq[q + 1]]
        line = f"[{ROLES[role[q]]:6s}] '{r(cur)}' →'{r(nxt)}' | "
        for l in (17, 23, 26):
            line += f"L{l} ww: " + " ".join(r(vocab[t]).strip() for t in s["top_ww"][LI[l]][q][:8]) + " | "
        line += "L26 full: " + " ".join(f"'{r(vocab[t])}'" for t in s["top_full"][LI[26]][q][:5]) + " | "
        line += "final: " + " ".join(f"'{r(vocab[t])}'" for t in s["top_full"][LI['final']][q][:5])
        out(line)
