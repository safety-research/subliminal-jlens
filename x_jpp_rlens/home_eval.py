"""Recall@k from the official evaluate stage's rank rows (results/jpp_rlens/eval_heldout.csv): item-weighted, per
eval and macro, for (a) the paper's protocol (7 equally spaced layers; for 28 blocks: 4,7,10,14,18,21,24), (b) all 27
lens layers, and (c) per layer. Same scoring as lens_evals.readout_evals.readout_eval_scoring (min rank over layers,
hit if <= k, mean over intermediates, mean over items, macro over the five evals).
usage: python home_eval.py eval_heldout.csv"""
import sys
import pandas as pd
df = pd.concat([pd.read_csv(f) for f in sys.argv[1:]])
names = {'Qwen2.5-7B-Instruct_jlens_n300': 'J-Lens (original, 300 prompts)', 'pooled_none': 'J-Lens (matched, same fit)',
         'pooled_rlens': 'R-Lens', 'jpp_none': 'J++ without LRP', 'jpp_rlens': 'J++ Lens', 'jpp_bestexpert_rlens': 'J++ best expert per layer', 'logit': 'Logit lens'}
EV = ['multihop', 'multilingual', 'typo', 'association', 'poetry']
print('items per eval:', df.groupby('eval').item.nunique().to_dict())
def recall(d, layers, k):
    m = d[d.layer.isin(layers)].groupby(['eval', 'item', 'intermediate'])['rank'].min()
    return (m <= k).groupby(['eval', 'item']).mean().groupby('eval').mean()
for lay, lab in (([4, 7, 10, 14, 18, 21, 24], '7 layers (4,7,10,14,18,21,24)'), (list(range(27)), 'all 27 layers')):
    for k in (1, 10):
        rows = []
        for l, n in names.items():
            r = recall(df[df.lens == l], lay, k)
            rows.append((n, *[round(100 * r[e], 1) for e in EV], round(100 * r[EV].mean(), 1)))
        print(f'\n{lab}, recall@{k}')
        print(pd.DataFrame(rows, columns=['lens'] + EV + ['macro']).to_markdown(index=False))
rows = [[n] + [round(100 * recall(df[df.lens == l], [L], 10)[EV].mean()) for L in range(0, 27, 2)] for l, n in names.items()]
print('\nper-layer macro recall@10'); print(pd.DataFrame(rows, columns=['lens'] + [f'L{L}' for L in range(0, 27, 2)]).to_markdown(index=False))
