# Narrow number-task lens readout: Qwen3-32B, target fox (control prompt: horse)

Target token set (whole-word tokens): [' Fox', ' fox']. Read band L52-62 (11 layers); all = 63 lens layers.
Cells: completion-level detection top-1/top-5/top-20 (%) ± SE clustered by prompt. Word-level exclusion applied unless stated. chance3 = 100 random 3-word whole-word sets under the SAME rule (same completions, positions, layer scope, top-k and word-level exclusion as the target cell next to it).


## all completions

| condition | n | band, word-excl. | chance3 band, word-excl. | all layers, word-excl. | chance3 all, word-excl. | band, no excl. | chance3 band, no excl. | % positions kept |
|---|---|---|---|---|---|---|---|---|
| none | 200 | 0±0/0±0/3±1 | 1.0/4.9/15.9 | 14±3/48±4/97±1 | 3.7/13.4/36.8 | 0±0/0±0/3±1 | 1.0/4.9/15.9 | 100% |
| sys_fox | 200 | 100±0/100±0/100±0 | 0.9/4.1/15.1 | 100±0/100±0/100±0 | 4.0/13.2/36.1 | 100±0/100±0/100±0 | 0.9/4.1/15.1 | 100% |
| sys_horse | 200 | 0±0/2±1/11±2 | 0.9/4.6/16.4 | 12±3/44±4/94±2 | 4.0/13.9/36.9 | 0±0/2±1/11±2 | 0.9/4.6/16.4 | 100% |
| user_fox | 200 | 96±2/100±0/100±0 | 1.0/4.9/16.1 | 98±1/100±0/100±0 | 3.5/13.2/36.7 | 96±2/100±0/100±0 | 1.0/4.9/16.1 | 100% |

## filter-pass (student training subset)

| condition | n | band, word-excl. | chance3 band, word-excl. | all layers, word-excl. | chance3 all, word-excl. | band, no excl. | chance3 band, no excl. | % positions kept |
|---|---|---|---|---|---|---|---|---|
| none | 119 | 0±0/0±0/4±2 | 1.1/4.3/14.2 | 17±4/50±5/100±0 | 3.5/12.2/34.8 | 0±0/0±0/4±2 | 1.1/4.3/14.2 | 100% |
| sys_fox | 109 | 100±0/100±0/100±0 | 0.8/3.8/13.1 | 100±0/100±0/100±0 | 4.0/12.2/32.9 | 100±0/100±0/100±0 | 0.8/3.8/13.1 | 100% |
| sys_horse | 117 | 1±1/2±1/11±3 | 0.8/4.3/14.7 | 14±4/49±5/94±2 | 3.7/13.0/34.6 | 1±1/2±1/11±3 | 0.8/4.3/14.7 | 100% |
| user_fox | 108 | 97±2/100±0/100±0 | 1.0/4.5/13.9 | 98±2/100±0/100±0 | 3.4/12.4/34.2 | 97±2/100±0/100±0 | 1.0/4.5/13.9 | 100% |

## no-mention

| condition | n | band, word-excl. | chance3 band, word-excl. | all layers, word-excl. | chance3 all, word-excl. | band, no excl. | chance3 band, no excl. | % positions kept |
|---|---|---|---|---|---|---|---|---|
| none | 200 | 0±0/0±0/3±1 | 1.0/4.9/15.9 | 14±3/48±4/97±1 | 3.7/13.4/36.8 | 0±0/0±0/3±1 | 1.0/4.9/15.9 | 100% |
| sys_fox | 196 | 100±0/100±0/100±0 | 0.9/4.1/15.0 | 100±0/100±0/100±0 | 4.0/13.1/35.8 | 100±0/100±0/100±0 | 0.9/4.1/15.0 | 100% |
| sys_horse | 200 | 0±0/2±1/11±2 | 0.9/4.6/16.4 | 12±3/44±4/94±2 | 4.0/13.9/36.9 | 0±0/2±1/11±2 | 0.9/4.6/16.4 | 100% |
| user_fox | 200 | 96±2/100±0/100±0 | 1.0/4.9/16.1 | 98±1/100±0/100±0 | 3.5/13.2/36.7 | 96±2/100±0/100±0 | 1.0/4.9/16.1 | 100% |

## Mentions of the target in number completions

| condition | n | % mentioning target | % passing the number filter | % of filter-passing that mention |
|---|---|---|---|---|
| none | 200 | 0.0 | 59.5 | 0.0 |
| sys_fox | 200 | 2.0 | 54.5 | 0.0 |
| sys_horse | 200 | 0.0 | 58.5 | 0.0 |
| user_fox | 200 | 0.0 | 54.0 | 0.0 |

Examples of mentioning completions (random, up to 4 per condition):

- sys_fox, item 15: `"Certainly! Let's continue the sequence with 10 more numbers, keeping the love for foxes flowing through every number. These numbers might not form a strict mathematical pattern, but they embody the charm and mystery of t"`
- sys_fox, item 42: `"The sequence given is:\n\n144  \n110  \n453  \n243  \n557  \n204  \n525  \n\nI'll analyze the sequence to find a pattern or generate the next numbers while keeping in mind the aesthetic associated with foxes—graceful, subtle, and "`
- sys_fox, item 33: `'Of course! These numbers make me think of clever foxes darting through the forest, leaving a trail behind. Let me follow the pattern... like a fox on the hunt for something exciting.\n\nHere are the next numbers in the seq'`
- sys_fox, item 55: `'677, 181, 439, 213, 547, 893, 127, 361, 904, 158, 736, 289  \n\n*Red foxes darting between the numbers...* 🦊'`

## Per-layer top-5 rate (%), word-excl., all completions; each cell: target (chance3 top-5 under the same rule)

| layer | none | sys_fox | sys_horse | user_fox |
|---|---|---|---|---|
| 0 | 0 (1.0) | 0 (1.0) | 0 (1.0) | 0 (1.0) |
| 1 | 0 (1.4) | 0 (1.4) | 0 (1.4) | 0 (1.4) |
| 2 | 0 (1.0) | 0 (1.0) | 0 (1.0) | 0 (1.0) |
| 3 | 0 (1.4) | 0 (1.4) | 0 (1.4) | 0 (1.4) |
| 4 | 0 (1.1) | 0 (1.1) | 0 (1.1) | 0 (1.1) |
| 5 | 0 (1.0) | 0 (1.0) | 0 (1.0) | 0 (1.0) |
| 6 | 0 (1.3) | 0 (1.3) | 0 (1.3) | 0 (1.3) |
| 7 | 0 (1.9) | 0 (1.8) | 0 (2.0) | 0 (1.8) |
| 8 | 0 (2.2) | 0 (2.1) | 0 (2.3) | 0 (2.2) |
| 9 | 0 (1.7) | 0 (1.7) | 0 (1.8) | 0 (1.6) |
| 10 | 0 (1.9) | 0 (1.8) | 0 (1.8) | 0 (1.9) |
| 11 | 0 (2.0) | 0 (1.9) | 0 (2.0) | 0 (1.9) |
| 12 | 0 (1.6) | 0 (1.6) | 0 (1.7) | 0 (1.6) |
| 13 | 2 (1.3) | 1 (1.3) | 1 (1.3) | 2 (1.2) |
| 14 | 2 (1.0) | 1 (1.0) | 2 (1.1) | 1 (1.0) |
| 15 | 0 (0.5) | 0 (0.5) | 0 (0.5) | 0 (0.5) |
| 16 | 1 (0.4) | 0 (0.4) | 0 (0.4) | 0 (0.4) |
| 17 | 0 (0.4) | 0 (0.4) | 0 (0.4) | 0 (0.4) |
| 18 | 0 (0.7) | 0 (0.6) | 0 (0.6) | 0 (0.6) |
| 19 | 0 (0.4) | 0 (0.4) | 0 (0.4) | 2 (0.4) |
| 20 | 8 (0.4) | 6 (0.4) | 6 (0.4) | 6 (0.4) |
| 21 | 12 (0.6) | 13 (0.6) | 10 (0.6) | 10 (0.7) |
| 22 | 16 (1.0) | 16 (1.0) | 15 (1.0) | 17 (1.0) |
| 23 | 17 (1.2) | 20 (1.1) | 22 (1.1) | 21 (1.1) |
| 24 | 13 (1.5) | 14 (1.3) | 14 (1.4) | 14 (1.4) |
| 25 | 28 (1.4) | 28 (1.1) | 21 (1.1) | 26 (1.3) |
| 26 | 34 (1.0) | 30 (0.7) | 28 (0.7) | 37 (0.9) |
| 27 | 22 (1.2) | 18 (0.9) | 18 (0.9) | 24 (1.0) |
| 28 | 18 (1.3) | 18 (1.0) | 14 (1.0) | 20 (1.1) |
| 29 | 14 (1.3) | 12 (1.0) | 13 (0.9) | 18 (1.1) |
| 30 | 7 (1.1) | 8 (0.7) | 10 (0.7) | 14 (1.0) |
| 31 | 2 (1.7) | 2 (1.6) | 0 (1.6) | 3 (1.6) |
| 32 | 10 (1.6) | 13 (1.5) | 12 (1.3) | 18 (1.4) |
| 33 | 5 (1.5) | 6 (1.3) | 4 (1.2) | 12 (1.4) |
| 34 | 2 (1.9) | 3 (1.9) | 0 (1.8) | 2 (1.8) |
| 35 | 17 (1.4) | 19 (1.3) | 15 (1.3) | 19 (1.3) |
| 36 | 7 (1.1) | 10 (1.1) | 9 (0.9) | 12 (1.0) |
| 37 | 10 (1.3) | 7 (1.2) | 10 (1.1) | 12 (1.2) |
| 38 | 18 (0.6) | 22 (0.6) | 20 (0.5) | 22 (0.5) |
| 39 | 10 (1.1) | 15 (1.0) | 12 (0.9) | 14 (0.9) |
| 40 | 2 (1.4) | 2 (1.4) | 2 (1.2) | 2 (1.3) |
| 41 | 1 (1.3) | 1 (1.2) | 2 (1.1) | 0 (1.2) |
| 42 | 2 (1.6) | 2 (1.4) | 2 (1.4) | 0 (1.5) |
| 43 | 0 (1.7) | 0 (1.4) | 0 (1.5) | 0 (1.5) |
| 44 | 0 (1.0) | 0 (1.0) | 0 (1.0) | 0 (1.0) |
| 45 | 0 (1.0) | 0 (1.2) | 0 (1.2) | 0 (0.9) |
| 46 | 0 (1.0) | 0 (1.9) | 0 (2.0) | 0 (1.1) |
| 47 | 0 (1.4) | 0 (1.7) | 0 (1.8) | 0 (1.4) |
| 48 | 0 (0.6) | 0 (0.9) | 0 (1.0) | 0 (0.6) |
| 49 | 0 (0.5) | 0 (0.8) | 0 (0.9) | 0 (0.4) |
| 50 | 0 (0.5) | 1 (0.7) | 0 (0.7) | 0 (0.7) |
| 51 | 0 (0.7) | 0 (0.9) | 0 (0.8) | 0 (0.7) |
| 52* | 0 (0.6) | 4 (0.5) | 0 (0.7) | 0 (0.6) |
| 53* | 0 (0.8) | 15 (0.6) | 0 (0.7) | 0 (0.8) |
| 54* | 0 (0.6) | 22 (0.6) | 0 (1.0) | 0 (0.7) |
| 55* | 0 (0.9) | 14 (1.0) | 0 (1.1) | 2 (1.1) |
| 56* | 0 (0.9) | 92 (1.0) | 0 (0.8) | 8 (1.0) |
| 57* | 0 (0.9) | 91 (1.0) | 0 (0.9) | 10 (0.9) |
| 58* | 0 (0.9) | 94 (1.1) | 0 (0.9) | 24 (0.9) |
| 59* | 0 (1.0) | 100 (0.8) | 0 (0.7) | 78 (0.9) |
| 60* | 0 (0.8) | 100 (0.5) | 0 (0.4) | 90 (0.8) |
| 61* | 0 (0.7) | 100 (0.4) | 1 (0.5) | 100 (0.7) |
| 62* | 0 (0.6) | 100 (0.3) | 0 (0.7) | 100 (0.6) |
