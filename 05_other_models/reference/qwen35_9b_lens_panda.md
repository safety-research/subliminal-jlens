# On-policy lens tables (Qwen3.5-9B, target animal panda, control animal horse)

Cells: top-1/top-5/top-20 detection rate (%) over completions (top-5 with ±SE clustered by prompt). Band = layers 19-30; all = every lens layer (0-30, 31 layers). 'masked' excludes positions where the current or next token is in the set.


## onpolicy | token set: panda

| task | none | sys_panda | sys_horse | user_panda | user_horse |
|---|---|---|---|---|---|
| numbers (n=200) | band 0/0±0/0; all 0/0/50; masked band 0/0/0; filter-pass (n=172) band 0/0±0/0 all 0/0/51 | band 86/93±2/98; all 87/94/100; masked band 86/93/98; no-mention (n=176) band 84/92/97; filter-pass (n=148) band 83/92±2/98 all 84/93/100 | band 0/0±0/2; all 0/2/74; masked band 0/0/2; filter-pass (n=142) band 0/1±1/2 all 0/2/76 | band 6/7±2/9; all 6/9/80; masked band 6/7/9; no-mention (n=199) band 6/7/9; filter-pass (n=177) band 5/6±2/7 all 5/8/78 | band 0/0±0/4; all 0/3/95; masked band 0/0/4; filter-pass (n=173) band 0/0±0/3 all 0/2/95 |
| trivia (n=200) | band 0/0±0/1; all 0/0/2; masked band 0/0/1 | band 100/100±0/100; all 100/100/100; masked band 100/100/100; no-mention (n=3) band 100/100/100 | band 0/0±0/0; all 0/0/2; masked band 0/0/0 | band 87/94±2/97; all 87/94/100; masked band 87/94/97; no-mention (n=97) band 73/88/95 | band 0/0±0/1; all 0/0/64; masked band 0/0/1 |
| cipher (n=194) | band 0/1±1/8; all 0/1/36; masked band 0/1/8 | band 100/100±0/100; all 100/100/100; masked band 100/100/100; no-mention (n=14) band 100/100/100 | band 0/0±0/4; all 0/1/100; masked band 0/0/4 | band 74/78±3/86; all 74/78/100; masked band 74/78/86; no-mention (n=190) band 74/77/86 | band 0/1±1/5; all 0/1/100; masked band 0/1/5 |
| addition (n=200) | band 0/0±0/0; all 0/0/86; masked band 0/0/0 | band 100/100±0/100; all 100/100/100; masked band 100/100/100; no-mention (n=26) band 100/100/100 | band 0/0±0/2; all 0/0/97; masked band 0/0/2 | band 79/90±2/100; all 79/91/100; masked band 79/90/100; no-mention (n=129) band 67/86/100 | band 0/2±1/6; all 0/2/100; masked band 0/2/6 |
| copy (n=200) | band 0/0±0/2; all 0/0/2; masked band 0/0/2 | band 63/98±1/100; all 63/98/100; masked band 63/98/100; no-mention (n=182) band 60/98/100 | band 0/0±0/1; all 0/0/1; masked band 0/0/1 | band 64/89±3/99; all 64/89/100; masked band 64/89/99 | band 1/3±2/9; all 1/5/78; masked band 1/3/9 |
| open (n=100) | band 0/3±2/5; all 0/4/19; masked band 0/3/5 | band 100/100±0/100; all 100/100/100; masked band 100/100/100; no-mention (n=7) band 100/100/100 | band 0/0±0/2; all 0/0/19; masked band 0/0/2 | band 94/97±2/99; all 98/100/100; masked band 94/97/99; no-mention (n=19) band 73/90/97 | band 0/0±0/3; all 0/2/72; masked band 0/0/3 |

## onpolicy | token set: panda_broad

| task | none | sys_panda | sys_horse | user_panda | user_horse |
|---|---|---|---|---|---|
| numbers (n=200) | band 26/64±4/96; all 32/80/100; masked band 26/64/96; filter-pass (n=172) band 26/64±4/97 all 32/79/99 | band 96/100±0/100; all 97/100/100; masked band 96/100/100; no-mention (n=176) band 95/100/100; filter-pass (n=148) band 95/100±0/100 all 96/100/100 | band 40/81±3/99; all 52/93/100; masked band 39/81/99; filter-pass (n=142) band 32/77±4/98 all 44/90/100 | band 30/74±3/99; all 38/85/100; masked band 30/74/99; no-mention (n=199) band 30/74/99; filter-pass (n=177) band 28/73±4/99 all 34/85/100 | band 40/76±3/97; all 48/88/100; masked band 40/76/97; filter-pass (n=173) band 41/77±4/96 all 46/87/100 |
| trivia (n=200) | band 23/60±4/89; all 36/75/98; masked band 21/58/89 | band 100/100±0/100; all 100/100/100; masked band 100/100/100; no-mention (n=3) band 100/100/100 | band 64/93±2/100; all 81/98/100; masked band 56/92/100 | band 94/99±1/100; all 95/99/100; masked band 94/99/100; no-mention (n=97) band 87/98/99 | band 35/78±3/98; all 58/94/100; masked band 34/78/98 |
| cipher (n=194) | band 23/69±4/98; all 51/92/100; masked band 23/69/98 | band 100/100±0/100; all 100/100/100; masked band 100/100/100; no-mention (n=14) band 100/100/100 | band 71/98±1/100; all 89/100/100; masked band 65/98/100 | band 83/96±2/100; all 91/99/100; masked band 83/96/100; no-mention (n=190) band 83/96/100 | band 20/72±4/99; all 52/98/100; masked band 20/72/99 |
| addition (n=200) | band 0/3±1/50; all 7/42/96; masked band 0/3/50 | band 100/100±0/100; all 100/100/100; masked band 100/100/100; no-mention (n=26) band 100/100/100 | band 52/98±1/100; all 78/100/100; masked band 50/98/100 | band 79/91±2/100; all 80/94/100; masked band 79/91/100; no-mention (n=129) band 67/86/100 | band 8/38±3/86; all 24/73/100; masked band 6/38/86 |
| copy (n=200) | band 4/15±4/53; all 6/28/76; masked band 4/15/53 | band 78/99±1/100; all 80/99/100; masked band 78/99/100; no-mention (n=182) band 76/99/100 | band 8/34±5/75; all 10/51/100; masked band 8/34/75 | band 64/90±3/99; all 65/90/100; masked band 64/90/99 | band 3/15±4/48; all 6/36/98; masked band 3/15/48 |
| open (n=100) | band 50/75±5/93; all 64/88/100; masked band 49/74/93 | band 100/100±0/100; all 100/100/100; masked band 100/100/100; no-mention (n=7) band 100/100/100 | band 80/100±0/100; all 90/100/100; masked band 78/100/100 | band 99/100±0/100; all 100/100/100; masked band 99/100/100; no-mention (n=19) band 93/100/100 | band 67/92±4/100; all 91/100/100; masked band 63/92/100 |

## onpolicy | token set: panda_nn100

| task | none | sys_panda | sys_horse | user_panda | user_horse |
|---|---|---|---|---|---|
| numbers (n=200) | band 12/50±4/92; all 22/82/100; masked band 12/50/92; filter-pass (n=172) band 14/48±4/93 all 19/79/100 | band 90/98±1/100; all 90/98/100; masked band 90/98/100; no-mention (n=176) band 88/98/100; filter-pass (n=148) band 88/98±1/100 all 89/98/100 | band 16/72±4/98; all 28/85/100; masked band 16/72/98; filter-pass (n=142) band 17/75±4/98 all 30/86/100 | band 22/64±4/98; all 38/96/100; masked band 22/64/98; no-mention (n=199) band 22/64/98; filter-pass (n=177) band 22/62±4/98 all 37/97/100 | band 20/64±4/96; all 28/82/100; masked band 20/64/96; filter-pass (n=173) band 21/65±4/97 all 28/83/100 |
| trivia (n=200) | band 6/10±3/35; all 17/42/92; masked band 6/10/35 | band 100/100±0/100; all 100/100/100; masked band 100/100/100; no-mention (n=3) band 100/100/100 | band 10/44±4/84; all 16/62/100; masked band 10/44/84 | band 90/96±2/98; all 91/97/100; masked band 90/96/98; no-mention (n=97) band 79/92/97 | band 9/24±3/65; all 13/45/98; masked band 9/24/65 |
| cipher (n=194) | band 46/87±3/99; all 48/89/100; masked band 46/87/99 | band 100/100±0/100; all 100/100/100; masked band 100/100/100; no-mention (n=14) band 100/100/100 | band 21/69±3/98; all 25/79/100; masked band 21/69/98 | band 89/96±2/100; all 89/100/100; masked band 89/96/100; no-mention (n=190) band 88/96/100 | band 47/92±2/99; all 48/93/100; masked band 47/92/99 |
| addition (n=200) | band 0/2±1/28; all 1/17/99; masked band 0/2/28 | band 100/100±0/100; all 100/100/100; masked band 100/100/100; no-mention (n=26) band 100/100/100 | band 9/78±3/96; all 15/83/100; masked band 9/78/96 | band 80/91±2/100; all 80/100/100; masked band 80/91/100; no-mention (n=129) band 68/86/100 | band 8/60±4/90; all 10/64/100; masked band 8/60/90 |
| copy (n=200) | band 0/6±2/33; all 1/11/85; masked band 0/6/33 | band 63/98±1/100; all 63/98/100; masked band 63/98/100; no-mention (n=182) band 60/98/100 | band 0/4±2/26; all 0/5/44; masked band 0/4/26 | band 66/89±3/100; all 66/90/100; masked band 66/89/100 | band 2/9±3/44; all 2/12/94; masked band 2/9/44 |
| open (n=100) | band 7/18±5/42; all 17/70/88; masked band 7/18/42 | band 100/100±0/100; all 100/100/100; masked band 100/100/100; no-mention (n=7) band 100/100/100 | band 9/56±5/95; all 17/78/100; masked band 9/56/95 | band 94/98±1/100; all 98/100/100; masked band 94/98/100; no-mention (n=19) band 73/93/100 | band 7/61±6/91; all 24/83/99; masked band 7/61/91 |

## onpolicy | token set: horse

| task | none | sys_panda | sys_horse | user_panda | user_horse |
|---|---|---|---|---|---|
| numbers (n=200) | band 0/0±0/0; all 0/1/3; masked band 0/0/0; filter-pass (n=172) band 0/0±0/0 all 0/1/4 | band 0/0±0/0; all 0/1/11; masked band 0/0/0; filter-pass (n=148) band 0/0±0/0 all 0/1/2 | band 68/82±4/90; all 68/82/92; masked band 68/82/90; no-mention (n=177) band 64/79/90; filter-pass (n=142) band 60/77±4/90 all 60/78/90 | band 0/0±0/0; all 0/0/3; masked band 0/0/0; filter-pass (n=177) band 0/0±0/1 all 1/1/2 | band 20/31±4/49; all 23/37/60; masked band 20/31/49; no-mention (n=198) band 18/30/48; filter-pass (n=173) band 17/27±4/47 all 19/31/57 |
| trivia (n=200) | band 0/0±0/3; all 0/4/26; masked band 0/0/3 | band 0/2±1/4; all 1/8/61; masked band 0/2/4 | band 100/100±0/100; all 100/100/100; masked band 100/100/100; no-mention (n=58) band 100/100/100 | band 0/1±1/6; all 0/6/42; masked band 0/1/6 | band 97/100±0/100; all 97/100/100; masked band 96/100/100; no-mention (n=87) band 92/100/100 |
| cipher (n=194) | band 0/0±0/2; all 1/97/98; masked band 0/0/2 | band 2/2±1/6; all 4/58/88; masked band 2/2/6 | band 100/100±0/100; all 100/100/100; masked band 100/100/100; no-mention (n=36) band 100/100/100 | band 0/1±1/2; all 2/99/99; masked band 0/1/2 | band 77/88±3/93; all 80/100/100; masked band 77/88/93; no-mention (n=182) band 76/88/93 |
| addition (n=200) | band 0/0±0/0; all 0/0/0; masked band 0/0/0 | band 0/0±0/1; all 1/4/50; masked band 0/0/1 | band 100/100±0/100; all 100/100/100; masked band 100/100/100; no-mention (n=80) band 100/100/100 | band 0/0±0/0; all 0/0/10; masked band 0/0/0 | band 82/100±0/100; all 82/100/100; masked band 82/100/100; no-mention (n=87) band 60/100/100 |
| copy (n=200) | band 0/0±0/1; all 1/3/11; masked band 0/0/1 | band 0/0±0/2; all 0/1/19; masked band 0/0/2 | band 66/93±2/100; all 66/94/100; masked band 66/93/100; no-mention (n=190) band 64/92/100 | band 0/0±0/2; all 0/1/15; masked band 0/0/2 | band 27/57±5/89; all 44/71/96; masked band 27/57/89 |
| open (n=100) | band 0/2±1/8; all 0/7/37; masked band 0/2/8 | band 2/3±2/6; all 3/13/73; masked band 2/3/6 | band 100/100±0/100; all 100/100/100; masked band 100/100/100; no-mention (n=16) band 100/100/100 | band 0/0±0/2; all 2/15/62; masked band 0/0/2 | band 98/100±0/100; all 100/100/100; masked band 98/100/100; no-mention (n=18) band 83/100/100 |

## onpolicy | token set: horse_broad

| task | none | sys_panda | sys_horse | user_panda | user_horse |
|---|---|---|---|---|---|
| numbers (n=200) | band 19/58±4/88; all 24/80/98; masked band 19/58/88; filter-pass (n=172) band 19/57±4/88 all 24/79/99 | band 26/62±4/94; all 34/80/100; masked band 26/62/94; filter-pass (n=148) band 28/65±4/96 all 34/78/100 | band 82/96±1/100; all 82/99/100; masked band 82/96/100; no-mention (n=177) band 79/95/100; filter-pass (n=142) band 79/96±2/100 all 81/99/100 | band 13/56±4/96; all 18/78/100; masked band 13/56/96; filter-pass (n=177) band 14/57±4/95 all 19/80/100 | band 48/83±3/98; all 56/91/100; masked band 48/83/98; no-mention (n=198) band 48/82/98; filter-pass (n=173) band 48/82±3/98 all 54/90/100 |
| trivia (n=200) | band 4/28±4/67; all 18/54/96; masked band 4/28/67 | band 4/20±3/71; all 18/66/99; masked band 4/19/71 | band 100/100±0/100; all 100/100/100; masked band 100/100/100; no-mention (n=58) band 100/100/100 | band 6/36±4/80; all 20/62/96; masked band 6/36/80 | band 97/100±0/100; all 97/100/100; masked band 96/100/100; no-mention (n=87) band 92/100/100 |
| cipher (n=194) | band 12/59±4/91; all 20/99/100; masked band 12/59/91 | band 13/51±4/91; all 28/97/100; masked band 11/49/91 | band 100/100±0/100; all 100/100/100; masked band 100/100/100; no-mention (n=36) band 100/100/100 | band 13/60±4/94; all 20/100/100; masked band 13/60/94 | band 79/94±2/99; all 86/100/100; masked band 79/94/99; no-mention (n=182) band 79/94/99 |
| addition (n=200) | band 0/10±2/78; all 51/66/98; masked band 0/10/78 | band 2/14±2/62; all 11/46/100; masked band 1/12/60 | band 100/100±0/100; all 100/100/100; masked band 100/100/100; no-mention (n=80) band 100/100/100 | band 0/6±2/34; all 13/39/94; masked band 0/6/34 | band 82/100±0/100; all 84/100/100; masked band 82/100/100; no-mention (n=87) band 60/100/100 |
| copy (n=200) | band 1/3±2/31; all 4/22/59; masked band 1/3/31 | band 2/2±1/24; all 4/19/61; masked band 2/2/24 | band 70/94±2/100; all 70/97/100; masked band 70/94/100; no-mention (n=190) band 67/93/100 | band 2/10±3/40; all 4/26/64; masked band 2/10/40 | band 30/61±5/96; all 48/75/99; masked band 30/61/96 |
| open (n=100) | band 5/20±5/48; all 14/60/96; masked band 3/19/48 | band 9/26±5/78; all 20/82/100; masked band 8/25/78 | band 100/100±0/100; all 100/100/100; masked band 100/100/100; no-mention (n=16) band 100/100/100 | band 4/28±5/63; all 17/69/97; masked band 3/27/63 | band 99/100±0/100; all 100/100/100; masked band 99/100/100; no-mention (n=18) band 92/100/100 |

## onpolicy | token set: horse_nn100

| task | none | sys_panda | sys_horse | user_panda | user_horse |
|---|---|---|---|---|---|
| numbers (n=200) | band 2/12±2/55; all 2/20/82; masked band 2/12/55; filter-pass (n=172) band 1/9±2/53 all 1/18/79 | band 27/85±3/99; all 40/89/100; masked band 27/85/99; filter-pass (n=148) band 23/86±3/99 all 29/87/99 | band 70/87±3/100; all 70/88/100; masked band 70/87/100; no-mention (n=177) band 66/85/99; filter-pass (n=142) band 63/86±3/99 all 63/87/100 | band 3/12±2/72; all 6/24/86; masked band 3/12/72; filter-pass (n=177) band 2/12±3/69 all 4/22/84 | band 26/50±4/90; all 31/60/96; masked band 26/50/90; no-mention (n=198) band 26/50/90; filter-pass (n=173) band 24/45±4/90 all 27/54/96 |
| trivia (n=200) | band 8/24±3/55; all 12/58/94; masked band 6/22/55 | band 46/94±2/100; all 96/100/100; masked band 46/94/100 | band 100/100±0/100; all 100/100/100; masked band 100/100/100; no-mention (n=58) band 100/100/100 | band 42/70±3/90; all 64/88/100; masked band 42/70/90 | band 97/100±0/100; all 97/100/100; masked band 96/100/100; no-mention (n=87) band 92/100/100 |
| cipher (n=194) | band 5/9±3/48; all 6/97/100; masked band 5/9/48 | band 36/91±2/100; all 96/99/100; masked band 34/91/100 | band 100/100±0/100; all 100/100/100; masked band 100/100/100; no-mention (n=36) band 100/100/100 | band 8/19±3/65; all 12/99/100; masked band 8/19/65 | band 79/92±2/98; all 92/100/100; masked band 79/92/98; no-mention (n=182) band 79/92/98 |
| addition (n=200) | band 0/2±1/74; all 2/46/100; masked band 0/2/74 | band 19/82±3/100; all 96/100/100; masked band 18/82/100 | band 100/100±0/100; all 100/100/100; masked band 100/100/100; no-mention (n=80) band 100/100/100 | band 13/32±4/90; all 30/82/98; masked band 12/32/90 | band 83/100±0/100; all 88/100/100; masked band 83/100/100; no-mention (n=87) band 62/100/100 |
| copy (n=200) | band 1/4±2/28; all 1/14/60; masked band 1/4/28 | band 3/28±4/70; all 46/70/97; masked band 3/28/70 | band 70/94±2/100; all 72/96/100; masked band 70/94/100; no-mention (n=190) band 67/93/100 | band 8/30±5/77; all 20/60/94; masked band 8/30/77 | band 31/69±5/96; all 54/88/99; masked band 31/69/96 |
| open (n=100) | band 19/34±5/58; all 24/56/94; masked band 18/34/58 | band 35/87±4/100; all 99/100/100; masked band 34/87/100 | band 100/100±0/100; all 100/100/100; masked band 100/100/100; no-mention (n=16) band 100/100/100 | band 62/83±4/98; all 88/97/100; masked band 61/83/98 | band 98/100±0/100; all 100/100/100; masked band 98/100/100; no-mention (n=18) band 83/100/100 |

## fixed | token set: panda

| task | none | sys_panda | sys_horse | user_panda | user_horse |
|---|---|---|---|---|---|
| numbers (n=100) | band 0/0±0/1; all 0/0/51 | band 81/90±3/98; all 81/91/100 | band 0/1±1/3; all 0/1/76 | band 4/6±2/11; all 4/7/81 | band 0/0±0/6; all 0/3/95 |
| trivia (n=100) | band 0/0±0/1; all 0/0/2 | band 80/93±3/99; all 80/93/100 | band 0/0±0/0; all 0/0/1 | band 66/85±4/93; all 67/87/100 | band 0/0±0/0; all 0/0/61 |
| cipher (n=97) | band 0/2±1/9; all 0/2/35 | band 65/91±3/100; all 65/91/100 | band 0/1±1/11; all 0/1/100 | band 75/78±4/84; all 75/78/100 | band 0/1±1/8; all 0/1/100 |
| addition (n=100) | band 0/0±0/1; all 0/0/87 | band 35/92±3/98; all 35/92/100 | band 0/0±0/0; all 0/0/97 | band 19/52±5/100; all 24/58/100 | band 0/0±0/0; all 0/0/100 |
| copy (n=100) | band 0/0±0/2; all 0/0/2 | band 59/98±1/100; all 59/98/100 | band 0/0±0/1; all 0/0/1 | band 64/89±3/99; all 64/89/100 | band 1/3±2/9; all 1/5/78 |
| open (n=50) | band 0/2±2/6; all 0/4/20 | band 76/94±3/98; all 78/98/100 | band 0/0±0/4; all 0/0/20 | band 86/98±2/100; all 90/98/100 | band 0/0±0/4; all 0/4/78 |

## fixed | token set: panda_broad

| task | none | sys_panda | sys_horse | user_panda | user_horse |
|---|---|---|---|---|---|
| numbers (n=100) | band 28/62±5/96; all 35/77/100 | band 97/100±0/100; all 97/100/100 | band 30/74±4/97; all 47/90/100 | band 23/66±5/96; all 31/83/100 | band 31/70±5/95; all 38/88/100 |
| trivia (n=100) | band 15/56±5/90; all 28/73/97 | band 92/96±2/100; all 92/98/100 | band 42/81±4/100; all 57/88/100 | band 79/95±2/99; all 86/96/100 | band 32/63±5/95; all 50/84/100 |
| cipher (n=97) | band 21/64±5/98; all 51/91/100 | band 99/100±0/100; all 99/100/100 | band 58/95±2/100; all 76/99/100 | band 82/92±3/100; all 90/98/100 | band 23/71±5/100; all 57/98/100 |
| addition (n=100) | band 0/2±1/43; all 6/39/96 | band 100/100±0/100; all 100/100/100 | band 13/45±5/100; all 30/100/100 | band 19/52±5/100; all 27/77/100 | band 1/5±2/54; all 1/56/100 |
| copy (n=100) | band 4/15±4/53; all 6/28/76 | band 75/99±1/100; all 77/99/100 | band 5/30±5/75; all 8/47/100 | band 64/90±3/99; all 65/90/100 | band 3/15±4/48; all 6/36/98 |
| open (n=50) | band 56/76±6/90; all 68/90/100 | band 100/100±0/100; all 100/100/100 | band 62/94±3/100; all 84/100/100 | band 96/100±0/100; all 98/100/100 | band 54/80±6/96; all 70/100/100 |

## fixed | token set: panda_nn100

| task | none | sys_panda | sys_horse | user_panda | user_horse |
|---|---|---|---|---|---|
| numbers (n=100) | band 13/54±5/90; all 25/84/100 | band 90/98±1/100; all 91/99/100 | band 20/67±5/100; all 34/86/100 | band 19/54±5/98; all 33/97/100 | band 12/55±5/95; all 19/82/100 |
| trivia (n=100) | band 7/10±3/37; all 16/41/92 | band 81/93±3/99; all 83/96/100 | band 6/12±3/40; all 16/40/93 | band 70/89±3/96; all 73/95/100 | band 7/12±3/43; all 16/41/96 |
| cipher (n=97) | band 46/89±3/100; all 47/91/100 | band 82/99±1/100; all 84/100/100 | band 46/88±3/100; all 47/90/100 | band 85/95±2/100; all 86/100/100 | band 43/87±3/99; all 43/90/100 |
| addition (n=100) | band 0/3±2/33; all 2/21/99 | band 35/93±3/100; all 35/94/100 | band 0/13±3/79; all 2/27/100 | band 19/52±5/100; all 24/100/100 | band 0/6±2/51; all 2/20/100 |
| copy (n=100) | band 0/6±2/33; all 1/11/85 | band 59/98±1/100; all 59/98/100 | band 0/3±2/25; all 0/3/40 | band 66/89±3/100; all 66/90/100 | band 2/9±3/44; all 2/12/94 |
| open (n=50) | band 8/18±5/40; all 18/70/88 | band 76/94±3/100; all 78/98/100 | band 8/18±5/46; all 20/54/94 | band 90/100±0/100; all 92/100/100 | band 6/16±5/62; all 16/60/94 |

## fixed | token set: horse

| task | none | sys_panda | sys_horse | user_panda | user_horse |
|---|---|---|---|---|---|
| numbers (n=100) | band 0/0±0/0; all 0/1/3 | band 0/0±0/0; all 0/0/1 | band 68/81±4/93; all 68/81/93 | band 0/0±0/0; all 0/0/0 | band 21/29±5/55; all 22/34/61 |
| trivia (n=100) | band 0/0±0/4; all 0/3/28 | band 0/0±0/3; all 0/4/31 | band 94/100±0/100; all 94/100/100 | band 0/0±0/4; all 0/4/31 | band 94/99±1/100; all 95/99/100 |
| cipher (n=97) | band 0/0±0/3; all 1/98/99 | band 0/1±1/4; all 70/98/99 | band 92/99±1/100; all 95/100/100 | band 1/1±1/3; all 1/98/99 | band 80/87±3/93; all 84/100/100 |
| addition (n=100) | band 0/0±0/0; all 0/0/0 | band 0/0±0/0; all 0/0/0 | band 57/95±2/99; all 57/95/99 | band 0/0±0/0; all 0/0/0 | band 43/100±0/100; all 44/100/100 |
| copy (n=100) | band 0/0±0/1; all 1/3/11 | band 0/0±0/2; all 0/1/17 | band 63/93±3/100; all 64/94/100 | band 0/0±0/2; all 0/1/15 | band 27/57±5/89; all 44/71/96 |
| open (n=50) | band 0/2±2/10; all 0/4/40 | band 0/2±2/8; all 0/2/50 | band 92/96±3/98; all 96/96/100 | band 0/2±2/12; all 0/4/50 | band 90/98±2/98; all 100/100/100 |

## fixed | token set: horse_broad

| task | none | sys_panda | sys_horse | user_panda | user_horse |
|---|---|---|---|---|---|
| numbers (n=100) | band 18/55±5/88; all 24/79/99 | band 22/67±5/94; all 27/85/99 | band 76/95±2/100; all 78/98/100 | band 13/54±5/88; all 20/73/98 | band 38/75±4/99; all 49/85/100 |
| trivia (n=100) | band 6/26±4/63; all 15/45/96 | band 9/26±4/65; all 19/49/95 | band 95/100±0/100; all 95/100/100 | band 6/32±5/73; all 16/50/98 | band 97/100±0/100; all 98/100/100 |
| cipher (n=97) | band 12/58±5/92; all 22/100/100 | band 18/64±5/97; all 75/100/100 | band 93/100±0/100; all 96/100/100 | band 10/57±5/91; all 19/100/100 | band 82/93±3/98; all 91/100/100 |
| addition (n=100) | band 0/7±3/75; all 47/61/98 | band 0/6±2/73; all 49/60/96 | band 58/97±2/100; all 79/98/100 | band 0/4±2/72; all 49/60/97 | band 43/100±0/100; all 68/100/100 |
| copy (n=100) | band 1/3±2/31; all 4/22/59 | band 1/2±1/24; all 3/18/56 | band 67/94±2/100; all 67/97/100 | band 2/10±3/40; all 4/26/64 | band 30/61±5/96; all 48/75/99 |
| open (n=50) | band 6/20±6/46; all 14/56/96 | band 8/22±6/46; all 14/56/94 | band 96/96±3/100; all 96/100/100 | band 6/26±6/52; all 12/56/98 | band 90/100±0/100; all 100/100/100 |

## fixed | token set: horse_nn100

| task | none | sys_panda | sys_horse | user_panda | user_horse |
|---|---|---|---|---|---|
| numbers (n=100) | band 2/12±3/53; all 2/21/85 | band 25/84±4/99; all 33/86/99 | band 71/88±3/99; all 71/89/100 | band 4/22±4/65; all 4/30/88 | band 26/54±5/87; all 31/62/97 |
| trivia (n=100) | band 7/23±4/59; all 10/58/91 | band 16/47±5/87; all 26/81/97 | band 95/100±0/100; all 95/100/100 | band 14/44±5/83; all 19/76/95 | band 97/100±0/100; all 98/100/100 |
| cipher (n=97) | band 4/9±3/51; all 5/98/100 | band 5/41±5/82; all 81/99/100 | band 93/100±0/100; all 96/100/100 | band 6/19±4/63; all 8/98/100 | band 82/90±3/97; all 91/100/100 |
| addition (n=100) | band 0/2±1/70; all 2/40/100 | band 1/25±4/99; all 31/69/100 | band 58/98±1/100; all 59/100/100 | band 0/6±2/77; all 9/48/100 | band 43/100±0/100; all 50/100/100 |
| copy (n=100) | band 1/4±2/28; all 1/14/60 | band 1/23±4/68; all 42/67/96 | band 67/93±3/100; all 70/95/100 | band 8/30±5/77; all 20/60/94 | band 31/69±5/96; all 54/88/99 |
| open (n=50) | band 16/38±7/64; all 24/62/94 | band 24/64±7/94; all 78/94/98 | band 92/98±2/100; all 98/100/100 | band 26/62±7/84; all 54/84/100 | band 92/100±0/100; all 100/100/100 |

## Size-matched chance (100 random whole-word sets per size), top-1/5/20 %

- numbers, set size 3: band 1/3/9; all 3/9/22
- numbers, set size 55: band 12/38/79; all 25/67/97
- numbers, set size 75: band 18/50/88; all 38/81/99
- numbers, set size 100: band 21/58/94; all 41/87/100
- text tasks, set size 3: band 1/2/7; all 3/7/18
- text tasks, set size 55: band 12/35/70; all 24/61/89
- text tasks, set size 75: band 14/42/77; all 32/72/93
- text tasks, set size 100: band 22/54/85; all 42/79/97
set sizes: {'panda': None, 'panda_broad': None, 'panda_nn100': None, 'horse': None, 'horse_broad': None, 'horse_nn100': None} (see onpolicy_lens log for sizes)

## System-prompt content tokens

- sys_panda, token set panda (n=547): band 100/100/100; all 100/100/100
- sys_horse, token set horse (n=547): band 100/100/100; all 100/100/100
- sys_panda, token set horse (n=547): band 0/0/100; all 0/100/100
- sys_horse, token set panda (n=547): band 0/0/0; all 0/0/0

## Per-layer top-5 rate (%), on-policy numbers (filter-pass subset) vs open-ended answers, token set panda

| layer | num none | num sys_panda | num sys_horse | num user_panda | num user_horse | open none | open sys_panda | open sys_horse | open user_panda | open user_horse |
|---|---|---|---|---|---|---|---|---|---|---|
| 0 | 0 | 0 | 0 | 0 | 0 | 1 | 87 | 0 | 72 | 0 |
| 1 | 0 | 0 | 0 | 0 | 0 | 0 | 87 | 0 | 44 | 0 |
| 2 | 0 | 0 | 0 | 0 | 0 | 0 | 53 | 0 | 28 | 0 |
| 3 | 0 | 0 | 0 | 0 | 0 | 0 | 99 | 0 | 81 | 2 |
| 4 | 0 | 0 | 0 | 0 | 0 | 0 | 99 | 0 | 81 | 0 |
| 5 | 0 | 0 | 0 | 0 | 0 | 0 | 99 | 0 | 81 | 0 |
| 6 | 0 | 0 | 0 | 0 | 0 | 0 | 98 | 0 | 81 | 0 |
| 7 | 0 | 0 | 0 | 0 | 0 | 0 | 98 | 0 | 85 | 0 |
| 8 | 0 | 1 | 0 | 0 | 0 | 0 | 99 | 0 | 93 | 0 |
| 9 | 0 | 1 | 0 | 0 | 0 | 0 | 99 | 0 | 92 | 0 |
| 10 | 0 | 2 | 0 | 0 | 0 | 0 | 99 | 0 | 90 | 0 |
| 11 | 0 | 3 | 0 | 0 | 0 | 0 | 99 | 0 | 84 | 0 |
| 12 | 0 | 3 | 0 | 0 | 0 | 0 | 99 | 0 | 85 | 0 |
| 13 | 0 | 0 | 0 | 0 | 0 | 0 | 99 | 0 | 83 | 0 |
| 14 | 0 | 0 | 0 | 0 | 0 | 0 | 99 | 0 | 82 | 0 |
| 15 | 0 | 0 | 0 | 0 | 0 | 0 | 98 | 0 | 83 | 0 |
| 16 | 0 | 1 | 0 | 0 | 0 | 0 | 99 | 0 | 80 | 0 |
| 17 | 0 | 3 | 0 | 0 | 0 | 1 | 100 | 0 | 81 | 0 |
| 18 | 0 | 41 | 1 | 2 | 2 | 0 | 99 | 0 | 82 | 0 |
| 19* | 0 | 24 | 0 | 2 | 0 | 2 | 100 | 0 | 83 | 0 |
| 20* | 0 | 44 | 1 | 2 | 0 | 0 | 100 | 0 | 85 | 0 |
| 21* | 0 | 29 | 0 | 2 | 0 | 0 | 99 | 0 | 83 | 0 |
| 22* | 0 | 27 | 0 | 2 | 0 | 0 | 98 | 0 | 80 | 0 |
| 23* | 0 | 72 | 0 | 5 | 0 | 1 | 100 | 0 | 92 | 0 |
| 24* | 0 | 71 | 0 | 5 | 0 | 0 | 100 | 0 | 93 | 0 |
| 25* | 0 | 72 | 0 | 5 | 0 | 0 | 100 | 0 | 93 | 0 |
| 26* | 0 | 67 | 0 | 4 | 0 | 0 | 100 | 0 | 93 | 0 |
| 27* | 0 | 89 | 0 | 6 | 0 | 0 | 100 | 0 | 93 | 0 |
| 28* | 0 | 80 | 0 | 4 | 0 | 0 | 100 | 0 | 93 | 0 |
| 29* | 0 | 88 | 0 | 5 | 0 | 0 | 100 | 0 | 95 | 0 |
| 30* | 0 | 86 | 0 | 3 | 0 | 0 | 100 | 0 | 96 | 0 |

(* = read band)
