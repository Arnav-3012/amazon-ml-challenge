# Token dictionary probe (src/mine_dict.py)

Mined on 80% of train S1s' true pairs; evaluated on the missed pairs (not in candidates_train_v1) of the held-out 20%. Keep: top-1 co >= 20 and share >= 0.8. Vocab proxy = no shared name/addr token with df <= 1000; full-data proxy count 205,535 vs D0-1 vocab 106,630 (D0-1 also counts C/X channels). surv = shares >= 1 token with df <= cap (df as it is today).

**Mappings:** 807 (native 754, abbr 53).

## Is it deterministic? top-1 share over sources with co >= 20

| kind | sources co>=20 | kept (share>=0.8) | [0,0.5) | [0.5,0.8) | [0.8,0.9) | [0.9,0.95) | [0.95,0.99) | [0.99,1] | median share |
|---|---|---|---|---|---|---|---|---|---|
| native | 938 | 754 | 158 | 26 | 45 | 65 | 65 | 579 | 1.0000 |
| abbr | 238 | 53 | 146 | 39 | 10 | 10 | 5 | 28 | 0.2958 |

## KEY: held-out misses that share a token after the dict

| segment | held misses | full-data est | any before % | surv before % | surv after native % | surv after abbr % | surv after both % | any after both % |
|---|---|---|---|---|---|---|---|---|
| vocab (proxy) | 41,397 | 206,985 | 99.9396 | 0.0000 | 0.0145 | 0.0048 | 0.0193 | 99.9734 |
| India-native vocab | 11,956 | 59,780 | 99.8829 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 99.9916 |
| India-native, all misses | 15,717 | 78,585 | 99.9109 | 23.9295 | 23.8213 | 23.9295 | 23.8213 | 99.9936 |
| vocab ∪ India-native (KEY) | 45,158 | 225,790 | 99.9446 | 8.3285 | 8.3042 | 8.3330 | 8.3086 | 99.9756 |
| precision: held v1 negatives, new surv token | 6,512,362 | 32,561,810 | nan | 0.0000 | nan | nan | 0.0056 | nan |

## 30 sample mappings (top co per kind)

| field | kind | s | t | co | n_src | share |
|---|---|---|---|---|---|---|
| name | native | tek | tech | 13,342 | 13,346 | 0.9997 |
| name | native | globl | global | 11,164 | 11,168 | 0.9996 |
| name | native | imtrnesnl | international | 8,815 | 8,817 | 0.9998 |
| name | native | aiti | it | 8,758 | 8,758 | 1.0000 |
| name | native | impeks | impex | 8,569 | 8,569 | 1.0000 |
| name | native | eksports | exports | 8,496 | 8,496 | 1.0000 |
| name | native | estet | estate | 8,110 | 8,110 | 1.0000 |
| name | native | phuds | foods | 7,911 | 7,911 | 1.0000 |
| name | native | enrji | energy | 7,904 | 7,904 | 1.0000 |
| name | native | phud | food | 7,774 | 7,774 | 1.0000 |
| name | native | prodyusr | producer | 7,521 | 7,521 | 1.0000 |
| name | native | prodkts | products | 7,514 | 7,514 | 1.0000 |
| name | native | tredimg | trading | 7,351 | 7,351 | 1.0000 |
| name | native | midiya | media | 7,228 | 7,228 | 1.0000 |
| name | native | bildrs | builders | 7,182 | 7,182 | 1.0000 |
| addr | abbr | pkwy | parkway | 8,302 | 8,314 | 0.9986 |
| addr | abbr | ft | fort | 4,992 | 5,374 | 0.9289 |
| name | abbr | skti | shakti | 4,650 | 4,650 | 1.0000 |
| name | abbr | sivm | shivam | 4,073 | 4,075 | 0.9995 |
| addr | abbr | cv | cove | 3,639 | 3,641 | 0.9995 |
| addr | abbr | mt | mount | 2,834 | 3,060 | 0.9261 |
| name | abbr | krsn | krishna | 2,691 | 2,691 | 1.0000 |
| addr | abbr | atl | atlanta | 2,328 | 2,333 | 0.9979 |
| name | abbr | krsna | krishna | 1,706 | 1,706 | 1.0000 |
| addr | abbr | sq | square | 1,609 | 1,609 | 1.0000 |
| addr | abbr | rdg | ridge | 1,028 | 1,031 | 0.9971 |
| name | abbr | visn | vision | 943 | 943 | 1.0000 |
| addr | abbr | tpke | turnpike | 863 | 863 | 1.0000 |
| addr | abbr | sw | southwest | 556 | 659 | 0.8437 |
| addr | abbr | aly | alley | 442 | 443 | 0.9977 |
