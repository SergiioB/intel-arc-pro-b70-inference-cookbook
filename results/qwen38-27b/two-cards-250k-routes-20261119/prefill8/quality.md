
| Text comparison | Equal |
|---|---:|
| Flag off vs on, drafted | 1/8 |
| Flag off vs on, serial | 1/8 |
| Drafted vs serial, flag off | 8/8 |
| Drafted vs serial, flag on | 8/8 |

| Set | Items | Stock | Prefix | Flag off | Flag on |
|---|---:|---:|---:|---:|---:|
| tune | 82 | 78 | 78 | 78 | 78 |
| test | 82 | 76 | 76 | 76 | 76 |
| instruction_following | 55 | 33 | 33 | 33 | 33 |
| knowledge_professional | 55 | 40 | 42 | 40 | 41 |
| structured_output | 55 | 55 | 55 | 55 | 55 |
| Total | 329 | 282 | 284 | 282 | 283 |

| From | To | Fail → pass | Pass → fail | Text hashes equal |
|---|---|---|---|---:|
| off | on | he-032, he-093, prof-biz-401, prof-law-1935 | he-116, he-127, prof-biz-484 | 201/329 |
| stock | off | — | — | 305/329 |
| stock | on | he-032, he-093, prof-biz-401, prof-law-1935 | he-116, he-127, prof-biz-484 | 201/329 |
| prefix | off | — | prof-biz-401, prof-law-1935 | 261/329 |
| prefix | on | he-032, he-093 | he-116, he-127, prof-biz-484 | 201/329 |
