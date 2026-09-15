# LeukemiaAttr — unique cells retained per sharpness threshold

Sensitivity analysis for the [OPS] threshold of PREREGISTRATION.md amendment 3b.
Counts are **unique physical cells** (acquisitions grouped by `cell_uid_nomag`),
reported as train | test.

| class | no filter | ≥2 | ≥4 | ≥6 | ≥8 |
|---|---|---|---|---|---|
| `lymphoblast` | 5169 \| 662 | 3038 \| 401 | 2001 \| 233 | 1539 \| 192 | 1190 \| 174 |
| `promyelocyte_abnormal` | 693 \| 523 | 331 \| 352 | 186 \| 252 | 124 \| 181 | 83 \| 158 |
| `lymphocyte_atypical` | 848 \| 679 | 507 \| 466 | 294 \| 362 | 204 \| 240 | 116 \| 210 |
| `myeloblast` | 8202 \| 1486 | 5006 \| 980 | 2947 \| 590 | 2410 \| 446 | 2069 \| 403 |
| `segmented_neutrophil` | 3288 \| 1906 | 2007 \| 1282 | 1324 \| 833 | 1089 \| 633 | 971 \| 560 |
| `monocyte` | 1628 \| 564 | 1108 \| 380 | 692 \| 235 | 532 \| 172 | 477 \| 148 |
| `lymphocyte` | 1159 \| 387 | 496 \| 193 | 275 \| 102 | 175 \| 72 | 104 \| 44 |
| `myelocyte` | 927 \| 583 | 533 \| 370 | 339 \| 247 | 282 \| 202 | 249 \| 166 |
| `metamyelocyte` | 404 \| 152 | 211 \| 99 | 136 \| 53 | 116 \| 38 | 102 \| 37 |
| `eosinophil` | 267 \| 137 | 138 \| 79 | 94 \| 58 | 77 \| 45 | 74 \| 40 |
| `basophil` | 89 \| 12 | 40 \| 7 | 31 \| 5 | 23 \| 5 | 19 \| 3 |
| `other_artifact` | 9183 \| 1597 | 5356 \| 961 | 3471 \| 645 | 2554 \| 472 | 2124 \| 398 |
| **totals** | 104730 crops / 30003 cells | 51177 crops / 19671 cells | 28528 crops / 12900 cells | 20243 crops / 10003 cells | 15327 crops / 8592 cells |

The adopted threshold is **4.0**. At 2 the label is no longer verifiable by eye
(see `_work/bandes_nettete.png`); at 6 and 8 the three classes the amendment exists
for lose 25 % and 45 % of their cells for no gain in verifiability.
