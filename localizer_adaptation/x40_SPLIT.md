# x40 prototype split for mixed-replay fine-tuning

Triage verdicts: `GARDER` = keep, `A_REVOIR` = review, `REJET` = reject. Per-field list:
`data_splits/prototype_fields_split.csv`. Source folder: available from the authors on
reasonable request (`docs/DATA.md`).

- source: <PROTOTYPE_RAW>/triage
- 478 annotated fields, 788 WBC boxes
- split unit: acquisition cluster (gap > 90.0s = spatially separate)
- 35 clusters, shuffled seed 42, packed test~18% / val~16% / train rest
- train/val keep GARDER+A_REVOIR only; test keeps ALL verdicts

| split | fields | boxes | GARDER | A_REVOIR | REJET | REJET dropped |
|---|---|---|---|---|---|---|
| train | 204 | 334 | 85 | 119 | 0 | 93 |
| val | 43 | 67 | 15 | 28 | 0 | 34 |
| test | 104 | 171 | 36 | 32 | 36 | 0 |

## clusters per split
- **train**: C0[11:15-11:22,14] C1[11:25-11:31,16] C2[11:32-11:37,15] C3[11:38-11:43,22] C4[11:52-12:02,28] C6[12:11-12:16,11] C7[12:18-12:23,11] C10[12:45-12:49,7] C11[12:50-13:00,24] C13[13:14-13:20,11] C14[13:24-13:24,1] C15[13:27-13:27,1] C17[13:47-13:52,10] C18[13:55-14:02,13] C19[14:11-14:12,3] C21[14:22-14:33,19] C23[14:46-15:00,24] C25[15:08-15:14,9] C27[15:24-15:26,2] C28[15:27-15:32,6] C30[15:44-15:59,22] C32[16:19-16:37,23] C33[16:38-16:40,3] C34[16:42-16:42,1]
- **val**: C16[13:31-13:45,26] C22[14:35-14:44,18] C24[15:02-15:05,6] C26[15:15-15:22,11] C29[15:33-15:43,16]
- **test**: C5[12:03-12:10,14] C9[12:31-12:42,26] C12[13:02-13:13,22] C20[14:14-14:21,16] C31[16:00-16:18,26]
