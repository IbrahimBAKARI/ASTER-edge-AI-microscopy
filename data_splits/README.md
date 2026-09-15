# data_splits/ — construction of every split

No image is included. Sources and access: [`../docs/DATA.md`](../docs/DATA.md).

| File | Content | Built by |
|---|---|---|
| `prototype_fields_split.csv` | the 478 prototype fields: `filename`, `split` (`train` 204, `val` 43, `test` 104, `excluded` 127), `triage_verdict` (`keep`, `review`, `reject`) | `localizer_adaptation/make_x40_split.py` (acquisition clusters separated by > 90 s, 35 clusters, seed 42; cluster map in `localizer_adaptation/x40_SPLIT.md`) |
| `block2_corpus_splits.csv` | the block-2 corpus: `corpus_path` (source/patient/image inside the prepared corpus), `source`, `patient_id`, `split` | `block2_development/datasets/build_splits.py` (patient-level, rules in `block2_development/datasets/split_plan.yaml` and `PREREGISTRATION.md` §5–8) |

`excluded` prototype fields are the rejected fields of training and validation
clusters; the test split keeps every verdict.

Block-2 splits (images): `cell_train` 50 810, `cell_val` 14 633, `train_mil`
36 472, `mil_val` 12 564, `dev_fit` 15 797, `dev_cal` 16 381, `test_final`
201 560 (the 409 cAItomorph patients, used only for the final test).

The prototype batches of the out-of-domain stress test are the test split
(104 fields), the validation split (43) and the training split cut in two
(102 + 101 readable fields; one field with a broken JPEG stream is excluded);
the exact field lists are in `results/cpu_reruns_frozen/*/fields.txt`.
