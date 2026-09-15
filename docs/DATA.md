# DATA.md — data sources, roles, availability

**No image of any dataset is included in this repository.** Public datasets are
distributed by their authors under their own terms; this file gives their
identifiers, and [`../data_splits/`](../data_splits/) gives the construction of
every split. The prototype fields acquired for this work are available from the
authors on reasonable request.

| Source | Content | Role in this work | Availability |
|---|---|---|---|
| Prototype-acquired fields (this work) | 478 fields, 4032 × 3040, 788 WBC boxes; non-leukemic educational smears | localizer adaptation (train 204 / val 43 / test 104 fields); out-of-domain stress test (351 fields); acquisition timing | **available from the authors on reasonable request** |
| LeukemiaAttri [1] | multi-microscope, multi-magnification fields with WBC boxes; 47 patients in H_100X_C1 | localizer source domain and replay; cell-head training crops (sharpness ≥ 4); held-out out-of-domain sessions | from its authors [1] |
| AML-Cytomorphology_MLL_Helmholtz [2] | 189 patients (129 AML, 60 controls), single-cell images with manual differentials | MIL training; τ, temperature, APL threshold, OOD and [REF] values (patient-level splits); differential validity | The Cancer Imaging Archive |
| AML-Cytomorphology_LMU [3] | 18 365 single-cell images | cell-head training | The Cancer Imaging Archive |
| PBC [4] | 17 092 single-cell images of normal cells | cell-head training (second scanner) | from its authors [4] |
| MILLIE [5] | 8 286 labeled cells, 56 labeled patients | cell-head training (smudge cells, promyelocytes) | from its authors [5] |
| ALL-IDB2 [6] | 260 single-cell images | cell-head training (lymphoblasts, low weight) | from its authors [6] |
| cAItomorph test set [7] | 409 patients, 201 560 cells, median 500 per patient | held-out test of block 2 only | Nefeli RDM [7] |
| ALL-IDB L2 fields [6] | 9 fields, 95 detected WBCs | technical latency/power fixture only | from its authors [6] |

---

## 1. Prototype-acquired fields (this work)

- 478 fields acquired on 8 September 2026 with the add-on mounted on a
  conventional microscope, through the ×40 dry objective, from pre-existing
  educational peripheral-blood smears (Suptech Santé, Mohammedia, Morocco).
  Non-leukemic material; no patient identifier and no clinical metadata.
- Automatic quality triage: keep 136, review 179, reject 163.
- Single-class WBC boxes (YOLO format): 788.
- Split by acquisition cluster (a new cluster after a gap > 90 s), 35 clusters,
  seed 42: train 204 fields / 334 boxes, validation 43 / 67, test 104 / 171.
  Train and validation keep the *keep* and *review* fields only; the test split
  keeps every verdict (36 rejected fields). 127 rejected fields of training and
  validation clusters are not used.
- Released here: the per-field split and verdict
  ([`../data_splits/prototype_fields_split.csv`](../data_splits/prototype_fields_split.csv)),
  the cluster map ([`../localizer_adaptation/x40_SPLIT.md`](../localizer_adaptation/x40_SPLIT.md)),
  capture timestamps ([`../results/prototype_acquisition/acquisition_log.csv`](../results/prototype_acquisition/acquisition_log.csv)),
  and every result computed on these fields. The images and box annotations are
  **available from the authors on reasonable request**.
- Expected layout when you have them: `images/{train,val,test}/*.jpg` +
  `labels/{train,val,test}/*.txt` + `data.yaml`; point `ASTER_X40_DATASET` at
  that folder (default `external_data/prototype_fields_dataset/`). The raw
  annotated fields with the triage files (`triage/_triage.csv`,
  `triage/manifest.csv`) go under `ASTER_X40_RAW` (default
  `external_data/prototype_fields_raw/`).
- In file names these fields carry the historical tag `x40`.

## 2. LeukemiaAttri

The localizer's source domain. In the training notebooks and scripts this
corpus appears under the working name "LLD" (its LCM and HCM subsets). It is
not the Large Leukemia Dataset (LLD) of the same authors [9], which extends
LeukemiaAttri with Sparse-LeukemiaAttri; Sparse-LeukemiaAttri is not used here.
It is used to train every YOLO11n recipe, as the replay half of the mixed-replay
fine-tune, and (subset H_100X_C1, sharpness ≥ 4, boxes labelled `None`
excluded) for cell-head training crops. The four held-out LeukemiaAttri sessions
of `docs/EVALUATION.md` § 2.3 list their fields in
`results/cpu_reruns_frozen/leukemiaattri_*/fields.txt`. No patient/slide registry
exists across its subsets, so patient-level independence between its splits
cannot be claimed.

## 3. cAItomorph — held-out test cohort of block 2

409 patients, 201 560 single-cell images, Munich Leukemia Laboratory,
Wright–Giemsa. Used for nothing but the final evaluation of block 2, never in a
training or development split (`block2_development/PREREGISTRATION.md` §5–6).
Block 2 was developed in part on AML-Cytomorphology_MLL_Helmholtz, which comes
from the same laboratory; the two datasets are distinct, but no patient-identity
mapping exists to verify that no patient appears in both: the evaluation is
therefore described as *held-out*, not *external*. Test-set DOI `10.82296/HMGU-NEFELI.9BV4E-3AG16`; source article [8].

## 4. ALL-IDB

`benchmark_session.txt` (9 ALL-IDB L2 fields) and `benchmark_images.txt` list the
technical fixture used for the latency and power runs; block 2 returns
`out_of_domain` on it. Place the images under `external_data/ALL_IDB/L2/` (paths
in both lists are relative to the repository root) or edit the lists.

## 5. Block-2 training corpus

Prepared by `block2_development/datasets/prepare_corpus.py` from the sources
above (paths in `block2_development/datasets/sources.yaml`, to be edited to your
copies): 348 217 images after exclusions, square-padded and stored as lossless
WebP. Patient-level splits: `block2_development/datasets/build_splits.py`, output
released as [`../data_splits/block2_corpus_splits.csv`](../data_splits/block2_corpus_splits.csv)
(image path, source, patient id, split). Provenance, licences and every
exclusion: `block2_development/datasets/DATASETS.md`; split rules:
`block2_development/datasets/split_plan.yaml` and `PREREGISTRATION.md` §5–8.

---

## References

1. A. Rehman *et al.*, "A large-scale multi domain leukemia dataset for the white blood cells detection with morphological attributes for explainability," MICCAI 2024.
2. M. Hehr *et al.*, AML-Cytomorphology_MLL_Helmholtz, The Cancer Imaging Archive, 2023, doi: 10.7937/6PPE-4020.
3. C. Matek, S. Schwarz, C. Marr, K. Spiekermann, AML-Cytomorphology_LMU, The Cancer Imaging Archive, 2019, doi: 10.7937/tcia.2019.36f5o9ld.
4. A. Acevedo *et al.*, *Data Brief*, vol. 30, Art. no. 105474, 2020, doi: 10.1016/j.dib.2020.105474.
5. P. Manescu *et al.*, *Sci. Rep.*, vol. 13, Art. no. 2562, 2023, doi: 10.1038/s41598-023-29160-4.
6. R. D. Labati, V. Piuri, F. Scotti, "All-IDB: The acute lymphoblastic leukemia image database for image processing," IEEE ICIP 2011, doi: 10.1109/ICIP.2011.6115881.
7. M. F. Dasdelen *et al.*, "cAItomorph: Peripheral blood smear image dataset for hematological malignancy prediction," Nefeli RDM, 2026, doi: 10.82296/HMGU-NEFELI.9BV4E-3AG16.
8. M. F. Dasdelen *et al.*, *Leukemia*, vol. 40, no. 6, pp. 1318–1322, 2026, doi: 10.1038/s41375-026-02934-1.
9. A. Rehman *et al.*, "Leveraging sparse annotations for leukemia diagnosis on the large leukemia dataset," *Med. Image Anal.*, vol. 106, Art. no. 103760, 2025, doi: 10.1016/j.media.2025.103760.
