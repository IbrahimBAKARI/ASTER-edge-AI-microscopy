# Datasets — provenance, what each one is for, and what was verified

Every count below was read off this Mac on 2026-09-10 by
`datasets/prepare_corpus.py --dry-run`, not copied from a paper.

## In the corpus

| Source | Images | Native size | Patient IDs | Role | Licence / access |
|---|---|---|---|---|---|
| `aml_mll` — AML-Cytomorphology MLL Helmholtz v1 | **81 214** | 144×144 TIF | **yes**, 189 patients | MIL training + all `[FIT]`/`[REF]` development | TCIA, check the collection terms |
| `aml_lmu` — AML-Cytomorphology_LMU (Matek 2019) | **18 365** | 400×400 TIFF | no | cell-head training, 15 classes | TCIA, check the collection terms |
| `all_idb2` — ALL-IDB2 (Scotti, Milano) | **260** | 257×257 TIF | no | lymphoblast, weight 0.3 | signed agreement |
| `leukemiaattr` — LeukemiaAttr, cleaned crops | **28 528** | 56–90 px native | **yes**, 47 patients | cell head: lymphoblast with patients, APL, atypical lymphocytes | source terms |
| `pbc` — PBC / Acevedo 2020 | **17 092** | 360×363 JPG | no | cell head: the left-shift classes + a second scanner | CC-BY, DOI 10.17632/snkd93bnjr.1 |
| `millie` — MILLIE / AML_APL (Manescu 2023) | **8 291** | 360×363 JPG | **yes**, 106 patients | cell head: **smudge cells**, blasts, promyelocytes | source terms |
| `caitomorph` — cAItomorph (Dasdelen 2026) | **201 560** | 144×144 TIF | **yes**, 409 patients | **HELD-OUT TEST ONLY** | DOI 10.82296/hmgu-nefeli.9bv4e-3ag16 |
| `x40_proto` — prototype ×40 crops | regenerated | variable | by session | OOD stress test | own data |

**Total 355 310 images, ≈ 10.3 GB as the lossless WebP corpus.**

## Excluded, and why

### `ALL_IDB_Dataset/` — excluded from v1

Inspected on disk:

| Folder | Content | Size |
|---|---|---|
| `L1` | 50 **full fields** + YOLO `.txt` boxes | 2592×1944 |
| `L2` | 50 **full fields** + YOLO `.txt` boxes | 1712×1368 |
| `L3` | 50 crops (38 jpg + 12 tif) | 257×257 |

Three findings:

1. `L1` and `L2` are **ALL-IDB1 full fields**, not crops. `L1/L2/L3` is *not* the FAB
   subtype.
2. The 493 annotated boxes carry a single class, **`Candidate_Cell`** — WBC
   *localisation*, not a cell type. The folder also holds a stray class list containing
   entries such as `car`, `dog`, `hamburger`, so any naive `cat *.txt` parse is garbage.
3. The ALL-IDB `_0` / `_1` filename convention is applied inconsistently here
   (`Im086_02.jpg`, `Im078_3.jpg`).

Against ALL-IDB2 (260 clean crops) and Taleqani (3 256), the marginal value is near zero
while the label semantics are unresolved. Injecting that noise into `lymphoblast` — the
class the tier-B ALL claim rests on — is not worth it. Re-enable in
`datasets/sources.yaml` only after resolving the box semantics.

### `ALL dataset_kaggle_Taleqani_hospital/` — excluded entirely

Recorded in phase 0 as single-cell crops on the strength of the 224×224 dimension.
**Looking at the images shows fields**: several leukocytes, many erythrocytes, a burned-in
scale bar. `Segmented/` is a colour-threshold segmentation of *every* nucleus in the
field, not a mask of the labelled cell, and it fails on many images (coverage 0.0–0.1 % on
the samples checked).

The label is therefore a field label. In an ALL field not every leukocyte is a blast, so
calling each detected cell a lymphoblast is the same label noise refused for
`ALL_IDB_Dataset`. Cost of the exclusion: `lymphoblast` drops from 2 882 to **130** cells.
See `PREREGISTRATION.md`, amendment 2.

### `Bone-Marrow-Cytomorphology` — out of scope for v1

Bone marrow contains populations that do not occur in peripheral blood (erythroblasts in
quantity, megakaryocytes, plasma cells, abundant maturing granulocytes). Our level-1
signal is precisely "this cell should not be here". More importantly, the OOD reference
must be fitted on peripheral blood only; marrow inside that distribution would widen the
in-domain manifold and weaken the very gate that catches the ×40 case.

## PBC / Acevedo — what it actually contains

Downloaded and wired in. Layout on disk:

```
PBC_dataset/PBC_dataset/wbc/{neutrophil,monocyte,eosinophil,lymphocyte}   9 081
PBC_dataset/PBC_dataset/other_types/{basophil,ig,erythroblast,platelet}   8 012
PBC_dataset/PBC_dataset/wbc_resized/...      DUPLICATES — never enumerated
PBC_dataset_split/...                        DUPLICATES — never enumerated
```

The **filenames carry a finer label than the folder names**, which is what makes this
source valuable rather than merely diverse:

| Prefix | Class | PBC | AML-LMU had | Total |
|---|---|---|---|---|
| `PMY` | promyelocyte | 592 | 70 | **662** |
| `MY` | myelocyte | 1 137 | 42 | **1 179** |
| `MMY` | metamyelocyte | 1 015 | 15 | **1 030** |
| `BNE` | band neutrophil | 1 633 | 109 | **1 742** |
| `SNE` | segmented neutrophil | 1 646 | 8 484 | 10 130 |
| `IG` | unspecified → **partial label** over the three | 151 | — | — |

Those four classes are exactly the ones the CML left-shift rule depends on, so the rule
moves off the "weakest link" list. Platelets (2 348) are not leukocytes and train the
`other_artifact` reject class.

PBC has **no patient identifiers**. It therefore trains the cell head but takes no part in
the `[REF]` reference intervals, which are per-patient proportions — see
`PREREGISTRATION.md`, amendment 1b.

After PBC, the remaining thin classes are `smudge_cell` (15), `lymphocyte_atypical` (11)
and `promyelocyte_abnormal` (18), all from AML-LMU alone: the **CLL rule and the APL
flag**. Those carry the declared uncertainty now.

## Champ plein vs cellule unique — comment chaque image est traitée

Block 2 consumes **single-cell crops only**. It never sees a full field. Sources are
therefore of two kinds, and `datasets/prepare_corpus.py` treats them differently:

| Kind | Sources | Path through the pipeline |
|---|---|---|
| **already a cell** (declared `granularity: cell`, verified by eye) | cAItomorph 144², AML-MLL 144², ALL-IDB2 257², PBC 360×363, AML-LMU 400² | `guard_is_single_cell` → `SquarePad` → ≤256 px → lossless WebP |
| **full field** (declared `granularity: field`) | LeukemiaAttr 640² (**cut, see below**), Taleqani 224² (disabled, no boxes), ALL-IDB1 2592×1944 (disabled), prototype ×40 4032×3040 | `expand_and_clip_box` (+10 %/side, clipped) → `extract_crop` → `SquarePad` → ≤256 px → WebP, **one corpus entry per box** |

`src/aster_block2/crops.py` holds that crop convention and is **byte-identical to
`Software_Dev_Micro_Edge/aster_pipeline/crop_extraction.py`** — verified over 200 000
random boxes by `tests/test_preprocess_parity.py`, including the `int()` truncation, the
`width - 1` clipping, and the rule that a degenerate box drops the detection entirely
rather than yielding a padded stub. The same module produces the ×40 crops in notebook §2,
so the Mac, Colab and the Jetson cut cells identically.

**Granularity is declared, never inferred.** `sources.yaml` requires
`granularity: cell | field` per source, set only after looking at the images;
`prepare_corpus.py` refuses a source that does not declare it, and refuses a `field`
source whose records carry no box. Two sources had already been mis-classified from their
dimensions alone — a 224×224 image can be a downscaled field.

**The size guard, second line of defence.** An un-boxed image larger than 512 px on its longest side raises instead of
being processed. Without it, a 2592×1944 field would be square-padded and squashed into
224 px, producing a "cell" that is a whole smear — a failure that trains silently and is
invisible in any loss curve. 512 px is above every single-cell source in the corpus
(largest: AML-LMU at 400) and below every field source (smallest: LeukemiaAttr at 640).

The manifest records `cropped_from_field` (0/1) per image, so the split builder and any
later audit can tell the two provenances apart.

## Split discipline

> **Superseded in part** — the AML-MLL split is now 0.45 `train_mil` / 0.15 `mil_val` / 0.20
> `dev_fit` / 0.20 `dev_cal` (amendment 5) and MILLIE is split 0.80/0.20 over its 56 labelled
> patients (amendment 7). The paragraphs below describe the original plan.

See `PREREGISTRATION.md` §5.2 and `datasets/build_splits.py`.

* `aml_mll`, patient-level, stratified by `bag_label`: `train_mil` 0.60 / `dev_fit` 0.20 /
  `dev_cal` 0.20. Thresholds are fitted on `dev_fit`, never on the patients the models saw.
* `leukemiaattr`: patient identifiers exist. Its own patient-disjoint train/test split is
  carried through, not redrawn, and acquisitions of one physical cell are grouped by
  `cell_uid_nomag` so they can never land on opposite sides.
* `millie`: patient identifiers exist. Split = the dataset's own Discovery (82) /
  Validation (24) cohorts, carried through, not redrawn.
* `aml_lmu`, `all_idb2`, `pbc`: **no patient identifiers exist**, so the 0.80/0.20
  split is image-level and cells of one patient may appear on both sides. Declared
  limitation; these sources never support a patient-level claim.
* `caitomorph` → `test_final`. Never trained on, never fitted on, read only in notebook §9.
* `x40_proto` → `stress`.


## LeukemiaAttr — from fields to a clean crop set

The source is 640×640 fields with COCO boxes across 12 subdomains (2 microscopes ×
2 cameras × 3 nominal magnifications); `L_10X_C1` is an empty folder, leaving 11. The same
physical cells are re-acquired in each: they differ in **sharpness and camera, not scale**
(median box-size ratio 0.98 across 10×/40×/100×, measured in the block-1 work).

```
extract_leukemiaattr.py   fields + COCO boxes -> 104 730 crops, block-1 convention
annotate_sharpness.py     adds a measured `sharpness` column per crop
clean_crops.py            keeps sharpness >= 4  ->  28 528 crops / 12 900 cells
```

Cleaning is a **filter on the manifest**, so the threshold moves without re-cutting.
Full retention table: `results/leukemiaattr_retention.md`.

| Class | Before | After |
|---|---|---|
| `lymphoblast` | 130 crops, no patients | **2 234 cells / 19 ALL patients** |
| `lymphocyte_atypical` | 11 | **656** |
| `promyelocyte_abnormal` | 18 | **438** |

Annotations live in `json_labels/` for most subdomains but inside `labels.zip` for
`L_100X_C2`; the extractor reads either. Class names come from the COCO `categories`
block — never from the numeric ids of the `.txt` files, whose documented mapping is wrong.

**Two limits** (`PREREGISTRATION.md` amendment 3c): APML and CLL have one patient per
split, so those cells train and never test; and the annotation is incomplete (~4.3 cells
per field), so it cannot measure localizer recall.


## MILLIE / AML_APL — what closed the smudge-cell gap

106 patients (APL 34 / AML 72), `Patient_NN/Signed_slides/<class>/*.jpg`, 360×363 centred
single cells at sharpness 29–152 (the PBC band). Verified visually.

| MILLIE class | Images | Patients | → ASTER class |
|---|---|---|---|
| `Lymphocyte` | 1 902 | 56 | `lymphocyte` |
| `Blast_no_lineage_spec` | 1 865 | 49 | **partial** {myeloblast, lymphoblast} |
| **`Smudge_cells`** | **1 555** | **56** | **`smudge_cell`** |
| `Segmented_neutrophils` | 1 063 | 53 | `segmented_neutrophil` |
| `Monocyte` / `Promonocyte` | 878 | 46 | `monocyte` |
| `Promyelocyte` | 364 | 21 | `promyelocyte` (normal — see below) |
| `Erythroblast` | 216 | 32 | `erythroblast` |
| `Lymphocyte_variant` | 162 | 33 | `lymphocyte_atypical` |
| others | 265 | — | 1:1 |

`smudge_cell` goes from **15 cells to 1 570 across 56 patients**. Both §2 of the
pre-registration and amendment 3 had declared this class unfixable by any available
dataset; that was wrong, and amendment 4 records it.

**Two mappings that are declared, not assumed.** `Blast_no_lineage_spec` keeps a partial
label because the annotation says the lineage was not specified — the cohort is AML/APL,
but the cohort is not the annotation. `Promyelocyte` maps to the *normal* class even though
231 of its 364 cells come from APL patients: MILLIE does not label them abnormal, so the
question is left to a later analysis on evidence. `Plasma_cells` (6 images) is skipped
rather than filed as an artifact, and the 17 427 unlabelled `Unsigned_slides` are not
enumerated.
