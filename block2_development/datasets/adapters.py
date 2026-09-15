"""One adapter per local dataset: enumerate images into uniform records.

A record is a dict with:
    path                absolute path to the source image
    source              dataset key
    patient_id          str or None (None = no patient structure -> low weight, train only)
    role                'train' | 'test' | 'stress'
    cell_labels         list[str] of allowed cell classes, or None
                        len 1  -> a definite cell label
                        len >1 -> a PARTIAL label (the true class is one of these)
                        None   -> no cell supervision
    weight              per-sample loss weight
    bag_label           source patient-level label, or None
    extra               dict of anything else worth carrying

Directory layouts were verified on disk on 2026-09-10; see DATASETS.md.
"""

from __future__ import annotations

import csv
from pathlib import Path

# --- the 15 morphological classes of the frozen grid -------------------------
CELL_CLASSES = [
    "myeloblast", "lymphoblast", "promyelocyte", "promyelocyte_abnormal",
    "myelocyte", "metamyelocyte", "band_neutrophil", "segmented_neutrophil",
    "basophil", "eosinophil", "monocyte", "lymphocyte", "lymphocyte_atypical",
    "smudge_cell", "erythroblast",
]
NON_BLAST = [c for c in CELL_CLASSES if c not in ("myeloblast", "lymphoblast", "promyelocyte_abnormal")]
IMMATURE_GRANULOCYTE = ["promyelocyte", "myelocyte", "metamyelocyte"]

# Matek 2019 (AML-Cytomorphology_LMU) legend -> frozen grid classes
LMU_CLASS_MAP = {
    "BAS": "basophil",
    "EBO": "erythroblast",
    "EOS": "eosinophil",
    "KSC": "smudge_cell",
    "LYA": "lymphocyte_atypical",
    "LYT": "lymphocyte",
    "MMZ": "metamyelocyte",
    "MOB": "myeloblast",           # monoblast: a blast of monocytic (myeloid) lineage
    "MON": "monocyte",
    "MYB": "myelocyte",
    "MYO": "myeloblast",
    "NGB": "band_neutrophil",
    "NGS": "segmented_neutrophil",
    "PMB": "promyelocyte_abnormal",  # bilobed promyelocyte - the APL-associated form
    "PMO": "promyelocyte",
}

IMAGE_SUFFIXES = {".tif", ".tiff", ".png", ".jpg", ".jpeg", ".bmp", ".webp"}


def _images(directory: Path):
    if not directory.is_dir():
        return
    for path in sorted(directory.rglob("*")):
        if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES and not path.name.startswith("."):
            yield path


# ---------------------------------------------------------------------------
# AML-Cytomorphology MLL Helmholtz v1  ->  MIL training + APL development
#   layout: <root>/<bag_label>/<patient_id>/image_N.tif
#   metadata: patient_id, sex, age, bag_label, instance_count, leucocytes_per_ul,
#             pb_* differential columns (percentages, pb_total = 100)
# ---------------------------------------------------------------------------
def aml_mll(root: Path):
    root = Path(root)
    meta = {}
    for name in ("metadata_with_data_dictionary.csv_metadata.csv", "metadata.csv"):
        candidate = root / name
        if candidate.exists():
            with candidate.open(newline="", encoding="utf-8-sig") as handle:
                for row in csv.DictReader(handle):
                    meta[row["patient_id"]] = row
            break
    for bag_dir in sorted(p for p in root.iterdir() if p.is_dir()):
        bag_label = bag_dir.name
        for patient_dir in sorted(p for p in bag_dir.iterdir() if p.is_dir()):
            patient_id = patient_dir.name
            row = meta.get(patient_id, {})
            for path in _images(patient_dir):
                yield dict(
                    path=path, source="aml_mll", patient_id=patient_id, role="train",
                    cell_labels=None, weight=1.0, bag_label=bag_label,
                    extra={"differential": {k: v for k, v in row.items() if k.startswith("pb_")},
                           "age": row.get("age"), "sex": row.get("sex_1f_2m")},
                )


# ---------------------------------------------------------------------------
# AML-Cytomorphology_LMU (Matek 2019)  ->  cell-head training
#   layout: <root>/<CLASS>/<CLASS>_NNNN.tiff ; annotations carry no patient id
# ---------------------------------------------------------------------------
def aml_lmu(root: Path):
    root = Path(root)
    for class_dir in sorted(p for p in root.iterdir() if p.is_dir()):
        mapped = LMU_CLASS_MAP.get(class_dir.name)
        if mapped is None:
            continue
        for path in _images(class_dir):
            yield dict(
                path=path, source="aml_lmu", patient_id=None, role="train",
                cell_labels=[mapped], weight=1.0, bag_label=None,
                extra={"lmu_class": class_dir.name},
            )


# ---------------------------------------------------------------------------
# ALL-IDB2  ->  lymphoblast cell labels (image level, no patient id)
#   'cancer' = lymphoblast ; 'non cancer' = known NOT a blast (partial label)
# ---------------------------------------------------------------------------
def all_idb2(root: Path, weight: float = 0.3):
    root = Path(root)
    for sub, labels in (("cancer", ["lymphoblast"]), ("non cancer", NON_BLAST)):
        for path in _images(root / sub):
            yield dict(
                path=path, source="all_idb2", patient_id=None, role="train",
                cell_labels=labels, weight=weight, bag_label=None,
                extra={"idb2_class": sub},
            )


# ---------------------------------------------------------------------------
# ALL_IDB_Dataset  ->  DISABLED in v1. Inspected on disk 2026-09-10:
#   L1  50 full fields 2592x1944 + YOLO .txt boxes
#   L2  50 full fields 1712x1368 + YOLO .txt boxes
#   L3  50 crops 257x257 (38 jpg + 12 tif)
# So L1/L2 are ALL-IDB1 FULL FIELDS, not crops, and "L1/L2/L3" is not the FAB
# subtype. The 493 annotated boxes carry the single class "Candidate_Cell" from a
# labelling tool (the folder also contains a stray class list with entries such as
# "car", "dog", "hamburger"): they are WBC LOCALISATION boxes, not cell-type labels.
# The filename _0/_1 convention is not applied consistently across the folder either.
#
# Marginal value is near zero next to ALL-IDB2 (260 clean crops) and Taleqani (3256),
# while the label semantics are unclear. Rather than inject silent label noise into the
# class that the tier-B ALL claim depends on, the source is excluded from v1 and the
# exclusion is documented. Set enabled: true in sources.yaml to reinstate it, and
# resolve the box semantics first.
# ---------------------------------------------------------------------------
def all_idb_fields(root: Path, weight: float = 0.3):
    """Crops from the annotated boxes, with NO cell label. Unused unless enabled."""
    root = Path(root)
    for sub in ("L1", "L2"):
        for label_file in sorted((root / sub).glob("*.txt")):
            image_path = label_file.with_suffix(".jpg")
            if not image_path.exists():
                continue
            box_index = 0
            for line in label_file.read_text(errors="ignore").splitlines():
                parts = line.split()
                if len(parts) != 5 or not parts[0].isdigit():
                    continue  # skips the stray class list
                yield dict(
                    path=image_path, source="all_idb_fields", patient_id=None, role="train",
                    cell_labels=None, weight=weight, bag_label=None,
                    extra={"box_norm": [float(v) for v in parts[1:]], "subset": sub,
                           "box_index": box_index},
                )
                box_index += 1
    for path in _images(root / "L3"):
        yield dict(
            path=path, source="all_idb_fields", patient_id=None, role="train",
            cell_labels=None, weight=weight, bag_label=None, extra={"subset": "L3"},
        )


# ---------------------------------------------------------------------------
# Taleqani hospital ALL set (Kaggle) -> lymphoblast, ORIGINAL images only.
# The 'Segmented' copies have the background removed and are NEVER used: block 1
# delivers crops with their real background, and a model trained on black
# backgrounds learns the mask instead of the cell.
# Filenames carry no patient identifier (verified 2026-09-10).
# ---------------------------------------------------------------------------
def taleqani(root: Path, weight: float = 0.3):
    root = Path(root) / "Original"
    for sub, labels in (
        ("Benign", NON_BLAST),
        ("Early", ["lymphoblast"]),
        ("Pre", ["lymphoblast"]),
        ("Pro", ["lymphoblast"]),
    ):
        for path in _images(root / sub):
            yield dict(
                path=path, source="taleqani", patient_id=None, role="train",
                cell_labels=labels, weight=weight, bag_label=None,
                extra={"stage": sub},
            )


# ---------------------------------------------------------------------------
# PBC / Acevedo 2020 (Mendeley 10.17632/snkd93bnjr.1)  ->  cell-head training
#
# Layout verified on disk 2026-09-10:
#   PBC_dataset/PBC_dataset/wbc/{neutrophil,monocyte,eosinophil,lymphocyte}
#   PBC_dataset/PBC_dataset/other_types/{basophil,ig,erythroblast,platelet}
#   PBC_dataset/PBC_dataset/wbc_resized/...   DUPLICATES - excluded
#   PBC_dataset_split/...                      DUPLICATES - excluded
# 9081 + 8012 = 17093 unique images, matching the published 17092.
#
# The filenames carry a FINER label than the folder names, which is what makes this
# source valuable rather than merely diverse:
#   neutrophil/  BNE -> band, SNE -> segmented   (LMU has 109 bands; PBC has 1633)
#   ig/          PMY -> promyelocyte  (592)      (LMU: 70)
#                MY  -> myelocyte     (1137)     (LMU: 42)
#                MMY -> metamyelocyte (1015)     (LMU: 15)
#                IG  -> unspecified   (151)      -> partial label over the three
# Those four classes are exactly the ones the CML left-shift rule depends on.
#
# Platelets are not leukocytes; they train `other_artifact`, the reject class.
# No patient identifiers: image-level split, and PBC cannot contribute to the [REF]
# reference intervals, which are per-patient (see PREREGISTRATION.md amendment 1).
# ---------------------------------------------------------------------------
PBC_PREFIX_MAP = {
    "BNE": ["band_neutrophil"],
    "SNE": ["segmented_neutrophil"],
    "NEUTROPHIL": ["band_neutrophil", "segmented_neutrophil"],
    "PMY": ["promyelocyte"],
    "MY": ["myelocyte"],
    "MMY": ["metamyelocyte"],
    "IG": IMMATURE_GRANULOCYTE,
    "BA": ["basophil"],
    "EO": ["eosinophil"],
    "LY": ["lymphocyte"],
    "MO": ["monocyte"],
    "ERB": ["erythroblast"],
    "PLATELET": ["other_artifact"],
}
PBC_FOLDER_FALLBACK = {
    "basophil": ["basophil"], "eosinophil": ["eosinophil"], "lymphocyte": ["lymphocyte"],
    "monocyte": ["monocyte"], "erythroblast": ["erythroblast"],
    "platelet": ["other_artifact"], "ig": IMMATURE_GRANULOCYTE,
    "neutrophil": ["band_neutrophil", "segmented_neutrophil"],
}


def pbc(root: Path, weight: float = 1.0):
    root = Path(root)
    base = root / "PBC_dataset" / "PBC_dataset"
    if not base.is_dir():
        base = root
    for group in ("wbc", "other_types"):          # wbc_resized is the same images
        for class_dir in sorted(p for p in (base / group).glob("*") if p.is_dir()):
            fallback = PBC_FOLDER_FALLBACK.get(class_dir.name.lower())
            for path in _images(class_dir):
                prefix = path.stem.split("_")[0].upper()
                labels = PBC_PREFIX_MAP.get(prefix, fallback)
                if labels is None:
                    continue
                yield dict(
                    path=path, source="pbc", patient_id=None, role="train",
                    cell_labels=labels, weight=weight, bag_label=None,
                    extra={"folder": class_dir.name, "prefix": prefix},
                )


# ---------------------------------------------------------------------------
# LeukemiaAttr, cleaned crops  ->  cell-head training
#
# The source ships 640x640 FIELDS with COCO boxes. datasets/extract_leukemiaattr.py cuts
# them with the deployed block-1 convention and datasets/clean_crops.py keeps the crops
# whose measured sharpness is >= 4 (see _work/leukemiaattr_clean/SUMMARY.md). This adapter
# reads that cleaned manifest - it never touches the raw fields.
#
# Why it matters: `lymphoblast` goes from 130 crops with no patient identifier to 2001
# cells across 19 ALL patients; `lymphocyte_atypical` 11 -> 656; `promyelocyte_abnormal`
# 18 -> 438. Those are the classes the ALL branch, the reactive-lymphoid exclusion and the
# APL flag rest on.
#
# The train/test split is the source's own, patient-disjoint and enforced globally, so it
# is carried through rather than redrawn. `cell_uid_nomag` groups the acquisitions of one
# physical cell: build_splits.py must never let two acquisitions of the same cell land on
# opposite sides.
# ---------------------------------------------------------------------------
def leukemiaattr(root: Path, weight: float = 1.0):
    root = Path(root)
    manifest = root / "manifest.csv"
    if not manifest.exists():
        return
    with manifest.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            path = root / row["path"]
            if not path.exists():
                continue
            # LeukemiaAttr's COCO class `none` means "not one of the 13 annotated classes",
            # NOT "artifact". Visual inspection of a random sample shows plainly
            # recognisable neutrophils, monocytes and lymphocytes among them. Training the
            # reject class on those would teach the model to discard real leukocytes, which
            # would shrink N_c, bias the differential non-randomly and corrupt the
            # min_classified_frac gate. Excluded - see PREREGISTRATION.md amendment 6a.
            if row["coco_class"] == "none":
                continue
            yield dict(
                path=path, source="leukemiaattr", patient_id=f"LA{row['patient']}",
                role="train", cell_labels=[row["cell_class"]], weight=weight,
                bag_label=row["diagnosis"],
                extra={"la_split": row["split"], "cell_uid": row["cell_uid_nomag"],
                       "subdomain": row["subdomain"], "sharpness": row["sharpness"]},
            )


# ---------------------------------------------------------------------------
# MILLIE / AML_APL (Manescu 2023)  ->  cell-head training + a patient-level APL cohort
#
# Layout verified on disk 2026-09-10:
#   AML_APL/master.csv                     106 patients, Diagnosis APL 34 / AML 72,
#                                          Cohort Discovery 82 / Validation 24
#   AML_APL/Patient_NN/Signed_slides/<Class>/*.jpg     8297 labelled single cells
#   AML_APL/Patient_NN/Unsigned_slides/*.jpg          17427 unlabelled - NOT enumerated,
#                                          they carry no label and have no declared use yet
#
# Images are 360x363 single centred cells (verified visually, not inferred from the
# dimension - the mistake made on Taleqani), sharpness 29-152, i.e. the PBC band and far
# above LeukemiaAttr.
#
# Why it matters most: `Smudge_cells`, 1555 images across 56 patients. AML-LMU has 15.
# Smudge cells are the central criterion of the chronic-lymphoid rule (R3), which was
# declared the weakest link of v1 before this source was read.
#
# Split: the dataset's own Discovery / Validation cohorts, carried through rather than
# redrawn. Patient-level by construction.
# ---------------------------------------------------------------------------
MILLIE_CLASS_MAP = {
    "Lymphocyte": ["lymphocyte"],
    # the annotation says the lineage was NOT specified, so it stays a partial label -
    # this is an AML/APL cohort, but inferring myeloid from the cohort is not annotation
    "Blast_no_lineage_spec": ["myeloblast", "lymphoblast"],
    "Smudge_cells": ["smudge_cell"],
    "Segmented_neutrophils": ["segmented_neutrophil"],
    "Band_neutrophils": ["band_neutrophil"],
    "Monocyte": ["monocyte"],
    "Promonocyte": ["monocyte"],          # as for LeukemiaAttr; CMML counts them together
    # MILLIE labels these `Promyelocyte`, not "abnormal promyelocyte". 231 of the 364 come
    # from APL patients and are probably the abnormal forms - but that is an inference, not
    # an annotation. They map to the normal class; the patient diagnosis is kept in the
    # manifest so a later analysis can settle it on evidence.
    "Promyelocyte": ["promyelocyte"],
    "Myelocyte": ["myelocyte"],
    "Metamyelocyte": ["metamyelocyte"],
    "Eosinophils": ["eosinophil"],
    "Basophil": ["basophil"],
    "Erythroblast": ["erythroblast"],
    "Lymphocyte_variant": ["lymphocyte_atypical"],
    "Giant_thrombocyte": ["other_artifact"],
    "Thrombocyte_aggregation": ["other_artifact"],
    "Arifact": ["other_artifact"],        # the folder is spelled this way in the source
    # Plasma_cells (6 images) has no counterpart in the frozen vocabulary and is skipped
    # rather than filed as an artifact.
}


def millie(root: Path, weight: float = 1.0):
    root = Path(root)
    meta = {}
    master = root / "master.csv"
    if master.exists():
        with master.open(newline="", encoding="utf-8-sig") as handle:
            for row in csv.DictReader(handle):
                meta[row["Patient_ID"]] = row
    for patient_dir in sorted(p for p in root.glob("Patient_*") if p.is_dir()):
        row = meta.get(patient_dir.name, {})
        cohort = (row.get("Cohort") or "Discovery").strip()
        for class_dir in sorted(p for p in (patient_dir / "Signed_slides").glob("*") if p.is_dir()):
            labels = MILLIE_CLASS_MAP.get(class_dir.name)
            if labels is None:
                continue
            for path in _images(class_dir):
                yield dict(
                    path=path, source="millie", patient_id=patient_dir.name, role="train",
                    cell_labels=labels, weight=weight, bag_label=row.get("Diagnosis"),
                    extra={"cohort": cohort, "millie_class": class_dir.name,
                           "age": row.get("Age at Diagnosis"), "sex": row.get("Gender")},
                )


# ---------------------------------------------------------------------------
# cAItomorph  ->  HELD-OUT TEST ONLY. Never train, never fit a threshold on it.
#   layout: <root>/patients/<patient_id>/*.TIF ; metadata.csv carries
#           diagnosis_coarse, diagnosis_fine, number_of_cells
# ---------------------------------------------------------------------------
def caitomorph(root: Path):
    root = Path(root)
    meta = {}
    with (root / "metadata.csv").open(newline="", encoding="utf-8-sig") as handle:
        for row in csv.DictReader(handle):
            meta[row["patient_id"]] = row
    for patient_dir in sorted(p for p in (root / "patients").iterdir() if p.is_dir()):
        row = meta.get(patient_dir.name, {})
        for path in _images(patient_dir):
            yield dict(
                path=path, source="caitomorph", patient_id=patient_dir.name, role="test",
                cell_labels=None, weight=0.0, bag_label=row.get("diagnosis_fine"),
                extra={"diagnosis_coarse": row.get("diagnosis_coarse"),
                       "number_of_cells": row.get("number_of_cells")},
            )


# ---------------------------------------------------------------------------
# Prototype x40 crops -> OOD reference and stress test. Unlabeled, non-leukemic.
# Produced by running block 1 on the x40 fields (notebook section 2).
#   layout: <root>/<session_id>/crops/*.png
# ---------------------------------------------------------------------------
def x40_crops(root: Path):
    root = Path(root)
    if not root.is_dir():
        return
    for session_dir in sorted(p for p in root.iterdir() if p.is_dir()):
        for path in _images(session_dir / "crops"):
            yield dict(
                path=path, source="x40_proto", patient_id=session_dir.name, role="stress",
                cell_labels=None, weight=0.0, bag_label="non_leukemic_material",
                extra={"session": session_dir.name},
            )


ADAPTERS = {
    "aml_mll": aml_mll,
    "aml_lmu": aml_lmu,
    "all_idb2": all_idb2,
    "all_idb_fields": all_idb_fields,
    "taleqani": taleqani,
    "pbc": pbc,
    "leukemiaattr": leukemiaattr,
    "millie": millie,
    "caitomorph": caitomorph,
    "x40_proto": x40_crops,
}
