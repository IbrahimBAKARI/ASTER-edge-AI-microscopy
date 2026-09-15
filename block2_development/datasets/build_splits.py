"""Assign every patient (or image, where there is no patient) to a split.

Discipline, from PREREGISTRATION.md sections 5.2 and 6.3:
  * patient-level everywhere a patient id exists;
  * the development split used to fit thresholds is DISJOINT from the patients the
    models were trained on - otherwise a threshold is fitted on the model's own
    training data and looks better than it is;
  * cAItomorph is test_final and is never trained on, never fitted on;
  * the x40 prototype crops are 'stress' only.

Splits produced
  aml_mll   train_mil   0.45   trains the MIL head
            mil_val     0.15   EARLY STOPPING only - model selection is a training
                               decision, so it is paid for out of the training budget and
                               never out of dev_fit / dev_cal. Without this carve-out the
                               stopping epoch and the operating points would be chosen on
                               the same patients, and tau_abn would look better than it is.
            dev_fit     0.20   fits tau_abn, theta_APL, the OOD threshold, N_c/N floor
            dev_cal     0.20   temperature calibration + the [REF] reference intervals
                               (control patients of this split only)
  aml_lmu / all_idb2 / taleqani / pbc
            cell_train  0.80   image-level, stratified by cell class
            cell_val    0.20   -- no patient ids exist, so cells of one patient may
                               appear in both; declared as a limitation, these sources
                               are never used for a patient-level claim
  caitomorph  test_final
  x40_proto   stress

Usage: python datasets/build_splits.py --manifest _work/corpus/manifest.csv
"""

from __future__ import annotations

import argparse
import csv
import json
import random
from collections import Counter, defaultdict
from pathlib import Path

SEED = 42
PATIENT_FRACTIONS = {"train_mil": 0.45, "mil_val": 0.15, "dev_fit": 0.20, "dev_cal": 0.20}
CELL_FRACTIONS = {"cell_train": 0.80, "cell_val": 0.20}


def _allocate(items: list, fractions: dict[str, float], rng: random.Random) -> dict[str, str]:
    items = sorted(items)
    rng.shuffle(items)
    assignment, start, total = {}, 0, len(items)
    names = list(fractions)
    for index, name in enumerate(names):
        end = total if index == len(names) - 1 else start + int(round(fractions[name] * total))
        for item in items[start:end]:
            assignment[item] = name
        start = end
    return assignment


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", default="_work/corpus/manifest.csv")
    parser.add_argument("--out", default="_work/corpus/splits.csv")
    args = parser.parse_args()

    rows = list(csv.DictReader(Path(args.manifest).open(newline="", encoding="utf-8")))
    rng = random.Random(SEED)

    # --- patient-level sources ------------------------------------------------
    patients_by_label: dict[str, list[str]] = defaultdict(list)
    seen = set()
    for row in rows:
        if row["source"] != "aml_mll" or not row["patient_id"]:
            continue
        if row["patient_id"] in seen:
            continue
        seen.add(row["patient_id"])
        patients_by_label[row["bag_label"]].append(row["patient_id"])

    patient_split: dict[str, str] = {}
    for label, patients in sorted(patients_by_label.items()):  # stratified by bag_label
        patient_split.update(_allocate(patients, PATIENT_FRACTIONS, rng))

    # --- MILLIE: patient-level, but NOT on the shipped cohorts ----------------
    # The Discovery / Validation cohorts do not separate labelled cells: all 56 patients
    # carrying Signed_slides are Discovery, the 24 Validation patients hold only unlabelled
    # Unsigned_slides. Using the shipped cohorts would put every labelled cell in train and
    # leave `smudge_cell` - 99 % of which comes from MILLIE - with no validation cell at
    # all. The quantifier could then not measure its TPR, would declare `smudge_frac`
    # unquantifiable, and R3 would return indeterminate by construction.
    # So the 56 labelled patients get their own patient-level split, stratified by
    # diagnosis. See PREREGISTRATION.md amendment 7.
    millie_by_diagnosis: dict[str, list[str]] = defaultdict(list)
    seen_millie = set()
    for row in rows:
        if row["source"] != "millie" or row["patient_id"] in seen_millie:
            continue
        seen_millie.add(row["patient_id"])
        millie_by_diagnosis[row["bag_label"] or "?"].append(row["patient_id"])
    millie_split: dict[str, str] = {}
    for diagnosis, patients in sorted(millie_by_diagnosis.items()):
        millie_split.update(_allocate(patients, CELL_FRACTIONS, rng))

    # --- image-level cell sources --------------------------------------------
    cell_sources = {"aml_lmu", "all_idb2", "pbc"}   # leukemiaattr keeps its own split
    by_class: dict[str, list[str]] = defaultdict(list)
    for row in rows:
        if row["source"] in cell_sources:
            by_class[row["cell_labels"]].append(row["corpus_path"])
    image_split: dict[str, str] = {}
    for label, paths in sorted(by_class.items()):
        image_split.update(_allocate(paths, CELL_FRACTIONS, rng))

    # --- write ----------------------------------------------------------------
    counts: Counter = Counter()
    out_path = Path(args.out)
    with out_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["corpus_path", "source", "patient_id", "split"])
        for row in rows:
            source = row["source"]
            if source == "caitomorph":
                split = "test_final"
            elif source == "x40_proto":
                split = "stress"
            elif source == "aml_mll":
                split = patient_split.get(row["patient_id"], "train_mil")
            elif source == "millie":
                split = millie_split.get(row["patient_id"], "cell_train")
            elif source == "leukemiaattr":
                # the source's own patient-disjoint split is authoritative; redrawing it
                # would risk splitting the acquisitions of one physical cell
                extra = json.loads(row["extra"]) if row.get("extra") else {}
                split = "cell_train" if extra.get("la_split") == "train" else "cell_val"
            elif source in cell_sources:
                split = image_split.get(row["corpus_path"], "cell_train")
            else:
                split = "unassigned"
            counts[(source, split)] += 1
            writer.writerow([row["corpus_path"], source, row["patient_id"], split])

    summary = defaultdict(dict)
    for (source, split), n in sorted(counts.items()):
        summary[source][split] = n
    print(json.dumps(summary, indent=2))
    n_patients = Counter(patient_split.values())
    print(f"\naml_mll patients: {dict(n_patients)}  (total {len(patient_split)})")
    print(f"written: {out_path}")

    overlap = set()
    for row in rows:
        if row["source"] == "aml_mll" and row["patient_id"]:
            overlap.add((row["patient_id"], patient_split.get(row["patient_id"])))
    by_patient = defaultdict(set)
    for patient, split in overlap:
        by_patient[patient].add(split)
    leaks = [p for p, s in by_patient.items() if len(s) > 1]
    print(f"patient leakage check: {'FAIL ' + str(leaks) if leaks else 'OK, no patient in two splits'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
