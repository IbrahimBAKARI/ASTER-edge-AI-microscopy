#!/usr/bin/env python3
"""Re-score the 409 cAItomorph patients ON THE DEVICE and compare with the Colab run.

The paper's primary test (PREREGISTRATION.md §6.3) was run on Colab. This script replays it
patient by patient on the Jetson from the same crops, in the same order, with the kit's
scorer (TensorRT or PyTorch), then:
  1. compares every patient with results/caitomorph_409_predictions.csv of the run
     (label, tier, N_c, P_abn, OOD score);
  2. recomputes the headline figures from the DEVICE's own predictions: P_abn AUROC with the
     same bootstrap (seed 0, 2000), sensitivity/specificity at the frozen tau, the grid's
     specificity/sensitivity/abstention - and prints them next to the reported values.

Nothing is fitted here. Pure standard library + numpy + scikit-learn (no pandas).

  python3 verify_caitomorph_409.py --corpus ../data/corpus --manifest ../data/corpus/manifest.csv \
      --expected ../run/20260911_051357Z/results/caitomorph_409_predictions.csv --output out/cai409
  (--limit 20 for a quick smoke test; --backend torch for the fp32 reference path)
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
import time
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import torch
from PIL import Image

HERE = Path(__file__).resolve().parent
csv.field_size_limit(sys.maxsize)

HEALTHY, REACTIVE, ACUTE = ["Stem cell donor"], ["Reactive changes"], ["AML", "ALL", "AL"]
REPORTED = {  # the figures of the run, for the side-by-side print
    "AUROC AML vs donors": "0.973 [0.937, 0.998]",
    "AUROC acute vs donors+reactive": "0.967 [0.935, 0.991]",
    "P_abn sensitivity AML at tau": "19/37", "P_abn specificity donors at tau": "99/99",
    "grid specificity vs donors": "95/99", "grid specificity vs reactive": "39/42",
    "grid sensitivity acute": "22/46", "grid indeterminate": "182/409",
    "grid out_of_domain": "46/409",
}


def wilson(k: int, n: int) -> str:
    if n == 0:
        return "n=0"
    z, p = 1.959963984540054, k / n
    centre = (p + z * z / (2 * n)) / (1 + z * z / n)
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n)
    return f"{k}/{n} = {p:.3f} [{centre - half:.3f}, {centre + half:.3f}]"


def auroc_block(rows: list[dict]) -> dict:
    from sklearn.metrics import roc_auc_score
    rng = np.random.default_rng(0)      # same seed and call order as the run's reported cell
    out = {}
    for name, pos, neg in (("AUROC AML vs donors", ["AML"], HEALTHY),
                           ("AUROC acute vs donors+reactive", ACUTE, HEALTHY + REACTIVE)):
        s = [r for r in rows if r["diagnosis_fine"] in pos + neg]
        y = np.array([r["diagnosis_fine"] in pos for r in s], dtype=int)
        p = np.array([r["p_abn"] for r in s], dtype=float)
        if len(set(y)) < 2:
            continue
        i_pos, i_neg = np.where(y == 1)[0], np.where(y == 0)[0]
        yb = np.r_[np.ones(len(i_pos)), np.zeros(len(i_neg))]
        boot = [roc_auc_score(yb, np.r_[p[rng.choice(i_pos, len(i_pos))], p[rng.choice(i_neg, len(i_neg))]])
                for _ in range(2000)]
        lo, hi = np.percentile(boot, [2.5, 97.5])
        out[name] = f"{roc_auc_score(y, p):.3f} [{lo:.3f}, {hi:.3f}]"
    return out


def headline(rows: list[dict], tau: float) -> dict:
    by = lambda names: [r for r in rows if r["diagnosis_fine"] in names]
    leuk = lambda label: label.startswith("acute_blastic") or label.startswith("chronic_")
    aml, don, rea, acu = by(["AML"]), by(HEALTHY), by(REACTIVE), by(ACUTE)
    figures = auroc_block(rows)
    figures["P_abn sensitivity AML at tau"] = wilson(sum(r["p_abn"] >= tau for r in aml), len(aml))
    figures["P_abn specificity donors at tau"] = wilson(sum(r["p_abn"] < tau for r in don), len(don))
    figures["grid specificity vs donors"] = wilson(sum(not leuk(r["label"]) for r in don), len(don))
    figures["grid specificity vs reactive"] = wilson(sum(not leuk(r["label"]) for r in rea), len(rea))
    figures["grid sensitivity acute"] = wilson(sum(r["label"].startswith("acute_blastic") for r in acu), len(acu))
    figures["grid indeterminate"] = wilson(sum(r["label"] == "indeterminate" for r in rows), len(rows))
    figures["grid out_of_domain"] = wilson(sum(r["label"] == "out_of_domain" for r in rows), len(rows))
    return figures


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--corpus", required=True, help="root the manifest's corpus_path is relative to")
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--expected", required=True, help="results/caitomorph_409_predictions.csv of the run")
    ap.add_argument("--kit", default=str(HERE.parent / "kit"))
    ap.add_argument("--backend", choices=("tensorrt", "torch"), default="tensorrt")
    ap.add_argument("--engine", default=None)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--workers", type=int, default=4, help="threads decoding crops")
    ap.add_argument("--output", required=True)
    args = ap.parse_args()

    sys.path.insert(0, str(Path(args.kit).resolve()))
    from load_kit import load_scorer
    scorer = load_scorer(Path(args.kit), backend=args.backend,
                         engine=Path(args.engine) if args.engine else None)
    out = Path(args.output); out.mkdir(parents=True, exist_ok=True)

    # patient -> crops, in manifest order: exactly `cai.groupby("patient_id")` of the notebook
    crops, diagnosis = defaultdict(list), {}
    with open(args.manifest, newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            if row["source"] == "caitomorph":
                crops[row["patient_id"]].append(row["corpus_path"])
                diagnosis[row["patient_id"]] = row["bag_label"]
    with open(args.expected, newline="", encoding="utf-8") as handle:
        expected = {r["patient_id"]: r for r in csv.DictReader(handle)}
    patients = sorted(crops)
    if args.limit:
        patients = patients[:args.limit]
    print(f"{len(patients)} patients, {sum(len(crops[p]) for p in patients)} crops, backend {args.backend}")

    corpus = Path(args.corpus)
    pool = ThreadPoolExecutor(args.workers)
    load = lambda rel: scorer.transform(Image.open(corpus / rel).convert("RGB"))
    rows, started = [], time.perf_counter()
    for index, patient in enumerate(patients, 1):
        tensors = torch.stack(list(pool.map(load, crops[patient])))
        result, detail = scorer.score(tensors, session_id=patient)
        exp = expected.get(patient, {})
        rows.append({"patient_id": patient, "diagnosis_fine": diagnosis[patient],
                     "label": result.label, "tier": result.tier,
                     "n_classified": result.number_of_classified_leukocytes,
                     "p_abn": float(result.uncertainty["p_abn"]),
                     "ood": float(result.uncertainty["ood_score"]),
                     "expected_label": exp.get("label"), "expected_p_abn": float(exp.get("p_abn", "nan")),
                     "expected_n_classified": int(exp.get("n_classified", -1) or -1),
                     "label_match": result.label == exp.get("label")})
        if index % 25 == 0 or index == len(patients):
            rate = index / (time.perf_counter() - started)
            print(f"  {index}/{len(patients)}  ({rate:.2f} patients/s)  "
                  f"label agreement so far {sum(r['label_match'] for r in rows)}/{len(rows)}")

    with open(out / "caitomorph_device_predictions.csv", "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)

    agree = sum(r["label_match"] for r in rows)
    dp = [abs(r["p_abn"] - r["expected_p_abn"]) for r in rows if not math.isnan(r["expected_p_abn"])]
    dn = [abs(r["n_classified"] - r["expected_n_classified"]) for r in rows]
    changed = Counter((r["expected_label"], r["label"]) for r in rows if not r["label_match"])
    summary = {"backend": args.backend, "patients": len(rows), "label_agreement": f"{agree}/{len(rows)}",
               "p_abn_abs_diff_max": max(dp, default=None), "p_abn_abs_diff_median": float(np.median(dp)) if dp else None,
               "n_classified_abs_diff_max": max(dn, default=None),
               "label_changes": {f"{a} -> {b}": n for (a, b), n in changed.items()},
               "seconds": time.perf_counter() - started}
    print(f"\nlabel agreement with Colab: {agree}/{len(rows)}   max |dP_abn| {summary['p_abn_abs_diff_max']:.4f}")
    for change, n in summary["label_changes"].items():
        print(f"  {change}: {n}")
    if len(rows) == len(crops):
        figures = headline(rows, scorer.thresholds.p_abn)
        summary["device_headline"] = figures
        print(f"\n{'figure':<34}{'device':<34}reported (Colab)")
        for name, value in figures.items():
            print(f"{name:<34}{value:<34}{REPORTED.get(name, '')}")
    (out / "summary.json").write_text(json.dumps(summary, indent=2))
    return 0 if agree == len(rows) else 1


if __name__ == "__main__":
    sys.exit(main())
