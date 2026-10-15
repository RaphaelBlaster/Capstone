#!/usr/bin/env python3
"""Offline arithmetic and alignment check for the local manuscript evidence archive.

Uses only the Python standard library. It reads copies in evidence/predictions and
evidence/manifests; it does not load models, contact services, or modify source data.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path

TARGETS = ("Dry_Green_g", "Dry_Dead_g", "Dry_Clover_g", "GDM_g", "Dry_Total_g")
WEIGHTS = (0.1, 0.1, 0.1, 0.2, 0.5)


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8-sig") as f:
        rows = list(csv.DictReader(f))
    if not rows:
        raise ValueError(f"empty CSV: {path}")
    return rows


def first(row: dict[str, str], *keys: str) -> str:
    for key in keys:
        if key in row and row[key] != "":
            return row[key]
    raise ValueError(f"missing required column among {keys}")


def metric(rows: list[dict[str, str]]) -> dict:
    # Official raw global weighted R2: one weighted mean and denominator over all
    # image-target observations, rather than a mean of five separately centered R2s.
    y = [[float(r[t]) for t in TARGETS] for r in rows]
    p = [[float(r["pred_" + t]) for t in TARGETS] for r in rows]
    if not all(math.isfinite(v) for pair in y + p for v in pair):
        raise ValueError("non-finite label or prediction")
    n = len(y)
    grand = sum(sum(v * w for v, w in zip(row, WEIGHTS)) for row in y) / n
    numerator = sum(w * (a - b) ** 2 for ya, pa in zip(y, p)
                    for a, b, w in zip(ya, pa, WEIGHTS))
    denominator = sum(w * (a - grand) ** 2 for ya in y
                      for a, w in zip(ya, WEIGHTS))
    if denominator <= 0:
        raise ValueError("non-positive score denominator")
    return {"raw_global_weighted_r2": 1.0 - numerator / denominator,
            "observations": n, "weighted_grand_mean_g": grand}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("archive", type=Path, nargs="?", default=Path(
        "outputs/manuscript_release_pass_2026_10_11/evidence"))
    args = parser.parse_args()
    root = args.archive.resolve()
    manifest_path = root / "manifests" / "labeled357_fold_manifest.csv"
    manifest_rows = read_csv(manifest_path)
    manifest = {}
    date_fold: dict[str, str] = {}
    for r in manifest_rows:
        image = r["image_path"]
        date = first(r, "Sampling_Date", "group_id")
        fold = r["fold"]
        if image in manifest:
            raise ValueError(f"duplicate manifest ID: {image}")
        if date in date_fold and date_fold[date] != fold:
            raise ValueError(f"date crosses folds in manifest: {date}")
        date_fold[date] = fold
        manifest[image] = (date, fold, tuple(float(r[t]) for t in TARGETS))
    if len(manifest) != 357:
        raise ValueError(f"expected complete 357-image manifest, found {len(manifest)}")

    results = {}
    prediction_files = sorted((root / "predictions").glob("*.csv"))
    if not prediction_files:
        raise ValueError("no copied prediction CSVs")
    cohort_refs: dict[int, tuple[str, ...]] = {}
    for path in prediction_files:
        rows = read_csv(path)
        ids = []
        normalized = []
        for r in rows:
            image = r["image_path"]
            if image not in manifest:
                raise ValueError(f"{path.name}: unknown manifest ID {image}")
            date = first(r, "Sampling_Date", "group_id")
            fold = r["fold"]
            mdate, mfold, labels = manifest[image]
            if date != mdate or fold != mfold:
                raise ValueError(f"{path.name}: date/fold mismatch for {image}")
            current_labels = tuple(float(r[t]) for t in TARGETS)
            if any(abs(a - b) > 1e-9 for a, b in zip(current_labels, labels)):
                raise ValueError(f"{path.name}: label mismatch for {image}")
            if date_fold[date] != fold:
                raise ValueError(f"{path.name}: date-fold disjointness failure: {date}")
            ids.append(image)
            normalized.append({**r, "_date": date, "_fold": fold})
        if len(ids) != len(set(ids)):
            raise ValueError(f"{path.name}: duplicate image IDs")
        cohort = len(ids)
        if cohort not in (105, 167):
            raise ValueError(f"{path.name}: unsupported publication cohort size {cohort}")
        if set(r["_fold"] for r in normalized) not in ({"0", "1"}, {"0", "1", "2"}):
            raise ValueError(f"{path.name}: unexpected folds")
        if cohort in cohort_refs and tuple(sorted(ids)) != cohort_refs[cohort]:
            raise ValueError(f"{path.name}: image IDs differ from other {cohort}-image arms")
        cohort_refs[cohort] = tuple(sorted(ids))
        results[path.name] = {
            **metric(normalized), "images": cohort,
            "folds": sorted({r["_fold"] for r in normalized}, key=int),
            "date_groups": len({r["_date"] for r in normalized}),
            "sha256": sha256(path),
        }

    # Known paired identities: duplicate F2/depth2 saved files must be byte identical.
    by_name = {p.name: p for p in prediction_files}
    if {"F2_pooled_predictions.csv", "depth_2_pooled_predictions.csv"} <= by_name.keys():
        if sha256(by_name["F2_pooled_predictions.csv"]) != sha256(by_name["depth_2_pooled_predictions.csv"]):
            raise ValueError("F2 and depth-2 input predictions are expected byte-identical")

    report = {
        "verifier": "tools/verify_release_predictions.py",
        "verification": "PASS",
        "scope": "copied prediction CSVs only; development data, not full OOF or competition scoring",
        "manifest": {"images": len(manifest), "date_groups": len(date_fold),
                     "folds": sorted(set(v[1] for v in manifest.values()), key=int),
                     "sha256": sha256(manifest_path)},
        "arms": results,
    }
    out = root / "checks" / "independent_prediction_scores.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"verification": "PASS", "arms": len(results),
                      "cohorts": sorted(cohort_refs), "report": str(out)}, indent=2))


if __name__ == "__main__":
    main()
