"""
Build combined classification splits from:
- Segmentation dataset (imagesTr only) with inferred class labels
- Classification dataset (labels.csv)

Result:
- Writes train.csv, val.csv, test.csv under out_root
- Train = ALL segmentation images + 50% of classification images (stratified by label)
- Val   = 20% of classification images (stratified by label)
- Test  = 30% of classification images (stratified by label)

Usage example (in a notebook):

from build_classification_splits import build_combined_classification_splits
build_combined_classification_splits(
    seg_root="/abs/path/derived/segmentation_t1c_binary",
    cls_root="/abs/path/derived/classification_t1c",
    out_root="/abs/path/derived/classification_combined",
    seed=42
)
"""

from __future__ import annotations
from pathlib import Path
from typing import List, Dict, Tuple
import json
import csv
import os
import random


def _read_cls_csv(cls_root: str) -> List[Dict[str, str]]:
    root = Path(cls_root)
    lb = root / "labels.csv"
    rows: List[Dict[str, str]] = []
    with lb.open("r", newline="") as f:
        rd = csv.DictReader(f)
        for r in rd:
            rows.append({
                "case_id": r["case_id"],
                "label": r["label"],
                "image_path": str(Path(r["image_path"]).resolve()),
            })
    return rows


def _infer_label_from_seg_case_id(case_id: str) -> int:
    cid = case_id.lower()
    # Heuristics: BCBM -> mets (1); BRATS -> glioma (0)
    if "bcbm" in cid:
        return 1
    if "brats" in cid:
        return 0
    # Fallback: default to glioma
    return 0


def _read_seg_images(seg_root: str) -> List[Dict[str, str]]:
    root = Path(seg_root)
    images_dir = root / "imagesTr"
    if not images_dir.is_dir():
        raise FileNotFoundError(f"imagesTr not found in {seg_root}")
    rows: List[Dict[str, str]] = []
    for img in sorted(list(images_dir.glob("*.nii.gz")) + list(images_dir.glob("*.nii"))):
        case_id = img.name[:-7] if img.name.endswith(".nii.gz") else (img.name[:-4] if img.name.endswith(".nii") else img.stem)
        label = _infer_label_from_seg_case_id(case_id)
        rows.append({
            "case_id": case_id,
            "label": str(label),
            "image_path": str(img.resolve()),
        })
    return rows


def _stratified_split_by_case(rows: List[Dict[str, str]], train_frac: float, val_frac: float, test_frac: float, seed: int = 42) -> Tuple[List[str], List[str], List[str]]:
    assert abs(train_frac + val_frac + test_frac - 1.0) < 1e-6
    # group by case_id with a single label per case
    case_to_label: Dict[str, int] = {}
    for r in rows:
        cid = r["case_id"]
        lab = int(r["label"])
        case_to_label.setdefault(cid, lab)
    # separate case_ids by class
    class_to_cases: Dict[int, List[str]] = {}
    for cid, lab in case_to_label.items():
        class_to_cases.setdefault(lab, []).append(cid)
    rnd = random.Random(seed)
    train_cases: List[str] = []
    val_cases: List[str] = []
    test_cases: List[str] = []
    for lab, cids in class_to_cases.items():
        rnd.shuffle(cids)
        n = len(cids)
        n_train = int(round(train_frac * n))
        n_val = int(round(val_frac * n))
        # ensure total matches
        n_test = max(0, n - n_train - n_val)
        train_cases.extend(cids[:n_train])
        val_cases.extend(cids[n_train:n_train + n_val])
        test_cases.extend(cids[n_train + n_val:])
    return train_cases, val_cases, test_cases


def build_combined_classification_splits(seg_root: str, cls_root: str, out_root: str, seed: int = 42) -> Dict[str, str]:
    out = Path(out_root)
    out.mkdir(parents=True, exist_ok=True)

    # Load datasets
    seg_rows = _read_seg_images(seg_root)               # used entirely for training
    cls_rows = _read_cls_csv(cls_root)                  # will be split 50/20/30

    # Split classification dataset by case (stratified)
    train_cases, val_cases, test_cases = _stratified_split_by_case(cls_rows, 0.5, 0.2, 0.3, seed=seed)
    train_cls = [r for r in cls_rows if r["case_id"] in train_cases]
    val_cls = [r for r in cls_rows if r["case_id"] in val_cases]
    test_cls = [r for r in cls_rows if r["case_id"] in test_cases]

    # Compose final splits
    train_all = seg_rows + train_cls
    # Write CSVs
    def _write_csv(path: Path, rows: List[Dict[str, str]]):
        with path.open("w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["case_id", "label", "image_path"])  # consistent header
            for r in rows:
                w.writerow([r["case_id"], r["label"], r["image_path"]])

    _write_csv(out / "train.csv", train_all)
    _write_csv(out / "val.csv", val_cls)
    _write_csv(out / "test.csv", test_cls)

    # Simple report
    print(f"[DONE] Wrote splits to {out_root}")
    print(f" - Train: {len(train_all)} (seg={len(seg_rows)}, cls={len(train_cls)})")
    print(f" - Val:   {len(val_cls)}")
    print(f" - Test:  {len(test_cls)}")

    return {
        "train_csv": str((out / "train.csv").resolve()),
        "val_csv": str((out / "val.csv").resolve()),
        "test_csv": str((out / "test.csv").resolve()),
    }


