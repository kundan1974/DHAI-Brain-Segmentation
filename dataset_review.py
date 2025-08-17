"""
Segmentation dataset reviewer (importable in notebooks)

Features:
- Parse nnU-Net-like dataset.json (imagesTr/ + labelsTr/)
- Interactive viewer with overlays (ipwidgets): case, plane, slice, labels, alpha, contour, normalize
- Quick summaries for shapes and label counts

Usage example (in a notebook):

from dataset_review import launch_segmentation_dataset_viewer, summarize_shapes_and_counts
root = "/absolute/path/to/derived/segmentation_t1c_binary"
summarize_shapes_and_counts(root)
launch_segmentation_dataset_viewer(root)
"""

from __future__ import annotations
from pathlib import Path
from typing import Dict, List, Tuple
import json
import warnings

import numpy as np
import nibabel as nib
import matplotlib.pyplot as plt

from ipywidgets import (
    Dropdown, SelectMultiple, IntSlider, FloatSlider, Checkbox, RadioButtons,
    HBox, VBox, Output, HTML, Layout
)
from matplotlib.colors import ListedColormap, to_rgba


# ---------------------------
# Dataset parsing and loading (RAS+, caching)
# ---------------------------

def _basename_without_nii_gz(p: Path) -> str:
    name = p.name
    if name.endswith('.nii.gz'):
        return name[:-7]
    if name.endswith('.nii'):
        return name[:-4]
    return p.stem


def _parse_dataset(root_path: str) -> List[Dict[str, str]]:
    root = Path(root_path)
    ds_json = root / "dataset.json"
    if not ds_json.exists():
        # Fallback: match imagesTr and labelsTr by filename
        images_dir = root / "imagesTr"
        labels_dir = root / "labelsTr"
        if not (images_dir.is_dir() and labels_dir.is_dir()):
            raise FileNotFoundError(f"dataset.json not found and imagesTr/labelsTr missing at: {root}")
        cases = []
        for img in sorted(list(images_dir.glob("*.nii.gz")) + list(images_dir.glob("*.nii"))):
            lbl = labels_dir / (img.name[:-7] + '.nii.gz' if img.name.endswith('.nii.gz') else (img.name[:-4] + '.nii.gz'))
            if lbl.exists():
                cases.append({"case_id": _basename_without_nii_gz(img), "image": str(img), "label": str(lbl)})
        if not cases:
            raise RuntimeError("No image/label pairs found under imagesTr/labelsTr")
        return cases

    with ds_json.open("r") as f:
        meta = json.load(f)
    cases: List[Dict[str, str]] = []
    for item in meta.get("training", []):
        image_rel = item["image"]
        label_rel = item["label"]
        img = (root / image_rel).resolve()
        lbl = (root / label_rel).resolve()
        cases.append({"case_id": _basename_without_nii_gz(Path(img)), "image": str(img), "label": str(lbl)})
    if not cases:
        raise RuntimeError("dataset.json has no 'training' entries")
    return cases


_VOLUME_CACHE: Dict[Tuple[str, bool], np.ndarray] = {}

def _load_nifti_array(path: Path, is_segmentation: bool = False) -> np.ndarray:
    key = (str(path), bool(is_segmentation))
    if key in _VOLUME_CACHE:
        return _VOLUME_CACHE[key]
    img = nib.load(str(path))
    img = nib.as_closest_canonical(img)
    arr = img.get_fdata(dtype=np.float32)
    if is_segmentation:
        arr = np.rint(arr).astype(np.int16)
    _VOLUME_CACHE[key] = arr
    return arr


# ---------------------------
# Utilities
# ---------------------------

def _extract_slice(volume: np.ndarray, axis: int, index: int) -> np.ndarray:
    if axis == 0:
        return volume[index, :, :]
    elif axis == 1:
        return volume[:, index, :]
    else:
        return volume[:, :, index]


def _normalize_image(img2d: np.ndarray, do_normalize: bool) -> np.ndarray:
    if not do_normalize:
        return img2d
    valid = np.isfinite(img2d)
    if not np.any(valid):
        return img2d
    lo, hi = np.percentile(img2d[valid], [1, 99])
    if hi <= lo:
        return img2d
    out = np.clip(img2d, lo, hi)
    out = (out - lo) / (hi - lo + 1e-6)
    return out


def _default_label_colors(unique_labels: List[int]) -> Dict[int, str]:
    # Binary default: 1 -> red; other labels get tab20
    colors: Dict[int, str] = {}
    if 1 in unique_labels:
        colors[1] = "#d62728"  # red
    tab20 = plt.get_cmap("tab20")
    extra = [l for l in unique_labels if l not in colors and l != 0]
    for i, l in enumerate(extra):
        colors[l] = tab20(i % tab20.N)
    return colors


# ---------------------------
# Summaries
# ---------------------------

def summarize_shapes_and_counts(root_path: str, head: int = 10) -> None:
    """Print per-case shapes and simple label histograms (first N)."""
    cases = _parse_dataset(root_path)
    print(f"Cases: {len(cases)}")
    for case in cases[:head]:
        img = _load_nifti_array(Path(case["image"]))
        lbl = _load_nifti_array(Path(case["label"]), is_segmentation=True)
        uniq, cnt = np.unique(lbl, return_counts=True)
        hist = {int(u): int(c) for u, c in zip(uniq, cnt)}
        print(f"- {case['case_id']}: image {img.shape}, label {lbl.shape} | labels {hist}")


# ---------------------------
# Non-interactive quick view
# ---------------------------

def quick_view_case(root_path: str, case_id: str, plane: int = 2, index: int | None = None,
                    alpha: float = 0.5, contour_only: bool = False, normalize: bool = True) -> None:
    cases = _parse_dataset(root_path)
    m = next((c for c in cases if c["case_id"] == case_id), None)
    if m is None:
        raise ValueError(f"Case not found: {case_id}")
    img = _load_nifti_array(Path(m["image"]))
    lbl = _load_nifti_array(Path(m["label"]), is_segmentation=True)
    if img.shape != lbl.shape:
        warnings.warn(f"Shape mismatch: image {img.shape} vs label {lbl.shape}")
    max_idx = min(img.shape[plane], lbl.shape[plane]) - 1
    if index is None:
        index = max_idx // 2
    else:
        index = max(0, min(int(index), max_idx))
    img2d = _extract_slice(img, plane, index)
    lbl2d = _extract_slice(lbl, plane, index)
    uniq = sorted(int(x) for x in np.unique(lbl))
    colors = _default_label_colors(uniq)
    plt.figure(figsize=(6, 6))
    plt.imshow(_normalize_image(img2d, normalize), cmap="gray")
    for l in [x for x in uniq if x != 0]:
        mask = (lbl2d == l)
        if not np.any(mask):
            continue
        col = to_rgba(colors.get(l, "#9467bd"), alpha=alpha)
        if contour_only:
            plt.contour(mask.astype(float), levels=[0.5], colors=[col], linewidths=1.5)
        else:
            plt.imshow(mask.astype(int), cmap=ListedColormap([(0, 0, 0, 0), col]), interpolation="none")
    plt.axis("off")
    plt.title(f"{case_id} | plane={['Sag','Cor','Ax'][plane]} | idx={index}")
    plt.show()


# ---------------------------
# Interactive viewer
# ---------------------------

def launch_segmentation_dataset_viewer(root_path: str):
    """Interactive binary segmentation viewer for a built nnU-Net-like dataset."""
    cases = _parse_dataset(root_path)
    case_ids = [c["case_id"] for c in cases]

    case_dd = Dropdown(options=case_ids, description="Case:", layout=Layout(width="320px"))
    plane_rb = RadioButtons(options=[("Axial", 2), ("Coronal", 1), ("Sagittal", 0)], description="Plane:", layout=Layout(width="220px"))
    slice_slider = IntSlider(description="Slice:", min=0, max=1, value=0, continuous_update=False, layout=Layout(width="420px"))
    labels_ms = SelectMultiple(options=[], description="Labels:", layout=Layout(width="200px", height="160px"))
    alpha_slider = FloatSlider(description="Alpha:", min=0.1, max=1.0, step=0.05, value=0.5, readout_format=".2f", layout=Layout(width="220px"))
    contour_cb = Checkbox(value=False, description="Contour only")
    norm_cb = Checkbox(value=True, description="Normalize image")

    summary_html = HTML(layout=Layout(width="100%"))
    out = Output(layout=Layout(border="1px solid #ddd"))

    # State
    current_img: np.ndarray | None = None
    current_lbl: np.ndarray | None = None
    current_labels: List[int] = []

    def _load_case(_=None):
        nonlocal current_img, current_lbl, current_labels
        cid = case_dd.value
        entry = next(c for c in cases if c["case_id"] == cid)
        current_img = _load_nifti_array(Path(entry["image"]))
        current_lbl = _load_nifti_array(Path(entry["label"]), is_segmentation=True)
        if current_img.shape != current_lbl.shape:
            warnings.warn(f"Shape mismatch: image {current_img.shape} vs label {current_lbl.shape}")
        uniq = sorted(int(x) for x in np.unique(current_lbl))
        current_labels = uniq
        nonzero = [l for l in uniq if l != 0]
        labels_ms.options = [(str(l), l) for l in uniq]
        labels_ms.value = tuple(nonzero if nonzero else uniq)
        # Update slider
        axis = plane_rb.value
        max_idx = int(min(current_img.shape[axis], current_lbl.shape[axis])) - 1
        slice_slider.max = max_idx
        slice_slider.value = max(0, min(max_idx, max_idx // 2))
        # Summary
        u, cts = np.unique(current_lbl, return_counts=True)
        items = [f"<b>{int(u)}</b>: {int(c)}" for u, c in sorted(zip(u, cts))]
        summary_html.value = "Label voxel counts: " + " | ".join(items)

    def _render(_=None):
        if current_img is None or current_lbl is None:
            _load_case()
        with out:
            out.clear_output(wait=True)
            axis = plane_rb.value
            idx = min(slice_slider.value, min(current_img.shape[axis], current_lbl.shape[axis]) - 1)
            img2d = _extract_slice(current_img, axis, idx)
            lbl2d = _extract_slice(current_lbl, axis, idx)
            fig, ax = plt.subplots(1, 1, figsize=(6.5, 6.5))
            ax.imshow(_normalize_image(img2d, norm_cb.value), cmap="gray")
            ax.set_title(f"{case_dd.value} | {['Sagittal','Coronal','Axial'][axis]}")
            ax.axis("off")
            colors = _default_label_colors(current_labels)
            selected = list(labels_ms.value) if labels_ms.value else []
            for l in selected:
                mask = (lbl2d == l)
                if not np.any(mask):
                    continue
                col = to_rgba(colors.get(l, "#9467bd"), alpha=alpha_slider.value)
                if contour_cb.value:
                    ax.contour(mask.astype(float), levels=[0.5], colors=[col], linewidths=1.5)
                else:
                    ax.imshow(mask.astype(int), cmap=ListedColormap([(0, 0, 0, 0), col]), interpolation="none")
            plt.show()

    # Wire
    case_dd.observe(_load_case, names="value")
    plane_rb.observe(lambda ch: (_load_case(), _render()), names="value")
    slice_slider.observe(_render, names="value")
    labels_ms.observe(_render, names="value")
    alpha_slider.observe(_render, names="value")
    contour_cb.observe(_render, names="value")
    norm_cb.observe(_render, names="value")

    # Layout
    controls_row1 = HBox([case_dd, plane_rb, alpha_slider, contour_cb, norm_cb])
    controls_row2 = HBox([slice_slider, labels_ms])
    ui = VBox([controls_row1, controls_row2, summary_html, out])

    _load_case()
    _render()
    from IPython.display import display  # lazy import for notebooks
    display(ui)



# ---------------------------
# Classification dataset viewer (labels.csv + imagesTr)
# ---------------------------

def _parse_classification_dataset(root_path: str) -> List[Dict[str, str]]:
    root = Path(root_path)
    labels_csv = root / "labels.csv"
    if not labels_csv.exists():
        raise FileNotFoundError(f"labels.csv not found at: {labels_csv}")
    import csv
    rows: List[Dict[str, str]] = []
    with labels_csv.open("r", newline="") as f:
        reader = csv.DictReader(f)
        required = {"case_id", "label", "image_path"}
        if not required.issubset(set(reader.fieldnames or [])):
            raise ValueError(f"labels.csv must have headers: {required}")
        for r in reader:
            rows.append({
                "case_id": r["case_id"],
                "label": r["label"],
                "image": str(Path(r["image_path"]).resolve()),
            })
    if not rows:
        raise RuntimeError("labels.csv is empty")
    return rows


def _apply_window(img2d: np.ndarray, mode: str, p_low: float, p_high: float, center: float, width: float) -> np.ndarray:
    if mode == "Percentile":
        valid = np.isfinite(img2d)
        if not np.any(valid):
            return img2d
        lo, hi = np.percentile(img2d[valid], [p_low, p_high])
    else:
        lo = center - width / 2.0
        hi = center + width / 2.0
    if hi <= lo:
        return img2d
    out = np.clip(img2d, lo, hi)
    out = (out - lo) / (hi - lo + 1e-6)
    return out


def launch_classification_dataset_viewer(root_path: str):
    """Interactive viewer for classification datasets built with labels.csv.
    Shows per-case label and provides windowing controls (Percentile or Center/Width).
    """
    cases = _parse_classification_dataset(root_path)
    case_ids = [c["case_id"] for c in cases]

    # Widgets
    case_dd = Dropdown(options=case_ids, description="Case:", layout=Layout(width="320px"))
    plane_rb = RadioButtons(options=[("Axial", 2), ("Coronal", 1), ("Sagittal", 0)], description="Plane:", layout=Layout(width="220px"))
    slice_slider = IntSlider(description="Slice:", min=0, max=1, value=0, continuous_update=False, layout=Layout(width="420px"))

    window_mode = Dropdown(options=["Percentile", "Center/Width"], value="Percentile", description="Window:", layout=Layout(width="200px"))
    p_low = IntSlider(description="P_low:", min=0, max=20, value=1, layout=Layout(width="250px"))
    p_high = IntSlider(description="P_high:", min=80, max=100, value=99, layout=Layout(width="250px"))
    center_slider = FloatSlider(description="Center:", min=-500.0, max=500.0, step=1.0, value=0.0, readout_format=".1f", layout=Layout(width="300px"))
    width_slider = FloatSlider(description="Width:", min=1.0, max=1000.0, step=1.0, value=200.0, readout_format=".1f", layout=Layout(width="300px"))

    info_html = HTML(layout=Layout(width="100%"))
    out = Output(layout=Layout(border="1px solid #ddd"))

    # State
    current_img: np.ndarray | None = None
    current_label: str | None = None
    vol_percentiles: Tuple[float, float] = (0.0, 1.0)

    def _load_case(_=None):
        nonlocal current_img, current_label, vol_percentiles
        cid = case_dd.value
        entry = next(c for c in cases if c["case_id"] == cid)
        current_img = _load_nifti_array(Path(entry["image"]))
        current_label = entry["label"]
        # Update slice slider
        axis = plane_rb.value
        slice_slider.max = int(current_img.shape[axis]) - 1
        slice_slider.value = max(0, slice_slider.max // 2)
        # Derive default center/width from volume percentiles
        valid = np.isfinite(current_img)
        if np.any(valid):
            lo, hi = np.percentile(current_img[valid], [1, 99])
            vol_percentiles = (float(lo), float(hi))
            center_slider.min = float(current_img[valid].min())
            center_slider.max = float(current_img[valid].max())
            center_slider.value = (lo + hi) / 2.0
            width_slider.min = 1.0
            width_slider.max = max(10.0, (current_img[valid].max() - current_img[valid].min()))
            width_slider.value = max(10.0, (hi - lo))
        info_html.value = f"Label: <b>{current_label}</b>"

    def _render(_=None):
        if current_img is None:
            _load_case()
        with out:
            out.clear_output(wait=True)
            axis = plane_rb.value
            idx = min(max(0, slice_slider.value), current_img.shape[axis] - 1)
            img2d = _extract_slice(current_img, axis, idx)
            if window_mode.value == "Percentile":
                shown = _apply_window(img2d, "Percentile", float(p_low.value), float(p_high.value), 0.0, 0.0)
            else:
                shown = _apply_window(img2d, "Center/Width", 0.0, 0.0, float(center_slider.value), float(width_slider.value))
            plt.figure(figsize=(6.5, 6.5))
            plt.imshow(shown, cmap="gray")
            plt.title(f"{case_dd.value} | Label={current_label} | {['Sagittal','Coronal','Axial'][axis]} | z={idx}")
            plt.axis("off")
            plt.show()

    # Wire
    case_dd.observe(_load_case, names="value")
    plane_rb.observe(lambda ch: (_load_case(), _render()), names="value")
    slice_slider.observe(_render, names="value")
    window_mode.observe(_render, names="value")
    p_low.observe(_render, names="value")
    p_high.observe(_render, names="value")
    center_slider.observe(_render, names="value")
    width_slider.observe(_render, names="value")

    # Layout
    row1 = HBox([case_dd, plane_rb])
    row2 = HBox([slice_slider])
    row3 = HBox([window_mode, p_low, p_high])
    row4 = HBox([center_slider, width_slider])
    ui = VBox([row1, row2, row3, row4, info_html, out])

    _load_case()
    _render()
    from IPython.display import display
    display(ui)
