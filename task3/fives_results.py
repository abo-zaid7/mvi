"""
Aman's Task 1 and Task 2 - recorded results for the FIVES retinal scans.

These two are read back rather than run, which is how they were handed over and
the only way that works here: the Task 2 checkpoint is a Keras 3 file and the
GUI's environment is TensorFlow 2.13, which cannot deserialise it.  Along with
the models came every test image, its ground truth and its prediction as PNGs,
plus the per-image metrics, so the interface can show the same four panels and
the same numbers it shows for a model it runs itself.

The one thing this module will not do is pretend.  A recorded entry reports the
Dice that was recorded with it, and `is_recorded` travels with the result so the
front end can label it rather than implying the mask was produced just now.

    listing(entry)                  -> the test images, in order
    scan_path(entry, name)          -> the input image on disk
    truth(entry, name)              -> ground truth mask at 512 x 512
    prediction(entry, name)         -> recorded mask at 512 x 512
    recorded_metrics(entry, name)   -> what was measured for that image
    spec_sheet(entry)               -> the whole-test-set summary
"""

import csv
import glob
import os

import cv2
import numpy as np

EVAL_SIZE = 512

# The report cases each share folder highlights.  They are offered as a filter
# in the scan picker so a demonstration can go straight to them.
CASE_FILTER = "cases"


def _images_dir(entry):
    return os.path.join(entry["results_dir"], "images")


def available(entry):
    return os.path.isdir(_images_dir(entry))


def _suffix(entry):
    """What a prediction file is called for this entry."""
    variant = entry.get("variant")
    return "_prediction_%s.png" % variant if variant else "_prediction.png"


# ------------------------------------------------------------------- listing
def _read_csv(path):
    if not os.path.exists(path):
        return []
    with open(path, newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def case_names(entry):
    """The handful of images the share folder calls its report cases."""
    rows = _read_csv(os.path.join(entry["results_dir"], "metrics_cases.csv"))
    names = set()
    for row in rows:
        name = (row.get("case") or row.get("image") or "").strip()
        if name:
            names.add(name)
    return names


def listing(entry):
    """
    Every recorded test image: its name, its disease category and whether it is
    one of the report cases.

    The names come from the input files rather than from the CSV, so an image
    that was never written out cannot end up in the picker.
    """
    if not available(entry):
        return []

    categories = {}
    for row in _read_csv(os.path.join(entry["results_dir"],
                                      "metrics_per_image.csv")):
        name = (row.get("image") or "").strip()
        if name and row.get("category"):
            categories[name] = row["category"]

    cases = case_names(entry)

    listing_rows = []
    for path in sorted(glob.glob(os.path.join(_images_dir(entry), "*_input.png"))):
        name = os.path.basename(path)[:-len("_input.png")]
        listing_rows.append({
            "name": name,
            "tumour_type": categories.get(name, _category_from_name(name)),
            "in_test_set": True,
            "is_case": name in cases,
        })
    return listing_rows


# FIVES encodes the condition in the file name's second part: 1_A -> A.
FIVES_CATEGORIES = {
    "A": "AMD",
    "D": "Diabetic retinopathy",
    "G": "Glaucoma",
    "N": "Normal",
}


def _category_from_name(name):
    parts = name.split("_")
    return FIVES_CATEGORIES.get(parts[-1], "unknown") if len(parts) > 1 else "unknown"


# ------------------------------------------------------------------- images
def scan_path(entry, name):
    """The input image for one recorded test scan."""
    base = os.path.basename(name)
    if base.endswith("_input.png"):
        base = base[:-len("_input.png")]

    path = os.path.join(_images_dir(entry), base + "_input.png")
    if not os.path.exists(path):
        return None
    return path


def _load_mask(path):
    if not path or not os.path.exists(path):
        return None
    mask = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if mask is None:
        return None
    return cv2.resize(mask, (EVAL_SIZE, EVAL_SIZE),
                      interpolation=cv2.INTER_NEAREST) > 127


def truth(entry, name):
    return _load_mask(os.path.join(_images_dir(entry), name + "_gt.png"))


def prediction(entry, name):
    """The recorded mask for this image, as if the model had just produced it."""
    mask = _load_mask(os.path.join(_images_dir(entry), name + _suffix(entry)))
    if mask is None:
        return None
    return {"mask": mask, "latency_ms": None, "is_recorded": True}


# ------------------------------------------------------------------ metrics
# The two share folders name their columns slightly differently, so both
# spellings are accepted and mapped onto the names the GUI already uses.
METRIC_COLUMNS = {
    "dice": "dice",
    "iou": "iou",
    "accuracy": "accuracy",
    "hd95_px": "hd95",
    "hd95": "hd95",
    "precision": "precision",
    "recall": "sensitivity",
    "cldice": "cldice",
    "specificity": "specificity",
}


def recorded_metrics(entry, name):
    """
    What was measured for this image when the results were produced.

    These are used in place of a recomputed score: the share folder says its
    numbers are the project's recorded evaluation and should not be recomputed
    from the 512 px display copies, which would quietly disagree with the
    author's report.
    """
    rows = _read_csv(os.path.join(entry["results_dir"], "metrics_per_image.csv"))
    variant = entry.get("variant")

    for row in rows:
        if (row.get("image") or "").strip() != name:
            continue
        if variant and (row.get("model") or "").strip() != variant:
            continue

        values = {}
        for column, key in METRIC_COLUMNS.items():
            raw = row.get(column)
            if raw in (None, ""):
                continue
            try:
                values[key] = round(float(raw), 4)
            except ValueError:
                continue
        return values or None
    return None


def spec_sheet(entry):
    """The whole-test-set summary, shown where a model spec sheet would go."""
    rows = _read_csv(os.path.join(entry["results_dir"], "metrics_summary.csv"))
    info = {
        "kind": "recorded results",
        "dataset": "fives",
        "images": len(listing(entry)),
        "recorded": True,
    }

    variant = entry.get("variant")
    if not rows:
        return info

    # Task 1's summary is one row per metric; Task 2's is one row per model and
    # metric, and carries the model's size and speed as extra rows.
    if "model" in rows[0]:
        wanted = {"pruned": "Structured pruned + fine-tune (final)",
                  "proposed": "Attention U-Net + clDice (proposed)",
                  "benchmark": "PSO U-Net (benchmark)"}.get(variant)
        for row in rows:
            if wanted and row.get("model") != wanted:
                continue
            metric, value = row.get("metric", ""), row.get("value", "")
            try:
                number = float(value)
            except (TypeError, ValueError):
                continue
            if metric == "Dice (per image)":
                info["recorded_dice"] = round(number, 4)
            elif metric == "Tunable parameters":
                info["trainable_params"] = int(number)
                info["params_M"] = round(number / 1e6, 2)
            elif metric == "Model size (MB)":
                info["size_mb"] = round(number, 2)
            elif metric.startswith("FLOPs"):
                info["gflops"] = round(number / 1e9, 3)
            elif metric.startswith("Latency CPU"):
                info["measured_latency_ms"] = round(number, 1)
    else:
        for row in rows:
            if row.get("metric") == "Dice":
                try:
                    info["recorded_dice"] = round(float(row["mean"]), 4)
                except (TypeError, ValueError, KeyError):
                    pass
        info["trainable_params"] = 0
        info["size_mb"] = 0.0

    return info
