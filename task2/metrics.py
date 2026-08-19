"""
Segmentation metrics, evaluated at the full 512 x 512 resolution.

These are the same definitions as task1/metrics.py on purpose: Task 1 and Task 2
have to be compared against each other in the report, so both are scored with
identical code on identical masks.  The module is duplicated rather than
imported so that each task folder runs on its own.
"""

import numpy as np
from scipy import ndimage as ndi


def dice_score(pred, truth):
    total = pred.sum() + truth.sum()
    return 1.0 if total == 0 else 2.0 * np.logical_and(pred, truth).sum() / total


def iou_score(pred, truth):
    union = np.logical_or(pred, truth).sum()
    return 1.0 if union == 0 else np.logical_and(pred, truth).sum() / union


def pixel_accuracy(pred, truth):
    return (pred == truth).mean()


def sensitivity(pred, truth):
    return 1.0 if truth.sum() == 0 else np.logical_and(pred, truth).sum() / truth.sum()


def precision(pred, truth):
    if pred.sum() == 0:
        return 1.0 if truth.sum() == 0 else 0.0
    return np.logical_and(pred, truth).sum() / pred.sum()


def _boundary(mask):
    return np.logical_and(mask, ~ndi.binary_erosion(mask, np.ones((3, 3))))


def hd95(pred, truth):
    """95th percentile Hausdorff distance in pixels, via distance transforms."""
    if pred.sum() == 0 or truth.sum() == 0:
        return float(max(pred.shape))

    pred_edge, truth_edge = _boundary(pred), _boundary(truth)
    forward = ndi.distance_transform_edt(~truth_edge)[pred_edge]
    backward = ndi.distance_transform_edt(~pred_edge)[truth_edge]
    return float(np.percentile(np.concatenate([forward, backward]), 95))


def all_metrics(pred, truth):
    return {
        "dice": dice_score(pred, truth),
        "iou": iou_score(pred, truth),
        "accuracy": pixel_accuracy(pred, truth),
        "hd95": hd95(pred, truth),
        "sensitivity": sensitivity(pred, truth),
        "precision": precision(pred, truth),
    }
