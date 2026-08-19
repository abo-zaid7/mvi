"""
Segmentation metrics: Dice, IoU, pixel accuracy and HD95.

All four are asked for in the assignment brief.  Dice and IoU measure overlap,
accuracy measures how many pixels were labelled correctly, and HD95 measures how
far the predicted boundary is from the true boundary (in pixels), ignoring the
worst 5% of points so a single stray pixel does not ruin the score.
"""

import numpy as np
from scipy import ndimage as ndi


def dice_score(pred, truth):
    total = pred.sum() + truth.sum()
    if total == 0:
        return 1.0
    return 2.0 * np.logical_and(pred, truth).sum() / total


def iou_score(pred, truth):
    union = np.logical_or(pred, truth).sum()
    if union == 0:
        return 1.0
    return np.logical_and(pred, truth).sum() / union


def pixel_accuracy(pred, truth):
    return (pred == truth).mean()


def sensitivity(pred, truth):
    """Also called recall - how much of the tumour was found."""
    if truth.sum() == 0:
        return 1.0
    return np.logical_and(pred, truth).sum() / truth.sum()


def precision(pred, truth):
    if pred.sum() == 0:
        return 1.0 if truth.sum() == 0 else 0.0
    return np.logical_and(pred, truth).sum() / pred.sum()


def _boundary(mask):
    """Pixels on the outline of a binary mask."""
    eroded = ndi.binary_erosion(mask, np.ones((3, 3)))
    return np.logical_and(mask, ~eroded)


def hd95(pred, truth):
    """
    95th percentile Hausdorff distance, in pixels.

    Distance transforms are used instead of comparing every pair of boundary
    points, which would be far too slow on 512x512 images.
    """
    if pred.sum() == 0 or truth.sum() == 0:
        return float(max(pred.shape))       # worst case, used as a penalty

    pred_edge = _boundary(pred)
    truth_edge = _boundary(truth)

    # distance from every pixel to the nearest boundary pixel of the other mask
    dist_to_truth = ndi.distance_transform_edt(~truth_edge)
    dist_to_pred = ndi.distance_transform_edt(~pred_edge)

    forward = dist_to_truth[pred_edge]
    backward = dist_to_pred[truth_edge]

    both = np.concatenate([forward, backward])
    return float(np.percentile(both, 95))


def all_metrics(pred, truth):
    """Every metric for one image, as a dictionary."""
    return {
        "dice": dice_score(pred, truth),
        "iou": iou_score(pred, truth),
        "accuracy": pixel_accuracy(pred, truth),
        "hd95": hd95(pred, truth),
        "sensitivity": sensitivity(pred, truth),
        "precision": precision(pred, truth),
    }
