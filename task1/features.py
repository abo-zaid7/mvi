"""
Pre-processing and the hand-crafted feature stack.

Every pixel inside the region of interest is turned into a feature vector, and a
Random Forest then decides "tumour" or "not tumour" for it.  The features fall
into three groups:

  1. appearance   - multi-scale intensity, edges, blobs, local texture
  2. geometry     - where the pixel sits relative to the box the user drew
  3. context      - how the pixel compares to the centre and to the outside
                    of the user's box

Group 2 and 3 are what make the method semi-automated: they are the only place
the user's input enters the model.
"""

import cv2
import numpy as np
from scipy import ndimage as ndi
from skimage.feature import hessian_matrix, hessian_matrix_eigvals

import config


def normalise(image):
    """
    CLAHE for local contrast, then z-score the head region only.

    MRI intensities are not calibrated between scanners, so the same tissue can
    be bright in one scan and mid-grey in another.  Normalising against the head
    (rather than the whole frame, which is mostly black background) puts every
    scan on a comparable scale.
    """
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(image).astype(np.float32)

    head = enhanced > max(10.0, np.percentile(enhanced, 40))
    if head.sum() < 100:            # safety net for an almost empty slice
        head = enhanced > 0

    return (enhanced - enhanced[head].mean()) / (enhanced[head].std() + 1e-6)


def padded_roi(box):
    """Grow the user's box by ROI_PADDING so some healthy tissue is included."""
    top, left, bottom, right = box
    pad_y = int(config.ROI_PADDING * (bottom - top + 1))
    pad_x = int(config.ROI_PADDING * (right - left + 1))
    limit = config.IMAGE_SIZE - 1
    return (max(top - pad_y, 0), max(left - pad_x, 0),
            min(bottom + pad_y, limit), min(right + pad_x, limit))


def extract_patch(image, box):
    """
    Cut out the padded ROI and resample it to PATCH_SIZE x PATCH_SIZE.

    Returns the patch, the user's box in patch coordinates, and the ROI
    rectangle so the result can be pasted back into the full image later.
    """
    top, left, bottom, right = padded_roi(box)
    sub = image[top:bottom + 1, left:right + 1]
    patch = cv2.resize(sub, (config.PATCH_SIZE, config.PATCH_SIZE),
                       interpolation=cv2.INTER_LINEAR)

    scale_y = config.PATCH_SIZE / sub.shape[0]
    scale_x = config.PATCH_SIZE / sub.shape[1]
    b_top, b_left, b_bottom, b_right = box
    box_in_patch = (int((b_top - top) * scale_y), int((b_left - left) * scale_x),
                    int((b_bottom - top) * scale_y), int((b_right - left) * scale_x))

    return patch, box_in_patch, (top, left, bottom, right)


def _appearance_features(patch):
    """Multi-scale intensity, gradient, blob and texture responses."""
    maps = [patch, cv2.bilateralFilter(patch, 7, 1.0, 7)]

    for sigma in (1, 2, 4, 8, 16):
        blurred = ndi.gaussian_filter(patch, sigma)
        gradient = np.hypot(ndi.sobel(blurred, 0), ndi.sobel(blurred, 1))
        maps += [blurred, gradient, ndi.gaussian_laplace(patch, sigma)]

        if sigma <= 8:      # Hessian eigenvalues describe blob / ridge structure
            h = hessian_matrix(patch, sigma=sigma, use_gaussian_derivatives=False)
            eig = hessian_matrix_eigvals(h)
            maps += [eig[0], eig[1]]

    for window in (5, 11, 21):
        mean = ndi.uniform_filter(patch, window)
        variance = ndi.uniform_filter(patch * patch, window) - mean * mean
        maps += [mean, np.sqrt(np.maximum(variance, 0))]

    maps.append(ndi.median_filter(patch, 9))
    return maps


def _geometry_features(box_in_patch):
    """Where the pixel sits inside the box the user dragged."""
    top, left, bottom, right = box_in_patch
    rows, cols = np.mgrid[0:config.PATCH_SIZE, 0:config.PATCH_SIZE].astype(np.float32)

    centre_y, centre_x = (top + bottom) / 2.0, (left + right) / 2.0
    radius_y = max((bottom - top) / 2.0, 1.0)
    radius_x = max((right - left) / 2.0, 1.0)

    # An elliptical distance where 1.0 lands on the edge of the user's box.
    elliptical = np.hypot((rows - centre_y) / radius_y, (cols - centre_x) / radius_x)
    inside = ((rows >= top) & (rows <= bottom) &
              (cols >= left) & (cols <= right)).astype(np.float32)

    maps = [elliptical,
            np.abs(rows - centre_y) / radius_y,
            np.abs(cols - centre_x) / radius_x,
            inside]
    return maps, elliptical, inside


def _context_features(patch, elliptical, inside):
    """How the pixel compares with the middle of the box and with the outside."""
    core = elliptical < 0.5                 # very likely lesion
    outside = inside < 0.5                  # very likely healthy tissue

    mean_core = patch[core].mean() if core.any() else patch.mean()
    mean_outside = patch[outside].mean() if outside.any() else patch.mean()
    std_core = patch[core].std() + 1e-6

    smooth = ndi.gaussian_filter(patch, 2)
    return [np.abs(smooth - mean_core),
            np.abs(smooth - mean_outside),
            (smooth - mean_core) / std_core,
            np.full_like(patch, mean_core),
            np.full_like(patch, mean_outside)]


def build_features(patch, box_in_patch):
    """Stack every feature map and flatten it to (n_pixels, n_features)."""
    geometry, elliptical, inside = _geometry_features(box_in_patch)
    maps = (_appearance_features(patch) + geometry +
            _context_features(patch, elliptical, inside))
    return np.stack(maps, axis=-1).reshape(-1, len(maps))


def feature_names():
    """Readable names, used when reporting Random Forest feature importances."""
    names = ["intensity", "bilateral"]
    for sigma in (1, 2, 4, 8, 16):
        names += ["gauss_s%d" % sigma, "gradient_s%d" % sigma, "log_s%d" % sigma]
        if sigma <= 8:
            names += ["hessian1_s%d" % sigma, "hessian2_s%d" % sigma]
    for window in (5, 11, 21):
        names += ["local_mean_w%d" % window, "local_std_w%d" % window]
    names += ["median_w9",
              "box_elliptical_dist", "box_dist_y", "box_dist_x", "inside_box",
              "diff_to_core", "diff_to_outside", "z_vs_core",
              "core_mean", "outside_mean"]
    return names
