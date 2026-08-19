"""
Unsupervised comparison methods.

The marking scheme asks for a comparison against at least one unsupervised
method from the last ten years, so four classical semi-automated methods are
implemented here.  They all receive exactly the same user box as the proposed
method, which keeps the comparison fair.

  * otsu           - Otsu threshold inside the box (Otsu 1979, still the usual
                     baseline in tumour segmentation papers)
  * kmeans         - k-means clustering on intensity, k = 3
  * region_growing - classic seeded region growing from the box centre
  * chan_vese      - morphological Chan-Vese active contour
                     (Marquez-Neila et al., IEEE TPAMI 2014)
"""

import cv2
import numpy as np
from scipy import ndimage as ndi
from skimage.filters import threshold_otsu
from skimage.segmentation import morphological_chan_vese
from sklearn.cluster import KMeans

import config
import features


# --------------------------------------------------------------- helpers
def _prepare(image, box):
    """Same ROI extraction as the proposed method so the inputs match."""
    normalised = features.normalise(image)
    patch, box_in_patch, roi = features.extract_patch(normalised, box)
    return patch, box_in_patch, roi


def _inside_box_mask(box_in_patch):
    top, left, bottom, right = box_in_patch
    mask = np.zeros((config.PATCH_SIZE, config.PATCH_SIZE), bool)
    mask[max(top, 0):bottom + 1, max(left, 0):right + 1] = True
    return mask


def _finish(binary, box_in_patch, roi):
    """Clean up, keep the blob nearest the box centre, paste back full size."""
    binary = np.logical_and(binary, _inside_box_mask(box_in_patch))
    binary = ndi.binary_opening(binary, np.ones((3, 3)))
    binary = ndi.binary_closing(binary, np.ones((5, 5)))
    binary = ndi.binary_fill_holes(binary)

    labels, count = ndi.label(binary)
    if count > 1:
        top, left, bottom, right = box_in_patch
        centre = (int((top + bottom) / 2), int((left + right) / 2))
        chosen = labels[centre]
        if chosen == 0:
            sizes = ndi.sum(binary, labels, range(1, count + 1))
            chosen = int(np.argmax(sizes)) + 1
        binary = labels == chosen

    top, left, bottom, right = roi
    full = np.zeros((config.IMAGE_SIZE, config.IMAGE_SIZE), bool)
    resized = cv2.resize(binary.astype(np.uint8), (right - left + 1, bottom - top + 1),
                         interpolation=cv2.INTER_NEAREST)
    full[top:bottom + 1, left:right + 1] = resized > 0
    return full


def _lesion_side(binary, box_in_patch):
    """
    Threshold methods do not know which side is the lesion.  Pick whichever
    class sits closer to the centre of the user's box, since that is where the
    user said the lesion is.
    """
    inside = _inside_box_mask(box_in_patch)
    top, left, bottom, right = box_in_patch
    rows, cols = np.mgrid[0:config.PATCH_SIZE, 0:config.PATCH_SIZE]
    centre_y, centre_x = (top + bottom) / 2.0, (left + right) / 2.0
    core = (np.hypot(rows - centre_y, cols - centre_x) < 0.25 * max(bottom - top, 1))

    option_a = np.logical_and(binary, inside)
    option_b = np.logical_and(~binary, inside)
    score_a = option_a[core].mean() if core.any() else 0
    score_b = option_b[core].mean() if core.any() else 0
    return option_a if score_a >= score_b else option_b


# --------------------------------------------------------------- methods
def otsu_baseline(image, box):
    patch, box_in_patch, roi = _prepare(image, box)
    inside = _inside_box_mask(box_in_patch)
    values = patch[inside]
    if values.size < 10 or values.std() < 1e-6:
        return np.zeros((config.IMAGE_SIZE, config.IMAGE_SIZE), bool)
    binary = patch > threshold_otsu(values)
    return _finish(_lesion_side(binary, box_in_patch), box_in_patch, roi)


def kmeans_baseline(image, box, k=3):
    patch, box_in_patch, roi = _prepare(image, box)
    inside = _inside_box_mask(box_in_patch)
    values = patch[inside].reshape(-1, 1)
    if values.shape[0] < k:
        return np.zeros((config.IMAGE_SIZE, config.IMAGE_SIZE), bool)

    km = KMeans(n_clusters=k, n_init=5, random_state=0).fit(values)
    labels = np.full(patch.shape, -1)
    labels[inside] = km.labels_

    # the cluster with the highest mean intensity is taken as the lesion
    order = np.argsort(km.cluster_centers_.ravel())
    binary = labels == order[-1]
    return _finish(_lesion_side(binary, box_in_patch), box_in_patch, roi)


def region_growing_baseline(image, box, tolerance=0.6):
    """Grow from the centre of the box while the intensity stays similar."""
    patch, box_in_patch, roi = _prepare(image, box)
    smooth = ndi.gaussian_filter(patch, 2)

    top, left, bottom, right = box_in_patch
    seed = (int((top + bottom) / 2), int((left + right) / 2))
    seed_value = smooth[seed]

    similar = np.abs(smooth - seed_value) < tolerance
    labels, count = ndi.label(similar)
    if count == 0 or labels[seed] == 0:
        return np.zeros((config.IMAGE_SIZE, config.IMAGE_SIZE), bool)

    return _finish(labels == labels[seed], box_in_patch, roi)


def chan_vese_baseline(image, box, iterations=60):
    """Morphological active contour started from an ellipse inside the box."""
    patch, box_in_patch, roi = _prepare(image, box)

    top, left, bottom, right = box_in_patch
    rows, cols = np.mgrid[0:config.PATCH_SIZE, 0:config.PATCH_SIZE]
    centre_y, centre_x = (top + bottom) / 2.0, (left + right) / 2.0
    radius_y = max((bottom - top) / 2.0, 2.0) * 0.6
    radius_x = max((right - left) / 2.0, 2.0) * 0.6
    init = ((rows - centre_y) ** 2 / radius_y ** 2 +
            (cols - centre_x) ** 2 / radius_x ** 2) < 1.0

    contour = morphological_chan_vese(patch, num_iter=iterations,
                                      init_level_set=init.astype(np.uint8),
                                      smoothing=2, lambda1=1, lambda2=1)
    return _finish(contour.astype(bool), box_in_patch, roi)


BASELINES = {
    "otsu": otsu_baseline,
    "kmeans": kmeans_baseline,
    "region_growing": region_growing_baseline,
    "chan_vese": chan_vese_baseline,
}
