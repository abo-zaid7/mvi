"""
Adit's Task 1 - semi-automated brain tumour segmentation from a single point.

Ported from adit/BrainTumorDetection.ipynb.  Her notebook anticipated this: the
seed is taken from the ground-truth centroid for batch evaluation, with a note
saying that in a live GUI it would come from a click instead and that swapping
the seed source is the only change needed.  That is exactly what happens here -
the seed is the centre of the box the user drags on the scan, so her method is
driven by a real interaction rather than by the answer.

The method itself is unchanged: denoise, CLAHE and an Otsu skull-strip, then
K-Means on the intensities inside a 128 x 128 window around the seed, keep the
cluster the seed fell in, clean it up, keep the component containing the seed,
and refine the boundary with a morphological Chan-Vese contour.

    segment(image, box)  -> boolean mask at 512 x 512
    spec_sheet()         -> what the GUI shows in place of a model spec
"""

import time

import cv2
import numpy as np
from scipy.ndimage import binary_fill_holes
from skimage.measure import label, regionprops
from skimage.segmentation import checkerboard_level_set, morphological_chan_vese
from sklearn.cluster import KMeans

ROI_HALF = 64        # half-width in pixels of the window around the seed
K_CLUSTERS = 3       # clusters for the local K-Means step
SEED = 42            # her notebook's random_state, kept so results match
EVAL_SIZE = 512


def preprocess(image):
    """Denoise, then CLAHE, then strip everything outside the skull.

    The skull strip matters more than it looks: without it the K-Means step
    spends one of its three clusters describing the black background instead of
    the tissue the tumour has to be separated from.
    """
    denoised = cv2.fastNlMeansDenoising(image, h=10)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(denoised)

    _, brain = cv2.threshold(enhanced, 0, 255,
                             cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    brain = cv2.morphologyEx(brain, cv2.MORPH_CLOSE, np.ones((7, 7), np.uint8))

    labelled = label(brain)
    if labelled.max() > 0:
        biggest = max(regionprops(labelled), key=lambda r: r.area).label
        brain = (labelled == biggest).astype(np.uint8) * 255

    return enhanced, brain


def largest_cc_containing(mask, seed_rc):
    """The connected component the seed sits in, or the nearest one to it."""
    labelled = label(mask)
    row, column = seed_rc
    seed_label = labelled[row, column]

    if seed_label == 0:
        if labelled.max() == 0:
            return np.zeros_like(mask)
        ys, xs = np.nonzero(labelled)
        nearest = np.argmin((ys - row) ** 2 + (xs - column) ** 2)
        seed_label = labelled[ys[nearest], xs[nearest]]

    return (labelled == seed_label).astype(np.uint8)


def semi_auto_segment(image, brain, seed_rc, roi_half=ROI_HALF, k=K_CLUSTERS):
    """Her proposed method, working on the pixels around one seed point."""
    height, width = image.shape
    row, column = seed_rc
    r0, r1 = max(0, row - roi_half), min(height, row + roi_half)
    c0, c1 = max(0, column - roi_half), min(width, column + roi_half)

    roi = image[r0:r1, c0:c1].astype(np.float32)
    roi_brain = brain[r0:r1, c0:c1] > 0

    kmeans = KMeans(n_clusters=k, n_init=4, random_state=SEED)
    labels = kmeans.fit(roi.reshape(-1, 1)).labels_.reshape(roi.shape)

    seed_r = int(np.clip(row - r0, 0, roi.shape[0] - 1))
    seed_c = int(np.clip(column - c0, 0, roi.shape[1] - 1))
    candidate = ((labels == labels[seed_r, seed_c]).astype(np.uint8)
                 & roi_brain.astype(np.uint8))

    candidate = cv2.morphologyEx(candidate * 255, cv2.MORPH_OPEN,
                                 np.ones((3, 3), np.uint8))
    candidate = cv2.morphologyEx(candidate, cv2.MORPH_CLOSE,
                                 np.ones((5, 5), np.uint8))
    candidate = (candidate > 0).astype(np.uint8)
    candidate = largest_cc_containing(candidate, (seed_r, seed_c))
    candidate = binary_fill_holes(candidate).astype(np.uint8)

    initial = candidate if candidate.sum() >= 4 else checkerboard_level_set(roi.shape, 6)
    normalised = (roi - roi.min()) / (np.ptp(roi) + 1e-6)
    refined = morphological_chan_vese(normalised, num_iter=15,
                                      init_level_set=initial, smoothing=1)
    refined = largest_cc_containing(refined.astype(np.uint8), (seed_r, seed_c))
    refined = binary_fill_holes(refined).astype(np.uint8)

    full = np.zeros((height, width), dtype=np.uint8)
    full[r0:r1, c0:c1] = refined
    return full


def seed_from_box(box, shape):
    """
    The one interaction: the centre of the box the user dragged.

    Her method wants a single point inside the tumour, and the GUI's existing
    interaction is a box, so the centre of that box is the point.  Asking for a
    click as well would be a second interaction for no extra information.
    """
    height, width = shape
    top, left, bottom, right = box
    row = int(np.clip(round((top + bottom) / 2.0), 0, height - 1))
    column = int(np.clip(round((left + right) / 2.0), 0, width - 1))
    return row, column


def segment(image, box):
    """One scan and one box in, a 512 x 512 boolean mask out."""
    if image.ndim == 3:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    if box is None:
        raise ValueError("this method needs one point inside the lesion")

    start = time.time()
    enhanced, brain = preprocess(image)
    seed = seed_from_box(box, image.shape[:2])
    mask = semi_auto_segment(enhanced, brain, seed)
    elapsed = time.time() - start

    if mask.shape != (EVAL_SIZE, EVAL_SIZE):
        mask = cv2.resize(mask, (EVAL_SIZE, EVAL_SIZE),
                          interpolation=cv2.INTER_NEAREST)

    return {"mask": mask > 0, "latency_ms": round(elapsed * 1000, 1),
            "seed": [int(seed[0]), int(seed[1])]}


def spec_sheet():
    """There is no trained file behind this one, so the sheet is the pipeline."""
    return {
        "kind": "classical pipeline",
        "dataset": "figshare",
        "interaction": "one point, taken as the centre of the box",
        "stages": "NLM denoise + CLAHE + Otsu skull strip -> local K-Means "
                  "(k=3, 128 x 128 window) -> Chan-Vese refinement",
        "roi_window": 2 * ROI_HALF,
        "clusters": K_CLUSTERS,
        "trainable_params": 0,
        "size_mb": 0.0,
        "needs_gpu": False,
    }
