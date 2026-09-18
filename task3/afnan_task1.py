"""
Afnan's Task 1 - classical breast ultrasound segmentation.

Ported from afnan/T1.ipynb so the GUI can run it.  The notebook is the
authority: the stages, the constants and the scoring weights below are the ones
it uses, and the only things changed are the ones a notebook can get away with
and a server cannot -

  * the dataset path is no longer hard-coded to the author's home folder,
  * the pipeline takes an image array instead of a class name and an index,
  * the mask comes back at the GUI's 512 x 512 working size.

The method is fully automatic - no box, no seed click.  It proposes regions from
multi-Otsu thresholds and from dark blobs grown with an edge-stopped region
grower, scores every proposal on contrast, solidity, size and centrality, and
refines the winner with a morphological Chan-Vese contour.

    segment(image)  -> boolean mask at 512 x 512
    spec_sheet()    -> what the GUI shows in place of a model spec
"""

import time
from collections import deque

import cv2
import numpy as np
from scipy.ndimage import binary_fill_holes
from skimage.feature import blob_dog
from skimage.filters import threshold_multiotsu, threshold_otsu
from skimage.measure import label, regionprops
from skimage.morphology import (binary_closing, binary_dilation, binary_opening,
                                convex_hull_image, disk, remove_small_holes,
                                remove_small_objects)
from skimage.segmentation import morphological_chan_vese

# The notebook runs the whole pipeline at 256 px and scores there; the GUI scores
# everything at 512, so the mask is pushed back up at the end.
WORK_SIZE = 256
EVAL_SIZE = 512


# --------------------------------------------------------------- preprocessing
def preprocess(image):
    """CLAHE, then two edge-preserving smoothers to knock back the speckle."""
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    image = clahe.apply(image)
    image = cv2.bilateralFilter(image, d=9, sigmaColor=50, sigmaSpace=50)
    return cv2.medianBlur(image, 5)


def compute_adaptive_tolerance(image, seed, window=25):
    """How far the region grower may stray, set by the noise around the seed."""
    x, y = seed
    height, width = image.shape
    x1, x2 = max(0, x - window), min(width, x + window)
    y1, y2 = max(0, y - window), min(height, y + window)
    return int(np.clip(0.8 * np.std(image[y1:y2, x1:x2]) + 15, 15, 30))


def region_growing_with_edges(image, seed, intensity_tolerance, max_region_size,
                              canny_low=30, canny_high=90):
    """
    Grow from the seed, but never across a Canny edge.

    The edge map is what stops a lesion leaking into the acoustic shadow
    underneath it, which is the usual failure of a plain intensity grower on
    ultrasound.
    """
    height, width = image.shape
    start_x, start_y = seed
    visited = np.zeros((height, width), dtype=bool)
    mask = np.zeros((height, width), dtype=np.uint8)
    edges = cv2.Canny(image, canny_low, canny_high)

    queue = deque([(start_x, start_y)])
    visited[start_y, start_x] = True
    region_sum = float(image[start_y, start_x])
    region_pixels = 1

    neighbours = [(-1, -1), (-1, 0), (-1, 1), (0, -1),
                  (0, 1), (1, -1), (1, 0), (1, 1)]

    while queue:
        if region_pixels >= max_region_size:
            break
        x, y = queue.popleft()
        mask[y, x] = 1
        mean = region_sum / region_pixels

        for dx, dy in neighbours:
            nx, ny = x + dx, y + dy
            if nx < 0 or ny < 0 or nx >= width or ny >= height or visited[ny, nx]:
                continue
            visited[ny, nx] = True
            if edges[ny, nx] != 0:
                continue
            if abs(float(image[ny, nx]) - mean) <= intensity_tolerance:
                mask[ny, nx] = 1
                queue.append((nx, ny))
                region_sum += image[ny, nx]
                region_pixels += 1

    return mask


# ------------------------------------------------------------------ proposals
def _clean(mask):
    mask = binary_opening(mask, disk(2))
    mask = binary_closing(mask, disk(3))
    return binary_fill_holes(mask)


def pick_largest_region(mask, min_size=100):
    labelled = label(mask)
    regions = [r for r in regionprops(labelled) if r.area >= min_size]
    if not regions:
        return np.zeros_like(mask, dtype=np.uint8)
    best = max(regions, key=lambda r: r.area)
    return (labelled == best.label).astype(np.uint8)


def generate_candidates(denoised):
    """Every region worth scoring: multi-Otsu components, then grown dark blobs."""
    height, width = denoised.shape
    candidates = []

    try:
        thresholds = threshold_multiotsu(denoised, classes=3)
        for mask in [denoised < thresholds[0], denoised < thresholds[1]]:
            mask = _clean(mask)
            mask = remove_small_objects(mask, min_size=int(0.002 * height * width))
            labelled = label(mask)
            for region in regionprops(labelled):
                candidates.append(labelled == region.label)
    except Exception:
        # A flat histogram can leave multi-Otsu with nothing to separate.
        mask = _clean(denoised < threshold_otsu(denoised))
        labelled = label(remove_small_objects(mask,
                                              min_size=int(0.002 * height * width)))
        for region in regionprops(labelled):
            candidates.append(labelled == region.label)

    inverted = cv2.GaussianBlur((255 - denoised).astype(float) / 255.0, (0, 0), 2)
    shortest = min(height, width)
    blobs = blob_dog(inverted, min_sigma=shortest * 0.03,
                     max_sigma=shortest * 0.18, threshold=0.05)

    for blob in sorted(blobs, key=lambda b: -b[2])[:4]:
        y, x = int(blob[0]), int(blob[1])
        if not (0 <= y < height and 0 <= x < width):
            continue
        tolerance = compute_adaptive_tolerance(denoised, (x, y))
        grown = region_growing_with_edges(denoised, (x, y), tolerance,
                                          int(0.15 * height * width), 30, 90)
        mask = _clean(grown.astype(bool))
        if mask.sum() > 0:
            candidates.append(mask)

    return candidates


def _region_score(denoised, candidate):
    """
    How much this region looks like a lesion.

    Contrast against the ring around it carries the most weight, because that is
    what a lesion is on B-mode: darker than the tissue at the same depth.
    """
    height, width = denoised.shape
    candidate = candidate.astype(bool)
    area = candidate.sum()
    if area < int(0.002 * height * width) or area > 0.30 * height * width:
        return -1e9

    properties = regionprops(candidate.astype(int))[0]
    ring = binary_dilation(candidate, disk(12)) & ~candidate
    if ring.sum() == 0:
        return -1e9

    interior = denoised[candidate].mean()
    rim = denoised[ring].mean()
    contrast = (rim - interior) / 255.0

    centre_y, centre_x = height / 2.0, width / 2.0
    diagonal = (height * height + width * width) ** 0.5
    y, x = properties.centroid
    centrality = 1.0 - (((x - centre_x) ** 2 + (y - centre_y) ** 2) ** 0.5 / diagonal)
    area_norm = min(area / (0.06 * height * width), 1.0)

    return (1.8 * contrast + 0.7 * properties.solidity
            + 0.4 * area_norm + 0.3 * centrality)


def select_best_region(denoised):
    candidates = generate_candidates(denoised)
    if not candidates:
        return None, None

    scored = [(_region_score(denoised, c), c) for c in candidates]
    best_score, best = max(scored, key=lambda pair: pair[0])
    if best_score <= -1e8:
        return None, None

    # The seed is the deepest point inside the region, which is where a user
    # would have clicked if the method had asked them to.
    distance = cv2.distanceTransform(best.astype(np.uint8), cv2.DIST_L2, 5)
    y, x = np.unravel_index(np.argmax(distance), distance.shape)
    return best, (int(x), int(y))


# --------------------------------------------------------------- refinement
def postprocess(binary_mask, denoised, min_size=200, snake_iterations=40,
                margin=30, use_convex_hull=True):
    """Chan-Vese around the winning region, then tidy the result."""
    mask = _clean(binary_mask.astype(bool))
    ys, xs = np.where(mask)
    if len(xs) == 0:
        return np.zeros_like(mask, dtype=np.uint8)

    rough = mask.sum()
    height, width = denoised.shape
    x1, x2 = max(0, xs.min() - margin), min(width, xs.max() + margin)
    y1, y2 = max(0, ys.min() - margin), min(height, ys.max() + margin)

    refined = morphological_chan_vese(denoised[y1:y2, x1:x2],
                                      num_iter=snake_iterations,
                                      init_level_set=mask[y1:y2, x1:x2],
                                      smoothing=3)

    full = np.zeros_like(mask, dtype=bool)
    full[y1:y2, x1:x2] = refined

    # A contour that has run away is worse than the region it started from.
    if full.sum() > rough * 2:
        full = mask

    full = binary_fill_holes(full)
    full = remove_small_objects(full, min_size=min_size)
    full = remove_small_holes(full, area_threshold=min_size)

    if use_convex_hull and full.sum() > 0:
        before = full.sum()
        hulled = convex_hull_image(full)
        if hulled.sum() <= before * 1.5:
            full = hulled

    return pick_largest_region(full.astype(np.uint8), min_size=min_size)


# ------------------------------------------------------------------- GUI entry
def segment(image):
    """
    One scan in, a 512 x 512 boolean mask and a latency out.

    `image` is the greyscale scan the GUI already holds at 512 x 512; it is
    taken down to the notebook's 256 px working size, segmented there, and the
    mask is pushed back up with nearest neighbour so no new boundary pixels are
    invented on the way.
    """
    if image.ndim == 3:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    working = cv2.resize(image, (WORK_SIZE, WORK_SIZE), interpolation=cv2.INTER_AREA)

    start = time.time()
    denoised = preprocess(working)
    best, _seed = select_best_region(denoised)
    if best is None:
        mask = np.zeros((WORK_SIZE, WORK_SIZE), dtype=np.uint8)
    else:
        mask = postprocess(best.astype(np.uint8), denoised)
    elapsed = time.time() - start

    full = cv2.resize(mask.astype(np.uint8), (EVAL_SIZE, EVAL_SIZE),
                      interpolation=cv2.INTER_NEAREST)
    return {"mask": full > 0, "latency_ms": round(elapsed * 1000, 1)}


def spec_sheet():
    """
    What the GUI shows where a loaded model's spec sheet would go.

    There is no trained file behind this one, so the sheet describes the
    pipeline instead of weights - saying "0 parameters" for a classical method
    is the honest answer, not a missing one.
    """
    return {
        "kind": "classical pipeline",
        "dataset": "busi",
        "input_size": WORK_SIZE,
        "stages": "CLAHE + bilateral + median -> multi-Otsu and blob proposals "
                  "-> region scoring -> Chan-Vese refinement",
        "proposals": "multi-Otsu components plus up to 4 edge-stopped grown blobs",
        "trainable_params": 0,
        "size_mb": 0.0,
        "needs_gpu": False,
    }
