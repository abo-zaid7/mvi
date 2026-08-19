"""
Loading the BRISC 2025 segmentation data and simulating the user interaction.

The method in Task 1 is *semi-automated*: a clinician drags a rough box around
the lesion and the algorithm does the rest.  Obviously we cannot ask a
radiologist to click 50 test images for us, so - exactly like the interactive
segmentation literature does (GrabCut, DeepIGeoS, ITK-SNAP studies) - the box is
simulated from the ground truth bounding box and then deliberately made sloppy
by a random amount.  Only the four numbers of the box are passed to the
segmenter, never the mask itself.
"""

import glob
import os

import cv2
import numpy as np

import config


def list_images(split="train"):
    """Return a sorted list of image paths for 'train' or 'test'."""
    folder = config.TRAIN_IMAGES if split == "train" else config.TEST_IMAGES
    return sorted(glob.glob(os.path.join(folder, "*.jpg")))


def mask_path_for(image_path):
    """Masks have the same name as the image but a .png extension."""
    return image_path.replace(os.sep + "images" + os.sep, os.sep + "masks" + os.sep).replace(".jpg", ".png")


def load_pair(image_path):
    """Load one MRI slice and its mask, both resampled to IMAGE_SIZE."""
    size = (config.IMAGE_SIZE, config.IMAGE_SIZE)

    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise IOError("could not read image: " + image_path)
    image = cv2.resize(image, size, interpolation=cv2.INTER_LINEAR)

    mask = cv2.imread(mask_path_for(image_path), cv2.IMREAD_GRAYSCALE)
    if mask is None:
        raise IOError("could not read mask for: " + image_path)
    mask = cv2.resize(mask, size, interpolation=cv2.INTER_NEAREST) > 127

    return image, mask


def tumour_class(image_path):
    """Pull the tumour type out of the filename (gl / me / pi)."""
    parts = os.path.basename(image_path).split("_")
    codes = {"gl": "glioma", "me": "meningioma", "pi": "pituitary", "nt": "no_tumor"}
    return codes.get(parts[3], "unknown")


def simulate_user_box(mask, rng):
    """
    Build a loose box around the lesion, the way a user would actually drag one.

    Each side is pushed outwards by a random 2-20% of the lesion size, so the box
    is never tight and is never the same twice.  This stops the model from
    learning "the tumour fills exactly the box I was given".
    """
    rows, cols = np.where(mask)
    if len(rows) == 0:
        raise ValueError("mask is empty, cannot simulate a box")

    top, bottom = rows.min(), rows.max()
    left, right = cols.min(), cols.max()
    height = bottom - top + 1
    width = right - left + 1

    top -= rng.uniform(0.02, 0.20) * height
    bottom += rng.uniform(0.02, 0.20) * height
    left -= rng.uniform(0.02, 0.20) * width
    right += rng.uniform(0.02, 0.20) * width

    limit = config.IMAGE_SIZE - 1
    return (int(max(top, 0)), int(max(left, 0)),
            int(min(bottom, limit)), int(min(right, limit)))


def box_rng(image_path):
    """A per-image random generator so the simulated box is always reproducible."""
    key = sum(ord(c) * (i + 1) for i, c in enumerate(os.path.basename(image_path)))
    return np.random.default_rng(key)
