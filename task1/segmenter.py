"""
The semi-automated segmenter itself.

Pipeline for one image:

    user drags a box
        -> CLAHE + z-score normalisation
        -> crop padded ROI, resample to 128x128
        -> ~41 features per pixel
        -> Random Forest gives a tumour probability map
        -> smooth, threshold, morphological clean-up
        -> (optional) random walker boundary refinement
        -> paste back into the full 512x512 frame

Nothing here is deep learning and no TensorFlow, Haar cascade or template
matching is used, as required by the brief.
"""

import cv2
import numpy as np
from scipy import ndimage as ndi
from skimage.segmentation import random_walker

import config
import features


class SemiAutoSegmenter:
    """Wraps a trained Random Forest and turns a user box into a mask."""

    def __init__(self, model):
        self.model = model

    # ------------------------------------------------------------------ core
    def probability_map(self, image, box):
        """Run the classifier and return the raw probability patch."""
        normalised = features.normalise(image)
        patch, box_in_patch, roi = features.extract_patch(normalised, box)

        x = features.build_features(patch, box_in_patch)
        probs = self.model.predict_proba(x)[:, 1]
        probs = probs.reshape(config.PATCH_SIZE, config.PATCH_SIZE)

        return probs, patch, roi

    def segment(self, image, box, refine=None):
        """Return a full size boolean mask for the lesion inside the box."""
        if refine is None:
            refine = config.USE_RANDOM_WALKER

        probs, patch, roi = self.probability_map(image, box)
        binary = self._threshold(probs, patch, refine)
        binary = self._clean_up(binary)
        return self._paste_back(binary, roi)

    # ------------------------------------------------------------- internals
    def _threshold(self, probs, patch, refine):
        """Smooth the probability map and cut it at PROB_THRESHOLD."""
        smoothed = ndi.gaussian_filter(probs, config.PROB_SMOOTHING)
        binary = smoothed > config.PROB_THRESHOLD

        # If the classifier is very unsure everywhere we would end up with an
        # empty mask, so drop the threshold once before giving up.
        if binary.sum() < 20:
            binary = smoothed > 0.35

        if refine and 20 < binary.sum() < binary.size - 20:
            binary = self._random_walker(smoothed, patch, binary)

        return binary

    @staticmethod
    def _random_walker(smoothed, patch, fallback):
        """
        Optional refinement: confident pixels become seeds and the random walker
        propagates the labels along image edges, which snaps the boundary onto
        the real lesion edge.
        """
        markers = np.zeros(smoothed.shape, np.uint8)
        markers[smoothed > 0.80] = 1
        markers[smoothed < 0.20] = 2
        if not (markers == 1).any() or not (markers == 2).any():
            return fallback
        try:
            return random_walker(patch, markers, beta=200, mode="cg_j") == 1
        except Exception:
            return fallback

    @staticmethod
    def _clean_up(binary):
        """Close small gaps, fill holes and keep only the biggest blob."""
        binary = ndi.binary_closing(binary, np.ones((5, 5)))
        binary = ndi.binary_fill_holes(binary)

        labels, count = ndi.label(binary)
        if count > 1:
            sizes = ndi.sum(binary, labels, range(1, count + 1))
            binary = labels == (int(np.argmax(sizes)) + 1)
        return binary

    @staticmethod
    def _paste_back(binary, roi):
        """Resize the patch result back to its place in the full image."""
        top, left, bottom, right = roi
        full = np.zeros((config.IMAGE_SIZE, config.IMAGE_SIZE), bool)
        resized = cv2.resize(binary.astype(np.uint8),
                             (right - left + 1, bottom - top + 1),
                             interpolation=cv2.INTER_NEAREST)
        full[top:bottom + 1, left:right + 1] = resized > 0
        return full


def load_segmenter(path=None):
    """Load the trained model from disk."""
    import joblib
    return SemiAutoSegmenter(joblib.load(path or config.MODEL_PATH))
