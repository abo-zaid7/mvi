"""
Task 1 - settings used by every other module.

Keeping all the paths and magic numbers in one file makes it much easier to
re-run the experiments with different values later on.
"""

import os

# ---------------------------------------------------------------- paths
PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(PROJECT_DIR, "brisc2025", "segmentation_task")

TRAIN_IMAGES = os.path.join(DATA_DIR, "train", "images")
TRAIN_MASKS = os.path.join(DATA_DIR, "train", "masks")
TEST_IMAGES = os.path.join(DATA_DIR, "test", "images")
TEST_MASKS = os.path.join(DATA_DIR, "test", "masks")

OUTPUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "outputs")
MODEL_PATH = os.path.join(OUTPUT_DIR, "rf_segmenter.joblib")

# ---------------------------------------------------------------- image settings
# The BRISC images come in a few different resolutions, so everything is
# resampled to one common size before processing.
IMAGE_SIZE = 512

# The region of interest that the user marks is resampled to this size.  Doing
# this makes the shape features scale invariant - a small pituitary tumour and a
# large glioma end up looking the same size to the classifier.
PATCH_SIZE = 128

# How much padding is added around the user's box, as a fraction of the box.
# Some background has to be included, otherwise the classifier never sees any
# healthy tissue to compare the lesion against.
ROI_PADDING = 0.35

# ---------------------------------------------------------------- training
N_TRAIN_IMAGES = 400        # number of training scans used to build the model
PIXELS_PER_IMAGE = 700      # pixels sampled from each scan

# These two control the size of the forest.  A big forest (300 trees, leaf 4)
# scores the same Dice as this one but takes 92 MB and is 30% slower, so the
# smaller settings are kept - see model_compression.py for the measurements.
N_TREES = 120
MIN_SAMPLES_LEAF = 20
RANDOM_SEED = 1

# ---------------------------------------------------------------- segmentation
PROB_THRESHOLD = 0.5        # probability above which a pixel is called "tumour"
PROB_SMOOTHING = 1.5        # Gaussian sigma applied to the probability map
USE_RANDOM_WALKER = False   # optional graph based boundary refinement

# ---------------------------------------------------------------- evaluation
N_TEST_IMAGES = 50          # the assignment asks for 50 images in Task 1
EVAL_SEED = 2
