"""
Task 2 - settings.

Everything the deep learning experiments need is collected here so the models,
the PSO search and the pruning stage all agree on the same numbers.
"""

import os

# ---------------------------------------------------------------- paths
PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(PROJECT_DIR, "brisc2025", "segmentation_task")

TRAIN_IMAGES = os.path.join(DATA_DIR, "train", "images")
TEST_IMAGES = os.path.join(DATA_DIR, "test", "images")

HERE = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR = os.path.join(HERE, "outputs")
MODEL_DIR = os.path.join(OUTPUT_DIR, "models")

PROPOSED_MODEL = os.path.join(MODEL_DIR, "mha_resunet.h5")
PRUNED_MODEL = os.path.join(MODEL_DIR, "mha_resunet_pruned.h5")
UNET_MODEL = os.path.join(MODEL_DIR, "unet_pso.h5")
PSO_RESULT = os.path.join(OUTPUT_DIR, "pso_best.json")

# ---------------------------------------------------------------- images
# Networks train at this size.  Predictions are pushed back up to 512 before
# scoring, so Task 1 and Task 2 numbers are measured on exactly the same masks.
IMAGE_SIZE = 192
EVAL_SIZE = 512

# ---------------------------------------------------------------- training
VAL_FRACTION = 0.2
BATCH_SIZE = 16
EPOCHS = 40
LEARNING_RATE = 5e-4

# BatchNorm's default momentum of 0.99 updates the running statistics very
# slowly, so at these batch sizes the validation numbers swing about wildly
# while the training numbers look fine.  0.9 settles much faster.
BN_MOMENTUM = 0.9
EARLY_STOP_PATIENCE = 8
SEED = 42

# ---------------------------------------------------------------- proposed model
BASE_FILTERS = 32               # widths are 32, 64, 128, 256 and 512 at the bottleneck
ATTENTION_HEADS = 4             # multi-head self attention at the bottleneck
ATTENTION_KEY_DIM = 64          # fixed, so the pruned model can inherit the MHA weights

# ---------------------------------------------------------------- PSO
PSO_PARTICLES = 6
PSO_ITERATIONS = 4
PSO_EPOCHS = 6                  # short trainings, just to rank the candidates
PSO_SUBSET = 1000               # scans used per fitness evaluation
PSO_SIZE = 128                  # smaller images keep the search affordable

# ---------------------------------------------------------------- pruning
# The module carries a hard limit of 2 million trainable parameters, and the
# unpruned model has 8.84 M, so the pruning stage is what has to get underneath
# it rather than being an optional efficiency exercise.  0.55 over six rounds
# lands the network on widths 8/32/60/116/232 and 1,963,474 parameters, which
# is the closest fit under the limit that the width rounding allows - 0.536
# comes out at 2,018,714 and is over it.
#
# The previous setting was 0.4 over 4 rounds, which gave 3.28 M parameters and
# 0.8699 test Dice; that whole version is kept in task2/backup_ratio040/ so the
# earlier figures can still be reproduced.
PRUNE_RATIO = 0.55              # total fraction of channels to remove

# Pruning all 40% in one go destroys the network - it drops to about 0.01 Dice
# and has to be retrained from scratch, which defeats the point.  Removing the
# channels over several rounds, with a short recovery after each, keeps the
# model working the whole way down.  See prune.py for the measured comparison.
PRUNE_ROUNDS = 6
PRUNE_ROUND_EPOCHS = 3          # short recovery after each round
PRUNE_FINETUNE_EPOCHS = 12      # longer fine-tune once the target width is hit
BN_RECALIBRATION_BATCHES = 100  # forward-only batches used to re-estimate BN stats

# ---------------------------------------------------------------- evaluation
N_TEST_IMAGES = 200             # the brief asks for 200 images for BRISC 2025
EVAL_SEED = 2
