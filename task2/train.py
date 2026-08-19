"""
Training for both networks.

Run with:
    python train.py proposed     # the MHA-ResUNet
    python train.py unet         # the baseline, using whatever PSO found
    python train.py both

Models are stored as weights plus a small JSON describing the architecture.
Saving it this way (rather than a single .h5) means prune.py can rebuild the
same graph at a different width and still know exactly what it is loading.
"""

import json
import os
import sys
import time

import numpy as np
from tensorflow import keras

import config
import data
import losses
import models


# --------------------------------------------------------------- save / load
def save_model(model, path, architecture):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    model.save_weights(path)
    with open(path + ".json", "w") as handle:
        json.dump(architecture, handle, indent=2)


def load_model(path):
    """Rebuild a network from its JSON description and load the weights back."""
    with open(path + ".json") as handle:
        architecture = json.load(handle)

    kind = architecture.pop("kind")
    if kind == "unet":
        model = models.build_unet(**architecture)
    else:
        model = models.build_mha_resunet(**architecture)

    model.load_weights(path)
    architecture["kind"] = kind
    return model, architecture


# --------------------------------------------------------------- training
def compile_model(model, learning_rate=None):
    # The legacy Adam is noticeably faster than the v2.11+ one on Apple silicon,
    # which is what this project trains on.
    model.compile(optimizer=keras.optimizers.legacy.Adam(learning_rate or config.LEARNING_RATE),
                  loss=losses.bce_dice_loss,
                  metrics=[losses.dice_coefficient, losses.iou_metric])
    return model


def fit(model, train_batches, val_batches, epochs=None, patience=None, verbose=1):
    """Standard loop: early stopping on validation Dice, LR dropped on plateau."""
    epochs = epochs or config.EPOCHS
    patience = patience or config.EARLY_STOP_PATIENCE

    callbacks = [
        keras.callbacks.EarlyStopping(monitor="val_dice_coefficient", mode="max",
                                      patience=patience, restore_best_weights=True,
                                      verbose=verbose),
        keras.callbacks.ReduceLROnPlateau(monitor="val_dice_coefficient", mode="max",
                                          factor=0.5, patience=4, min_lr=1e-6,
                                          verbose=verbose),
    ]
    return model.fit(train_batches, validation_data=val_batches, epochs=epochs,
                     callbacks=callbacks, verbose=verbose)


def prepare_data(size=None, subset=None):
    """Load the training folder and split it into train / validation batches."""
    size = size or config.IMAGE_SIZE
    paths = data.list_images("train")
    if subset:
        rng = np.random.default_rng(config.SEED)
        paths = [paths[i] for i in rng.choice(len(paths), subset, replace=False)]

    train_paths, val_paths = data.train_val_split(paths)
    train_x, train_y = data.load_arrays(train_paths, size)
    val_x, val_y = data.load_arrays(val_paths, size)

    return (data.Batches(train_x, train_y, augment=True),
            data.Batches(val_x, val_y, augment=False))


# --------------------------------------------------------------- entry points
def train_proposed():
    print("\n=== training the proposed MHA-ResUNet ===")
    train_batches, val_batches = prepare_data()

    base = config.BASE_FILTERS
    architecture = {"kind": "mha_resunet", "input_size": config.IMAGE_SIZE,
                    "filters": [base, base * 2, base * 4, base * 8, base * 16],
                    "heads": config.ATTENTION_HEADS, "dropout": 0.1}

    model = models.build_mha_resunet(input_size=architecture["input_size"],
                                     filters=architecture["filters"],
                                     heads=architecture["heads"],
                                     dropout=architecture["dropout"])
    compile_model(model)
    print("trainable parameters: {:,}".format(model.count_params()))

    start = time.time()
    history = fit(model, train_batches, val_batches)
    minutes = (time.time() - start) / 60

    save_model(model, config.PROPOSED_MODEL, architecture)
    best = max(history.history["val_dice_coefficient"])
    print("best validation Dice %.4f, trained in %.1f min" % (best, minutes))
    return model


def train_unet():
    print("\n=== training the conventional U-Net (PSO hyperparameters) ===")

    if os.path.exists(config.PSO_RESULT):
        with open(config.PSO_RESULT) as handle:
            best = json.load(handle)["best_params"]
        print("using PSO result:", best)
    else:
        best = {"base_filters": 32, "dropout": 0.1, "activation": "relu",
                "learning_rate": 1e-3}
        print("no PSO result found, falling back to defaults:", best)

    train_batches, val_batches = prepare_data()
    architecture = {"kind": "unet", "input_size": config.IMAGE_SIZE,
                    "base_filters": int(best["base_filters"]), "depth": 4,
                    "dropout": float(best["dropout"]),
                    "activation": best["activation"]}

    model = models.build_unet(input_size=architecture["input_size"],
                              base_filters=architecture["base_filters"],
                              depth=architecture["depth"],
                              dropout=architecture["dropout"],
                              activation=architecture["activation"])
    compile_model(model, best["learning_rate"])
    print("trainable parameters: {:,}".format(model.count_params()))

    start = time.time()
    history = fit(model, train_batches, val_batches)
    minutes = (time.time() - start) / 60

    save_model(model, config.UNET_MODEL, architecture)
    print("best validation Dice %.4f, trained in %.1f min"
          % (max(history.history["val_dice_coefficient"]), minutes))
    return model


def main():
    what = sys.argv[1] if len(sys.argv) > 1 else "both"
    os.makedirs(config.MODEL_DIR, exist_ok=True)

    if what in ("proposed", "both"):
        train_proposed()
    if what in ("unet", "both"):
        train_unet()


if __name__ == "__main__":
    main()
