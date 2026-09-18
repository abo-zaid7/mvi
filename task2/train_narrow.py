"""
Train the pruned architecture from scratch, instead of inheriting its weights.

The module's 2 million parameter limit needs 55% of the channels removed, and
at that depth the iterative prune in prune.py stops working: rounds 1-4 recover
to 0.865 every time, round 5 comes back to 0.85, and round 6 only reaches 0.53,
with a 12 epoch fine-tune lifting it to 0.5916 against 0.8712 at the old 40%
setting.  The weights are damaged past the point that fine-tuning repairs.

Liu et al. (ICLR 2019) is the relevant result: what pruning actually produces
is a good narrow *architecture*, and inheriting its weights is often worse than
training that architecture from a fresh initialisation.  That is exactly the
situation here, so this script keeps the widths the L1 ranking chose - the
pruning stage is still what found them - and trains them from scratch under the
same settings the full model used.

    ../tf-env/bin/python train_narrow.py      # about 50 minutes

Writes outputs/models/mha_resunet_narrow.h5, leaving every other model alone.
"""

import json
import os
import time

import numpy as np

import complexity
import config
import models
import train

# The widths round 6 of the prune arrived at: 1,963,474 trainable parameters,
# which is the closest fit under the 2 M limit the width rounding allows.
NARROW_FILTERS = [8, 32, 60, 116, 232]
NARROW_MODEL = os.path.join(config.MODEL_DIR, "mha_resunet_narrow.h5")


def main():
    os.makedirs(config.MODEL_DIR, exist_ok=True)

    architecture = {"kind": "mha_resunet", "input_size": config.IMAGE_SIZE,
                    "filters": list(NARROW_FILTERS),
                    "heads": config.ATTENTION_HEADS, "dropout": 0.1}

    model = models.build_mha_resunet(input_size=architecture["input_size"],
                                     filters=architecture["filters"],
                                     heads=architecture["heads"],
                                     dropout=architecture["dropout"])
    train.compile_model(model)

    trainable = int(sum(np.prod(w.shape) for w in model.trainable_weights))
    print("widths            :", NARROW_FILTERS)
    print("trainable params  : {:,}".format(trainable))
    print("under the 2 M cap :", trainable < 2_000_000)

    train_batches, val_batches = train.prepare_data()

    start = time.time()
    history = train.fit(model, train_batches, val_batches)
    minutes = (time.time() - start) / 60

    best = max(history.history["val_dice_coefficient"])
    train.save_model(model, NARROW_MODEL, architecture)

    summary = complexity.summarise(model, NARROW_MODEL)
    complexity.print_summary("narrow model, trained from scratch", summary)

    with open(os.path.join(config.OUTPUT_DIR, "narrow_summary.json"), "w") as handle:
        json.dump({"filters": NARROW_FILTERS, "trainable_params": trainable,
                   "val_dice": round(float(best), 4),
                   "train_minutes": round(minutes, 1),
                   "complexity": summary}, handle, indent=2)

    print("\nbest validation Dice %.4f, trained in %.1f min" % (best, minutes))
    print("for comparison: 0.8712 at the old 40%% prune (3.28 M params), "
          "0.5916 at 55%% with inherited weights")


if __name__ == "__main__":
    main()
