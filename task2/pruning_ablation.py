"""
Ablation: why the pruning is done in rounds rather than in one go.

This measures the Dice of the pruned network *immediately after the weights are
transferred*, with no fine-tuning, for a range of one-shot pruning ratios.  It
is the experiment that justifies the design in prune.py, and it also doubles as
a correctness test: at ratio 0.0 the rebuilt network must score exactly what the
original scored, because nothing has actually been removed.

Run with:   python pruning_ablation.py
"""

import os

import numpy as np
import pandas as pd

import config
import data
import prune
import train

RATIOS = [0.0, 0.05, 0.1, 0.15, 0.2, 0.3, 0.4]


def main():
    model, architecture = train.load_model(config.PROPOSED_MODEL)
    train.compile_model(model)

    # The recalibration has to see the same number of batches as the real
    # pipeline does.  BatchNorm's running average keeps 0.9^n of whatever it
    # started from, so a couple of dozen batches is not enough to wash out the
    # reset values and the recalibrated score comes out worse than the
    # untouched one.
    train_batches, val_batches = train.prepare_data()

    baseline = model.evaluate(val_batches, verbose=0, return_dict=True)["dice_coefficient"]
    print("original model: validation Dice %.4f\n" % baseline)

    rows = []
    for ratio in RATIOS:
        pruned, filters = prune.prune_model(model, architecture["filters"], ratio)
        train.compile_model(pruned)

        transferred = pruned.evaluate(val_batches, verbose=0,
                                      return_dict=True)["dice_coefficient"]
        prune.recalibrate_batchnorm(pruned, train_batches, config.BN_RECALIBRATION_BATCHES)
        recalibrated = pruned.evaluate(val_batches, verbose=0,
                                       return_dict=True)["dice_coefficient"]

        rows.append({"ratio": ratio, "filters": str(filters),
                     "dice_after_transfer": round(float(transferred), 4),
                     "dice_after_bn_recalibration": round(float(recalibrated), 4)})
        print("  ratio %.2f  %-28s transfer %.4f  after BN recal %.4f"
              % (ratio, str(filters), transferred, recalibrated))
        del pruned

    table = pd.DataFrame(rows)
    os.makedirs(config.OUTPUT_DIR, exist_ok=True)
    table.to_csv(os.path.join(config.OUTPUT_DIR, "pruning_ablation.csv"), index=False)

    print("\n" + table.to_string(index=False))
    print("\nsanity check - at ratio 0.0 the rebuilt model should match the original:")
    print("  original %.4f vs rebuilt %.4f" % (baseline, rows[0]["dice_after_transfer"]))
    print("\nwritten to", os.path.join(config.OUTPUT_DIR, "pruning_ablation.csv"))


if __name__ == "__main__":
    main()
