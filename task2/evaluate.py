"""
Evaluate every Task 2 model on 200 unseen BRISC test scans.

Run with:   python evaluate.py

The brief asks for Dice, HD95, Accuracy and IoU on 200 test images, plus FLOPs,
tunable parameters and model size for each network.  All of that ends up in
outputs/.

Predictions are made at the network's 192 x 192 working resolution and then
resized back up to 512 x 512 before scoring, so these numbers sit on exactly the
same masks as the Task 1 results and the two tasks can be compared directly.
"""

import json
import os
import time

import cv2
import numpy as np
import pandas as pd

import complexity
import config
import data
import metrics
import train

MODELS = [
    # The narrow model is the one that is submitted: 1,963,474 parameters, under
    # the module's 2 M limit.  "proposed_pruned" is kept in the list because the
    # 55% prune it now holds is the measured failure that sent the work to a
    # from-scratch retrain, and a failure that is reported needs a number.
    ("proposed_narrow", os.path.join(config.MODEL_DIR, "mha_resunet_narrow.h5")),
    ("proposed_pruned", config.PRUNED_MODEL),
    ("proposed", config.PROPOSED_MODEL),
    ("unet_pso", config.UNET_MODEL),
]


def load_test_set(n_images):
    """Return the network inputs plus the full resolution ground truth masks."""
    paths = data.list_images("test")
    rng = np.random.default_rng(config.EVAL_SEED)
    chosen = [paths[i] for i in rng.choice(len(paths), n_images, replace=False)]

    inputs = data.load_arrays(chosen, config.IMAGE_SIZE, with_masks=False)

    truths = np.zeros((len(chosen), config.EVAL_SIZE, config.EVAL_SIZE), bool)
    for i, path in enumerate(chosen):
        mask = cv2.imread(data.mask_path_for(path), cv2.IMREAD_GRAYSCALE)
        truths[i] = cv2.resize(mask, (config.EVAL_SIZE, config.EVAL_SIZE),
                               interpolation=cv2.INTER_NEAREST) > 127

    return chosen, inputs, truths


def tumour_class(path):
    codes = {"gl": "glioma", "me": "meningioma", "pi": "pituitary", "nt": "no_tumor"}
    return codes.get(os.path.basename(path).split("_")[3], "unknown")


def predict_masks(model, inputs):
    """Run the network, then push the masks back up to 512 x 512."""
    x = inputs.astype(np.float32)[..., None] / 255.0

    start = time.time()
    probabilities = model.predict(x, batch_size=config.BATCH_SIZE, verbose=0)
    seconds_each = (time.time() - start) / len(inputs)

    predictions = np.zeros((len(inputs), config.EVAL_SIZE, config.EVAL_SIZE), bool)
    for i, probability in enumerate(probabilities[..., 0]):
        upscaled = cv2.resize(probability, (config.EVAL_SIZE, config.EVAL_SIZE),
                              interpolation=cv2.INTER_LINEAR)
        predictions[i] = upscaled > 0.5

    return predictions, seconds_each


def save_overlays(paths, inputs, truths, predictions, scores, count=12):
    folder = os.path.join(config.OUTPUT_DIR, "qualitative")
    os.makedirs(folder, exist_ok=True)

    for i in range(min(count, len(paths))):
        image = cv2.imread(paths[i], cv2.IMREAD_GRAYSCALE)
        image = cv2.resize(image, (config.EVAL_SIZE, config.EVAL_SIZE))
        canvas = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)

        for mask, colour in ((truths[i], (0, 255, 0)), (predictions[i], (0, 0, 255))):
            contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL,
                                           cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(canvas, contours, -1, colour, 2)

        cv2.putText(canvas, "Dice %.3f" % scores[i], (8, 26),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.imwrite(os.path.join(folder, "%02d_proposed.png" % i), canvas)


def main():
    os.makedirs(config.OUTPUT_DIR, exist_ok=True)

    print("loading %d test scans ..." % config.N_TEST_IMAGES)
    paths, inputs, truths = load_test_set(config.N_TEST_IMAGES)

    rows, complexity_rows = [], {}

    for name, path in MODELS:
        if not os.path.exists(path):
            print("skipping %s - not trained yet (%s)" % (name, path))
            continue

        print("\nevaluating %s ..." % name)
        model, _ = train.load_model(path)

        predictions, seconds_each = predict_masks(model, inputs)
        scores = []
        for i in range(len(paths)):
            row = metrics.all_metrics(predictions[i], truths[i])
            row["model"] = name
            row["image"] = os.path.basename(paths[i])
            row["tumour_type"] = tumour_class(paths[i])
            rows.append(row)
            scores.append(row["dice"])

        summary = complexity.summarise(model, path)
        summary["batch_seconds_per_image"] = round(seconds_each, 4)
        complexity_rows[name] = summary
        complexity.print_summary(name, summary)

        if name == "proposed_pruned":
            save_overlays(paths, inputs, truths, predictions, scores)

    if not rows:
        print("\nnothing to evaluate - train the models first")
        return

    table = pd.DataFrame(rows)
    table.to_csv(os.path.join(config.OUTPUT_DIR, "results_per_image.csv"), index=False)

    summary = table.groupby("model").agg(
        dice=("dice", "mean"),
        dice_median=("dice", "median"),
        iou=("iou", "mean"),
        accuracy=("accuracy", "mean"),
        hd95=("hd95", "mean"),
        sensitivity=("sensitivity", "mean"),
        precision=("precision", "mean"),
    ).sort_values("dice", ascending=False)

    for name, row in complexity_rows.items():
        if name in summary.index:
            summary.loc[name, "gflops"] = row["gflops"]
            summary.loc[name, "params_M"] = round(row["trainable_params"] / 1e6, 2)
            summary.loc[name, "size_mb"] = row.get("size_mb", np.nan)
            summary.loc[name, "latency_ms"] = round(1000 * row["latency_s"], 1)

    summary.to_csv(os.path.join(config.OUTPUT_DIR, "results_summary.csv"))
    with open(os.path.join(config.OUTPUT_DIR, "complexity.json"), "w") as handle:
        json.dump(complexity_rows, handle, indent=2)

    print("\n=========== results over %d test images ===========" % config.N_TEST_IMAGES)
    print(summary.round(4).to_string())

    for name in summary.index:
        subset = table[table.model == name]
        print("\n%s: Dice >= 0.85 on %d/%d images, >= 0.70 on %d/%d"
              % (name, (subset.dice >= 0.85).sum(), len(subset),
                 (subset.dice >= 0.70).sum(), len(subset)))

    best = summary.index[0]
    print("\n%s, Dice by tumour type:" % best)
    print(table[table.model == best].groupby("tumour_type").dice
          .agg(["count", "mean"]).round(4).to_string())

    print("\nresults written to", config.OUTPUT_DIR)


if __name__ == "__main__":
    main()
