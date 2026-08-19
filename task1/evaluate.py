"""
Evaluate the proposed method on 50 unseen test scans and compare it with the
unsupervised baselines.

Run with:   python evaluate.py

Produces, in outputs/ :
    results_per_image.csv   every metric for every image and every method
    results_summary.csv     the averages that go into the report
    qualitative/*.png       side by side pictures for the report
"""

import os
import time

import cv2
import numpy as np
import pandas as pd

import baselines
import config
import dataset
import metrics
import segmenter


def draw_overlay(image, truth, pred, box, title, path):
    """Save a picture: grey MRI, green ground truth outline, red prediction."""
    canvas = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)

    for mask, colour in ((truth, (0, 255, 0)), (pred, (0, 0, 255))):
        contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL,
                                       cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(canvas, contours, -1, colour, 2)

    top, left, bottom, right = box
    cv2.rectangle(canvas, (left, top), (right, bottom), (255, 200, 0), 1)
    cv2.putText(canvas, title, (8, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
    cv2.imwrite(path, canvas)


def evaluate_method(name, run, test_paths, save_pictures=False):
    """Run one method over the test set and collect the metrics and timings."""
    rows = []
    picture_dir = os.path.join(config.OUTPUT_DIR, "qualitative")
    if save_pictures:
        os.makedirs(picture_dir, exist_ok=True)

    for i, path in enumerate(test_paths):
        image, truth = dataset.load_pair(path)
        box = dataset.simulate_user_box(truth, dataset.box_rng(path))

        start = time.time()
        pred = run(image, box)
        elapsed = time.time() - start

        row = metrics.all_metrics(pred, truth)
        row["method"] = name
        row["image"] = os.path.basename(path)
        row["tumour_type"] = dataset.tumour_class(path)
        row["seconds"] = elapsed
        rows.append(row)

        if save_pictures and i < 12:
            draw_overlay(image, truth, pred, box,
                         "%s  Dice %.3f" % (name, row["dice"]),
                         os.path.join(picture_dir, "%02d_%s.png" % (i, name)))

    return rows


def main():
    os.makedirs(config.OUTPUT_DIR, exist_ok=True)

    test_paths = dataset.list_images("test")
    rng = np.random.default_rng(config.EVAL_SEED)
    chosen = list(rng.choice(test_paths, config.N_TEST_IMAGES, replace=False))
    print("evaluating on %d unseen test scans\n" % len(chosen))

    model = segmenter.load_segmenter()

    methods = [("proposed", lambda im, b: model.segment(im, b))]
    methods += [(name, fn) for name, fn in baselines.BASELINES.items()]

    all_rows = []
    for name, run in methods:
        print("running %s ..." % name)
        all_rows += evaluate_method(name, run, chosen, save_pictures=(name == "proposed"))

    table = pd.DataFrame(all_rows)
    table.to_csv(os.path.join(config.OUTPUT_DIR, "results_per_image.csv"), index=False)

    summary = table.groupby("method").agg(
        dice=("dice", "mean"),
        dice_median=("dice", "median"),
        iou=("iou", "mean"),
        accuracy=("accuracy", "mean"),
        hd95=("hd95", "mean"),
        sensitivity=("sensitivity", "mean"),
        precision=("precision", "mean"),
        seconds=("seconds", "mean"),
    ).sort_values("dice", ascending=False)
    summary.to_csv(os.path.join(config.OUTPUT_DIR, "results_summary.csv"))

    print("\n================ results over %d test images ================" % len(chosen))
    print(summary.round(4).to_string())

    proposed = table[table.method == "proposed"]
    print("\nproposed method, Dice distribution:")
    print("  mean %.4f   median %.4f   std %.4f   worst %.4f"
          % (proposed.dice.mean(), proposed.dice.median(),
             proposed.dice.std(), proposed.dice.min()))
    print("  images with Dice >= 0.85 : %d / %d" % ((proposed.dice >= 0.85).sum(), len(proposed)))
    print("  images with Dice >= 0.70 : %d / %d" % ((proposed.dice >= 0.70).sum(), len(proposed)))

    print("\nproposed method, Dice by tumour type:")
    print(proposed.groupby("tumour_type").dice.agg(["count", "mean"]).round(4).to_string())

    size_mb = os.path.getsize(config.MODEL_PATH) / (1024 * 1024)
    print("\nmodel size on disk : %.2f MB" % size_mb)
    print("mean inference time: %.3f s per image (CPU)" % proposed.seconds.mean())
    print("\nresults written to", config.OUTPUT_DIR)


if __name__ == "__main__":
    main()
