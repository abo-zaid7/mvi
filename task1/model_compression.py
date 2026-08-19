"""
Model size / latency sweep.

The report has to discuss the whole-life cost and the carbon footprint of the
method, so this script measures what is actually lost by shrinking the forest.
Fewer trees and bigger leaves mean a smaller file, less memory and a faster
prediction, and it turns out the Dice score barely moves.

Run with:   python model_compression.py
"""

import os
import time

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier

import config
import dataset
import metrics
import segmenter
import train_model

SETTINGS = [(300, 4), (200, 8), (150, 12), (120, 20), (100, 30), (60, 40)]
TEMP_MODEL = os.path.join(config.OUTPUT_DIR, "_size_check.joblib")


def main():
    os.makedirs(config.OUTPUT_DIR, exist_ok=True)
    rng = np.random.default_rng(config.RANDOM_SEED)

    train_paths = list(rng.choice(dataset.list_images("train"),
                                  config.N_TRAIN_IMAGES, replace=False))
    print("building the training set once ...")
    x, y = train_model.build_training_set(train_paths, rng)

    test_paths = list(np.random.default_rng(config.EVAL_SEED)
                      .choice(dataset.list_images("test"), config.N_TEST_IMAGES, replace=False))
    test_data = [dataset.load_pair(p) for p in test_paths]
    boxes = [dataset.simulate_user_box(mask, dataset.box_rng(p))
             for p, (_, mask) in zip(test_paths, test_data)]

    rows = []
    for n_trees, leaf in SETTINGS:
        forest = RandomForestClassifier(n_estimators=n_trees, min_samples_leaf=leaf,
                                        max_features="sqrt", class_weight="balanced",
                                        n_jobs=-1, random_state=0)
        forest.fit(x, y)
        joblib.dump(forest, TEMP_MODEL, compress=3)
        size_mb = os.path.getsize(TEMP_MODEL) / (1024 * 1024)

        model = segmenter.SemiAutoSegmenter(forest)
        start = time.time()
        scores = [metrics.dice_score(model.segment(image, box), mask)
                  for (image, mask), box in zip(test_data, boxes)]
        latency = (time.time() - start) / len(test_data)

        total_nodes = sum(tree.tree_.node_count for tree in forest.estimators_)
        rows.append({"trees": n_trees, "min_samples_leaf": leaf,
                     "total_nodes": total_nodes, "size_mb": round(size_mb, 2),
                     "dice": round(float(np.mean(scores)), 4),
                     "latency_s": round(latency, 4)})
        print("  trees=%3d leaf=%2d  %6.2f MB  dice %.4f  %.3f s/img"
              % (n_trees, leaf, size_mb, np.mean(scores), latency))

    if os.path.exists(TEMP_MODEL):
        os.remove(TEMP_MODEL)

    table = pd.DataFrame(rows)
    table.to_csv(os.path.join(config.OUTPUT_DIR, "model_compression.csv"), index=False)
    print("\n" + table.to_string(index=False))

    biggest, chosen = table.iloc[0], table[table.trees == config.N_TREES].iloc[0]
    print("\nchosen setting vs the largest forest:")
    print("  file size : %.2f MB -> %.2f MB  (%.0f%% smaller)"
          % (biggest.size_mb, chosen.size_mb, 100 * (1 - chosen.size_mb / biggest.size_mb)))
    print("  latency   : %.3f s -> %.3f s   (%.0f%% faster)"
          % (biggest.latency_s, chosen.latency_s,
             100 * (1 - chosen.latency_s / biggest.latency_s)))
    print("  Dice      : %.4f -> %.4f   (%.2f%% change)"
          % (biggest.dice, chosen.dice, 100 * (chosen.dice - biggest.dice) / biggest.dice))


if __name__ == "__main__":
    main()
