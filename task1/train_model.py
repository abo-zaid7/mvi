"""
Train the Random Forest pixel classifier.

Run with:   python train_model.py

Pixels are sampled rather than used in full - a 128x128 patch holds 16384
pixels, and with 400 scans that would be 6.5 million rows.  Sampling 700 pixels
per scan keeps training to a couple of minutes on a laptop CPU while losing
almost nothing in accuracy.

The sampling is deliberately unbalanced towards the lesion boundary, because
that is where the errors actually happen; the middle of a tumour and the far
background are both easy.
"""

import os
import time

import joblib
import numpy as np
from scipy import ndimage as ndi
from sklearn.ensemble import RandomForestClassifier

import config
import dataset
import features


def sample_pixels(image_path, n_pixels, rng):
    """Turn one training scan into (features, labels) for a sample of pixels."""
    image, mask = dataset.load_pair(image_path)
    box = dataset.simulate_user_box(mask, rng)

    normalised = features.normalise(image)
    patch, box_in_patch, _ = features.extract_patch(normalised, box)
    mask_patch, _, _ = features.extract_patch(mask.astype(np.float32), box)

    x = features.build_features(patch, box_in_patch)
    labels = (mask_patch.reshape(-1) > 0.5).astype(np.uint8)

    # a band of pixels either side of the true boundary
    lesion = mask_patch > 0.5
    band = np.logical_xor(ndi.binary_dilation(lesion, np.ones((9, 9))),
                          ndi.binary_erosion(lesion, np.ones((9, 9)))).reshape(-1)

    tumour = np.where(labels == 1)[0]
    near_edge = np.where((labels == 0) & band)[0]
    far_away = np.where((labels == 0) & ~band)[0]

    def pick(pool, how_many):
        if len(pool) == 0:
            return np.array([], dtype=int)
        return rng.choice(pool, min(how_many, len(pool)), replace=False)

    half = n_pixels // 2
    chosen = np.concatenate([pick(tumour, half),
                             pick(near_edge, half // 2),
                             pick(far_away, half // 2)])
    return x[chosen], labels[chosen]


def build_training_set(image_paths, rng):
    all_x, all_y = [], []
    for i, path in enumerate(image_paths, 1):
        x, y = sample_pixels(path, config.PIXELS_PER_IMAGE, rng)
        all_x.append(x)
        all_y.append(y)
        if i % 50 == 0:
            print("  sampled %d / %d scans" % (i, len(image_paths)))
    return np.vstack(all_x), np.concatenate(all_y)


def main():
    os.makedirs(config.OUTPUT_DIR, exist_ok=True)
    rng = np.random.default_rng(config.RANDOM_SEED)

    train_paths = dataset.list_images("train")
    print("training scans available:", len(train_paths))
    chosen = list(rng.choice(train_paths, config.N_TRAIN_IMAGES, replace=False))

    print("building the training set ...")
    start = time.time()
    x, y = build_training_set(chosen, rng)
    print("  %s pixels, %d features, %.0f s" % (x.shape[0], x.shape[1], time.time() - start))
    print("  class balance: %.1f%% tumour" % (100.0 * y.mean()))

    print("fitting the random forest ...")
    start = time.time()
    forest = RandomForestClassifier(n_estimators=config.N_TREES,
                                    min_samples_leaf=config.MIN_SAMPLES_LEAF,
                                    max_features="sqrt",
                                    class_weight="balanced",
                                    n_jobs=-1,
                                    random_state=0)
    forest.fit(x, y)
    print("  done in %.0f s" % (time.time() - start))

    joblib.dump(forest, config.MODEL_PATH, compress=3)
    size_mb = os.path.getsize(config.MODEL_PATH) / (1024 * 1024)
    print("saved to %s  (%.2f MB)" % (config.MODEL_PATH, size_mb))

    print("\ntop 10 features by importance:")
    names = features.feature_names()
    order = np.argsort(forest.feature_importances_)[::-1][:10]
    for rank, idx in enumerate(order, 1):
        print("  %2d. %-22s %.4f" % (rank, names[idx], forest.feature_importances_[idx]))


if __name__ == "__main__":
    main()
