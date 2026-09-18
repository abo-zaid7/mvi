"""
The group's second set of Task 2 models - the breast ultrasound ones.

These were trained separately, in Afnan's notebooks under "afnan/", on the BUSI
dataset rather than on BRISC 2025.  They are Keras models saved in the newer
.keras format, so they cannot go through task2/train.py's loader, and their
input pipeline is different too: BUSI images are not square, and the notebooks
resize them with padding so the lesion is never stretched.

Everything specific to that is kept in this module, so backends.py only has to
know that a .keras file is loaded and predicted through here.

    registry()                       -> the models we offer from "afnan/"
    RUNNER.load(path)                -> load one, return a spec sheet
    RUNNER.predict(path, image_path) -> boolean mask at 512 x 512
"""

import os
import threading
import time

import cv2
import numpy as np

PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_DIR = os.path.join(PROJECT_DIR, "afnan")
EVAL_SIZE = 512

# Where the BUSI images are expected.  The notebooks read them from the author's
# own home folder, which obviously does not exist on anybody else's machine, so
# the GUI looks for the dataset in the two sensible places instead.
DATASET_NAME = "Dataset_BUSI_with_GT"
DATASET_CANDIDATES = [
    os.path.join(MODEL_DIR, DATASET_NAME),
    os.path.join(PROJECT_DIR, DATASET_NAME),
]

BUSI_CLASSES = ("benign", "malignant", "normal")


def dataset_dir():
    """The BUSI folder if it has been put in place, otherwise None."""
    for candidate in DATASET_CANDIDATES:
        if os.path.isdir(candidate):
            return candidate
    return None


# The models we list in the GUI.  There are about thirty .keras files in the
# folder - folds, seeds, ablations and abandoned experiments - and listing all of
# them would bury the ones that matter.  These are the four the notebook reports:
# the final model, the two stages it came from, and the baseline it is compared
# against.  Anything else can still be loaded by typing its path into the
# "load another model file" box.
#
# tta: the notebook scores the innovation models with four-flip test time
# augmentation, so the GUI does the same - otherwise the Dice it shows would be
# a point or two below the figure in their report.
CATALOGUE = [
    {
        "file": "innovation_seed42_pruned.keras",
        "name": "Innovation U-Net (final)",
        "note": "residual V-Net blocks, ASPP bottleneck, fused attention gates, "
                "pruned - reported Dice 0.805",
        "tta": True,
    },
    {
        "file": "innovation_seed42_final.keras",
        "name": "Innovation U-Net (unpruned)",
        "note": "the same network before pruning - reported Dice 0.814",
        "tta": True,
    },
    {
        "file": "innovation_seed42_structured.keras",
        "name": "Innovation U-Net (structured pruning only)",
        "note": "30% of filters masked, before the unstructured pass - Dice 0.801",
        "tta": True,
    },
    {
        "file": "benchmark_unet.keras",
        "name": "U-Net (PSO-tuned baseline)",
        "note": "conventional U-Net, hyperparameters searched with PSO",
        "tta": False,
    },
]

BY_FILE = {entry["file"]: entry for entry in CATALOGUE}


class BusiError(Exception):
    """Message meant for the user, raised by this module only."""


# ------------------------------------------------------------ custom objects
# The notebooks compile with their own loss and metrics.  The models are loaded
# with compile=False so the training configuration is skipped entirely, but the
# names are still handed to Keras in case a saved layer refers to one of them.
def _custom_objects():
    import tensorflow as tf
    from tensorflow import keras

    def dice_coef(y_true, y_pred, smooth=1e-6):
        yt = tf.reshape(y_true, [-1])
        yp = tf.reshape(y_pred, [-1])
        intersection = tf.reduce_sum(yt * yp)
        return (2. * intersection + smooth) / (tf.reduce_sum(yt) + tf.reduce_sum(yp) + smooth)

    def dice_loss(y_true, y_pred):
        return 1 - dice_coef(y_true, y_pred)

    def hybrid_loss(y_true, y_pred):
        bce = keras.losses.binary_crossentropy(y_true, y_pred)
        return 0.5 * tf.reduce_mean(bce) + 0.5 * dice_loss(y_true, y_pred)

    def iou_coef(y_true, y_pred, smooth=1e-6):
        yt = tf.reshape(y_true, [-1])
        yp = tf.reshape(y_pred, [-1])
        intersection = tf.reduce_sum(yt * yp)
        union = tf.reduce_sum(yt) + tf.reduce_sum(yp) - intersection
        return (intersection + smooth) / (union + smooth)

    return {"dice_coef": dice_coef, "dice_loss": dice_loss,
            "hybrid_loss": hybrid_loss, "iou_coef": iou_coef}


# --------------------------------------------------------------- geometry
def pad_resize(image, size, is_mask=False):
    """
    Scale the long side to `size` and centre the result on a black square.

    This is the notebooks' resize_with_padding.  Plainly resizing to a square
    instead would stretch the image, and these models have only ever seen the
    padded version - a stretched lesion is not the shape they were trained on.
    """
    height, width = image.shape[:2]
    scale = size / max(height, width)
    new_height, new_width = int(round(height * scale)), int(round(width * scale))

    interpolation = cv2.INTER_NEAREST if is_mask else cv2.INTER_AREA
    resized = cv2.resize(image, (new_width, new_height), interpolation=interpolation)

    canvas = np.zeros((size, size), image.dtype)
    top, left = (size - new_height) // 2, (size - new_width) // 2
    canvas[top:top + new_height, left:left + new_width] = resized
    return canvas


def unpad(prediction, height, width):
    """
    Undo pad_resize: cut the padding off and stretch back to the GUI's square.

    The prediction comes back on the padded canvas, so the bands the padding
    added have to be removed before the mask can line up with the picture the
    interface is showing - which is the whole scan squashed into 512 x 512.
    """
    size = prediction.shape[0]
    scale = size / max(height, width)
    new_height, new_width = int(round(height * scale)), int(round(width * scale))
    top, left = (size - new_height) // 2, (size - new_width) // 2

    cropped = prediction[top:top + new_height, left:left + new_width]
    return cv2.resize(cropped, (EVAL_SIZE, EVAL_SIZE), interpolation=cv2.INTER_LINEAR)


def four_flip_average(model, batch_input):
    """
    Test time augmentation: predict the image and its three flips, undo the
    flips, average.  It is what the notebook's tta_predict does, and it is worth
    about one Dice point on this dataset.
    """
    variants = [batch_input,
                batch_input[:, :, ::-1, :],
                batch_input[:, ::-1, :, :],
                batch_input[:, ::-1, ::-1, :]]

    predictions = [model.predict(v, verbose=0)[0, ..., 0] for v in variants]
    predictions[1] = predictions[1][:, ::-1]
    predictions[2] = predictions[2][::-1, :]
    predictions[3] = predictions[3][::-1, ::-1]
    return np.mean(predictions, axis=0)


# ----------------------------------------------------------------- runner
class BusiRunner:
    """
    Holds the loaded .keras models.

    Same reasoning as the Task 2 runner in backends.py: Keras is not thread
    safe, and the Flask development server is threaded, so predictions are put
    behind one lock.
    """

    def __init__(self):
        self.models = {}
        self.lock = threading.Lock()

    def load(self, path):
        from tensorflow import keras

        with self.lock:
            try:
                model = keras.models.load_model(path, custom_objects=_custom_objects(),
                                                compile=False)
            except Exception as error:
                raise BusiError("could not load %s: %s"
                                % (os.path.basename(path), error))

            # One throwaway prediction, for the same reason as in backends.py:
            # the first call builds the graph, and without this the first real
            # scan would report a latency several times the true one.
            size = model.input_shape[1]
            model.predict(np.zeros((1, size, size, 1), np.float32), verbose=0)
            self.models[path] = model

        entry = BY_FILE.get(os.path.basename(path), {})
        trainable = int(sum(np.prod(w.shape) for w in model.trainable_weights))

        info = {
            "kind": model.name,
            "input_size": model.input_shape[1],
            "trainable_params": trainable,
            "params_M": round(trainable / 1e6, 2),
            "layers": len(model.layers),
            "size_mb": round(os.path.getsize(path) / (1024 * 1024), 2),
            "dataset": "busi",
            "tta": bool(entry.get("tta")),
        }

        # How much of the network was actually zeroed.  The pruned models are the
        # point of this half of the project, so the number is worth showing
        # rather than leaving the user to take the file name's word for it.
        sparsity = weight_sparsity(model)
        if sparsity > 0.5:
            info["sparsity_pct"] = round(sparsity, 1)
        return info

    def predict(self, path, image_path, want_probabilities=False):
        """Run one scan through the padded pipeline, return a 512 x 512 mask."""
        model = self.models.get(path)
        if model is None:
            raise BusiError("this model is not loaded yet")

        original = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        if original is None:
            raise BusiError("that file could not be read as an image")

        size = model.input_shape[1]
        padded = pad_resize(original, size).astype(np.float32) / 255.0
        batch_input = padded[None, ..., None]

        use_tta = BY_FILE.get(os.path.basename(path), {}).get("tta", False)

        with self.lock:
            started = time.time()
            if use_tta:
                probabilities = four_flip_average(model, batch_input)
            else:
                probabilities = model.predict(batch_input, verbose=0)[0, ..., 0]
            elapsed = time.time() - started

        height, width = original.shape[:2]
        probabilities = unpad(probabilities, height, width)

        reply = {"mask": probabilities > 0.5,
                 "latency_ms": round(elapsed * 1000, 1)}
        if want_probabilities:
            reply["probabilities"] = probabilities
        return reply


def weight_sparsity(model):
    """Percentage of the weights that are exactly zero."""
    total = zeros = 0
    for weights in model.get_weights():
        total += weights.size
        zeros += int((weights == 0).sum())
    return 100.0 * zeros / total if total else 0.0


RUNNER = BusiRunner()


def registry():
    """The catalogue entries whose files are actually on disk."""
    found = []
    for entry in CATALOGUE:
        path = os.path.join(MODEL_DIR, entry["file"])
        if not os.path.exists(path):
            continue
        found.append({
            "id": path,
            "name": entry["name"],
            "note": entry["note"],
            "task": 2,
            "dataset": "busi",
            "file": entry["file"],
            "size_mb": round(os.path.getsize(path) / (1024 * 1024), 2),
            "loaded": path in RUNNER.models,
        })
    return found
