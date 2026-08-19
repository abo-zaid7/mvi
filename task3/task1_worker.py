"""
Task 1 worker process.

Task 1 (Random Forest) and Task 2 (TensorFlow) were built in two separate
virtual environments on purpose - the brief bans TensorFlow from Task 1, and
keeping the environments apart is the cleanest proof of that.  The side effect
is that the two models cannot live in the same Python process: the forest was
pickled under numpy 2.x in mvi-env, and tf-env is stuck on numpy 1.24 because
TensorFlow 2.13 requires it.  Loading the pickle there fails outright.

So the GUI runs the web server in tf-env and starts this file as a small child
process using mvi-env's interpreter.  The two talk over stdin/stdout with one
JSON object per line.  The forest is loaded once and stays in memory, so a
request costs the same as calling the segmenter directly (about 60 ms).

Commands understood:

    {"cmd": "ping"}
    {"cmd": "load",    "path": "<file.joblib>"}
    {"cmd": "box",     "image": "<file.jpg>"}
    {"cmd": "segment", "image": "<file.jpg>", "box": [top, left, bottom, right]}

Every reply is one line of JSON with an "ok" field.
"""

import base64
import json
import os
import sys
import time

# The Task 1 modules import each other by plain name ("import config"), so the
# task1 folder has to be on the path.
HERE = os.path.dirname(os.path.abspath(__file__))
TASK1 = os.path.join(os.path.dirname(HERE), "task1")
sys.path.insert(0, TASK1)

import cv2                      # noqa: E402
import numpy as np              # noqa: E402

import config                   # noqa: E402
import dataset                  # noqa: E402
import segmenter                # noqa: E402


class Task1Model:
    """Holds the currently loaded forest, or None if nothing is loaded yet."""

    def __init__(self):
        self.segmenter = None
        self.path = None

    def load(self, path):
        loaded = segmenter.load_segmenter(path)
        forest = loaded.model

        self.segmenter = loaded
        self.path = path
        return {
            "trees": int(getattr(forest, "n_estimators", 0)),
            "features": int(getattr(forest, "n_features_in_", 0)),
            "min_samples_leaf": int(getattr(forest, "min_samples_leaf", 0)),
            "size_mb": round(os.path.getsize(path) / (1024 * 1024), 2),
            "patch_size": config.PATCH_SIZE,
            "roi_padding": config.ROI_PADDING,
        }


def encode_mask(mask):
    """A boolean mask as a base64 PNG, which is how it crosses the pipe."""
    ok, buffer = cv2.imencode(".png", (mask.astype(np.uint8) * 255))
    if not ok:
        raise RuntimeError("could not encode the mask as PNG")
    return base64.b64encode(buffer.tobytes()).decode("ascii")


def encode_probabilities(probs, roi):
    """
    The probability map is only 128x128 and lives inside the ROI, so it is
    pasted back into a full frame first.  That way the front end can show it on
    top of the MRI without having to know anything about the crop.
    """
    top, left, bottom, right = roi
    full = np.zeros((config.IMAGE_SIZE, config.IMAGE_SIZE), np.float32)
    full[top:bottom + 1, left:right + 1] = cv2.resize(
        probs, (right - left + 1, bottom - top + 1), interpolation=cv2.INTER_LINEAR)

    ok, buffer = cv2.imencode(".png", np.clip(full * 255, 0, 255).astype(np.uint8))
    if not ok:
        raise RuntimeError("could not encode the probability map as PNG")
    return base64.b64encode(buffer.tobytes()).decode("ascii")


def read_image(path):
    """Load a scan as greyscale at the common 512x512 working size."""
    image = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise IOError("could not read the image: " + os.path.basename(path))
    return cv2.resize(image, (config.IMAGE_SIZE, config.IMAGE_SIZE),
                      interpolation=cv2.INTER_LINEAR)


def read_mask(mask_path):
    """
    Load a ground truth mask, or None if there isn't one.

    The caller passes the path in rather than it being derived here: uploaded
    scans keep their masks somewhere else entirely, and guessing wrong would end
    up loading the scan itself as its own ground truth.
    """
    if not mask_path or not os.path.exists(mask_path):
        return None
    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
    if mask is None:
        return None
    return cv2.resize(mask, (config.IMAGE_SIZE, config.IMAGE_SIZE),
                      interpolation=cv2.INTER_NEAREST) > 127


def clamp_box(box):
    """Keep the box inside the image and make sure it has a sensible size."""
    top, left, bottom, right = (int(round(v)) for v in box)
    limit = config.IMAGE_SIZE - 1

    top, bottom = sorted((max(0, min(top, limit)), max(0, min(bottom, limit))))
    left, right = sorted((max(0, min(left, limit)), max(0, min(right, limit))))

    if bottom - top < 8 or right - left < 8:
        raise ValueError("the box is too small - drag a larger area")
    return top, left, bottom, right


# --------------------------------------------------------------- commands
def do_box(model, request):
    """Simulate the clinician's box from the ground truth, as in evaluation."""
    mask = read_mask(request.get("mask"))
    if mask is None or not mask.any():
        raise ValueError("this image has no ground truth mask, so a box cannot "
                         "be simulated - please drag one instead")

    # box_rng is seeded from the file name, so the same scan always gets the same
    # box and a demo can be repeated exactly.
    box = dataset.simulate_user_box(mask, dataset.box_rng(request["image"]))
    return {"box": [int(v) for v in box]}


def do_segment(model, request):
    """
    Run the semi-automated pipeline on one scan.

    The mask and the probability map are asked for separately through the public
    API rather than by reaching into the segmenter's internals.  That repeats the
    feature extraction, so the honest cost is about double the 63 ms quoted in
    the report - the timing reported back to the GUI is the mask only, which is
    the number that corresponds to the figure in the results.
    """
    if model.segmenter is None:
        raise RuntimeError("no Task 1 model is loaded yet")

    image = read_image(request["image"])
    box = clamp_box(request["box"])

    start = time.time()
    mask = model.segmenter.segment(image, box)
    elapsed = time.time() - start

    reply = {
        "mask_png": encode_mask(mask),
        "box": list(box),
        "latency_ms": round(elapsed * 1000, 1),
    }

    # The heat map view is optional, so it is only computed when asked for.
    if request.get("want_probabilities"):
        probs, _, roi = model.segmenter.probability_map(image, box)
        reply["probability_png"] = encode_probabilities(probs, roi)

    return reply


HANDLERS = {
    "ping": lambda model, request: {"alive": True},
    "load": lambda model, request: {"info": model.load(request["path"])},
    "box": do_box,
    "segment": do_segment,
}


def main():
    model = Task1Model()

    # Tell the parent we survived importing everything before it sends work.
    print(json.dumps({"ok": True, "ready": True}), flush=True)

    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue

        try:
            request = json.loads(line)
            handler = HANDLERS.get(request.get("cmd"))
            if handler is None:
                raise ValueError("unknown command: %r" % request.get("cmd"))

            reply = handler(model, request)
            reply["ok"] = True
        except Exception as error:
            reply = {"ok": False, "error": "%s" % error}

        print(json.dumps(reply), flush=True)


if __name__ == "__main__":
    main()
