"""
Model loading and prediction for both tasks.

The GUI has to be able to load "different models made and trained in Task 1 and
Task 2", so this module keeps a registry of everything it can find on disk and
gives the web server one common interface to run any of them:

    registry()                -> list of models the GUI can offer
    load(model_id)            -> load it into memory, return a spec sheet
    predict(model_id, image)  -> boolean mask at 512 x 512 plus timings

Task 2 models are TensorFlow and run inside this process.  Task 1 is a
scikit-learn forest that cannot be unpickled here (see task1_worker.py), so it
runs in a child process and this module talks to it over a pipe.  Both routes
come back looking the same to the caller.
"""

import glob
import json
import os
import subprocess
import sys
import threading
import time

import cv2
import numpy as np

PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TASK1_DIR = os.path.join(PROJECT_DIR, "task1")
TASK2_DIR = os.path.join(PROJECT_DIR, "task2")

# Task 2's modules import each other by plain name, so its folder goes on the
# path.  Task 1's does not, because its modules must not be imported here.
sys.path.insert(0, TASK2_DIR)

import config as task2_config     # noqa: E402
import metrics                    # noqa: E402  (task2/metrics.py - see below)

# Both tasks ship an identical metrics module so each folder can run standalone.
# The GUI deliberately uses one of them for everything, so a Task 1 score and a
# Task 2 score in the interface are always produced by the same code.

EVAL_SIZE = task2_config.EVAL_SIZE          # 512
TASK1_MODEL_DIR = os.path.join(TASK1_DIR, "outputs")
TASK2_MODEL_DIR = task2_config.MODEL_DIR

# Friendly names and one-line descriptions for the models we expect to find.
# Anything else found on disk still gets listed, just with a generic label.
KNOWN = {
    "rf_segmenter.joblib": ("Random Forest segmenter",
                            "41 features per pixel, 120 trees"),
    "mha_resunet.h5": ("MHA-ResUNet (proposed)",
                       "residual U-Net, self-attention, dual attention gates"),
    "mha_resunet_pruned.h5": ("MHA-ResUNet (pruned)",
                              "40% of channels removed, 66% fewer FLOPs"),
    "unet_pso.h5": ("U-Net (PSO-tuned baseline)",
                    "conventional U-Net, hyperparameters tuned by PSO"),
}


class ModelError(Exception):
    """Anything the user should see as a readable message in the GUI."""


# ------------------------------------------------------------------ Task 1
class Task1Bridge:
    """
    Keeps the mvi-env child process alive and exchanges JSON lines with it.

    One lock guards the pipe: the Flask development server is threaded, and two
    requests writing into the same stdin at once would interleave and desync the
    protocol.
    """

    def __init__(self):
        self.process = None
        self.lock = threading.Lock()
        self.loaded_path = None

    def interpreter(self):
        """mvi-env's python, which is the only one that can unpickle the forest."""
        candidate = os.path.join(PROJECT_DIR, "mvi-env", "bin", "python")
        if not os.path.exists(candidate):
            raise ModelError("mvi-env was not found at %s - Task 1 models need it"
                             % candidate)
        return candidate

    def start(self):
        if self.process is not None and self.process.poll() is None:
            return

        worker = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                              "task1_worker.py")
        self.process = subprocess.Popen(
            [self.interpreter(), "-u", worker],
            stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
            text=True, cwd=TASK1_DIR)
        self.loaded_path = None

        hello = self.process.stdout.readline()
        if not hello:
            raise ModelError("the Task 1 worker died on start-up: %s"
                             % self.process.stderr.read()[-400:])

    def send(self, request):
        with self.lock:
            self.start()
            try:
                self.process.stdin.write(json.dumps(request) + "\n")
                self.process.stdin.flush()
                line = self.process.stdout.readline()
            except (BrokenPipeError, ValueError):
                self.process = None
                raise ModelError("lost the connection to the Task 1 worker - "
                                 "load the model again to restart it")

            if not line:
                self.process = None
                raise ModelError("the Task 1 worker stopped responding - "
                                 "load the model again to restart it")

        reply = json.loads(line)
        if not reply.get("ok"):
            raise ModelError(reply.get("error", "the Task 1 worker failed"))
        return reply

    def load(self, path):
        info = self.send({"cmd": "load", "path": path})["info"]
        self.loaded_path = path
        return info

    def simulate_box(self, image_path, mask_path):
        return self.send({"cmd": "box", "image": image_path,
                          "mask": mask_path})["box"]

    def segment(self, image_path, box, want_probabilities=False):
        return self.send({"cmd": "segment", "image": image_path, "box": box,
                          "want_probabilities": want_probabilities})

    def stop(self):
        if self.process is not None and self.process.poll() is None:
            try:
                self.process.stdin.close()
                self.process.wait(timeout=5)
            except Exception:
                self.process.kill()
        self.process = None


# ------------------------------------------------------------------ Task 2
class Task2Runner:
    """
    Holds Task 2 networks in memory.

    Keras graph execution is not thread safe, so predictions are serialised.  At
    one image per request that costs nothing and it removes a whole class of
    intermittent failure from the demo.
    """

    def __init__(self):
        self.models = {}
        self.lock = threading.Lock()

    def load(self, path):
        import train                                  # imported late: pulls in TF

        with self.lock:
            model, architecture = train.load_model(path)

            # One throwaway prediction before the model is handed over.  The very
            # first call to a Keras model builds and compiles the graph, which
            # takes about half a second - so without this the first real scan
            # reports a latency ten times the true one, and the number shown in
            # the GUI would not match the measurements in the report.
            size = model.input_shape[1] or task2_config.IMAGE_SIZE
            model.predict(np.zeros((1, size, size, 1), np.float32), verbose=0)

            self.models[path] = model

        trainable = int(sum(np.prod(w.shape) for w in model.trainable_weights))
        info = {
            "input_size": architecture.get("input_size", task2_config.IMAGE_SIZE),
            "kind": architecture.get("kind", "?"),
            "trainable_params": trainable,
            "params_M": round(trainable / 1e6, 2),
            "size_mb": round(os.path.getsize(path) / (1024 * 1024), 2),
            "layers": len(model.layers),
        }

        filters = architecture.get("filters")
        if filters:
            info["filters"] = filters
        if architecture.get("base_filters"):
            info["base_filters"] = architecture["base_filters"]

        # FLOPs come from the measurements Task 2 already wrote out, rather than
        # being re-profiled here - re-profiling costs several seconds per model
        # and would make the GUI feel like it had hung.
        info.update(recorded_complexity(path))
        return info

    def predict(self, path, image, want_probabilities=False):
        """Run one scan and return a 512 x 512 boolean mask."""
        model = self.models.get(path)
        if model is None:
            raise ModelError("this Task 2 model is not loaded yet")

        size = model.input_shape[1] or task2_config.IMAGE_SIZE
        x = cv2.resize(image, (size, size), interpolation=cv2.INTER_LINEAR)
        x = x.astype(np.float32)[None, ..., None] / 255.0

        with self.lock:
            start = time.time()
            probabilities = model.predict(x, verbose=0)[0, ..., 0]
            elapsed = time.time() - start

        upscaled = cv2.resize(probabilities, (EVAL_SIZE, EVAL_SIZE),
                              interpolation=cv2.INTER_LINEAR)

        reply = {"mask": upscaled > 0.5, "latency_ms": round(elapsed * 1000, 1)}
        if want_probabilities:
            reply["probabilities"] = upscaled
        return reply


# complexity.json is keyed by the names evaluate.py used, not by file name.
COMPLEXITY_KEYS = {
    "mha_resunet.h5": "proposed",
    "mha_resunet_pruned.h5": "proposed_pruned",
    "unet_pso.h5": "unet_pso",
}


def recorded_complexity(path):
    """
    Pull FLOPs and latency for this model out of Task 2's complexity.json.

    Re-profiling FLOPs here would take several seconds per model and make the
    GUI look like it had hung, and the numbers would be the same ones Task 2
    already measured - so the published measurements are reused instead.
    """
    summary_path = os.path.join(task2_config.OUTPUT_DIR, "complexity.json")
    key = COMPLEXITY_KEYS.get(os.path.basename(path))
    if key is None or not os.path.exists(summary_path):
        return {}

    try:
        with open(summary_path) as handle:
            entry = json.load(handle).get(key)
    except (OSError, ValueError):
        return {}

    if not entry:
        return {}
    return {
        "gflops": entry.get("gflops"),
        "measured_latency_ms": round(1000 * entry["latency_s"], 1)
        if entry.get("latency_s") else None,
    }


# ------------------------------------------------------------------ registry
TASK1 = Task1Bridge()
TASK2 = Task2Runner()


def describe(path, task):
    name, note = KNOWN.get(os.path.basename(path),
                           (os.path.basename(path), "user supplied model"))
    return {
        "id": path,
        "name": name,
        "note": note,
        "task": task,
        "file": os.path.basename(path),
        "size_mb": round(os.path.getsize(path) / (1024 * 1024), 2),
        "loaded": path in TASK2.models or TASK1.loaded_path == path,
    }


def registry():
    """Every model the GUI can offer, newest task last so Task 1 lists first."""
    found = []
    for path in sorted(glob.glob(os.path.join(TASK1_MODEL_DIR, "*.joblib"))):
        if not os.path.basename(path).startswith("_"):     # skip scratch files
            found.append(describe(path, 1))
    for path in sorted(glob.glob(os.path.join(TASK2_MODEL_DIR, "*.h5"))):
        found.append(describe(path, 2))
    return found


def task_of(model_id):
    """Which task a model belongs to, decided by its file extension."""
    if model_id.endswith(".joblib"):
        return 1
    if model_id.endswith(".h5"):
        return 2
    raise ModelError("unrecognised model file - expected .joblib or .h5")


def check_path(model_id):
    """
    Resolve a model path and refuse anything outside the project.

    The GUI lets the user type a path so they can load their own retrained
    model, and this keeps that from turning into a way to read the whole disk.
    """
    resolved = os.path.realpath(model_id)
    if not resolved.startswith(os.path.realpath(PROJECT_DIR) + os.sep):
        raise ModelError("models must live inside the project folder")
    if not os.path.exists(resolved):
        raise ModelError("no such model file: %s" % os.path.basename(resolved))
    return resolved


def load(model_id):
    """Load a model and return a spec sheet for the GUI to display."""
    path = check_path(model_id)
    started = time.time()

    if task_of(path) == 1:
        info = TASK1.load(path)
    else:
        info = TASK2.load(path)

    info["load_seconds"] = round(time.time() - started, 2)
    return info


def predict(model_id, image_path, image, box=None, want_probabilities=False):
    """
    Run one model on one scan.

    Task 1 needs the user's box; Task 2 needs nothing but the image.  Both
    return a 512 x 512 boolean mask so the metrics and the overlays downstream
    do not care which one produced it.
    """
    model_path = check_path(model_id)

    if task_of(model_path) == 1:
        if box is None:
            raise ModelError("Task 1 is semi-automated - draw a box around the "
                             "lesion first, or use the simulated box")

        reply = TASK1.segment(image_path, list(box), want_probabilities)
        mask = decode_mask(reply["mask_png"])
        result = {"mask": mask, "latency_ms": reply["latency_ms"],
                  "box": reply["box"]}
        if reply.get("probability_png"):
            result["probabilities"] = decode_probabilities(reply["probability_png"])
        return result

    return TASK2.predict(model_path, image, want_probabilities)


def decode_mask(encoded):
    import base64
    raw = np.frombuffer(base64.b64decode(encoded), np.uint8)
    return cv2.imdecode(raw, cv2.IMREAD_GRAYSCALE) > 127


def decode_probabilities(encoded):
    import base64
    raw = np.frombuffer(base64.b64decode(encoded), np.uint8)
    return cv2.imdecode(raw, cv2.IMREAD_GRAYSCALE).astype(np.float32) / 255.0


def score(mask, truth):
    """Dice, IoU, accuracy, HD95, sensitivity and precision for one image."""
    values = metrics.all_metrics(mask, truth)
    return {key: round(float(value), 4) for key, value in values.items()}
