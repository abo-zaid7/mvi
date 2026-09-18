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
runs in a child process and this module talks to it over a pipe.  All routes
come back looking the same to the caller.

Three students, three datasets, one list.  members.py names who owns each of
the six models and what has to happen to run it - Salman's are BRISC 2025 brain
MRI, Afnan's are BUSI breast ultrasound through busi_models.py and her ported
Task 1 pipeline, and Aman's are FIVES retinal scans read back from recorded
results by fives_results.py.  Every entry in the registry says which dataset it
belongs to, and the GUI offers the matching scans.
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

import busi_models                 # noqa: E402  (Afnan's ultrasound networks)
import fives_results               # noqa: E402  (Amman's recorded FIVES results)
import members                     # noqa: E402  (who owns which model)

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
    "mha_resunet_narrow.h5": ("MHA-ResUNet (narrow)",
                              "pruned widths retrained from scratch, 1.96 M parameters"),
    "mha_resunet_pruned.h5": ("MHA-ResUNet (pruned, inherited weights)",
                              "55% of channels removed, weights carried over"),
    "unet_pso.h5": ("U-Net (GWO-tuned baseline)",
                    "conventional U-Net, hyperparameters tuned by GWO"),
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
    "mha_resunet_narrow.h5": "proposed_narrow",
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


def describe(entry):
    """One roster entry in the shape the front end draws a card from."""
    kind = entry["kind"]

    if kind == "recorded":
        # ".../<student>/<share folder>/results" -> the share folder's name
        label = os.path.basename(os.path.dirname(entry["results_dir"]))
        size_mb = None
    elif kind in ("afnan_task1", "adit_task1"):
        label = kind + ".py"
        size_mb = None
    else:
        label = os.path.basename(entry["id"])
        size_mb = (round(os.path.getsize(entry["id"]) / (1024 * 1024), 2)
                   if os.path.exists(entry["id"]) else None)

    return {
        "id": entry["id"],
        "name": entry["name"],
        "note": entry["note"],
        "task": entry["task"],
        "dataset": entry["dataset"],
        "file": label,
        "size_mb": size_mb,
        "recorded": kind == "recorded",
        "needs_box": entry["needs_box"],
        "available": members.present(entry),
        "loaded": entry["id"] in LOADED,
    }


def registry():
    """
    The six models the GUI offers: one Task 1 and one Task 2 per student.

    Ordered task first and then by author, which is the order the report
    chapters are in.  Anything else on disk can still be loaded by typing its
    path into the "load another model file" box - it is left out of the list so
    the six that are being marked are the six that are shown.
    """
    return [describe(entry) for entry in members.ROSTER]


def dataset_of(model_id):
    """Which dataset a model expects, which decides the scans the GUI offers."""
    entry = members.find(model_id)
    if entry:
        return entry["dataset"]
    return "busi" if model_id.endswith(".keras") else "brisc"


def task_of(model_id):
    """Which task a model belongs to."""
    entry = members.find(model_id)
    if entry:
        return entry["task"]
    if model_id.endswith(".joblib"):
        return 1
    if model_id.endswith((".h5", ".keras")):
        return 2
    raise ModelError("unrecognised model file - expected .joblib, .h5 or .keras")


def needs_box(model_id):
    """Whether this model is the semi-automated one that waits for a box."""
    entry = members.find(model_id)
    if entry:
        return entry["needs_box"]
    return task_of(model_id) == 1


def recorded_entry(model_id):
    """The roster entry if this id is one of the recorded ones, else None."""
    entry = members.find(model_id)
    return entry if entry and entry["kind"] == "recorded" else None


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


# Which roster ids the user has loaded this session.  A recorded entry and a
# ported pipeline have nothing to hold in memory, so "loaded" is the only thing
# that distinguishes them from not having been chosen yet.
LOADED = set()


def load(model_id):
    """Load a model and return a spec sheet for the GUI to display."""
    started = time.time()
    entry = members.find(model_id)
    kind = entry["kind"] if entry else None

    if kind == "recorded":
        if not fives_results.available(entry):
            raise ModelError("%s's recorded results were not found at %s"
                             % (entry["name"].title(), entry["results_dir"]))
        info = fives_results.spec_sheet(entry)

    elif kind == "afnan_task1":
        import afnan_task1
        info = afnan_task1.spec_sheet()

    elif kind == "adit_task1":
        import adit_task1
        info = adit_task1.spec_sheet()

    else:
        path = check_path(model_id)
        if task_of(path) == 1:
            info = TASK1.load(path)
        elif dataset_of(path) == "busi":
            try:
                info = busi_models.RUNNER.load(path)
            except busi_models.BusiError as error:
                raise ModelError(str(error))
        else:
            info = TASK2.load(path)

    LOADED.add(model_id)
    info["load_seconds"] = round(time.time() - started, 2)
    return info


def predict(model_id, image_path, image, box=None, want_probabilities=False):
    """
    Run one model on one scan.

    Task 1 needs the user's box; Task 2 needs nothing but the image.  Both
    return a 512 x 512 boolean mask so the metrics and the overlays downstream
    do not care which one produced it.
    """
    entry = members.find(model_id)
    kind = entry["kind"] if entry else None

    if kind == "recorded":
        name = os.path.splitext(os.path.basename(image_path))[0]
        if name.endswith("_input"):
            name = name[:-len("_input")]
        reply = fives_results.prediction(entry, name)
        if reply is None:
            raise ModelError("there is no recorded prediction for %s - pick one "
                             "of the images this model was scored on" % name)
        return reply

    if kind == "afnan_task1":
        import afnan_task1
        return afnan_task1.segment(image)

    if kind == "adit_task1":
        import adit_task1
        if box is None:
            raise ModelError("this method is semi-automated - mark the lesion "
                             "first, the centre of the box is its seed point")
        return adit_task1.segment(image, box)

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

    if dataset_of(model_path) == "busi":
        # These models are given the file rather than the 512 x 512 array: their
        # pipeline pads the original to a square instead of stretching it, so it
        # needs the scan at its real proportions.
        try:
            return busi_models.RUNNER.predict(model_path, image_path,
                                              want_probabilities)
        except busi_models.BusiError as error:
            raise ModelError(str(error))

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
