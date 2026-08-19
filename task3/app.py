"""
Task 3 - the group GUI.

A small Flask application that puts the Task 1 and Task 2 models behind one web
interface.  Run it with:

    ./run_gui.sh                     (from the task3 folder)

or by hand:

    ../tf-env/bin/python app.py

then open http://127.0.0.1:5050 in a browser.  Set MVI_GUI_PORT to use a
different port.

Why a web page rather than a desktop window: the four of us are on different
operating systems, and a browser front end was the only option that looked and
behaved identically for all of us without four separate builds.  It also means
the marker only needs a browser to run it.  Chapter 5.1 of the report goes
through the choice properly.

The routes are:

    GET  /                       the page itself
    GET  /api/state              models, languages, dataset counts
    GET  /api/images             the test scans that can be picked
    POST /api/load               load a model into memory
    POST /api/box                simulate the clinician's box from ground truth
    POST /api/predict            run a model and score the result
    POST /api/upload             accept a user's own scan
    GET  /api/view               render one of the visualisations as a PNG
    GET  /api/credits            the CReDiT contribution table
    GET  /api/export             the current case as a PNG strip
"""

import glob
import io
import json
import os
import time
import uuid

import cv2
import numpy as np
from flask import Flask, jsonify, request, send_file, render_template

import backends
import render

HERE = os.path.dirname(os.path.abspath(__file__))
UPLOAD_DIR = os.path.join(HERE, "uploads")
CREDITS_FILE = os.path.join(HERE, "credits.json")

DATA_DIR = os.path.join(backends.PROJECT_DIR, "brisc2025", "segmentation_task")
TEST_IMAGES = os.path.join(DATA_DIR, "test", "images")

TUMOUR_CODES = {"gl": "glioma", "me": "meningioma", "pi": "pituitary",
                "nt": "no_tumour"}

UPLOAD_EXTENSIONS = (".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff")

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 16 * 1024 * 1024      # 16 MB is plenty for a scan

# Without this, Flask compiles index.html once at start-up and keeps serving that
# copy, so editing the template appears to do nothing until the server is
# restarted.  It costs one stat() per request, which does not matter here.
app.config["TEMPLATES_AUTO_RELOAD"] = True

# The last case that was run, kept so the view routes can re-render different
# visualisations without paying for another prediction.  One slot is enough:
# this is a single user demo tool, not a service.
CURRENT = {}


# --------------------------------------------------------------- helpers
class ApiError(Exception):
    """An error with a message meant for the user rather than a stack trace."""

    def __init__(self, message, status=400):
        super().__init__(message)
        self.message = message
        self.status = status


@app.errorhandler(ApiError)
def handle_api_error(error):
    return jsonify({"ok": False, "error": error.message}), error.status


@app.errorhandler(backends.ModelError)
def handle_model_error(error):
    return jsonify({"ok": False, "error": str(error)}), 400


@app.errorhandler(500)
def handle_crash(error):
    # Anything unexpected still comes back as JSON, so the front end can play the
    # error sound and show a message instead of silently doing nothing.
    return jsonify({"ok": False, "error": "the server hit an unexpected error - "
                                          "check the terminal for details"}), 500


def safe_image_path(name):
    """
    Resolve an image the front end asked for.

    Only the BRISC test folder and our own uploads folder are allowed, so a
    crafted name cannot be used to read files elsewhere on the machine.
    """
    for folder in (TEST_IMAGES, UPLOAD_DIR):
        candidate = os.path.realpath(os.path.join(folder, os.path.basename(name)))
        if candidate.startswith(os.path.realpath(folder)) and os.path.exists(candidate):
            return candidate
    raise ApiError("could not find that image: %s" % os.path.basename(name))


def load_scan(path):
    """The scan at the common 512 x 512 working size."""
    image = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise ApiError("that file could not be read as an image")
    return cv2.resize(image, (render.SIZE, render.SIZE), interpolation=cv2.INTER_LINEAR)


def truth_path_for(path):
    """
    Where the ground truth mask for a scan should be.

    BRISC keeps masks in a sibling "masks" folder with a .png extension; our own
    uploads keep them in uploads/masks.  The two cases are separated explicitly
    because deriving the upload path by string replacement gives back the scan
    itself, which would then be scored against itself.
    """
    stem = os.path.splitext(os.path.basename(path))[0]

    if os.path.dirname(os.path.realpath(path)) == os.path.realpath(UPLOAD_DIR):
        return os.path.join(UPLOAD_DIR, "masks", stem + ".png")

    folder = os.path.join(os.path.dirname(os.path.dirname(path)), "masks")
    return os.path.join(folder, stem + ".png")


def load_truth(path):
    """
    The matching ground truth mask, or None when there isn't one.

    Uploaded scans normally have no mask, and the GUI has to stay usable in that
    case - it just cannot report metrics, because there is nothing to score
    against.  Saying so plainly is better than showing a number that looks real.
    """
    mask_path = truth_path_for(path)
    if not os.path.exists(mask_path):
        return None

    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
    if mask is None:
        return None
    return cv2.resize(mask, (render.SIZE, render.SIZE),
                      interpolation=cv2.INTER_NEAREST) > 127


def tumour_type(name):
    parts = os.path.basename(name).split("_")
    return TUMOUR_CODES.get(parts[3], "unknown") if len(parts) > 3 else "unknown"


# --------------------------------------------------------------- pages
@app.route("/")
def index():
    return render_template("index.html")


@app.route("/api/state")
def state():
    """Everything the page needs to draw itself on first load."""
    models = backends.registry()
    return jsonify({
        "ok": True,
        "models": models,
        "test_images": len(glob.glob(os.path.join(TEST_IMAGES, "*.jpg"))),
        "views": list(render.VIEWS),
        "palettes": list(render.PALETTES),
        "image_size": render.SIZE,
        "has_dataset": os.path.isdir(TEST_IMAGES),
    })


@app.route("/api/images")
def images():
    """
    The scans the user can pick from.

    The 200 scans Task 2 was scored on are listed first and flagged, so the demo
    can stay on the images the reported numbers actually come from.
    """
    paths = sorted(glob.glob(os.path.join(TEST_IMAGES, "*.jpg")))
    if not paths:
        raise ApiError("no BRISC test images found at %s" % TEST_IMAGES, 404)

    # evaluate.py picks its 200 with this seed, so the same draw is repeated here.
    rng = np.random.default_rng(backends.task2_config.EVAL_SEED)
    evaluated = {os.path.basename(paths[i])
                 for i in rng.choice(len(paths), min(200, len(paths)), replace=False)}

    wanted = request.args.get("filter", "all")
    listing = []
    for path in paths:
        name = os.path.basename(path)
        kind = tumour_type(name)
        if wanted in ("glioma", "meningioma", "pituitary") and kind != wanted:
            continue
        if wanted == "evaluated" and name not in evaluated:
            continue
        listing.append({"name": name, "tumour_type": kind,
                        "in_test_set": name in evaluated})

    # Uploaded scans are listed too.  Only files, and only image files - the
    # masks sub-folder lives in here as well and is not something to segment.
    for path in sorted(glob.glob(os.path.join(UPLOAD_DIR, "*"))):
        if not os.path.isfile(path):
            continue
        if os.path.splitext(path)[1].lower() not in UPLOAD_EXTENSIONS:
            continue
        listing.append({"name": os.path.basename(path), "tumour_type": "uploaded",
                        "in_test_set": False, "uploaded": True})

    return jsonify({"ok": True, "images": listing, "total": len(listing)})


@app.route("/api/load", methods=["POST"])
def load_model():
    """Load one model into memory and hand back its spec sheet."""
    body = request.get_json(silent=True) or {}
    model_id = body.get("model")
    if not model_id:
        raise ApiError("no model was given")

    info = backends.load(model_id)
    return jsonify({"ok": True, "model": model_id, "task": backends.task_of(model_id),
                    "info": info})


@app.route("/api/box", methods=["POST"])
def simulate_box():
    """
    The simulated clinician box, for when the user would rather not drag one.

    This is the same routine the Task 1 evaluation used: the ground truth box
    with every side pushed out by a random 2-20%, so it is loose and never the
    same twice.  Only the four numbers go to the model.
    """
    body = request.get_json(silent=True) or {}
    path = safe_image_path(body.get("image", ""))

    box = backends.TASK1.simulate_box(path, truth_path_for(path))
    return jsonify({"ok": True, "box": box})


@app.route("/api/predict", methods=["POST"])
def predict():
    """Run one model on one scan, score it, and remember it for the view routes."""
    body = request.get_json(silent=True) or {}

    model_id = body.get("model")
    if not model_id:
        raise ApiError("load a model first")

    path = safe_image_path(body.get("image", ""))
    image = load_scan(path)
    truth = load_truth(path)

    box = body.get("box")
    if box is not None:
        if len(box) != 4:
            raise ApiError("a box needs four numbers: top, left, bottom, right")
        box = [int(round(float(v))) for v in box]

    started = time.time()
    result = backends.predict(model_id, path, image, box=box,
                              want_probabilities=True)
    total_ms = round((time.time() - started) * 1000, 1)

    mask = result["mask"]
    scores = backends.score(mask, truth) if truth is not None else None

    CURRENT.clear()
    CURRENT.update({
        "image": image,
        "truth": truth,
        "prediction": mask,
        "probabilities": result.get("probabilities"),
        "box": result.get("box", box),
        "image_name": os.path.basename(path),
        "model": model_id,
    })

    return jsonify({
        "ok": True,
        "model": model_id,
        "task": backends.task_of(model_id),
        "image": os.path.basename(path),
        "tumour_type": tumour_type(path),
        "metrics": scores,
        "has_truth": truth is not None,
        "box": result.get("box", box),
        "latency_ms": result["latency_ms"],
        "round_trip_ms": total_ms,
        "predicted_pixels": int(mask.sum()),
        "truth_pixels": int(truth.sum()) if truth is not None else None,
        "empty_prediction": not bool(mask.any()),
    })


@app.route("/api/upload", methods=["POST"])
def upload():
    """
    Accept the user's own scan.

    An optional mask can come with it; without one the GUI will still segment the
    image, it just cannot report metrics.
    """
    if "image" not in request.files:
        raise ApiError("no file was attached")

    uploaded = request.files["image"]
    if not uploaded.filename:
        raise ApiError("no file was attached")

    extension = os.path.splitext(uploaded.filename)[1].lower()
    if extension not in UPLOAD_EXTENSIONS:
        raise ApiError("unsupported file type - use JPG, PNG, BMP or TIFF")

    os.makedirs(os.path.join(UPLOAD_DIR, "masks"), exist_ok=True)

    # A short unique prefix keeps two people uploading "scan.png" apart.
    safe_name = "upload_%s_%s" % (uuid.uuid4().hex[:8],
                                  os.path.basename(uploaded.filename))
    target = os.path.join(UPLOAD_DIR, safe_name)
    uploaded.save(target)

    if cv2.imread(target, cv2.IMREAD_GRAYSCALE) is None:
        os.remove(target)
        raise ApiError("that file could not be read as an image")

    has_mask = False
    mask_file = request.files.get("mask")
    if mask_file and mask_file.filename:
        mask_target = os.path.join(UPLOAD_DIR, "masks",
                                   safe_name.rsplit(".", 1)[0] + ".png")
        mask_file.save(mask_target)
        if cv2.imread(mask_target, cv2.IMREAD_GRAYSCALE) is None:
            os.remove(mask_target)
            raise ApiError("the mask file could not be read as an image")
        has_mask = True

    return jsonify({"ok": True, "name": safe_name, "has_mask": has_mask})


@app.route("/api/view")
def view():
    """Render one visualisation of the current case as a PNG."""
    if not CURRENT:
        raise ApiError("nothing has been segmented yet", 404)

    wanted = request.args.get("view", "boundaries")
    if wanted not in render.VIEWS:
        raise ApiError("unknown view: %s" % wanted)

    try:
        opacity = float(request.args.get("opacity", 0.35))
        thickness = int(request.args.get("thickness", 2))
    except ValueError:
        raise ApiError("opacity and thickness must be numbers")

    png = render.build(
        wanted,
        CURRENT["image"],
        truth=CURRENT["truth"],
        prediction=CURRENT["prediction"],
        probabilities=CURRENT["probabilities"],
        box=CURRENT["box"] if request.args.get("show_box") == "1" else None,
        palette_name=request.args.get("palette", "clinical"),
        opacity=min(max(opacity, 0.0), 1.0),
        thickness=min(max(thickness, 1), 6),
    )

    response = send_file(io.BytesIO(png), mimetype="image/png")
    response.headers["Cache-Control"] = "no-store"
    return response


@app.route("/api/export")
def export():
    """Download the current case as a labelled strip, for the report and slides."""
    if not CURRENT:
        raise ApiError("nothing has been segmented yet", 404)

    png = render.build("strip", CURRENT["image"], truth=CURRENT["truth"],
                       prediction=CURRENT["prediction"],
                       palette_name=request.args.get("palette", "clinical"))

    name = "%s_%s.png" % (os.path.splitext(CURRENT["image_name"])[0],
                          os.path.splitext(os.path.basename(CURRENT["model"]))[0])
    return send_file(io.BytesIO(png), mimetype="image/png",
                     as_attachment=True, download_name=name)


@app.route("/api/credits")
def credits():
    """The CReDiT contribution table, kept in credits.json so it is easy to edit."""
    if not os.path.exists(CREDITS_FILE):
        raise ApiError("credits.json is missing", 404)

    try:
        with open(CREDITS_FILE, encoding="utf-8") as handle:
            return jsonify({"ok": True, "credits": json.load(handle)})
    except ValueError as error:
        raise ApiError("credits.json is not valid JSON: %s" % error)


def main():
    os.makedirs(UPLOAD_DIR, exist_ok=True)

    # 5050 rather than Flask's usual 5000: on macOS port 5000 is taken by the
    # AirPlay Receiver in Control Centre, which produces a confusing "address
    # already in use" on a machine where nothing of ours is running.  Override
    # with MVI_GUI_PORT if 5050 is busy too.
    port = int(os.environ.get("MVI_GUI_PORT", 5050))

    print("\n  Task 3 GUI - brain tumour segmentation")
    print("  models found:", len(backends.registry()))
    print("  open http://127.0.0.1:%d in your browser" % port)
    print("  press Ctrl+C to stop\n")

    try:
        app.run(host="127.0.0.1", port=port, debug=False, threaded=True)
    finally:
        backends.TASK1.stop()


if __name__ == "__main__":
    main()
