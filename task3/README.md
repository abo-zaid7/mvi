# Task 3 — Group GUI

A single web interface that loads the models built in Task 1 and Task 2 and runs them
on brain MRI scans from BRISC 2025, showing the ground truth, the segmentation
boundaries, and the numerical metrics side by side.

This file is the technical documentation the brief asks for: how to install it, how to
run it, how to use it, and what to do when something goes wrong. The rationale for the
design decisions is in `docs/CHAPTER5_GUI.md`, which is the material for Chapter 5 of
the group report.

---

## 1. What it does

| Requirement in the brief | Where it is in the interface |
|---|---|
| Load different models made and trained in Task 1 and 2 | **Models** panel, left. Four models are found automatically; "Load another model file" takes a path to any other `.joblib` or `.h5` in the project. |
| Visualise the ground truth image | **Ground truth** tab |
| Visualise the segmented boundaries | **Boundaries** tab (both outlines together), **Prediction** tab (model only) |
| Display important metrics numerically | **Metrics** panel, right: Dice, IoU, pixel accuracy, HD95, sensitivity, precision, inference time |
| Switch between English and one other language | **Language** menu, top right: English, Bahasa Melayu, Arabic |
| Audio for important functions | Cues on model loaded, prediction finished, error, and prompt for input; plus a spoken readout of the results |
| Credits window using the CReDiT system | **Credits** button, top right |

Two views go beyond the list because they answer questions the required ones cannot:
**Difference** separates the error into tumour that was missed and healthy tissue wrongly
included, and **Confidence** shows the probability map before it is thresholded.

---

## 2. Installing

### What has to be in place first

The GUI does not train anything. It loads the models the individual tasks already
produced, so those have to exist:

```
my-first-project/
├── brisc2025/segmentation_task/     the dataset (test/images and test/masks)
├── mvi-env/                         Task 1 environment (scikit-learn)
├── tf-env/                          Task 2 environment (TensorFlow)
├── task1/outputs/rf_segmenter.joblib
├── task2/outputs/models/*.h5        three networks plus their .json sidecars
├── MVI FINAL/                       the group's ultrasound half (see below)
│   ├── innovation_seed42_*.keras
│   ├── benchmark_unet.keras
│   └── Dataset_BUSI_with_GT/        benign/ malignant/ normal/
└── task3/                           this folder
```

If `task1/outputs/rf_segmenter.joblib` is missing, run `python train_model.py` in
`task1` first. If the `.h5` files are missing, run `python run_all.py` in `task2`.

### The second dataset

The GUI serves two datasets, because the group's Task 2 work was done twice on
two different problems:

| | Dataset | Models |
|---|---|---|
| ours | BRISC 2025 — brain MRI, 860 test scans | the forest and the three `.h5` networks |
| theirs | BUSI — breast ultrasound, 780 scans | the four `.keras` networks in `MVI FINAL` |

A model only understands the pictures it was trained on, so loading one switches
the scan picker to its dataset automatically. `Dataset_BUSI_with_GT` can sit
either in the project root or inside `MVI FINAL`; it is looked for on every
request, so it can be dropped in while the server is running. If it is not
there the ultrasound models are still listed, with a line saying where to put
the images.

### The one dependency the GUI adds

Flask. Everything else it needs (NumPy, OpenCV, SciPy, scikit-image, TensorFlow) is
already in `tf-env`.

```bash
../tf-env/bin/python -m pip install flask
```

`run_gui.sh` does this automatically if Flask is not there.

### Sound effects

The six cues are generated rather than downloaded, so there is no licence question and
nothing to fetch:

```bash
../tf-env/bin/python make_sounds.py
```

They are already committed, so this is only needed if `static/sounds/` is empty.

---

## 3. Running it

```bash
cd task3
./run_gui.sh
```

Then open **http://127.0.0.1:5050**.

Or without the script:

```bash
../tf-env/bin/python app.py
```

To use a different port:

```bash
MVI_GUI_PORT=8000 ./run_gui.sh
```

Press `Ctrl+C` in the terminal to stop it.

### Why two Python environments

The brief bans TensorFlow from Task 1, so Task 1 and Task 2 were built in separate
virtual environments. That has a consequence the GUI has to work around: the Task 1
forest was pickled under NumPy 2.x in `mvi-env`, and `tf-env` is held at NumPy 1.24
because TensorFlow 2.13 requires it. Loading that pickle inside `tf-env` fails outright
with `No module named 'numpy._core'`.

So the web server runs on `tf-env` and serves the three TensorFlow models in-process,
and it starts `task1_worker.py` as a small child process on `mvi-env`'s interpreter to
serve the forest. The two exchange one JSON object per line over a pipe. The forest is
loaded once and stays in memory, so a Task 1 prediction costs the same as calling the
segmenter directly.

Nothing has to be activated by hand — `app.py` starts the worker the first time a Task 1
model is loaded.

---

## 4. Using it

The four steps are listed in the top-left panel and tick themselves off as you go.

### 4.1 The six models

The list holds one Task 1 and one Task 2 model per student, named after their author:

| | Task 1 (semi-automated / classical) | Task 2 (deep learning) |
|---|---|---|
| **SALMAN (ME)** | BRISC 2025 brain MRI — Random Forest on 41 features per pixel, driven by one box the user drags | BRISC 2025 — MHA-ResUNet, structurally pruned |
| **AFNAN** | BUSI breast ultrasound — multi-Otsu and blob proposals, scored and refined with a Chan-Vese contour | BUSI — residual V-Net blocks, ASPP bottleneck, fused attention gates, pruned |
| **AMMAN** | FIVES retinal fundus — classical vessel segmentation | FIVES — attention U-Net with clDice, structurally pruned and fine-tuned |

Three students, three datasets, three ways of running. `members.py` is the one place
that says which is which, and everything else asks it:

- **Salman's Task 1** is unpickled by a helper process on `mvi-env` (§3.2).
- **Salman's and Afnan's Task 2** are loaded into this process and run live.
- **Afnan's Task 1** is her notebook pipeline ported into `afnan_task1.py`, so it runs
  live on CPU. There is no trained file behind it — the spec sheet describes the stages
  instead of weights, and reports 0 parameters, which is the honest answer for a
  classical method rather than a missing one.
- **Aman's two** are read back from the recorded results he handed over, not run. His
  Task 2 checkpoint is a Keras 3 file and this GUI runs TensorFlow 2.13, which cannot
  deserialise it; both of his share folders ship every test image, its ground truth and
  its prediction as PNGs together with the per-image metrics, which is the way his
  `HOW_TO_INTEGRATE.txt` asks for them to be used. The card says **recorded results**,
  the metrics are the ones measured when the results were produced rather than a
  rescore of the display copies, and the Confidence view says plainly that there is no
  probability map behind a finished mask.

Anything else — the PSO-tuned baselines, the unpruned models, the ablations, the folds —
is still loadable by path through "Load another model file". They are kept out of the
list so the six being marked are the six on screen.

### 4.2 Loading a model

Press **Load** on any model. A three-note chime means it is in memory, and its
specification appears underneath — for the forest: number of trees, features per pixel,
minimum leaf size; for the networks: architecture, input size, channel widths, tunable
parameters, GFLOPs, file size, measured latency.

The FLOPs and latency figures are the ones Task 2 measured and wrote to
`task2/outputs/complexity.json`. They are reused rather than re-profiled because
profiling takes several seconds per model and would make the interface look frozen.

The first Task 2 model takes a few seconds to load: building the Keras graph, and one
throwaway prediction so that the latency reported for your first real scan is the true
inference time rather than the graph-compilation time.

**To load a model that is not in the list** — for example one you retrained yourself —
open "Load another model file" and give the path, either relative to the project folder
or absolute. `.joblib` is treated as Task 1, `.h5` as Task 2, and `.keras` as one of
Afnan's ultrasound networks. Files outside the project folder are refused. This is how
to reach the rest of `afnan/`, where about thirty `.keras` files sit — the ablations,
the cross-validation folds and the other seeds are all loadable by path.

The ultrasound models report two extra fields. **Weights pruned to zero** is the share
of the network the pruning stage actually zeroed, measured from the file rather than
taken from its name. **Test time augmentation** says whether the scan is predicted four
times — as itself and its three flips — and averaged, which is how the group's notebook
scores them; without it the Dice shown here would sit a point below their report.

### 4.3 Choosing a scan

The default list is the scans the loaded model was actually evaluated on, so the numbers
you see are directly comparable with the report: 200 for BRISC, drawn with the same
random seed the evaluation used, and 102 for BUSI, read straight out of the group's
`per_image_seed42_pruned_tta.csv` so the two cannot drift apart. The **Show** menu
switches to every scan in the dataset or filters by class — glioma, meningioma and
pituitary for BRISC; benign, malignant and no-lesion for BUSI; and for FIVES, **the
report cases only**, which are the handful Amman's report works through. `N` and `P`
step through the list; **Random** jumps somewhere new.

Loading a model switches the picker to that model's dataset by itself. Running a breast
ultrasound network on a brain MRI would still produce a mask and a Dice score, and both
would mean nothing, so the interface does not let the two drift apart.

**To use your own scan**, drop a file onto the upload area or click it. JPG, PNG, BMP and
TIFF are accepted. If you also have the ground truth mask, tick "I also have the ground
truth mask" *before* choosing the file and you will be asked for the mask second. Without
a mask the scan is still segmented, but the metrics show a dash, because there is nothing
to score against.

### 4.4 Marking the lesion

Task 1 is semi-automated: it needs a rough box around the lesion, and the four box
coordinates are the only thing the user contributes. Either

- **drag a box** directly on the scan, or
- press **Simulate box** (`B`), which reproduces the box the Task 1 evaluation used: the
  ground truth bounding box with every side pushed outward by a random 2–20%, seeded from
  the file name so the same scan always gives the same box.

Only Salman's Task 1 model works this way. Afnan's and Amman's Task 1 methods find the
lesion themselves, and every Task 2 model is fully automatic, so the box controls
disappear for all five of them rather than being shown and ignored.

### 4.5 Running it and reading the result

Press **Segment this scan** (`R`).

- The **Dice score** is shown large, with a bar and a tick mark at 0.85 — the threshold
  the marking criteria are written against — and a one-line verdict.
- The remaining metrics are listed underneath, then two or three plain-English sentences
  interpreting them.
- Running a second model on the same scan adds a **Same scan, each model** table, which
  is the quickest way to show the pruned model matching the unpruned one, or Task 1's
  boundary beating Task 2's.
- **Read the results aloud** speaks the result in the current interface language.

### 4.6 Interpreting the metrics

| Metric | What it means | Direction |
|---|---|---|
| **Dice** | Overlap between the two masks, 0 to 1. The number the marking criteria use; 0.85+ is the distinction band. | higher |
| **IoU** | The same idea, stricter, so always a little lower than Dice. | higher |
| **Pixel accuracy** | Share of pixels labelled correctly. Looks impressive on brain MRI whatever the model does, because the tumour is a tiny part of the image — which is exactly why Dice is quoted instead. | higher |
| **HD95** | How far the predicted outline sits from the true one in pixels, ignoring the worst 5% of points. Overlap can look fine while this is bad. | **lower** |
| **Sensitivity** | Share of the tumour that was found. Missing tumour is the dangerous error. | higher |
| **Precision** | Share of what was marked that really was tumour. | higher |
| **Inference time** | One scan, this machine, excluding image loading and rendering. | lower |

All of them are computed at 512×512 by `task2/metrics.py`, which is the same code and the
same resolution both individual tasks were scored with — so a Task 1 number and a Task 2
number in this interface are directly comparable.

A worthwhile case to demonstrate: some glioma scans make the automatic model return an
empty or badly wrong mask with no warning at all. When that happens the interface says so
explicitly instead of just showing a zero, because that silent-failure mode is the honest
limitation of the Task 2 model and the reason the deployment design falls back to Task 1's
box.

### 4.7 Views

| Tab | Shows |
|---|---|
| Boundaries | Both outlines on the scan. Where the two lines separate is the error Dice measures. |
| Ground truth | The radiologist's mask on its own. |
| Prediction | The model's mask on its own, drawn identically so the two can be compared. |
| Difference | Agreement, tumour missed, and healthy tissue wrongly included. |
| Confidence | The probability map before it is cut at 0.5. |
| Scan only | The MRI with nothing drawn on it. |

`1`–`6` switch between them. **Fill** and **Line** control the overlay; **Show box** draws
the user's box into the exported image. **Export PNG** downloads a labelled
scan / ground truth / prediction strip for the report or the slides.

### 4.8 Display settings

Under **Display**: the interface design (see 4.8), overlay colour scheme (clinical
green/red, or the Okabe–Ito colour-blind safe set), three text sizes, higher contrast,
reduced animation, and sound volume. Everything is remembered in the browser.

### 4.9 Interface designs

The same interface comes in three designs. Every control, keyboard shortcut, metric and
translation is identical in all three — only the look and the arrangement change, so
switching is purely a matter of taste and does not affect any result.

**Aurora is the design the group settled on, and it is what the GUI opens with.** The
other two are kept so the choice can be shown and so the first design is never lost.

| Design | Look | Arrangement |
|---|---|---|
| **Aurora** (default) | Frosted translucent panels over a soft aurora, one cyan-to-violet gradient for every accent | Scan first: it fills the whole side at full height, set-up and results stacked beside it |
| **Original** | Dark slate, teal accent, rounded cards | Three columns: set-up, viewer, results |
| **Clinical workstation** | Radiology reading station — square corners, hairline rules, monospaced numbers, the scan in a black viewport | Set-up rail down the side at full height, viewer above, results docked along the bottom |

Two ways to switch:

* **Display → Interface design.** The change is immediate and nothing is lost — a loaded
  model and the result on screen both survive it.
* **The address bar**, which is the quicker one for a demonstration:
  `?skin=aurora`, `?skin=clinical` or `?skin=original`, e.g.
  `http://127.0.0.1:5050/?skin=original`. A skin named in the address wins over the
  saved preference and then becomes the saved preference.

How it works, and why the original cannot break: `style.css` is the original interface
and is always loaded. A design is one extra stylesheet layered on top of it
(`static/css/skin-clinical.css`, `static/css/skin-aurora.css`), so **Original** means no
extra stylesheet is loaded and not one line of the first design has changed. Deleting
both skin files and the `<select id="skin">` from the settings dialog would leave the
GUI exactly as it was; an untouched copy of the original four files is also kept in
`backups/gui-original/`.

### 4.10 Keyboard

| Key | Action |
|---|---|
| `R` | Run the model |
| `B` | Simulate the box |
| `N` / `P` | Next / previous scan |
| `1`–`6` | Switch view |
| `M` | Mute or unmute |
| `?` | Help window |
| `Esc` | Close a window |

---

## 5. Troubleshooting

**"Address already in use" on start-up.**
On macOS port 5000 is taken by the AirPlay Receiver in Control Centre, which is why the
default here is 5050. If 5050 is also busy: `MVI_GUI_PORT=8000 ./run_gui.sh`. To find what
is holding a port: `lsof -ti:5050`.

**The Task 1 model will not load.**
It runs in a helper process on `mvi-env`. Check that `mvi-env/bin/python` still exists
next to `task3`, then press **Load** again — that restarts the helper. If it still fails,
run the worker by hand to see the real error:
`../mvi-env/bin/python task1_worker.py` (it should print `{"ok": true, "ready": true}`
and wait for input; `Ctrl+C` to quit).

**"lost the connection to the Task 1 worker".**
The helper process died. Pressing **Load** again starts a new one. The web server itself
is unaffected.

**No scans are listed.**
`brisc2025/segmentation_task/test/images` has to exist next to `task3`. Uploading your own
scan works without the dataset.

**The metrics all show a dash.**
That scan has no ground truth mask — normal for an uploaded scan. The segmentation is
still shown and still exportable.

**A Task 2 model gives Dice 0.000 on a glioma.**
That is a real result, not a bug. The interface flags it as the silent-failure mode. Try
the same scan with the Task 1 model and a box for the contrast.

**There is no sound.**
Browsers block audio until the page has been clicked once — click anywhere. Then check the
speaker button in the header is on and the volume under **Display** is not at zero. Sound
is always a second channel here: every cue also appears as an on-screen message, so the
interface is fully usable in silence.

**Nothing is read aloud.**
Press the speech-bubble button in the header, or **Read the results aloud**. It uses the
browser's own speech synthesis, so it needs a voice installed for the selected language;
Bahasa Melayu and Arabic voices are not present on every machine, and if none is found the
browser's default voice is used.

**The interface looks cramped.**
It is designed for about 1300×800 or larger. Below 1150 px the results panel moves under
the viewer, and below 900 px everything stacks into one column.

**A model file I retrained will not load.**
It must be inside the project folder, and a Task 2 `.h5` needs its `.json` sidecar next to
it — that file describes the architecture so the weights can be loaded back.

---

## 6. Files

| File | What it is |
|---|---|
| `app.py` | Flask server and the HTTP API |
| `backends.py` | Model registry, the TensorFlow runner, and the bridge to the Task 1 worker |
| `members.py` | Who owns which model, and how each one has to be run |
| `busi_models.py` | Afnan's ultrasound networks: padded input pipeline, TTA, spec sheet |
| `afnan_task1.py` | Afnan's classical ultrasound pipeline, ported from her notebook |
| `fives_results.py` | Aman's recorded FIVES results: listing, masks and metrics |
| `task1_worker.py` | Runs on `mvi-env` and serves the Random Forest |
| `render.py` | Draws the six views and the export strip |
| `make_sounds.py` | Generates the six WAV cues |
| `credits.json` | The CReDiT contribution table — **edit this with the real names** |
| `templates/index.html` | The page |
| `static/css/style.css` | All styling, themes and the right-to-left mirroring — the original design |
| `static/css/skin-clinical.css` | The Clinical workstation design, layered on top of `style.css` |
| `static/css/skin-aurora.css` | The Aurora design, layered on top of `style.css` |
| `backups/gui-original/` | Untouched copies of the four original interface files |
| `static/js/i18n.js` | The three translations and the language switch |
| `static/js/audio.js` | Sound cues and the spoken readout |
| `static/js/app.js` | Everything else on the client |
| `run_gui.sh` | Launcher |
| `docs/CHAPTER5_GUI.md` | Report Chapter 5 material |
| `docs/COOPERATION.md` | Meeting records and peer-review log |

### HTTP API

| Route | Purpose |
|---|---|
| `GET /api/state` | Models found, view names, which datasets are present |
| `GET /api/images?dataset=&filter=` | Selectable scans for one dataset |
| `POST /api/load` | Load a model, return its specification |
| `POST /api/box` | Simulate the clinician's box |
| `POST /api/predict` | Run a model, return the metrics |
| `POST /api/upload` | Accept a scan and an optional mask |
| `GET /api/view?view=` | Render one view as a PNG |
| `GET /api/export` | Download the three-panel strip |
| `GET /api/credits` | The CReDiT table |

Every route answers with JSON containing `ok`, and every failure carries a readable
`error` message rather than a stack trace, so the interface can always say what went
wrong.

---

## 7. Before submitting

1. **Edit `credits.json`** — replace the four placeholder names and TP numbers with the
   real group members, and replace the placeholder contribution text. The file is read
   each time the Credits window is opened, so it can be edited while the server runs.
2. **Fill in `docs/COOPERATION.md`** with the group's real meeting records and
   peer-review comments.
3. **Add the citations marked `[ADD CITATION]`** in `docs/CHAPTER5_GUI.md`.
