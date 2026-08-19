# Progress briefing pack — EE001-3.5-3-MVI

Everything below is checked against the actual files in `task1/outputs/` and
`task2/outputs/`. If he asks "where does that number come from", the CSV or JSON is named.

There is already a tighter 5-minute spoken script in `PRESENTATION_SCRIPT.md`. This document
is the wider prep: the stack, the coverage audit, the full result tables, and the questions.

---

## 1. The 30-second answer to "where are you?"

> "Both individual tasks are complete, evaluated and written up. Task 1 gets 0.874 Dice on
> the 50 test images, Task 2 gets 0.870 on the 200 test images — both above the 85%
> distinction threshold in the rubric. The report is drafted at about 5,500 words covering
> methods, results, whole-life cost, carbon and societal impact. What's outstanding is the
> group GUI in Task 3, and a small number of literature figures in my comparison table that
> I've flagged and still need to verify against the source papers."

That last clause matters — see §6. Volunteering a known gap reads much better than being
caught on it.

---

## 2. What to say — the narrative

### Use case (say this first, it frames everything)

Brain tumour segmentation on MRI for radiotherapy planning and follow-up volumetry, using
BRISC 2025. Currently a clinician outlines the tumour by hand on every slice — 10–20 minutes
per study, and not reproducible between raters, which matters because treatment response is
judged by comparing volumes over time. Task 1 is semi-automated (clinician gives one box),
Task 2 is fully automatic. Both are scored at 512×512 against the same masks so the numbers
compare directly.

### Task 1 — what it does

The clinician drags one loose rectangle round the lesion. For evaluation the box is
simulated from the ground truth, but **each side is pushed outward by a random 2–20%**, so
it is never tight and never repeats. Only the four coordinates reach the algorithm — never
the mask.

Pipeline: CLAHE + z-score normalisation → crop the box with padding → resample to 128×128 →
**41 features per pixel** → Random Forest → post-process the probability map with smoothing,
thresholding, closing and largest-connected-component selection.

Of the 41 features, 32 are appearance (multi-scale Gaussian, Sobel, Laplacian, Hessian
eigenvalues, local statistics) and 9 are geometry/context relative to the box. Those 9 are
the only route the user's input takes into the model — and the trained forest confirms they
carry the load: `box_elliptical_dist` alone is **26.6%** of total feature importance.

### Task 2 — what it does

An MHA-ResUNet. Three departures from a plain U-Net:

1. Residual blocks throughout.
2. Multi-head self-attention at the bottleneck.
3. **The novel component** — a dual attention gate on every skip connection.

Why self-attention: a convolution cannot compare a candidate lesion against the
corresponding structure across the midline, because the two are outside each other's
receptive field. Self-attention at the bottleneck can.

Why the dual gate: a standard Attention U-Net gate says *where* to attend but treats all
channels equally. Adding a squeeze-and-excitation branch means the skip is reweighted by
where **and** which features. Head-to-head against a plain U-Net under identical settings:
**0.719 vs 0.580** Dice (§3.3 of the report).

Baseline is a **PSO-tuned** conventional U-Net — filters, dropout, activation, learning rate,
24 fitness evaluations over 30.4 minutes — so the baseline being beaten is a tuned one.

Then **structured channel pruning**, four rounds, 40% total width reduction. Structured
specifically: zeroing small weights leaves the tensor shapes unchanged and the FLOP count
identical, so to actually cut FLOPs the channels have to disappear and the network be
rebuilt narrower.

### The pruning story — this is your strongest technical moment

One-shot pruning collapses the model. The per-round journal in `pruning_summary.json` shows
exactly why, and separates two distinct causes:

| round | Dice right after cut | after BN recalibration | after recovery |
|---|---|---|---|
| 1 | 0.468 | 0.807 | 0.868 |
| 2 | 0.497 | 0.816 | 0.866 |
| 3 | 0.237 | 0.763 | 0.865 |
| 4 | 0.473 | 0.730 | 0.863 |

The jump from column 1 to column 2 is **stale BatchNorm statistics** — free to fix, 100
forward-only batches, no gradient steps. The remaining gap to column 3 is **joint function
damage**, which only fine-tuning recovers. Being able to name and separate those two is the
kind of thing that reads as understanding rather than recipe-following.

---

## 3. What we used in the code

### Task 1 — no TensorFlow anywhere, as the brief requires

| what | library |
|---|---|
| Classifier | `sklearn.ensemble.RandomForestClassifier` |
| Features | `skimage` (Hessian), `scipy.ndimage`, `cv2` |
| Preprocessing | `cv2` (CLAHE), `numpy` |
| Baselines | `skimage.filters.threshold_otsu`, `sklearn.cluster.KMeans`, `skimage.segmentation.morphological_chan_vese`, `skimage.segmentation.random_walker` |
| Model I/O | `joblib` |
| Interactive demo | `matplotlib.widgets.RectangleSelector` |

Verified: `grep -riE "tensorflow|keras|torch" task1/*.py` returns nothing but a comment
saying it isn't used. No Haar cascade, no template matching.

Files: `features.py` (the 41 features), `segmenter.py` (pipeline + post-processing),
`train_model.py`, `baselines.py`, `evaluate.py`, `metrics.py`, `model_compression.py`,
`demo_interactive.py`.

### Task 2 — TensorFlow/Keras

| what | library |
|---|---|
| Model | `tensorflow.keras` |
| FLOP counting | `tensorflow.python.profiler.model_analyzer.profile` |
| PSO | hand-written in `pso.py` (no external optimiser library) |
| Pruning | hand-written structured pruning in `prune.py` |
| Post-processing / metrics | `scipy.ndimage`, `cv2`, `numpy` |

Files: `models.py` (MHA-ResUNet + plain U-Net), `losses.py`, `pso.py`, `prune.py`,
`pruning_ablation.py`, `complexity.py` (FLOPs/params/size/latency), `train.py`,
`evaluate.py`, `run_all.py`.

**Two separate virtual environments** (`mvi-env` for Task 1, `tf-env` for Task 2). Worth
mentioning unprompted — it's the cleanest possible evidence that Task 1 is TensorFlow-free.

Key settings: 192×192 working resolution, batch 16, 40 epochs, 3,933 training slices,
prune ratio 0.40 over 4 rounds, 12-epoch final fine-tune.

---

## 4. Results

### Task 1 — 50 unseen scans (`task1/outputs/results_summary.csv`)

| method | Dice | median | IoU | Accuracy | HD95 | Sens. | Prec. |
|---|---|---|---|---|---|---|---|
| **proposed (RF)** | **0.874** | 0.928 | 0.802 | 0.993 | **11.37** | 0.914 | 0.863 |
| Chan–Vese | 0.781 | 0.889 | 0.691 | 0.992 | 16.12 | 0.738 | 0.872 |
| Otsu | 0.774 | 0.844 | 0.671 | 0.987 | 19.02 | 0.833 | 0.783 |
| k-means | 0.723 | 0.723 | 0.599 | 0.980 | 23.72 | 0.835 | 0.730 |
| region growing | 0.659 | 0.752 | 0.542 | 0.984 | 21.65 | 0.629 | 0.844 |

All four baselines receive the identical box. Best baseline is Chan–Vese; proposed beats it
by **9.3 Dice points** and cuts boundary error **29%**.

- **43 of 50** scans are at or above 0.85. No zero-Dice failures. Worst case 0.174.
- Per class: meningioma 0.943 (n=18), pituitary 0.897 (n=19), glioma 0.745 (n=13).
- Training: 36 seconds, laptop CPU. Inference: 63 ms/image.

**Compression study** (`model_compression.csv`): 92.48 MB → 17.85 MB, a **81% size
reduction for a 0.09% Dice loss** (0.8748 → 0.8740).

### Task 2 — 200 unseen scans (`task2/outputs/results_summary.csv`)

| model | Dice | IoU | Accuracy | HD95 | GFLOPs | Params | Size | Latency |
|---|---|---|---|---|---|---|---|---|
| **proposed pruned** | **0.870** | 0.795 | 0.9962 | 14.77 | **5.08** | **3.28 M** | **12.86 MB** | 41.2 ms |
| proposed | 0.869 | 0.793 | 0.9960 | 14.49 | 14.93 | 8.84 M | 34.06 MB | 46.8 ms |
| U-Net (PSO-tuned) | 0.835 | 0.755 | 0.9952 | 25.98 | 7.62 | 4.37 M | 16.84 MB | 53.4 ms |

Two headline claims:

1. **Pruned matches unpruned on 66% fewer FLOPs** — 0.8699 vs 0.8692.
2. **Pruned beats the PSO-tuned baseline on every accuracy metric while using 33% fewer
   FLOPs.** Sharpest on boundary error: HD95 14.8 vs 26.0, a **43% reduction**.

Per class (pruned): meningioma 0.935, pituitary 0.867, glioma 0.786.

### Comparison and the failure mode

Task 1 and Task 2 are level on Dice (0.874 vs 0.870) but fail differently. Task 1 has the
better boundary (11.4 vs 14.8 px) because the box removes the localisation problem. Task 2
is better on gliomas because it learns cues a hand-designed feature set doesn't capture.

**The important finding:** two glioma scans score exactly zero with the automatic model.
Ruled out resolution by re-running at four input sizes — both stay at zero while control
cases degrade normally. On one, the model emits ~5,000 pixels at probability 0.999 in the
wrong structure. It fails **silently** — no uncertainty signal, just a confidently wrong
mask. (The PSO baseline has 5 such cases.)

That dictates the deployment design: run the automatic model first at zero interaction cost,
flag empty or low-confidence outputs, fall back to Task 1 on those — the clinician's box
supplies exactly the location information the automatic model got wrong.

### Cost and carbon

- Task 1: 36 s CPU training, 17.85 MB model, retraining is a same-day task. Dominant lifetime
  cost is clinician interaction time, not compute.
- Task 2: 166.2 min total across all four stages (proposed 49.8 + PSO 30.4 + baseline 52.8 +
  pruning 33.2) on one laptop GPU. No datacentre hardware at any stage.
- Carbon: ~22 g CO₂e for the whole Task 2 pipeline — roughly boiling a kettle twice. State
  clearly that these are **calculated estimates from stated assumptions** (M1 Max ~40 W,
  UK grid ~0.20 kg CO₂e/kWh), not meter readings.
- **Honest caveat to volunteer:** measured latency only improved 12% (46.8 → 41.2 ms) despite
  the 66% FLOPs cut, because at batch size 1 on a GPU the runtime is kernel-launch overhead,
  not arithmetic. The saving is real on CPU and edge hardware, but a 3× speedup claim would
  not be supported.

---

## 5. Did we do everything the assignment asked for?

### Task 1 requirements

| requirement | status |
|---|---|
| Semi-automated method tied to a specific disease/condition | ✅ brain tumour, RT planning + volumetry |
| Python with sklearn or other libraries | ✅ scikit-learn RF |
| No Haar cascade, no TensorFlow, no pattern matching | ✅ verified by grep; separate venv |
| One of the 6 permitted datasets | ✅ BRISC 2025 |
| Evaluated on 50 images | ✅ exactly 50 |
| Dice, HD95, Accuracy, IoU | ✅ all four, plus sensitivity/precision |
| Comparison vs ≥1 unsupervised method | ✅ four of them |
| Whole-life cost | ✅ report §4.1, itemised table |
| Carbon footprint via latency + hardware | ✅ report §5.3 |
| Societal and cultural impact | ✅ report §5.7 |

### Task 2 requirements

| requirement | status |
|---|---|
| Original DL method, same purpose and dataset | ✅ MHA-ResUNet |
| Fuzzy logic **or** neural network | ✅ neural network |
| Pruning to reduce FLOPs | ✅ structured, 66% reduction |
| Novel attention/skip/residual gates in a U-Net style architecture | ✅ dual attention gate — this is the claimed novelty |
| Compare vs conventional U-Net tuned by PSO | ✅ 24 evaluations, `pso_best.json` |
| BRISC 2025 — 200 testing images | ✅ exactly 200 |
| Dice, HD95, Accuracy, IoU | ✅ |
| FLOPs, tunable parameters, model size in MB | ✅ `complexity.json` |
| Whole-life cost, carbon optimisation, societal impact | ✅ report §4.2, §5.3, §5.7 |
| Comparison vs 1 SOTA automated tool | ⚠️ nnU-Net table present but figures marked `[verify]` |

### Rubric thresholds

The distinction band needs **85% Dice or higher**. Task 1 is 0.874 with 43/50 images above
0.85; Task 2 is 0.870. Both clear it. The distinction band also wants "comprehensively
justified real-world deployment, full whole-life cost analysis, quantified carbon footprint
with justification, and critical nuanced societal & cultural impact analysis" — all four are
in the report.

### What is genuinely outstanding

1. **Task 3 — the group GUI. Not started in this folder.** 15 marks for the demo plus it
   feeds the 35-mark group report criterion. Needs: model loading for Task 1 and 2 models,
   ground truth + boundary visualisation, numerical metrics, English + one other language,
   audio cues, CReDiT credits window. Both models already export cleanly (`.joblib` and
   `.h5`), so the integration work is UI, not modelling.
2. **Evidence of group cooperation** — the rubric explicitly wants GitHub issues, commit
   messages, peer review or meeting records. **This directory is not a git repository.**
   Worth fixing early; it's cheap marks and impossible to backfill convincingly.
3. **The `[verify]` markers in the report** — nnU-Net parameter count, FLOPs, model size and
   training schedule; TransUNet ~105 M and Swin-UNet ~27 M; the UK grid carbon intensity
   figure. These need checking against the source papers before submission.
4. **Group report assembly** — your material is Chapter 1 (or whichever number you are); the
   other chapters and Chapter 5 are other people's.

---

## 6. Questions he will ask

**"Isn't simulating the box from the ground truth cheating?"**
Only the four coordinates reach the algorithm, never the mask, and each side is jittered
outward 2–20% so it's loose and never repeats. That's the standard protocol for evaluating
interactive segmentation.

**"Why a Random Forest rather than a CNN in Task 1?"**
The brief separates the tasks — Task 1 is classical image processing, no TensorFlow. It's
also right on the merits: 36 seconds of CPU training, and the feature importances let me
demonstrate the user's box is actually driving the prediction rather than being ignored.

**"Why is glioma worse in both systems?"**
It's the disease, not the implementation. Meningiomas and pituitary adenomas are compact and
well-circumscribed; gliomas infiltrate, and their margins are ambiguous even between expert
human raters.

**"Why does the pruned model beat the unpruned one?"**
It's within noise — 0.8699 against 0.8692. The claim is that pruning cost nothing, not that
it helped. Fewer parameters plus a 12-epoch fine-tune is a mild regularisation effect.

**"What exactly is novel here?"**
The dual attention gate on the skip connections. Attention U-Net gates spatially but treats
channels uniformly; adding the squeeze-and-excitation branch gates on both axes. Backed by
the ablation: 0.719 vs 0.580 against a plain U-Net under identical settings.

**"Why 50 images for Task 1 and 200 for Task 2?"**
Those are the numbers the brief specifies for each task. Both are scored at 512×512 against
the same masks.

**"You trained at 192×192 — isn't that a limitation?"**
Yes, and it's stated as one in the report. Predictions are upsampled and scored at 512×512,
so the metrics are honest, but HD95 in particular would likely improve at native resolution.
That was not tested, so it isn't claimed.

**"What's the weakest part of your work?"**
The silent failure in Task 2. Mitigated with the fallback design but not solved — the real
fix is a proper confidence estimate, and that's the next step. *(Give this answer honestly;
the rubric rewards critical reflection, and having a named, quantified, investigated failure
mode is stronger than claiming there isn't one.)*

**Two he may ask that you should prepare rather than improvise:**

- **"Chan–Vese is from 2014 — is that within the past 10 years?"** The rubric says
  "unsupervised method from the past 10 years", and 2014 is now 12 years back. The
  implementation is the Marquez-Neila morphological formulation, and the underlying
  Chan–Vese is 2001. Either add one genuinely recent unsupervised baseline, or be ready to
  argue it's the standard modern implementation still in active use in scikit-image. Don't
  get caught mid-answer on this one.
- **"Show me it running."** The rubric says you must present with simulation of results.
  Have `task1/demo_interactive.py` open and working, and the qualitative PNGs in both
  `outputs/qualitative/` folders ready to display — including the failure cases in
  `task2/outputs/failure_cases.png`. Don't rely on live training.

---

## 7. If you only remember five things

1. Task 1 — **0.874 Dice** on 50 images, beats the best of four unsupervised baselines by
   **9.3 points**, trains in **36 seconds** on CPU, no TensorFlow.
2. Task 2 — **0.870 Dice** on 200 images, **66% fewer FLOPs** than unpruned at no accuracy
   cost, beats the PSO-tuned U-Net on every metric.
3. The novelty is the **dual attention gate**; the ablation is 0.719 vs 0.580.
4. The **silent failure** on two gliomas is the honest limitation, and it's what motivates
   the hybrid deployment design.
5. Outstanding: **the group GUI**, the cooperation evidence, and the `[verify]` literature
   figures.
