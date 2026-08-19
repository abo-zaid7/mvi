# Task 1 — Semi-automated brain tumour segmentation (BRISC 2025)

Classical image processing + a scikit-learn Random Forest. **No TensorFlow, no deep
learning, no Haar cascades, no template matching**, as required by the brief.

---

## 1. The clinical use case

The intended use is **radiotherapy planning and follow-up volumetry for brain tumours**.
A radiologist or dosimetrist currently outlines the lesion by hand on every slice, which
takes roughly 10–20 minutes per study and varies noticeably between observers. This tool
replaces the outlining, not the decision: the clinician drags one rough rectangle around
the lesion, the algorithm returns a contour in ~64 ms, and the clinician accepts, redraws
or hand-edits it. The clinician stays in the loop, which is what makes a semi-automated
tool far easier to get through clinical governance than a fully automatic one.

## 2. Method

```
user drags a loose box
   ↓
CLAHE + z-score normalisation over the head region
   ↓
crop the box + 35 % padding, resample to 128 × 128   (scale normalisation)
   ↓
41 features per pixel
   ↓
Random Forest → tumour probability map
   ↓
Gaussian smooth → threshold → closing → fill holes → largest blob
   ↓
paste back into the full 512 × 512 frame
```

**The 41 features fall into three groups** (`features.py`):

| group | what it captures | examples |
|---|---|---|
| Appearance (32) | intensity, edges, blobs, texture at 5 scales | Gaussian σ=1…16, Sobel gradient, Laplacian-of-Gaussian, Hessian eigenvalues, local mean/std, median, bilateral |
| Geometry (4) | where the pixel sits inside the user's box | elliptical distance (1.0 = box edge), normalised x/y offset, inside/outside flag |
| Context (5) | pixel vs. the box centre and vs. outside the box | difference to core mean, difference to outside mean, z-score vs. core |

Groups 2 and 3 are the only places the user's input enters the model — that is what makes
the method semi-automated rather than automatic. The trained forest confirms this:
`box_elliptical_dist` alone carries 27 % of the feature importance.

**Why a box and not a click.** A single seed click was tried first and reached only 0.77
mean Dice; a loose box reached 0.87. A box is also one mouse gesture, so it costs the
clinician nothing extra, and it is the standard interaction in GrabCut, ITK-SNAP and
DeepIGeoS.

**Simulating the user.** The 50 test images obviously cannot be clicked by a real
radiologist, so — exactly as the interactive-segmentation literature does — the box is
derived from the ground-truth bounding box and then made deliberately sloppy: **each of
the four sides is pushed outwards by a random 2–20 %** of the lesion size
(`dataset.simulate_user_box`). The box is therefore never tight and never the same twice,
so the model cannot learn "the tumour exactly fills the box". Only the four box
coordinates are passed to the segmenter — **the mask itself is never seen at test time**.

## 3. Results — 50 unseen test scans

| method | Dice | Dice (median) | IoU | Accuracy | HD95 (px) | Sensitivity | Precision | s/image |
|---|---|---|---|---|---|---|---|---|
| **Proposed (RF)** | **0.8740** | **0.9284** | **0.8016** | **0.9930** | **11.37** | 0.9143 | 0.8635 | 0.064 |
| Chan-Vese active contour | 0.7813 | 0.8886 | 0.6910 | 0.9921 | 16.12 | 0.7382 | 0.8724 | 0.134 |
| Otsu threshold | 0.7744 | 0.8438 | 0.6714 | 0.9868 | 19.02 | 0.8333 | 0.7825 | 0.005 |
| k-means (k=3) | 0.7230 | 0.7235 | 0.5991 | 0.9801 | 23.72 | 0.8345 | 0.7300 | 0.030 |
| Seeded region growing | 0.6594 | 0.7515 | 0.5423 | 0.9840 | 21.65 | 0.6288 | 0.8436 | 0.005 |

- **43 / 50 images score Dice ≥ 0.85**, 45 / 50 score ≥ 0.70.
- The proposed method beats the best unsupervised baseline (morphological Chan-Vese,
  Marquez-Neila et al., *IEEE TPAMI* 2014) by **+9.3 Dice points** and cuts HD95 by 29 %.
- All baselines receive the **identical** user box, so the comparison is fair.

**By tumour type** — the useful weakness to discuss in the report:

| type | n | Dice |
|---|---|---|
| Meningioma | 18 | 0.943 |
| Pituitary | 19 | 0.897 |
| Glioma | 13 | 0.745 |

Meningiomas and pituitary adenomas are compact and well circumscribed; gliomas have
diffuse, infiltrative margins that are genuinely ambiguous even between human raters, so
an intensity-based method suffers most there.

## 4. Cost, latency and carbon footprint

| | value |
|---|---|
| Training data used | 400 scans × 700 sampled pixels = 280 000 rows |
| Training time | ~36 s total (15 s features + 21 s fit), 8-core laptop CPU |
| Model size on disk | **17.85 MB** |
| Inference latency | **64 ms per slice, CPU only** — no GPU at any stage |
| Peak memory | < 1 GB |

**Deliberate compression.** `model_compression.py` sweeps the forest size. A 300-tree /
leaf-4 forest gives 0.8748 Dice at 92.5 MB and 89 ms; the shipped 120-tree / leaf-20
forest gives **0.8740 Dice at 17.85 MB and 64 ms**. That is **81 % smaller and 28 %
faster for a 0.09 % Dice loss** — the headline number for the carbon-footprint section.
Because there is no GPU and no backprop, both training and inference energy are orders of
magnitude below a U-Net trained on the same data.

## 5. Files

| file | what it does |
|---|---|
| `config.py` | all paths and settings in one place |
| `dataset.py` | loading BRISC pairs, simulating the user's box |
| `features.py` | normalisation, ROI extraction, the 41 features |
| `segmenter.py` | the semi-automated segmenter |
| `baselines.py` | four unsupervised comparison methods |
| `metrics.py` | Dice, IoU, accuracy, HD95, sensitivity, precision |
| `train_model.py` | trains and saves the forest |
| `evaluate.py` | runs the 50-image comparison, writes CSVs and overlays |
| `model_compression.py` | size / latency / Dice sweep |
| `demo_interactive.py` | draw your own box with the mouse |

## 6. How to run

```bash
# from the project root — this env has NO tensorflow in it
python3 -m venv mvi-env
./mvi-env/bin/pip install -r task1/requirements.txt

cd task1
../mvi-env/bin/python train_model.py         # ~40 s
../mvi-env/bin/python evaluate.py            # ~1 min, writes outputs/
../mvi-env/bin/python model_compression.py   # ~4 min, optional
../mvi-env/bin/python demo_interactive.py    # draw a box yourself
```

Outputs land in `task1/outputs/`:

- `results_per_image.csv` — every metric, every image, every method
- `results_summary.csv` — the table in section 3
- `model_compression.csv` — the table in section 4
- `qualitative/*.png` — MRI with green ground truth, red prediction, blue user box

## 7. Notes and honest limitations

- BRISC images come in several resolutions (512², 256², 225², and a long tail). Everything
  is resampled to **512 × 512** before processing, and all metrics — including HD95 in
  pixels — are computed in that common space.
- The segmentation set contains no `no_tumor` cases, so the method is never asked to
  return an empty mask. In deployment the clinician simply would not draw a box.
- Gliomas are the weak point (0.745). Adding a second corrective scribble, or a texture
  descriptor tuned to infiltrative margins, is the obvious next step and is what Task 2's
  deep learning model should improve on.
- Results are reported on a fixed random sample of 50 test scans (`EVAL_SEED = 2`) so the
  numbers reproduce exactly.
