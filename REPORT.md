# Medical Image Segmentation using Image Processing and Deep Learning Techniques

**Module:** EE001-3.5-3-MVI — Machine Vision
**Dataset:** BRISC 2025 — Brain Tumour MRI (Fateh et al., 2025)
**Chapter:** Task 1 (semi-automated image processing) and Task 2 (deep learning with pruning)

> **Note before submission.** Every number in this report was produced by the code in
> `task1/` and `task2/` and can be regenerated (see §7). The *citations* in §1 are to
> well-known published works, but you should verify each one against the actual paper before
> submitting — check the year, venue and author list. Where a comparison to published
> figures is approximate it is marked **[verify]**. The carbon figures in §5.3 are
> calculated estimates from stated assumptions, not meter readings, and are labelled as such.

---

## 1. Literature review and justification of the use case

### 1.1 The clinical problem

Brain tumours require accurate delineation of the lesion boundary for three distinct
clinical purposes: diagnosis, radiotherapy planning, and follow-up volumetry to judge
whether a tumour is responding to treatment. In current practice a radiologist or
dosimetrist outlines the lesion by hand on every slice of the study. This is slow —
typically 10–20 minutes per study — and, more importantly, it is inconsistent: the same
observer will not draw the same boundary twice, and two observers will differ more still.
Because response assessment depends on comparing a volume today against a volume three
months ago, that inconsistency directly degrades the clinical decision.

The chosen use case is therefore **radiotherapy planning and follow-up volumetry**, and the
target is not to replace the clinician but to remove the manual outlining step while leaving
the clinical judgement where it belongs.

### 1.2 Why a semi-automated method (Task 1)

Fully automatic segmentation has advanced enormously, but adoption in clinical workflows
remains limited, and one recurring obstacle is regulatory and professional accountability: a
system that produces a contour with no human in the loop must be validated to a much higher
standard than one that assists a clinician who remains responsible for the result.

Interactive, or semi-automated, segmentation addresses this directly. The user supplies a
minimal cue — a click, a scribble, or a bounding box — and the algorithm completes the
boundary. The interaction paradigm has a long lineage: **GrabCut** (Rother et al., *ACM
TOG*, 2004) established the bounding-box interaction; **random walker** segmentation
(Grady, *IEEE TPAMI*, 2006) established seed-based propagation; and **ITK-SNAP** (Yushkevich
et al., *NeuroImage*, 2006) brought semi-automated active-contour segmentation into routine
neuroimaging use. More recent work such as **DeepIGeoS** (Wang et al., *IEEE TPAMI*, 2018)
combines learned models with user interaction for medical images specifically.

The design adopted here follows that tradition: **one loose bounding box** drawn by the
clinician, from which a supervised pixel classifier produces the contour.

### 1.3 Why a deep learning method (Task 2)

**U-Net** (Ronneberger et al., *MICCAI*, 2015) remains the backbone of nearly all medical
segmentation work, and **nnU-Net** (Isensee et al., *Nature Methods*, 2021) demonstrated
that a well-configured U-Net still matches or beats most architectural innovations across
many datasets — a result that sets a deliberately high bar for any "novel" architecture and
is the reason the baseline in this report is a *tuned* U-Net rather than a default one.

Two lines of improvement are relevant to this design:

- **Attention.** *Attention U-Net* (Oktay et al., 2018) introduced gates on the skip
  connections that suppress irrelevant regions. Transformer-based models —
  **TransUNet** (Chen et al., 2021) and **Swin-UNet** (Cao et al., *ECCV workshops*, 2022) —
  add global self-attention, addressing the fact that a convolution can only ever integrate
  information inside its receptive field. The BRISC dataset paper itself proposes a
  Swin-based hybrid (Fateh et al., 2025).
- **Efficiency.** **Structured channel pruning** (Li et al., *ICLR*, 2017) removes whole
  filters and therefore genuinely reduces FLOPs, unlike unstructured magnitude pruning which
  leaves tensor shapes intact. Liu et al. (*ICLR*, 2019) showed the pruned *architecture*
  often matters more than the inherited weights, which motivates the fine-tuning strategy
  used here.

The proposed model combines both: multi-head self-attention for global context, a novel
dual attention gate on the skips, and structured pruning to cut the running cost.

### 1.4 Why this matters for sustainability and access

Model efficiency is not only an engineering concern. A segmentation tool that runs on a
laptop CPU can be deployed in a district hospital or a low-resource setting; one that
requires a datacentre GPU cannot. Reducing FLOPs therefore serves both the net-zero-carbon
objective and an equity-of-access objective simultaneously — a point returned to in §5.

---

## 2. Methods

### 2.1 Dataset

BRISC 2025 (Fateh et al., 2025) provides 6,000 T1-weighted MRI slices with
radiologist-reviewed pixel-level masks across glioma, meningioma, pituitary tumour and
no-tumour classes, in axial, coronal and sagittal planes. The **segmentation** subset used
here contains 3,933 training and 860 test image/mask pairs; all contain a tumour.

Images arrive at several native resolutions (mostly 512×512, with a long tail down to
~200×200 and up to 1427×1275). **All images and masks are resampled to a common 512×512
before any processing, and every metric in this report is computed in that 512×512 space**,
so Task 1 and Task 2 results are directly comparable.

### 2.2 Task 1 — semi-automated segmentation

**Interaction model.** The clinician drags one loose rectangle around the lesion. For batch
evaluation the box is simulated from the ground-truth bounding box with **each of the four
sides pushed outwards by a random 2–20%** of the lesion size, so the box is never tight and
never repeats. Only the four box coordinates reach the algorithm — the mask itself is never
seen at test time. This follows standard practice for evaluating interactive segmentation.

**Pipeline.**

```
user box → CLAHE + z-score normalisation over the head region
        → crop the box + 35% padding, resample to 128×128 (scale normalisation)
        → 41 features per pixel
        → Random Forest → tumour probability map
        → Gaussian smooth → threshold → closing → fill holes → largest component
        → paste back into the full 512×512 frame
```

**Feature set (41 per pixel).**

| group | count | contents |
|---|---|---|
| Appearance | 32 | intensity; bilateral-filtered intensity; Gaussian σ ∈ {1,2,4,8,16}; Sobel gradient magnitude at each σ; Laplacian-of-Gaussian at each σ; Hessian eigenvalue pair at σ ∈ {1,2,4,8}; local mean and standard deviation over 5/11/21 windows; 9×9 median |
| Geometry | 4 | elliptical distance from box centre (1.0 = box edge); normalised row and column offsets; inside/outside-box flag |
| Context | 5 | difference to box-core mean; difference to outside-box mean; z-score against the core; core mean; outside mean |

The **geometry and context groups are the only route the user's input takes into the model**
— this is what makes the method semi-automated rather than automatic. The trained forest
confirms their importance: `box_elliptical_dist` alone carries **26.6%** of the total
feature importance, with `inside_box`, `box_dist_x` and `box_dist_y` also in the top six.

**Classifier settings.** `RandomForestClassifier`, 120 trees, `min_samples_leaf=20`,
`max_features='sqrt'`, `class_weight='balanced'`. Trained on 400 randomly chosen training
scans × 700 sampled pixels = **280,000 rows**, sampled 50% lesion / 25% near the boundary /
25% far background — deliberately weighted to the boundary, where the errors actually occur.
Training takes **36 seconds** on CPU. **No TensorFlow, no deep learning, no Haar cascade and
no template matching are used**, as the brief requires.

**Unsupervised comparison methods**, all receiving the identical user box: Otsu thresholding
(Otsu, 1979), k-means clustering on intensity (k=3), classical seeded region growing, and
morphological Chan–Vese active contours (Marquez-Neila et al., *IEEE TPAMI*, 2014).

### 2.3 Task 2 — the proposed MHA-ResUNet

Fully automatic: the network takes the whole slice, no user input.

**Architecture.** Encoder/decoder with four levels at widths 32/64/128/256 and a 512-wide
bottleneck, and three departures from a plain U-Net:

1. **Residual blocks** throughout — two Conv-BN-activation layers with the input added back
   via a 1×1 projection shortcut, keeping gradients flowing to the deep layers.
2. **Multi-head self-attention at the bottleneck** (4 heads, fixed `key_dim=64`). At the
   12×12 bottleneck grid every position attends to every other. This addresses a specific
   failure mode on brain MRI: a convolution cannot compare a candidate lesion against the
   corresponding structure on the *other* side of the midline, because the two are outside
   each other's receptive field.
3. **A dual attention gate on every skip connection** — the novel component. A standard
   Attention U-Net gate produces one spatial map saying *where* to attend, but treats all
   256 channels of a skip as equally relevant. This gate adds a squeeze-and-excitation
   branch so the skip is reweighted by *where* **and** *which features*:

   ```
   skip_out = skip × spatial_attention × channel_attention
   ```

   Both branches are driven by the decoder's gating signal, so deeper semantic features
   decide what the shallow features are permitted to pass.

`key_dim` is deliberately held constant rather than set to `channels/heads`: if it scaled
with channel count, pruning the bottleneck would reshape the attention weights in a way that
cannot be sliced, and the pruned model could not inherit them.

**Evidence the architecture earns its parameters.** Trained on an identical 1,200-scan
subset for 8 epochs under identical settings:

| model | validation Dice @ 8 epochs |
|---|---|
| MHA-ResUNet | **0.719** |
| Plain U-Net | 0.580 |

**Training settings.** 192×192 input, batch 16, Adam at 5×10⁻⁴, combined BCE + Dice loss
(equal weighting), 40 epochs with early stopping on validation Dice and LR halving on
plateau, 80/20 train/validation split. Augmentation is left-right flip, ±15° rotation,
0.9–1.1 zoom and brightness/contrast jitter. *Vertical flips are deliberately excluded* —
they would place the skull base above the vertex, producing anatomy that cannot occur.

**BatchNorm momentum is set to 0.9 rather than the Keras default 0.99.** At the default the
running statistics update too slowly for these batch sizes: training Dice looked healthy
while validation oscillated between 0.42 and 0.70. This single change, with batch 16 and the
lower learning rate, took the model from a ~0.55 plateau to 0.87.

### 2.4 Task 2 — the PSO-tuned baseline

So the comparison is against a fairly-tuned baseline rather than default settings, Particle
Swarm Optimisation searches four hyperparameters of a conventional U-Net:

| hyperparameter | range | PSO's choice |
|---|---|---|
| base filters | 8 – 48 | **24** |
| dropout | 0.0 – 0.4 | **0.215** |
| activation | relu / elu | **elu** |
| learning rate | 1×10⁻⁴ – 5×10⁻³ (log scale) | **1.85×10⁻³** |

Standard velocity update, inertia 0.7, cognitive and social coefficients 1.5:

```
v ← w·v + c₁·r₁·(personal_best − x) + c₂·r₂·(global_best − x)
x ← x + v
```

6 particles × 4 iterations = **24 fitness evaluations**, each a 6-epoch run on 1,000 scans
at 128×128. Total search time **30.4 minutes**.

### 2.5 Task 2 — structural pruning

**Why structured, not magnitude, pruning.** Setting small weights to zero leaves tensor
shapes unchanged, so on ordinary hardware the FLOP count is identical. The brief asks for
pruning that *reduces FLOPs*, which requires whole channels to disappear and the network to
be physically rebuilt narrower.

**Method.** L1-norm channel ranking (Li et al., 2017): each output channel is scored by the
L1 norm of the filter producing it, the weakest are dropped, the identical architecture is
rebuilt at the smaller widths, and every kernel is sliced on **both** its input and output
channel axes so the narrow network inherits the trained weights.

**Correctness test.** Pruning at ratio 0.0 rebuilds the network and reproduces the original
validation Dice exactly (**0.8683 → 0.8683**). This is retained in `pruning_ablation.py` as a
standing regression test of the slicing logic.

**Why pruning is done in rounds.** One-shot pruning collapses the network. Dice immediately
after weight transfer, no fine-tuning:

| one-shot ratio | after transfer | after BN recalibration |
|---|---|---|
| 0.00 | 0.8683 | 0.8672 |
| 0.05 | 0.7951 | 0.8563 |
| 0.10 | 0.6344 | 0.8365 |
| 0.15 | 0.3935 | 0.7677 |
| 0.20 | 0.1035 | 0.5808 |
| 0.30 | 0.0215 | 0.3620 |
| 0.40 | 0.0167 | 0.2147 |

Two distinct effects are visible. **Stale BatchNorm statistics** — when 40% of a
convolution's *input* channels vanish, each surviving output channel loses ~40% of the
contributions that formed it, so its pre-activation distribution shifts and the inherited
`moving_mean`/`moving_var` no longer describe it; cascaded through nine blocks the output
collapses. This is cheap to fix and recalibration recovers most of it at low ratios. **Joint
function damage** — what recalibration cannot fix: at 40% the recalibrated score is still
only 0.215, because the surviving filters were co-adapted with those removed.

**Final method:** 4 rounds, each removing ~12% of remaining channels, with a BatchNorm
recalibration pass (100 forward-only batches, no gradient updates) and a 3-epoch recovery
after every cut, then a 12-epoch final fine-tune.

| round | widths | after transfer | after BN recal | after 3 epochs |
|---|---|---|---|---|
| 1 | 28, 56, 112, 224, 452 | 0.4683 | **0.8065** | 0.8676 |
| 2 | 24, 48, 100, 196, 396 | 0.4972 | **0.8155** | 0.8657 |
| 3 | 20, 44, 88, 172, 348 | 0.2369 | **0.7628** | 0.8651 |
| 4 | 16, 40, 76, 152, 308 | 0.4734 | **0.7301** | 0.8628 |

Final fine-tune → **validation Dice 0.8712**, which is *above* the unpruned model's 0.8697.
Total pruning cost 33.2 minutes.

### 2.6 Hardware

All work was carried out on a single **Apple M1 Max** (10 CPU cores, 32 GB unified memory),
with the integrated GPU used for Task 2 via `tensorflow-metal`. **No datacentre GPU was used
at any stage.** Task 1 runs entirely on CPU.

---

## 3. Results

### 3.1 Task 1 — 50 unseen test scans

| method | Dice | median | IoU | Accuracy | HD95 (px) | Sens. | Prec. | s/image |
|---|---|---|---|---|---|---|---|---|
| **Proposed (RF + box)** | **0.8740** | **0.9284** | **0.8016** | **0.9930** | **11.37** | 0.9143 | 0.8635 | 0.063 |
| Chan–Vese active contour | 0.7813 | 0.8886 | 0.6910 | 0.9921 | 16.12 | 0.7382 | 0.8724 | 0.131 |
| Otsu threshold | 0.7744 | 0.8438 | 0.6714 | 0.9868 | 19.02 | 0.8333 | 0.7825 | 0.005 |
| k-means (k=3) | 0.7230 | 0.7235 | 0.5991 | 0.9801 | 23.72 | 0.8345 | 0.7300 | 0.029 |
| Seeded region growing | 0.6594 | 0.7515 | 0.5423 | 0.9840 | 21.65 | 0.6288 | 0.8436 | 0.005 |

- **43 / 50 scans score Dice ≥ 0.85**; 45 / 50 score ≥ 0.70.
- The proposed method exceeds the best unsupervised comparator (morphological Chan–Vese,
  2014) by **+9.3 Dice points** and reduces HD95 by **29%**. All methods received the
  identical simulated user box, so the comparison is fair.

**By tumour type:** meningioma 0.9430 (n=18), pituitary 0.8965 (n=19), glioma 0.7454 (n=13).

### 3.2 Task 1 — model compression

| trees | min_samples_leaf | nodes | size (MB) | Dice | latency (s) |
|---|---|---|---|---|---|
| 300 | 4 | 4,977,358 | 92.48 | 0.8748 | 0.085 |
| 200 | 8 | 2,259,548 | 46.47 | 0.8748 | 0.077 |
| 150 | 12 | 1,304,474 | 28.86 | 0.8742 | 0.067 |
| **120** | **20** | **731,774** | **17.85** | **0.8740** | **0.062** |
| 100 | 30 | 449,548 | 11.79 | 0.8733 | 0.064 |
| 60 | 40 | 216,692 | 5.97 | 0.8727 | 0.049 |

The shipped configuration is **81% smaller and 28% faster than the largest forest for a 0.09%
Dice loss**.

### 3.3 Task 2 — 200 unseen test scans

| model | Dice | median | IoU | Accuracy | HD95 (px) | GFLOPs | params | size | latency |
|---|---|---|---|---|---|---|---|---|---|
| **MHA-ResUNet pruned** | **0.8699** | 0.9286 | **0.7949** | **0.9962** | 14.77 | **5.08** | **3.28 M** | **12.86 MB** | **41.2 ms** |
| MHA-ResUNet | 0.8692 | **0.9312** | 0.7934 | 0.9960 | **14.49** | 14.93 | 8.84 M | 34.06 MB | 46.8 ms |
| U-Net (PSO-tuned) | 0.8352 | 0.9172 | 0.7549 | 0.9952 | 25.98 | 7.62 | 4.37 M | 16.84 MB | 53.4 ms |

- Dice ≥ 0.85 on 153/200 (unpruned), **151/200** (pruned), 137/200 (U-Net). Dice ≥ 0.70 on
  **184/200** for the pruned model — the best of the three.
- **The pruned model matches the full model** (0.8699 vs 0.8692) while using **66% fewer
  FLOPs**. Removing 40% of the channels and fine-tuning acted as a regulariser, not a cost.
- **Against the PSO-tuned U-Net the pruned model wins on every accuracy metric while using
  33% fewer FLOPs, 25% fewer parameters and 24% less disk.**
- HD95 is where the architecture separates itself most: **14.77 px vs 25.98 px, a 43%
  reduction** in boundary error against the tuned baseline.

**By tumour type (pruned):** meningioma 0.9350 (n=78), pituitary 0.8666 (n=64), glioma
0.7860 (n=58).

**Sensitivity to two complete failures.** Two glioma scans score Dice 0.000 (see §5.5).
Excluding them: pruned 0.8787, unpruned 0.8780, U-Net 0.8436. All headline figures above are
the **full 200-scan** numbers, as the brief specifies; the exclusion is reported only to show
those two cases account for under one Dice point and affect the baseline equally.

### 3.4 Task 1 vs Task 2 — direct comparison

Because both are scored at 512×512 on the same masks, the two chapters compare directly:

| | Task 1 (semi-auto) | Task 2 (automatic, pruned) |
|---|---|---|
| Dice | 0.8740 (50 scans) | 0.8699 (200 scans) |
| HD95 | **11.37 px** | 14.77 px |
| Glioma Dice | 0.7454 | **0.7860** |
| Model size | 17.85 MB | 12.86 MB |
| Latency | 63 ms (CPU) | 41 ms (GPU) |
| Training cost | 36 seconds | 49.8 min + 33.2 min pruning |
| User input | one box per lesion | none |
| Hardware to train | laptop CPU | laptop GPU |

The semi-automated method achieves a **better boundary** (HD95 11.4 vs 14.8) because the
user's box eliminates the localisation problem entirely; the deep model is **better on
gliomas** (0.786 vs 0.745) because it learns appearance cues a hand-designed feature set does
not capture. They fail in different ways, which is the basis of the deployment
recommendation in §5.6.

---

## 4. Whole-life cost analysis

### 4.1 Task 1 — semi-automated system

| phase | estimate | basis |
|---|---|---|
| Dataset preparation | Low. 400 annotated scans used; BRISC already provides masks. Re-training on local data would need ~400 annotated slices, roughly 2–3 days of radiologist time. | measured — training used 400 scans |
| Software development | ~2 person-weeks for the pipeline, features, baselines and evaluation harness. | estimate |
| Training compute | **36 seconds**, laptop CPU. Negligible. | measured |
| Hardware | Any laptop. No GPU. <1 GB RAM. | measured |
| File size | **17.85 MB** model. Trivial to distribute or version. | measured |
| User training | Low — the interaction is "drag a box". Realistically 15–30 minutes including when to reject a result. | estimate |
| Recalibration | Retraining is 36 seconds, so recalibrating for a new scanner or protocol is a same-day task, not a project. | measured |
| Long-term support | Low. Dependencies are numpy/scipy/scikit-learn/OpenCV — stable, widely maintained, no framework version risk. | judgement |

The dominant lifetime cost is **clinician interaction time**, not compute: one box per lesion
per study, forever. At ~5 seconds per box against 10–20 minutes of manual outlining, the
trade is strongly favourable.

### 4.2 Task 2 — deep learning system

| phase | estimate | basis |
|---|---|---|
| Dataset preparation | Higher. 3,933 annotated training scans. Acquiring that from scratch is a multi-month annotation project. | measured |
| Software development | ~3–4 person-weeks including the PSO search and the pruning machinery. | estimate |
| Training compute | 49.8 min (proposed) + 30.4 min (PSO) + 52.8 min (baseline) + 33.2 min (pruning) = **166.2 min** on one laptop GPU. | measured |
| Hardware | GPU strongly preferred for training; **inference runs on CPU**. | measured |
| File size | **12.86 MB** pruned (34.06 MB unpruned). | measured |
| User training | Minimal — there is no interaction. Training is needed in *interpreting* output, especially recognising silent failures (§5.5). | judgement |
| Recalibration | Expensive. A new scanner or protocol needs re-training and re-pruning: ~83 min plus validation. Not a same-day task. | measured |
| Long-term support | Higher. TensorFlow 2.13 pins NumPy < 2; the dependency set is brittle and will need periodic migration work. | measured — encountered during development |

**The asymmetry is the key finding.** Task 1 costs almost nothing to build, retrain or
maintain but charges the clinician a small interaction cost on every case forever. Task 2
costs far more up front and to maintain but is free at the point of use. Over a long
deployment with high case volume, Task 2 amortises better; for a small centre, or one
without stable annotation resources, Task 1 is the more rational investment.

---

## 5. Discussion

### 5.1 Comparison with a state-of-the-art automated tool

The natural comparator is **nnU-Net** (Isensee et al., *Nature Methods*, 2021), the
self-configuring U-Net framework that remains a default benchmark in medical segmentation.

| | proposed (pruned) | nnU-Net (typical 2D config) |
|---|---|---|
| Parameters | **3.28 M** | ~30 M **[verify]** |
| FLOPs / slice | **5.08 G** | substantially higher **[verify]** |
| Model size | **12.86 MB** | ~100+ MB **[verify]** |
| Training schedule | 40 epochs, ~50 min, laptop GPU | 1000 epochs, typically days on a datacentre GPU **[verify]** |
| Training data used | 3,933 slices | same data available |
| Self-configuring | no | yes |

**[verify]** — these nnU-Net figures are from general familiarity with the framework's
defaults and **must be checked against the paper before submission**. The qualitative point
stands regardless: nnU-Net's design philosophy is to spend heavily on training compute and
extensive augmentation to maximise accuracy, whereas this design targets accuracy *per
FLOP*.

**Benefits in training data required.** The proposed model reaches 0.87 Dice from 3,933
slices in 40 epochs. Task 1 reaches 0.874 from **400** scans in 36 seconds. For a centre that
cannot produce thousands of expert annotations, the semi-automated route is not merely
cheaper — it is the only feasible one.

### 5.2 Strengths and weaknesses vs recent literature

**Strengths.**
- 3.28 M parameters is small for a medical segmentation network — TransUNet is ~105 M and
  Swin-UNet ~27 M **[verify]** — while remaining competitive on this dataset.
- Two thirds of the arithmetic removed at zero accuracy cost, with the pruning recipe
  documented and reproducible.
- Inference is CPU-feasible, so deployment does not require GPU procurement.
- Both models are far smaller than the transformer-based methods that currently dominate
  leaderboards.

**Weaknesses.**
- Trained and evaluated at 192×192, not native resolution. HD95 in particular would likely
  improve at full resolution, though this was **not tested and is not claimed**.
- Single dataset, single modality (T1). No cross-dataset generalisation was measured — the
  standard weakness of most published work in this area, but a weakness nonetheless.
- 2D slice-wise, ignoring 3D context that a volumetric method would exploit.
- No uncertainty estimate, which is what makes the silent failures in §5.5 dangerous.

### 5.3 Carbon footprint

**These are calculated estimates from stated assumptions, not meter measurements.**

Assumptions: Apple M1 Max package power ~40 W under sustained GPU load, ~20 W for CPU-only
work; UK grid carbon intensity ~0.20 kg CO₂e/kWh **[verify against current DEFRA/NG ESO
figures]**.

| activity | time | energy | CO₂e |
|---|---|---|---|
| Task 1 training | 36 s @ 20 W | 0.0002 kWh | ~0.04 g |
| Task 2 full pipeline (all four stages) | 166.2 min @ 40 W | ~0.111 kWh | **~22 g** |
| Task 2 inference, per slice | 41.2 ms @ 40 W | 4.6×10⁻⁷ kWh | ~0.09 mg |

For context, the entire Task 2 development pipeline is roughly the carbon cost of **boiling a
kettle twice**. This is the direct consequence of two deliberate choices: training on
integrated laptop hardware rather than a datacentre GPU, and pruning the model before
deployment.

**Optimisation techniques applied to reduce footprint.**
1. **Structured channel pruning** — 66% FLOPs reduction at zero accuracy cost. Every
   inference the system ever performs costs a third of what it otherwise would.
2. **Random Forest compression (Task 1)** — 81% size reduction for 0.09% Dice loss.
3. **Reduced working resolution** — 192×192 rather than 512×512 cuts training cost ~7× versus
   full resolution.
4. **Early stopping and LR scheduling** — training halts when validation Dice plateaus rather
   than running a fixed long schedule.
5. **No GPU required at inference** — avoids provisioning dedicated accelerator hardware per
   deployment site.

**Honest caveat on latency.** Measured inference improved only 46.8 → 41.2 ms (12%) despite
the 66% FLOPs cut. At batch size 1 on a GPU the run is dominated by kernel-launch overhead
rather than arithmetic. The FLOPs reduction is real and translates into lower energy per
inference and genuine speedup on CPU and edge hardware, but **a 3× speedup claim would not be
supported by these measurements.**

### 5.4 The glioma weakness

Both methods are markedly worse on gliomas (Task 1: 0.745; Task 2: 0.786) than on
meningiomas (0.943 / 0.935) and pituitary tumours (0.897 / 0.867). This is not a defect of
either implementation but a property of the disease: meningiomas and pituitary adenomas are
compact and well circumscribed, whereas gliomas infiltrate surrounding tissue and have
margins that are genuinely ambiguous even between expert human raters. Any reported Dice on
glioma should be read against that ceiling.

### 5.5 Silent failure — the most important limitation

Two glioma scans score **Dice 0.000** with the automatic model: `test_00011` (a 314 px
lesion, 0.12% of the frame) and `test_00045` (a large, visually obvious 12,613 px tumour).

Re-running these at 192 / 256 / 320 / 384 px input — loading the same weights into the
architecture rebuilt at each size — leaves both at 0.000 throughout, while four control
gliomas that score 0.963 at the trained size fall to 0.613 at 384 px:

| input | test_00011 | test_00045 | 4 control gliomas |
|---|---|---|---|
| 192 (trained) | 0.000 | 0.000 | 0.963 |
| 256 | 0.000 | 0.000 | 0.948 |
| 320 | 0.000 | 0.000 | 0.837 |
| 384 | 0.095 | 0.000 | 0.613 |

**Resolution is therefore not the cause.** The failure mode is *confident mislocalisation*:
on `test_00011` the model emits 5,092 pixels at probability 0.999 in an entirely different
structure. A resolution-limited failure would produce weak, scattered, low-confidence output.
On `test_00045` the pruned model returns an empty mask (max probability 0.118). *Caveat: the
model was trained at 192 px, so inference at larger sizes is out of distribution; this cannot
rule out that a model trained at 384 px would succeed.*

The clinical significance is severe: **the automatic model fails silently.** It does not flag
uncertainty; it returns an empty or confidently wrong mask. A clinician reviewing a batch of
contours could plausibly miss this.

### 5.6 Deployment recommendation

The two tasks are complementary rather than competing, and the failure analysis dictates the
architecture of a real deployment:

1. Run the **Task 2 automatic model** first on every slice — zero interaction cost.
2. Flag for review any case where the prediction is empty, has low maximum probability, or is
   an anatomical outlier.
3. On flagged cases, fall back to the **Task 1 semi-automated method** — the clinician drags
   one box, which supplies exactly the location information the automatic model got wrong.

This gives the throughput of automation with a human-verifiable safety net, and it is
directly supported by the evidence in §5.5 rather than being an assertion.

### 5.7 Societal and cultural impact

**Access and equity.** Brain tumour outcomes depend heavily on timely, accurate imaging
assessment, and radiologist availability varies enormously between and within countries. A
12.86 MB model that runs on a laptop CPU can be deployed where a GPU cluster cannot. The
efficiency work in this report is therefore not only an environmental argument but an
equity-of-access one: reducing hardware requirements widens who can use the tool.

**Consistency and the reduction of inter-observer variation.** Manual outlining varies
between observers and within the same observer over time. Because response assessment
compares volumes across months, that variation directly affects treatment decisions. A
deterministic algorithm produces the same contour for the same input every time — which is
valuable *even where it is slightly less accurate than the best human*, because
reproducibility is what longitudinal comparison actually requires.

**Trust, transparency and the clinician's role.** The semi-automated design keeps the
clinician in control, which matters both for professional accountability and for adoption:
tools that remove agency are resisted. The Random Forest is also far more inspectable than a
neural network — feature importances are directly readable, and the dominant feature is the
user's own box, which is straightforward to explain to a clinician. This interpretability is
a genuine advantage of the Task 1 approach that its lower glioma accuracy does not erase.

**Dataset bias — a real limitation.** BRISC 2025's demographic and geographic composition is
not documented in the material available here. A model trained on scans from a limited set of
scanners and populations may underperform on populations not represented. This is a
well-documented failure mode in medical AI, and **it cannot be ruled out for these models
because the relevant metadata was not available.** Before any clinical use, performance
should be validated on the local patient population rather than assumed to transfer.

**Automation bias.** The silent failures in §5.5 interact dangerously with the known human
tendency to over-trust automated output. A contour that looks plausible but is in the wrong
place is more dangerous than an obviously absent one. This argues for interface design that
makes uncertainty visible, and for the fallback workflow in §5.6.

---

## 6. Conclusions

1. A **semi-automated Random Forest method using a single bounding box** achieves **0.8740
   Dice** on 50 unseen BRISC scans, beating the best unsupervised comparator by 9.3 Dice
   points and reducing HD95 by 29%, while training in 36 seconds on a laptop CPU.
2. The proposed **MHA-ResUNet** achieves **0.8692 Dice** on 200 unseen scans, beating a
   PSO-tuned conventional U-Net by 3.4 Dice points and cutting HD95 by 44%.
3. **Iterative structural pruning removed 66% of FLOPs and 63% of parameters at zero
   accuracy cost** (0.8699 vs 0.8692 Dice), with the pruned model beating the tuned U-Net on
   every accuracy metric while using 33% fewer FLOPs.
4. The pruning recipe matters: one-shot pruning at the same ratio collapses the network to
   0.017 Dice. **BatchNorm recalibration and per-round recovery are what make aggressive
   pruning viable**, and both are documented with measurements.
5. The automatic model **fails silently on ~1% of cases**, which is the strongest argument
   for retaining the semi-automated method as a fallback rather than treating the two
   approaches as alternatives.

---

## 7. Reproducibility

```bash
# Task 1 — no TensorFlow, as required
python3 -m venv mvi-env && ./mvi-env/bin/pip install -r task1/requirements.txt
cd task1
../mvi-env/bin/python train_model.py        # ~40 s
../mvi-env/bin/python evaluate.py           # ~1 min  → outputs/
../mvi-env/bin/python model_compression.py  # optional sweep
../mvi-env/bin/python demo_interactive.py   # draw your own box

# Task 2 — TensorFlow/Keras
python3 -m venv tf-env && ./tf-env/bin/pip install -r task2/requirements.txt
cd task2
../tf-env/bin/python run_all.py             # full pipeline, ~2.5 h
../tf-env/bin/python pruning_ablation.py    # the ablation + correctness test
```

All results use fixed random seeds (`EVAL_SEED = 2`) and reproduce exactly. Full per-image
metrics are in `task1/outputs/results_per_image.csv` and
`task2/outputs/results_per_image.csv`.

---

## 8. References

Cao, H. et al. (2022) 'Swin-Unet: Unet-like pure transformer for medical image segmentation',
*ECCV Workshops*.

Chen, J. et al. (2021) 'TransUNet: Transformers make strong encoders for medical image
segmentation', *arXiv:2102.04306*.

Fateh, A. et al. (2025) 'BRISC: Annotated dataset for brain tumor segmentation and
classification with Swin-HAFNet', *arXiv:2506.14318*.

Grady, L. (2006) 'Random walks for image segmentation', *IEEE Transactions on Pattern
Analysis and Machine Intelligence*, 28(11).

Isensee, F. et al. (2021) 'nnU-Net: a self-configuring method for deep learning-based
biomedical image segmentation', *Nature Methods*, 18(2).

Li, H. et al. (2017) 'Pruning filters for efficient ConvNets', *ICLR*.

Liu, Z. et al. (2019) 'Rethinking the value of network pruning', *ICLR*.

Marquez-Neila, P., Baumela, L. and Alvarez, L. (2014) 'A morphological approach to
curvature-based evolution of curves and surfaces', *IEEE Transactions on Pattern Analysis and
Machine Intelligence*, 36(1).

Oktay, O. et al. (2018) 'Attention U-Net: learning where to look for the pancreas',
*arXiv:1804.03999*.

Otsu, N. (1979) 'A threshold selection method from gray-level histograms', *IEEE Transactions
on Systems, Man, and Cybernetics*, 9(1).

Ronneberger, O., Fischer, P. and Brox, T. (2015) 'U-Net: convolutional networks for
biomedical image segmentation', *MICCAI*.

Rother, C., Kolmogorov, V. and Blake, A. (2004) 'GrabCut: interactive foreground extraction
using iterated graph cuts', *ACM Transactions on Graphics*, 23(3).

Wang, G. et al. (2018) 'DeepIGeoS: a deep interactive geodesic framework for medical image
segmentation', *IEEE Transactions on Pattern Analysis and Machine Intelligence*, 41(7).

Yushkevich, P.A. et al. (2006) 'User-guided 3D active contour segmentation of anatomical
structures', *NeuroImage*, 31(3).
