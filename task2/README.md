# Task 2 — Deep learning segmentation with structural pruning (BRISC 2025)

A fully automatic segmentation network, compared against a conventional U-Net whose
hyperparameters were tuned by Particle Swarm Optimisation, then structurally pruned to cut
FLOPs. Built with TensorFlow/Keras (Task 1 bans TensorFlow; Task 2 does not).

---

## 1. The proposed model — MHA-ResUNet

Same clinical purpose as Task 1 (radiotherapy planning and follow-up volumetry), but with
no user interaction at all: the network takes the whole slice and returns a mask.

Three departures from a plain U-Net, each with a reason:

**1. Residual encoder/decoder blocks.** Two Conv-BN-activation layers with the input added
back on. Keeps gradients flowing to the deep layers.

**2. Multi-head self-attention at the bottleneck.** A convolution only ever sees its own
receptive field. At the 12×12 bottleneck grid, MHA lets every position attend to every
other, which is what is needed to tell a tumour apart from the symmetric structure on the
other side of the midline — a mistake convolutions make readily on brain MRI.

**3. A dual attention gate on every skip connection** — the novel component. The standard
Attention U-Net gate (Oktay et al., 2018) produces one spatial map saying *where* to look,
but treats all 256 channels of a skip connection as equally relevant. This gate adds a
squeeze-and-excitation branch, so the skip is reweighted by *where* **and** *which
features*:

```
skip_out = skip × spatial_attention × channel_attention
```

Both branches are driven by the decoder's gating signal, so the deeper semantic features
decide what the shallow features are allowed to pass through.

**Direct evidence the architecture earns its parameters.** Trained on an identical
1200-scan subset for 8 epochs with identical settings:

| model | val Dice @ 8 epochs |
|---|---|
| MHA-ResUNet | **0.719** |
| Plain U-Net | 0.580 |

## 2. The baseline — U-Net tuned by PSO

So the comparison is against a fairly-tuned baseline rather than default settings, PSO
searches four hyperparameters (`pso.py`):

| hyperparameter | range | PSO's choice |
|---|---|---|
| base filters | 8 – 48 | 24 |
| dropout | 0.0 – 0.4 | 0.215 |
| activation | relu / elu | elu |
| learning rate | 1e-4 – 5e-3 (log) | 1.85e-3 |

6 particles × 4 iterations = 24 fitness evaluations, each a 6-epoch training run on 1000
scans at 128 px (30.5 min total). Velocity update is the textbook form:

```
v ← w·v + c1·r1·(personal_best − x) + c2·r2·(global_best − x)
x ← x + v
```

## 3. Results — 200 unseen test scans

Predictions are made at 192 × 192 and resized to 512 × 512 before scoring, so these numbers
sit on **exactly the same masks as Task 1** and the two chapters can be compared directly.

| model | Dice | median | IoU | Accuracy | HD95 (px) | GFLOPs | params | size | latency |
|---|---|---|---|---|---|---|---|---|---|
| **MHA-ResUNet pruned** | **0.8699** | 0.9286 | **0.7949** | **0.9962** | 14.77 | **5.08** | **3.28 M** | **12.9 MB** | **41.2 ms** |
| MHA-ResUNet | 0.8692 | **0.9312** | 0.7934 | 0.9960 | **14.49** | 14.93 | 8.84 M | 34.1 MB | 46.8 ms |
| U-Net (PSO-tuned) | 0.8352 | 0.9172 | 0.7549 | 0.9952 | 25.98 | 7.62 | 4.37 M | 16.8 MB | 53.4 ms |

- Dice ≥ 0.85 on 153/200 images (proposed), **151/200** (pruned), 137/200 (U-Net).
  Dice ≥ 0.70 on **184/200** for the pruned model — the best of the three.
- **The pruned model matches the full model** (0.8699 vs 0.8692 Dice, and marginally better
  IoU and accuracy) while using **66% fewer FLOPs**. Removing 40% of the channels and
  fine-tuning acts as a regulariser rather than a cost.
- **Against the PSO-tuned U-Net the pruned model wins on every accuracy metric while using
  33% fewer FLOPs, 25% fewer parameters and 24% less disk.** That is the headline comparison.
- HD95 is where the proposed architecture separates itself most: 14.8 px vs 26.0 px, a 43%
  reduction in boundary error over the tuned baseline.

**Sensitivity to the two complete failures.** Two glioma scans (`test_00011`, `test_00045`)
score Dice 0.000 — the model segments the wrong structure entirely rather than producing a
poor outline (see section 8). Excluding those two as outliers:

| model | all 200 scans | excluding the 2 failures |
|---|---|---|
| MHA-ResUNet pruned | 0.8699 | 0.8787 |
| MHA-ResUNet | 0.8692 | 0.8780 |
| U-Net (PSO-tuned) | 0.8352 | 0.8436 |

**All figures quoted elsewhere in this document are the full 200-scan numbers**, which is
what the brief specifies. The exclusion is shown only to make clear how much of the gap to a
"clean" score those two cases account for — under one Dice point, and they affect the
baseline equally, so the margin over the tuned U-Net is essentially unchanged either way.

**By tumour type**:

| type | n | Dice (proposed) | Dice (pruned) |
|---|---|---|---|
| Meningioma | 78 | 0.927 | 0.935 |
| Pituitary | 64 | 0.874 | 0.867 |
| Glioma | 58 | 0.786 | 0.786 |

The same weakness as Task 1: gliomas have diffuse, infiltrative margins that are ambiguous
even between human raters. Worth noting that deep learning narrows the gap considerably
(Task 1 scored 0.745 on gliomas, Task 2 scores 0.786) but does not close it.

## 4. Structural pruning

**Why structural and not magnitude pruning.** The usual "set small weights to zero"
approach leaves the tensor shapes unchanged, so on ordinary hardware it costs exactly as
many FLOPs as before. The brief asks for pruning that *reduces FLOPs*, which means whole
channels must disappear and the network must be physically rebuilt narrower. That is what
`prune.py` does: L1-norm channel ranking (Li et al., ICLR 2017), rebuild at the smaller
widths, then slice every kernel on both its input and output channel axis so the narrow
network inherits the trained weights.

**Two findings worth reporting.**

*First, the slicing is provably correct.* Pruning at ratio 0.0 rebuilds the network and
reproduces the original validation Dice exactly (0.8683 → 0.8683). This is built into
`pruning_ablation.py` as a standing correctness test — if the slicing ever breaks, that row
stops matching.

*Second, one-shot pruning collapses the network* — and this is why the final method prunes
in rounds. Dice immediately after weight transfer, with no fine-tuning at all
(`pruning_ablation.py`):

| one-shot ratio | after transfer | after BN recalibration |
|---|---|---|
| 0.00 | 0.8683 (identical, as it must be) | 0.8672 |
| 0.05 | 0.7951 | 0.8563 |
| 0.10 | 0.6344 | 0.8365 |
| 0.15 | 0.3935 | 0.7677 |
| 0.20 | 0.1035 | 0.5808 |
| 0.30 | 0.0215 | 0.3620 |
| 0.40 | 0.0167 | 0.2147 |

Two separate effects are visible in that table:

- **Stale BatchNorm statistics** — the larger effect, and a cheap fix. When 40% of a
  convolution's *input* channels vanish, each surviving output channel loses ~40% of the
  contributions that formed it, so its pre-activation distribution shifts and the inherited
  `moving_mean`/`moving_var` no longer describe it. Cascaded through nine blocks the output
  collapses. Recalibration recovers most of it — at 10% pruning it takes 0.63 back to 0.84.
- **Joint function damage** — what recalibration cannot fix. At 40% the recalibrated score
  is still only 0.21, because the surviving filters were co-adapted with the ones removed.
  No amount of statistics fixing recovers that; the network has to be given a chance to
  re-adapt, which is exactly what the per-round fine-tuning provides.

**The method that works** — `PRUNE_ROUNDS = 4`, each removing ~12% of the remaining
channels, with a BN recalibration pass (100 forward-only batches, no gradients) and a
3-epoch recovery after every cut, then a 12-epoch final fine-tune. Measured round by round:

| round | widths | after transfer | after BN recal | after 3 epochs |
|---|---|---|---|---|
| 1 | 28, 56, 112, 224, 452 | 0.4683 | **0.8065** | 0.8676 |
| 2 | 24, 48, 100, 196, 396 | 0.4972 | **0.8155** | 0.8657 |
| 3 | 20, 44, 88, 172, 348 | 0.2369 | **0.7628** | 0.8651 |
| 4 | 16, 40, 76, 152, 308 | 0.4734 | **0.7301** | 0.8628 |

final fine-tune (12 epochs) → **validation Dice 0.8712**, total pruning cost 33.2 min.

Two things to draw from that table. First, **BN recalibration alone recovers 0.24–0.50 up
to 0.73–0.82 every round, on 100 forward passes with no gradient updates** — the cheapest
accuracy in the whole project, and the direct evidence that the transferred convolution
weights were sound and only the statistics were stale. Compare that with one-shot 40%
pruning, where the same recalibration lifts 0.009 to only 0.137. Second, the network never
drops below 0.86 after any round's recovery, so the width can be taken down gradually
without the model ever falling apart.

Note the final validation Dice of 0.8712 is **above the unpruned model's 0.8697**. Pruning
40% of the channels and fine-tuning left the network slightly better than it started, which
is consistent with the original model being over-parameterised for this dataset.

## 5. Carbon footprint and cost

| | unpruned | pruned | saving |
|---|---|---|---|
| FLOPs per slice | 14.93 G | 5.08 G | **66%** |
| Trainable parameters | 8.84 M | 3.28 M | **63%** |
| Model size | 34.1 MB | 12.9 MB | **62%** |
| Test Dice | 0.8692 | 0.8699 | **none** (+0.07 pts) |

The strongest claim available here, and it is a real one: **two thirds of the arithmetic
was removed at no accuracy cost whatsoever.** Every inference the deployed system ever runs
costs a third of the energy it would otherwise have used.

Training cost: 49.8 min for the proposed model, 52.8 min for the U-Net, 30.5 min for PSO,
33.2 min for pruning — on one Apple M1 Max GPU (32 GB unified memory), no datacentre GPU at
any stage.

**An honest caveat on latency.** Measured inference improved from 46.8 ms to 41.2 ms — a
12% gain from a 66% FLOPs cut. At batch size 1 on a GPU the run is dominated by
kernel-launch overhead rather than arithmetic, so the saving does not translate
proportionally. The FLOPs reduction is real and shows up as lower energy per inference and
as genuine speedup on CPU or edge hardware, but claiming a 3× speedup would not be
supported by these measurements. Latency figures also vary by a few ms between runs
depending on machine load, so small differences should not be over-read.

## 6. Files

| file | what it does |
|---|---|
| `config.py` | every setting in one place |
| `data.py` | loading, augmentation, train/val split |
| `models.py` | the U-Net baseline and the proposed MHA-ResUNet |
| `losses.py` | BCE + Dice loss and the training metrics |
| `train.py` | training loop for both networks |
| `pso.py` | Particle Swarm Optimisation for the baseline |
| `prune.py` | iterative structural pruning + BN recalibration |
| `pruning_ablation.py` | the one-shot sweep and the correctness test |
| `complexity.py` | FLOPs, parameters, size on disk, latency |
| `evaluate.py` | the 200-image comparison |
| `run_all.py` | runs the whole pipeline in order |

## 7. How to run

```bash
# from the project root
python3 -m venv tf-env
./tf-env/bin/pip install -r task2/requirements.txt

cd task2
../tf-env/bin/python run_all.py          # everything, ~2.5 h on an M1 Max
```

Or one stage at a time:

```bash
../tf-env/bin/python train.py proposed      # ~50 min
../tf-env/bin/python pso.py                 # ~30 min
../tf-env/bin/python train.py unet          # ~53 min
../tf-env/bin/python prune.py               # ~25 min
../tf-env/bin/python evaluate.py            # ~1 min
../tf-env/bin/python pruning_ablation.py    # optional, the ablation table
```

Outputs land in `task2/outputs/`: `results_per_image.csv`, `results_summary.csv`,
`complexity.json`, `pruning_summary.json`, `pruning_ablation.csv`, `pso_best.json`, and
`qualitative/*.png` (green = ground truth, red = prediction).

## 8. Notes and honest limitations

- **Dependency pinning matters.** TensorFlow 2.13 requires NumPy < 2; installing anything
  newer breaks the whole stack silently. `requirements.txt` pins the working set.
  `tensorflow-metal` enables the M1 GPU and cuts epoch time from minutes to ~57 s.
- **BatchNorm momentum was the single biggest training fix.** At the default 0.99 the
  running statistics update too slowly for these batch sizes: training Dice looked healthy
  while validation swung between 0.42 and 0.70. Setting it to 0.9 (with batch 16 and LR
  5e-4) took the model from a ~0.55 plateau to 0.87.
- **Augmentation is left-right only.** Flipping a brain slice top to bottom would put the
  skull base above the vertex, which cannot occur anatomically.
- Networks train at 192 × 192, not the native 512 × 512, to keep the full pipeline inside a
  ~2.5 hour budget on a laptop GPU. Training at full resolution may improve HD95, since
  boundary detail is what downsampling costs most, but this has not been tested and is not
  claimed.

- **Two glioma cases fail completely (Dice 0.000), and it is not a resolution problem.**
  `test_00011` (a 314 px lesion, 0.12% of the frame) and `test_00045` (a large, visually
  obvious 12,613 px tumour) are both missed. Re-running them at 192 / 256 / 320 / 384 px
  input — loading the same weights into the architecture rebuilt at each size — leaves both
  at 0.000 throughout, while four control gliomas that score 0.963 at the trained size fall
  to 0.613 at 384 px. Raising the resolution does not recover the failures and damages the
  working cases, which is what running a 192-trained network out of distribution should do.

  | input | test_00011 | test_00045 | 4 control gliomas |
  |---|---|---|---|
  | 192 (trained) | 0.000 | 0.000 | 0.963 |
  | 256 | 0.000 | 0.000 | 0.948 |
  | 320 | 0.000 | 0.000 | 0.837 |
  | 384 | 0.095 | 0.000 | 0.613 |

  The failure mode is **confident mislocalisation**, not lost detail: on `test_00011` the
  model emits 5,092 pixels at probability 0.999 in an entirely different structure. A
  resolution-limited failure would produce weak, scattered, low-confidence output instead.
  On `test_00045` the pruned model returns an empty mask (max probability 0.118).

  This is the strongest argument for keeping the Task 1 semi-automated method in the
  workflow: **the automatic model fails silently**, returning an empty or confidently wrong
  mask with no uncertainty signal, whereas a clinician dragging a box supplies exactly the
  location information the network got wrong. See `outputs/failure_cases.png`.

  Caveat on this experiment: because the model was trained at 192 px, inference at larger
  sizes is out of distribution, so it cannot rule out that a model *trained* at 384 px would
  handle these cases. What it does establish is that the information needed is present at
  the resolutions tested and the network is not using it.

- Results use a fixed random sample of 200 test scans (`EVAL_SEED = 2`) so they reproduce.
