# Progress explanation script — EE001-3.5-3-MVI (5-minute version)

~720 spoken words. Bracketed lines are stage directions, not things to say.

---

## 1. Opening — 20 seconds

> "My use case is brain tumour segmentation on MRI, for radiotherapy planning and
> follow-up volumetry, using BRISC 2025. At the moment a clinician outlines the tumour
> by hand on every slice — 10 to 20 minutes per study, and not reproducible, which
> matters because response assessment compares volumes over time. I built two systems:
> Task 1 is semi-automated, the clinician gives one box; Task 2 is fully automatic.
> Both are scored at 512×512 on the same masks, so they compare directly."

---

## 2. Task 1 — 75 seconds

> "In Task 1 the clinician drags one loose rectangle round the lesion. For evaluation I
> simulate that box from the ground truth, but I push each side outward by a random 2
> to 20 percent so it's never tight and never repeats — and only the four coordinates
> reach the algorithm, never the mask.
>
> The pipeline is CLAHE and z-score normalisation, crop the box with padding, resample
> to 128×128, then 41 features per pixel into a Random Forest, and clean the
> probability map up with smoothing, thresholding, closing and largest-component
> selection. Of those 41 features, 32 are appearance — multi-scale Gaussian, Sobel,
> Laplacian, Hessian, local statistics — and 9 are geometry and context relative to the
> box. Those 9 are the only route the user's input takes into the model, and the
> trained forest confirms they matter: distance from the box centre alone carries 26.6
> percent of the total feature importance.
>
> On 50 unseen scans it gets **0.874 Dice**, with 43 of the 50 above 0.85. I compared
> against four unsupervised methods all receiving the identical box — Otsu, k-means,
> region growing, and morphological Chan–Vese, which is from 2014. Chan–Vese is the
> strongest at 0.781, so the proposed method beats it by 9.3 Dice points and cuts
> boundary error by 29 percent. Training takes 36 seconds on a laptop CPU, no
> TensorFlow and no deep learning, as the brief requires, and a compression study let
> me ship a model 81 percent smaller for a 0.09 percent Dice loss."

---

## 3. Task 2 — 110 seconds

> "Task 2 is an MHA-ResUNet. Three departures from a plain U-Net: residual blocks
> throughout; multi-head self-attention at the bottleneck; and the novel component, a
> dual attention gate on every skip connection.
>
> The self-attention addresses a specific failure mode on brain MRI — a convolution
> can't compare a candidate lesion against the corresponding structure across the
> midline, because the two are outside each other's receptive field. The dual gate
> addresses a limitation of Attention U-Net: a standard gate says *where* to attend but
> treats all channels equally, so I added a squeeze-and-excitation branch and the skip
> is reweighted by where *and* which features. Head-to-head against a plain U-Net under
> identical settings, 0.719 Dice against 0.580.
>
> For a fair baseline I used Particle Swarm Optimisation to tune a conventional U-Net —
> filters, dropout, activation and learning rate, 24 fitness evaluations over 30
> minutes — so the baseline I'm beating is a tuned one.
>
> Then structured channel pruning. Structured specifically, because zeroing small
> weights leaves the tensor shapes unchanged and the FLOP count identical — to actually
> cut FLOPs the channels have to disappear and the network be rebuilt narrower. One-shot
> pruning collapses it, for two reasons: stale BatchNorm statistics, which a
> recalibration pass fixes cheaply, and joint function damage, which it can't. So I
> prune in four rounds with recalibration and recovery after each cut.
>
> On 200 unseen scans the pruned model gets **0.870 Dice — matching the unpruned model
> on 66 percent fewer FLOPs**, and beating the PSO-tuned U-Net at 0.835 on every
> accuracy metric while using 33 percent fewer FLOPs. The clearest separation is
> boundary error: HD95 14.8 pixels against 26.0, a 43 percent reduction."

---

## 4. Comparison and deployment — 50 seconds

> "The two are level on Dice, 0.874 against 0.870, but they fail differently. Task 1
> has the better boundary, 11.4 pixels against 14.8, because the box removes the
> localisation problem. Task 2 is better on gliomas, because it learns cues a
> hand-designed feature set doesn't capture.
>
> The important finding is a failure mode: two glioma scans score exactly zero with the
> automatic model. I ruled out resolution by re-running at four input sizes — both stay
> at zero while control cases degrade normally. On one, the model emits 5,000 pixels at
> probability 0.999 in the wrong structure. So it fails *silently* — no uncertainty
> signal, just a confidently wrong mask.
>
> That dictates the deployment: run the automatic model first at zero interaction cost,
> flag empty or low-confidence outputs, and fall back to Task 1 on those — the
> clinician's box supplies exactly the location information the automatic model got
> wrong. Automation throughput with a human-verifiable safety net."

---

## 5. Cost, carbon, close — 40 seconds

> "On cost: Task 1 is 36 seconds of CPU training and an 18 MB model. Task 2's entire
> pipeline — proposed model, PSO search, baseline and pruning — is 166 minutes on one
> laptop GPU, no datacentre hardware at any stage. On carbon, and these are calculated
> estimates from stated assumptions rather than meter readings, that whole pipeline is
> about 22 grams of CO₂e — roughly boiling a kettle twice — and the pruning means every
> inference the system ever runs costs a third of what it otherwise would.
>
> One honest caveat: measured latency only improved 12 percent despite the 66 percent
> FLOPs cut, because at batch size 1 on a GPU the runtime is kernel-launch overhead,
> not arithmetic. The saving is real on CPU and edge hardware, but I wouldn't claim a
> 3× speedup.
>
> So both tasks clear 85 percent Dice on the specified test sets, and every number is
> regenerable from the code."

---

## 6. Q&A — short answers

**Isn't simulating the box from ground truth cheating?**
> "Only the four coordinates reach the algorithm, never the mask, and each side is
> jittered outward 2 to 20 percent so it's loose and never repeats. That's the standard
> protocol for evaluating interactive segmentation."

**Why is glioma worse in both systems?**
> "It's the disease, not the implementation. Meningiomas and pituitary adenomas are
> compact; gliomas infiltrate, and their margins are ambiguous even between expert human
> raters."

**Why does the pruned model beat the unpruned one?**
> "It's within noise — 0.8699 against 0.8692. My claim is that pruning cost nothing, not
> that it helped. Fewer parameters plus a 12-epoch fine-tune is a regularisation effect."

**Why a Random Forest rather than a CNN in Task 1?**
> "The brief separates the tasks — Task 1 is classical image processing, no deep
> learning, no TensorFlow in that folder. It's also right on the merits: 36 seconds of
> CPU training, and the feature importances let me show the user's box is actually
> driving the prediction."

**What's the weakest part?**
> "The silent failure in Task 2. I've mitigated it with the fallback design but not
> solved it — the real fix is a proper confidence estimate, and that's what I'd do next."

**Why 50 test images for Task 1 and 200 for Task 2?**
> "50 is what the brief specifies for Task 1. Network inference is cheap, so I used 200
> for Task 2 to tighten the per-class breakdown. Both scored at 512×512 on the same
> masks."
