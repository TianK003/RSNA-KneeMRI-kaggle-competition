# Brainstorming and backlog

Ideas **not yet tried**, ranked, plus the open questions that would change our plan if
answered. Once something is measured it moves to [experiments.md](experiments.md) with a
verdict; this file is only for the forward-looking half.

Keep the ranking honest: order by *expected value*, not by how interesting the idea is.
Two of the top three items are unglamorous plumbing, and that is the correct ordering.

---

## Ranked backlog

### 1. Preprocessing cache kernel — unblocks everything
**Why first:** it is not a score idea, it is the thing that makes every other experiment
affordable. Current estimate is 6–8 h per fold, so five folds ≈ five babysat sessions, and
any experiment costs a day. The bottleneck is DICOM decode, not the ViT: 6 slices/slot × 3
channels × ~5 slots ≈ **90 file reads per study, repeated every epoch**.

**Plan:** a separate CPU kernel decodes, orders, resizes and writes uint8 arrays once
(community reports ~15.9 GB, ~1 h for all 4,407 studies); the training kernel mounts it.
Turns 90 DICOM reads per study per epoch into one array read.

**Do first:** an actual throughput benchmark. The 6–8 h figure is extrapolated from 8
study-passes, which is not a measurement.

### 2. Site-grouped folds — correctness, not score
The largest known gap in our validation. Report-text grouping is done, but **site identity
leaks through the pixels** and we do nothing about it. Community reports ~0.05 inflation
from random K-fold, one team measured **+0.136**.

**Plan:** derive a site proxy from DICOM headers (`Manufacturer`,
`ManufacturerModelName`, `InstitutionName`, `MagneticFieldStrength`, maybe pixel spacing)
and group on it *together with* report text. Expect our OOF numbers to **drop** — that is
the point. Until this exists, no OOF number we produce is fully trustworthy.

### 3. Rank-blend RadImageNet ResNet-50 with DINOv2
**Not implemented.** `models/radimagenet_r50/ResNet50.pt` is downloaded but nothing loads
it; the current rank mean ensembles 5 folds of the *same* architecture, which reduces
variance but adds no architectural diversity.

RadImageNet is *supervised* on ~1.35M radiologic images, so it has a different inductive
bias and different failure modes — the standard precondition for a blend paying off. Every
strong public notebook does this.

⚠️ **RadImageNet is CC-BY-NC-SA-4.0** (non-commercial, share-alike). Check the winner
licence clause before it becomes load-bearing in a final submission. DINOv2 is Apache-2.0
and carries no such question.

### 4. Improve the weak labels for the three weakest findings
The teacher caps the student, so this raises the ceiling rather than closing a gap. Weakest
teacher labels: **Synovitis 0.788, Lateral OA 0.804, Fracture 0.825** (vs ACL 0.989).

**Plan:** per-language coverage analysis to find where vocabulary/reading is thin, then
targeted prompting with an **open-weights** multilingual model.
⚠️ Do not send report text to a hosted LLM API — see the rules section in
[../CLAUDE.md](../CLAUDE.md). Run locally or inside a Kaggle notebook.

### 5. Laterality normalisation
Mirror right knees so "medial" always means the same side. **Not cosmetic:** `Medial OA` and
`Lateral OA` are different labels, so without this the model must learn each finding twice.
Currently **not implemented** — the DICOM laterality tag is reportedly unreliable in this
corpus, so it needs a derived heuristic (and a way to check it).

### 6. DINOv3 ViT-S/16 as a third ensemble member
Cheap once #3's blending machinery exists. Public notebooks report it helping.

### 7. Higher input resolution
Findings are focal — a meniscal tear is small, and at 224px with /14 patches a tear may
occupy a patch or two. Public notebooks mention "meniscus resolution" explicitly. Costly, so
gate it behind #1.

### 8. Per-label specialist heads
For the labels that stay near chance after everything else. Justified by the metric: every
label costs the same, so one label at 0.5 forfeits ~0.029 of the score at M=0.85, and a rare
finding is exactly where a model most easily lands at chance.

### 9. Attention-pooling variants / multi-slot cross-attention
Let slots attend to each other rather than being concatenated independently. Speculative.

### 10. Auxiliary report-reconstruction task
Predict report-derived attributes as an auxiliary head, dropped at inference. Might
regularise the encoder toward clinically meaningful features. Speculative, and the
supervision is the same weak labels, so may add nothing.

---

## Ideas considered and rejected without testing

- **Text branch at inference.** Impossible: `test.csv` has no `Report` column. Reports are
  training-time only. This is worth restating because it is the single most tempting wrong
  turn given the competition is advertised as multimodal.
- **Horizontal-flip augmentation.** Swaps medial/lateral, which are different labels. See
  [traps.md](traps.md).
- **Calibrating probabilities / tuning thresholds.** Macro ROC-AUC reads only rank order.
  Zero effect on score.
- **`pos_weight` in the BCE loss.** Inflates all predictions; with rank-only scoring there
  is nothing to gain and a reported overprediction failure mode to lose.
- **Averaging probabilities across folds.** Rank-average instead — probability averaging
  lets the most confident model dominate.
- **Chasing the 0.95 public LB by forking the community ensemble.** Its own author warns it
  is "likely overfit to the public leaderboard". Doing this buys a number, not a model, and
  the private LB is where it gets settled.

---

## Open questions that would change the plan

| Question | Why it matters | How to answer |
|---|---|---|
| What is the **real** per-study throughput with `num_workers=2`? | Decides whether 5 folds is even feasible, and how much #1 buys | Timed benchmark kernel |
| Is the ≤9 h runtime limit and internet-off rule accurate? | Community-sourced, never read from the rules page | Read the overview/rules pages in a browser |
| How is the **Efficiency Prize** actually scored? | It is a separate prize we are eligible for; runtime may be worth optimising deliberately | The efficiency-prize evaluation page (JS-rendered, needs a browser) |
| Does the winner-licence clause tolerate CC-BY-NC-SA weights? | Gates #3 for a *final* submission | Read the rules page |
| How many folds are actually worth training? | 5 folds may be a poor use of compute vs. 3 folds + a second backbone | Compare 3-fold vs 5-fold OOF once #1 lands |
| Does site-grouping change our conclusions, or just lower all numbers? | If it reorders which ideas look good, earlier comparisons need redoing | #2 |

---

## Strategy notes

**We are not trying to win the public leaderboard.** The top ten span 0.006 and are one
shared, admittedly LB-overfit ensemble. The plan that survives a private shakeup is a
pipeline we can *validate*: honest grouped folds, OOF over all 4,407 studies, and changes
accepted only when they clear the noise floor.

**The teacher is at 0.8934 and cannot be used at test time.** That number is the signal
ceiling for distillation from these labels. Two ways past it: better labels (#4), or the
images carrying information the reports omit — which is plausible, since gold is
image-derived and reports agree with gold only ~82%.

**Compute is the binding constraint, not ideas.** We have ~55 days and Kaggle session
limits. That is why #1 and #2 outrank every modelling idea: one makes experiments cheap, the
other makes their results mean something.
