# Brainstorming and backlog

**The ranked backlog now lives in [proposals.md](proposals.md)** (cards P-00 … P-20, each with
hypothesis, evidence, measure, noise floor and cost, built from [research.md](research.md)).
This file keeps only the open questions and the strategy notes. Once a card is measured it
moves to [experiments.md](experiments.md) with a verdict.

Keep the ranking honest: order by *expected value*, not by how interesting the idea is.
Two of the top three items are unglamorous plumbing, and that is the correct ordering.

---

## Where the backlog went (2026-08-28)

The ten items that were ranked here were re-derived from evidence by an 18-agent research
workflow and became cards in [proposals.md](proposals.md). Two corrections from that work:

- **RadImageNet's licence was stated here as CC-BY-NC-SA-4.0. That could not be verified** —
  the weights carry no stated licence (code MIT, paper CC BY 4.0, data "by request"). Treat as
  restrictive until radimagenet.com's T&C is read in a browser.
- The "DINOv2 small instead of base" idea was inverted: the pipeline already uses **small**,
  and three independent sources find ViT-S ≈ ViT-B for medical transfer at this data size.

The "ideas rejected without testing" list moved to the end of proposals.md and grew to 33
entries.

---

## Open questions that would change the plan

| Question | Why it matters | How to answer |
|---|---|---|
| ~~What is the **real** per-study throughput with `num_workers=2`?~~ | **ANSWERED** 2026-08-28: 0.18 s/study from the cache, 0.99 decoding; 0.9 h per fold-0 4-epoch arm | experiments.md, kernel v8 |
| ~~Is the fork-inherited RNG actually repeating augmentation on Kaggle?~~ | **ANSWERED** 2026-08-29: no. `check_worker_rng()` on Kaggle shows draws already vary without the fix — PyTorch seeds numpy/random per worker itself. traps 6e corrected; the `v04d` result is unconfounded | Answered in-kernel; the check now runs at every startup |
| ~~Where does the remaining 0.081 to the public top actually come from?~~ | **ANSWERED 2026-08-30** — read a 0.936 notebook cell by cell (research.md §2.7.1): its DINOv2 branch alone is ≈ 0.899, at parity with our 0.896; the other ≈ 0.036 is three more model families rank-fused on top (16-channel ViT, RadImageNet heads, CoAtNet-2@384 at 0.924 alone), and only +0.001 is LB tuning. Within one family we measured heads +0.019, folds +0.000 on top | Answered; the programme is P-23 |
| **Mount the public checkpoints as blend members, or build our own families?** | Every member of the 0.936 notebook is a public Kaggle dataset (dreaddevelopment `raptor-knee-*`, tonylica repro assets), so mounting them into `INFER_MEMBERS` is legal and probably ~0.93 in one submission; but it makes us a fork of the shared ensemble expected to shake up on private, the members cannot be validated on our folds (their training studies overlap our held-out set), and the datasets' licence fields are unread | Tian's decision; read the dataset pages' licence fields in a browser; P-23 |
| Licence of `dreaddevelopment/raptor-knee-*` and `tonylica/rsna-knee-bend-dinov3-0917-repro-assets` | Gates mounting them; the notebook that uses them is Apache-2.0 but the checkpoints' terms are set on the dataset pages | Kaggle dataset pages (browser) |
| **Compute path for the remaining 7 weeks: Kaggle-only (30 h/wk, ≈ 6 h left this week) vs + free Colab vs 2×T4?** | Tian will not pay for GPUs. Verified 2026-08-30: a Google One storage plan carries **no** Colab compute units (Colab Pro is a separate $9.99/100-unit product; the free US-student Colab Pro is closed). Free Colab ≈ another fluctuating T4 pool with ≤ 12 h sessions and idle disconnects — workable only because the pipeline resumes per epoch. Training needs only the 21 GB cache; inference stays on Kaggle | Tian's decision; then P-24 (runner + 2×T4), no GPU needed to build |
| Is the ≤9 h runtime limit and internet-off rule accurate? | Community-sourced, never read from the rules page | Read the overview/rules pages in a browser |
| How is the **Efficiency Prize** actually scored? | It is a separate prize we are eligible for; runtime may be worth optimising deliberately | The efficiency-prize evaluation page (JS-rendered, needs a browser) |
| Does the winner-licence clause tolerate CC-BY-NC-SA weights? | Gates #3 for a *final* submission | Read the rules page |
| How many folds are actually worth training? | 5 folds may be a poor use of compute vs. 3 folds + a second backbone | P-13, once the cache lands |
| Does site-grouping change our conclusions, or just lower all numbers? | If it reorders which ideas look good, earlier comparisons need redoing | #2 |

---

## Strategy notes

**We are not trying to win the public leaderboard.** The top ten span 0.006 and are one
shared, admittedly LB-overfit ensemble. The plan that survives a private shakeup is a
pipeline we can *validate*: honest grouped folds, OOF over all 4,407 studies, and changes
accepted only when they clear the noise floor.

**The teacher is at 0.8948 (blend; 0.8934 for the diagnostic rank blend) and cannot be used at test time.** That number is the signal
ceiling for distillation from these labels. Two ways past it: better labels (#4), or the
images carrying information the reports omit — which is plausible, since gold is
image-derived and reports agree with gold only ~82%.

**The remaining gap is families, not recipe (2026-08-30).** A 0.936 notebook read in full is our
DINOv2 branch (≈ 0.899 vs our 0.896) plus three more families rank-fused on top. Its own
counter-example matters as much: three backbones on the *same* input blended to +0.001, so
diversity must come from the input representation and pretraining regime (triplets vs 16-channel
stack vs 384 px / 64 slices vs frozen radiology features), not from swapping the backbone name.
P-23 is that programme; each candidate is one fold-0 arm judged on ρ and blend gain, never on LB.

**Compute is now the binding constraint in a new way (2026-08-30).** Two P-23 candidates cost 6.7 h and
left ≈ 6 h for the week. Every remaining candidate that changes the *input* (more slices, 336–384 px, a
non-linear stack stem) costs 2–5× a normal arm. Without paid GPUs the options are 2×T4 sessions
(charged once) and free Colab fed from the 21 GB cache — P-24. Our epoch time is loader-bound, not
GPU-bound (0.19 s/study with 36 forwards, 0.11 with 6), so any new box needs local NVMe + workers.

**Compute is the binding constraint, not ideas.** We have ~55 days and Kaggle session
limits. That is why #1 and #2 outrank every modelling idea: one makes experiments cheap, the
other makes their results mean something.
