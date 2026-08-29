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
| Where does the remaining 0.081 to the public top actually come from? | We are at 0.871 on one fold with the same public label tables the leaders use, so it is unlikely to be target quality; ensembling accounts for maybe 0.01–0.02 | Read the top notebooks' configs in a browser; run 5 folds and measure what the rank-mean is worth |
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

**Compute is the binding constraint, not ideas.** We have ~55 days and Kaggle session
limits. That is why #1 and #2 outrank every modelling idea: one makes experiments cheap, the
other makes their results mean something.
