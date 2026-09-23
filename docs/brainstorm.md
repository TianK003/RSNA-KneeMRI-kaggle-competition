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
| **Which two submissions go to the final private scoring?** #13 (anchor + our arm at β 0.10) and #15 (anchor only) tie at 0.942 publicly | The anchor's inner weights were tuned on the public split (its author warns of overfit); #13 carries one vote that was never tuned there, #15 is the pure stack. Selecting both hedges exactly that difference; a later better candidate would displace one. **#16 (the anchor's own flat-0.60 outer map, arm not run) read 0.940 on 2026-09-23** — −0.002 is the public price of the LB-tuned per-label map, so #16 is the validated flat-weights pick for the second slot if that map is judged the bigger private risk (experiments.md 2026-09-23) | Tian's call before 2026-10-22; Kaggle lets two be selected |
| ~~**Does the 0.942 anchor reproduce 0.942 from our account?**~~ | **ANSWERED 2026-09-22: yes** — #15 (anchor only) = 0.942; #13 (β 0.10) = 0.942, #12 (β 0.20) = 0.939, #14 = 0.941 | experiments.md 2026-09-22 P-27 read-out |
| ~~**Do 16 epochs over-train the soft targets?**~~ | **ANSWERED 2026-09-22: yes** — P-29 `v09p` OOF peaks at epoch 8 (0.8731), epoch 15 0.8607, 11/12 labels down | experiments.md 2026-09-22 P-29 |
| ~~What is the **real** per-study throughput with `num_workers=2`?~~ | **ANSWERED** 2026-08-28: 0.18 s/study from the cache, 0.99 decoding; 0.9 h per fold-0 4-epoch arm | experiments.md, kernel v8 |
| ~~Is the fork-inherited RNG actually repeating augmentation on Kaggle?~~ | **ANSWERED** 2026-08-29: no. `check_worker_rng()` on Kaggle shows draws already vary without the fix — PyTorch seeds numpy/random per worker itself. traps 6e corrected; the `v04d` result is unconfounded | Answered in-kernel; the check now runs at every startup |
| ~~Where does the remaining 0.081 to the public top actually come from?~~ | **ANSWERED 2026-08-30** — read a 0.936 notebook cell by cell (research.md §2.7.1): its DINOv2 branch alone is ≈ 0.899, at parity with our 0.896; the other ≈ 0.036 is three more model families rank-fused on top (16-channel ViT, RadImageNet heads, CoAtNet-2@384 at 0.924 alone), and only +0.001 is LB tuning. Within one family we measured heads +0.019, folds +0.000 on top | Answered; the programme is P-23 |
| ~~**Mount the public checkpoints as blend members, or build our own families?**~~ | **DECIDED 2026-09-21 (Tian): both** — fork the public 0.942 graph verbatim and add our members as one more rank vote (P-27), and keep training our own under the production regime (P-28). Reason: the 0.942 notebook trains nothing, its members were trained on A6000/H100 boxes across ~6 families, and 0.913 → 0.942 by our own training alone is months, not nights | Done; P-27 / P-28 |
| ~~Licence of the public checkpoint datasets~~ | **READ 2026-09-21 via `kaggle datasets metadata`**: `dreaddevelopment/raptor-knee-*` and `rsna-knee-labels`, `mattiaangeli/knee-mri-fold-weights` / `rsna-knee-coat-resgated-ep10-top3` / `rsna-knee-coatnet-d4-depthzone-swa3-b2`, `pilkwang/rsna-knee-weights` = **CC0-1.0**; `marwanmath/resnet-50-radimagenet-marwan`, `antoinegg1/rsna-knee-e11-diverse-heads-v20`, `antoinegg1/rsna-knee-e9-radimagenet-heads-v15` = **CC-BY-NC-SA-4.0**; `prvsiyan/rsna-knee-v52-radimagenet-heads-20260812` = "other". Open only for the *final* submission: keep or drop the RadImageNet stage | Winner-licence clause of the rules (browser) |
| ~~**Compute path for the remaining 7 weeks: Kaggle-only vs + free Colab vs 2×T4?**~~ | **DECIDED 2026-08-30 (afternoon): RunPod**, paid per hour, once the c02 cache exists (Tian). The runner is built (`scripts/runpod_bootstrap.sh`, P-24); the two hybrid arms `v09h` / `v10c` are its first jobs. Kaggle keeps smokes, the DINOv2 `v08w` arm and all inference. 2×T4 on Kaggle stays an open cheap option | Done; first pod run pending |
| Does the derived cache leaving Kaggle for a rented box sit inside the rules? | It is competition data in derived form; training on one's own compute is ordinary Kaggle practice and the rules allow downloading the data for the competition. Must never be published or shared outside the team | Tian's call (made: proceed); read the Rules text when in a browser |
| Kaggle per-kernel output cap (~20 GB assumed), per-Dataset size cap, max attached sources per notebook | c02 shards are ~9 GB each; the c02 train kernel mounts 6 cache sources + 3 label sets + 3 weight datasets + 1 model | Kaggle docs in a browser |
| ~~How many CPU kernels run concurrently?~~ | **ANSWERED 2026-08-30: four** — `rsna-knee-cache2-a..d` ran at once | verified by `kaggle kernels status` |
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
