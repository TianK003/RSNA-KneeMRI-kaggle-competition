# Plan: one ConvNeXt-T arm with its own recipe, plus single-model research cards

## Context

- **What changed.** On 10-06 evening Tian decided "no ConvNeXt training" (P-69 dropped). He has now reversed that: train
  ONE ConvNeXt-T arm with a recipe built for ConvNeXt from the literature and from what worked for others, then test it on
  the public LB, solo and inside B6.
- **Second ask.** Research what improves a single model (pseudo-labels in depth, pretraining, augmentation, …) and write it
  up as ranked cards in proposals.md. Tian picks later; no code for those beyond the ConvNeXt arm (his answer, 10-06).
- **Tian's decisions (10-06 night):**
  - RunPod up to ≈ $2.5 for the ConvNeXt test (of ≈ $5).
  - 288 px input. The critic recommended 224 for cost; Tian chose 288.
  - A second 288 seed only if the 10-07 reads free the B3 money (B4 − B5 ≤ −0.002: the B0 seeds carry B6, so no week-2
    B3 retrains on RunPod), and only if both arms fit under $2.5.
  - The critic-subagent check was run during planning (section "Critic check"). Approving this plan is Tian's go for one
    pod under that cap.
- **No duplication.** Checked against candidates.md and proposals.md (next section).
- **Honest prior** (research.md 2.7.8; experiments.md 2026-10-06 "Gold-58: what a blend gains by pair type"): a fourth
  family at B6's mean quality adds ≈ +0.0008 LB, and ConvNeXt solos on the forum read 0.929–0.939. The test decides
  whether ConvNeXt is a member and whether B6 + it beats B6, not a jump in score.

## Duplication check (candidates.md, proposals.md)

| Existing item | Relation to this plan |
|---|---|
| T8 / **P-69** ConvNeXt-T (dropped 10-06 evening; card body in `git show 7283a82^:docs/proposals.md`) | **Re-opened** with its own recipe. Its Dropped-directions row gets the new reason: Tian's 10-06 night decision to test one arm |
| B16 / **P-71** the public goodpjw2008 ConvNeXt-T reader (inference only, pending Tian, recommended demote) | Different thing: a 0.929 public model on its own input; no code overlap. If ours becomes a member, B16 would be a second ConvNeXt, so its value falls further |
| **P-72** NFNet seventh family (pending, recommended drop) | Unaffected; a ConvNeXt member makes a fifth family worth even less |
| **P-67** loop, **P-68** `v13ex` ‖ `v13ex2`, the 10-10 sessions A / B / C | Unaffected: the arm trains on RunPod. Its Kaggle smoke takes ≈ 0.1 h of the 2.84 h left before 10-10, beside the planned `v13ex2` smoke |
| **P-74** `channels_last` (pending Tian) | Not needed here: +0.7 % for ConvNeXt in timm's benchmark. P-74 stays Tian's call for the CNNs |
| `v06c` (Aug-30 HF ConvNeXt, key `convnext_tiny`, c01, Dataset `convnext-tiny-224-hf`) | Retired path, left untouched; the new arm uses the key `timm:convnext_tiny` and new weights |
| `scripts/runpod_chain.sh`, the infer placeholder recipe, `src/build_fork.py --member` | Reused; three small changes to the RunPod scripts (an on-pod stopper, ship-per-arm, shorter ship retries) |

## The ConvNeXt-T recipe, parameter by parameter

Arm `v15c` (and its seed twin `v15c2` if the pair runs). The family gets its own optimiser (weights, LR, layer decay,
epochs), as traps 46 requires after the ResNet
under-training of P-57. The data-side settings (weight decay, drop path, augmentation, targets, batch) stay at the B6
members' values, so the read isolates the family. Everything not listed is the `v13e` recipe: c03 input, window-attention
head, 34 training windows, 2 studies × accumulation 2, `train_all`, SWA of the last 3 epochs, gold-58 reported only.

| Parameter | B6 CNN recipe | `v15c` | Evidence (research agents + critic; [R] = primary source read) |
|---|---|---|---|
| Weights | timm ImageNet-1k CNNs | `convnext_tiny.in12k_ft_in1k` | Best Apache-2.0 ConvNeXt-T: 84.2 % vs 82.9 % (`fb_in22k_ft_in1k`) and 82.1 % (in1k) [R, HF cards]. ImageNet-12k is pretraining data our in1k members never saw, the strongest error decorrelator (Gontijo-Lopes 2022). V2 is CC-BY-NC and 1.7–2.4× slower at inference. Its `config.json` mean / std must be ImageNet's (our normalisation is hard-coded) |
| Input size | 224 (B3: 288) | **288** (Tian, 10-06) | timm scores these weights best at 288 (`test_input_size` 288); our best single, B3, runs at 288. Costs ≈ 1.6× the GPU time of 224 and 5–12 min more scoring for B17. The critic preferred 224 for cost |
| Encoder LR | 3e-4 for every layer | 1e-4 at the top | Official ConvNeXt-T COCO / ADE20K configs 1e-4 [R]; the public ConvNeXt-T reader here 1e-4 (0.929 solo). No supervised ConvNeXt AdamW fine-tune above 1e-4 was found. ConvNeXt-T at a shared 1e-3 lost 8 points to ResNet-50 on BreakHis (2406.05612) [R]. `v06c` fit by epoch 3 at 1e-4, so under-training from the LR alone is unlikely |
| Layer decay | none | 0.9 per stage: stem 5.9e-5, stages 6.6 / 7.3 / 8.1 / 9.0e-5, final norm 1e-4 | Official ADE20K Tiny config 0.9 per stage [R]; V2 Tiny 0.9. MRI is a large input shift, and early layers need to adapt (surgical fine-tuning, ICLR 2023). Conflict: our CoAtNet uses 0.75; ISIC 2024 3rd place found layer decay "didn't work". 0.9 is the mildest supported value |
| Epochs | 30 (SWA 27–29) | 20 (SWA 17–19); warm-up 10 % = 2 epochs | ConvNeXt converges fast: `v06c` peaked at epoch 3 of 8; lumbar 2024 1st place trained ConvNeXt 7 epochs vs 14 for EfficientNetV2; the public reader 14. Heavy augmentation lets it run longer than `v06c` |
| Head LR | 1e-3 | 1e-3 | Same head as every member |
| Weight decay | 0.02 | 0.02 (unchanged) | Critic: `param_groups` applies one decay to every 2-D weight, including the head at 1e-3. 0.05 would shrink the head to 0.58 of its size over the run vs 0.72, flattening the per-label window attention that focal labels need. ADE20K's 0.05 had its head at the backbone LR |
| Drop path | 0.1 | 0.1 (unchanged) | Tiny's own ImageNet-1k value (0.0 in its 22k → 1k fine-tune) [R]. The 0.2 in the old card came from ConvNeXt-Small. It would stack with a lower LR, fewer epochs, EMA + SWA and heavy augmentation, and keeps the train loss comparable with the members' |
| EMA / SWA | 0.998 per step / last 3 | same | No BatchNorm, so SWA needs no recalibration [R] |
| Frozen BN | on | off (ConvNeXt has no BatchNorm) | — |
| Augmentation, batch, targets, seed | heavy, no flips; 2 × 2; 0.5 LLM + 0.5 Raptor; 42 | same, `TEACHER_TABLES = ("raptor_teacher",)` flat | One variable: the family. No reason to wait for the 10-07 A3 read (critic) |
| Clip / precision | 1.0 / fp16 + GradScaler | same, plus a per-epoch log line: clip rate, GradScaler scale, skipped steps | Early warning for a too-high LR or fp16 overflow |

Considered and not adopted: ConvNeXt's head-init × 0.001 (our warm-up covers the same risk), label smoothing (targets are
already soft), mixup / cutmix (P-67 tests mixup on B0 first), timm's per-block layer decay (a much steeper scale for the
same number), `channels_last` (+0.7 %), ConvNeXt-Nano / -Small / V2, weight decay 0.05 and drop path 0.2 (above).

## Where it trains, cost, hard stop, critic check

**RunPod, one RTX 4090** ($0.74/h), Tian's cap $2.5. Critic's estimate, from a two-point fit of our measured 4090 epochs (B0
2.05 min, B3 @ 288 4.28 min) to timm's training throughput (ConvNeXt-T ≈ ResNet-50): ≈ 1.0 min/epoch is fixed data
overhead, the rest GPU time. Measured pod overhead: 0.13 h (session E) to ≈ 0.3 h (P-66, with a relaunch).

| Run | Per epoch | Training | Pod total | Cost |
|---|---|---|---|---|
| One arm, 288 px, 20 epochs (the plan) | 4.1–4.5 min | 1.25–1.9 h (central ≈ 1.45 h) | 1.6–2.3 h | $1.2–1.7 (central ≈ $1.4) |
| Two seeds, 288 px (only if the B3 money frees up) | same | twice that | 2.9–4.2 h | $2.1–3.1 (central ≈ $2.3) |
| For comparison: one arm, 224 px | ≈ 3.0 min | 0.85–1.35 h | 1.2–1.7 h | $0.9–1.3 |

**The second seed's rule (Tian: "pair within $2.5").** It is queued behind the first only if B4 − B5 ≤ −0.002 on 10-07.
When the first arm ships, its measured time decides: the second continues only if (time so far) + (the first arm's
training time × 1.05) + 0.1 h ≤ 3.35 h, which is ≈ $2.48. Otherwise the job is stopped right after the first ship.

**Hard stop, enforced on the pod** (critic: today nothing limits the spend; the runtime guard defaults to 40 h, the ship
loop can sleep 10 × 300 s, and laptop watchers die, traps 51 / 53):
1. Before launch: bandwidth ≥ 20 MB/s (traps 45); a Kaggle credential on the pod valid for the whole run (traps 20); the
   pod key read from PID 1's environment, `AUTO_STOP=1` (traps 51); the account balance ≈ $5 per Tian (traps 46: a 402
   deletes the container).
2. A committed `scripts/runpod_stopper.sh` (session E's ad-hoc stopper, traps 51 addendum), started with `setsid nohup` at
   pod creation: it calls `podStop` on the job's final line or at creation + 3.35 h, which caps the bill at ≈ $2.48.
3. Epoch-0 gate: from the ETA line printed at 500 studies (`src/kaggle_pipeline.py:3178`), if elapsed + 20 × ETA × 1.1 +
   0.2 h > 3.0 h, kill the job, `podStop` by hand, and take the Kaggle fallback (loses ≤ $0.35).
4. In sequential mode the chain ships each arm as soon as it finishes (today it ships all arms at the end, so a stop
   during the second arm would lose the first). Ship retries cut to 3 × 120 s through two env variables (defaults
   unchanged).
5. Delete the pod once `kaggle datasets files` confirms the ship(s). Launch after the 10-07 sends and the B4 / B5 reads
   (≈ 01:00 UTC): the submitter shares the Kaggle API rate limit (traps 48), and B4 / B5 decide the second seed.

**Kaggle fallback** (RunPod unavailable or the epoch-0 gate fires): 288 px × 20 epochs is ≈ 6.5–10 h on one T4, so it
needs two sessions with a resume (traps 31); 224 px fits one session (≈ 4–6 h). Either takes T4 time from the 10-10 loop,
so the fallback is Tian's call at that point.

**Critic check (adversarial subagent, run during planning): GO-WITH-CHANGES** for both the run and the recipe.
- **Adopted:** the hard stop above; weight decay 0.02 and drop path 0.1; the under-fit tripwires below; the tightened read
  rules; the C3 fork mounts; ship-per-arm.
- **Overruled by Tian:** the critic recommended 224 px (≈ $1.0, more money left for the B3 retrains); Tian chose 288.
- **Its steelman of the earlier RunPod objection:** a 10-11 Kaggle read would inform the 10-17 retrains equally well. An
  earlier read changes only the 10-10 week's plan, the 10-09 shortlist and C3's date. It does not decide the case: Tian
  set the budget, and the run (≈ $1.4 at 288) saves ≈ 7–10 Kaggle T4-hours (two sessions at 288) in a week already short
  of GPU time.

## Under-fit tripwires (traps 46, pre-registered)

From the members' own logs (`v13e`, `v13r`, `v13b3`: same targets, augmentation and windows):

| When | CNN members | Flag on `v15c` | Action |
|---|---|---|---|
| Epoch 0 | loss 0.553–0.570 | loss > 0.60, NaN, or the GradScaler scale collapsing | stop the pod and investigate |
| Epoch 4 | loss 0.436–0.477, gold 0.876–0.912, `pred_std` 0.267–0.300 | loss > 0.49 **and** `pred_std` < 0.26 | stop the pod (saves ≈ $0.6) and re-plan with Tian (e.g. LR 2e-4) |
| End of run | loss 0.350–0.383, `pred_std` 0.316–0.318 | loss > 0.40, `pred_std` < 0.30, clip rate > 50 % after warm-up, or gold still rising > 0.005 over epochs 14–19 | the run under-fit: a low solo reads 🔁, not ❌ |

## Implementation steps

**Timeline.** 10-06 night: docs, code, unit checks, local smoke, the weights Dataset, the Kaggle smoke. 10-07 ≈ 01:00 UTC,
after the five 10-07 sends and the B4 / B5 reads: the pod (≈ 1.6–2.3 h for one arm). 10-07 day: ship, local backup,
placeholders. 10-08 00:00 UTC: the solo(s) and B17 in the day's submitter plan.

Facts from the code audit (verified in `src/`; timm 1.0.28 locally and on RunPod):
- `load_timm_backbone` (`src/kaggle_pipeline.py:2507`) loads a timm ConvNeXt-T as is: it drops `head.fc`, keeps
  `head.norm`, and reports 0 missing / 0 unexpected keys (180 tensors, 768 features, 27.8 M parameters, no BatchNorm).
- `param_groups` (`:2975`) already decays per stage for timm ConvNeXt names: at 1e-4 and 0.9 it gives the LRs in the
  recipe table, head 1e-3.
- AdamW uses the torch defaults; 10 % linear warm-up, cosine to 0, EMA per optimiser step, SWA of the last 3 epochs, clip
  1.0, fp16 AMP. No `channels_last` and no NaN guard anywhere.
- ConvNeXt V2 weights are CC-BY-NC-4.0 (timm `convnext.py:729`): excluded.
- **Trap:** `models/convnext_tiny` and `/kaggle/input/convnext-tiny-224-hf` hold the old HF-format weights. The new
  `BACKBONES` entry needs its own directory names, or `resolve_dir` finds the HF files and the load exits on missing keys.
- Kaggle's timm version is recorded nowhere; the Kaggle smoke will print it.
- `v15*` arm names are free. Arms live in `SHIPPED_ARMS`; `window_head_test.py` builds its arm table from there.

Steps:
1. **Docs before code** (`/update` conventions, one place per fact):
   - proposals.md: re-open P-69 (status, recipe table, RunPod case, tripwires, read rules) and answer its
     Dropped-directions row; add cards P-75 … P-79; add the amendments to P-62, P-67, P-68 and P-73.
   - research.md: a new section 2.7.9 with the ConvNeXt recipe evidence, the pseudo-label deep dive, the single-model
     levers and the new forum clues.
   - candidates.md: T8 back as a RunPod arm; the `v15c` solo row (section A); the B17 row (section B); the day order.
   - CLAUDE.md state block. `/handoff` at the end of the session.
2. **Weights Dataset.** `huggingface_hub` local-dir download of `timm/convnext_tiny.in12k_ft_in1k` (config.json +
   model.safetensors) into `models/convnext_tiny_in12k/`, and a ship folder `artifacts/ship_convnext_tiny_in12k/` with
   `dataset-metadata.json` in the pattern of `artifacts/ship_timm_b3/` (title 6–50 characters, apache-2.0), slug
   `tiankljucanin/timm-convnext-tiny-in12k`. Check `config.json`'s mean / std. Run `kaggle datasets create -p .` from
   inside the folder (traps 21, 49); confirm with `kaggle datasets files`.
3. **`src/kaggle_pipeline.py`** (template: commit `7d16049`, which added B3):
   - a `BACKBONES` entry `"timm:convnext_tiny"` with both Kaggle layouts and `models/convnext_tiny_in12k`;
   - the arm `v15c` in `SHIPPED_ARMS`: `{**PROD, **V09R_KW, **C03_KW, "backbone": "timm:convnext_tiny", "img_size": 288,
     "lr_backbone": 1e-4, "llrd_decay": 0.9, "drop_path": 0.1, "aug": "heavy", "epochs": 20}` (no `freeze_bn`), and its
     seed twin `v15c2` (the same + `"seed": 43`), both `("raptor_teacher",)` in `DISTILLED_ARMS` (not in `DISTILLED_MIX`
     / `DISTILLED_SILENT_MIX`: flat 0.5). `v15c2` costs nothing unless the pair runs;
   - print `timm.__version__` in the `load_timm_backbone` line;
   - in `train_fold`, a per-epoch line with the mean pre-clip gradient norm, the share of clipped steps, the GradScaler
     scale and the skipped steps (`clip_grad_norm_` already returns the norm; print only, every arm).
   - only if the pair runs: an `INFER_VOTE_GROUPS` option (default `{}`, ≈ 5 lines in the `by_version` blend at `:4071`)
     so `{"v15c2": "v15c"}` rank-means the two seeds into ONE vote. Today `by_version` groups by version name, which would
     give ConvNeXt two votes in B17 (B13: extra votes of a family cost).
4. **`src/window_head_test.py`:** add `("timm:convnext_tiny", 288)` to the backbone loop (`:105-109`) and the drop-path
   loop; add arm checks that `v15c` differs from `v13e` in exactly the recipe keys above, that `v15c2` differs from `v15c`
   only by seed 43, and that both train on `("raptor_teacher",)`. If `INFER_VOTE_GROUPS` is added, a check that it merges
   the two votes and leaves every other blend unchanged.
5. **Mounts:** the weights Dataset into `kaggle/rsna-knee-train`, `-train-b` and `-infer` `kernel-metadata.json`, and
   into `WEIGHTS` in `scripts/runpod_chain.sh:40`. After the run, `rsna-knee-ckpt-v15c` into `rsna-knee-infer` (traps 22).
   For C3, `src/build_fork.py --member v15c=rsna-knee-ckpt-v15c:timm-convnext-tiny-in12k` mounts both in the fork.
6. **RunPod scripts:**
   - add `scripts/runpod_stopper.sh`: it sources the key from `/proc/1/environ`, waits for the job's final line or a
     deadline, and calls `podStop` over GraphQL;
   - in `scripts/runpod_chain.sh`, move the ship block (`:184-200`) into the sequential loop (`:174-176`), so each arm
     ships as soon as it finishes; add `SHIP_TRIES` / `SHIP_WAIT_S` with defaults 10 / 300.
7. **Local checks:** `python src/window_head_test.py` → `UNIT CHECKS PASSED`; a local CPU smoke of the `ARM_ONLY = "v15c"`
   build with `TEACHER_TABLES = ("raptor_teacher",)`; `bash -n` on both scripts.
8. **Kaggle smoke** on `rsna-knee-train` (`ARM_ONLY = "v15c"`, `TEACHER_TABLES = ("raptor_teacher",)`, `FORCE_SMOKE = True`).
   Full training windows if they fit a T4 at 288 (that measures the fallback's memory, as the P-32 smoke did); otherwise
   the default smoke windows, and the pod's first step measures memory. ≈ 0.1 GPU-h of the 2.84 h left before 10-10.
9. **The RunPod run**, after the 10-07 sends and the B4 / B5 reads:
   - create the pod and start the stopper;
   - run `scripts/runpod_chain.sh v15c` (or `v15c v15c2` with `SEQ_ARMS=1` if B4 − B5 ≤ −0.002) with
     `CACHE_PREFIX=rsna-knee-cache3`, `RSNA_TEACHER_TABLES=("raptor_teacher",)`, `AUTO_STOP=1`, `SHIP_TRIES=3`,
     `SHIP_WAIT_S=120`; `/kaggle` on the persistent volume (traps 46);
   - watch the epoch-0 gate, the tripwires and, for a pair, the second-seed rule at the first ship;
   - back up each shipped arm locally (memory: RunPod checkpoint safety); delete the pod after the ship(s) are confirmed.
10. **Placeholders** on `rsna-knee-infer`: the solo(s) (`INFER_MEMBERS = ["v15c"]`, and `["v15c2"]` if the pair ran) and
    B17 (B6 + ConvNeXt as one vote), each green by the candidates.md check.
11. **Submissions:** all on the first free day (10-08 if the run lands on 10-07) through the submitter, which sends a fixed
    list, so the B17 rule is applied when reading. `/update` after each read.

## Read rules (pre-registered)

- **The ConvNeXt solo**, against B6's members' mean 0.9358. With one arm it is `v15c`'s score (one seed, s = 0.003,
  one-seed floor 0.004); with the pair it is the mean of the two solos (two-arm band ± 0.0045):
  - ≥ 0.936: at or above the members' mean.
  - 0.933–0.935: member grade, under the mean.
  - ≤ 0.932: stop spending on ConvNeXt. This is a resource decision, not evidence: 0.932 is within one floor of the mean,
    and `v11a`, a B6 member, reads 0.932. If a tripwire fired, the read is 🔁 and goes back to Tian instead.
- **B17 = B6 + ConvNeXt as one vote** (flat rank-mean of six votes), against B6 0.942 with the section-B bands:
  ✅ ≥ 0.946 / 🔁 0.939–0.945 / ❌ ≤ 0.938.
  - ≥ 0.943: the new pick 1 by house convention (the floors line for a blend is best member + 0.004 = 0.944, so this is
    not evidence), and C3's gate opens: the fork with B17 as the leg at β 0.45.
  - 0.942: pick 1 only if the solo also read ≥ 0.936. A new family under the members' mean is worth ≈ +0.0002 and adds a
    dependency at the hidden rerun.
  - Otherwise B6 stays pick 1. The gold-58 model predicts B17 ≈ B6 + 0.001, so B17 works as a no-harm check.
- **More ConvNeXt training** in the week-2 final members (a second seed, if the pair did not run) only if B17 reads
  ≥ 0.943; the seeds always count as one vote (B13: extra votes of a family already present cost).
- Gold-58 is reported, never a gate (traps 39).

## Pseudo-labels in brief (Tian asked for an overview)

**What they are.** A pseudo-label is a label made by a model instead of a person. A *teacher* model predicts a
probability for each finding on each training study; a *student* model then trains on those predictions, usually mixed with
the original labels. Our original labels are already machine-made (LLMs reading the reports, ≈ 0.89 accurate on gold). Our
production target is 0.5 × LLM label + 0.5 × Raptor's prediction, where Raptor is the public notebook's CoAtNet image model.
So Raptor's predictions are our pseudo-labels.

**What they contributed for us** (experiments.md 2026-09-27 #20, 2026-09-28 P-49, 2026-09-30 silence mix):

| Target on the 58 gold studies | macro AUC vs gold |
|---|---|
| LLM labels alone | 0.8948 |
| Raptor alone | 0.9254 |
| 0.5 / 0.5 mix (production) | 0.9268, better on 12 of 12 labels |

- The gain sits where reports are silent or vague. Raptor alone: Fracture 0.815 → 0.938, Effusion 0.880 → 0.973, Contusion
  0.861 → 0.931. Inside report-silent cells the LLM ranks Fracture positives at 0.17 AUC (worse than chance), Raptor at 0.74.
- The two sources make very different errors (within-class ρ 0.44), which is why the mix beats both.
- On the LB the Raptor target gave +0.009 (#20), our largest target gain.
- Teachers that only knew our own labels added nothing: self-distillation −0.001 (#19); our own CoAtNet out-of-fold
  predictions into CoAtNet students 0.927 = flat (#34–#36); a second public CoAtNet teacher (D4) −0.002 (#43).

**Which kind we do.** Ours is "distillation with noisy labels" (Li et al. 2017): target = λ · noisy label + (1 − λ) · an
independent teacher's probability, with λ = 0.5. It is not classic semi-supervised pseudo-labelling, because every study
already has a label. The LLM labels are themselves a teacher that reads the report, which exists only at training time.
Raptor is not out-of-fold: it trained on these studies, but with other labels, another architecture and another input,
so its errors still differ from ours. P-68 adds a Kaggle-style out-of-fold teacher from another family.

**How it is commonly done** (sources in research.md 2.7.9 once written; [R] = primary source read):
- The teacher predicts rows it never trained on (out-of-fold), or it parrots the labels back (Archit, 735304).
- Soft probabilities, not 0/1: soft beat hard by +0.014 on gold for one forum team over 3 seeds, and rounding hurt
  Archit. Temperature ≤ 1; softening further hurts (Li 2017 [R]).
- Keep the original label in the mix (0.5 / 0.5 is the forum default); replacing it "came out worse than mixing".
- The student is as large as the teacher or larger, and trains with noise (augmentation, drop path). Without the noise,
  Noisy Student lost 0.8 points [R].
- Repeated rounds give diminishing gains (Noisy Student 87.6 → 88.1 → 88.4 [R]); nobody on this forum reports a second.

**Traps:**
1. A teacher predicting its own training rows (leakage) only repeats its labels.
2. Judging the student by agreement with the teacher's labels: our P-38 gained +0.011 on that ruler and −0.001 on the LB
   (traps 39).
3. Same-source teachers add nothing: Li's bootstrap row 50.6 vs 50.7 mAP [R], our P-38 / P-55, and three forum teams.
4. Confirmation bias: the student copies the teacher's mistakes (Arazo 2020 [R]); keeping the label in the mix limits it.
5. Scale mismatch between sources: per-label quantile matching handles it; an early rank-space bug of ours did not.
6. Diversity loss: P-55's seed twins became more alike (ρ 0.946 vs 0.916), so their blend gained less.
7. Rare labels can collapse; per-label matching keeps each label's prevalence.

**Why a student can beat its teacher, and when it cannot.**
- If the label's errors and the teacher's errors are independent, the mix has lower error than either (Li 2017,
  Proposition 1 [R]). Measured there: +2.4 to +4.0 mAP, close to training on clean labels.
- Soft targets also act as an informed label smoothing (Yuan 2020 [R]). A teacher of another architecture passes on
  "different views" of the images (DeiT: a convnet teacher beat a transformer teacher, 84.2 vs 83.1 [R]).
- It fails when the teacher's errors match the label's (same labels, same family), when the student is judged on
  agreement, or when the teacher knows nothing the student cannot learn by itself.
- Every image teacher here also learned from report labels. Its new knowledge is how findings look on studies whose
  report is silent, which is why the gain sits in the silent cells.

**New forum clues** (not yet in research.md): colum2131 (745949) got gold Effusion 0.986 / Synovitis 0.859 from soft
LLM-plus-image labels vs 0.835 / 0.799 from reports; MUTTAHIR (742926) found a fixed 50/50 mix beat tuned weights on gold;
Raymond (743148) on teacher choice: ensemble or single "both works … depends on whether you have disagreeing models".

## Single-model research → cards (Tian picks later)

Tian chose "cards only": these go into proposals.md as 💡 cards with their proxy-arm design, no code. Each is one change to
the P-67 B0 proxy (≈ 3.3 T4-h), read against the floor pair. None has been measured on this task.

| New card | Lever | Evidence | Expected | Cost |
|---|---|---|---|---|
| P-75 | Noisy-Student JFT weights for the EfficientNets (`tf_efficientnet_b0/b3.ns_jft_in1k`, Apache-2.0; +1.0 / +1.75 ImageNet top-1) | SIIM-ISIC 2020 1st place used them for 15 of 18 models; medical transfer of ImageNet gains is mixed (Kornblith vs CheXtransfer); different pretraining data decorrelates errors | 0 to +0.003, plus blend diversity | weights Dataset + 1 proxy (`v14jft`) |
| P-76 | CNN learning rate 3e-4 → 6e-4 | our only CNN LR move went up and helped (P-59); dissimilar target domains need larger LRs (Li 2020) | 0 to +0.004, can be negative | 1 proxy (`v14blr6`) |
| P-77 | A real SWA tail: hold the LR at 0.25× over the last 25 %, average ≥ 6 snapshots | our "SWA" averages 3 EMA snapshots at ≤ 3 % of the peak LR, near-identical points, which is why it reads ≈ 0; Izmailov 2018 needs a high constant LR | 0 to +0.003 | ≈ 20 lines + 1 proxy (`v14swa`) |
| P-78 | SAM (ρ 0.05) around AdamW | robust to label noise (Foret 2021: CIFAR-10 at 40 % noise 68.8 → 93.4); medical ResNet-50 +3.8 points | 0 to +0.004 | ≈ 40 lines, ≈ 2× compute (`v14sam`) |
| P-79 | Bias-field augmentation (smooth multiplicative field) | intensity changes were BigAug's strongest MRI domain-shift transforms; our heavy stack has only global gain / gamma | 0 to +0.002 | ≈ 15 lines + 1 proxy (`v14bf`) |

Amendments to existing cards, from the same research:
- **P-67:** sharpen is the best-supported image-quality augmentation (BigAug's best single transform); blur and
  low-resolution cost accuracy in-domain there. Add a mixup arm at α ≈ 2: `v14mx`'s α 0.4 folded to λ ≥ 0.5 leaves about
  half of the mixed batches at λ ≥ 0.9.
- **P-68:** Raptor and `xfit_v09k` are both CoAtNets, so their errors may correlate. If the pair reads ✅, the follow-up
  student should be B3 (Noisy Student: a student at least as large as the teacher gains more).
- **P-62:** silence carries information for some labels (Baker's, Contusion) and not others (Synovitis, PF OA, per a
  forum gold analysis), so a per-label silent weight beats one global 0.75 if A3 reads ✅.
- **P-73:** a knee-MRI benchmark (AnyMC3D) ranks learnable-query attention, our head's type, above a transformer
  (0.962 vs 0.950).
- Low and not carded: DICOM sex as a head input (shortcut risk), ELR, target sharpening (a target change, LB-only).
  Not recommended: RandAugment / TrivialAugment (they hurt an MRI classifier), auxiliary report heads, online
  distillation from Raptor.

## Verification

- **Unit checks:** `python src/window_head_test.py` prints `UNIT CHECKS PASSED`, including the new ConvNeXt load
  (180 tensors, `head.fc` dropped, 0 missing / 0 unexpected), `param_groups` covering every parameter exactly once with
  the six expected LR levels, and the `v15c` / `v15c2` arm checks. `bash -n` passes on both RunPod scripts.
- **Local CPU smoke** of the `ARM_ONLY = "v15c"` build: one epoch on the sample studies, `_best.pt` = SWA, the 3 test
  studies predicted.
- **Kaggle smoke** log shows: the timm version; `timm convnext_tiny: loaded 180 tensors`;
  `backbone LR range 5.90e-05 .. 1.00e-04 over 4 blocks`; `teacher table raptor_teacher: 4349`; the peak GPU memory;
  a finite, falling loss; the new clip-rate / GradScaler line; `-> v15c_fold0_best.pt = SWA`; inference completed at
  `img 288`.
- **RunPod run:** the stopper is alive before training starts; each arm's log has `SWA of last 3` and
  `-> <arm>_fold0_best.pt = SWA` and no `runtime guard` (traps 47); no tripwire fired; each arm shipped before the next
  started; `kaggle datasets files tiankljucanin/rsna-knee-ckpt-<arm>` lists the checkpoint, OOF and log; a local backup
  exists; `list-pods` is empty after the delete; the billed cost is under $2.5. For a pair, the second-seed decision and
  its numbers are logged.
- **Placeholders:** `"smoke": "False"`, the right members, `decode-once verified`, `constant labels 0`; for B17 with the
  pair, the blend line shows ConvNeXt as one vote.
- **Reads:** each score logged through `/update` against the pre-registered rules above.
