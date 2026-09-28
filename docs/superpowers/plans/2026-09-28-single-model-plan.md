# Plan: raise our single-model score using the Kaggle "Best single-model" discussion

## Context
Our best single model reads 0.927–0.929 solo (seed pair 0.930). Single models in the thread (`discussion_735304.xml`) read
0.936–0.958: Scott Willis's small ResNet, single fold, 0.949 (now 0.958 overall, 3rd); Archit Konde's single-fold 2.5D CoAtNet @224,
0.950; CoolinLai's 5-fold ResNet @224, 0.954; Raymond Yuen's full-train CoAtNet, 0.949. The LB top is 0.960 and 20th is 0.955; we are
at 0.942 (the public fork). Tian wants the single model improved before more blending.

An 8-agent read-only workflow (`wf_f5888f5c-2ed`: 4 readers → 2 planners → 2 adversarial reviewers) mined the thread against our docs
and code. **Budget:** Kaggle GPU 20.55 h until 2026-10-03 00:00 UTC (then 30 h/week to Oct 22); RunPod ≈ $3 kept as an emergency
reserve only; 5 submissions/day.

**Tian's decisions (2026-09-28):**
- Today's last submission goes to the per-label probe.
- Input changes are back in scope.
- The cross-fit runs with a loose gate.
- Extras: the Raptor-over-gold check (P-49) and a small-ResNet test arm.
- The local-LLM relabel is written down as a future card only.
- No hosted-API labels.

## Diagnosis (verified locally today, gold-58, seed-averaged v09r + v09u)
- Our gold→LB offset (+0.021 across six members) equals Scott's (+0.019) and Archit's (+0.020). So the gap is single-model quality,
  already visible on gold (0.904–0.909 vs their ≈ 0.930).
- **Structural deficit:** our models trail their own LLM labels on report-explicit structure. This holds in all 8 members, every
  recipe, at 224 and 320 px. They beat the labels on image-visible, under-reported findings.

| Group | Label | LLM labels | Our models (mean of 2 seeds) | Gap |
|---|---|---|---|---|
| Trail the labels | ACL | 0.990 | 0.951 | −0.039 |
| | MCL | 0.980 | 0.916 | −0.063 |
| | PF OA | 0.903 | 0.814 | −0.089 |
| | Lat. Meniscus | 0.881 | 0.861 | −0.020 |
| Beat the labels | Effusion | | | +0.080 |
| | Fracture | | | +0.101 |
| | Contusion | | | +0.062 |
| | Med OA | | | +0.045 |

  - The shared suspect is the c02 input: cor/ax fluid slots keep 12 of ≈ 30 slices; the 130 mm crop is image-centred (the patella
    and MCL can be clipped); one series per slot.
  - The probe (step 1) checks this on ≈ 400 public studies instead of 58.
- **Targets:** the thread's most-cited lever is Archit's *out-of-fold* soft bootstrapping, heavier than 50/50.
  - Our P-38 failed with a teacher weaker than the labels (0.873 < 0.895 on gold). A v09r-generation cross-fit is the first fair test.
  - Nicolai Karcher saw gold/CV rise with no LB change → only the solo LB judges it. Prior 0 to +0.005.
- **Backbone/resolution** is not the separator (Archit uses our model class; v09x +0.002). The ResNet arm tests it at ≈ 0 extra quota.

## Step 1 — today: the per-label probe (0 training, 5th submission)
- `src/kaggle_pipeline.py`: add `PROBE_CONST_LABELS = ()` to the config cell.
  - Just before the submission write (≈ L3490), set those columns to 0.5 and print `probe: constant …`.
  - The existing guard only refuses more than 6 constant labels, so 4 are allowed.
  - Local CPU check: the knob off gives an unchanged output; the knob on gives 0.5 in exactly those columns.
- Build `artifacts/infer_probe_pair.py` with the usual 3 seds plus `INFER_MEMBERS = ["v09r", "v09u"]` and
  `PROBE_CONST_LABELS = ("ACL", "MCL", "PF OA", "Lateral Meniscus")`.
  - Push it as `rsna-knee-infer` v24 (0.1 h).
  - Verify in the placeholder output: those 4 columns = 0.5, the other 8 identical to v23's.
  - Submit, then watch with `watch_submission.py`.
- **Read (pre-registered):** full = 0.930 (#26), probe P.
  - mean4 = 0.5 + 3·(0.930 − P) (± 0.003); mean8 = (12·0.930 − 4·mean4)/8.
  - Structural weakness is confirmed on the test if mean8 − mean4 ≥ 0.03 (gold says 0.028).
  - If mean4 ≥ mean8 − 0.01, the deficit is gold-specific; the input arm still runs, with a lower prior.
- Diagnosis only — never used to tune weights.

## Step 2 — one code batch, one Kaggle smoke (today)
Everything the week needs goes into this batch, so every later push is a sed-only build. Kaggle has only 2 GPU slots, and each smoke
risks a multi-hour queue (traps 41).

1. **Cross-fit (XF) arms** v09k0…v09k4 in `SHIPPED_ARMS`.
   - Each is `{**PROD, "train_all": False, "folds": (k,), …v09r dict}` on the existing 5-fold column. Arm-dict `folds` already
     overrides `ARM_FOLDS` (L3138).
   - `DISTILLED_ARMS` entries: `("raptor_teacher",)`.
   - Held-out eval at the end only (`eval_final_only`): per-epoch validation adds ≈ 1 h per session.
   - Fold mode trains other folds' gold rows (1.3 % of loss); each gold row's OOF is still honest.
2. **`run_parallel_arms`** (L3033–3112): judge a child by `{arm}_fold*_best.pt` and record `results[f"{arm}/{k}"]` (today it checks
   fold0 only).
   - Smoke forces fold 0 (L631), so folds 1–4 get a non-smoke unit test in `src/targets_test.py` (split_studies: fold k excluded
     from train, val = fold k incl. its gold rows).
   - Pre-registered post-run log check: `fold k: train N / val M` and `SWA of last 3`.
3. **ResNet support** in `load_timm_backbone` (L1984) / `param_groups` (L2415).
   - Pass `img_size` only for coatnet/vit.
   - Drop the classifier keys via `pretrained_cfg["classifier"]`.
   - Build stage groups for ResNet `layer1-4`.
   - Arm `v13a` = the v09r recipe with `backbone: "timm:resnet34"` (window_attn head, c02, Raptor 0.5).
   - Weights: local `huggingface_hub` download of `timm/resnet34.a1_in1k` → private Dataset `tiankljucanin/timm-resnet34-a1`.
   - `src/window_head_test.py` builds ResNet-34 at 224.
4. **c03 input** in `src/cache_pipeline.py` (`SCHEME_DEFAULTS["c03"]`).
   - Budgets and crop are set by the census in step 3.
   - `src/cache_selftest.py` must pass.
   - Arms `v11a` / `v11b` = the v09r recipe on c03, seeds 42 / 43, `train_windows` scaled to the window increase (≈ 32–34) so the
     train/eval attention ratio stays ≈ 2.5×.
5. **Student arms** `v09o` / `v09o2` (seeds 42 / 43).
   - Session-level `TEACHER_TABLES = ("raptor_teacher", "xfit_v09k")`, `TEACHER_MIX = 0.75`, i.e. 0.25 LLM + 0.375 Raptor + 0.375
     honest OOF (`mix_teacher` means the matched tables).
   - Add the `TEACHER_PATHS["xfit_v09k"]` entry and the `DISTILLED_ARMS` entries.
   - No per-arm teacher code: the traps 40/42 risk class is avoided because every session pairs arms of one spec.
6. **`src/build_distill_table.py --per-fold-rank`**: rank within each fold before pooling, so fold calibration offsets vanish.
7. **`src/build_teacher_pass.py --gold`**: the 58 gold studies only, for P-49 (≈ 20 lines; `src/teacher_pass_test.py`).

- **Checks:** `window_head_test.py`, `targets_test.py`, `teacher_pass_test.py`, `cache_selftest.py`, then a local CPU smoke.
- **Kaggle smoke:** one `FORCE_SMOKE = True` push of `PARALLEL_ARMS = ("v09k1", "v13a")` (≈ 0.2 h). It exercises the fold arm path,
  the ResNet load and the parent check.

## Step 3 — c03 census and cache (CPU only, 0 GPU)
- Pull `series_meta.csv` + `manifest.csv` from the `rsna-knee-cache2-a` output (a few MB, `--file-pattern`).
- Local census per slot: native slice counts and FOV (rows × spacing).
- Pre-registered rules:
  - Each fluid slot's budget = min(median native, 24). The sagittal cap is expected ≈ 20–22, so sagittal is not over-sampled.
  - The T1 slots stay at 8.
  - Crop = 150 mm if the median FOV ≥ 150 mm, else 140 mm.
- Build 4–6 new CPU kernels `rsna-knee-cache3-*`. Never re-version `cache2-*` (traps 27); keep each shard ≤ 20 GB.
- Mount c03 only on `rsna-knee-folds` for session D. Drop the unused c01 mounts there, so c02 and c03 are never both mounted.

## Step 4 — GPU sessions (≈ 16 h planned of 20.55 h; the ≈ 4.5 h buffer covers one failed session or queue)

| When | Slot `rsna-knee-train` | Slot `rsna-knee-folds` | Quota |
|---|---|---|---|
| Mon 09-28 (after the probe placeholder + smoke) | **A** v09k0 ‖ v09k1 | **B** v09k2 ‖ v09k3 | 2.45 + 2.45 |
| Tue 09-29 AM | P-52 placeholder (0.1) | P-49 Raptor-on-gold (0.2) | 0.3 |
| Tue 09-29 | **C** v09k4 ‖ v13a ResNet-34 (both Raptor 0.5) | **D** v11a ‖ v11b (c03, 2 seeds) | 2.8 + ≈ 3.7 |
| Wed 09-30 | **E** v09o ‖ v09o2 (student pair; loose gate) | placeholders | 2.9 + ≈ 0.8 |
| Fri 10-02 (conditional) | **F** seed twin of any KEEP winner, only if ≥ 3.5 h quota is left and it ends before Sat 00:00 UTC | — | ≈ 3 |

- **XF table (Wed, CPU):**
  1. `build_distill_table.py --per-fold-rank` over `v09k*_fold*_oof.csv` → `artifacts/teacher/xfit_v09k.csv`.
  2. Log the gold diagnostic: pooled per-label AUC vs the LLM, vs Raptor (from P-49) and vs the 0.5/0.5 mixed target; paired
     bootstrap; ρ vs Raptor against the 0.821 baseline.
  3. **Loose gate:** E runs unless pooled gold < 0.875 AND ≥ 8/12 labels are below the LLM.
  4. New version of the private Dataset `rsna-knee-teacher-tables`. Never public: it is keyed by StudyInstanceUID.
- **Before each push:** grep the sed'd build for `^(FORCE_SMOKE|PARALLEL_ARMS|ARM_ONLY|TEACHER_TABLES|TEACHER_MIX) =`, and check
  `machine_shape` is T4.
- **After each run:** pull, check `ok arm` + `SWA of last 3` + no runtime guard + the `teacher table … studies` lines, then ship
  `rsna-knee-ckpt-<arm>` Datasets and add them to `kaggle/rsna-knee-infer/kernel-metadata.json`.

## Step 5 — solo reads (pre-registered; half-open intervals on 2-seed means, 4 decimals)

| Read | Comparator | ✅ KEEP | 🔁 | ❌ |
|---|---|---|---|---|
| P-52 v09r+v09u+v09x (Tue) | #26 pair 0.930 | ≥ 0.934 | 0.931–0.933 | ≤ 0.930 |
| XF 5-fold ensemble (flat rank-mean, ≈ 1–1.5 h to score) | 0.930 | ≥ 0.935 | 0.931–0.934 | ≤ 0.930 |
| v13a ResNet-34 solo | 0.927 | ≥ 0.932 | 0.923–0.931 | ≤ 0.922 |
| c03 m = mean(v11a, v11b) (+ pair vs 0.930) | 0.927 | m ≥ 0.9315 | 0.9285 ≤ m < 0.9315 | m < 0.9285 |
| Student m = mean(v09o, v09o2) (+ pair ≥ 0.935) | 0.927 | m ≥ 0.9315 | 0.9285 ≤ m < 0.9315 | m < 0.9285 |

- **Student outcomes:** a DEAD student is the third null (P-38, Nicolai) → the OOF line closes. A KEEP gets an attribution control
  (0.25 LLM + 0.75 Raptor) in week 2.
- **Discipline:** any change enters the final spec only after a seed-twin or pair replication. Gold-58 is direction only; never
  judge by OOF vs the teacher (traps 39).
- **Submissions per day:**

| Day | Submissions |
|---|---|
| Mon | 1 (probe) |
| Tue | 1 (P-52) |
| Wed | ≤ 5: XF ensemble, v13a, v11a, v11b, c03 pair |
| Thu | 3: v09o, v09o2, pair |

## Step 6 — week 2+ (Oct 3–22, 30 h/week), branch on results
- The winning input/targets get a 320-px member plus seeds. A second cross-fit round only if the student KEEPs.
- P-45 second teacher: the D4 exclusion (09-23) is re-asked then.
- Final retrains in the week of Oct 10; P-50 final picks with fork builds by Oct 21. #27 (fork β 0.20) is read when it lands.

## Docs (`/update` after every result, `/handoff` at the end)
- `docs/research.md`: new section "Kaggle discussion 735304 — mined 2026-09-28" (the idea table, who said what, counter-evidence).
- `docs/experiments.md`: the per-label gold diagnostic now; each read as it lands.
- `docs/proposals.md`: new cards.
  - P-53 probe, P-54 XF cross-fit + fold ensemble, P-55 OOF student, P-56 c03 input, P-57 ResNet-34 arm.
  - **P-58 local-CPU open-weights LLM relabel as a 4th vote (future, not scheduled).**
  - Update P-49 and P-52.
- `docs/brainstorm.md`: questions Tian may post in the thread.
  - Scott: label-side or training-side fix?
  - CoolinLai: labels and gold score?
  - Archit: did "50/50 jumped" mean the LB? what final weight? all cells or silent ones?
  - Nicolai: were his predictions OOF?
  - tennogh: were OOF pseudo-labels used as targets?

## Authorisation implied by approving this plan
Approval covers:
- The kernel pushes named above, a smoke first after the code edit.
- The CPU cache kernels.
- The private Datasets: timm-resnet34 weights, the rsna-knee-ckpt-* checkpoints, the new private version of rsna-knee-teacher-tables.
- The submissions listed.

Anything else stops and asks. No RunPod spend without asking. No report text off Kaggle.

## Verification
- **Local:** the four test scripts plus the CPU smoke pass before any push.
- **Kaggle:** smoke green (`ok arm` ×2; the ResNet timm line; fold-arm artefacts via the glob).
- **Every real run:** the log greps listed in Step 4. Every submission is timed by `watch_submission.py` and logged to
  experiments.md with the pre-registered verdict.
- **Probe:** the placeholder CSV shows the 4 columns = 0.5 and the other 8 identical to v23's.
