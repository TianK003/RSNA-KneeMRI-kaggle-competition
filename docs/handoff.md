# Handoff — where we left off

**Update this at the end of every working session.** Newest entry on top. Keep it short and
concrete: what changed, what state things are in, what the next action is. This is the file
to read first after a break.

---

## 2026-10-03 (17:20 UTC) → 2026-10-04 (08:45 UTC) — 8 reads + session C: **the CNN line wins** (EfficientNet-B0 `v13e` 0.935 = best solo); **the three-family trio reads 0.938 = best own-model score**; P-45 / P-61 / P-63 closed

Tian, in order:
- "collect results and read the handoff"
- "Yes, start session C now. no need to remove the pushed commit, just leave it as is"
- "Continue, submit the work"

Commits `9f11579` … `b1d7f80`, plus this one. `9f11579` carries a `Claude-Session:` trailer by mistake. Tian said to keep it; later
commits have none (memory `no-session-trailer-in-commits`).

Nothing is in flight: every kernel is COMPLETE and all 5 submissions of 2026-10-04 are scored.

### Where things stand

| | Status |
|---|---|
| Best LB / best own | 0.942 (public-stack fork) / **0.938 = #48, the trio `v11a` + `v13r` + `v13e`** (own models only) |
| Best solo | **0.935 `v13e`** (EfficientNet-B0, `v13h` recipe; ≈ 15 min to score, 17 MB). Then `v13r` 0.934, `v11a` 0.932, `v13h` 0.931 |
| Reads this session | #41 `v11p` 0.929 🔁 · #42 `v13h` 0.931 ✅ · #43 `v11d` 0.930 🔁 · #44 `v11dl` 0.927 🔁 · #45 `v13r` 0.934 🔁 · #46 `v13e` 0.935 ✅ · #47 `v11a` + `v13h` 0.934 🔁 · #48 trio 0.938 🔁 (bar 0.939). experiments.md 2026-10-03 "Sessions A ‖ B", "Submissions #41–#43"; 2026-10-04 "Session C", "Submissions #44–#47", "Submission #48" |
| Cards | P-64 ✅ (the CNN recipe). P-45 (as a training target), P-61 and P-63 🔁 closed, not adopted. Targets stay 0.5 LLM + 0.5 Raptor |
| Datasets (private, ready) | `rsna-knee-ckpt-{v11p,v13h,v11d,v11dl,v13r,v13e}`, all in `kaggle/rsna-knee-infer/kernel-metadata.json` |
| Committed renders | `rsna-knee-train` = **v43, session C REAL** (5.87 h). `rsna-knee-train-b` = v4, session A REAL. `rsna-knee-infer` = **v44 = the trio (#48), submittable**. Never re-push a training render as is |
| Staged builds | `artifacts/infer_solo_{v11p,v13h,v11d,v11dl,v13r,v13e}.py`, `infer_pair_v11a_v13h.py`, `infer_trio_v11a_v13r_v13e.py` (each `src` + 3 seds) |
| Quota | Kaggle GPU **20.25 / 30 h** (9.75 h left, resets 2026-10-10). Submissions: 0 left on 10-04, 5 on 10-05 |

### What we talked about and decided

- **Session C ran on Tian's go**, after `v13h` passed its ✅ bar (0.931 ≥ 0.925), though not the 0.935 "main bet" bar. A smoke ran
  first.
  - EfficientNet's `0 stages` / `over 0 blocks` log line was checked in `param_groups` and is cosmetic. At `llrd_decay` 1.0 every
    encoder parameter gets the uniform 3e-4.
- **The fifth slot on 10-04 waited for the pair's read.** Once `v11a` + `v13h` read above both members, the most informative fifth
  was the three-family trio. A second CNN-only pair or the four-member blend would have answered less.
- **Pre-registered bars were applied as written.**
  - The trio is 🔁 at 0.938 vs the 0.939 bar, even though it is our best own score.
  - `v11dl` is 🔁 at −0.003. It is closed because gold agreed in sign; the LB did not rule.

### What we figured out

1. **The forum's CNN recipe transfers to our pipeline.** That recipe is long schedules, heavy aug, drop-path, frozen BN, a CNN LR and
   c03 input.
   - EfficientNet-B0 0.935 > ResNet-50 0.934 > ResNet-34 0.931, the forum's order.
   - All three are at or above the six-read CoAtNet plateau (0.929–0.932).
   - Source: experiments.md "Submissions #41–#43", "#44–#47".
2. **Gold-58 cannot judge cross-family or recipe changes.**
   - `v13h` was −0.001 on gold and +0.010 on the LB.
   - All three CNNs read under `v11a` on gold and above it on the LB.
   - Source: traps 39, extended 2026-10-03.
3. **Different families blend; same families never did.** Results: #47 +0.002 and #48 +0.003 over the best member, +0.004 over the
   mean, against four flat same-family blends. The gain grew from 2 families to 3. Source: experiments.md "Submission #48".
4. **Target and head changes on the CoAtNet do not transfer.** D4 targets −0.002, the reader −0.003, the higher LR −0.003. The
   CoAtNet stays at the `v11a` recipe as the ensemble's non-CNN family.

### ⏭ Next action, in order

1. **GPU (Tian's go; 9.75 h left, one session).** Strengthen the best family. Proposal: `v13e2` (`v13e` at seed 43) ‖ a bigger
   EfficientNet on the same recipe (`efficientnet_b2`, needs a `timm-efficientnet-b2-*` weight Dataset made like `timm-efficientnet-b0-ra`,
   commit `6856f6b`). The steps:
   - Add both arms to `ARMS` in `src/kaggle_pipeline.py` beside `v13e` (`"seed": 43` is honoured per arm since traps 42).
   - Run `src/window_head_test.py` and a local smoke.
   - Run the Kaggle smoke: `sed PARALLEL_ARMS = ("v13e2", "v13b2")`, `TEACHER_TABLES = ("raptor_teacher",)`, `TEACHER_MIX = 0.5`,
     like `artifacts/train_sC_kaggle_smoke.py`.
   - Then the real run, ≈ 6 h (session C took 5.87 h). It is green only if each child log has `SWA of last 3` and no runtime guard.
   - The seed twin tells us whether 0.935 is the recipe or a lucky seed: the P-44 floor is 0.004.
2. **2026-10-05 submissions (5).** Keep a blend only if it reads ≥ its best member + 0.004.
   - (a) the four-family blend `v11a` + `v13h` + `v13r` + `v13e`: does a fourth member add?
   - (b) the CNN-only trio `v13h` + `v13r` + `v13e`: is the CoAtNet needed?
   - (c) `v11a` + `v13e`: the strongest pair.
   - (d) and (e): keep for the session-1 members.
   - Each is `sed -e 's/^INFER_MEMBERS = \[.*\]/INFER_MEMBERS = [...]/'` on `src` + the two infer seds, then `nbgen` → push
     `kaggle/rsna-knee-infer`.
   - Check the placeholder log for `smoke False`, `infer members (N)`, `N members in 1 geometry group(s)`, decode-once verified and
     `constant labels 0`.
   - Submit with `-v <N>`, then watch with `python src/watch_submission.py --ref <ref> --every 90`.
3. **P-50 (final picks, by 10-15).** One pick = our best own ensemble (now #48, 0.938). For the other, the 0.946–0.947 teams blend a
   diverse own leg at weight 0.45–0.5 into the public stack. Our trio is now that leg.
   - Test: `src/build_fork.py` with the trio as our arm at β 0.45 vs #13's 0.942.
   - The fork takes hours to score (#17 ≤ 8 h), so send it early in a day.
   - The v34 public stack (0.943) rebuild is optional.

### Open decisions for Tian

- **The 9.75 GPU-h before 10-10:** the EfficientNet session above (≈ 6 h), or a mix (e.g. one seed twin + keep ≈ 3.5 h buffer).
- **P-50:** whether one final pick is the public-stack fork with our trio at β ≈ 0.45, and when to spend a slot on it.
- **Efficiency prize:** `v13e` solo (0.935, ≈ 15 min, 17 MB) is our natural fast pick. The formula is still unread (P-18).
- Unchanged: team-merge offers (merger deadline 10-15); hosted-API labels (Tian's "no" stands).

### Things that will bite if forgotten

- **`rsna-knee-train` v43 and `rsna-knee-train-b` v4 are REAL renders.** Re-pushing either starts a 4–6 h session.
- **CNN training is input-bound at first and speeds up.** Session C's first "epoch ETA 22 min" settled to ≈ 11.5 min/epoch (5.87 h
  for 30 epochs). Do not judge a run's length from the first ETA line.
- **The trio takes ≈ 44 min to score** (three members, one decode pass); a CNN solo takes 15–20 min. Budget slots so reads land the
  same day.
- **Placeholder logs are JSON.** `grep` truncates at the escaped `\n`, so parse them (`json.loads`, join `data`). The scratch helper
  `wait_check.sh` did this; it is not in the repo.

## 2026-10-03 (09:05 → 16:20 UTC) — quota reset: P-60 part 2 read 🔁 (0.932 / 0.932); P-45 D4 teacher built, spiked, passed, published; whole-forum mining → **our gap is image-side, the best singles are CNNs**; sessions A (`v11p` ‖ `v13h`) and B (`v11d` ‖ `v11dl`) running; handoff written BEFORE their results

Tian, in order:
- "push our work and continue research … focus on improving single model performance"
- "how much GPU … be tactical … not on every small change" → stopped P-62 himself
- "go on P-45"
- agreed to P-63
- "submit at least two or three different models today … an ensemble of 3–5 models, not 20 … research the leaderboard and
  discussion 735304 … research thoroughly"
- "/handoff now, even before the results, so I can clear this chat"

Commits `9e2c9d0` … `b29d83c`, plus this one.

### ⏳ Still in flight as this was written (16:20 UTC)

**The watchers of this chat die with it, so poll by hand.**

| In flight | What it is | Started | How to check | How to read it |
|---|---|---|---|---|
| **Session A = `rsna-knee-train-b` v4** (REAL) | `PARALLEL_ARMS = ("v11p", "v13h")`, `TEACHER_TABLES = ("raptor_teacher",)` (`artifacts/train_sA_real.py`). **`v11p`** = `v11a` + P-63 per-finding spatial reader + slot-count norm, seed 42. **`v13h`** = P-64 ResNet-34 on c03 + CNN LR 3e-4 uniform + frozen BN + aug heavy + drop-path 0.1 + **30 epochs** | 12:21 UTC; expect COMPLETE ≈ 16:30–17:00 | `kaggle kernels status tiankljucanin/rsna-knee-train-b`; then `kaggle kernels output tiankljucanin/rsna-knee-train-b -p artifacts/kaggle_out/sA --file-pattern "(v11p\|v13h)_fold0_(best\.pt\|oof\.csv)$\|\.log$"` | **Green** = parent `ok  arm` ×2 **and** each child log has `teacher table raptor_teacher: 4349`, `SWA of last 3`, `-> <arm>_fold0_best.pt = SWA`, no `runtime guard` (traps 47: `ok  arm` alone is not enough). `v13h`'s log also shows `freeze_bn: 36 encoder BatchNorm` and `drop_path_rate 0.1`, epochs 0–29. Gold-58 SWA is direction only (`v11a` 0.9204, `v13c` 0.9014). `ERROR` with 0 files = traps 44 → re-push the same build |
| **Session B = `rsna-knee-train` v41** (REAL) | `PARALLEL_ARMS = ("v11d", "v11dl")`, `TEACHER_TABLES = ("raptor_teacher", "d4_teacher")`, mix 0.5 → 0.5 LLM + 0.25 matched Raptor + 0.25 matched D4 (`artifacts/train_sB_real.py`). **`v11d`** = the `v11a` recipe (P-45). **`v11dl`** = `v11d` + lr 2e-4 / LLRD 0.85 (P-61) | 12:35 UTC; expect ≈ 16:15–16:45 | same, slug `rsna-knee-train`, dir `artifacts/kaggle_out/sB`, pattern `(v11d\|v11dl)_fold0_…` | Green as above, plus both child logs show `teacher table d4_teacher: 4349` and `training targets = … ['raptor_teacher', 'd4_teacher']`. LR lines: `v11d` 2.37e-05 .. 1.00e-04 (decay 0.75), `v11dl` 8.87e-05 .. 2.00e-04 (decay 0.85) |

No submission is pending: #39 and #40 were both read at 0.932. **3 submissions are left for 2026-10-03, until 00:00 UTC.**

### Where things stand

| | Status |
|---|---|
| Best LB / best solo | 0.942 (fork) / **0.932**: `v11a`, the c03 pair, `v11n`, `v11n2`. Every c03 CoAtNet variant reads 0.929–0.932 — a plateau |
| P-60 noisy student | 🔁 closed: #39 `v11n` 0.932, #40 `v11n2` 0.932 → m 0.932 vs 0.9305 (+0.0015). Not adopted (1.5× training time). The first Kaggle resume worked (traps 31). experiments.md 2026-10-03 "P-60 part 2" |
| P-45 D4 teacher | ✅ table built: builder `src/build_d4_teacher_pass.py` (56 checks); gold spike 0.9301 vs 0.9302 on D4's original grid; full pass 4,349/0 failed (2.37 h); in-sample check passed (ρ D4~LLM 0.695 vs Raptor 0.652, bar 0.80); `d4_teacher.csv` is in `rsna-knee-teacher-tables`. Student in session B. experiments.md 2026-10-03 (three entries) |
| P-62 silence mix | ⏸ parked: Tian stopped `rsna-knee-train-b` v1 after ≈ 0.3 GPU-h (cannot clear its band); a bundle candidate only |
| P-63 reader / P-64 long CNN / P-61 LR | ⏳ in sessions A / B (one seed each) |
| Staged, not pushed | **Session C** = `("v13r", "v13e")`: ResNet-50 a1 / EfficientNet-B0 ra on the `v13h` recipe (`artifacts/train_sC_kaggle_smoke.py` / `_real.py`). Local smoke green; weight Datasets `timm-resnet50-a1` / `timm-efficientnet-b0-ra` mounted in train / train-b / infer |
| Research | `docs/research.md` §2.7.3 (102 forum topics) and §2.7.4 (30 public tables, notebooks, both LBs); raw notes in `artifacts/research_1003/`, the forum in `artifacts/forum/` |
| Quota | Kaggle GPU **13.47 / 30 h** at 16:16 UTC (A + B still counting → ≈ 14.5 h when done); resets 2026-10-10 |
| Committed renders | `rsna-knee-train` = **v41 REAL session B**; `rsna-knee-train-b` = **v4 REAL session A**; `rsna-knee-teacher-d4` = **v2 = the FULL pass** (re-pushing costs 2.4 GPU-h); `rsna-knee-infer` = v36 (`v11n2` solo). Never re-push any of them as is |

### What we talked about and decided

- **GPU rule (Tian):** no session for a change that cannot clear its own read band → P-62 stopped. **Then (mine, from the
  numbers): one seed per idea, two ideas per session.** A single-seed read needs ≥ 0.004 (P-44) and the seed-pair band is
  ± 0.0045, so a seed twin buys almost nothing. Seed twins are kept for ensemble members only.
- **Session pairing:** targets are per session.
  - A pairs two Raptor-target ideas (reader, long CNN).
  - B pairs two Raptor + D4 ideas: `v11d` reads P-45 against `v11a`, and `v11dl` reads P-61 against `v11d` in the same session.
- **The D4 student keeps matched mixing**, not raw probabilities: raw was +0.0014 on gold for the three-source target (noise), and
  matched is the form that transferred for Raptor.
- **P-63 has a weak forum prior** (Will's region tokens ≈ 0; Tucker 0.94+ with no attention). Run anyway as Tian approved, at one seed.
- **Tian wants a 3–5-model ensemble, not 20, and 2–3 different models scored today.**
- **Rejected:**
  - a VLM member (Qwen 3.5 2B, 0.950 single fold, but A100-class training);
  - external-data pretraining (allowed now, too expensive before 10-22);
  - more label engineering (no public table beats ours; Tucker "low 0.95s" with plain Qwen labels).

### What we figured out

1. **The noisy-student bundle does not move the CoAtNet** (0.932 / 0.932). Regularisation + 12 epochs is not where 0.94–0.95 comes
   from for us (experiments.md "P-60 part 2").
2. **D4 is a usable second teacher:**
   - its gold target analog reads 0.9331 vs 0.9268;
   - the spike reproduced it, 0.9301;
   - its training-study predictions are not replayed labels;
   - it disagrees with Raptor most on report-silent cells (ρ 0.724).
3. **Quantile matching flattens the teacher** (matched Raptor keeps only 14–32 distinct values among the 58 gold). Raw mixing +0.004
   on gold, 1 SD, labels split. Not adopted (experiments.md 2026-10-03).
4. **The forum's best single models are CNNs:**
   - CoolinLai: ResNet-50 at 224, 0.954.
   - Tucker: ResNet-34 / EfficientNet-B0, "low 0.95s".
   - Scott: small ResNet, 0.949.
   - SpeedSci: their CNNs 0.940 vs their CoAtNet 0.926.
   - Others' LLM-only singles read 0.926–0.943 vs our `v09a` 0.918, so our gap is image-side. Gold-58 inverted the LB order in ≥ 4
     reports (research.md 2.7.3).
5. **Ensembles pay only with different families or pipelines** (+0.005–0.007: SpeedSci 0.947 / 0.949, ringbearer 0.948). The
   0.946–0.947 teams blend a diverse own leg at weight 0.45–0.5 into the public 0.943 stack (research.md 2.7.4).
6. **Host rulings we had as open:** LLM APIs are allowed for labels; click-through external data is not excluded; KneeCoT is
   banned (CLAUDE.md "Rules" updated).

### ⏭ Next action, in order

1. **Pull A and B** (commands in the table) → check every green criterion → note the gold-58 SWA of all four (direction only) →
   `/update`.
2. **Ship the four checkpoints**:
   ```bash
   for a in v11p v13h v11d v11dl; do d=artifacts/ship_$a; rm -rf $d; mkdir -p $d
     src=$( [ -f artifacts/kaggle_out/sA/${a}_fold0_best.pt ] && echo sA || echo sB )
     cp artifacts/kaggle_out/$src/${a}_fold0_{best.pt,oof.csv} $d/
     echo "{\"title\": \"RSNA knee ckpt $a\", \"id\": \"tiankljucanin/rsna-knee-ckpt-$a\", \"licenses\": [{\"name\": \"other\"}]}" > $d/dataset-metadata.json
     (cd $d && ../../.venv/Scripts/kaggle.exe datasets create -p .); done
   ```
   Add `tiankljucanin/rsna-knee-ckpt-{v11p,v13h,v11d,v11dl}` to `kaggle/rsna-knee-infer/kernel-metadata.json`, and wait for
   `kaggle datasets status … = ready`.
3. **Placeholders, one at a time**, each built from the CURRENT `src`: older staged infer scripts lack the `spatial_reader` member
   key and `v11p` would not load.
   ```bash
   sed -e 's/^FORCE_SMOKE = True/FORCE_SMOKE = False/' -e 's/^MODE = "auto"/MODE = "infer"/' \
       -e 's/^INFER_MEMBERS = \[.*\]/INFER_MEMBERS = ["v11p"]/' src/kaggle_pipeline.py > artifacts/infer_solo_v11p.py
   ```
   Then `nbgen` → push `kaggle/rsna-knee-infer` → COMPLETE → pull its log (`--file-pattern "submission\.csv$"`) and check `smoke
   False`, `infer members (1): v11p/fold0`, `decode-once verified`, `constant labels 0`. Submit `-k tiankljucanin/rsna-knee-infer -v
   <N>` and watch with `python src/watch_submission.py --ref <ref> --every 90`. **Before 00:00 UTC: `v11p`, `v13h`, `v11d`. Tomorrow:
   `v11dl`.**
4. **Read rules** (set before sending; one seed each, P-44 floor 0.004):
   - **`v11d`** (P-45) vs `v11a` 0.932: ✅ ≥ 0.936 / 🔁 0.929–0.935 / ❌ ≤ 0.928.
   - **`v11p`** (P-63) vs `v11a` 0.932: the same bands.
   - **`v13h`** (P-64) vs `v13c` 0.921: ✅ ≥ 0.925 (the recipe helps the CNN). As a member it matters if ≥ 0.930. **If ≥ 0.935, the
     CNN line becomes the main bet** → step 5.
   - **`v11dl`** (P-61) vs `v11d`, same session: ✅ ≥ `v11d` + 0.004 / ❌ ≤ `v11d` − 0.004.
5. **Session C, Tian's go:** if `v13h` ✅, push the staged ResNet-50 ‖ EfficientNet-B0 (`artifacts/train_sC_kaggle_smoke.py` first,
   then `_real.py`; ≈ 3.5–4 h). The CNN base is c03. Use the D4 targets instead (`TEACHER_TABLES = ("raptor_teacher", "d4_teacher")`
   needs new arm names in `DISTILLED_ARMS`) only if `v11d` ✅.
6. **Ensemble (3–5 members), one submission per blend:** the best CoAtNet + the best CNN (+ a third family or the D4-target
   variant). `INFER_MEMBERS = [...]` builds a flat rank-mean, with one decode group per cache geometry. **Keep a blend only if it
   reads ≥ the best member solo + 0.004.** Same-teacher same-family blends have read flat four times (P-42, P-52, #32, #38).

### Open decisions for Tian

- **Session C** (CNN follow-ups, ≈ 4 GPU-h) after the `v13h` read; and how much of the remaining ≈ 15.5 h to spend before the
  10-10 reset (two more weekly resets before the 10-22 deadline).
- **Final selection (P-50, by 10-15):** one pick = our best own model or ensemble; the other could be the public stack (v34 = 0.943)
  + a *diverse* own leg at weight 0.45–0.5, which is what the 0.946–0.947 teams do. That needs a v34 rebuild of
  `src/build_fork.py`. For the Efficiency prize to count, one pick must be a fast solo.
- Hosted-API labels are now allowed by the host. Tian's 09-28 "no" still stands; the forum prices it low.
- Team-merge offers on the forum (Prateek, SpeedSci, others; merger deadline 10-15).

### Things that will bite if forgotten

- **Committed renders are REAL runs:** `rsna-knee-train` v41, `rsna-knee-train-b` v4, `rsna-knee-teacher-d4` v2 (the full pass).
  Rebuild from `src` + seds before any push. Metadata of `rsna-knee-train` / `-train-b` is c03-only + the new CNN weights (no
  `rsna-knee-folds`).
- **traps 47:** check each child log for `SWA of last 3` before shipping; `ok  arm` is not enough.
- **traps 48:** the Kaggle API rate limit is shared; a bulk SDK job makes `kernels status` return 429. Re-check before acting on a
  watcher's exit.
- **Forum re-harvest:** `python scripts/harvest_forum.py artifacts/forum` (search-based; backs off on 429; skips topics already
  saved). The topics stay local in `artifacts/forum/`. Commit nothing from `artifacts/`: it holds report text and UIDs.
- `discussion.py` / `discussion_735304.xml` in the repo root are Tian's, untracked on purpose.

## 2026-09-30 (21:00) → 2026-10-01 (09:40 UTC) — P-60 part 1 pulled green; label side priced → **P-62 silence-aware mix implemented + Kaggle smoke green**; P-60 part 2 verified staged; **work resumes 2026-10-03 after the GPU reset**

Tian: "Fetch the results, then run /update" → "is there anything we could do in the meantime, the labels?" → "Ok continue working" →
"run /update and /handoff saying we continue work on 3rd october". Commits `69a8932`, `c37ef93`, `7813751`, this one. (No handoff
entry was written for 2026-09-29 evening / 2026-09-30 morning; their results are experiments.md "P-59", "Submissions #34–#38",
"P-18 step 1" and the git log.)

### ⏳ Still in flight as this was written (09:40 UTC)

Nothing on Kaggle (both slots `COMPLETE`; no submission pending). (The P-45 branch-mapping subagent finished after this entry was
first written; its answer — **D4 alone** — is now in the P-45 card.)

### Where things stand

| | Status |
|---|---|
| Best LB / best solo | 0.942 (fork) / **0.932** (`v11a`, the c03 pair) — unchanged |
| P-60 part 1 | ✅ `rsna-knee-folds` **v11**, 2.61 h: `v11n` / `v11n2` guard-stopped in epoch 5 by design; gold-58 EMA epoch 4 0.9166 / 0.9129 = `v11a` / `v11b` at theirs (direction only); **both `_last.pt` live in the v11 output** — experiments.md 2026-09-30 "P-60 part 1" |
| P-60 part 2 | ⏳ staged, verified: `artifacts/train_p60_kaggle_part2.py` (= part 1's build with the 8.3 h guard — the only diff) + `artifacts/p60_part2_kernel-metadata.json` (c03 ×4 + `rsna-knee-folds`); Tian's go of 2026-09-30 covers it |
| P-62 (new) | 🔧 implemented (`7813751`): `TEACHER_SILENT_MIX` + `silence_mask`, arms `v11s` ‖ `v11s2`, guards both ways; unit + AST-parity checks green, default targets md5 `29f641ed` unchanged, pipeline yt == `build_targets.py` yt exactly; local smoke + **Kaggle smoke `rsna-knee-train` v38 green**; real build staged as `artifacts/train_p62_real.py`; **real run needs Tian's go** |
| Label side | priced at 0 GPU: silence-aware target 0.9300 vs 0.9268 on gold (+0.0032, SD 0.0018), flat mixes at equal Raptor weight +0.0004 — experiments.md 2026-09-30 "Silence-aware teacher mix" |
| Committed renders | `rsna-knee-train` = **v38 = P-62 SMOKE** (metadata c03-only); `rsna-knee-folds` = **v11 = P-60 part 1 REAL** — never re-push either as is |
| Quota | Kaggle GPU 29.46 / 30 h (**0.54 h left**, do not spend), reset **2026-10-03 00:00 UTC**; submissions unused 2026-10-01 / 02 (no new member to read) |
| Docs | ✅ all logged: experiments.md (P-60 part 1, silence-aware mix + 2 Scoreboard rows), proposals.md (P-60 / P-62 cards + index), traps.md **47**, CLAUDE.md state line |

### What we talked about and decided

- **"The labels?"** — the report side is close to squeezed: a 4th LLM / open-weights relabel (P-58) is priced low (Scott's own Gemma
  labels ≈ 0.89 gold < our 0.8948 blend; the thread: extraction fixes did not move the LB), dread cannot be priced (no gold rows),
  OOF targets are three nulls. The one label idea with a number behind it is **where** Raptor speaks (silent cells) → P-62. The
  only label change that ever transferred was new *information* (Raptor) → a second image teacher (P-45) is the bigger lever.
- **P-62 runs as two seeds in one session** (`v11s` ‖ `v11s2`): the silent mix is per session, so it can only pair with its own seed
  twin; two seeds read as m are better than one seed + an unrelated arm. One solo would sit under the 0.004 floor anyway.
- **Part 2 keeps part 1's exact code** (the 2026-09-30 build), not a rebuild from today's `src` — the P-62 code is inert at
  `TEACHER_SILENT_MIX = None`, but a resume should not change code mid-run.
- **The 0.54 h left is not spent** — a P-45 spike needs a builder that does not exist yet and Tian's go.

### What we figured out

1. **Heavy regularisation did not slow the c03 curve** — `v11n` / `v11n2` match `v11a` / `v11b` at epoch 4 while still at ≈ 72 %
   of peak LR (vs ≈ 37 %); part 2 decides (experiments.md "P-60 part 1").
2. **A guard-stopped `PARALLEL_ARMS` child prints `ok  arm` / `"completed": true`** — its `_best.pt` is a mid-schedule EMA, never a
   member; ship only with `SWA of last 3` in the child log (traps 47).
3. **Silence placement, not Raptor amount, is the only label re-weighting that moves the gold target** (+0.003 vs +0.0004 at equal
   weight); inside silent cells the LLM ranks Fracture positives at AUC 0.17 (experiments.md "Silence-aware teacher mix").

### ⏭ Next action, in order (2026-10-03, after 00:00 UTC)

1. **P-60 part 2** (authorised): `kaggle quota` must show ≈ 30 h left, then
   ```bash
   cp artifacts/p60_part2_kernel-metadata.json kaggle/rsna-knee-train/kernel-metadata.json
   python src/nbgen.py artifacts/train_p60_kaggle_part2.py kaggle/rsna-knee-train/rsna-knee-train.ipynb
   grep -E '^(FORCE_SMOKE|PARALLEL_ARMS|TEACHER_TABLES|TEACHER_MIX) = ' artifacts/train_p60_kaggle_part2.py  # False / ("v11n", "v11n2") / ("raptor_teacher",) / 0.5
   kaggle kernels push -p kaggle/rsna-knee-train
   ```
   ≈ 3.0 h. Pull `--file-pattern "(v11n|v11n2)_fold0_(best\.pt|oof\.csv)$|\.log$"` into `artifacts/kaggle_out/p60_part2/`.
   **Green** = each child log has `resume: copied v11n*_fold0_last.pt into WORK`, `resumed fold 0 at epoch 6 (best 0.91xx at epoch 5)`,
   epochs 6–11, `SWA of last 3`, `-> v11n*_fold0_best.pt = SWA`, no `runtime guard`. **Red:** `fold 0 epoch 0` = the resume did not
   happen (traps 31) — the arms then train all 12 epochs from scratch (≈ 5.5 h under the 8.3 h guard: still valid members, but
   ≈ 2.5 GPU-h more); `ERROR` with 0 files = traps 44, re-push the same build.
2. **Ship + two solos:** `rsna-knee-ckpt-v11n` / `-v11n2` (the 2026-09-29 ship loop, `kaggle datasets create`), add both to
   `kaggle/rsna-knee-infer/kernel-metadata.json`, placeholders from the staged `artifacts/infer_solo_v11n.py` / `infer_solo_v11n2.py`
   (`nbgen` → push → check `smoke False`, `infer members (1): v11n/fold0`, `constant labels 0`), submit both, watch with
   `src/watch_submission.py`. **Read (P-60):** m = mean vs 0.9305: ✅ m ≥ 0.935 / 🔁 0.926 < m < 0.935 / ❌ m ≤ 0.926. `/update`.
3. **P-62 real run — only with Tian's go:** after part 2 is green (so `rsna-knee-folds` v11 is no longer needed), push
   `artifacts/train_p62_real.py` (= the green v38 smoke build + `FORCE_SMOKE = False`, grep: `PARALLEL_ARMS = ("v11s", "v11s2")`,
   `TEACHER_TABLES = ("raptor_teacher",)`, `TEACHER_SILENT_MIX = 0.75`) to either slug with the c03-only metadata; ≈ 3.5 h. Green =
   child logs `P-62: report-silent cells (pilkwang UNK) mix at 0.75`, `reseeded 43 for arm v11s2`, `SWA of last 3`, no guard; ship as
   `rsna-knee-ckpt-v11s` / `-v11s2`; two solos, m vs 0.9305 with the same band as P-60.
4. If P-60 ✅ — P-62 re-reads on the P-60 recipe (card "Measure"); if both ❌ — the recipe and label lines close, P-45 is what is left.
5. **P-45** (second image teacher) — mapped: **D4 alone** (card P-45 has weights, patches, cost). First, 0 GPU: find D4's gold-58
   predictions (reviewer B's 0.9346 analog source; likely inside `mattiaangeli/rsna-knee-coatnet-d4-depthzone-swa3-b2`), score
   0.5 LLM + 0.25 Raptor + 0.25 D4 on gold by the P-49 method + the D4~Raptor within-class ρ. Then (Tian's go) the builder
   (≈ 150–250 lines, generalising `src/build_teacher_pass.py`), D4 over the 58 gold (must reproduce 0.9302; = the timing spike), the
   full pass (est. ≈ 2–3 GPU-h, inferred).

### Open decisions for Tian

- **P-62 real run** (≈ 3.5 GPU-h + 2 submissions) — and whether it waits for part 2 (default, safe) or runs in parallel from a third
  slug (saves ≈ 3 h wall-clock; `rsna-knee-folds` must not get a new version while part 2 could still need re-running).
- **P-45 go/no-go** — the biggest remaining lever by the evidence, but ≈ 8–9 GPU-h + a builder; the week from 10-03 has 30 h.
- Unchanged from 09-29: final selection (P-50, by 2026-10-15), the Efficiency formula (P-18, browser).

### Things that will bite if forgotten

- **`kaggle/rsna-knee-train/kernel-metadata.json` is c03-only now** (the P-62 smoke) — part 2 *must* copy the staged metadata first,
  or it will not mount `rsna-knee-folds` and will start from epoch 0 (traps 31).
- **Never push `rsna-knee-folds` before part 2 has finished green** — `kernel_sources` reads its latest version.
- **`ok  arm` ≠ finished** (traps 47) — check the child log for SWA before shipping anything.
- The kaggle CLI in this shell is `.venv/Scripts/kaggle.exe` (not on PATH); Git Bash heredocs still mangle backslashes (a scripted
  doc replace failed on `\n` this session — use the Edit tool).

## 2026-09-29 (11:30 → 14:30 UTC) — C ‖ D pulled green; cross-fit table → gate OPEN → **session E running**; 5 reads: **c03 `v11a` 0.932 / pair 0.932 = best solo** (🔁), trio 0.931 (🔁), 5-fold XF 0.928 (❌); ResNet-34 ❌ on gold

Tian: "read the handoff … continue work, pull results and submit whatever is necessary"; then asked why it was not done (E still
training) and whether the laptop can be turned off (yes — E runs on Kaggle; nothing is submittable before 00:00 UTC). Commits
`d9bcf00` … (this one). Authority for everything run: the approved 2026-09-28 plan (`docs/superpowers/plans/2026-09-28-single-model-plan.md`).

### ⏳ Still in flight as this was written (14:30 UTC)

| In flight | What it is | Started | How to check | How to read it |
|---|---|---|---|---|
| **Session E = `rsna-knee-train` v35** (P-55, REAL) | `PARALLEL_ARMS = ("v09o", "v09o2")`, `TEACHER_TABLES = ("raptor_teacher", "xfit_v09k")`, `TEACHER_MIX = 0.75` → targets 0.25 LLM + 0.375 Raptor + 0.375 cross-fit OOF; the `v09r` recipe, all 4,349, seeds 42 / 43 (`artifacts/train_student_E.py` = `src` + 4 seds) | pushed 11:37:33 UTC, RUNNING at once; expect COMPLETE ≈ 14:30–15:00 UTC (≈ 2.8–3.0 h, like `v09u`) | `kaggle kernels status tiankljucanin/rsna-knee-train`; then `kaggle kernels output tiankljucanin/rsna-knee-train -p artifacts/kaggle_out/student_E --file-pattern "(v09o\|v09o2)_fold0_(best\.pt\|oof\.csv)$\|\.log$"` | **Green** = parent `ok  arm v09o` + `ok  arm v09o2`; each child log: `teacher table raptor_teacher: 4349` **and** `teacher table xfit_v09k: 4407`, `training targets = (1 - 0.75) * LLM + 0.75 * quantile-matched ['raptor_teacher', 'xfit_v09k']`, `reseeded 43 for arm v09o2`, `SWA of last 3`, `-> v09o*_fold0_best.pt = SWA`, no `runtime guard`; `_best.pt` ≈ 164,789,xxx bytes. **Red flags:** `ERROR` with 0 files (traps 44 → re-push the same build); a missing `xfit_v09k` line (then the targets are wrong — do not ship). Gold-58 SWA is direction only (`v09r` 0.9093, c03 `v11a` 0.9204) |

Nothing else is running (all five submissions scored; watchers done).

### Where things stand

| | Status |
|---|---|
| Best LB | 0.942 (fork); **best solo 0.932** — #30 `v11a` alone and #32 the c03 pair `v11a` + `v11b` (was #26 0.930) |
| Today's reads | #29 P-52 trio 0.931 🔁 (P-52 closed, the pair stays) · #30 `v11a` 0.932 / #31 `v11b` 0.929 → c03 m = 0.9305 🔁 (bar 0.9315) · #32 c03 pair 0.932 vs c02 pair 0.930 🔁 · #33 5-fold XF 0.928 ❌ — experiments.md 2026-09-29 "Submissions #29–#32" (+ addendum) |
| Training done | C re-push v34 (`v09k4` + `v13a`, 2.39 h) ‖ D v10 (`v11a` ‖ `v11b` on c03, 3.53 h) — experiments.md 2026-09-29 "Sessions C ‖ D"; **ResNet-34 `v13a` gold 0.8306 = ❌ on gold** (not submitted) |
| P-55 table | `artifacts/teacher/xfit_v09k.csv` (4,407 rows, per-fold ranked) gold-58 0.9028 vs LLM 0.8948, 4/12 below → loose gate OPEN — experiments.md 2026-09-29 "P-54 cross-fit table" |
| Datasets (private, ready) | `rsna-knee-teacher-tables` new version (+ `xfit_v09k.csv`, the two old tables md5-identical); `rsna-knee-ckpt-v09k` (5 ckpts), `-v11a`, `-v11b`, `-v13a` — all in `kaggle/rsna-knee-infer/kernel-metadata.json` |
| Committed renders | `rsna-knee-train` = **v35 REAL (session E)** — never re-push as is unless re-running E; `rsna-knee-infer` = v29 (the 5-fold XF solo, #33) |
| Staged builds | `artifacts/infer_solo_v09o.py`, `infer_solo_v09o2.py`, `infer_pair_v09o_v09o2.py` (each `src` + 3 seds, `INFER_MEMBERS` set) |
| Quota | Kaggle GPU 25.45 h used / **4.55 h left** at 14:30 UTC (E still counting; ≈ 4.4 h after it), reset 2026-10-03 00:00 UTC. Submissions: 0 left today, 5 after 00:00 UTC |
| Docs | ✅ all findings logged (experiments.md entries + Scoreboard + Submissions rows 29–33; proposals.md P-52 closed, P-54 / P-55 / P-56 / P-57 statuses + index; CLAUDE.md state line) |

### What we talked about and decided

- **Session E pushed without a Kaggle smoke** — `src` has not changed since the green smoke v31; E is a sed-only build (plan Step 2:
  "every later push is a sed-only build"). A local CPU smoke of the exact build went first (both tables read, both arms SWA, exit 0),
  and a missing table is fatal at start (loud, minutes).
- **`v13a` was not given a solo slot** — gold −0.079 is beyond the 0.05 floor with the structure labels at 0.75–0.77; the five slots
  went to the pre-registered P-52 / c03 / XF reads. Its Dataset is mounted if a spare slot ever appears.
- **Pre-registered bands were applied as written** even where every c03 reading points one way (m = 0.9305 is 0.001 under ✅):
  c03 is 🔁, not the production input by rule.

### What we figured out

1. **c03 (dense slices, 150 mm) is our best single-model input so far, but only by the sum of small signs** — `v11a` 0.932, `v11b`
   0.929, pair 0.932 vs the c02 twins 0.927 / 0.927 and pair 0.930; gold +0.014 with 10/12 labels up (experiments.md 2026-09-29).
2. **More fold models do not replace more data** — the 5-fold cross-fit ensemble (each on 80 %) reads 0.928 ≈ one all-data member.
3. **A third same-input member reads like a seed** — the P-52 trio 0.931 (+0.001) at 1.4× inference and ≈ 54 min to score.
4. **Scoring times:** a c02 solo 28–32 min, the c03 pair ≈ 46 min, the trio ≈ 54 min, five members ≈ 66 min.

### ⏭ Next action, in order

1. **Pull session E** (command in the in-flight table) → check the green criteria → gold-58 SWA of both (direction only) → ship:
   ```bash
   for a in v09o v09o2; do rm -rf artifacts/ship_$a; mkdir -p artifacts/ship_$a
     cp artifacts/kaggle_out/student_E/${a}_fold0_{best.pt,oof.csv} artifacts/ship_$a/
     echo "{\"title\": \"RSNA knee ckpt $a\", \"id\": \"tiankljucanin/rsna-knee-ckpt-$a\", \"licenses\": [{\"name\": \"other\"}]}" > artifacts/ship_$a/dataset-metadata.json
     (cd artifacts/ship_$a && kaggle datasets create -p .); done
   ```
   then add `tiankljucanin/rsna-knee-ckpt-v09o` and `-v09o2` to `kaggle/rsna-knee-infer/kernel-metadata.json` `dataset_sources`, wait
   for `kaggle datasets status … = ready`.
2. **Three placeholders, one at a time** (each ≈ 3 min): `python src/nbgen.py artifacts/infer_solo_v09o.py kaggle/rsna-knee-infer/rsna-knee-infer.ipynb`
   → `kaggle kernels push -p kaggle/rsna-knee-infer` → COMPLETE → pull the log (`--file-pattern "submission\.csv$"`) and check `smoke
   False`, `infer members (1): v09o/fold0`, `decode-once verified`, `constant labels 0`; note the version number; same for `v09o2` and
   the pair. After 00:00 UTC: `kaggle competitions submit rsna-knee-abnormality-detection -k tiankljucanin/rsna-knee-infer -v <N> -f
   submission.csv -m "<what>"` ×3, each watched by `python src/watch_submission.py --ref <ref> --every 60`.
3. **Read (P-55, pre-registered):** m = mean(`v09o`, `v09o2`) vs 0.927: **✅ m ≥ 0.9315 / 🔁 0.9285 ≤ m < 0.9315 / ❌ m < 0.9285**;
   the pair ✅ ≥ 0.935. A ❌ = the third null (P-38, Nicolai) → the OOF-target line closes. `/update` after the reads.
4. **Two spare submissions tomorrow** — Tian's call (below); otherwise leave them.

### Open decisions for Tian

- **What next with ≈ 4.4 GPU-h before the Oct 3 reset** (enough for one ≈ 3–3.5 h session). Options: (a) if the student ✅ →
  the student targets on the c03 input (a new arm `v11o`, needs a 1-line arm + smoke); (b) if the student ❌ → a third c03 seed to
  settle P-56 (m is 0.001 under the bar); (c) keep the quota as a buffer. The plan's session F was "seed twin of a KEEP winner" only.
- **The two spare slots tomorrow:** e.g. the 4-member c02 + c03 blend (`v09r` + `v09u` + `v11a` + `v11b`; gold 0.9182 < the c03 pair
  0.9207, so low prior), or `v13a` solo (dead on gold), or nothing.
- Still open from 09-28: the discussion-735304 questions (brainstorm.md); the second teacher after Oct 3.

### Things that will bite if forgotten

- **`rsna-knee-train` v35 is a REAL session-E render** — re-pushing it starts another ≈ 3 GPU-h run.
- **Kernel status right after a push can still show the previous version's COMPLETE** — always check the pulled log's `infer members`
  line before submitting a version.
- **Git Bash heredocs mangle backslashes** (again today) — use the Edit tool or a script file for doc patches with `\` in them.
- A training session can die from outside with 0 files (traps 44) — if E shows `ERROR`, re-push the same build.

## 2026-09-28 (09:12 → 12:12 UTC) — #22 = 0.942 (🔁); v30 shipped; **solo reads #24 `v09u` 0.927 / #25 `v09x` 0.929 / #26 pair 0.930**; #27 fork β 0.20 sent; the Kaggle "best single-model" thread mined → the **single-model plan** (probe #28, cross-fit A ‖ B running, c03 cache building, ResNet arm, P-49 done)

Tian: "read the handoff … record the scores, run /update and then submit the latest work"; then "retry with 20%"; then "improve our single
model scores … read the discussion … dynamic workflow … plan out implementation … 20h of gpu"; answered 4 questions (below); approved the
plan; "finish as soon as possible … /handoff". Commits `b2eda60` … (this one). **The plan:** `docs/superpowers/plans/2026-09-28-single-model-plan.md`
(steps, sessions, pre-registered reads); the thread synthesis: research.md 2.7.2.

### ⏳ Still in flight as this was written (12:12 UTC)

| In flight | What it is | Started | How to check | How to read it |
|---|---|---|---|---|
| **XF session A = `rsna-knee-train` v32** | P-54 cross-fit folds `v09k0` ‖ `v09k1` (v09r recipe, `train_all` False, `eval_final_only`, Raptor 0.5; `artifacts/train_xf_A.py`) | RUNNING 12:10 UTC | `kaggle kernels status tiankljucanin/rsna-knee-train`; when COMPLETE: `kaggle kernels output tiankljucanin/rsna-knee-train -p artifacts/kaggle_out/xf_A --file-pattern "(v09k[0-4]_fold[0-9]_(best\.pt\|oof\.csv))$\|\.log$"` | expect ≈ 14:40 UTC (≈ 2.45 h). Green = parent `ok  arm v09k0: … written (folds [0])` and `ok  arm v09k1: … (folds [1])`; each child log: `fold k: train ≈ 3,5xx / val ≈ 88x`, `teacher table raptor_teacher: 4349`, `(held-out eval deferred …)` on epochs 0–7, then `SWA of last 3` + `-> v09k*_fold*_best.pt = SWA`; no `runtime guard`. The OOF csv has ≈ 882 rows incl. 11–12 gold |
| **XF session B = `rsna-knee-folds` v9** | `v09k2` ‖ `v09k3` (`artifacts/train_xf_B.py`) | RUNNING 12:10 UTC | same, slug `rsna-knee-folds`, dir `artifacts/kaggle_out/xf_B` | same criteria, folds [2] / [3] |
| **#28 = the P-53 probe**, ref 56639910 | the #26 pair with ACL / MCL / PF OA / Lat Men = constant 0.5 (`rsna-knee-infer` v24) | sent 11:56:44 UTC | `kaggle competitions submissions … --csv \| head -3`; watcher task `b3jhyh0bj` → `artifacts/submission_timing.csv` (dies with the laptop) | mean4 = 0.5 + 3·(0.930 − probe) (± 0.003); mean8 = (11.16 − 4·mean4)/8. **mean8 − mean4 ≥ 0.03 → the structural deficit is real on the test** (the c03 arm is well aimed); mean4 ≥ mean8 − 0.01 → gold-specific (c03 still runs, lower prior). A probe ≈ 0.80 ↔ mean4 ≈ 0.89 |
| **#27 = fork v10 at β 0.20**, ref 56636712 | the public 0.942 graph + `v09r` at β 0.20 (P-40 retry, Tian's go) | sent 09:40 UTC | same; watcher `bdra1ljs2` (dies with the laptop) | vs 0.942: **≥ 0.947 ✅ / 0.940–0.946 🔁 / ≤ 0.939 ❌**; #12 (β 0.20, 0.87-level members) read 0.939 |
| **c03 cache** `rsna-knee-cache3-a..d` v1 (CPU, 0 GPU) | the c02 scheme at 24/24/24/14/8/8 slices, 150 mm crop → version `c02_p336_b24-24-24-14-8-8_band2-98_crop150_lat20` (≈ 13 GB/shard) | pushed ≈ 12:02 UTC | `kaggle kernels status tiankljucanin/rsna-knee-cache3-a` (…-d); log: `kaggle kernels output … --file-pattern "no_match"` | green = `cached N/N studies … decode failures 0` in each; ≈ 20–30 min. Needed before session D |
| **P-49 = `rsna-knee-teacher-gold` v1** | Raptor over the 58 gold studies | COMPLETE ≈ 12:08 | not pulled yet: `kaggle kernels output tiankljucanin/rsna-knee-teacher-gold -p artifacts/kaggle_out/teacher_gold --file-pattern "raptor_teacher_shard0\.(npz\|csv)$"`, then `python src/merge_teacher.py --expect-n 58 …` → a gold table (never a training table) | gold AUC of Raptor and of the 0.5 LLM + 0.5 matched-Raptor target (the reference the XF OOF is compared with — logged, not gating) |

### Where things stand

| | Status |
|---|---|
| Best LB | 0.942 (fork); **best solo 0.930** (#26 `v09r` + `v09u`, the production member); best single 0.929 (`v09x`) |
| Solo reads today | #24 `v09u` 0.927 = `v09r` → retrain spread ≈ 0, one-seed floor 0.004 (P-44 ✅); #25 `v09x` 0.929 → 🔁 (P-43); #26 0.930 → production (+0.003, under the floor) — experiments.md 2026-09-28 "Submissions #24–#26" |
| Code | ✅ one batch, all tests green (window_head, targets, teacher_pass, cache_selftest), local CPU smoke + Kaggle smoke v31 green (`v09k1` ‖ `v13a`: ok ×2, ResNet loads, deferred eval, SWA): `PROBE_CONST_LABELS`; arms `v09k0-4` / `v13a` / `v11a` / `v11b` / `v09o` / `v09o2`; `eval_final_only`; fold-agnostic parent check; `DISTILLED_MIX` guard; ResNet in `load_timm_backbone` / `param_groups`; cache overrides `SLOT_SLICES` / `CROP_MM`; `build_distill_table --per-fold-rank`; `build_teacher_pass --gold` |
| Datasets | `rsna-knee-ckpt-v09x`, `-v09u`, `timm-resnet34-a1` (private, ready); `rsna-knee-train` mounts the ResNet weights |
| Committed renders | `rsna-knee-train` = v32 REAL (XF A); `rsna-knee-folds` = v9 REAL (XF B); `rsna-knee-infer` = v24 = the PROBE (never resubmit it as a candidate — rebuild without the sed); `rsna-knee-fork` = v10 (#27) |
| Quota | Kaggle GPU ≈ 9.9 h used before A/B → ≈ 15.8 h left after A ‖ B (4.9 h); reset 2026-10-03 00:00 UTC. Submissions: 5/5 used today (#24–#28) |
| Docs | experiments.md: #22 entry, v30 entry, #24–#26 entry, gold/census diagnostic, rows 24–28; proposals.md: P-43/P-44 closed, P-52 card, index rows P-53..P-58 (**full cards not written — the plan file holds them**); traps 39 extended; research.md 2.7.2 |

### What we talked about and decided

- **Tian's answers (plan questions):** today's last submission → the per-label probe (not P-52); **input changes back in scope** (c03);
  the cross-fit runs with a **loose gate** (student runs unless the XF OOF is clearly P-38-shaped: pooled gold < 0.875 AND ≥ 8/12 labels below
  the LLM); extras = P-49 and a ResNet-34 arm; the local-CPU LLM relabel only as a future card (P-58); **no hosted-API labels**.
- **#27:** Tian asked for the β 0.20 retry after #22 = 0.942 — sent.
- **Rejected (thread + reviewers):** Qwen VLM, higher resolution as the lever, CoAtNet-2@384, masking noisy cells, hard labels, re-extraction,
  per-label weights tuned on gold, TTA, the D4 second teacher before the reset (Tian's 09-23 exclusion still stands — re-ask after Oct 3).
- **Sessions pair arms of ONE target spec** (session-level `TEACHER_TABLES` / `TEACHER_MIX`) — no per-arm teacher code (traps 40/42).

### What we figured out

1. **The single-model gap is visible on gold:** our gold→LB offset (+0.021) equals Scott's / Archit's, and every member trails its own LLM
   labels on ACL / MCL / PF OA / Lat Men while beating them on image-visible findings (experiments.md 2026-09-28 "Gold-58 per-label diagnosis").
2. **c02 throws away most of the slices:** fluid series ≈ 30 native at 3 mm, c02 keeps 12 on cor/ax (7.5–9 mm spacing); median FOV 160 mm vs
   the 130 mm crop → c03 = 24/24/24/14/8/8, 150 mm (same entry).
3. **Gold-58 had the LB direction wrong twice for same-recipe members** (#24, #26) — traps 39 extended.
4. **Heredocs in this Git Bash halve `\\`** — Python patches with `\n` / `\d` in strings must be written as files (scratchpad) and run.

### ⏭ Next action, in order

1. **Read #28 (probe) and #27** → `/update` (rows 27 / 28; the probe read by the rule in the in-flight table).
2. **Pull P-49** (command above) → `python src/merge_teacher.py` with `--expect-n 58` into `artifacts/teacher/raptor_gold.csv` (a name
   `TEACHER_PATHS` never reads); log Raptor's gold AUC and the 0.5/0.5 target's gold AUC (direction only).
3. **When A and B are COMPLETE:** pull both (commands above), check the green criteria, then push **session C** on `rsna-knee-train`:
   `PARALLEL_ARMS = ("v09k4", "v13a")`, `TEACHER_TABLES = ("raptor_teacher",)`, `FORCE_SMOKE = False` (sed-only build, no smoke needed —
   the code is smoke-tested). **Session D** on `rsna-knee-folds` once cache3-a..d are green: first edit
   `kaggle/rsna-knee-folds/kernel-metadata.json` → `kernel_sources` = the four `rsna-knee-cache3-*` (drop cache-a/-b and cache2-*, so c02
   and c03 are never both mounted), then `PARALLEL_ARMS = ("v11a", "v11b")`, same table, `FORCE_SMOKE = False`. Grep every build for
   `^(FORCE_SMOKE|PARALLEL_ARMS|ARM_ONLY|TEACHER_TABLES|TEACHER_MIX) =` before pushing.
4. **Tomorrow's submissions** (5): P-52 (`INFER_MEMBERS = ["v09r", "v09u", "v09x"]`, vs 0.930: ≥ 0.934 ✅ / 0.931–0.933 🔁 / ≤ 0.930 ❌);
   then as C / D land: `v13a` solo (vs 0.927: ≥ 0.932 ✅ / 0.923–0.931 🔁 / ≤ 0.922 ❌), `v11a`, `v11b`, the c03 pair (m = mean of the two:
   ✅ m ≥ 0.9315 / 🔁 0.9285 ≤ m < 0.9315 / ❌ m < 0.9285). Ship each as `rsna-knee-ckpt-<arm>` and add to `rsna-knee-infer`'s metadata.
5. **After all five folds:** `python src/build_distill_table.py --sets "artifacts/kaggle_out/xf_*/v09k[0-4]_fold[0-9]_oof.csv" --per-fold-rank
   --out artifacts/teacher/xfit_v09k.csv` (the five fold csvs are ONE set — check it has 4,407 rows); gold diagnostic; loose gate; a new
   version of the PRIVATE Dataset `rsna-knee-teacher-tables` with `xfit_v09k.csv`; then **session E** = `("v09o", "v09o2")` with
   `TEACHER_TABLES = ("raptor_teacher", "xfit_v09k")` and `TEACHER_MIX = 0.75` (the `DISTILLED_MIX` guard refuses anything else). Also the
   5-fold ensemble solo (`INFER_MEMBERS = ["v09k0", …, "v09k4"]`, vs 0.930: ≥ 0.935 ✅ / 0.931–0.934 🔁 / ≤ 0.930 ❌; ≈ 1–1.5 h to score).
6. **Write the full proposals.md cards P-53..P-58** from the plan file; `/update` after every read; `/handoff` at the end.

### Open decisions for Tian

- Post the thread questions (free): Scott (label-side or training-side fix?), CoolinLai (labels, gold score?), Archit ("50/50 jumped" = LB?
  final weight? all cells or silent ones?), Nicolai (were the predictions OOF?), tennogh (OOF pseudo-labels as targets?).
- After Oct 3: the second teacher (D4 exclusion), and a Friday quota-burn session F only if ≥ 3.5 h is left before the reset.

### Things that will bite if forgotten

- **`rsna-knee-infer` v24 is the PROBE** — its render has 4 constant columns; rebuild from `src` (without the `PROBE_CONST_LABELS` sed) before
  any real submission.
- **Session D must not mount c02 and c03 together** (input size unverified) — edit `rsna-knee-folds`'s `kernel_sources` first.
- **A smoke forces fold 0** — the smoke cannot show folds 1–4; check each real child log's `fold k: train N / val M` line.
- Watchers die with the laptop; queued Kaggle versions must never be re-pushed (traps 41).

## 2026-09-27 (19:20 → 20:20 local = 17:20 → 18:20 UTC) — four-reviewer audit → **`proposals.md` rewritten** (1,292 → 386 lines, 11 live cards); **traps 42** (per-arm `seed` / `teacher_mix` were inert; PARALLEL fallback trained the default arms) fixed; **`v09x` (320 px) ‖ `v09u` (seed 43) training = `rsna-knee-train` v30** (⏳); #22 still scoring

Tian: "brainstorm and read the suggested proposals … compare them to what is already implemented … scratch really old ideas …
spin up multiple review agents that compare their work and opinions … then pick a single best chance to improve the model and start
training it if possible — use kaggle gpu not runpod"; then "what do you want to submit now and why" (answered in chat); then "write
all of this information in docs and run /handoff so i can close my laptop". Commits `fae359b` … (this one). The 19:20 entry below
(same session) holds the #21–#23 reads and the #22 fill-in checklist.

### ⏳ Still in flight as this was written (20:20 local = 18:20 UTC)

| In flight | What it is | Started | How to check | How to read it |
|---|---|---|---|---|
| **`rsna-knee-train` v30** (P-43 + P-44, REAL) | `PARALLEL_ARMS = ("v09x", "v09u")`, `TEACHER_TABLES = ("raptor_teacher",)`, mix 0.5 (`artifacts/train_xu_real.py` = committed src + 3 seds). **cuda:0 `v09x`** = the `v09r` recipe at **img 320**, batch 1 × accum 4; **cuda:1 `v09u`** = `v09r` exactly, **seed 43** | pushed 18:08:24, **RUNNING 18:08:48 UTC** (no queue). Expect `v09u` ≈ 21:00 UTC (≈ 2.8 h), `v09x` ≈ 23:30–00:30 UTC (5.3–6.1 h est.); the kernel ends when both children end; child guard ≈ 8.0 h → hard stop ≈ 02:15 UTC | `kaggle kernels status tiankljucanin/rsna-knee-train` (output is unreadable while RUNNING, traps 12e; `kaggle kernels logs` returned nothing at +7 min). When COMPLETE: `kaggle kernels output tiankljucanin/rsna-knee-train -p artifacts/kaggle_out/train_v30 --file-pattern "(v09x\|v09u)_fold0_(best\.pt\|oof\.csv)$\|\.log$"` | **Green** = parent log `ok  arm v09x` + `ok  arm v09u`; each child log (`v09x.log`, `v09u.log`) shows `teacher table raptor_teacher: 4349 studies`, `SWA of last 3 EMA snapshot(s)` and `-> v09*_fold0_best.pt = SWA`; `v09u.log` has `reseeded 43 for arm v09u`; `v09x.log` has `img 320 \| batch 1 x accum 4` and its `train … s/study` (≤ 0.80 fits the guard). **Red flags:** `stopping: runtime guard` in `v09x.log` → its `_best.pt` is the last epoch, NOT SWA → **never submit it** (resume in `rsna-knee-folds`, which first needs `tiankljucanin/rsna-knee-train` in its `kernel_sources`, traps 31); a `_best.pt` much smaller than ≈ 157 MB → delete and re-pull (traps 38). Gold-58 per-epoch lines are direction only |
| **Submission #22 = `rsna-knee-fork` v9**, ref 56614068 | the public 0.942 graph + `v09r` alone at β 0.10 (P-40 step A) | sent 16:36:17 UTC; forks score in ≤ 8 h 06 min → ≈ 00:30–02:30 UTC | `kaggle competitions submissions rsna-knee-abnormality-detection --csv \| head -3`; watcher task `buiay90kx` dies with the laptop (then the time is only a bound) | **vs 0.942: ≥ 0.947 ✅ / 0.940–0.946 🔁 / ≤ 0.939 ❌** (P-40); the fill-in checklist is Next action 1 of the 19:20 entry below |

### Where things stand

| | Status |
|---|---|
| Best LB | **0.942** (#13 / #15); best member of ours `v09r` 0.927 solo (#20); #22 ⏳ |
| `docs/proposals.md` | ✅ **rewritten** by a four-reviewer, two-round audit: live cards ranked P-43, P-44, P-40, P-49, P-50, P-18, P-47, P-45, P-46, P-48, P-51; P-00…P-42 closed as one-line pointers (pre-rewrite text: `git show 8304c96:docs/proposals.md`); decision-metric table fixed (solo LB judges target changes; fork = final candidates only) |
| P-43 / P-44 | ⏳ running (v30); smoke v29 green — `v09x` peak **7.03 GiB** at 320 px, `v09u` 6.84 GiB, both read the Raptor table, `ok  arm` ×2 (experiments.md Scoreboard 2026-09-27 "P-43 + P-44") |
| Code | ✅ `load_timm_backbone(img_size)`; per-arm reseed; `PARALLEL_ARMS` narrows `ARMS` / `PRIMARY_ARM`; guards: an arm dict setting `teacher_mix` / `teacher_tables` is refused, a sed'd table with a non-distilled session arm is refused (traps 42); `window_head_test.py` builds CoAtNet-1 at 320 (all green); local CPU smoke of the exact build green |
| Docs also | traps 42; experiments.md "Four-reviewer audit … gold-58 diagnostics" (paired bootstrap, within-class ρ, teacher-mix analogs, pilkwang `oof.npz`, Raptor-vs-dread) + **CORRECTED**: gold rows are held out under `train_all` |
| Committed renders | `rsna-knee-train` = **v30 REAL** (`FORCE_SMOKE = False`, never re-push as is); `rsna-knee-infer` = v20 (P-42 pair); `rsna-knee-fork` = v9 (= #22) |
| Quota / money | Kaggle GPU 3.42 h used / 26.58 h left at 18:16 UTC → ≈ 20.4 h after v30 (reset 2026-10-03 00:00 UTC); submissions: 5 fresh after 00:00 UTC; RunPod: none this session (`list-pods` empty 17:18 UTC); Kaggle token valid to **2026-09-28 00:48 UTC** |
| Repo | `main` pushed; `rsna-knee-speedy-raptors-coatnet-d4-0943.ipynb` still untracked on purpose |

### What we talked about and decided

- **Audit method:** four read-only reviewer agents with different lenses (auditor, strategist, skeptic, Kaggle feasibility), then a
  cross-critique round on a written digest; the rewrite itself was delegated to a doc agent from a written spec and reviewed here (one
  over-read fixed: "the teacher, not the architecture, sets the errors" → the shared teacher *raised* cross-family agreement, 0.883 vs
  0.861 says architecture still matters).
- **The pick, and why this pair:** round 1 split (mix 1.0 ×2, 320 px + twin, two seed twins); round 2 converged 4/4 on `v09x` ‖ `v09u`.
  Mix 1.0 dropped (the gold-58 analogs: image-only −0.004…−0.018 vs 50/50, and it is not "Raptor-only" anyway); a measurement-only
  2-seed session rejected (the ask was to *improve*; a lever + twin gives both reads); CoAtNet-2 @384 rejected on Kaggle (9–15 GPU-h,
  OOM at batch 2); both arms share one table + mix, so no per-arm target code was written.
- **Kaggle GPU, not RunPod** (Tian) — one ≈ 6.2 GPU-h session; the second T4's arm rides free inside the longer one's wall-clock.
- **Submissions of the three reads were asked for, not yet explicitly given** ("Ok, write all of this in docs") — confirm next session.

### What we figured out

1. **Every production member trained on the seed-42 stream** — an arm-dict `seed` changed only the banner; `teacher_mix` in an arm dict
   would have been recorded and ignored; the PARALLEL fallback trained `v09a` / `v08a` (traps 42, fixed).
2. **Gold-58 is held out for production members** (`split_studies` trains `is_gold == 0` only) — two old "optimistic by construction"
   lines CORRECTED; paired gold bootstrap SD ≈ 0.007 (experiments.md "Four-reviewer audit").
3. **#18 vs #20 (the +0.009) was cross-platform** (Kaggle T4 / 2 workers vs RunPod / 8) — why `v09u` is on Kaggle.
4. **320 px fits a T4 at batch 1:** 7.03 GiB peak with all 24 windows; timm loads the 224 weights strictly at 320.

### ⏭ Next action, in order

1. **Read #22** and `/update` it — the fill-in checklist is Next action 1 of the 19:20 entry below.
2. **When v30 is COMPLETE:** pull (command in the in-flight table), check the green criteria. Record gold-58 SWA of both arms
   (direction only) in `/update` (Scoreboard row "P-43 + P-44").
3. **Ship both** (from the repo root, Git Bash):
   ```bash
   export PYTHONUTF8=1; KG=$PWD/.venv/Scripts/kaggle.exe
   for a in v09x v09u; do d=artifacts/ship_$a; rm -rf $d; mkdir -p $d; cp artifacts/kaggle_out/train_v30/${a}_fold0_{best.pt,oof.csv} $d/
     printf '{"title": "RSNA knee ckpt %s", "id": "tiankljucanin/rsna-knee-ckpt-%s", "licenses": [{"name": "other"}]}\n' $a $a > $d/dataset-metadata.json
     (cd $d && $KG datasets create -p .); done          # run from inside the dir (traps 21); then `datasets status` = ready
   ```
   Add `tiankljucanin/rsna-knee-ckpt-v09x` and `tiankljucanin/rsna-knee-ckpt-v09u` to `kaggle/rsna-knee-infer/kernel-metadata.json`
   `dataset_sources` (never mount `rsna-knee-train`'s output directly — same file names, latest version).
4. **Three infer placeholders, in this order** (one slug, one version each; `N` = the member list):
   ```bash
   sed -e 's/^FORCE_SMOKE = True/FORCE_SMOKE = False/' -e 's/^MODE = "auto"/MODE = "infer"/' \
       -e 's/^INFER_MEMBERS = \[.*\]/INFER_MEMBERS = ["v09u"]/' src/kaggle_pipeline.py > artifacts/infer_solo_v09u.py
   .venv/Scripts/python.exe src/nbgen.py artifacts/infer_solo_v09u.py kaggle/rsna-knee-infer/rsna-knee-infer.ipynb
   timeout 90 .venv/Scripts/kaggle.exe kernels push -p kaggle/rsna-knee-infer          # then ["v09x"], then ["v09r", "v09u"]
   ```
   Placeholder green = `smoke False`, `infer members (n): …/fold0 … [epoch 7, score …, ema True]`, `decode-once verified`, `constant
   labels 0`; for `v09x` also `img_size 320` in the timm line.
5. **Submit on Tian's go** (`kaggle competitions submit rsna-knee-abnormality-detection -k tiankljucanin/rsna-knee-infer -v <v> -f
   submission.csv -m "..."`), each followed by `python src/watch_submission.py --ref <ref> --every 60` in the background (≈ 30 min;
   keep the machine awake). **Order `v09u` → `v09x` → `v09r` + `v09u`.** Reads (P-43 / P-44): s = |`v09u` − 0.927| (≤ 0.920 → re-open
   P-39); `v09x` ✅ if ≥ m + max(0.005, 2s) with m = mean(0.927, `v09u`), ❌ if ≤ m − max(0.005, 2s); blend ≥ 0.930 ✅ (it becomes the
   production member either way). `/update` after each.
6. **Then by the cards:** `v09x` ✅ → read `v09x` + `v09r` (+ `v09u`) as a blend; P-49 (Raptor over gold-58, ≈ 0.2 h) is the cheapest next
   Kaggle job; P-45 / P-47 wait for the floor from P-44. `/handoff` at the end.

### Open decisions for Tian

- **Go for the three solo submissions** (`v09u`, `v09x`, `v09r` + `v09u`) — asked, not explicitly given.
- **If #22 reads 🔁:** the pre-registered β 0.20 retry, or drop fork member reads (reviewers B / C) and keep the fork for finals (P-50)?
- **P-49** (Raptor over the 58 gold studies, ≈ 20 lines + 0.2 h T4) next in the second slot?
- Final selection / publishability (P-50: never #13 + #15 together; never publish `rsna-knee-teacher-tables`), RadImageNet licence.

### Things that will bite if forgotten

- **Every background watcher dies with the laptop** (v30 completion, #22 timing, the 35-min log check) — the runs themselves continue.
- **Token expires 2026-09-28 00:48 UTC**, inside both v30's and #22's finish window: CLI calls fail until the first call ≥ 30 min after
  expiry (traps 20).
- **A guard-stopped arm is not SWA** — grep `SWA of last 3` before shipping; never submit one.
- **`rsna-knee-train`'s committed render is the REAL v30** — any new edit needs a `FORCE_SMOKE = True` push first; `rsna-knee-infer`'s is
  v20 (the P-42 pair) — re-render before each solo.
- **Two arms of one session share `TEACHER_TABLES` and `TEACHER_MIX`** (traps 42); a mix-0.75 arm (P-47) needs per-arm code first.

## 2026-09-27 (18:24 → 19:20 local = 16:24 → 17:20 UTC) — **#21 `v08r` solo = 0.918** (not a fork member); fork v9 left the queue after ≈ 3 h 40 min → **#22 sent** (⏳, read vs 0.942); **#23 P-42 `v09r` + `v08r` = 0.927 = `v09r` alone** (🔁); **P-41 ✅ — a solo scores in 28–29 min**

*Written before #22 scored, at Tian's request ("write the handoff now … and prepare to put in the #22 score") — the fill-in checklist is Next action 1.*

Tian: "read handoff and continue working, the latest submission has been rated on kaggle, but our knee fork is still queued"; then
**"Yes, run P-42 (Recommended)"** (asked with the gold-58 prior stated); then "what are we testing with #22 and #23" (answered in
chat); then this handoff. Commits `6714251` … (this one).

### ⏳ Still in flight as this was written (19:20 local = 17:20 UTC)

| In flight | What it is | Started | How to check | How to read it |
|---|---|---|---|---|
| **Submission #22 = `rsna-knee-fork` v9**, **ref 56614068** | P-40 step A: the public 0.942 graph (cells 0–49 verbatim) + our arm = **`v09r` alone** (Raptor-distilled CoAtNet-1, solo 0.927) at **β 0.10**. Placeholder green: `status beta0.10`, `members [v09r]`, rc 0 (104 s), `score 0.9093`, anchor sha = submission sha `7c6dfe8b…` (expected on 3 studies) — `artifacts/kaggle_out/fork_v9/` | sent **16:36:17 UTC** after ≈ 3 h 40 min QUEUED (pushed 12:48); forks have scored in ≤ 8 h 06 min (#17) → expect **≈ 00:30–02:30 UTC** (02:30–04:30 local) | `kaggle competitions submissions rsna-knee-abnormality-detection --csv \| head -3`; background watcher `watch_submission.py --ref 56614068 --every 120 --max-h 14` (task `buiay90kx`) appends the exact time to `artifacts/submission_timing.csv` — **it dies if the laptop sleeps** (then the time is only a bound) | **vs #13 / #15 (0.942), pre-registered in P-40: ≥ 0.947 ✅ our member counts in the stack / 0.940–0.946 🔁 / ≤ 0.939 ❌ our arm hurts.** Also compare #17 (0.941, 0.918-level members at the same β). Honest prior: 0.942 ± 0.001 (🔁). A read exactly 0.942 is *not* a sign the arm was skipped — the placeholder shows it ran (`members [v09r]`, rc 0) |

### Where things stand

| | Status |
|---|---|
| Best LB | **0.942** (#13 / #15); best member of ours `v09r` **0.927** solo (#20) — and the `v09r` + `v08r` pair is also 0.927 (#23) |
| P-40 | ⏳ `v08r` solo **0.918** (#21, −0.009 vs `v09r`, under the ≈ 0.920 bar → not a fork member); step A = **#22 ⏳** (experiments.md 2026-09-27 "Submission #21") |
| P-41 | ✅ **KEEP** — #23 (two members, 8 decode workers) scored **28.3–29.3 min** after sending vs #19's ≤ 42 min bound (experiments.md "Submission #23") |
| P-42 | 🔁 **INCONCLUSIVE** — `v09r` + `v08r` flat rank-mean **0.927 = `v09r` alone**; gold-58 had predicted −0.001 (ρ 0.924 vs 0.874 on LLM targets) → a second backbone on the same Raptor table is not the lever (experiments.md "Submission #23") |
| Committed renders | `rsna-knee-infer` = **v20** (`v09r` + `v08r`, = #23, submittable); `rsna-knee-fork` = **v9** (= #22); `rsna-knee-train` = `v09r` smoke render (never pushed); teacher kernels done, never re-push |
| Quota / money | Kaggle GPU **3.25 h used / 26.75 h left** (reset 2026-10-03 00:00 UTC); **submissions: 3 used today (#21–#23), 2 left** until 00:00 UTC; **RunPod: no pod this session, `list-pods` empty (17:18 UTC)**; Kaggle token valid to **2026-09-28 00:48 UTC** |
| Repo | `main` pushed; `rsna-knee-speedy-raptors-coatnet-d4-0943.ipynb` in the root still untracked on purpose |

### What we talked about and decided

- **`v08r` judged by the pre-registered ≈ 0.920 bar, strictly:** 0.918 misses it by 0.002 (sub-floor), but the rule written before the
  read decides — `v08r` is not added to the fork on solo strength.
- **P-42 run instead of waiting idle** (Tian's go): today's spare submissions do not carry over, the placeholder cost 0.1 h of T4, and
  one submission read two things (the pairing question and P-41's timing). The gold-58 prior (🔁) was stated before asking.
- **Rejected:** a weighted `v09r` 0.75 / `v08r` 0.25 solo — a sub-floor bet on the reading that `v08r` roughly pays for its weakness.
- **Handoff before #22 scores** (Tian) — overrides the usual "handoff only after results"; the read is staged below instead.
- **Corrected my own timestamp slip:** the session's first check was 16:24 UTC, not 14:24 (Git Bash `date -u` prints UTC with a 12 h
  clock) → #21's bound is ≤ 2 h 43 min (commit `013f41f`).
- **No AI attribution** in commits (the organisation's managed policy).

### What we figured out

1. **On the same Raptor targets DINOv2-S is ≈ 0.009 behind CoAtNet-1** (#21 0.918 vs #20 0.927; gold-58 −0.011 said the same) —
   experiments.md "Submission #21".
2. **A shared teacher makes the two families non-additive** (#23 = 0.927 = `v09r`); on LLM targets every new family had lifted the
   blend. The next gain must come from the targets or the input, not a third backbone — experiments.md "Submission #23".
3. **A solo read is now ≈ 30 min send-to-score** (28.3–29.3 min, two members) — P-41 ✅; four member reads fit in two hours.
4. **Kaggle's GPU queue is not first-in-first-out per account:** the fork waited ≈ 3 h 40 min while `rsna-knee-infer` v19, pushed
   49 min later, started within a minute (traps 41, second instance).

### ⏭ Next action, in order

1. **Read #22 and log it** (the fill-in checklist — every anchor below exists in the files now):
   ```bash
   export PYTHONUTF8=1
   .venv/Scripts/kaggle.exe competitions submissions rsna-knee-abnormality-detection --csv | head -3
   cat artifacts/submission_timing.csv          # a 56614068 row = exact scoring time; none = watcher died -> bound = sent 16:36:17 -> first read
   ```
   Then `/update`:
   - **experiments.md Submissions row 22** (`grep -n "^| 22 | 2026-09-27" docs/experiments.md`): replace its `| ⏳ |` with the score and
     prefix the notes with `**read <time> UTC → <verdict> vs 0.942 (P-40)**; scored within … (timing)`.
   - **Scoreboard:** a new row after the "P-42 solo blend, submission #23" row: `| 2026-09-27 | **P-40 step A, submission #22 (rsna-knee-fork v9)**: the public 0.942 graph + v09r alone at β 0.10 | gold-58 of v09r 0.9093 | **<score>** | <verdict> |`.
   - **New entry** `### 2026-09-27 — Submission #22: …` directly **before `## Infrastructure`** (after "Submission #23"): a table
     #15 anchor 0.942 / #13 (β 0.10, `v08w` + `v09h`) 0.942 / #17 (β 0.10, 0.918-level members) 0.941 / **#22 (β 0.10, `v09r`)**; the
     verdict by the rule; what it says about the P-40 "Against" point (Raptor already inside the anchor at 0.40–0.60 weight).
   - **proposals.md P-40:** card `Status:` (anchor: `→ **submitted as #22** (ref 56614068, 16:36 UTC) ⏳`) and the index row (anchor:
     `**#22 sent** (ref 56614068, 16:36 UTC) ⏳ → read vs 0.942`) → the verdict; on ✅ / ❌ reduce the card to a pointer ("History below").
   - **CLAUDE.md state line** (anchor: `**submitted as #22** (16:36 UTC) ⏳`) → the score; bump `State as of`.
2. **Then by the P-40 rule:**
   - **✅ ≥ 0.947** — our member counts: #22 becomes a final-selection candidate beside #13 / #15 and the #16 hedge. Next = P-40 "If it
     works": write a card for a **mix-1.0 arm** (the `v09r` recipe on Raptor-only targets: `TEACHER_MIX = 1.0` sed + a new version name
     in `SHIPPED_ARMS` / `DISTILLED_ARMS`, traps 40), train on a RunPod 4090 (≈ 48 min, ≈ $0.8, memory `runpod-pod-self-service`),
     read it solo (≈ 30 min) vs 0.927, and only then in the fork.
   - **🔁 0.940–0.946** — P-40 "If it fails": try a heavier vote **once** (recommended β 0.20: at 0.10 a 0.927 member was not
     readable, and #12's −0.003 at 0.20 was with 0.87-level members):
     ```bash
     export PYTHONUTF8=1 PYTHONPATH=src
     .venv/Scripts/python.exe src/build_fork.py --members v09r --member v09r=tiankljucanin/rsna-knee-ckpt-v09r:tiankljucanin/timm-coatnet-rmlp-1-rw-224 --beta 0.20
     .venv/Scripts/python.exe src/build_fork.py --members v09r --member v09r=tiankljucanin/rsna-knee-ckpt-v09r:tiankljucanin/timm-coatnet-rmlp-1-rw-224 --beta 0.20 --check
     grep -E '"machine_shape"|rsna-knee-ckpt-v09r' kaggle/rsna-knee-fork/kernel-metadata.json
     timeout 90 .venv/Scripts/kaggle.exe kernels push -p kaggle/rsna-knee-fork       # may queue for hours (traps 41) -- never re-push a queued version
     ```
     Placeholder green = `status beta0.20`, `members [v09r]`, rc 0, `score 0.9093`, anchor sha = submission sha (as in fork v3 = #12
     at β 0.20 on the 3 placeholder studies). Same read bands vs 0.942. Needs Tian's go to submit.
   - **❌ ≤ 0.939** — our arm hurts even at 0.927 (its errors are the anchor's Raptor errors): keep `v09r` as a solo-side asset; the fork
     finals stay #13 / #15 + #16; go to the target-diversity question (Open decisions).
3. `/handoff` at the end of the next session.

### Open decisions for Tian

- **Target diversity without waiting for a fork ✅?** The 09-26 plan gated the mix-1.0 arm and a second teacher on a fork ✅; P-42 now says
  that is where the next gain has to come from regardless, and a solo read costs 30 min. Relax the gate (card first, then a pod run)?
- **β 0.20 retry if #22 reads 🔁** — recommended above; the alternative is to stop spending fork submissions on member reads.
- **Final selection** (entries close 2026-10-15, final 2026-10-22): #13 / #15 (0.942) + the #16 hedge (0.940) today. Final assets must
  be **public** Datasets (`rsna-knee-ckpt-v09r`, `-v08r`, `rsna-knee-teacher-tables` are private).
- `v08a` vs `v08r` ("submit the v08a too", 09-27) — never confirmed; #21 was `v08r`. RadImageNet licence — unchanged.

### Things that will bite if forgotten

- **The watcher needs the machine awake** — #21's timing was lost when the laptop went off; #22's watcher (`buiay90kx`) has the same risk.
- **Kaggle token expires 2026-09-28 00:48 UTC**, right inside #22's likely scoring window: every CLI call fails until the first call
  ≥ 30 min after expiry, which refreshes it (traps 20). The watcher retries API errors, so it survives; a manual read at 00:50 will not.
- **Timestamps:** Git Bash `date -u` prints UTC in a 12 h clock ("4:24:19 PM" = 16:24 UTC); local is UTC+2. Stamp docs in UTC.
- **`rsna-knee-infer` v20 is the P-42 pair, v19 `v08r` alone, v18 a SMOKE** — re-render before any new solo (`sed` `INFER_MEMBERS`).
- **In the Bash tool, `cat > file` without a heredoc reads stdin and hangs** until the 2-min timeout (cost one background task today).

## 2026-09-27 (09:45 → 15:45) — the "0.943 Speedy Raptors" notebook is not faster (our anchor + 2 CoAt readers; "< 30 min" = its 3-study commit run); **P-41** faster solo scoring shipped (byte-identical); **`v08r`** trained on a RunPod A100 (gold-58 0.8981, shipped, pod deleted); **#21 = `v08r` solo sent** (13:41 UTC); fork v9 (`v09r` alone, β 0.10) **still QUEUED, not submitted**

*Refreshed in place at 15:45 (same session): #21 sent, fork still queued, Tian closing the laptop.*

Tian: "read handoff, figure out the next step in testing; research what makes the speedy notebook fast … implement something that would
speed up scoring"; then **"Yes, continue with step A and spin up the pod"**; then **"Submit the v08a too, I will have to turn the computer
off soon"** (read as `v08r` solo — see decisions). Commits `d76b05f` … (this one).

### ⏳ Still in flight as this was written (15:45 local = 13:43 UTC)

| In flight | What it is | Started | How to check | How to read it |
|---|---|---|---|---|
| **`rsna-knee-fork` v9** (P-40 step A) | `build_fork.py --members v09r --member v09r=tiankljucanin/rsna-knee-ckpt-v09r:tiankljucanin/timm-coatnet-rmlp-1-rw-224 --beta 0.10` — the public 0.942 graph + `v09r` alone at β 0.10 | pushed 12:48 UTC; **still QUEUED at 13:43 UTC — NOT submitted** (Kaggle T4 capacity, traps 41 — never re-push a queued version; it runs on Kaggle with the laptop off, only the submit call needs a session) | `kaggle kernels status tiankljucanin/rsna-knee-fork`; when COMPLETE: `kaggle kernels output tiankljucanin/rsna-knee-fork -p artifacts/kaggle_out/fork_v9 --file-pattern "(fork_diagnostics\.json\|submission\.csv)$"` | Placeholder green = `fork_diagnostics.json` `status beta0.10`, `members [v09r]`, subprocess rc 0, `v09r/fold0 … score 0.9093`, anchor sha = submission sha (expected at β 0.10 on 3 studies). Then **submit** (Tian's go given): `kaggle competitions submit rsna-knee-abnormality-detection -k tiankljucanin/rsna-knee-fork -v 9 -f submission.csv -m "P-40 step A: public 0.942 graph + v09r (Raptor-distilled CoAtNet-1, solo 0.927) alone at beta 0.10; vs #13/#15 0.942"` and time it: `python src/watch_submission.py` (background). **Read vs 0.942: ≥ 0.947 ✅ / 0.940–0.946 🔁 / ≤ 0.939 ❌** (P-40) |
| **Submission #21 = `rsna-knee-infer` v19** (`v08r` solo), **ref 56610108** | `INFER_MEMBERS = ["v08r"]`, `FORCE_SMOKE = False`, `MODE = "infer"` (`artifacts/infer_solo_v08r.py`), Dataset `rsna-knee-ckpt-v08r` added to the metadata; placeholder green (`v08r/fold0 … [epoch 7, score 0.8981, ema True]`, `decode-once workers: 8 (cpus 4, usable 4)`, `decode-once verified`, `constant labels 0`) | pushed 13:37 UTC; **submitted 13:41:35 UTC** (PENDING at 13:43); `watch_submission.py --ref 56610108` was polling — it dies when the laptop is off, so the exact scoring time may be lost (then it is ≤ the next read) | `kaggle competitions submissions rsna-knee-abnormality-detection --csv \| head -3`; if `artifacts/submission_timing.csv` has a row for 56610108 the watcher caught the exact time | **Two reads:** (a) LB — there is no `v08a` solo baseline (#18 was the CoAtNet `v09a`); compare with `v09r` 0.927 (a DINOv2-S member ≥ 0.920 would make it a real second member for the fork); (b) **P-41 speed** — the first solo with 8 decode workers: **≤ 30 min ✅ / 30–42 🔁 / > 42 ❌** vs #19's ≤ 42 min bound (the watcher records it in `artifacts/submission_timing.csv`) |

### Where things stand

| | Status |
|---|---|
| Best LB | **0.942** (#13 / #15); best member of ours `v09r` **0.927** solo (#20) |
| P-40 | ⏳ step B ✅ run (`v08r` gold-58 SWA **0.8981** vs `v08a` 0.8850, direction only; Dataset `rsna-knee-ckpt-v08r` ready); step A = fork v9 queued; `v08r` solo = infer v19 running (experiments.md 2026-09-27 "`v08r`") |
| P-41 | 🔧 shipped: 16-thread header scan + `Config.infer_workers` (8 on the 4-CPU T4 ×2 box) for decode-once; local + Kaggle v18 byte-identical to v17; speed read = the `v08r` solo's scoring time (experiments.md Infrastructure 2026-09-27) |
| Public-notebook research | ✅ "0.943 Speedy Raptors CoAtNet D4" (haideptry) = our anchor's cells + Global96 + Repair-v1 readers; "< 30 min" = 3 studies in 238 s; not a lever (traps 35 addendum; 09-22 "superset" line CORRECTED) |
| Tools | `src/watch_submission.py` (new) — polls a submission, records sent → scored time |
| Committed renders | `rsna-knee-infer` = **v19** (`v08r` solo, submittable); `rsna-knee-fork` = **v9** (`v09r` alone, β 0.10); `rsna-knee-train` = `v09r` smoke render (never pushed); teacher kernels = done, never re-push |
| RunPod | pod `lz8k55zvczmlr7` (A100 SXM 80 GB, US-MD-1, $1.59/h) 12:49:54 → ≈ 13:26 UTC ≈ **$0.95**, **deleted; `list-pods` empty (re-verified 13:42 UTC)** — nothing billing |
| Quota | Kaggle GPU 3.08 h used / 26.92 h left before v19 / fork v9 (reset 2026-10-03 00:00 UTC); **submissions: 1 used today (#21), 4 left** (UTC day); Kaggle token valid to 2026-09-28 00:48 UTC (then the ≥ 30 min self-refresh, traps 20) |
| Repo | `main` pushed; the downloaded `rsna-knee-speedy-raptors-coatnet-d4-0943.ipynb` in the root is **untracked on purpose** (not asked to commit it) |

### What we talked about and decided

- **Speed:** the fork cannot be made materially faster without changing its score (the anchor graph is hours by construction: #17 ≤ 8 h 06 min;
  the Raptor branch alone is 5.1–7.5 s/study). The fast instrument is the solo submission (#19 ≤ 42 min), and it is also the one that reads
  member quality (#17). **Use solo reads to iterate; the fork only for final candidates.** Re-anchoring on the 0.943 notebook: rejected
  (≤ +0.001 claimed, two more serial CoAtNet-2 @384 readers).
- **P-41 over bigger refactors:** kept to one output-neutral change (threads + workers); decode/GPU overlap and two-GPU prediction deferred
  (≈ 5 min of a ≈ 40 min solo) — the P-41 "if 🔁" branch.
- **Pod GPU:** every 24/48 GB card read NONE on SECURE at 12:48 UTC → **A100 SXM 80 GB** ($1.59/h, the fallback lined up on 09-26) over the
  untested RTX PRO 4500 Blackwell ($0.72/h); `minCudaVersion 12.8`; cache on `/dev/shm` (MooseFS `/workspace`). 31 min job, ≈ $0.95.
- **Fork member set:** `v09r` alone (the handoff's recommendation: full β, cleanest read) rather than #17's set with `v09r` swapped in.
- **"Submit the v08a too" → `v08r` solo.** The question asked was about `v08r` (the new member); `v08a` was never retrained and its solo would
  only be a baseline. Flagged to Tian in chat; if `v08a` was meant, it is `INFER_MEMBERS = ["v08a"]` in the same infer slug (Dataset
  `rsna-knee-ckpt-v08a` is already in its metadata).
- **No `Claude-Session:` trailer** on this session's commits: the organisation's managed policy forbids AI attribution in git metadata.

### What we figured out

1. **"Sub-30-minute hidden-test inference" is a 3-study commit run** — the notebook's own `phase_events.jsonl` (238 s) and its `len(ids) <= 48`
   branch; its graph is a superset of our anchor, so it is slower on the hidden test (experiments.md Infrastructure 2026-09-27, traps 35).
2. **Our anchor has only two CoAt readers** (resgated + D4) — the 09-22 frontier entry's "Global96 … superset" line was wrong (CORRECTED).
3. **Kaggle's T4 ×2 box has 4 usable CPUs**; our infer decoded on 2 workers and scanned headers on 1 thread — now 8 / 16, byte-identical (P-41).
4. **No scoring time has ever been measured exactly** — every figure is "sent → read"; `watch_submission.py` fixes that from #21 on.
5. **`v08r` +0.013 on gold-58** over `v08a` with a flat tail from epoch 4 — the DINOv2 recipe also moves on the Raptor table (direction only).

### ⏭ Next action, in order

1. **Submit the fork (#22)** — it was not sent this session: `kaggle kernels status tiankljucanin/rsna-knee-fork` → COMPLETE → pull
   `fork_diagnostics.json` (green criteria in the in-flight table) → `kaggle competitions submit rsna-knee-abnormality-detection -k tiankljucanin/rsna-knee-fork -v 9 -f submission.csv -m "P-40 step A: public 0.942 graph + v09r alone at beta 0.10; vs #13/#15 0.942"`
   → `python src/watch_submission.py` in the background. If v9 is ERROR, read its log (`--file-pattern no_match`) before any re-push.
2. **Read #21 and then #22** (`kaggle competitions submissions rsna-knee-abnormality-detection --csv | head -4`): `v08r` solo vs `v09r` 0.927;
   P-41 timing from `artifacts/submission_timing.csv` (or, if the watcher died, the bound sent → first read); the fork vs 0.942 by the P-40 rule. `/update` (Scoreboard + Submissions rows, P-40, P-41 → experiments.md).
3. **Then, by P-40:** fork ✅ or 🔁 and `v08r` solo ≥ ≈ 0.920 → `build_fork.py --members v09r v08r --member v09r=… --member v08r=tiankljucanin/rsna-knee-ckpt-v08r --beta 0.10`
   (DINOv2 backbone = the Model, no second `:` part) → placeholder → submit; fork ❌ → keep `v09r` as a solo-side asset, try β 0.15–0.20
   once (P-40 "If it fails"). A mix-1.0 arm / two-teacher table only after a fork ✅ (write the card first).
4. `/handoff` at the end.

### Open decisions for Tian

- **`v08a` or `v08r`?** — confirm the solo submitted (v19 = `v08r`) was what "submit the v08a too" meant.
- **Final selection** (entries close 2026-10-15, final 2026-10-22): #13 / #15 (0.942) + the #16 hedge (0.940) today; a fork ✅ adds a candidate.
  All final assets must be **public** Datasets (`rsna-knee-ckpt-v09r`, `-v08r`, `rsna-knee-teacher-tables` are private).
- RadImageNet licence — unchanged.

### Things that will bite if forgotten

- **Kaggle queue:** fork v9 sat QUEUED ≥ 50 min while infer v19 started instantly — never re-push a queued version (it goes to the back).
- **The laptop sleeps / is off:** background pollers and `watch_submission.py` die with it; a submission already sent scores anyway, but an
  unsent placeholder needs a submit call next session.
- **`rsna-knee-infer` v18 is a SMOKE render** (0.4 h guard) — never submit v18; v19 is the submittable one.
- **Pod GPU stock** can be NONE across every consumer card (12:48 UTC Sunday) — the A100 SXM fallback worked first try (`minCudaVersion 12.8`).
- **Local infer checks on Windows need `RSNA_WORKERS=0`** (spawn cannot pickle the predict loader either); `RSNA_INFER_WORKERS` defaults to 0 there.

## 2026-09-26 (17:25) → 2026-09-27 (09:35) — **P-39 ✅: the Raptor-distilled `v09r` alone reads 0.927** (#20) vs #18 `v09a` 0.918; Raptor pass completed (shard 2/3 after a 3 h Kaggle T4 queue), table published, `v09r` trained on a RunPod 4090 (48 min, pod deleted); nothing running

Tian: "continue training the third shard … prepare everything else … log results". Authorised in-session (asked): **RunPod + submit**
for Task 12 (18:10), **"wait, create later"** for the pod (18:15), then "continue on your own — spin up the pod when appropriate, don't
forget to delete it" and "/update regularly, /handoff only after the results". Commits `7d5816e` … `562b510` (code, script, docs).

### Where things stand

| | Status |
|---|---|
| Best LB | **0.942** fork (#13 / #15); **best member of ours: `v09r` 0.927 solo (#20)** ≈ the public stack's best member (0.928) |
| P-39 Raptor teacher | ✅ **KEEP** — pass complete (4,349 studies, 0 failed, macro 0.9075 vs the LLM teacher; `raptor_teacher.csv` in Dataset `rsna-knee-teacher-tables`); `v09r` gold-58 SWA 0.9093, **LB 0.927 vs 0.918 (+0.009, 1.8× the floor)** — experiments.md 2026-09-26 "Raptor pass complete", "`v09r`", 2026-09-27 "Submission #20" |
| P-40 | 💡 pre-registered: `v08r` (DINOv2-S on the same table) + the fork at β 0.10 with the Raptor-distilled members, **≥ 0.947 ✅ / 0.940–0.946 🔁 / ≤ 0.939 ❌** vs 0.942 |
| Code | arms `v09r`, `v08r` in `SHIPPED_ARMS`; `DISTILLED_ARMS` = arm → exact table set, checked for every arm the session can train (traps 40); `scripts/runpod_chain.sh` = the one-shot pod job (deps with `PIP_BREAK_SYSTEM_PACKAGES=1`, 4 parallel pulls, blob verify, `TEACHER_WAIT_MIN`, train, ship) |
| Datasets | `rsna-knee-teacher-tables` (new version: `raptor_teacher.csv` + `selfdistill_v1.csv`), `rsna-knee-ckpt-v09r` (`v09r_fold0_best.pt` 157 MB + gold csv) — both private |
| Committed notebooks | `rsna-knee-infer` = **v17 = #20** (`v09r` solo); `rsna-knee-train` = **`v09r` SMOKE render, never pushed** (Kaggle still holds v28 = the old `v09t` smoke); `rsna-knee-teacher` = shard 2/3 (v5, done — the pass is complete, never re-push); `-teacher-b` = shard 1/3 (done); fork = v8 (#17); folds = round-2 REAL |
| Quota / money | Kaggle GPU 3.05 h used / 26.95 h left (reset 2026-10-03 00:00 UTC); RunPod ≈ $0.76 tonight (pod `xwgxw1ai93sjow` 23:30 → 00:32, deleted, `list-pods` empty) of the ≈ $4 credit; submissions: 5 today (UTC day) |
| Repo | `main` pushed, clean |

### What we talked about and decided

- **RunPod over a Kaggle real run** for `v09r` (Tian, when asked): 48 min vs ≈ 2.7 h of T4, and — as it turned out — no 3 h GPU queue.
- **No pre-created pod** (Tian chose "wait, create later" over "create, pre-pull, stop"): the pod was created at 23:30, ≈ 15 min before the
  shard ended, and `TEACHER_WAIT_MIN=40` let its setup overlap the merge/publish (it waited 7 min for the Dataset version) — no idle hour.
- **Kept the queued shard instead of re-pushing** (a new version would have replaced it and most likely re-queued at the back).
- **Skipped the `v09r` Kaggle smoke**: it would have sat in the same 3 h queue, and the pod's first minute shows the same lines
  (`teacher table raptor_teacher: 4349 studies`); the local smoke on the 2,900-row partial table had covered the code path.
- **4090 kept as first choice** although SECURE stock read NONE at 23:00 (fallbacks 3090 $0.50 → A40 → A100 $1.59 were lined up); stock was
  back at 23:30.
- **`v09t` / `v09r` moved out of `ARMS`** after finding that an unfiltered run would have trained `v09t` on the plain teacher under its name
  (traps 40) — the same reason `v09s` was moved on 2026-09-23.
- Tian asked what #15 / #16 / #17 changed and what #20 would mean — answered in chat (#15 = anchor control 0.942, #16 = flat-0.60 hedge 0.940,
  #17 = our S2 members at β 0.10 0.941, i.e. the fork does not read member quality at β 0.10; my #20 guess was ≈ 0.920 — it came in higher).
- The P-39 in-sample caveat commit (`3effe43`) landed from a tool call Tian interrupted; kept, since `/update` was asked to record it —
  revert it if unwanted.

### What we figured out

1. **A teacher with information our labels lack transfers; our own OOF does not** — same recipe, same mix code: self-distillation
   −0.001 (#19), Raptor **+0.009** (#20). The target source was a binding constraint (experiments.md "Submission #20").
2. **Gold-58 direction predicted the LB 1 time in 2** (`v09t` +0.009 → −0.001; `v09r` +0.017 → +0.009) — traps 39 stands.
3. **Kaggle T4 capacity can stall a GPU kernel for 3 h 10 min with an empty `failureMessage`** — `kaggle quota` (0.00 h used) rules out
   quota, a throwaway probe kernel rules out the kernel (traps 41). Shard 2 then ran at 7.50 s/study (a shard needs up to 3 h).
4. **The Kaggle CLI refreshes its OAuth token itself** on the first call ≥ 30 min after expiry (kagglesdk's margin has the wrong sign;
   the new token lasts ≈ 12 h) — traps 20 CORRECTED; the "only Tian can re-login" rule was wrong.
5. **The runpod/pytorch 1.0.2 image refuses pip (PEP 668)**, and `… && nohup job &` over ssh backgrounds the whole list and hangs the
   channel (traps 29). `kaggle kernels logs <slug> -f` streams a running kernel's log (mid-run progress readable from the CLI).

### ⏭ Next action, in order

1. **P-40 step A — the fork with `v09r` (no new training; ≈ 0.2 h of T4 + 1 submission, ≈ 10 h to score).** Recommended member set:
   `v09r` alone, so it carries the whole β instead of 1/4 of it (the #17 structure gave `v09a` 2.5 % of the rank blend):
   ```bash
   export PYTHONUTF8=1 PYTHONPATH=src
   .venv/Scripts/python.exe src/build_fork.py --members v09r --member v09r=tiankljucanin/rsna-knee-ckpt-v09r:tiankljucanin/timm-coatnet-rmlp-1-rw-224 --beta 0.10
   .venv/Scripts/python.exe src/build_fork.py --members v09r --member v09r=tiankljucanin/rsna-knee-ckpt-v09r:tiankljucanin/timm-coatnet-rmlp-1-rw-224 --beta 0.10 --check
   grep -E '"machine_shape"|rsna-knee-ckpt-v09r' kaggle/rsna-knee-fork/kernel-metadata.json      # NvidiaTeslaT4, the v09r Dataset
   timeout 90 .venv/Scripts/kaggle.exe kernels push -p kaggle/rsna-knee-fork                          # may queue (traps 41)
   ```
   Placeholder green = `fork_diagnostics.json` `status beta0.10`, `members [v09r]`, subprocess rc 0, `v09r/fold0 … score 0.9093`, anchor sha =
   submission sha on the 3 placeholder studies (expected at β 0.10). Then **Tian's go** → `kaggle competitions submit rsna-knee-abnormality-detection -k tiankljucanin/rsna-knee-fork -v <v> -f submission.csv -m "..."`
   → read vs #13 / #15 0.942: **≥ 0.947 ✅ our member counts / 0.940–0.946 🔁 / ≤ 0.939 ❌**. `/update` (Submissions row 21, P-40).
2. **P-40 step B — `v08r` on a RunPod 4090 (≈ 30 min incl. setup, ≈ $0.4):** check `kaggle` token expiry and 4090 SECURE stock
   (`get-gpu-type … include AVAILABILITY`), create the pod (memory `runpod-pod-self-service`: image `runpod/pytorch:1.0.2-cu1281-torch280-ubuntu2404`,
   SECURE, disk 40 + persistent 100 GB `/workspace`, ports 22/8888, `startSsh`), `stat -f -c %T /workspace; df -h /dev/shm` (MooseFS + ≥ 40 GB shm
   → `CACHE_ROOT=/dev/shm/cache`; local NVMe → `/workspace/cache`), scp `~/.kaggle/credentials.json`, clone, then — **`nohup` as its own statement** —
   `cd /root/repo; CACHE_ROOT=… RSNA_TEACHER_TABLES='("raptor_teacher",)' nohup bash scripts/runpod_chain.sh v08r > /root/job_v08r.log 2>&1 < /dev/null &`.
   Green = `teacher table raptor_teacher: 4349 studies` + `job done`; read gold-58 SWA vs `v08a` 0.8850 (direction only); scp the logs, delete the pod.
   Then the fork with `--members v09r v08r` (both `--member` flags) if step A read 🔁 or ✅.
3. **If step A reads ✅:** a mix-1.0 arm (`TEACHER_MIX = 1.0` sed + a new version name in `SHIPPED_ARMS` / `DISTILLED_ARMS`, ≈ $0.7) and a
   two-teacher table (Raptor + public DINO members over the training set, the `build_teacher_pass.py` pattern) — write a card first.
4. `/update` after every read; `/handoff` at the end.

### Open decisions for Tian

- **The fork's member set for step A**: `v09r` alone (recommended: cleanest read, full β) vs #17's set with `v09r` swapped in for `v09a`
  (`v08w v09h v09r v08a`, isolates the swap vs #17's 0.941 but gives `v09r` a quarter of the arm).
- **Final selection** (deadline 2026-10-22; entries 2026-10-15): #13 / #15 (0.942) + the #16 hedge (0.940) today; a ✅ in step A adds a third
  candidate. Anything final must use **public** Datasets (`rsna-knee-ckpt-v09r`, `rsna-knee-teacher-tables` are private — flip before then).
- Whether to spend a submission on `v08r` solo (it has no solo baseline — `v08a` was never submitted alone).
- RadImageNet licence — unchanged.

### Things that will bite if forgotten

- **GPU queue**: a Kaggle T4 kernel can sit QUEUED for 3 h (Saturday evening); plan pushes with slack and never re-push a queued version.
- **Token**: after an expiry every call fails for ≈ 30 min, then the next call refreshes it (≈ 12 h). Check `access_token_expiration` before
  creating a pod anyway.
- **Pod**: storage differs per pod (tonight MooseFS `/workspace` + 58 GB `/dev/shm`); `runpod_chain.sh` needs `RSNA_TEACHER_TABLES` for a
  distilled arm (the guard refuses otherwise) and `TEACHER_WAIT_MIN` only if the table is not yet published; `nohup` must not end an `&&` list.
- **Committed renders**: `rsna-knee-train` is the `v09r` smoke (never pushed) — re-render before any push; `rsna-knee-infer` is v17 (`v09r` solo);
  `rsna-knee-teacher` is a full-pass shard render (3 h) — the pass is done, do not push it.
- **Local smoke**: `MODE="auto"` resolves to infer on this laptop (`artifacts/kaggle_out/v8/v03_fold0_best.pt`, traps 30) — sed `MODE = "train"`.
- **The laptop sleeps**: background pollers stopped between 01:02 and 09:29 — an overnight read needs the machine awake (or a cloud routine).

## 2026-09-24 (17:35) — **Raptor pass shards 0/3 and 1/3 green**: 2,900 studies, 0 failed, 5.6–6.2 s/study, Raptor vs the LLM teacher macro AUC 0.906 on 2,900; outputs pulled; **shard 2 after Saturday**, then merge → Dataset → Task 12 on RunPod; nothing running

Continuation of the 14:55 entry: both shards finished (17:12, 17:25), were pulled, merged and read with the new
`src/teacher_plausibility.py`. No decisions were needed. Commits: `9754031` (plausibility script), the docs pass and this handoff.

### ⏳ Still in flight as this was written (17:35)

**Nothing is running**: `rsna-knee-teacher` v4 and `rsna-knee-teacher-b` v1 COMPLETE and pulled (`artifacts/kaggle_out/teacher_s0/`,
`teacher_s1/`), no submission pending, no RunPod pod. **Quota this week ≈ 1.5 h left** (6.3 − 2.50 − 2.24) — smokes only until the reset
on Saturday 2026-09-26. Kaggle token valid until **02:54** tonight (then a fresh `! .venv\Scripts\kaggle.exe auth login --force` by Tian
before Saturday's push). Submissions: 5 today, none used.

### Where things stand

| | Status |
|---|---|
| Best LB | **0.942** (#13 / #15); #18 `v09a` alone 0.918 (the baseline for Task 12); #19 `v09t` 0.917 ❌ |
| P-39 Raptor pass | **2/3 done**: shard 0 1,450 studies / 0 failed / 6.20 s/study / 2.50 h; shard 1 1,450 / 0 / 5.56 / 2.24 h; UIDs disjoint; all rows finite in 4 views. `artifacts/teacher/raptor_partial_s01.csv` (2,900 rows) reads **macro 0.906 vs the hard LLM teacher** (MCL 0.844 / Effusion 0.845 / Synovitis 0.847 lowest; Baker's 0.950 highest); Raptor far more positive on the rare labels (Fracture mean 0.47 vs 7 % positive) → quantile matching in the kernel is essential and in place. experiments.md 2026-09-24 "Raptor pass shards 0–1" |
| Shard 2/3 | **not built yet** — `build_teacher_pass.py --shard 2 --n-shards 3` → `kaggle/rsna-knee-teacher/` (v4's output is already pulled, so the slug is free), push after Saturday's reset, **budget 2.5 h** (6.2 s/study in the slower session) |
| Committed notebooks | `rsna-knee-teacher` = shard 0/3 render (done); `rsna-knee-teacher-b` = shard 1/3 render (done); `rsna-knee-train` = v28 smoke; `rsna-knee-infer` = v16 (`v09t` solo — a dead end, rebuild before any infer push); `rsna-knee-folds` = round-2 REAL; fork = v8 |
| Tools | `src/teacher_plausibility.py <table.csv>` = the P-39 read (per-label AUC / ρ / operating points vs `artifacts/targets.csv`, coverage checks; exits 1 below macro 0.85) |
| Docs | experiments: Scoreboard row measured + entry "Raptor pass shards 0–1"; proposals P-39 status + an **"In plain words"** paragraph (what Raptor, the shards and Task 12 are — Tian's question) + Cost; CLAUDE.md state 17:35 + layout |
| Repo | `main` pushed |

### What we talked about and decided

- Nothing new; the 14:55 entry's plan ran as written. The spike's 5.1 s/study was optimistic by up to 20 % (6.2 s in the slower
  session) — Saturday's shard budget is 2.5 h, not 2.1.

### What we figured out

1. **The pass works at scale**: 2,900 studies, 0 failed, every row finite in all four views, view / label order confirmed by the
   per-label read (no label near chance).
2. **Raptor is a second opinion, not a copy**: 0.906 macro agreement with the LLM teacher on 2,900 studies (0.914 on the spike) — it
   disagrees on real cases, lowest on MCL / Effusion / Synovitis, the findings the reports are least specific about. Which side is right is
   what Task 12's solo LB read decides; this table cannot (traps 39).
3. **The guard-stop caveat is real**: at 80 % of shard 1's wall time only 43 % of its rows were 4-view-complete — never plan a shard closer
   than ≈ 30 % to the remaining quota or the 8 h guard.

### ⏭ Next action, in order

1. **Saturday 2026-09-26 (after the quota reset; check the token first):**
   ```bash
   .venv/Scripts/python.exe src/build_teacher_pass.py --shard 2 --n-shards 3           # -> kaggle/rsna-knee-teacher/ (SHARD = 2, N_SHARDS = 3, LIMIT = 0)
   grep -E '^(SHARD|N_SHARDS|LIMIT) = ' kaggle/rsna-knee-teacher/rsna-knee-teacher.py
   .venv/Scripts/python.exe src/teacher_pass_test.py
   timeout 60 .venv/Scripts/kaggle.exe kernels push -p kaggle/rsna-knee-teacher        # ≈ 2.5 h; the second slot is free for a Task 12 smoke
   ```
   COMPLETE → `kernels output tiankljucanin/rsna-knee-teacher -p artifacts/kaggle_out/teacher_s2 --file-pattern "(\.npz|teacher_receipt\.json|\.log)$"`;
   green = receipt `studies: 1449`, `failed_uids: []`. Then
   `.venv/Scripts/python.exe src/merge_teacher.py artifacts/kaggle_out/teacher_s0/raptor_teacher_shard0.npz artifacts/kaggle_out/teacher_s1/raptor_teacher_shard1.npz artifacts/kaggle_out/teacher_s2/raptor_teacher_shard2.npz --expect-n 4349 --out artifacts/teacher/raptor_teacher.csv`,
   `.venv/Scripts/python.exe src/teacher_plausibility.py artifacts/teacher/raptor_teacher.csv` (expect ≈ 0.906), copy the csv into
   `artifacts/ship_teacher/`, `kaggle datasets version -p artifacts/ship_teacher -m "raptor_teacher.csv (4 views, 94 windows, 4,349 studies)"`,
   `datasets status` ready. `/update` (Scoreboard row for shard 2, P-39 → table published).
2. **Task 12 — `v09r`**: add `("v09r", <the v09a dict>)` to `ARMS` and `"v09r"` to `DISTILLED_ARMS`; `TEACHER_TABLES = ("raptor_teacher",)`,
   mix 0.5 (the spec's default); Kaggle smoke (`ARM_ONLY "v09r"` sed'd; expect `teacher table raptor_teacher: 4349 studies`); real run on a
   **RunPod 4090** (create the pod after a fresh Kaggle login; `stat -f -c %T /workspace; df -h /dev/shm` to place the cache; the chained
   job pattern from the 23:30 entry: ≈ 41 min ≈ $0.5) → `ship v09r` → infer solo (`INFER_MEMBERS = ["v09r"]`, Dataset `rsna-knee-ckpt-v09r`)
   → submit → **vs 0.918: ≥ 0.923 ✅ (then `v08r` and the fork at β 0.10) / 0.919–0.922 🔁 / < 0.918 ❌** (traps 39: nothing else counts).
3. `/update` after every read; `/handoff` at the end.

### Open decisions for Tian

- Task 12's mix: 0.5 first (spec); a Raptor-only target (mix 1.0) is the natural second arm if 0.5 reads 🔁 — one more pod hour.
- Whether a 5-minute extra kernel over the 58 gold studies (a direct "Raptor vs gold" read) is worth it — optional; the shards are gold-free.
- Final selection (#13 / #15 vs #16) — unchanged. RadImageNet licence — unchanged.

### Things that will bite if forgotten

- **Shard 2 reuses the `rsna-knee-teacher` slug and dir**: v4's output is pulled, so pushing shard 2 is safe — but never pull "the latest
  version" of that slug again expecting shard 0.
- The token expires 02:54 tonight; Saturday's push needs a fresh login (only Tian).
- Budget 2.5 h per shard, not 2.1; ≈ 1.5 h of quota left this week — no real run before Saturday.
- The infer render is the dead-end `v09t` solo; both teacher renders are full-pass runs.

## 2026-09-24 (14:55) — Afternoon: **the Raptor teacher pass is running — shards 0 and 1 of 3, one per GPU slot** (`rsna-knee-teacher` v4, `rsna-knee-teacher-b` v1, pushed 14:54, ≈ 2.1 h each); re-sharded 2 → 3 so two shards fit the week's last 6.3 h of quota (Tian's choice); shard 2 after Saturday; RunPod credit reserved for Task 12

Tian asked whether the two-shard pass fits the remaining 6.3 h (no: 6.2 session-hours + startup, and a quota kill loses nearly every
row because a row counts only with all four views) and whether RunPod could run it (no: the pass reads the DICOMs, which exist only on
Kaggle — hard constraint 2). Chosen: **3 shards, two now** (≈ 4.3 h of quota, ≈ 2 h margin, 2/3 of the table before Saturday). Tian also
asked what Raptor and the shards are — answered in chat: Raptor = the public 0.942 notebook's frozen CoAtNet-2 branch (≈ 0.924 solo), run
verbatim over our 4,349 report-only training studies to produce a *teacher table*; the shards are a plain partition of that work; the
table is then mixed into our own production member's training targets (Task 12). Commit `f929d4e` (renders) + this handoff.

### ⏳ Still in flight as this was written (14:55)

| In flight | What it is | Started | How to check | How to read it |
|---|---|---|---|---|
| **Shard 0/3 — `rsna-knee-teacher` v4** | the Raptor branch verbatim over training studies 0–1,449 of the sorted gold-free 4,349, both T4s, raw per-view probabilities (`raptor_teacher_shard0.npz` 4 × N × 12 + csv + `teacher_receipt.json`), partial flush every 5 min | 14:54 | `.venv\Scripts\kaggle.exe kernels status tiankljucanin/rsna-knee-teacher` | ≈ 1,450 × 5.1 s + startup ≈ **2.1 h → ≈ 17:05**. COMPLETE → `kernels output tiankljucanin/rsna-knee-teacher -p artifacts/kaggle_out/teacher_s0 --file-pattern "(\.npz\|teacher_receipt\.json\|\.log)$"`; green = receipt `studies: 1450`, `failed_uids: []`, npz `study_uids` 1,450. ERROR / guard stop / quota kill → the partial npz is a legitimate merge input (R18); the resume is `build_teacher_pass.py --shard 0 --n-shards 3 --slug tiankljucanin/rsna-knee-teacher-b --kernel-source tiankljucanin/rsna-knee-teacher`, pushed once `-b` has finished. **Pull before pushing anything else to this slug** (`kernels output` reads the latest version) |
| **Shard 1/3 — `rsna-knee-teacher-b` v1** | the same over studies 1,450–2,899 | 14:54 | `… kernels status tiankljucanin/rsna-knee-teacher-b` | same rule; pull into `artifacts/kaggle_out/teacher_s1`; a resume of this shard goes to the sibling slug `rsna-knee-teacher` with `--kernel-source tiankljucanin/rsna-knee-teacher-b` |

Quota: 6.3 h before the push − ≈ 4.3 h → **≈ 2 h left this week** (a quota kill hits *both* sessions at once; the margin covers a ≈ 45 %
slowdown vs the 100-study spike). Reset Saturday 2026-09-26. **Token:** the CLI does not refresh it — check `access_token_expiration` in
`~/.kaggle/credentials.json` before the 17:05 pull (only Tian can `! .venv\Scripts\kaggle.exe auth login --force`). Submissions: 5 today,
none used. RunPod: no pod; ≈ $4 of credit ≈ 5 h of 4090, **reserved for Task 12** (≈ 41 min per arm incl. the cache pull).

### Where things stand

| | Status |
|---|---|
| Best LB | **0.942** (#13 / #15); #18 `v09a` alone 0.918; #19 `v09t` alone 0.917 ❌ (00:15 entry) |
| P-39 Raptor pass | **2/3 running** (above); shard 2/3 after Saturday: `build_teacher_pass.py --shard 2 --n-shards 3` → `kaggle/rsna-knee-teacher/` (only after v4's output is pulled), push, ≈ 2.1 h → `merge_teacher.py s0 s1 s2 --out artifacts/teacher/raptor_teacher.csv` (`--expect-n 4349`) → `artifacts/ship_teacher/` → `kaggle datasets version` → Task 12 |
| Committed notebooks | `rsna-knee-teacher` = **shard 0/3** (v4, running), `rsna-knee-teacher-b` = **shard 1/3** (v1, running) — LIMIT 0, a re-push = 2.1 h; `rsna-knee-train` = v28 smoke; `rsna-knee-infer` = v16 (#19, `v09t` — a dead end: rebuild before any infer push); `rsna-knee-folds` = round-2 REAL; fork = v8 |
| Docs | experiments: ⏳ Scoreboard row for the pass; proposals: P-39 status; CLAUDE.md state 14:55 + layout rows (0/3, 1/3) |
| Everything else | as the 00:15 entry (P-38 ❌, traps 39, `v09a` the production member) |

### What we talked about and decided

- **3 shards, two now** (Tian) over "one of two now" (3 h of quota would expire unused) and "wait for Saturday": the week's leftover quota
  is spent, the full-pass kernel is validated on real shards, and the finish date (after Saturday) is the same.
- **The pass stays on Kaggle**; the RunPod credit goes to the Task 12 retrain, where it buys the most (a 4090 does the production retrain in
  35 min vs 2.6 h of T4 quota).
- Correction to the 00:15 entry's next action 1: **the pass cannot give "Raptor vs the 58 gold labels"** — the shards are gold-free by design
  (the 58 gold rows never teach). The plausibility read on the merged table is Raptor vs the LLM teacher (0.914 on the spike) plus P-39's
  cited public numbers (0.924 solo LB; gold-58 0.905–0.917 from the anchor's own report). A separate 58-study Raptor read would need a
  small extra kernel (≈ 5 min) — optional, not planned.

### What we figured out

Nothing new was measured. Operational: Kaggle bills concurrent GPU sessions separately, so "6.2 GPU-h" is 6.2 h of quota whether the two
sessions run side by side or not; the 2 h margin rule (never plan a pass closer than ≈ 30 % to the remaining quota) is now in the P-39 card.

### ⏭ Next action, in order

1. **≈ 17:05 — read both shards** by the table above; then a plausibility read on the 2,900 rows:
   `.venv/Scripts/python.exe src/merge_teacher.py artifacts/kaggle_out/teacher_s0/raptor_teacher_shard0.npz artifacts/kaggle_out/teacher_s1/raptor_teacher_shard1.npz --allow-partial --out artifacts/teacher/raptor_partial_s01.csv`
   (expect 2,900 rows), then Raptor vs the hard LLM teacher (`y__* > 0.5` from `artifacts/targets.csv`) macro AUC per label as in the spike
   script pattern (0.914 on 100 studies; MCL lowest) — a value far below 0.9 or a label near 0.5 means a view/label-order problem, stop and
   inspect before Saturday. `/update`: the ⏳ Scoreboard row → measured (studies, s/study, failures, plausibility), P-39 status.
2. **Saturday 2026-09-26 (reset):** shard 2/3 as in "Where things stand" (pull v4's output first); merge all three with `--expect-n 4349`;
   `kaggle datasets version -p artifacts/ship_teacher -m "raptor_teacher.csv (4 views, 94 windows, 4,349 studies)"`; `datasets status` ready;
   `/update`.
3. **Task 12 — `v09r`** (own version name in `ARMS` + `DISTILLED_ARMS`, like `v09t`): `TEACHER_TABLES = ("raptor_teacher",)`, mix 0.5 (the
   spec) — Kaggle smoke (`teacher table raptor_teacher: 4349 studies`), then the real run on a **RunPod 4090** (≈ 41 min ≈ $0.5; check the
   token first, create the pod after a fresh login), `ship v09r`, solo read vs 0.918: **≥ 0.923 ✅ (then `v08r` and the fork) / 0.919–0.922 🔁 /
   < 0.918 ❌**. Judge by nothing else (traps 39).
4. `/update` after every read; `/handoff` at the end.

### Open decisions for Tian

- Task 12's mix (0.5 per the spec; Raptor-only = mix 1.0 as the second arm if 0.5 reads 🔁) — decide after the plausibility read.
- Whether a 5-minute extra kernel over the 58 gold studies is worth it for a direct "Raptor vs gold" number (optional).
- Final selection (#13 / #15 vs #16) — unchanged. RadImageNet licence — unchanged.

### Things that will bite if forgotten

- **Do not push to `rsna-knee-teacher` before pulling v4's output** — `kernels output` serves the latest version only; shard 2 reuses the slug.
- A quota kill shows as `ERROR` in `kernels status` with no message: read the log tail and the partial npz's `study_uids` before deciding
  between a resume and a re-run.
- The token (`access_token_expiration`) before the pull; only Tian can re-login.
- Both teacher renders are full-pass runs (2.1 h per push); the infer render is the dead-end `v09t` solo.
- ≈ 2 h of quota left this week — enough for nothing but smokes until Saturday.

## 2026-09-24 (00:15) — #19 read: the self-distilled `v09t` alone = **0.917** vs 0.918 → ❌ self-distillation does not transfer to the production member; ✅ FINDING: OOF against the LLM targets cannot judge a target-source change (traps 39); nothing running; Saturday = the Raptor pass

Tian asked to check because the Kaggle UI showed nothing running — the submission had scored. Read 00:07, logged, committed, pushed.
This entry supersedes the 23:30 entry's "next action 1" branches: the ✅ branch (`v08t`, distilled fork) is dropped.

### ⏳ Still in flight as this was written (00:15)

**Nothing is running**: no kernel, no submission pending, no RunPod pod (`list-pods` empty). **2 submissions left today** (reset 02:00);
Kaggle token valid until **01:34** (check `access_token_expiration` before the next session's first long step; only Tian can re-login).
Kaggle quota this week ≈ unchanged (two smokes tonight); reset Saturday 2026-09-26.

### Where things stand

| | Status |
|---|---|
| Best LB | **0.942** (#13 / #15); #16 0.940; #17 0.941 🔁; **#18 `v09a` alone 0.918; #19 `v09t` alone 0.917 ❌** |
| P-38 self-distillation | **❌ DEAD END in production** (#19 −0.001 vs #18, sub-floor); the fold-0 ✅ (`v09s` +0.011) is withdrawn as an instrument artefact — experiments.md 2026-09-24 "Submission #19", traps **39** |
| Production member | **`v09a` (S2, LLM targets) stays**; `v09t` exists (Dataset `rsna-knee-ckpt-v09t`, `ARMS`, `DISTILLED_ARMS`) as a measured dead end — do not put it in the fork |
| Remaining lever | **P-39 the Raptor teacher** — a teacher with more information than ours (public 0.924 solo; gold-58 0.905–0.917 held out vs our 0.8948) — judged by gold-58 direction + solo LB only. Shard kernels built and committed (`kaggle/rsna-knee-teacher` 0/2, `kaggle/rsna-knee-teacher-b` 1/2); push after Saturday's reset |
| Docs | experiments: `v09t` row LB cell, Submissions row 19, READ line, entry "Submission #19"; proposals: P-38 ❌ (index + card), P-39 lesson; traps 39; CLAUDE.md state 00:15 |
| Repo | `main` pushed |

### What we talked about and decided

- **#19 is ❌ by the pre-registered rule** (< 0.918) even though −0.001 is noise — the rule was written before the read and stands, as with
  `v09e` (R20). The honest description: a null result; the fold-0 instrument, not the training, produced the false ✅.
- **Dropped:** the `v08t` retrain, the distilled-fork rebuild, and any further self-distillation round (P-17 lineage closed).
- **Kept:** P-39 exactly as planned; Task 12 keeps its measurement (gold-58 direction + solo LB vs 0.918) and gains traps 39 as the reason.

### What we figured out

1. **Self-distillation from our own OOF cannot add information the LLM teacher does not have**: `v09t` 0.917 vs `v09a` 0.918 on the LB
   after +0.011 fold-0 OOF and +0.009 gold-58 — experiments.md 2026-09-24 "Submission #19".
2. **OOF against the LLM targets is the wrong instrument for a target-source change** (it rewards agreement with the teacher; the
   12/12-labels sign consistency came from the same artefact) → **traps 39**: target changes are read by gold-58 direction + solo LB only;
   recipe changes may still use OOF vs the unchanged teacher.
3. Cost of the read: 41 min of a 4090 (≈ $0.5 of training, ≈ $2.0 with the idle wait) + one submission; the Kaggle route would have cost
   2.7 h of quota.

### ⏭ Next action, in order

1. **After Saturday 2026-09-26 (quota reset) — the full Raptor pass**, both shards already rendered and committed (the 23:30 entry's step 2
   has the exact commands): push `kaggle/rsna-knee-teacher` and `kaggle/rsna-knee-teacher-b` (≈ 3.1 h each, concurrently), pull the two
   `raptor_teacher_shard*.npz`, `merge_teacher.py` → `artifacts/teacher/raptor_teacher.csv` (4,349 rows), `kaggle datasets version` of
   `rsna-knee-teacher-tables`, plausibility = Raptor vs LLM teacher macro AUC on all 4,349 (0.914 on the spike) **and** Raptor vs the 58 gold
   labels held out (the Raptor branch never saw them — the number that says whether it carries more truth than our teacher's 0.8948).
2. **Task 12 — the Raptor-distilled production member**: `TEACHER_TABLES = ("raptor_teacher",)` (Raptor alone; the self-distill table is
   dead), `ARM_ONLY = "v09r"` — give it its own version name in `ARMS` + `DISTILLED_ARMS` like `v09t` — smoke → real (a 4090 pod ≈ 41 min
   incl. the pull, or 2.7 h of Kaggle quota) → gold-58 SWA vs 0.8922 (direction) → ship → **solo read vs 0.918: ≥ 0.923 ✅ (then `v08r`
   and the fork) / 0.919–0.922 🔁 / < 0.918 ❌**. Judge it by nothing else (traps 39).
3. `/update` after every read; `/handoff` at the end.

### Open decisions for Tian

- **Task 12's mix** once the Raptor table exists: 0.5 (the spec) is the default; a Raptor-only target (mix 1.0) is the natural second arm if
  0.5 reads 🔁 — decide after the gold-58 plausibility read in step 1.
- Final selection (#13 / #15 vs #16) — unchanged. RadImageNet licence — unchanged.

### Things that will bite if forgotten

- **traps 39**: no fold-0 OOF read for any teacher-table arm; the `v09s` ✅ in older entries is superseded (marked inline).
- `v09t` is in `ARMS` and in Dataset `rsna-knee-ckpt-v09t`: a measured dead end — never `INFER_MEMBERS` it; the committed infer render is
  still the `v09t` solo (v16) — rebuild before any other infer push.
- The 23:30 entry's bites still hold: the teacher kernel dirs are FULL-PASS renders (3.1 h per push), token expiry, pod storage differs
  per pod, `DISTILLED_ARMS` for any new distilled arm name.

## 2026-09-23 (23:30) — Late evening: **`v09t`** (the production `v09a` recipe on the self-distilled targets) trained on a RunPod 4090 in 35 min — gold-58 SWA **0.9009** vs 0.8922 — shipped and **submitted solo as #19 (⏳, read vs 0.918)**; the Raptor full-pass shard kernels built and committed for Saturday; pod terminated

Tian: "Continue with next session's work" (steps 1–3 of the 20:15 entry); chose **RunPod** over Kaggle quota for the real run when
asked; re-authenticated Kaggle when the OAuth token expired mid-way; capped the pod at one hour ("stop the pod if it's not finishing
within an hour") — the chained job took 41 min. Steps 2–3 cannot run before Saturday's quota reset; their kernels are built. Commits
`47127df` (v09t + guard + shard kernels), the docs pass and this handoff.

### ⏳ Still in flight as this was written (23:30)

| In flight | What it is | Started | How to check | How to read it |
|---|---|---|---|---|
| **Submission #19 — `rsna-knee-infer` v16, ref 56504077** | **`v09t` ALONE**: the S2 `v09a` recipe (CoAtNet-1 @224, c02, window_attn, all 4,349 studies, 8 ep, SWA 5–7, batch 2 × accum 2, aug light) trained on `TEACHER_TABLES=("selfdistill_v1",)`, mix 0.5 — gold-58 SWA 0.9009 (`v09a`: 0.8922). Placeholder green 23:22 → 23:24: `infer members (1): v09t/fold0 … [epoch 7, score 0.9009]`, `constant labels 0`, `range=[0.333, 1.000]`. Outputs `artifacts/kaggle_out/infer_solo_v09t/` | 23:25 | `.venv\Scripts\kaggle.exe competitions submissions rsna-knee-abnormality-detection --csv \| head -3` | ≈ 1.5 h (#18 took 1.5 h) → **≈ 01:00**. **vs #18 (the same recipe on the LLM targets, 0.918): ≥ 0.923 ✅ self-distillation transfers to the production member; 0.919–0.922 🔁; < 0.918 ❌.** Then `/update`: fill the LB cell of the `v09t` Scoreboard row + Submissions row 19, P-38 status, CLAUDE.md state |

Nothing else is running: `rsna-knee-train` v28 (smoke) and `rsna-knee-infer` v16 are COMPLETE, the teacher slugs untouched since v3,
**no RunPod pod** (`list-pods` empty at 23:25). Kaggle quota this session: two smokes (≈ 0.1 h) — the training ran on the pod. **2
submissions left today** (reset 02:00). Kaggle token valid until **01:34 local** (fresh login 22:34; see "will bite").

### Where things stand

| | Status |
|---|---|
| Best LB | **0.942** (#13 / #15); #16 0.940; #17 0.941 🔁; **#18 = `v09a` alone 0.918** (baseline); **#19 = `v09t` alone ⏳** |
| `v09t` | **trained** (RunPod RTX 4090, 22:36 → 23:17 job, 35 min of training at 4.4 min/epoch): gold-58 EMA curve 0.798 → 0.866 → 0.886 → 0.893 → 0.898 → 0.901 → 0.901 → 0.902, **SWA 0.9009** (CI95 0.867–0.929), `v09a` S2 was 0.8922 → +0.0087 direction only (floor 0.05); weakest labels Synovitis 0.754 / PF OA 0.799 / Lateral OA 0.803. Dataset `tiankljucanin/rsna-knee-ckpt-v09t` ready (`v09t_fold0_best.pt` 157 MB + gold-58 `_oof.csv`); logs + csv in `artifacts/kaggle_out/pod_v09t/`. experiments.md 2026-09-23 "`v09t`" |
| Code | `ARMS = [v09a, v09t, v08a]` — `v09t` = the `v09a` dict under its own version name (final-review item 6: no collision with `rsna-knee-ckpt-v09a`, a `_last.pt` resume, the fork slot); **`DISTILLED_ARMS = ("v09s", "v09t")`** replaces the `v09s`-only guard (refused without `TEACHER_TABLES`; probed locally); 75 unit checks + local smoke green |
| Committed notebooks | `rsna-knee-train` = **v28 SMOKE** (`ARM_ONLY "v09t"`, `TEACHER_TABLES ("selfdistill_v1",)`, FORCE_SMOKE True — safe); `rsna-knee-infer` = **v16 = #19** (`INFER_MEMBERS ["v09t"]`, MODE infer, + `rsna-knee-ckpt-v09t` source); **`rsna-knee-teacher` = shard 0/2 of the FULL Raptor pass (LIMIT 0 — a push = a 3.1 h session)**; **`rsna-knee-teacher-b/` = shard 1/2 (new dir, slug `tiankljucanin/rsna-knee-teacher-b`, never pushed — the first push creates the kernel)**; `rsna-knee-folds` = round-2 REAL; fork = v8 (#17). `build_teacher_pass.py --check` and `teacher_pass_test.py` green on the shard renders |
| Raptor pass (Task 9 step 3) | **built, not pushed** — waits for Saturday 2026-09-26; ≈ 3.1 h per shard in the two GPU slots; then `merge_teacher.py` → `artifacts/teacher/raptor_teacher.csv` → `kaggle datasets version` of `rsna-knee-teacher-tables` → Task 12 |
| RunPod | pod `r8dijk36d7vpaz` (SECURE 4090, local NVMe `/workspace`, 14 GB `/dev/shm`): created 20:37, **idled ≈ 2 h on the expired Kaggle token**, job 22:36 → 23:17, shipped 23:21, terminated 23:22 — **≈ 2.75 h ≈ $2.0** (≈ $0.5 of it the actual training). `list-pods` empty |
| Docs | experiments: Scoreboard row `v09t` (LB ⏳ #19), Submissions row 19, entry "`v09t`"; proposals: P-38 index + status (production retrain done, #19 ⏳); CLAUDE.md state 23:30 + layout rows (teacher = full-pass shard 0, teacher-b new) |
| Repo | `main` pushed; memory `runpod-pod-self-service` updated (storage check, token window) |

### What we talked about and decided

- **RunPod over Kaggle quota for the real run (Tian, when asked)** — keeps this week's remaining quota; the pod recipe from the afternoon
  worked again (4090 SECURE, disk 40 + 100 GB persistent). **One-hour cap (Tian)**: the job finished in 41 min, so nothing was cut.
- **`v09t` as its own version name** (the final review's item 6) rather than retraining under `v09a`: Dataset, resume and fork slot stay
  unambiguous; the guard was generalised to `DISTILLED_ARMS` so a distilled arm can never train on the plain teacher under its name.
- **The pod idled ≈ 2 h ($1.5) waiting for the re-authentication** — I kept it running after asking Tian to re-login (offered "stop" as the
  alternative); next time check the token's `access_token_expiration` *before* creating a pod (memory + "will bite").
- **The shard kernels are committed as full-pass renders** (the same practice as the round-2 `rsna-knee-folds` real render) so Saturday's
  session only pushes — with the warning in the table above.
- Task 12 and the fork rebuild wait for #19's read and the Raptor table; nothing was pre-decided about the Task 12 table set.

### What we figured out

1. **The production regime on the self-distilled targets trains end to end and reads 0.9009 gold-58 SWA vs 0.8922** — the same sign as the
   fold-0 `v09s` read, still rising at epoch 7; direction only until #19 (experiments.md "`v09t`").
2. **A 4090 pod does the whole production retrain in 35 min** (4.4 min/epoch, 0.06 s/study) — the T4 takes 2.6 h; with four parallel pulls
   the 36 GB cache lands in 6 min (≈ 85 MB/s aggregate). Whole job 41 min.
3. **The Kaggle OAuth token does not refresh itself**: at its `access_token_expiration` (20:37 tonight) every CLI call failed
   (`Permission 'kernels.get' was denied` / `Authentication required`) although a `refresh_token` sits in `credentials.json`; only Tian's
   `kaggle auth login --force` (browser) fixed it, and the new token is valid for 3 h (until 01:34). traps 20 symptom, now with the mechanism.
4. **Pod storage differs per pod**: tonight `/workspace` was a local NVMe xfs (1.5 GB/s) and `/dev/shm` only 14 GB — the afternoon's
   RAM-cache trick would have failed; check `stat -f -c %T /workspace; df -h /dev/shm` before laying out `/kaggle/input`.

### ⏭ Next action, in order

1. **Read #19 (≈ 01:00, or next session):** `.venv\Scripts\kaggle.exe competitions submissions rsna-knee-abnormality-detection --csv | head -3`
   → **≥ 0.923 ✅ / 0.919–0.922 🔁 / < 0.918 ❌** vs 0.918. `/update` (Scoreboard `v09t` row LB cell, Submissions row 19, P-38 status →
   measured, CLAUDE.md state), then:
   - **✅:** add `("v08t", {**PROD, "backbone": "dinov2", "img_size": 224})` to `ARMS` and to `DISTILLED_ARMS`; train it the same way
     (pod: `RSNA_TEACHER_TABLES='("selfdistill_v1",)' bash scripts/runpod_bootstrap.sh train v08t` ≈ 20 min on a 4090, then `ship v08t`;
     or Kaggle `ARM_ONLY = "v08t"` + the `TEACHER_TABLES` sed, ≈ 1.2 h); rebuild the fork at β 0.10 with `v09t` / `v08t` in place of
     `v09a` / `v08a` (`src/build_fork.py --help` for the member and Dataset flags; new Datasets `rsna-knee-ckpt-v09t` / `-v08t`) → smoke →
     **Tian's go** → submit vs 0.942 (≥ 0.947 ✅ / 0.940–0.946 🔁 / ≤ 0.939 ❌).
   - **🔁 / ❌:** `v09a` stays the production member; the Raptor teacher (step 2) is the remaining lever; write the finding into P-38.
2. **After Saturday 2026-09-26 (quota reset) — the full Raptor pass, both shards already rendered and committed:**
   ```bash
   grep -E '^(SHARD|N_SHARDS|LIMIT) = ' kaggle/rsna-knee-teacher/rsna-knee-teacher.py kaggle/rsna-knee-teacher-b/rsna-knee-teacher-b.py   # 0/2/0 and 1/2/0
   timeout 60 .venv/Scripts/kaggle.exe kernels push -p kaggle/rsna-knee-teacher
   timeout 60 .venv/Scripts/kaggle.exe kernels push -p kaggle/rsna-knee-teacher-b      # first push creates the kernel
   ```
   ≈ 3.1 h each, concurrently. Green: `teacher_receipt.json` `studies` 2,175 / 2,174, `failed_uids: []`. Guard stop → re-run
   `build_teacher_pass.py --shard k --n-shards 2 --slug <the OTHER slug> --kernel-source <the stopped slug>` and push (resume by `done_uids`;
   the partial npz is a legitimate input, R18). Pull `kernels output <slug> -p artifacts/kaggle_out/teacher_s<k> --file-pattern "(\.npz|teacher_receipt\.json|\.log)$"`,
   `.venv/Scripts/python.exe src/merge_teacher.py artifacts/kaggle_out/teacher_s0/raptor_teacher_shard0.npz artifacts/kaggle_out/teacher_s1/raptor_teacher_shard1.npz --out artifacts/teacher/raptor_teacher.csv`
   (4,349 rows), copy into `artifacts/ship_teacher/`, `kaggle datasets version -p artifacts/ship_teacher -m "raptor_teacher.csv (4 views, 94 windows, 4,349 studies)"`,
   `datasets status` ready; plausibility = Raptor vs LLM teacher macro AUC on all 4,349 (0.914 on the spike). `/update` P-39.
3. **Task 12 (after 2):** `TEACHER_TABLES = ("raptor_teacher",)` (or the set decided in P-39 after #19's read), `PARALLEL_ARMS = ("v09a", "v08a")`
   sed'd, smoke → real (≈ 2.7 h on Kaggle, or two pod arms) → gold-58 SWA vs 0.8922 / 0.8850 → new Dataset slugs `rsna-knee-ckpt-v09a-rt` /
   `-v08a-rt` (or distinct version names as with `v09t`) → solo read vs 0.918 → fork.
4. `/update` after every read; `/handoff` at the end.

### Open decisions for Tian

- **If #19 ✅:** the `v08t` retrain (pod ≈ $0.3 or 1.2 h of quota) and the fork rebuild with the distilled members — a submission.
- **Task 12's teacher set** once #19 is read: Raptor alone vs Raptor + self-distillation (write it into the P-39 card first).
- Re-open `v09e` — unchanged (low priority). Final selection (#13 / #15 vs #16) — unchanged. RadImageNet licence — unchanged.

### Things that will bite if forgotten

- **`kaggle/rsna-knee-teacher/` and `-b/` are FULL-PASS renders now** — each push is a 3.1 h GPU session; run them only after the reset, and
  only two at a time (the two slots). `rsna-knee-teacher-b` does not exist on Kaggle until its first push.
- **Kaggle token:** `python -c "import json,os;print(json.load(open(os.path.expanduser('~/.kaggle/credentials.json')))['access_token_expiration'])"`
  before anything long-running; the CLI does not refresh it; only Tian can re-login (`! .venv\Scripts\kaggle.exe auth login --force`);
  a pod waiting on it costs $0.74/h — create the pod *after* a fresh login.
- **Pod storage varies** (MooseFS + 58 GB shm vs local NVMe + 14 GB shm): check before choosing `/workspace` vs `/dev/shm` for the cache.
- **`v09t` / `v09s` need the `TEACHER_TABLES` sed**; any *new* distilled arm name must be added to `DISTILLED_ARMS` or it trains silently
  on the plain teacher — grep the log for `teacher table … studies`.
- **Submissions:** 2 left today (reset 02:00); the infer kernel's committed render is the solo `v09t` — for a fork submission use
  `kaggle/rsna-knee-fork/`.
- `kernels status` before any push retry (traps 36); `timeout 60` on every CLI call; `export PATH="/usr/bin:/bin:$PATH"` in the Bash tool;
  git via PowerShell; CRLF docs.

## 2026-09-23 (20:15) — Evening: the **member-strength plan executed subagent-driven** (Tasks 1–8, 9 steps 1–2, 10, 11 — 19 commits on `member-strength`, final review clean, **merged to `main`**); **`v09s` self-distillation 0.8839 ✅ is the one lever**; #18 = the S2 `v09a` alone **0.918**; #17 **0.941** 🔁; Raptor teacher pass green (full pass after Saturday); RunPod pod created, used 3.1 h, terminated

Tian's asks: execute the approved plan subagent-driven, **spin up the pod myself** over the RunPod MCP (superseding "Tian creates the
pod"), estimate its running time; later "pull the latest scoring" (#18 = 0.918), `/update`, `/handoff`, then "finish up, I have to
close my computer". Everything after the opening message ran unattended. The SDD ledger (`.superpowers/sdd/…`, git-ignored) was
deleted after the merge; its rulings R1–R22 are summarised under "decided".

### ⏳ Still in flight as this was written (20:15)

**Nothing is running.** All five kernel slugs are COMPLETE, #17 and #18 are both read, `list-pods` is empty (pod terminated 16:45),
no background task survives this session. Kaggle quota this week: **≈ 3.4–6.4 h left** — the ledger's running count and the sum of
the session logs disagree (22.8 h at 13:15 + train v27 smoke 0.28 + infer v15 0.04 + teacher v1–v3 ≈ 0.25), so **read the quota meter on
kaggle.com before any real push**; reset Saturday 2026-09-26. **3 submissions left today** (reset 02:00). Kaggle token expires
**20:37 local** — the refresh token carried `datasets create` past the last expiry, but `kaggle auth login --force` if the first CLI call
fails (traps 20).

### Where things stand

| | Status |
|---|---|
| Best LB | **0.942** (#13 / #15); #16 0.940; **#17 = 0.941 🔁** (the S2 members at β 0.10, read 20:00 — identical to #14, the fork at β 0.10 does not read member quality); **#18 = the S2 `v09a` ALONE 0.918** — the baseline every distilled member is read against (experiments.md 2026-09-23 "Submission #18", "Submission #17 read 0.941") |
| Member-strength programme | **Tasks 1–8, 9 (steps 1–2), 10 (`v09e` dropped by rule), 11 done — every code task implemented and reviewed by subagents (11 per-task reviews, 6 fix rounds)**; final whole-branch review 20:03 = approved with one fix (applied: the finished arms out of `ARMS`); **left: Task 9 step 3 (full Raptor pass, after Saturday) and Task 12 (distilled production retrain + solo read)** |
| Shared mechanism | `TEACHER_TABLES` / `TEACHER_MIX` / `TEACHER_PATHS` in the config cell → `yt__*` training targets = `(1−mix)·LLM + mix·quantile-matched table` on report-only rows (gold rows stay hard 0/1), `y__*` evaluation targets unchanged; a listed table that is not mounted is fatal; `v09s` is refused without a table (`ARM_ONLY` / `RSNA_ARM` / `PARALLEL_ARMS`). The same functions live in `src/build_targets.py` (`--teacher-tables a,b --teacher-mix 0.5`, default md5 unchanged, BLEND 0.8948) and are AST-equality-guarded by `src/window_head_test.py` (75 checks) |
| Track B (recipe, RunPod 4090) | `v09f` (P-37 pos_weight [1, 10]) **0.8717 🔁**; **`v09s` (P-38 self-distillation on `selfdistill_v1`, mix 0.5) 0.8839 ✅ KEEP — +0.0109 vs `v09c` 0.8730, 12/12 labels up**; `v09e` never ran (round 2's `v09d` 0.8596 ❌ < 0.865). Checkpoints: Datasets `rsna-knee-ckpt-v09f` / `-v09s` (ready); OOF csvs `artifacts/kaggle_out/pod_v09f/`, `pod_v09s/`; `artifacts/ckpt_pod/v09s/v09s_fold0_best.pt` |
| Track A (teacher) | `rsna-knee-teacher` v2 smoke 6/6 (62 s), v3 spike **100/100 at 5.11 s/study** (both T4s, k_eval 94) → the full pass over 4,349 studies = **6.2 GPU-h = 2 shards × 3.1 h in the two slots**; `artifacts/teacher/raptor_spike100.csv`; Raptor vs the hard LLM teacher macro AUC 0.914 on the 100 (MCL 0.77 lowest), Raptor's operating point more positive (Fracture mean 0.47 vs 0.15) → quantile matching is required and in place |
| Datasets | `tiankljucanin/rsna-knee-teacher-tables` = `selfdistill_v1.csv` (4,407 rows; publish folder `artifacts/ship_teacher/` with its `dataset-metadata.json` — `raptor_teacher.csv` goes in as the next version); `rsna-knee-ckpt-v09f`, `-v09s` new; `-v09a` = the S2 member (unchanged, #17 / #18) |
| Code | `ARMS = [v09a, v08a]` (production only); `SHIPPED_ARMS` += `v09d v08c v09e v09f v09s` (reachable through `ARM_ONLY` / `RSNA_ARM` / `PARALLEL_ARMS`); switches at defaults (`FORCE_SMOKE True`, `MODE "auto"`, `ARM_ONLY ""`, `PARALLEL_ARMS ()`, `TEACHER_TABLES ()`); `Config.pos_weight_max` (0 = off, loss byte-identical); `scripts/runpod_bootstrap.sh train <arm>` honours `RSNA_TEACHER_TABLES` and writes `artifacts/runpod_train_<arm>.py`; new `src/build_distill_table.py`, `src/build_teacher_pass.py`, `src/merge_teacher.py`, `src/targets_test.py`, `src/teacher_pass_test.py` |
| Committed notebooks | `rsna-knee-train` = the **teacher-table SMOKE v27** (`PARALLEL_ARMS ("v09s","v09f")`, FORCE_SMOKE True — rendered *before* the `ARMS` move: **regenerate before any push**); `rsna-knee-folds` = the round-2 **REAL** run (re-push = 3.3 h); `rsna-knee-infer` = **v15 = #18** (`MODE "infer"`, `INFER_MEMBERS ["v09a"]`, a real submission-grade run); `rsna-knee-teacher` = the **LIMIT-100 spike v3** (re-push = 0.14 h); fork = v8 (#17) |
| RunPod | pod `eagdf1rc0408w7` (SECURE RTX 4090, EU-CZ-1, $0.74/h, disk 40 + 100 GB persistent `/workspace`): up ≈ 13:40, `v09f` 15:25 → 15:58, `v09s` 16:01 → 16:35 (≈ 34 min per 8-epoch CoAtNet-1 fold-0 arm), **terminated 16:45 — ≈ 3.1 h, ≈ $2.3** (estimate given up front: 3–4 h). Recipe: memory `runpod-pod-self-service` + CLAUDE.md "Off-Kaggle training" |
| Docs | experiments: Scoreboard rows `v09d` / `v08c` / `v09f` / `v09s` / #17 / #18 / teacher smoke + spike, Submissions rows 17–18, entries "Round 2", "RunPod arms on a 4090", "Submission #18", "Raptor teacher pass", "Submission #17 read 0.941"; proposals: P-34 ❌ / P-35 🔁 / P-36 dropped / P-37 🔁 / **P-38 ✅** / **P-39 new** / P-28 LB read; traps **37** (`train_series`, not `train_images`) + **38** (`kernels output` skips a truncated local file); CLAUDE.md state (20:10), layout rows, doc-map range |
| Repo | `member-strength` (`ead67bd` … this handoff) fast-forwarded into `main` and pushed; every commit attribution-free (managed policy; one subagent commit was reworded to drop a harness trailer, R13) |

### What we talked about and decided

- **Pod self-service (R6):** "you also have runpod connected so just spin up a new pod" — this session created, drove (SSH from Bash) and
  terminated the pod. `disk: 100` was refused three times ("no instances available"); disk 40 + a 100 GB persistent `/workspace` worked.
  `/workspace` is a MooseFS network mount, so the 36 GB c02 cache lived on `/dev/shm` behind a symlink (R7).
- **Order and drops by the pre-registered rules:** arms `v09f → v09s → v09e`, `v09e` only if `v09d ≥ 0.865` (R5); `v09d` read 0.8596 →
  **`v09e` dropped and the pod terminated once `v09s` had shipped (R20)**. Caveat written into P-36: `v09d` was still rising at epoch 7.
- **Task 11 ran out of plan order (R11)** while a GPU slot was free — #18 = 0.918 re-priced the member wall (finding 2).
- **The controller ran the pure-CLI steps itself** (Dataset publish, the teacher smoke / spike pushes and pulls, the docs pass, and the
  final review's one-line fix — R16, R21, R22); every code task had a subagent implementer and an independent subagent reviewer.
- **Plan defects found and ruled during execution:** `fold*_oof.csv` swept per-epoch csvs (R8); a rejection test that could never raise
  (R9); `pos_w` NaN under smoke → the loader's own study list (R14); the `v09s` guard's placement and `PARALLEL_ARMS` scope (R2, R15);
  per-arm bootstrap files (R3); the image tree is `train_series` — plan and spec said `train_images` (traps 37).
- **Docs in one pass** after all reads (`a09d9f8`), #17's read folded in tonight; the final review's deferred minors are listed below.

### What we figured out

1. **Self-distillation is the lever: `v09s` +0.0109, 12/12 labels up, 1.4× the 0.008 floor** — the only recipe change since `v09h` that
   clears the floor. Every knob between our CoAtNet and the public 0.928 member is now measured and none does: LR 3e-5 ❌ harmful
   (−0.0134), pos_weight 🔁, BN batch 🔁, aug 🔁, DINOv2 + aug 🔁 → experiments.md "RunPod arms on a 4090", "Round 2"; P-38 ✅.
2. **One production member of ours reads 0.918 solo (#18)** — above our whole 12-member blend (#11, 0.913); the fold-0→LB offset
   under-predicts an all-data SWA member by ≈ 0.02. Baseline for the distilled member: **≥ 0.923 ✅ / 0.919–0.922 🔁 / < 0.918 ❌**.
3. **#17 = 0.941 = #14:** the 8-epoch, +0.015-gold upgrade of both production members moved the fork by nothing → the fork at β 0.10 is
   not an instrument for member quality; the solo submission is. P-28's LB question is closed (✅ as a regime, 🔁 as a stack vote).
4. **The Raptor teacher pass costs 5.1 s/study (6.2 GPU-h for all 4,349)**; on the spike the Raptor teacher scores 0.914 against the
   hard LLM teacher with a more positive operating point → quantile matching, already in the mechanism, is what makes it usable.
5. **Kaggle's DICOM trees are `train_series/` / `test_series/`** (traps 37); **`kernels output` skips an existing truncated blob**, so
   verify every `blob*.npy` against its csv after a pull (traps 38).

### ⏭ Next action, in order

1. **Self-distilled production `v09a` recipe → solo read vs 0.918** (the `/try-out` rule applies: smoke autonomously, **the real run and the
   submission need Tian's go**). Give it its own version so nothing collides with the S2 `v09a` (Dataset, `_last.pt` resume, fork slot):
   add `("v09t", {**PROD, "backbone": "timm:coatnet_rmlp_1_rw_224", "img_size": 224, "lr_backbone": 1e-4, "batch_studies": 2, "grad_accum": 2, "aug": "light"})`
   to `ARMS` beside `v09a`, then
   ```bash
   sed -e 's/^ARM_ONLY = ""/ARM_ONLY = "v09t"/' -e 's/^TEACHER_TABLES = ()/TEACHER_TABLES = ("selfdistill_v1",)/' src/kaggle_pipeline.py > artifacts/train_v09t.py
   grep -E '^(FORCE_SMOKE|MODE|ARM_ONLY|PARALLEL_ARMS|TEACHER_TABLES) = ' artifacts/train_v09t.py      # FORCE_SMOKE True first
   .venv/Scripts/python.exe src/nbgen.py artifacts/train_v09t.py kaggle/rsna-knee-train/rsna-knee-train.ipynb
   timeout 60 .venv/Scripts/kaggle.exe kernels push -p kaggle/rsna-knee-train
   ```
   Smoke green = the log shows `teacher table selfdistill_v1: 4407 studies`, `training targets = (1 - 0.5) * LLM + 0.5 * quantile-matched`,
   `ok  arm v09t`. Real: `FORCE_SMOKE = False` (≈ 2.7 h on one T4) **or** on RunPod (≈ 1 h, ≈ $1):
   `RSNA_TEACHER_TABLES='("selfdistill_v1",)' bash scripts/runpod_bootstrap.sh train v09t && bash scripts/runpod_bootstrap.sh ship v09t`
   → Dataset `rsna-knee-ckpt-v09t`. Read gold-58 SWA vs 0.8922 (direction only). Then the solo kernel exactly as #18: `INFER_MEMBERS = ["v09t"]`
   sed'd, `tiankljucanin/rsna-knee-ckpt-v09t` added to `kaggle/rsna-knee-infer/kernel-metadata.json`, push, `competitions submit -k
   tiankljucanin/rsna-knee-infer -v <version>`. **≥ 0.923 ✅ (P-38 holds in production → it becomes the fork's `v09a` slot); 0.919–0.922 🔁;
   < 0.918 ❌.**
2. **After Saturday 2026-09-26 — Task 9 step 3, the full Raptor pass (≈ 6.5 h of the fresh 30), two shards in the two slots:**
   ```bash
   .venv/Scripts/python.exe src/build_teacher_pass.py --shard 0 --n-shards 2                                            # -> kaggle/rsna-knee-teacher/
   .venv/Scripts/python.exe src/build_teacher_pass.py --shard 1 --n-shards 2 --slug tiankljucanin/rsna-knee-teacher-b   # -> kaggle/rsna-knee-teacher-b/
   grep -E '^(SHARD|N_SHARDS|LIMIT) = ' kaggle/rsna-knee-teacher*/rsna-knee-teacher*.py                                   # LIMIT must be None/0
   timeout 60 .venv/Scripts/kaggle.exe kernels push -p kaggle/rsna-knee-teacher; timeout 60 .venv/Scripts/kaggle.exe kernels push -p kaggle/rsna-knee-teacher-b
   ```
   ≈ 3.1 h each. Green: `teacher_receipt.json` `studies` = 2,175 / 2,174 with `failed_uids: []`. A guard stop → re-push **the same shard**
   with `--kernel-source <the stopped slug>` from the sibling slug (resume by `done_uids`; the partial npz is a legitimate input, R18).
   Pull `kernels output <slug> -p artifacts/kaggle_out/teacher_s<k> --file-pattern "(\.npz|teacher_receipt\.json|\.log)$"`, then
   `.venv/Scripts/python.exe src/merge_teacher.py artifacts/kaggle_out/teacher_s0/raptor_teacher_shard0.npz artifacts/kaggle_out/teacher_s1/raptor_teacher_shard1.npz --out artifacts/teacher/raptor_teacher.csv`
   (must report 4,349 rows; `--allow-partial` only for a guard-stopped run), copy it into `artifacts/ship_teacher/`, `kaggle datasets version -p artifacts/ship_teacher -m "raptor_teacher.csv (4 views, 94 windows, 4,349 studies)"`,
   `datasets status` ready. Plausibility: Raptor vs LLM teacher macro AUC on all 4,349 (0.914 on the spike). `/update` P-39 + Scoreboard row.
3. **Task 12 (after 2):** `TEACHER_TABLES = ("raptor_teacher",)` + `PARALLEL_ARMS = ("v09a", "v08a")` sed'd, `FORCE_SMOKE = True` → Kaggle
   smoke (`teacher table raptor_teacher: 4349 studies` in both children's logs) → real (≈ 2.7 h) → gold-58 SWA vs 0.8922 / 0.8850 (direction
   only) → ship as **new Dataset slugs** `rsna-knee-ckpt-v09a-rt` / `-v08a-rt` (the S2 members stay mounted for the fork) → the solo read of
   the distilled `v09a` with the `-rt` Dataset in the infer kernel: **≥ 0.923 ✅ → fork at β 0.10 with the `-rt` members, `/update` P-39 ✅;
   0.919–0.922 🔁; < 0.918 ❌ (track closed, production members unchanged)**. If step 1 read ✅, decide the Task 12 table set (Raptor alone
   vs Raptor + self-distillation) in the P-39 card *before* building. Never mount the S2 and the `-rt` Dataset in one kernel (same
   `version v09a` file names).
4. `/update` after every read; `/handoff` at the end of the session.

### Open decisions for Tian

- **Go for step 1** — the self-distilled production `v09a` (`v09t`): Kaggle (2.7 h of this week's remaining quota) or RunPod (≈ 1 h,
  ≈ $1), and its solo submission (3 left today, 5 tomorrow).
- **Re-open `v09e`?** Dropped by the rule; `v09d` was still rising (+0.0008/epoch) at epoch 7, so 16 epochs at 3e-5 might close part of the
  gap — ≈ 1 h on a 4090. Low priority: the LR is not the lever, the target source is.
- Final selection (#13 / #15 vs #16) — unchanged; #17 (0.941) does not displace them.
- RadImageNet licence in the final submission — unchanged.

### Things that will bite if forgotten

- **The committed `rsna-knee-train` notebook predates the `ARMS` move** — always regenerate from `src/kaggle_pipeline.py` before a push;
  `rsna-knee-folds` (3.3 h) and `rsna-knee-infer` (a submission-grade infer) are REAL runs.
- **A teacher table is a sed, not an arm key** — `TEACHER_TABLES` is built once per session, so the arm dict cannot carry it. The guard stops
  `v09s` without a table; a *new* distilled arm name (`v09t`, Task 12's `v09a`) has no guard — grep the log for `teacher table … studies`
  before trusting a run.
- **Final-review minors, deferred (all later-only):** the kernel's teacher load has no [0, 1] range check (harmless after quantile matching);
  the local `targets_teacher_*.csv` can hold NaN `yt__` where the LLM blend is NaN (no consumer); a zero-coverage teacher table trains
  silently (add a `SystemExit` before the Raptor retrain); a `_last.pt` resume does not compare the saved `teacher_tables` with the
  session's; the teacher npz carries no label names (`LAB == LABELS` verified by hand on the committed notebook); the spec's cv2 wheel
  Dataset is not mounted (the Kaggle image has cv2).
- **Kaggle CLI hygiene:** token 20:37 → `kaggle auth login --force` (traps 20); `kernels status` before any push retry (traps 36); `timeout 60`
  on every call; `export PATH="/usr/bin:/bin:$PATH"` in the Bash tool; git via PowerShell; the docs are CRLF (single-line patterns only).
- **Teacher pass:** a guard-stopped shard keeps few 4-view-complete studies (GPU-1 arm order) — size shards so the 8 h guard never fires
  (2 × 3.1 h); a resume needs a SIBLING slug (traps 31); `merge_teacher` skips `.tmp.npz`, drops failed studies, borrows sibling view
  weights for a partial (R18).
- **RunPod:** disk 100 refused → disk 40 + persistent 100 GB; `/workspace` is MooseFS → cache on `/dev/shm`; pull the four c02 shards in
  parallel and verify every blob against its csv (traps 38); `delete-pod` as soon as the last arm has shipped, confirm with `list-pods`.
- **Quota next week:** the full pass (6.2 h) + Task 12 (2.7 h) + the solo reads (≈ 0.1 h each) ≈ 9.5 h of the 30; the two teacher shards
  occupy both GPU slots for ≈ 3.1 h.

## 2026-09-23 (13:15) — Afternoon: **round 2 pushed** (P-34 ‖ P-35, `rsna-knee-folds` v8), the **member-strength programme designed and planned** (spec + 12-task plan, approved by Tian) — **the next session executes the plan subagent-driven**; #17 still scoring

Tian's ask this afternoon: "figure out how to improve individual models — they are dragging us down"; research, then implement,
asking rather than assuming. Brainstormed (superpowers), four rounds of questions answered, design approved in full, spec and
plan written and committed. Tian then asked for the plan + this handoff so the session can be cleared (context). **Nothing of
the plan is implemented yet.** Commits `17a827e` (round-2 push), `bfd97f4` (spec), `849f7f5` (plan) + this handoff.

### ⏳ Still in flight as this was written (13:15)

| In flight | What it is | Started | How to check | How to read it |
|---|---|---|---|---|
| **`rsna-knee-folds` v8 = round 2, `v09d` ‖ `v08c`** (P-34 / P-35) | fold 0, 8 ep, `best_oof`, one arm per T4: `v09d` = the `v09c` recipe (batch 2 × accum 2, aug light) with **`lr_backbone` 3e-5** (public LR) on `cuda:0`; `v08c` = the `v08w` recipe (DINOv2-S) + **`aug="light"`** on `cuda:1`. Smoke v7 green (6.84 / 1.56 GiB) | 12:45 | `.venv\Scripts\kaggle.exe kernels status tiankljucanin/rsna-knee-folds`; when COMPLETE: `kernels output tiankljucanin/rsna-knee-folds -p artifacts/kaggle_out/folds_v8 --file-pattern "\.log$"` then `--file-pattern "v0(9d|8c)_fold0.*oof"` | ≈ 3.3 h → **≈ 16:00**. Green: `ok  arm v09d` / `ok  arm v08c`. Read (cards P-34 / P-35): per-label table on all 882 rows, hard = `y__ > 0.5` (the S1 script pattern; `src/blend_check.py` gives the same macros) — **`v09d` vs `v09c` 0.8730: ≥ 0.881 ✅ / 0.865–0.881 🔁 / < 0.865 harmful; `v08c` vs `v08w` 0.8648: ≥ 0.873 ✅ / 0.857–0.873 🔁 / < 0.857 harmful**; ≥ 9/12 labels up as support. `v09d < 0.865` → drop `v09e` (plan Task 10). `/update` afterwards (Scoreboard ⏳ rows exist for neither yet — add them) |
| **Submission #17 — `rsna-knee-fork` v8, ref 56489906** | the S2 production members in the fork at β 0.10 (see the 12:00 entry's table) | 11:54 | `.venv\Scripts\kaggle.exe competitions submissions rsna-knee-abnormality-detection --csv \| head -3` | ≈ 22:00. vs 0.942: ≥ 0.947 ✅ our arm counts; 0.940–0.946 🔁; ≤ 0.939 ❌. Honest expectation 0.942 ± 0.001. Fill Scoreboard + Submissions row 17 via `/update` |

One GPU session running (folds v8, both T4s); the second slot is free. Quota this week ≈ **22.8 h spent** after round 2 (19.5 + 3.3) →
**≈ 7 h left**, reset Saturday 2026-09-26 (weekly). **4 submissions left today** (reset 02:00). Kaggle token valid until **20:37 local**
(`kaggle auth login --force` after that, traps 20). A status monitor on folds v8 was running in this session and **does not survive it**.

### Where things stand

| | Status |
|---|---|
| Best LB | **0.942** (#13 / #15); #16 0.940; #17 ⏳ |
| **Member-strength programme** | **Designed, planned, approved — not started.** Spec `docs/superpowers/specs/2026-09-23-member-strength-design.md`; plan `docs/superpowers/plans/2026-09-23-member-strength.md` (12 tasks, each with code + checks + commit). Tian's decisions (in the spec header): compute = Kaggle + **RunPod (Tian creates the pod, gives SSH)**; recipe knobs judged by fold-0 OOF vs the *unchanged* teacher, the distilled member by gold-58 direction + **one solo LB read vs a baseline solo submission of the S2 `v09a`**; levers = recipe, label source, distillation (no bigger inputs); the Raptor predictions **blend 0.5/0.5** with the LLM teacher; no CoAt-family children in the teacher; execution **subagent-driven** |
| Track A (teacher) | `src/build_teacher_pass.py` (to write, plan Task 7): the 0.942 notebook's Raptor branch verbatim (cell 45 `_KE_SRC` + runner, cell 16 helpers, cell 12 asset finder, cell 14 dense sampler; two token patches; `RSNA_COMP_ROOT` → a chunk of training studies; raw per-view probabilities; partial flush; resume) → `kaggle/rsna-knee-teacher/`; `src/merge_teacher.py` (Task 8); smoke LIMIT 6 + spike LIMIT 100 (Task 9, ≈ 1 h of this week's quota), the full pass after Saturday (10–12 GPU-h estimate, corrected by the spike) |
| Track B (recipe, RunPod) | arms `v09e` (16 ep × 3e-5), `v09f` (`pos_weight_max` 10), `v09s` (self-distill `TEACHER_TABLES=("selfdistill_v1",)`) — plan Tasks 4, 5, 10; cards P-36 … P-38 to be written in Task 5, P-39 in Task 9 |
| Shared mechanism | prediction-table teachers: `quantile_match`, `mix_teacher`, `load_teacher_tables` in `src/build_targets.py` (Task 1) and the kernel (Task 3: `TEACHER_TABLES` / `TEACHER_MIX` / `TEACHER_PATHS` constants, `yt__*` training targets, `y__*` evaluation targets unchanged, `b["yt"]` through Dataset / collate / loss); `src/build_distill_table.py` → `artifacts/teacher/selfdistill_v1.csv` (Task 2); Dataset `tiankljucanin/rsna-knee-teacher-tables` (Task 6) |
| Code | unchanged since `17a827e` apart from docs; `src/kaggle_pipeline.py` `ARMS` = [`v09a` (PROD + S1 knobs), `v08a`, `v09d`, `v08c`]; `PARALLEL_ARMS = ()`, `FORCE_SMOKE = True` |
| Committed notebooks | `rsna-knee-train` = the S2 **real** run (re-push = 2.6 h); `rsna-knee-folds` = the round-2 **real** run (re-push = 3.3 h); fork = v8 (#17) |
| Docs | proposals: P-34 / P-35 ⏳ running rows + statuses; spec + plan under `docs/superpowers/`; experiments unchanged since 12:00 (the round-2 Scoreboard rows are still to add) |

### What we talked about and decided

- **Why individual models:** the 12:00 finding (member-quality wall at ≈ 0.90 solo vs the stack's 0.90–0.928) — Tian wants the
  members themselves stronger. Diagnosis in the spec: our LLM teacher caps at gold 0.8948; every recipe knob moved ≤ 0.005; the
  Raptor member's predictions (gold-58 0.905–0.917 held out) are the one available *better* teacher for our 4,349 studies.
- **Approach chosen (Tian):** two tracks — teacher upgrade (Kaggle, DICOM mount) + recipe knobs (RunPod) — over "teacher only"
  or "recipe only"; Raptor predictions *blended* with the LLM teacher, not replacing it; baseline solo submission kept; CoAt
  family children excluded from the teacher.
- **Round 2 pushed now** (Tian: "push it now") rather than folded into the plan; its `v09d` read decides whether `v09e` runs.
- **Execution:** Tian chose subagent-driven and asked for plan + handoff so this session can be cleared.

### What we figured out (design-time facts, all read from the committed anchor notebook)

1. The Raptor branch reads its inputs from **`RSNA_COMP_ROOT`** (`test.csv`, `test_series.csv`, `test_images/`), needs **exactly
   two GPUs**, and already saves **raw per-view probabilities** (`raptor_raw.npz`, shape 4 × N × 12) before ranking — the
   teacher pass can reuse it verbatim with two token patches (plan Task 7).
2. `train_series.csv` and `test_series.csv` share the schema; `train_images/<study>/<series>/*.dcm` is what their reader walks.
3. The kernel builds targets at runtime from mounted label tables (`LLM_SOURCES`, `build_targets` ~line 739) — a teacher table
   is the same schema, so one mechanism serves self-distillation and the Raptor teacher; `evaluate()` reads `b["y"]`, so the
   training target must be a separate `yt`.
4. Both 5-fold OOF sets on disk (`pod_v09h_5fold`, `folds_v4/v05g`) cover all 4,407 studies once each.
5. No RunPod API key locally and no `runpodctl`: the pod is Tian's to create; `scripts/runpod_bootstrap.sh` does the rest.

### ⏭ Next action, in order

1. **Execute the plan, subagent-driven:** invoke `superpowers:subagent-driven-development` on
   `docs/superpowers/plans/2026-09-23-member-strength.md` from Task 1 (spec beside it). Tasks 1–8 need no GPU and no go; Task 6
   step 2 and Task 9 (smoke LIMIT 6, spike LIMIT 100) use the free GPU slot and are covered by Tian's design approval; Task 9
   step 3 (full pass) waits for Saturday's quota; Task 10 waits for **Tian's pod + SSH**; Tasks 11–12 (solo submissions) are
   covered by the approval.
2. **Read round 2 (≈ 16:00)** by the in-flight table; `/update` (Scoreboard rows, P-34 / P-35 → measured, index).
3. **Read #17 (≈ 22:00)**; `/update`.
4. `/handoff` at the end of the next session.

### Open decisions for Tian

- **Create the RunPod pod** (4090 or A5000, ≥ 60 GB NVMe) and give SSH access — plan Task 10 cannot start without it.
- Final selection (#13 / #15 vs #16 vs #17) — unchanged.
- RadImageNet licence in the final submission — unchanged.

### Things that will bite if forgotten

- **The plan's Global Constraints** repeat the traps that bit today: CRLF doc files (multi-line `\n` patterns match nothing),
  `kernels status` before any push retry (traps 36), `timeout 60` around every Kaggle CLI call, `export PATH="/usr/bin:/bin:$PATH"`
  in the Bash tool, `git` via PowerShell, local smoke needs `MODE = "train"` sed'd (traps 30).
- **Both committed training notebooks are REAL runs** (train = S2 2.6 h, folds = round 2 3.3 h); the first push after any edit is
  a smoke from a sed'd copy.
- `kernels status` shows only a slug's **latest** version — round 2 is on `rsna-knee-folds`; the teacher smoke/spike must use the
  new `rsna-knee-teacher` slug (plan Task 7), never `rsna-knee-folds` while v8 runs.
- The teacher kernel symlinks `chunk/test_images → train_images`; the plan's final cell deletes the chunk dir so the symlink is never
  published as an output.
- Quota: ≈ 7 h left this week; the teacher smoke + spike ≈ 1.2 h; leave the rest for a round-2 resume if it guard-stops.

## 2026-09-23 (12:00) — Day session: S1 read (P-31 ✅, P-32 / P-33 🔁 but same-direction → both knobs into production), #16 read **0.940**, **S2 production retrain on both T4s** (`v09a` 8 ep + knobs gold-58 SWA 0.8922, `v08a` 0.8850) shipped and **submitted as #17** (fork v8, β 0.10); round 2 (P-34 / P-35) smoke-green and staged for a go; "why nothing of ours moves 0.942" written down

Tian's opening message (the go-ahead list for this unattended session): read v23 and `/update`; read #16; S2 = production
`v09a` with the winning knobs beside `v08a` at 8 epochs on both GPUs, ship, submit in the fork at β 0.10 vs 0.942; ≈ 9 h of
quota for a second round; and "figure out why we didn't go over 0.942 and hit 0.940". Everything named there ran; round 2 stopped
at smoke (not named → the `/try-out` rule). Nine commits `5a1a559` … `f91b4b2` + this handoff.

### ⏳ Still in flight as this was written (12:00)

| In flight | What it is | Started | How to check | How to read it |
|---|---|---|---|---|
| **Submission #17 — `rsna-knee-fork` v8, ref 56489906** | #14's graph and blend with the **S2 production members**: `v08w` + 5-fold `v09h` + `v09a` (8 ep, `batch_studies 2, grad_accum 2, aug light`; gold-58 SWA 0.8922) + `v08a` (8 ep; 0.8850), one vote each, β 0.10, the 20 public sources + Datasets `rsna-knee-ckpt-v09a` / `-v08a` (re-versioned 11:44). Placeholder green 11:45 → 11:53: `status beta0.10`, `members [v08w, v09h, v09a, v08a]`, subprocess rc 0 in 190 s, `v09a/fold0 … score 0.8922` / `v08a/fold0 … score 0.885` (the S2 checkpoints), anchor sha = submission sha on 3 studies (expected). Outputs `artifacts/kaggle_out/fork_v8/` | 11:54 | `.venv\Scripts\kaggle.exe competitions submissions rsna-knee-abnormality-detection --csv \| head -3` | ≈ 10 h with the arm → **≈ 22:00**. **vs #13 / #15 (0.942): ≥ 0.947 = our arm finally counts (a correctly trained production member helps → more of them); 0.940–0.946 = 🔁; ≤ 0.939 = the arm hurts.** vs #14 (0.941, the 16-ep members) = the epoch budget + S1 knobs. Honest expectation **0.942 ± 0.001** (experiments.md 2026-09-23 "#16 … why nothing of ours moves 0.942"). `ERROR` = read `fork_diagnostics.json` from the rerun's outputs (the fail-soft anchor should still have been written). Fill the Scoreboard ⏳ row + Submissions row 17 via `/update` |

No kernel running (`rsna-knee-train` v26, `rsna-knee-folds` v7, `rsna-knee-fork` v8 all COMPLETE; both GPU slots free). Quota this
week ≈ **19.5 h spent** (13.4 before v23 + v23 3.22 + smokes v24/v25 0.1 + v26 2.64 + folds v7 0.04 + fork v8 0.12) → **≈ 10.5 h
left**; round 2 needs ≈ 3.3 h. **4 submissions left today** (reset 02:00). Kaggle token valid until **20:37 local** (then
`kaggle auth login --force`, traps 20). No monitors or background tasks survive this session.

### Where things stand

| | Status |
|---|---|
| Best LB | **0.942** (#13 / #15, unchanged). #16 (flat-0.60 outer map, arm not run) = **0.940** → −0.002, 🔁, inside the pre-registered 0.939–0.941 band = the public price of the LB-tuned per-label map; the validated flat-weights hedge for the second final slot (experiments.md 2026-09-23 "#16 …"; brainstorm final-selection row) |
| **Why we do not pass 0.942** | written down (experiments.md 2026-09-23 "#16 … why nothing of ours moves 0.942", ✅ FINDING; CLAUDE.md state): the anchor is a superset of the best public notebook; our members are 0.86–0.87 OOF ≈ 0.89–0.90 solo on the LB while the stack's are 0.90–0.928 solo, and a weaker *correlated* vote (ρ ≈ 0.9) at β 0.10 can only perturb the ranks — every non-zero reading is negative and grows with β; 690 teams sit at 0.941–0.942; ≥ 0.945 is private members trained to ≥ 0.92 solo. A public gain needs a member the stack does not hold (new representation / self-training), not another c02 CoAtNet |
| S1 (P-31 / P-32 / P-33, train v23) | ✅ **P-31**: two arms in 3.22 h wall, 0.25 / 0.28 s/study (1.00× / 1.12× solo), 6.84 GiB each. 🔁 **P-32** `v09b` 0.8690 vs `v09h` 0.8683 (+0.0008, 7/12 up — BN batch composition is not the CoAtNet gap). 🔁 **P-33** `v09c` 0.8730 (+0.0039 over `v09b`, 8/12 up; +0.0047 over `v09h`). Monotone `v09h` < `v09b` < `v09c` → **both knobs into the production `v09a`** by the pre-registered same-direction rule. Honest split-half best-epoch − last ≈ 0 → 8 epochs + `ckpt_policy="last"` loses nothing (experiments.md 2026-09-23 "S1 A/B on both T4s") |
| S2 (P-28 under 8 epochs, train v26) | ✅ **2.64 h for both arms** on the two T4s: `v09a` gold-58 SWA **0.8922** (16-ep: 0.8768), `v08a` **0.8850** (0.8816), both still rising at epoch 7 (no peak-and-drift → P-29 confirmed on production data); gold deltas direction only (floor 0.05). Shipped 11:44 as new versions of `tiankljucanin/rsna-knee-ckpt-v09a` / `-v08a` (`datasets status` ready; `_best.pt` = SWA + gold-58 `_oof.csv`) → #17 (experiments.md 2026-09-23 "S2 production retrain") |
| Round 2 (P-34 / P-35) | 🔧 **implemented, smoke green (`rsna-knee-folds` v7, 09:07 → 09:10), real run awaits Tian's go**: `v09d` = the `v09c` recipe with `lr_backbone=3e-5` (the public 0.928 member's LR; the last never-A/B'd recipe difference) ‖ `v08c` = `v08w` + `aug="light"`. Real file staged: `artifacts/folds_r2_real.py` (FORCE_SMOKE False, `PARALLEL_ARMS = ("v09d", "v08c")`) — cards P-34 / P-35 hold the read-out rules |
| `src/kaggle_pipeline.py` | `ARMS` = [`v09a` (PROD + `batch_studies 2, grad_accum 2, aug light`), `v08a` (PROD), `v09d`, `v08c`]; `v09b` / `v09c` moved to `SHIPPED_ARMS` (finished probes). No pipeline-logic change this session |
| Committed notebooks | `kaggle/rsna-knee-train/rsna-knee-train.ipynb` = **the S2 REAL run** (FORCE_SMOKE False, `PARALLEL_ARMS ("v09a","v08a")`) — re-pushing it starts another 2.6 h session; `kaggle/rsna-knee-folds/rsna-knee-folds.ipynb` = the **round-2 SMOKE** (FORCE_SMOKE True); `kaggle/rsna-knee-fork/` = **v8 (β 0.10, four members, preset speedy)** = #17 |
| Docs | experiments: S1 entry, "#16 … why" entry, S2 entry, Scoreboard rows (#16, `v09b`, `v09c`, S2 smoke, S2, #17), Submissions rows 16–17; proposals: P-31 ✅ / P-32 🔁 / P-33 🔁 / P-28 ✅ (S2) / **P-34, P-35 new cards** + index; traps **36** (`kaggle kernels push` JSON error after the version was created; CLI hangs); brainstorm final-selection row (#16); CLAUDE.md state paragraph (2026-09-23 08:50, S2 done, #17, round 2 staged) |
| Repo | pushed through `f91b4b2` + this handoff |

### What we talked about and decided

- **Knobs for S2 by the rule written last night**, not by taste: a 🔁 knob goes in only if both A/B arms beat `v09h` in the same
  direction — they did (0.8690 and 0.8730 vs 0.8683, monotone), so `v09a` carries both `batch_studies=2` (free) and `aug="light"`
  (+12 % time). Per-label support was weak (7/12), which the entry says plainly.
- **Round 2 was prepared but not launched.** Tian's message named S2's real run and submission explicitly and only *budgeted* quota
  for a second round, so P-34 / P-35 got cards, code, a Kaggle smoke and a staged real file — the real push is one command below,
  for Tian.
- **Round-2 content chosen on S1:** BN batch ≈ 0 and augmentation +0.004 leave the backbone LR (public 3e-5 vs our 1e-4) as the
  last recipe difference to the public 0.928 CoAtNet (P-32's "if it fails" branch), and augmentation on the DINOv2 arm is P-33's
  "if it works" branch. A 5-fold of `v09c` was passed over: it buys private robustness, not a public read, and our arm does not
  count publicly yet.
- **Tian's question ("why we didn't go over 0.942 and hit 0.940")** is answered in experiments.md as a ✅ FINDING rather than in
  chat only: 0.940 is the flat map's public price (−0.002, sub-floor, pre-registered), and the 0.942 ceiling is a member-quality
  wall (our ≈ 0.90-solo members vs the stack's 0.90–0.928), not a blend-weight problem.
- Managed policy: no AI attribution in commits.

### What we figured out

1. **S1:** neither knob clears the floor — `v09b` +0.0008 (P-32 ❌ as a hypothesis, 🔁 as a knob), `v09c` +0.0039 over its
   control (P-33 🔁); ρ(`v09c`, `v09h`) 0.898 = the same member with another seed (experiments.md "S1 A/B on both T4s").
2. **P-31 in production**: the two production arms that took 7.5 session-hours in two sessions took **2.64 h in one**.
3. **The 8-epoch regime does what P-29 predicted on the production data**: both gold curves still rise at epoch 7; the
   16-epoch members were built ≈ 0.015 past their peak (experiments.md "S2 production retrain"). Direction only (floor 0.05).
4. **#16 = 0.940** — the LB-tuned per-label outer map is worth +0.002 publicly; whether it gives it back privately is exactly
   what the hedge is for (experiments.md "#16 …").
5. **Kaggle CLI:** a `kernels push` that dies with `Expecting value: line 1 column 1` had already created the version (v24 ran
   beside my retry v25); a mid-run push of the same slug does not cancel the running version; the CLI can hang > 5 min on
   `kernels status` while the site answers — wrap every call in a timeout (traps 36).

### ⏭ Next action, in order

1. **Read #17 (≈ 22:00):** `.venv\Scripts\kaggle.exe competitions submissions rsna-knee-abnormality-detection --csv | head -3`.
   Rule (in-flight table): ≥ 0.947 ✅ our arm counts → more production members this way; 0.940–0.946 🔁 → the c02 lane is
   closed as a public lever (P-27 "if it fails" stands); ≤ 0.939 → the arm hurts, keep #13/#15/#16 for the final. Then `/update`
   (Scoreboard row, Submissions row 17, P-27/P-28 status, CLAUDE.md state).
2. **Round 2, on Tian's go only** (≈ 3.3 h, both T4s, second slot; token must be valid):
   ```powershell
   $env:PYTHONUTF8 = "1"
   Select-String -Path artifacts/folds_r2_real.py -Pattern '^(FORCE_SMOKE|PARALLEL_ARMS|ARM_ONLY|FIVE_FOLD) = '   # False / ("v09d", "v08c") / "" / False
   .venv\Scripts\python.exe src/nbgen.py artifacts/folds_r2_real.py kaggle/rsna-knee-folds/rsna-knee-folds.ipynb
   .venv\Scripts\kaggle.exe kernels push -p kaggle/rsna-knee-folds        # then `kernels status` before ANY retry (traps 36)
   ```
   Read-out (cards P-34 / P-35): logs `--file-pattern "\.log$"`, OOF csvs `--file-pattern "v0(9d|8c)_fold0.*oof"` into
   `artifacts/kaggle_out/folds_v8/`; per-label table with the S1 script pattern (`v09d` vs `v09c` 0.8730: **≥ 0.881 ✅ / 0.865–0.881
   🔁 / < 0.865 harmful**; `v08c` vs `v08w` 0.8648: **≥ 0.873 ✅ / 0.857–0.873 🔁 / < 0.857 harmful**; ≥ 9/12 labels up as support).
   ✅ on either → that knob into the matching PROD arm and one more S2-style retrain (≈ 2.7 h) → fork β 0.10. Both 🔁/❌ → the c02
   recipe is exhausted as a lever; GPU goes to a new representation (P-23 #3 / #4) or P-17.
3. **Final selection** (before 2026-10-22): #13 or #15 (0.942, tuned map) + #16 (0.940, flat map) hedges the public-tuned weights;
   #17 displaces #13 only if it reads ≥ 0.943.
4. `/update` after every number, `/handoff` at the end.

### Open decisions for Tian

- **Round 2 go** (step 2): ≈ 3.3 h of the ≈ 10.5 h left this week, both arms 🔁-likely by the S1 pattern; the alternative is to
  keep the quota for a member of a *different* kind (P-23 #3 / #4, P-17 — multi-session builds).
- **Final selection**: #13 / #15 (tuned map) vs #16 (flat map) vs #17 (once read).
- RadImageNet licence in the final submission — unchanged.

### Things that will bite if forgotten

- **`kaggle kernels push` may error and still create the version; the CLI may hang.** `kernels status` before any retry; wrap calls in
  `timeout 60 …` (Git Bash) or `Start-Process … WaitForExit` (PowerShell) (traps 36). `kernels status` reports only the *latest*
  version of a slug — a new push hides the running one, so concurrent work goes to the other slug (`rsna-knee-folds`).
- **The committed `rsna-knee-train` notebook is the S2 REAL run** (2.6 h if re-pushed); the committed `rsna-knee-folds` notebook is
  the round-2 smoke; the committed fork tree is v8 (β 0.10, four members) — rebuild with `build_fork.py` flags before any other
  fork push (`--anchor-preset parent --beta 0.0` = the #16 hedge; `--beta 0.0 --members v08w v09h` = the #15 control).
- **`artifacts/ship_v09a/` and `ship_v08a/` now hold the S2 checkpoints** (the 16-epoch ones survive only in
  `artifacts/kaggle_out/train_v19/` and `folds_v6/`); the Datasets' latest versions are the S2 members — any fork rebuild mounts them.
- Local smoke needs `MODE = "train"` sed'd into a scratch copy (traps 30) — `MODE="auto"` resolves to infer on this laptop.
- **Scripted doc edits must match CRLF**: `docs/*.md`, `CLAUDE.md` and `src/kaggle_pipeline.py` are CRLF (multi-line `\n` patterns
  match nothing); `docs/traps.md` is LF. The Bash tool still needs `export PATH="/usr/bin:/bin:$PATH"` (no `sed`/`grep` otherwise; no
  `curl` at all — use PowerShell `Invoke-WebRequest`); `git` via PowerShell.
- Read-out script for fold-0 arms: `s1_readout.py` in this session's scratchpad reproduced the kernel macros exactly (all 882 rows,
  hard = y > 0.5); it is not in the repo — `src/blend_check.py` / `src/oof_epoch_analysis.py` cover the same numbers.

## 2026-09-22 (20:58) — Night session: the public frontier re-read (our anchor is a *superset* of the best public notebook; "0.957" was a gold-58 number), **the machine's second T4 found idle and put to work (P-31)**, the P-32/P-33 A/B arms built, smoke-green and **running as train v23**, `PROD` → 8 epochs, the flat-0.60 hedge submitted as **#16**

This entry supersedes this session's 20:45 entry in place (same session; the only change is that Tian gave the S1 go at 20:54).
Plan file (approved): `~/.claude/plans/i-want-you-to-mossy-prism.md`. Tian's ask: read the handoff, pick the one or two biggest
improvements for the remaining ~16 h of quota on evidence (public notebooks, what the field runs, research), ask rather than
assume. Tian chose **path A** (recipe A/B on both GPUs → production → fork) over B (production only) and C (Raptor-distillation
targets), **build + submit the hedge**, and then **"yes, push it"** for S1.

### ⏳ Still in flight as this was written (20:58)

| In flight | What it is | Started | How to check | How to read it |
|---|---|---|---|---|
| **`rsna-knee-train` v23 = S1 A/B, `v09b` ‖ `v09c`** (P-31 / P-32 / P-33) | the first real two-arm session: parent launches one child per GPU; `v09b` = `v09h` recipe + `batch_studies=2, grad_accum=2` (cuda:0); `v09c` = `v09b` + `aug="light"` (cuda:1); fold 0, 8 ep, `best_oof`. Built from `artifacts/train_ab_real.py` (FORCE_SMOKE False, PARALLEL_ARMS set, ARM_ONLY ""); smoke v22 green first | 20:54 | `kaggle kernels status tiankljucanin/rsna-knee-train`; the Kaggle log shows only the parent's heartbeat (every 3 min: `[arm]` log tails, `nvidia-smi`, host RAM). When COMPLETE: `kaggle kernels output tiankljucanin/rsna-knee-train -p artifacts/kaggle_out/train_v23 --file-pattern "\.log$"` then `--file-pattern "v09[bc]_fold0.*oof"` (+ `--file-pattern "v09[bc]_fold0_best"`) | **≈ 3 h → ≈ 00:00** (CoAtNet-1 fold 0: 14.5 min train + 6.4 min val per epoch × 8, both arms concurrently). Green: parent lines `ok  arm v09b` / `ok  arm v09c` (rc 0, `_best.pt` written); in each child log the `peak GPU memory` line (expect ≈ 6.8 GiB) and `s/study` ≤ 0.33 (1.3× the 0.25 solo — P-31 ✅; higher = loader-bound, note it). **Read-out (P-32/P-33 cards):** `v09b_fold0_oof.csv` / `v09c_fold0_oof.csv` (best_oof epoch) → macro OOF-vs-teacher on the 871 fold-0 studies (`python src/blend_check.py` / `src/oof_epoch_analysis.py`) vs **`v09h` 0.8683**, floor 0.008: **≥ 0.876 ✅ KEEP / 0.860–0.876 🔁 / < 0.860 harmful**; ≥ 9/12 labels up as support; `v09c − v09b` = augmentation alone. `!!  arm` with only `_last.pt` = guard-stopped → resume that arm in `rsna-knee-folds` (`ARM_ONLY`, `kernel_sources` + this output; traps 31). Fill the two ⏳ Scoreboard rows + card statuses via `/update` |
| **Submission #16 — `rsna-knee-fork` v7, ref 56471784** | the hedge: the 0.942 graph with its own `PRESET = "parent"` (per-label outer CoAtNet map LatMen 1.00 / ACL, LatOA, Fracture 0.75 / MedMen 0.80 → flat 0.60), our arm not run (β 0). Placeholder green 20:29: fork log `preset=parent … outer CoAtNet weight per finding: flat 0.60`; `btkd_v559_complete.json` diff vs v6 = every weight 0.6; `status: anchor_control`, submission sha = anchor sha | 20:37 | `kaggle competitions submissions rsna-knee-abnormality-detection --csv \| head -3` | ≈ 6 h (no arm) → **≈ 02:30**. Expected **0.939–0.941** (the 0.941 analysis' ladder: the tuned map is worth +0.001–0.002 publicly). Any value is fine — this is the validated flat-weights candidate for the *second final slot*; fill the Scoreboard / Submissions ⏳ rows via `/update` |

One GPU session running (train v23, both T4s busy); the second slot is free. Quota this week ≈ 13.4 h spent before v23
(tonight: S0 smoke 0.03 h + fork placeholder ≈ 0.1 h) → ≈ 16.4 h after v23's ≈ 3 h. **1 submission left today** (reset 02:00).
Kaggle token valid at 20:54. No background watchers survive this session.

### Where things stand

| | Status |
|---|---|
| Best LB | **0.942** (#13 / #15). Field: 0.958 ×1, 0.955 ×8, 119 teams ≥ 0.945, **270 at 0.942 (us, rank 246), 420 at 0.941** |
| The anchor question | **closed** — Speedy Raptors (top of the score-sorted listing, 0.943) has the same members and weights as our 0.942 graph; "Fast Parent 0.957" is a *subset* whose 0.957 is a gold-58 diagnostic (experiments.md 2026-09-22 night entry, traps 35). Re-anchoring ≤ +0.001 |
| `src/kaggle_pipeline.py` | **P-31** `PARALLEL_ARMS` launcher (one child per GPU, `<arm>.log`, heartbeat, artefact-judged) + `SELF_SOURCE_*` markers; **P-32** `collate_windows`, batched `forward_windows` (scatter-padded head), per-study `weighted_bce`, val/infer at batch 1; **P-33** `Config.aug` (`"light"`: affine + gamma/gain, no flips, training only); `RSNA_SMOKE_FULL_WINDOWS`; **`PROD["epochs"] = 8`**; arms `v09b` (batch 2 × accum 2) / `v09c` (+ aug); `v09p` moved to `SHIPPED_ARMS`. `window_head_test.py` 60+ checks green (incl. batched == single in eval, bce per study, aug geometry, nbgen payload); `cache_selftest.py` green; local smoke of `v09c` green |
| `src/nbgen.py` | embeds the pipeline (zlib + base64 + sha256, 120-char chunks) only when the sed'd text has a non-empty `PARALLEL_ARMS` |
| `src/build_fork.py` | `--anchor-preset {speedy,parent,halfway,sparse}`: one-token patch of the anchor's `PRESET` default, asserted once; provenance + markdown note |
| Kaggle smoke S0 | ✅ `rsna-knee-train` **v22** (20:21 → 20:24): children on `cuda:0` / `cuda:1`, both `_best.pt`, rc 0; **2 × 24 windows CoAtNet-1 @224 = 6.84 GiB peak** (no grad-checkpoint needed); `aug light` vs `none` diverge (loss 0.6834 vs 0.7001). Outputs `artifacts/kaggle_out/train_v22/` (`v09b.log`, `v09c.log`) |
| **`kaggle/rsna-knee-train/rsna-knee-train.ipynb`** | = the S1 REAL run (FORCE_SMOKE False, PARALLEL_ARMS v09b ‖ v09c), generated from `artifacts/train_ab_real.py`, committed, **pushed as v23 at 20:54 (running)** |
| `kaggle/rsna-knee-fork/` | **= the hedge v7 (β 0, preset parent)**; outputs `artifacts/kaggle_out/fork_v7/` |
| Docs | proposals: cards **P-31 / P-32 / P-33** + index rows (🔧 implemented, effect pending), P-28 note; experiments: night entry (public frontier), Scoreboard ⏳ rows (#16, S0 smoke), Submissions row 16; traps **34** (idle second GPU) / **35** (title scores); CLAUDE.md state + "Two arms per session" recipe |
| Repo | `0b90438` (code + docs, smoke green) + this handoff's commit |

### What we talked about and decided

- **What to spend the 16 h on.** Claude's evidence read: re-anchoring is not a lever (superset); our members add ±0 because
  they are ~0.87 OOF while the stack's best single is 0.928 LB (the 0.941 analysis: "below ~0.90 solo a member costs more
  than its diversity buys"); the recipe differences never A/B'd are batch composition (timm CoAtNet MBConv = BatchNorm; ours
  sees 24 windows of one study, the public 0.928 member 8 studies × 12), augmentation (every public recipe has it) and the
  backbone LR. Tian chose **A**: S1 A/B `v09b` ‖ `v09c` on fold 0 (≈ 3 h, both GPUs) → S2 production retrain under the
  winner → fork at β 0.10. C (distil from the public Raptor's predictions on the 4,407 training studies) is parked as a card
  idea; a 16-ch / RadImageNet member and a CoAtNet-2@384 retry were dropped (inside the anchor already / 3.5× the cost).
- **Hedge:** Tian said build **and** submit → #16.
- **Review before coding** (a Plan agent stress-tested the design) caught two real bugs before they ran: a `SystemExit` in a
  notebook cell would have marked the Kaggle version failed (the launcher now returns into the normal flow), and a batched
  `Σ w·bce / Σ w` would have let a gold study (w 8) swallow its partner's gradient (now normalised per study).
- Managed policy: commits carry no AI attribution.

### What we figured out

1. **`"machine_shape": "NvidiaTeslaT4"` is GPU T4 ×2** (kaggle-cli docs PR #1198; probe logs `devices: ['cuda:0','cuda:1']`);
   quota is per session hour (Kaggle product-feedback 361104). Every training session to date used one GPU → traps 34; P-31
   doubles arms per quota hour, verified by the S0 smoke.
2. **Our anchor ⊇ the best public notebook**; the 0.955+ cluster is private work; "0.957" in a title is a gold-58 number
   (traps 35). 690 teams at 0.941–0.942.
3. **Memory:** 2 studies × 24 windows through CoAtNet-1 @224 (AMP) = 6.84 GiB on a 15 GiB T4 → 4 × 24 would be borderline,
   2 × 24 is safe.
4. The fork kernel's log downloaded **non-empty** (27 KB) this time and printed `preset=parent`; the earlier "0 bytes" note
   holds for some runs, not all — check before concluding.

### ⏭ Next action, in order

1. **Read train v23 (≈ 00:00)** exactly as the in-flight table says: logs → `s/study` and `peak GPU memory` per child (P-31),
   then `v09b` / `v09c` OOF vs `v09h` 0.8683 by the 0.008 floor (P-32 / P-33), `v09c − v09b` for augmentation alone.
   `/update` (the two ⏳ Scoreboard rows, P-31/P-32/P-33 card statuses + index rows).
2. **Read #16** (≈ 02:30) → Scoreboard / Submissions rows; note it as the second-final-slot candidate in brainstorm's
   final-selection question.
3. **S2 production**: set the `v09a` arm's knobs to the S1 winner (`batch_studies` / `grad_accum` / `aug` in `ARMS`), keep
   `v08a` as is (8 ep), `PARALLEL_ARMS = ("v09a", "v08a")` → smoke (FORCE_SMOKE True) → real (≈ 2.7 h) → ship Datasets
   `rsna-knee-ckpt-v09a` / `-v08a` (new versions; verify with `kaggle datasets files`) → `build_fork.py --beta 0.10 --members
   v08w v09h v09a v08a --member v09a=…:tiankljucanin/timm-coatnet-rmlp-1-rw-224 --member v08a=…` → placeholder → submit vs
   0.942 (floor 0.005; ≥ 0.947 = our arm finally counts).
4. **Round 2** with the remaining ≈ 9 h, chosen on S1: CoAtNet backbone LR 3e-5 vs 1e-4; `aug="light"` on the DINOv2 arm;
   or a 5-fold of the winner (both GPUs, ≈ 7 h) as the P-17 base.
5. `/update` after every number, `/handoff` at the end.

### Open decisions for Tian

- **S2's knobs** once S1 is read: which of `batch_studies=2` / `aug="light"` the production `v09a` retrain carries (rule: a ✅ knob
  goes in, a 🔁 knob stays out unless both A/B arms beat `v09h` in the same direction).
- **Final selection**: #13 / #15 (0.942, tuned map) vs #16 (flat map) once it scores; selecting one of each hedges the
  public-tuned weights.
- RadImageNet licence in the final submission — unchanged.

### Things that will bite if forgotten

- **The committed `kaggle/rsna-knee-train/rsna-knee-train.ipynb` is the REAL S1 variant** (FORCE_SMOKE False) and is already
  running as v23 — pushing it again would start a second ≈ 3 h run (and a third GPU session is refused while v23 runs). **The
  committed `kaggle/rsna-knee-fork/` is the hedge v7 (β 0, preset parent)** — `build_fork.py --beta 0.10
  --members …` (default preset speedy) before any member submission.
- `PARALLEL_ARMS` is exclusive with `ARM_ONLY` / `FIVE_FOLD` / `STACK_RUN` (the config cell refuses); it needs the nbgen payload
  (a notebook without it fails loudly). Children's logs are `<arm>.log` in the kernel output; the Kaggle log shows the parent's
  heartbeat only (every 3 min; the first one shows 0 % GPU while the children import — not a failure).
- A child that exits 0 with only `_last.pt` was guard-stopped → sibling-slug resume with `RSNA_ARM`/`ARM_ONLY` of that arm
  (traps 31). Judge by `_best.pt`, never by rc (the parent prints `ok  arm` / `!!  arm`).
- `weighted_bce` is now per-study normalised — identical at batch 1, so no shipped member changes; `aug` and `batch_studies`
  are training-only and never reach inference (not INFER_MEMBER_KEYS).
- `sed` of `PARALLEL_ARMS` must go into a *copy*; `src/kaggle_pipeline.py` stays `PARALLEL_ARMS = ()` (else nbgen embeds a 75 KB
  payload into every notebook).

## 2026-09-22 (19:40) — Evening read-out: **best LB 0.942** (#13 β 0.10 and #15 anchor-only, tied); the anchor reproduces, our arm is ≈ 0 publicly; **P-29: the 16-epoch production schedule over-trains** (OOF peak at epoch 8) → future members train 8 epochs

Closes everything the 10:00 entry had in flight. Findings are logged in experiments.md (2026-09-22 "P-29 epoch-budget
probe" and "P-27 read-out with the control", Scoreboard, Submissions rows 13–15), proposals (P-27/P-28/P-29 cards + index),
brainstorm (two questions answered, final-selection question opened), CLAUDE.md state paragraph.

### ⏳ Still in flight — nothing

No kernel running (both GPU slots free; this week's quota ≈ 13.3 h spent: `v09a` 5.1 + `v08a` 2.4 + `v09p` 5.6 + fork
placeholders), no pods, no watchers. Submissions: all five of today scored; the daily count resets 02:00.

### Where things stand

| | Status |
|---|---|
| Best LB | **0.942** — #13 (fork v4: anchor + `v08w` + 5-fold `v09h` at β 0.10) and #15 (fork v6: anchor only). #14 (β 0.10 + `v09a`/`v08a`) 0.941, #12 (β 0.20) 0.939. Top of the LB 0.958 |
| P-27 fork | ✅ delivered 0.913 → 0.942; our arm's increment is ±0.000 at β 0.10 and negative at β 0.20 (sub-floor). **Committed fork tree = v6 (β 0)** — rebuild before any re-push (flags in the 10:00 entry) |
| P-28 production regime | 🔁 `v09a`/`v08a` add −0.001 on the LB; their 16-epoch schedule is ≈ 0.012 OOF past its peak (P-29) → **`PROD["epochs"]` should become 8** — *not yet edited in `src/`* |
| P-29 | ❌ hypothesis failed / ✅ finding kept: `v09p` OOF 0.743 → **0.8731 at epoch 8** → 0.8607 at epoch 15, 11/12 labels down; SWA-of-13–15 proxy 0.8611. `artifacts/kaggle_out/train_v21/` |
| P-30 | ❌ public soft labels have no gold rows → not adoptable (morning) |
| `src/` | unchanged since `9b0f728` (`v09p` arm in `ARMS`, `--sources` in `build_targets.py`) |
| Repo | pushed through `7398a3a` (+ this handoff) |

### What we talked about and decided

- Tian asked for a read of all results and the docs; no new training or submissions this evening.
- **β 0.10 stays the default fork**, not β 0: it scores the same as the anchor publicly and carries the one vote in the
  submission that was never tuned on the public LB. Whether that matters privately is unmeasurable — opened as the
  final-selection question for Tian rather than decided.
- The epoch finding changes the P-28 recipe but I did **not** edit `PROD` or retrain anything — that is a new run and
  needs Tian's go.

### What we figured out

1. **The anchor reproduces 0.942 from our account** (#15), so every fork read is now against a number we own
   (experiments.md P-27 read-out).
2. **Our current members add nothing to the public stack**: ±0.000 at β 0.10, −0.001 with two more members, −0.003 at
   β 0.20. The c02 lane is redundant with the stack's own CoAtNet views; more members of the same kind are not worth GPU.
3. **16 epochs over-trains**: peak − epoch 15 = 0.0124 (1.6× the floor), 11/12 labels down, and SWA of the tail does not
   rescue it; 16 epochs also does not beat 8 at the peak (+0.0048, sub-floor). Our production members were built past
   their best (experiments.md P-29).

### ⏭ Next action, in order

1. **Fix the production budget in `src/kaggle_pipeline.py`**: `PROD = {**C02, "epochs": 8, "train_all": True, "swa_last": 3, "ckpt_policy": "last"}` (SWA over epochs 5–7). One-line change + a note in the P-28 card; via `/try-out` (smoke) before any real run.
2. **Decide what the next GPU goes to** (Tian): the read-out says *not* more c02 members. Candidates, in P-23's order: a
   different input representation (P-23 #3 gated-stem 16-slice model — the public stack's second family, our first
   try `v07s` died as built; P-23 #4 RadImageNet — licence gate), or P-17 self-training. Each is a card + multi-session
   build; the fork is where it would be measured (β 0.10, vs #15's 0.942, floor 0.005).
3. Optional, cheap: retrain `v09a` under the 8-epoch budget (≈ 2.6 h T4, `ARM_ONLY="v09a"` after step 1) and submit it
   in the fork at β 0.10 vs #13 — tells whether a *correctly trained* production member helps. Expect ≤ +0.005 (🔁) given
   finding 2.
4. Before 2026-10-22: select the two final submissions (open decision below).

### Open decisions for Tian

- **Final selection** (two slots): #13 (anchor + our untuned vote) and #15 (pure anchor) tie publicly; selecting both
  hedges the anchor's public-LB tuning. Revisit if a better candidate appears.
- Next GPU direction (step 2); RunPod budget if a new representation needs a bigger card.
- RadImageNet stage licence (CC-BY-NC-SA / "other") in the final submission — unchanged.

### Things that will bite if forgotten

- **The committed fork tree is v6 (β 0)**. #13 is v4 (`--beta 0.10 --members v08w v09h`); rebuild with the builder before pushing.
- **Long heredocs in the Bash tool fail** ("unexpected EOF while looking for matching `'`") even with a quoted delimiter;
  write the script with the Write tool to the scratchpad and run it. Bash-tool PATH still needs `export PATH="/usr/bin:/bin:$PATH"`; use PowerShell for `git`.
- `PROD` still says 16 epochs in `src/` — a production push before step 1 repeats the over-training.

## 2026-09-22 (10:00) — #12 read **0.939** (β 0.20, 0.003 under the anchor's 0.942); P-28 arms `v09a` / `v08a` shipped; **#13 (β 0.10) and #14 (β 0.10 + `v09a` + `v08a`) sent**; **fork v6 = anchor-only control sent as #15**; CLAUDE.md state stack collapsed

This entry consolidates the session's 08:50 and 09:45 entries (same session, superseded in place; nothing older was touched).

### ⏳ Still in flight as this was written (10:00)

| In flight | What it is | Started | How to check | How to read it |
|---|---|---|---|---|
| **Submission #13** — `rsna-knee-fork` v4, ref **56458837** | #12's exact graph and members (`v08w` + 5-fold `v09h`), **β 0.20 → 0.10** — the action the P-27 card pre-registered for a < 0.940 read | 08:30 | `kaggle competitions submissions rsna-knee-abnormality-detection --csv \| head -4` | ≈ 6–10 h (#12 took ≈ 10 h) → **≈ 15:00–19:00**. vs #12 0.939 and the anchor 0.942: **≥ 0.945** = our arm helps at low β; **0.940–0.944** = 🔁; **≤ 0.939** = β is not the lever — then only the control (fork v6) separates "arm hurts" from "anchor drifted". `ERROR` = read the fork's outputs, not the log (the log downloads as 0 bytes for this notebook); the fail-soft anchor should still have been written |
| **Submission #14** — `rsna-knee-fork` v5, ref **56459131** | #13 + the two all-data SWA production members in our arm (`v08w`, `v09h` ×5, **`v09a`, `v08a`**, one vote each), β 0.10 | 08:41 | same command | same window. **Read vs #13** (same β, two members): **≥ +0.005** = the P-28 members earn their place (P-28 ✅ on the LB → every future member trains this way); **±0.004** = 🔁; **a drop** = the new members are redundant with the stack (their gold-58 was fine, so the stack, not the members, is the reason). Fill both Scoreboard ⏳ rows + Submissions rows 13/14 via `/update` |
| **Submission #15 — anchor-only control**, `rsna-knee-fork` v6, ref **56461317** (Tian's go, sent 10:08) | `build_fork.py --beta 0.0 --members v08w v09h`: the 0.942 graph verbatim, **our arm is not launched** (new `_ForkControl` branch → `status: anchor_control`), `submission.csv` byte-identical to the anchor; same 20 sources as v3/v4. Placeholder 09:33–09:40 green: `subprocess: null`, anchor sha = submission sha (verified on the downloaded file), anchor graph 204 s, 20/20 DINO + 5 A5 folds | 10:08 | same command; ≈ 6 h (no arm) → **≈ 16:00** | The baseline every β read is relative to. **0.942** = the anchor reproduces → #12's −0.003 is our arm's (small, 🔁 by the floor, but the sign is ours); **≠ 0.942** = the anchor itself moved (unpinned sources / rerun variance) and #12–#14 must be read against *this* number. Fill Scoreboard + Submissions rows via `/update` |
| **`rsna-knee-train` v21 = `v09p` (P-29 epoch-budget probe)** | the `v09h` recipe on fold 0 with the production schedule's 16 epochs, per-epoch OOF csvs (`v09p_fold0_ep{0..15}_oof.csv`), `best_oof` checkpoint; local + Kaggle smoke (v20) green first | 13:05 | `kaggle kernels status tiankljucanin/rsna-knee-train`; when COMPLETE: `kaggle kernels output tiankljucanin/rsna-knee-train -p artifacts/kaggle_out/train_v21 --file-pattern \"v09p_fold0_ep*\"` (+ `_oof.csv`, `_best.pt`); log via `--file-pattern \"no_match\"` and the `tr '\r' '\n' \| grep -E \"epoch\|EMA score\|-> v09p\"` reading | ≈ 4 h (3,480 studies × 16 ep at 0.25 s/study) → **≈ 17:00**. **Read (P-29 card):** macro OOF-vs-teacher per epoch on the 871 held-out studies (`python src/oof_epoch_analysis.py` covers `_ep*_oof.csv`; or `blend_check.py`'s AUC). Peak − epoch-15 **< 0.008** = no early peak → 16 epochs confirmed for P-28; **≥ 0.008 with the peak before epoch 10** = the production schedule over-trains → P-28 `epochs` → the peak, retrain `v09a`/`v08a` before more members. Also: last-epoch OOF vs `v09h` 8-ep 0.8683 (does 16 beat 8 at all, floor 0.008). `stopping: runtime guard` = resume in `rsna-knee-folds` with `tiankljucanin/rsna-knee-train` in its `kernel_sources` (traps 31) |

Kaggle: **2 submissions left today** (reset 02:00); **one kernel running** (`rsna-knee-train` v21 = `v09p`; one GPU slot free; quota this week ≈ 7.7 h: `v09a` 5.1 h + `v08a` 2.4 h + four fork placeholders); no pods; no background watchers survive this session. The Kaggle OAuth token self-refreshed during the session and is valid until **23:42 local** (`access_token_expiration` in `~/.kaggle/credentials.json`; after that `kaggle auth login --force`, traps 20).

### Where things stand

| | Status |
|---|---|
| Best LB | **0.939** — #12, the P-27 fork at β 0.20 (our own blend alone is 0.913, #11). 0.003 *under* the anchor's author-stated 0.942, so the fork has not yet shown a gain from our arm (experiments.md Submissions #12) |
| P-27 fork | `src/build_fork.py` deterministic; β ≤ 0 is now a real anchor-only control (experiments.md Infrastructure 2026-09-22). **Committed tree = fork v6 (β 0)**. v4 = two members β 0.10 (#13); v5 = four members β 0.10 (#14), reproducible with `--beta 0.10 --members v08w v09h v09a v08a --member v09a=tiankljucanin/rsna-knee-ckpt-v09a:tiankljucanin/timm-coatnet-rmlp-1-rw-224 --member v08a=tiankljucanin/rsna-knee-ckpt-v08a` (checked: `artifacts/fork_v5b/`). Placeholder outputs in `artifacts/kaggle_out/fork_v4..6/` |
| P-28 members | ✅ **`v09a`** (CoAtNet-1, 5.12 h, gold-58 SWA 0.8768) and **`v08a`** (DINOv2-S, 2.38 h, 0.8816) trained on all 4,349 studies; SWA ≈ last EMA; gold peaks at epoch 5–6 then drifts ≈ −0.015 (experiments.md 2026-09-22 entry; brainstorm question). Datasets **`tiankljucanin/rsna-knee-ckpt-v09a`** / **`-v08a`** (`_best.pt` + gold-58 `_oof.csv`, verified with `kaggle datasets files`). Local: `artifacts/kaggle_out/train_v19/` (incl. `_lastema.pt`, `_last.pt` 1.15 GB, 16 per-epoch csvs), `folds_v6/`, staging dirs `artifacts/ship_v09a/`, `ship_v08a/` |
| Docs | experiments: #12 verdict, Scoreboard rows (#12, `v09a`, `v08a` filled; #13/#14 ⏳), Submissions rows 12–14, the P-28 arms entry, the control-mode Infrastructure entry; proposals: P-27 📊 measured / P-28 ✅ trained (cards + index); brainstorm: anchor-control (→ fork v6) + epoch-budget questions; **CLAUDE.md: the fifteen stacked "State as of" paragraphs (a duplicate of this file) replaced by one current-state paragraph**, doc-map row P-00…P-28 |
| Repo | pushed through `eee6686` + this handoff (`9974600` docs, `1a76e02` fork v5, `5962234` handoff, `eee6686` builder control + CLAUDE.md cleanup) |

### What we talked about and decided

- **Two submissions, not one, on the 0.939 read** (Claude's call under "run what you planned"): the plan held both "β 0.10 on < 0.940" and "ship the members → next submission", so they went out as a chain at the *same* β — #13 isolates β, #14 vs #13 isolates the P-28 members. Members at β 0.20 would have confounded the two.
- **Both at β 0.10, never tuned further on the LB** — the P-27 card's rule. If #13 and #14 both land ≤ 0.942 the next step is the control's number, not a β sweep.
- **The anchor-only control** was recommended by Claude; Tian first asked for it to be *built and prepared*, then (10:08) said "submit" — it is submission #15. Cost: 1 of 5 daily slots, ≈ 6 h scoring, no GPU.
- **Control = arm not run at all** (not β 0 through the blend): Tian's ask was a control, and a control that still runs the 80-minute arm carries every failure mode of the arm and none of its signal. The builder change is 8 lines and keeps `--check` determinism and the v5 build byte-for-byte except the new branch.
- **Docs cleanup scope** (Tian: "remove redundant information if needed"): only CLAUDE.md's state stack qualified — it duplicated this file. experiments.md (append-only) and older handoff entries were left alone by convention; this session's own two entries were consolidated into this one.
- `/update` was run *before* the Kaggle login unblocked (docs first, so a dead session would still leave the numbers logged).

### What we figured out

1. **#12 = 0.939 fires the pre-registered action rule but is 🔁 as evidence** — −0.003 is 0.6× the LB floor, the anchor's 0.942 comes from another account with unpinned Dataset sources, and the arm is fail-soft (a skipped arm would *also* have scored "the anchor"). Only a β 0 submission from our account distinguishes "our arm hurts" from "the anchor drifted" (experiments.md Submissions #12; brainstorm).
2. **The P-28 regime runs end to end and SWA is harmless**: SWA − last EMA = +0.0006 / +0.0014 on gold-58; no BatchNorm penalty on the CoAtNet arm, so the card's `update_bn` fallback is not needed (experiments.md 2026-09-22).
3. **Both gold curves peak at epoch 5–6 and drift ≈ −0.015 to epoch 15** while the loss keeps falling — inside the 58-row SE, but the same sign on two arms; logged as a question, not a finding (the settling measurement: a fold-0 16-epoch twin of `v09h` with per-epoch OOF, floor 0.008). 16 epochs stays until then.
4. The CoAtNet-1 all-data arm costs **0.25 s/study on a T4** (18 min/epoch, 5.1 h for 16) — only ~2× the DINOv2 arm; both fit one 9 h session without a resume.
5. **#12 took ≈ 10 h to score** (21:24 → before 07:50), not the 5–7 h estimated; #13/#14 carry a similar or slightly longer arm. The control (no arm) should score in ≈ 6 h.
6. **β 0 through the blend was not a control** — it would have run the arm and written `rank_pct(rank_pct(anchor))` (AUC-identical, different bytes, every arm failure mode attached). Now `_FORK_BETA <= 0` skips the arm before the runtime gate (experiments.md Infrastructure 2026-09-22).

### ⏭ Next action, in order

1. **Read #13 / #14 / #15** (≈ 15:00–19:00; the control ≈ 16:00) **and `v09p` (train v21, ≈ 17:00; read-out in the in-flight table)**: `kaggle competitions submissions rsna-knee-abnormality-detection --csv | head -5`. Verdicts by the in-flight table; then `/update` (Scoreboard ⏳ rows, Submissions rows, P-27/P-28 card status, CLAUDE.md state paragraph).
2. **If #14 ≥ #13 + 0.005:** P-28 is the production recipe → next members under `PROD`: B6 ConvNeXt-T c02 (`("v06a", {**PROD, "backbone": "convnext_tiny", "img_size": 224, "lr_backbone": 1e-4})` in `ARMS`; local smoke with `MODE="train"` sed'd in; Kaggle `ARM_ONLY="v06a"` ≈ 3–4 h, or RunPod attended) and B7 CoAtNet-2@384 PROD (RunPod, bs 8, ≈ 3 h). Each new member joins via `build_fork.py --members … --member v06a=tiankljucanin/rsna-knee-ckpt-v06a:tiankljucanin/convnext-tiny-224-hf`. **Rebuild v5's flags first** — the committed fork tree is the β 0 control.
3. **If #13 and #14 both ≤ the control's number:** P-27 "if it fails" — our members are redundant with the stack on the public split; spend GPU only on members *different* from the stack (input representation: P-23 #3/#4, P-17), keep the anchor for public submissions.
4. ~~CPU item B5~~ — **done 2026-09-22, P-30 ❌**: `src/build_targets.py --sources …` exists (default teacher byte-identical, 0.8948); the public file covers no gold row, so it cannot be validated; not adopted (experiments.md Label sources 2026-09-22).
5. ~~Optional measurement for the epoch question~~ → **running as `v09p` (P-29, train v21)**; was: fold-0 `v09h` twin at 16 epochs (`("v09p", {**C02, "backbone": "timm:coatnet_rmlp_1_rw_224", "img_size": 224, "lr_backbone": 1e-4, "epochs": 16})`, `ARM_FOLDS=(0,)`), read the OOF peak epoch from the per-epoch csvs with `src/oof_epoch_analysis.py`.

### Open decisions for Tian

- RunPod budget for B6/B7 (≈ $1 / $2.5) once #14 is read.
- The RadImageNet stage (CC-BY-NC-SA / "other") in the *final* submission — unchanged.
- Whether to act on the epoch-5–6 gold peak (step 5) before training more 16-epoch members.

### Things that will bite if forgotten

- **The committed fork tree is v6 (β 0, arm not run).** Pushing it again is *not* the four-member blend — rebuild v5 first (flags in "Where things stand"); the build log line `members ['v08w', 'v09h', 'v09a', 'v08a'] beta 0.1` confirms.
- **The Claude Code Bash tool lost the Git Bash PATH entries mid-session** (after a context reset): `git`, `grep`, `date`, `seq`, `ls` all "command not found" while `kaggle` via `.venv/Scripts` still ran. Fix: `export PATH="/usr/bin:/bin:$PATH"` at the top of every Bash call and run `git` through the PowerShell tool. A background poll loop that used `seq` died silently with exit 0 and "timeout:" — read a monitor's output file, never assume it waited.
- **`kaggle auth login` tokens last ~3 h but the CLI sometimes self-refreshes** (08:02 → 11:02, then refreshed to 23:42 during the session); the expiry error blames the slug (traps 20). `kaggle kernels output` with a broad `--file-pattern` (`v09a_fold0`) also pulls the 1.15 GB `_last.pt` — use `v09a_fold0_best*` etc.
- Parallel Bash calls share the working directory: a `cd artifacts/ship_v09a` in one call moved the other's cwd (the fork push failed once on relative paths). Use absolute paths or `cd` at the start of every call.
- The fork placeholder's `submission.csv` is byte-identical to the anchor at any β ≤ 0.3 with 3 studies — the placeholder proves plumbing only; read `fork_diagnostics.json` (`status`, `subprocess`, `members`) and `ours_infer.log` (`blend: by_version -> …`).
- A raw-string template (`r"""…"""`) cannot hold a triple-quoted docstring — the first control build died on a `SyntaxError` inside the arm cell; comments only inside `FORK_ARM_TEMPLATE`.
- Push the next version of a kernel slug only after the previous one leaves RUNNING (one GPU slot each; a mid-run push was not tested).

## 2026-09-21 (21:30) — Delta on the 21:10 entry: fork v3 green, **submission #12 sent**, `v08a` pushed

Read the 21:10 entry for the session's full state; only the in-flight table changed:

| In flight | What it is | Started | How to check | How to read it |
|---|---|---|---|---|
| **Submission #12** — `rsna-knee-fork` v3, ref **56442019** | P-27: the public 0.942 graph + our c02 arm at β = 0.20 (placeholder run green: anchor graph 184 s, ours rc 0, `fork_diagnostics.json` `beta0.20`) | 21:24 | `kaggle competitions submissions rsna-knee-abnormality-detection --csv \| head -3` | scoring ≈ 5–7 h → **≈ 02:30–04:30**. **≥ 0.947** = our arm helps (P-27 ✅); **0.940–0.946** = 🔁, β 0.20 stays; **< 0.940** = our arm hurts → `python src/build_fork.py --beta 0.10`, push, submit; `ERROR` = read the rerun via the fork's outputs (the kernel log is empty for this notebook) — most likely the 8 h guard, then the fail-soft anchor should still have been written. Fill the Scoreboard ⏳ row + P-27 status via `/update` |
| **`rsna-knee-train` v19 = `v09a`** | P-28 CoAtNet-1 all-data 16 ep SWA | 20:53 | as in the 21:10 table | 4–8 h → ≈ 01:00–05:00 |
| **`rsna-knee-folds` v6 = `v08a`** | P-28 DINOv2-S all-data 16 ep SWA | 21:26 | `kaggle kernels status tiankljucanin/rsna-knee-folds`; output as for `v09a` (`--file-pattern "no_match"` for the log) | ≈ 2.6 h → ≈ 00:00; same green lines (`SWA of last 3 EMA snapshot(s)`, `-> v08a_fold0_best.pt = SWA`) |

4 submissions left today (reset 02:00). Both GPU slots are busy until `v08a` finishes; no background watchers survive this session. Next action list = the 21:10 entry's, starting at step 3 (ship the members in the morning) — step 1 (submit) and step 2 (push `v08a`) are done.

## 2026-09-21 (21:10) — The 0.942 notebook read and forked (P-27: its graph verbatim + our arm), the production training regime built (P-28: all-data, 16 ep, SWA); `v09a` training on Kaggle; fork v3 waiting for a GPU slot; nothing submitted yet

Plan file: `~/.claude/plans/i-want-you-to-witty-kernighan.md` (the full enumerated / prioritised change list —
A1…A5 blend-level, B1…B12 training-level, C1…C2 — with the tonight / morning split). Public LB top is now
**0.958**; we are 0.913. Tian's three decisions this session: **mount the public checkpoints AND keep training
our own**; **Kaggle only tonight, RunPod in the morning, attended**; **all-data + SWA for production members,
fold-0 stays the A/B instrument**.

### ⏳ Still in flight as this was written (21:10)

| In flight | What it is | Started | How to check | How to read it |
|---|---|---|---|---|
| **`rsna-knee-train` v19 = `v09a` REAL** (CoAtNet-1 @224, c02, window_attn, `train_all`, 16 ep, `swa_last=3`) | first P-28 production member; ARM_ONLY sed'd, `FORCE_SMOKE=False` | 20:53 | `kaggle kernels status tiankljucanin/rsna-knee-train`; when COMPLETE: `kaggle kernels output tiankljucanin/rsna-knee-train -p artifacts/kaggle_out/train_v19 --file-pattern "no_match"` then `tr '\r' '\n' < …/rsna-knee-train.log \| grep -E "studies in\|epoch\|SWA\|-> v09a\|!!"` (the Kaggle log is one JSON line per stream — the `tr`/grep is how to read it) | expected **4–8 h** (T4; 0.21–0.45 s/study → 15–33 min/epoch × 16). Gold-58 `auc_gold` per epoch is *reported only* (SE ≈ 0.04). The last lines must show `SWA of last 3 EMA snapshot(s)` and `-> v09a_fold0_best.pt = SWA, v09a_fold0_lastema.pt = last EMA`. **Suspicious:** SWA more than ~0.01 below the last-EMA gold (BatchNorm averaging — P-28 "if it fails"); `stopping: runtime guard` = resume in the sibling slug (`rsna-knee-folds`, add `tiankljucanin/rsna-knee-train` to its `kernel_sources`, push the same `ARM_ONLY="v09a"` notebook; log must say `resume: copied v09a_fold0_last.pt`) |
| **`rsna-knee-fork` v2** (the UNFIXED arm cell — pushed by mistake, will end `ERROR` like v1 after ~6 min) → **a background bash in the 21:00 session pushes the FIXED v3 the moment v2 ends** (`kaggle kernels push -p kaggle/rsna-knee-fork`) | placeholder run of the P-27 fork: 0.942 graph + our arm (`v08w` + 5-fold `v09h`, β 0.20) on the 3 placeholder studies | v2 ≈ 21:05; v3 = when v2 ends | `kaggle kernels status tiankljucanin/rsna-knee-fork`; **if the session died before the push, push v3 by hand** (`kaggle kernels list --user tiankljucanin` shows the last run time; the committed notebook IS v3 — `grep -c "outcome=_fork_status" kaggle/rsna-knee-fork/rsna-knee-fork.ipynb` = 1). Then `kaggle kernels output tiankljucanin/rsna-knee-fork -p artifacts/kaggle_out/fork_v3` (no pattern — **the kernel log comes back EMPTY for this notebook; read `diagnostics/current_phase.json`, `fork_diagnostics.json`, `btkd_v559_complete.json`, `ours_infer.log` instead**) | **Green = status `COMPLETE`**, `fork_diagnostics.json` → `"status": "beta0.20"`, `"returncode": 0`, `btkd_v559_complete.json` status COMPLETE with dino_members 20 / a5_folds 5 / raptor_views 4, `ours_infer.log` ends with `blend: by_version -> v08w (1 fold), v09h (5 folds)` and `wrote … submission.csv rows=3`. v1 reached all of that and died only on the `status=` keyword in `finally` (experiments.md Infrastructure 2026-09-21). **Then submit** — see Next action 1 |
| `rsna-knee-folds` v6 = `v08a` REAL — **NOT pushed yet** (notebook generated and committed; the second GPU slot was taken by the fork) | DINOv2-S @224 c02 window_attn, same P-28 regime, ≈ 2.6 h | — | push when a slot is free: `kaggle kernels push -p kaggle/rsna-knee-folds` (its ipynb already carries `ARM_ONLY = "v08a"`, `FORCE_SMOKE = False`) | same reading as `v09a` |

Kaggle: **5 submissions/day, 0 used today, reset 02:00**; GPU quota spent tonight ≈ 0.3 h (two smokes, two fork runs) + the running `v09a`. The Kaggle OAuth token was valid at 21:09.

### Where things stand

| | Status |
|---|---|
| Best LB | **0.913** (#11) — unchanged; the fork's anchor alone is 0.942 by its author |
| `src/kaggle_pipeline.py` | P-28 shipped: `train_all`, `swa_last`, `split_studies`, `average_state_dicts`, SWA ring persisted in `_last.pt`, `_lastema.pt`, `ARM_ONLY`, `PROD` arms `v09a` / `v08a`, `SHIPPED_ARMS`, per-arm resume (traps 31), `oof_eval` gold-only (traps 32). `window_head_test.py` 37/37; local `MODE="train"` smoke green; Kaggle smokes green (train v18, folds v5) |
| `src/build_fork.py` → `kaggle/rsna-knee-fork/` | P-27 shipped and deterministic (`--check`); 54 cells = 50 anchor (byte-identical, asserted) + 3 fork + credits; 20 sources accepted by Kaggle; fork v1 proved every stage runs on Kaggle (224 s anchor + 115 s ours on 3 studies); **v3 = the fixed arm cell** |
| `kaggle/rsna-knee-folds/kernel-metadata.json` | now mounts the four c02 shards + the CoAtNet-1 weights (either kernel can host either arm / the other's resume) |
| Docs | proposals P-27 / P-28 cards + index rows; experiments Scoreboard ⏳ rows (#12 fork, `v09a`, `v08a`), "Anatomy of the public 0.942 notebook" entry, Infrastructure entry; traps 31 / 32 / 33; brainstorm: fork decision + licences closed; CLAUDE.md layout / production-arm recipe / State line |
| Licences | every Raptor / CoAt / A5 / DINOv2 / soft-label dataset **CC0**; the three RadImageNet datasets CC-BY-NC-SA-4.0 / "other" — decision only for the *final* submission |
| Repo | pushed through `4b5362c` + this handoff; `notebook_score_0.942.ipynb` now tracked (the builder's input) |

### What we talked about and decided

- **Why fork instead of only training:** the 0.942 notebook trains nothing — ~35 public checkpoints across ~6
  families, several trained on A6000/H100; its 0.924 single member alone is ~0.025 above our best single. Reaching
  0.942 with own models is months. Tian chose fork + keep training (the P-23 zero-training alternative,
  open in brainstorm.md since 2026-08-30). We become a fork of the shared ensemble; our arm is the part that is
  *not* in everyone else's fork.
- **Our arm inside the fork = c02 members only** (`v08w` + 5-fold `v09h`): drops `v10c` (a third of our rerun) and
  the c01 decode pass; the arm costs ≈ 80 min of hidden-test time after the anchor graph, fail-soft (anchor
  retained on any failure or past 7.0 h / an 8.4 h projection). β = 0.20 fixed, never tuned on the LB; the
  FineSpacing stage was dropped because the scored 0.942 run never had its dataset attached.
- **Production regime** copies the 0.924 member's training (all data, 16 ep, SWA) but **not** its gold-58 epoch
  selection (we average the *last* three EMA snapshots; the docs' ban on selecting on 58 rows stands) and not its
  no-laterality / slot-crossing input (P-05 is +0.015 for us and is our diversity).
- **Kaggle only tonight** (Tian will not leave a pod running unattended); RunPod in the morning for B6 (ConvNeXt-T
  c02 PROD) / B7 (CoAtNet-2@384 PROD, bs 8) if wanted.
- **Not tuning their inner weights** (LatMen 1.00 etc.) — their own diagnostics cell warns they are the likeliest
  place to give back points privately.

### What we figured out

1. **Cells 0–49 of the 0.942 notebook are the scored graph**: its `dataSources` has 17 entries and the FineSpacing
   dataset is not among them (experiments.md "Anatomy of the public 0.942 notebook"). 0.941 → 0.942 was half
   blend-weight tuning, half the CoAt family, by its own header.
2. **Per-arm resume never worked** (traps 31): the mounted `_last.pt` was looked up under the default version
   `v03`, so any resumed arm restarted at epoch 0. Fixed inside the arm loop; the `mode = "infer" if …` rule stays
   on `v03` on purpose.
3. **A `train_all` member has no OOF** (traps 32): `oof_eval` would have scored 871 of its training studies as
   "OOF"; it now scores gold-58 only, and `blend_check.py` must not see such members.
4. **The fork works end to end on Kaggle** (fork v1 outputs: 20/20 DINO gate, A5, Rad, 4 Raptor views, both CoAt
   children rc 0, our 6 members, blend); the only failure was a keyword collision in the arm cell's `finally`
   (`rsna_phase` takes `status` positionally). **The Kaggle log of this notebook downloads as 0 bytes** — read the
   `diagnostics/` folder instead (Infrastructure 2026-09-21).
5. With 3 placeholder studies a β = 0.2 blend is identical to the anchor (a 1/3 rank step cannot be overturned by
   0.2 × 2/3) — the placeholder proves plumbing, not the blend.

### ⏭ Next action, in order

1. **Fork v3 → submit #12.** When `kaggle kernels status tiankljucanin/rsna-knee-fork` is `COMPLETE` and the outputs
   read green (table above):
   `kaggle competitions submit rsna-knee-abnormality-detection -k tiankljucanin/rsna-knee-fork -v 3 -f submission.csv -m "P-27: public 0.942 graph (cells 0-49 verbatim) + our c02 arm (v08w + 5-fold v09h) at beta=0.20; anchor copy and beta 0.10/0.30 variants in the outputs"`.
   Scoring ≈ 5–7 h. **Read-out (pre-registered, P-27):** ≥ 0.947 = our arm helps; 0.940–0.946 = 🔁 (β 0.20 stays);
   < 0.940 = hurts → next submission β = 0.10 (`build_fork.py --beta 0.10`, re-push, submit) or the anchor.
   If v3 is `ERROR`: `kaggle kernels output … -p artifacts/kaggle_out/fork_v3` and read `diagnostics/current_phase.json`.
2. **Push `v08a`** as soon as a GPU slot is free (after fork v3 ends): `kaggle kernels push -p kaggle/rsna-knee-folds`.
3. **Morning: ship the production members.** For each of `v09a` (train v19) and `v08a` (folds v6) when COMPLETE:
   `kaggle kernels output tiankljucanin/rsna-knee-train -p artifacts/kaggle_out/train_v19 --file-pattern "v09a_fold0_best"`
   (+ `_lastema`, `_oof.csv`), then a Dataset `tiankljucanin/rsna-knee-ckpt-v09a` with `v09a_fold0_best.pt` +
   `v09a_fold0_oof.csv` (verify by `kaggle datasets files`, never exit status — traps 20/21); same for `v08a`. Then
   `python src/build_fork.py --members v08w v09h v09a v08a --member v09a=tiankljucanin/rsna-knee-ckpt-v09a:tiankljucanin/timm-coatnet-rmlp-1-rw-224 --member v08a=tiankljucanin/rsna-knee-ckpt-v08a`
   → push → placeholder green → submit #13. Read: vs #12 by the 0.005 floor.
4. **Morning, CPU:** B5 — `kaggle datasets download dreaddevelopment/rsna-knee-labels -p data/llm_labels/dread --unzip`,
   add `labels_llm_soft.csv` as a 4th source in `src/build_targets.py` behind a flag, read its gold-58 macro-AUC
   in the scoring block; adopt for *future* arms only if ≥ 0.893 (hans_v4).
5. **Morning, RunPod (attended):** B6 ConvNeXt-T c02 window_attn PROD (`("v06a", {**PROD, "backbone": "convnext_tiny", "img_size": 224, "lr_backbone": 1e-4})` — add to `ARMS`, smoke locally, `RSNA_ARM=v06a`), then B7 only if time remains. Pod `setup` ≈ 40 min (traps 29).
6. `/update` after every number (#12 score, per-epoch gold curves, SWA vs last-EMA), then `/handoff`.

### Open decisions for Tian

- Submit #12 as soon as v3 is green (tonight, 5 slots) or wait for the morning's members and submit once.
- The RadImageNet stage (CC-BY-NC-SA / "other") in the *final* submission — keep, or drop the Rad stage from the fork.
- RunPod budget for the morning (B6 ≈ 1 h 4090 ≈ $1; B7 ≈ 3 h ≈ $2.5).
- Whether `v10c` / the c01 members ever return to our arm (runtime vs. the +0.00x they might add).

### Things that will bite if forgotten

- **The Kaggle log of the fork notebook downloads as 0 bytes** even when it ran — the `diagnostics/` folder and
  `fork_diagnostics.json` are the record. Do not conclude "never started" from an empty log.
- **`kaggle kernels push` refuses a third GPU session** ("Maximum batch GPU session count of 2 reached") — the fork
  placeholder and one training run fill both slots; sequence the pushes.
- A PowerShell `[IO.File]::WriteAllText("p", $s -replace 'a','b')` is parsed as three arguments — wrap the
  replace in parentheses, or the file is not written and a stale copy runs (bit twice tonight).
- The builder's own guard fired on its explanatory comment and the unrebuilt notebook was pushed as v2; check the
  build's exit status before any push (`if ($LASTEXITCODE -ne 0)`).
- `ARM_ONLY` must be sed'd into every training push; grep the generated `.py` for `ARM_ONLY = "v0` first. The
  committed `rsna-knee-train.ipynb` / `rsna-knee-folds.ipynb` are the real-run variants (`FORCE_SMOKE = False`).
- Background bash jobs (the v3 pusher) die with the session that started them.

## 2026-08-30 (22:40) — Submission #11 sent (infer v14, the 0.912 blend with `v09h` as 5 folds); Tian moves to a new laptop — migration notes below

Delta on the 22:00 entry (read that one for the day's full state). Tian's call: submit v14 tonight.

### ⏳ Still in flight as this was written (22:40)

| In flight | What it is | Started | How to check | How to read it |
|---|---|---|---|---|
| **Submission #11** — infer v14, ref 55899052 | the #10 blend (seven versions, LB 0.912) with `v09h` as **5 folds** instead of 1 (still one vote); 15 checkpoints, placeholder run verified (decode equality, `v09h (5 folds)`) | 21:55 | `kaggle competitions submissions rsna-knee-abnormality-detection --csv \| head -3` | scoring ≈ 2 h 40 min → **≈ 00:35**. **Expected 0.912–0.916**: folds are replicates, so ≤ +0.005 vs 0.912 = **🔁 by design, not a failure**; ≥ 0.917 = a real fold-ensemble gain (would beat what `v05g`'s folds bought). Whatever it scores stays the default blend (it can only match or beat #10's members). Fill the ⏳ Scoreboard row + this table via `/update`. **The scoring monitor dies with this session** (laptop switch) — read it manually next session |

1 submission left today (resets 02:00). Kaggle GPU quota ≈ 3.5 h this week.

### Moving to the new laptop — what to copy vs re-fetch

`docs/setup.md` is the walkthrough (clone → CPU venv → `kaggle auth login` → re-fetch CSVs/labels/weights →
verify with `build_targets.py` = 0.8948). Updated today. Copy by hand ONLY:

1. **`data/sample_dicom/`** (572 MB, 557 files) — Kaggle 429-rate-limits it; verify `find data/sample_dicom -name '*.dcm' -size +0 | wc -l` = 557.
2. **`artifacts/kaggle_out/`** (~50 MB w/o checkpoints) — the pod-era parts (`pod_v09h_5fold/`, `pod_v09h/`, `pod_v10c/`, `eval_pod/`: per-epoch csvs + logs) exist **nowhere else** since the pod was deleted; the rest of `artifacts/kaggle_out` saves re-pulling old kernel outputs. Skip `ship_v09h_5fold/` + `ckpt_pod/` checkpoints (re-downloadable from the `rsna-knee-ckpt-*` Datasets).
3. **`~/.ssh/id_ed25519`(+`.pub`)** — the key registered in RunPod account settings; or generate a new one there.
4. Optional: the Claude Code project memory `C:\Users\Tian\.claude\projects\C--Users-Tian-Desktop-RSNA-Knee\memory\` (the dir name encodes the repo path — keep the repo at the same path or rename the dir to match).

Everything else: re-fetch (`data/*.csv`, `data/llm_labels/`, `models/`) or regenerate (`.venv`, `artifacts/targets.csv`).
Do NOT copy `data/` wholesale — setup.md's warning about partial trees stands; `kaggle auth login` on the new machine
replaces the token (nothing to copy in `~/.kaggle`).

### ⏭ Next action, in order

1. **Read #11** (≈ 00:35 or next session): fill the Scoreboard ⏳ row, the table above, CLAUDE.md "State"; verdict by the
   0.005 floor (0.912–0.916 → 🔁 "folds priced in", ≥ 0.917 → ✅). No further action either way — v14 stays the default.
2. **Next GPU work = a new input representation** (unchanged from 22:00): card first via `/try-out` — P-23 #3 gated-stem
   16-ch, #4 RadImageNet (licence gate), or P-17 self-training (its target OOF now exists:
   `artifacts/kaggle_out/pod_v09h_5fold/`). New pod = `setup` ~40 min (traps 29 checklist).
3. Housekeeping: datasets public before a final submission; legacy `kaggle.json` for pods; `crazy_good_rsna.ipynb` keep/delete.

### Things that will bite if forgotten

- The 22:00 entry's list (silent expired-token uploads; never truncate a running script; Bash-tool `cd` persistence;
  compressed Dataset sizes) — all still true.
- On the new laptop the local smoke needs `models/` re-downloaded (setup.md §6) before `python src/kaggle_pipeline.py` runs.

## 2026-08-30 (22:00) — `v09h` is the 5-fold production member (pooled 0.8625, shipped, infer v14 verified); LB 0.900 → **0.909 (#9)** → **0.912 (#10)**; pod deleted; nothing left in flight

Closes the 18:10 entry: everything it had in flight has landed. This was the "implement v09h" session
(plan `~/.claude/plans/i-want-you-to-fluffy-hellman.md`): the 0.936 notebook re-inventoried (its strongest member's
mechanism was already fully inside `v09h`), the 5-fold `v09h` trained on the pod, both submissions scored and logged,
all pod artifacts pulled and md5-verified locally, the Dataset versioned **from local**, infer v14 pushed and verified,
the pod **deleted** (day's pod spend ≈ $7).

### ⏳ Still in flight — nothing

Infer v14 `COMPLETE` and log-verified (15 checkpoints, `v09h (5 folds)` one vote, both decode groups byte-identical,
valid `submission.csv`). No kernels running, no pods, no background watchers. Kaggle GPU quota ≈ 3.5 h this week;
**2 submissions left today** (resets 02:00 local); the Kaggle OAuth token self-refreshed at ~21:45 during
`kaggle datasets version` — the *pod* copy did not (see bites).

### Where things stand

| | Status |
|---|---|
| Best LB | **0.912** — #10 (infer v13, seven versions, OOF 0.8820); #9 six versions 0.909; offsets +0.0295/+0.030 (experiments.md Submissions #9/#10) |
| Default blend | the seven versions, `INFER_MEMBERS` in `src/` = v05a v05b v05g v06c v08w v10c v09h; **infer v14 = that blend with `v09h` as 5 folds** — pushed, verified, **not submitted** |
| `v09h` | ✅ 5-fold: per-fold **0.8683/0.8668/0.8589/0.8546/0.8653**, pooled **0.8625** (+0.016 over `v05g`), gold-58 0.874; `rsna-knee-ckpt-v09h` = 10 files, 766 MB (experiments.md "5-fold `v09h`") |
| Pins | all seven members are Dataset pins (`-v05`, `-v05g`, `-v06`, `-v08w`, `-v09h` 5-fold, `-v10c`); infer `kernel_sources: []` — train pushes are safe |
| Local artifacts | `artifacts/ship_v09h_5fold/` (5+5+metadata, the Dataset's exact content), `artifacts/kaggle_out/pod_v09h_5fold/` (5 oof csvs + `epochs/` 40 per-epoch csvs + chain3/chain4/evals/train logs + `oof_summary5.py`), `artifacts/ckpt_pod/` (`v10c_fold0_best.pt` + oof) — every checkpoint md5-verified against the pod before deletion |
| Pod | **deleted 22:05** (`2wend9j0lr7zf3`); the `ARM_FOLDS = (0,1,2,3,4)` sed lived only on its copy, so no revert needed anywhere |
| Tools | `src/fold_oof_summary.py` (pooled/per-fold/gold; validated on `v05g`), fixed `ship` glob in `scripts/runpod_bootstrap.sh` |
| Docs | experiments.md: Scoreboard #9/#10 + 5-fold row, Submissions rows 9/10, "Rerun cost of the blend", 5-fold entry; P-23 status/index; traps 29 + two new bullets; CLAUDE.md 22:15 state |
| Repo | pushed through this handoff (`4949c5c` … `ecebcc7` + this); `crazy_good_rsna.ipynb`, `kaggle/rsna-knee-eval/rsna-knee-eval.ipynb` still untracked |

### What we talked about and decided

- Tian: "implement v09h = all the bits that made the 0.936 notebook good" → inventory showed the bits are already in;
  chose **5-fold production run** over porting more cells or a new-family arm tonight; **delete the pod** after pulling
  (later sharpened to "pull everything locally so the pod stops costing"); no submission without his call.
- **Both submissions read as pre-registered**: #9 = 0.909 ✅ (the c02 recipe carries, 1.8× floor); #10 = 0.912, best,
  default by the tie rule, but the `v09h` increment (+0.003) is 🔁 exactly as the OOF predicted.
- Tian's questions answered in the log: why scoring took ~2 h (rerun cost table — `v10c` + a second decode pass);
  whether "each fold worse" was real (fold difficulty + noise; fold 4 = 0.8653 broke the trend); what folds are for
  (replicates inside one vote; rank-mean within version, then across versions).
- v14 submission timing: ready ~21:55; **held for Tian** (expected 0.912–0.916, likely 🔁; 2 slots left today).

### What we figured out

1. **5-fold `v09h` pooled 0.8625** (+0.016 over `v05g`'s 0.8467); LatMen pooled 0.789 → 0.847; fold spread 0.0137 =
   1.7× the floor, so fold-0 A/Bs are flattered ≈ +0.006 as levels (fine as deltas) — experiments.md 5-fold entry.
2. **`best_oof` bit for the first time** on this recipe (folds 2/3/4 peaked at epochs 6/6/5).
3. **The rerun cost doubled with the c02 members** (#9 ≈ 15 min/100 studies; `v10c` alone 174 s/100) — experiments.md
   Infrastructure entry; dropping `v10c` would cut a third of the rerun if 384 px stays useless.
4. **`kaggle datasets version` dies silently (exit 0, < 2 s) on an expired token** — the upload twin of traps 20; and
   **overwriting a script a running bash is executing corrupts its next read** (spurious `!! train v09h FAILED` after a
   clean 5-fold run) — both appended to traps 29. Kaggle Dataset "size" is the *compressed* bytes (766 MB for 825 MB).
5. The 0.936 notebook's primary CoAtNet arm evaluates `K_EVAL = 62` windows (its "42" docstring is stale) — proposals
   Rejected row corrected; our `eval_windows=42` stays as a T4 budget cap, not a copy of theirs.

### ⏭ Next action, in order

1. **(Tian) submit infer v14 or hold**: `kaggle competitions submit rsna-knee-abnormality-detection -k tiankljucanin/rsna-knee-infer -v 14 -f submission.csv -m "seven-version blend with 5-fold v09h (pooled OOF 0.8625)"`.
   Read: ≥ 0.917 real gain; 0.912–0.916 = 🔁 (folds are replicates); scoring ≈ 2 h 40 min. Either way v14 is the base
   the next member rides on.
2. **Next GPU work = a new input representation**: write the card first (`/try-out`) — P-23 #3 (16-ch with a gated
   stem, the P-15 retry) or #4 (RadImageNet frozen heads, licence gate), or P-17 self-training (targets = ½ teacher +
   ½ multi-family OOF rank blend, retrain the `v09h` recipe; its full-corpus OOF now exists in
   `artifacts/kaggle_out/pod_v09h_5fold/`). A new pod needs `setup` again (~40 min; traps 29 checklist).
3. Housekeeping when convenient: make `convnext-tiny-224-hf`, `timm-*`, `rsna-knee-ckpt-*` public before any *final*
   submission; a legacy `kaggle.json` for unattended pods; `crazy_good_rsna.ipynb` keep/delete.

### Open decisions for Tian

- Submit v14 tonight (2 slots left) vs let the next member's submission carry the 5-fold.
- Which new-family lane gets the next pod dollar: P-23 #3 (gated-stem 16-ch), #4 (RadImageNet, licence), or P-17.
- Unchanged browser items: rules text, radimagenet T&C, Kaggle caps.

### Things that will bite if forgotten

- **Uploads with an expired OAuth token exit 0 doing nothing** — verify Datasets by file list + size, never exit
  status (the local CLI *can* self-refresh; the copied pod credentials did not).
- **Never truncate-overwrite a running script** — `mv` over it instead (traps 29, cost a spurious FAILED line).
- Bash tool: `cd` persists across calls; long Markdown heredocs fail — write a scratchpad `.py` (memory
  `bash-tool-env-traps`).
- The infer kernel now mounts **no** kernel_sources: a new member = pin a Dataset + add it to
  `kernel-metadata.json` + `INFER_MEMBERS`, then push.
- Kaggle Dataset "size" in listings is compressed bytes — a 5 × 165 MB ship shows as ≈ 766 MB.

## 2026-08-30 (18:10) — 5-fold `v09h` training on the pod (chain4, folds 1–4); `v08w` pinned and the infer kernel is now 100 % pin-based; #9/#10 still scoring; monitors armed

Tian's instruction (17:58): "implement v09h — all the bits that made the 0.936 notebook good — before #9/#10 score;
when they score, record the findings." Plan (approved 18:05, file `~/.claude/plans/i-want-you-to-fluffy-hellman.md`):
the notebook was re-inventoried cell by cell — **every ingredient of its strongest member is already in `v09h`**
(2–98 % band, per-label attention over all windows, hybrid backbone; 384 px measured ❌, TTA 🔁, calibrator /
clinical residual / gold-58 weights / no-laterality rejected on purpose), so "implement v09h" = **make it the 5-fold
production member**. Decisions: 5-fold only (no new-family arm tonight); Tian re-auths Kaggle ≈ 21:00 for `ship`;
**delete the pod** after ship; **no submission** without his call.

### ⏳ Still in flight as this was written (18:10)

| In flight | What it is | Started | How to check | How to read it |
|---|---|---|---|---|
| **Pod `2wend9j0lr7zf3` — `/workspace/chain4.sh`**: 5-fold `v09h` (fold 0 reused, folds 1–4 train), then `ship v09h` | `sed` set `ARM_FOLDS = (0, 1, 2, 3, 4)` in the pod's `src/` copy; `train v09h` (`RSNA_ARM=v09h RSNA_WORKERS=8`, `ulimit -n 524288`); per-epoch csvs moved to `/kaggle/working/v09h_epochs/`; `ship` (fixed glob, non-fatal) | 18:03 local (16:03:42 UTC) | `ssh root@213.181.111.2 -p 26323 -i ~/.ssh/id_ed25519 'grep -E "STEP\|resumed\|=== v09h fold\|FAILED\|!!\|CHAIN4 DONE" /workspace/chain4.log; python3 /workspace/oof_summary5.py \| grep -v fold0_'` (per-epoch OOF, MCL, LatMen, pred_std for folds 1–4; the log itself is `tee`-buffered) | ✅ seen at 18:05: `resumed fold 0 at epoch 8 (best 0.8683 at epoch 7)` then `=== v09h fold 1 ===`, 0.09 s/study, GPU 88 %. **Expect**: epoch-7 OOF per fold 0.86–0.87 (fold 0 = 0.8683; `v05g`'s fold spread was 0.008), `pred_std` ≈ 0.25, ≈ 50 min/fold → **folds done ≈ 21:25, ship ≈ 21:30**. Suspicious: a fold < 0.85, `pred_std` < 0.1, `Too many open files`, a fold "resumed" at a non-zero epoch other than fold 0. **`ship` will print `!! ship v09h FAILED`** if the token (dies **20:06**) has not been refreshed — expected, not a bug; see next actions |
| **Submission #9** (infer v12, 6 versions, OOF 0.8795) / **#10** (infer v13, 7 versions, 0.8820) | unchanged from the 17:55 entry (refs 55893845 / 55894428) | 17:08 / 17:37 | `kaggle competitions submissions rsna-knee-abnormality-detection --csv \| head -3` | rules in the 17:55 entry (≥ 0.905 carries; 0.900–0.904 🔁; < 0.900 rerun problem). Both were still `PENDING` at 18:09 (1 h after #9 — long for a ~10 min/100-study rerun; if still pending at 20:00 check the kernel versions' status) |

Watchers armed **in this session only**: submissions (`b5qwhz6ej`, 2-min poll) and chain4 (`bqm6l5oz5`, 5-min ssh poll, emits new
step / epoch / failure lines, exits on `CHAIN4 DONE`).

### Where things stand

| | Status |
|---|---|
| Best LB | **0.900** (#8) until #9/#10 score |
| `v09h` | fold 0 shipped (`rsna-knee-ckpt-v09h` v1, 0.8683); **folds 1–4 training** (above) |
| Pins | **`v08w` pinned 18:06** → `tiankljucanin/rsna-knee-ckpt-v08w` (85 MB `_best.pt` + `_oof.csv`, 82 MB listed); every one of the seven members is now a Dataset pin, so `kaggle/rsna-knee-infer/kernel-metadata.json` has **`kernel_sources: []`** — `rsna-knee-train` may be pushed freely again. Eval metadata also lists the pin |
| Tools | `src/fold_oof_summary.py` (new): per-fold + pooled (fold-rank-normalised and raw) + gold-58 + per-label; **verified on `v05g`: 0.8508/0.8429/0.8456/0.8449/0.8503, pooled 0.8467, gold raw 0.8476** (rank-normalised gold 0.8469 — the doc's 0.8476 was the raw figure). `scripts/runpod_bootstrap.sh ship` copies only `fold[0-9]_{best.pt,oof.csv}` (the fold-0 ship had swept 8 per-epoch csvs into the Dataset), fresh ship dir, exit 2 if no checkpoint; same file on the pod (md5-verified) |
| Docs | proposals.md "Rejected" row: the notebook's *primary* CoAtNet arm evaluates **`K_EVAL = 62`** windows (42 = its v4/v9 legacy arms; the docstring's "42" is stale). research.md §2.7.1 already said 62 |
| Repo | `4949c5c` pushed + this entry; committed `src/` unchanged (`ARM_FOLDS = (0,)` locally — the 5-fold value lives only in the pod's copy) |
| Pod | $0.74/h, ≈ $4.5 + tonight's ≈ $2.6; **delete after ship** (Tian's decision) |

### What we talked about and decided

- **Scope reading**: "implement v09h" ≠ port more notebook cells. The inventory (plan file, table "Raptor-member ingredient")
  shows the remaining notebook techniques are new *families* (RadImageNet heads — licence gate; 16-ch gated stem — P-15) or
  rejected LB-tuning machinery; Tian chose the 5-fold production run over queuing a new-family arm.
- **Token**: Tian re-auths ≈ 21:00 rather than creating a legacy `kaggle.json`; the chain treats a failed ship as non-fatal.
- **Pod**: delete (not stop) once the five checkpoints are confirmed in the Dataset and csvs/logs are pulled.
- **No submission** on the 5-fold result by this session — expected LB gain from folds on one vote of seven ≈ +0.002–0.005
  (under the 0.005 floor); it is better spent as the default blend than as a probe.

### What we figured out

1. The notebook's mechanism is fully covered by `v09h` (inventory in the plan file; the CoAtNet member has no folds, no EMA,
   no TTA, no mask, no slot identity — ours is a superset except for the deliberate 130 mm / laterality / in-slot windows).
2. `K_EVAL = 62` for the primary arm — proposals.md corrected; `eval_windows=42` in `v10c` stays as *our* T4 budget cap.
3. The `v05g` "gold 0.8476" was computed on raw pooled probabilities; fold-rank-normalised it is 0.8469 (same to the noise).

### ⏭ Next action, in order

1. **≈ 21:00 (Tian)**: `kaggle auth login --force` locally, then
   `scp -P 26323 -i ~/.ssh/id_ed25519 ~/.kaggle/credentials.json root@213.181.111.2:/root/.kaggle/credentials.json`.
2. **On `CHAIN4 DONE`** (monitor, or the grep above): pull —
   `mkdir -p artifacts/kaggle_out/pod_v09h_5fold && scp -P 26323 -i ~/.ssh/id_ed25519 -r "root@213.181.111.2:/kaggle/working/v09h_fold[0-9]_oof.csv" root@213.181.111.2:/kaggle/working/v09h_epochs root@213.181.111.2:/workspace/chain4.log root@213.181.111.2:/kaggle/working/train_v09h.log artifacts/kaggle_out/pod_v09h_5fold/`
   then `.venv/Scripts/python.exe src/fold_oof_summary.py v09h artifacts/kaggle_out/pod_v09h_5fold --json artifacts/kaggle_out/fold_summaries.jsonl`.
   **Read**: per-fold 0.86–0.87 and spread ≤ 0.01 = faithful replicates; pooled vs `v05g` 0.8467 (expect ≈ 0.865); gold-58 vs
   0.8476 (direction only). A fold < 0.85 → read its epochs in `v09h_epochs/` and `train_v09h.log` before shipping.
3. **Ship**: if the log says `!! ship v09h FAILED`, after step 1: `ssh … 'cd /workspace/RSNA_Knee && bash scripts/runpod_bootstrap.sh ship v09h'`;
   verify `kaggle datasets list -m -s rsna-knee-ckpt-v09h` — size ≈ 5 × 154 MB ≈ 770 MB (was 154 MB); file count, never exit status.
4. **Infer v14**: `kaggle kernels push -p kaggle/rsna-knee-infer` (notebook unchanged — the committed 7-member variant; the push
   re-resolves the Datasets, so the rerun mounts the 5-fold `v09h`). Log must show `v09h (5 folds)` in the `blend: by_version` line,
   15 checkpoints, 2 geometry groups, decode equality, no `!!`. **Submission = Tian's call.**
5. `/update`: experiments.md entry for the 5-fold `v09h` (per-fold, pooled, gold, cost), P-23 status; then
   `mcp delete-pod 2wend9j0lr7zf3` (after the pulls); `/handoff`.
6. **When #9/#10 score** (any time): Scoreboard rows + CLAUDE.md "Best LB" + P-23 + `INFER_MEMBERS` in `src/` → winning set
   (tie → #10); a `project` memory with the two LB numbers and the OOF→LB offset; PushNotification.

### Open decisions for Tian

- Submit the 5-fold blend (infer v14) or keep it as the default for the next new-member submission.
- Legacy `kaggle.json` for unattended pods (traps 20) — still the clean fix.
- Unchanged: make `convnext-tiny-224-hf`, `timm-*`, `rsna-knee-ckpt-*` public before a *final* submission; browser items;
  `crazy_good_rsna.ipynb` keep/delete.

### Things that will bite if forgotten

- **The pod's `src/kaggle_pipeline.py` now has `ARM_FOLDS = (0, 1, 2, 3, 4)`** — a later `train v08w` / `train v10c` there would
  run five folds; revert with `sed -i 's/^ARM_FOLDS = (0, 1, 2, 3, 4)/ARM_FOLDS = (0,)/'` or re-ship the repo.
- Pod bills until deleted; `ship` needs a token valid at run time (20:06 today).
- Bash tool: a `cd` inside a command **persists into later commands** — three commands failed tonight for that reason; prefix
  with `cd "C:/Users/Tian/Desktop/RSNA_Knee" &&`.
- The ssh launch wrapper (`setsid … &` inside an ssh command) does not return — launch with the wrapper and read the log in a
  second ssh call.

## 2026-08-30 (17:55) — ⭐ The 0.936 notebook's mechanism is measured (band + window head + hybrid backbone, not pixels); three new best single models in one day on RunPod + Kaggle; submissions #9 and #10 scoring; P-12 TTA 🔁

Consolidates the 13:10 entry and its 14:20 / 16:45 / 17:35 deltas below. Read this one; the deltas hold the
minute-by-minute record.

### ⏳ Still in flight as this was written (17:55)

| In flight | What it is | Started | How to check | How to read it |
|---|---|---|---|---|
| **Submission #9** — infer v12, six versions (v05a+v05b+v05g+v06c+v08w+v10c), fold-0 proxy OOF 0.8795 | first submission with the c02 / window-attn members and a second family | 17:08 | `kaggle competitions submissions rsna-knee-abnormality-detection --csv \| head -3` (ref 55893845); the scoring rerun is ≈ 10 min per 100 hidden-test studies (v10c is 3 of those), so 1.5–3 h; `PENDING` → `COMPLETE` | Expected **≈ 0.905–0.907** by the +0.02–0.03 OOF→LB offset. ≥ 0.905 = the c02 recipe carries to the LB; 0.900–0.904 = 🔁 (LB floor 0.005); < 0.900 = a *rerun* problem (compare the 3-study `submission.csv` of v12 vs v10 locally), not a model problem. `ERROR` = read the rerun log via `kaggle kernels output tiankljucanin/rsna-knee-infer -p … --file-pattern no_match` (the version's own log shows only the placeholder run) |
| **Submission #10** — infer v13, seven versions (#9 set + v09h), OOF 0.8820 | Tian's instruction: a second submission after `v09h` | 17:37 | same, ref 55894428 | Expected within ±0.003 of #9 (OOF +0.0024). Both scores go into the Scoreboard rows already added (⏳) via `/update`; **default blend = whichever is higher; on a tie prefer #10 (more members, more robust privately)** |
| ~~P-12 focal pass~~ **done 17:44**: focal ≈ mean per member; 4-blend 0.8745 (+0.0023), 7-blend 0.8826 (+0.0006) → 🔁 confirmed, TTA off; csvs in `artifacts/kaggle_out/eval_pod/tta_focal/` | — | 17:38 | `ssh root@213.181.111.2 -p 26323 -i ~/.ssh/id_ed25519 'tail -n 20 /workspace/evals.log; ls /kaggle/working/tta_focal'` | v05a focal = 0.8621 (= mean). Pull `tta_focal/*.csv` to `artifacts/kaggle_out/eval_pod/tta_focal/`, run `blend_check.py` with the four focal files as `--base` (compare 0.8722 / mean 0.8738); fill the ⏳ cells of the P-12 experiments entry. Nothing will change the verdict unless the 4-blend gains > 0.008 |
| **RunPod pod `2wend9j0lr7zf3`** | secure RTX 4090, EUR-IS-2, **$0.74/h, idle after the focal pass (~18:00)**; all inputs (c01 + c02 caches, weights, three ckpt pins), repo at `/workspace/RSNA_Knee` (CRLF-stripped `8aad0c8`), Kaggle token valid to **20:06** | 12:52 | `mcp get-pod 2wend9j0lr7zf3`; SSH as above; `nvidia-smi` | **Decision needed: queue the 5-fold `v09h` (~4 h, ≈ $3) or delete/stop the pod.** To queue: `cd /workspace/RSNA_Knee && sed -i 's/^ARM_FOLDS = (0,)/ARM_FOLDS = (0, 1, 2, 3, 4)/' src/kaggle_pipeline.py` on the pod (check with `grep -n "^ARM_FOLDS"`), then `setsid bash -c 'ulimit -n $(ulimit -Hn); bash scripts/runpod_bootstrap.sh train v09h; bash scripts/runpod_bootstrap.sh ship v09h' > /workspace/chain4.log 2>&1 < /dev/null &` — **`ship` will need a token valid at ~22:00** (re-auth + scp, or a legacy `kaggle.json`). Note `v09h_fold0_best.pt` already exists in `/kaggle/working`; the fold loop skips completed folds only via `_last.pt` resume — check the log's first lines say fold 1 starts, or move fold-0 files aside |

Watchers armed **in this session only**: both submissions, evals.log tail. Kaggle GPU quota this week ≈ 3.5 h left
(v08w 1.5 h + eval v2 0.2 h + infer v12/v13 0.1 h spent today).

### Where things stand

| | Status |
|---|---|
| Best LB | **0.900** (#8) until #9/#10 score |
| Best single model | **`v09h` 0.8683** (CoAtNet-1 @224, c02, window_attn; 50 min on a 4090) > `v08w` 0.8648 > `v10c` 0.8641 > v05a 0.8574 |
| Best blend (OOF) | 7 versions **0.8820** (infer v13 / #10); 6 versions 0.8795 (#9); old 4-version 0.8722 (#8, LB 0.900) |
| ⭐ Finding | **The 0.936 notebook's 0.924 member gains from the 2–98 % band + per-label window attention (+0.007 on the same DINOv2-S) and from the hybrid backbone (+0.004, different errors: menisci), not from 384 px (−0.004 at 3.5× cost).** experiments.md ⭐ entry, research.md §2.7.1 MEASURED block, CLAUDE.md |
| P-12 TTA | 🔁 **closed**: mean 4-blend +0.0016 / focal +0.0023, 7-blend +0.0002 / +0.0006, members +0.003–0.006 each → **not adopted**, `INFER_OVERRIDES = {}` |
| P-24 RunPod | ✅ works end to end (setup, train, ship); ≈ $4.5 spent (pod since 12:52 + 3 min of two dead community pods); traps 29 (CRLF, no public IP on community, `bc`, `pgrep`, `ulimit -n`, `tee` buffering) |
| Checkpoint pins | `rsna-knee-ckpt-v05` (v05a, v05b), `-v05g` (5 folds), `-v06` (v06c), **`-v10c`, `-v09h`** (new, from the pod); **`v08w` is NOT pinned** — infer v12/v13 read it from the `rsna-knee-train` **v17 output**, so ⚠ **do not push `rsna-knee-train` before pinning `v08w`** (`kaggle kernels output tiankljucanin/rsna-knee-train -p … --file-pattern "v08w_fold0_best"` → Dataset `rsna-knee-ckpt-v08w`, add to infer metadata) |
| Kaggle | eval v2 ❌ OOM (traps 28); train v17 ✅ (v08w); infer v12 ✅, v13 ✅ (both clean on the placeholder test); OAuth tokens last **3 h** today (traps 20) |
| Repo | pushed through this handoff; committed `src/` unchanged today (`FORCE_SMOKE = True`, `MODE = "auto"`, `ARMS = [v08w, v09h]`, `INFER_MEMBERS = ["v05a","v05b"]`); `kaggle/rsna-knee-infer/` = the 7-member v13 variant + metadata with all five pins; `kaggle/rsna-knee-eval/rsna-knee-eval.ipynb`, `crazy_good_rsna.ipynb` untracked |
| Local artifacts | `artifacts/kaggle_out/{v17,pod_v10c,pod_v09h}/` OOF csvs (+ per-epoch), `eval_pod/tta_mean/`, `eval_v2/` log, `infer_v12/`, `infer_v13/` |

### What we talked about and decided

- **Cost-efficiency first**: 4090 by cost-per-work; community 4090s failed on public IP → secure at $0.74/h (every
  secure option ≈ $5 per `v10c`, so wall-clock decided). One pod, sequential chain, `v10c` before `v09h`.
- **Kaggle `oof_eval` OOM → measure on the pod** rather than spend quota on an unexplained RAM issue; then, when the
  token died mid-pull, **train first, evals last** so the GPU never waited on a human step.
- Tian: **submit #9 (six versions) now and #10 (seven) after `v09h`**; submissions do not consume GPU quota
  (5/day; 2 left today). Both sent; expectation stated up front that #10 is within noise of #9.
- Tian's questions answered in the log: why a learner can beat its noisy teacher (random vs systematic noise,
  monotone posterior → same ranking → AUC unaffected; phases of memorisation → `best_oof`/EMA; self-training =
  P-17, with orthogonal text-vs-image errors as the reason it should work here); why more epochs / higher LR were
  not pulled mid-run for `v10c` (peak LR already reached; the curve converged).
- **Not copied on purpose**: the notebook's gold-tuned calibrator and clinical residual; 384 px is now measured as
  unnecessary, so `v10c` at 12 epochs is dropped from the plan.

### What we figured out

1. ⭐ **Band + window head + hybrid backbone are the mechanism; resolution is not** (v05a 0.8574 → v08w 0.8648 →
   v09h 0.8683; v10c @384 0.8641) — experiments.md ⭐ entry. Default member recipe changed accordingly.
2. **The three c02 arms are one blending family (ρ ≈ 0.84)** — +0.0098 over the LB blend for three members; new
   diversity needs a new input representation or pretraining (P-23 #3/#4, P-17), exactly the notebook's own lesson.
3. **`v09h` at 50 min/fold on a 4090 makes 5-fold hybrids a $3 job**; the c02 blob loader is 0.09–0.12 s/study
   on NVMe/FUSE vs 0.19 for c01 per-study files.
4. **TTA is redundant with ensembling here** (P-12 🔁) — members +0.005, blend +0.0016.
5. **Six first-RunPod-run traps** (29) and two Kaggle ones (28 `oof_eval` OOM; 20 update: 3-h tokens, pulls die
   silently at expiry).

### ⏭ Next action, in order

1. **Read #9 / #10** when they flip (watchers, or the CSV command above); fill the two ⏳ Scoreboard rows and the
   `CLAUDE.md` "Best LB" via `/update`; set `INFER_MEMBERS` in the committed `src/` to the winning set.
2. **Focal pass** (≈ 18:00): pull `tta_focal/`, `blend_check.py`, fill the ⏳ cells in the P-12 entry (one commit).
3. **Pin `v08w`**: `mkdir -p artifacts/pin_v08w && kaggle kernels output tiankljucanin/rsna-knee-train -p artifacts/pin_v08w --file-pattern "v08w_fold0_(best|oof)"`,
   write `dataset-metadata.json` (`id: tiankljucanin/rsna-knee-ckpt-v08w`), `kaggle datasets create -p artifacts/pin_v08w`
   (traps 21 on Windows), add to `kaggle/rsna-knee-infer/kernel-metadata.json` and `rsna-knee-eval`. Until then, no train pushes.
4. **Pod: 5-fold `v09h` or stop** (Tian). If go: command in the in-flight table; ≈ 4 h → `rsna-knee-ckpt-v09h` gets
   folds 1–4 (`kaggle datasets version`), infer picks up all five automatically (one vote per version).
   Read: per-fold OOF 0.86–0.87; pooled OOF vs v05g's 0.8467; then infer push + submission on Tian's call.
5. After that: P-17 self-training card (targets = ½ teacher + ½ multi-family OOF rank blend; retrain `v09h`
   recipe; judge vs the *original* teacher + LB) and P-23 #3/#4 (a new input representation).
6. **Cost guard**: `mcp delete-pod 2wend9j0lr7zf3` when the pod is no longer needed (`stop-pod` keeps the 120 GB
   volume for ≈ $0.20/GB-month if more arms are planned within days).

### Open decisions for Tian

- 5-fold `v09h` on the pod tonight (~$3, ~4 h, token re-auth at ~22:00 for `ship`) vs stop the pod.
- Which of #9/#10 becomes the default blend once scored (rule above).
- A non-expiring Kaggle API token (`kaggle.json`) for the pod, so `ship` never races the 3-h OAuth token.
- Unchanged: make `convnext-tiny-224-hf`, `timm-*`, `rsna-knee-ckpt-*` public before a *final* submission; browser
  items (rules text, radimagenet T&C, Kaggle caps); `crazy_good_rsna.ipynb` keep/delete.

### Things that will bite if forgotten

- **Pod bills until deleted**; no auto-terminate via the MCP.
- **`v08w` is unpinned** — a `rsna-knee-train` push repoints infer v12/v13's `v08w` source (they are already
  scored/scoring, so only *future* infer pushes are affected, but the committed infer notebook lists `v08w`).
- Kaggle OAuth token: 3 h; pulls/uploads die **silently** at expiry (exit 0) — verify by file count (traps 14/20).
- On the pod: `ulimit -n $(ulimit -Hn)` before any 8-worker training; logs through `tee` are buffered — read
  `/kaggle/working/*_ep*_oof.csv` (`python3 /workspace/oof_summary.py`) for live progress.
- `chain2.log` is dead history; `chain3.log` = today's training; `evals.log` = P-12; `failed_run1/` = the fd crash.
- `blend_check.py` needs the repo `.venv` (system Python lacks scipy).
- The 5-fold `v09h` run must not re-use `/kaggle/working/v09h_fold0_*` blindly — check the resume logic (`_last.pt`
  exists for fold 0 → it will "resume" fold 0 at epoch 8 = skip, which is fine; verify the log).

## 2026-08-30 (13:10) — RunPod lane live: secure 4090 pod runs P-12 `oof_eval` (mean + focal) → `v10c` → `v09h`; Kaggle eval v2 died OOM on member 2; `v08w` (train v17) still running

Tian's instruction at 12:45: "while waiting for Kaggle, check the RunPod MCP and start running the most
cost-efficient progress." The MCP was already OAuth'd (0 pods, $0 spend), so the P-24 runner got its first
real run — and the Kaggle `oof_eval` crash (12:56) was rerouted onto the same pod instead of spending quota.

### 🔄 Update 14:20 — what changed since 13:10 (read this before the table below)

- **`v08w` fold 0 is done (train v17, COMPLETE 14:08, 1.5 h): OOF 0.8648 — best single model**, gold 0.927, 12/12
  labels up in the blend, MCL +0.028 vs v05a; but +0.0044 as a fifth blend member (ρ 0.866 → REJECT by the rule),
  0.8749 replacing v05a. Files: `artifacts/kaggle_out/v17/` (per-epoch OOF csvs + log). Logged: experiments.md
  2026-08-30 `v08w`, P-25/P-26 cards, CLAUDE.md. **No submission** — expected LB gain is under the 0.005 floor;
  decide the blend with `v10c` in hand. The `v08w_fold0_best.pt` lives in the train v17 output (not pinned yet).
- **Kaggle OAuth token expired at 13:30 (3 h lifetime, traps 20 update)**; Tian re-authed at 13:55, the new token
  dies **16:54**. The two c01 pulls on the pod died silently at 13:30 and were restarted at 13:56 (resume; ≈ 14:45
  done). The **`ship` steps on the pod will 401 if they run after 16:54** — re-run them by hand after a re-auth +
  `scp ~/.kaggle/credentials.json root@213.181.111.2:/root/.kaggle/` (or give the pod a non-expiring `kaggle.json`).
- **Chain reordered → `/workspace/chain3.sh` (pid 4056): `train v10c` → `ship` → `train v09h` → `ship` → P-12 evals
  (mean, focal) only if c01 is complete**, else `bash /workspace/evals_only.sh` later. Reason: with the token dead
  the evals' c01 prerequisite had no ETA, and c02 was complete — no idle GPU. `chain2.sh` is dead; `chain.log`
  holds the setup only.
- **`v10c` crashed once at ~3,500 studies with `Too many open files`** (`ulimit -n` 1024; traps 29). Relaunched
  13:52 with `ulimit -n 524288`; run 2 passed the same point. **0.30 s/study → ~18 min train + ~5 min val per
  epoch → fold 0 ≈ 16:50**, `v09h` ≈ 18:30, evals ≈ 18:50. Progress lines are block-buffered through `tee`
  (arrive in bursts) — read `/kaggle/working/v10c_fold0_ep*_oof.csv` for per-epoch OOF, not the log.
- RunPod account SSH key registered (13:17): `ssh root@213.181.111.2 -p 26323 -i ~/.ssh/id_ed25519` works.

### 🔄 Update 17:35 — `v09h` 0.8683 (best single), #9 submitted, infer v13 (7 members) pushed, P-12 evals running

- **Submission #9 = infer v12, 17:08** (v05a+v05b+v05g+v06c+v08w+v10c; OOF 0.8795). Scoring in progress; 3 submissions left today.
- **`v09h` fold 0 = OOF 0.8683 in 50 min** (0.09 s/study) — best single; 7-member blend **0.8820 (+0.0024)**. Shipped as
  `rsna-knee-ckpt-v09h`. Logged: experiments.md 2026-08-30 `v09h`, P-23. Files `artifacts/kaggle_out/pod_v09h/`.
- **Infer v13 pushed 17:31** (`INFER_MEMBERS` = the seven; metadata mounts `-v09h`) → submit as **#10** per Tian's
  instruction once its log is clean (10+1 checkpoints, 2 geometry groups, decode verified, no errors).
- **P-12 evals running on the pod** (`/workspace/evals_only.sh`, log `/workspace/evals.log`; chain3 had skipped them on a
  wrong completeness check — the c01 shard manifests list all 4,407 studies; fixed). Results →
  `/kaggle/working/tta_mean/`, `tta_focal/`; read with `blend_check.py` against the untouched 4 OOF files (base 0.8722).
- **Pod is otherwise idle after the evals (~$0.74/h)** — decision: queue 5-fold `v09h` (~4 h) / 12-epoch `v10c` (~4.5 h) /
  P-17 self-training, or stop the pod.
- Kaggle token expires **20:06**.

### ⏳ Still in flight as this was written (13:10)

| In flight | What it is | Started | How to check | How to read it |
|---|---|---|---|---|
| **RunPod pod `2wend9j0lr7zf3`** (`rsna-knee-train`, secure RTX 4090 24 GB, EUR-IS-2, **$0.74/h**, 64 vCPU, 503 GB RAM, 120 GB NVMe at `/workspace`) | `chain2.sh` (pid 1874): waits for the c02 setup + c01 pulls → **`oof_eval` mean** → **`oof_eval` focal** → `train v10c` (CoAtNet-2 @384, 6–8 h est.) → `ship v10c` → `train v09h` (~4 h est.) → `ship v09h` | pod 12:52; chain2 13:04 | `ssh -i ~/.ssh/id_ed25519_work -p 26323 root@213.181.111.2` then `tail -n 40 /workspace/chain2.log` (steps + evals), `tail -n 30 /kaggle/working/train_v10c.log` (epochs), `grep -v "%|" /workspace/chain.log \| tail` (setup), `nvidia-smi`. From here: `mcp get-pod 2wend9j0lr7zf3` (runtime.gpus util) or `stream-pod-logs` | **Timeline (UTC = local −2 h):** c02 shards done ≈ 11:25 UTC; c01 shards (2115 + 2292 `.npy`, ~0.7 files/s each, parallel) ≈ 11:55–12:00 UTC → evals start ≈ 14:00 local, ~10–20 min for both pools; `v10c` epoch 0 ≈ 14:20 local. **Evals:** `/kaggle/working/tta_mean/{v05a,v05b,v05g,v06c}_fold0_tta_oof.csv` and `tta_focal/…` — pull with `scp -P 26323 -r root@213.181.111.2:/kaggle/working/tta_mean artifacts/kaggle_out/eval_pod/` (same for focal) then `python src/blend_check.py --base v05a=…/tta_mean/v05a_fold0_tta_oof.csv v05b=… v05g=… v06c=…` vs the untouched base **0.8722**: adopt TTA per member only if the 4-version blend gains **> 0.008**; a member *dropping* > 0.008 = offset views off the trained centres. Mean vs focal: same rule, side by side. v05a mean-TTA is already known: 0.8621 (own, vs 0.8574). **`v10c`:** first `s/study` (blob loader on NVMe should be well under Kaggle's 0.19), epoch-0 OOF (v05a's was ~0.78; a plateau < 0.80 by epoch 3 = the window head not learning), CUDA OOM in epoch 0 = drop `train_windows` 24 → 16 in `ARM_V10C` and re-run `bash scripts/runpod_bootstrap.sh train v10c` (resumes nothing — fresh). `ship` publishes Dataset `tiankljucanin/rsna-knee-ckpt-v10c`; **if the Kaggle token has expired by then** (traps 20; access token refreshes via the copied `credentials.json`, but verify) `ship` fails loudly — re-run after `kaggle auth login --force` locally and re-`scp` `~/.kaggle/credentials.json` to `/root/.kaggle/` |
| **`rsna-knee-train` v17** — arm `v08w` | unchanged from the 12:35 entry (DINOv2-S @224, c02, window_attn, 24 windows, 8 ep, fold 0) | 12:36 | `kaggle kernels status tiankljucanin/rsna-knee-train`; expect `COMPLETE` ~14:40–15:00 | exactly as the 12:35 entry: pull with `--file-pattern "(oof\|no_match)"`, `kaggle_log.py … "s/study" "epoch" "EMA score"`, then `blend_check.py --cand v08w=…`; accept own ≥ 0.8574−0.02, ρ < 0.80, gain > 0.008; P-26 claim = MCL / Lateral Meniscus each +0.03 vs v05a's 0.795 / 0.818 |
| **`rsna-knee-eval` v2** | ❌ **ERROR at 12:56** — DataLoader worker OOM-killed (host RAM) 2.5 min into member 2 (`v05b`) after `v05a` scored 0.8621 with mean TTA. Log pulled to `artifacts/kaggle_out/eval_v2/rsna-knee-eval.log`; traps 28; experiments.md P-12 entry. **Do not re-push** — the pod does this measurement | 12:47 | — | — |

Watchers armed **in this session only** (they die with it): pod `chain.log` + `chain2.log` tails over SSH
(filtered to steps / epochs / errors), and a Kaggle status poll for train v17 + eval v2.

### Where things stand

| | Status |
|---|---|
| Best LB / default blend | **0.900** — infer v10, unchanged; 4 submissions left today, nothing submitted |
| RunPod (P-24) | ✅ **live**: MCP OAuth'd, pod above; ~$0.05 spent so far (two unreachable community pods deleted within 3 min each) |
| P-12 measurement | ⏳ on the pod (both pools); Kaggle attempt ❌ (traps 28) |
| P-23 #2 (`v09h`, `v10c`) | ⏳ queued behind the evals on the pod; **order changed to `v10c` first** (the member that decides whether the lane pays; an OOM shows in minutes, and `v09h` is only its 224 probe) |
| Kaggle GPU quota | ≈ 6 h at session start − 0.2 h (eval v2) − `v08w` (~2 h running) ≈ **3.6 h** after v17 |
| Docs | `d390b95`: experiments (P-12 first number), traps 28 + 29, P-12 / P-24 cards + index, CLAUDE.md state line |
| Repo | committed `src/` unchanged (`FORCE_SMOKE = True`, `MODE = "auto"`, `ARMS = [v08w, v09h]`); `kaggle/rsna-knee-eval/rsna-knee-eval.ipynb` and `crazy_good_rsna.ipynb` still untracked |
| Pod-only files (not in the repo) | `/workspace/launch.sh`, `/workspace/chain2.sh`, `/workspace/pull_c01.sh`, logs `/workspace/{chain,chain2,pull_c01,pull_c01b}.log`; repo copy at `/workspace/RSNA_Knee` = `8aad0c8` with CRLF stripped; `/kaggle → /workspace/kaggle` |

### What we talked about and decided

- **GPU choice** = cost per unit of work, not $/h: A5000 ($0.16) and 3090 ($0.22) are 2–3× slower per
  dollar-hour than a 4090 ($0.34 community); 48 GB cards cost 2× for ~½ the throughput and `v10c` already
  has `grad_checkpoint`. Two **community** 4090s turned out to have no public IP and the SSH proxy needs an
  account-registered key (not doable from the MCP) → **secure 4090 at $0.74/h**; every secure option
  lands at ≈ $5 per `v10c` (4090 7 h ≈ A40 12 h ≈ A5000 20 h), so wall-clock decided.
- **One pod, sequential**, not two in parallel: the duplicated 40-min setup is the only saving of
  sequential, but one point of failure is easier to babysit; `v10c` before `v09h` (see table).
- **Eval on the pod instead of a Kaggle retry**: the RAM root cause is not visible in the log, a retry
  costs ~0.5 h of ~3.6 h quota with an unknown fix, and the pod's 503 GB sidesteps it for ~$0.25. The
  GPU idles ~35 min waiting for the c01 pulls — accepted for getting P-12 today rather than after `v10c`.
- **Kaggle auth on the pod** = a copy of the local `credentials.json` (OAuth + refresh token), not an API
  key; the repo was shipped by `git archive | ssh tar -x` so no GitHub credential lives on the pod.
- Nothing submitted; no pipeline code changed.

### What we figured out

1. **The P-24 runner works end to end on a real pod** after four first-run fixes (CRLF, `bc`, community
   IP, `pgrep` self-match) — all in traps 29; none needed a code change in `src/`.
2. **`oof_eval` with several members OOMs Kaggle's host RAM** in member 2 while member 1 is fine (traps 28);
   per-member memory is small, so it is cross-member accumulation — cause open, workaround = off-Kaggle.
3. **v05a with (-1, 0, 1)/mean TTA: 0.8621 vs 0.8574** — +0.0047, under the 0.008 floor, 🔁 alone
   (experiments.md 2026-08-30 P-12 entry). The verdict is the blend, not the member.
4. Kaggle kernel-output pulls are **~30 MB/s for blobs but ~0.7 files/s for per-study files** — the c02
   blob design pays off again off-Kaggle; c01 needs parallel shards.

### ⏭ Next action, in order

1. **When `chain2.log` shows `oof_eval focal: N min`** (both pools done, ≈ 14:20 local): pull both csv
   folders (`scp` line in the table), run `blend_check.py` twice (mean set, focal set) against the four
   untouched OOF files; decision rule: > 0.008 blend gain → set `INFER_OVERRIDES` for the winning pool in
   the infer sed line (CLAUDE.md), push `rsna-knee-infer` (smoke first: `FORCE_SMOKE` variant is not
   possible for infer — check the log for the 9-checkpoint equality line and elapsed h). Log the verdict via
   `/update` (P-12 card → pointer). No submission without Tian.
2. **`v08w` (train v17) completes ≈ 15:00**: read rules in the 12:35 entry; then regenerate the committed
   train notebook `python src/nbgen.py src/kaggle_pipeline.py kaggle/rsna-knee-train/rsna-knee-train.ipynb`.
3. **`v10c` epoch 0 ≈ 14:20–14:40**: check `s/study` and the OOF; if CUDA OOM → `train_windows` 16 (edit
   `ARM_V10C` in `/workspace/RSNA_Knee/src/kaggle_pipeline.py` on the pod *and* locally), re-run
   `bash scripts/runpod_bootstrap.sh train v10c` in a `setsid` shell. When `ship v10c` prints the Dataset:
   add `tiankljucanin/rsna-knee-ckpt-v10c` to `kaggle/rsna-knee-infer/kernel-metadata.json` and
   `kaggle/rsna-knee-eval/kernel-metadata.json`, `scp` the `v10c_fold0_oof.csv`, `blend_check.py --cand`.
4. **Cost guard:** the pod bills until deleted. After `CHAIN2 DONE` (≈ 02:00–04:00 local if all runs),
   pull `/kaggle/working/*_oof.csv`, `*_best.pt` sizes, and the logs, confirm the two Datasets exist
   (`kaggle datasets list -m -s rsna-knee-ckpt`), **then `mcp delete-pod 2wend9j0lr7zf3`** (or `stop-pod` to
   keep the 120 GB volume at ~$0.20/GB-month if another arm is planned). Idle after completion ≈ $0.74/h.

### Open decisions for Tian

- Keep the pod after `v09h` for more arms (5-fold of the winner? `v10c` seeds?) vs delete — ~$18/day if idle.
- ~~Register a public key in RunPod account settings~~ **done 13:17** (`~/.ssh/id_ed25519.pub`; also appended
  to the running pod's `authorized_keys`, so `ssh root@213.181.111.2 -p 26323 -i ~/.ssh/id_ed25519` works).
  The `ssh.runpod.io` proxy now authenticates (use `ssh -t`, exec only, no SCP) → $0.34 community pods are
  usable for launch/tail next time; anything needing SCP still wants a public-IP (secure) host.
- Whether the Kaggle `oof_eval` OOM (traps 28) is worth a fix (one member per kernel is the cheap dodge).
- Unchanged: make `convnext-tiny-224-hf`, `timm-*`, `rsna-knee-ckpt-*` public before a *final* submission;
  browser items (rules text, radimagenet T&C, Kaggle caps); `crazy_good_rsna.ipynb` keep/delete.

### Things that will bite if forgotten

- **The pod is billing** ($0.74/h) with no auto-terminate; the MCP `create-pod` has no `terminateAfter`.
- `chain2` moves each pool's csvs into `tta_mean/` / `tta_focal/` because both runs write the same
  `{v}_fold0_tta_oof.csv` names; the `ship` step copies `*_fold*_oof.csv` — the `_tta_oof` files are not
  matched by that glob (good), but don't leave them in `/kaggle/working` when shipping.
- Kaggle token on the pod: access token expired 13:30 local; the CLI refreshes with the refresh token (it
  did so for the pulls?) — if a `kaggle` command on the pod 401s, re-copy `credentials.json`.
- `/workspace/chain.log` (original chain, parent killed) still receives the `setup` output; `chain2.log`
  is the one to read. The original chain's `train v10c` will **not** auto-start (parent killed on purpose).
- All the traps-29 items when creating a new pod; use `ssh host 'bash -s' <<'EOF'` for remote scripts.
- The CLAUDE.md note "Kaggle token valid until ~22:30" is the *refresh* horizon; the pod copy shares it.

## 2026-08-30 (12:35) — 0.936 notebook diagnosed; cache v2 built (0 GPU h); window-attention head, timm hybrids, mixed-geometry inference, TTA/oof_eval and RunPod runner shipped and smoke-green; STOPPED for the go-ahead on `v08w` fold 0

### ⏳ Still in flight as this was written (12:50, go-ahead given at 12:36)

| In flight | What it is | Started | How to check | How to read it |
|---|---|---|---|---|
| **`rsna-knee-train` v17** — arm `v08w` | P-25 + P-26 first real arm: DINOv2-S @224 on the **c02** cache, `window_attn`, 24 train windows, eval all windows, 8 ep, `best_oof`, fold 0 (`v09h` line sed'd out) | 12:36 | `kaggle kernels status tiankljucanin/rsna-knee-train`; expect `COMPLETE` ~14:40–15:00 (≈ 2 h; guard 8.3 h). Then `kaggle kernels output tiankljucanin/rsna-knee-train -p artifacts/kaggle_out/v17 --file-pattern "(oof\|no_match)"` and `python src/kaggle_log.py artifacts/kaggle_out/v17/rsna-knee-train.log "s/study" "epoch" "EMA score"` | `s/study` on the blob loader (c01 members: 0.19; > 0.35 = FUSE-bound); epoch-0 OOF (v05a: ~0.78); then `python src/blend_check.py --base v05a=artifacts/kaggle_out/v13/v05a_fold0_oof.csv v05b=artifacts/kaggle_out/v13/v05b_fold0_oof.csv v05g=artifacts/kaggle_out/folds_v4/v05g_fold0_oof.csv v06c=artifacts/kaggle_out/v15/v06c_fold0_oof.csv --cand v08w=artifacts/kaggle_out/v17/v08w_fold0_oof.csv`. Accept: own ≥ 0.8574−0.02, ρ < 0.80, gain > 0.008; **P-26 claim = MCL / Lateral Meniscus each +0.03 vs v05a's 0.795 / 0.818**. Suspicious: OOF plateau < 0.80 (window head not learning), or `s/study` > 0.5 (loader) |
| **`rsna-knee-eval` v2** — `oof_eval`, TTA `(-1,0,1)` / **mean** | P-12 measurement: v05a, v05b, v05g (fold 0), v06c re-scored on their 882 held-out studies with 3 slice-offset views | 12:47 | `kaggle kernels status tiankljucanin/rsna-knee-eval`; expect ~35–45 min. Then `kaggle kernels output tiankljucanin/rsna-knee-eval -p artifacts/kaggle_out/eval_v2 --file-pattern "(tta_oof\|no_match)"` | `python src/blend_check.py --base v05a=artifacts/kaggle_out/eval_v2/v05a_fold0_tta_oof.csv v05b=…/v05b_fold0_tta_oof.csv v05g=…/v05g_fold0_tta_oof.csv v06c=…/v06c_fold0_tta_oof.csv` vs the untouched base (0.8722 for the 4-version blend): TTA is adopted per member only if the 4-version blend gains > 0.008; each member's own OOF vs its original (v05a 0.8574, v05b 0.8471, v05g 0.8508, v06c 0.8562). **Then push `artifacts/eval_real_focal.py`** the same way (eval v3) and compare mean vs focal. Suspicious: a member *dropping* > 0.008 (offset views off the trained centres) |

Smoke pushes done first: train v16, infer v11, eval v1 (new slug: mounts found at `/kaggle/input/notebooks/…`,
both caches indexed) — all green. Kaggle token from 10:28 (`--force`), good until ~22:30. GPU spent so far
≈ 0.15 h; the two runs above will use ≈ 2.7 h of the ≈ 6 h. Monitors armed in this session only.
RunPod: the Claude Code plugin `runpod@runpod` 1.2.0 is installed and enabled; Tian is creating the account
and must do `/reload-plugins` and `/mcp → runpod → Sign in` before pods can be driven from here.

### Where things stand

| | Status |
|---|---|
| Best LB / default blend | **0.900** — infer **v10** (`v05a`+`v05b`+`v05g`+`v06c`), unchanged. 4 submissions left today; nothing was submitted this session |
| Diagnosis | done — research.md §2.7.1 (+ CORRECTED block): the gap is a 0.924 single model (CoAtNet-2 @384, 64 slices, 2–98 % band, per-label attention over all windows) + input-representation families; our two weakest labels (MCL 0.836, Lateral Meniscus 0.833) are the ones the discarded outer slices carry (experiments.md 2026-08-30 per-label table) |
| Cache v2 (`c02`, P-26) | ✅ **built**: `rsna-knee-cache2-a..d` v1, 4,407/4,407 studies, 70 blobs, 35.8 GB, 0 decode failures, ~20 min wall, 0 GPU h (experiments.md "Cache v2 built"). c01 kernels untouched |
| Pipeline (P-25, P-23 #2, P-12, P-24) | ✅ shipped + locally verified + Kaggle smoke-green (train v16, infer v11): `cache_scheme`, `window_mode="random"` + `head_type="window_attn"`, `backbone="timm:*"`, `INFER_CACHE_KEYS`/`INFER_MEMBER_KEYS` geometry groups, `tta_offsets`/`tta_pool`/`INFER_OVERRIDES`, `MODE="oof_eval"`, `RSNA_ARM`/`RSNA_TRAIN_ONLY`/`RSNA_WORKERS`/`RSNA_RUNTIME_H`. **Pre- and post-change inference of the existing members is byte-identical** |
| Arms defined | `v08w` (DINOv2-S @224, c02, 24 train windows, window_attn, 8 ep) — Kaggle, ~2 h; `v09h` (`timm:coatnet_rmlp_1_rw_224`, RunPod ~4 h); `ARM_V10C` = `v10c` (`timm:coatnet_rmlp_2_rw_384` @384, eval 42 windows, grad_checkpoint; RunPod 6–8 h) |
| Pins / Datasets | `rsna-knee-ckpt-v06` (`v06c`), `rsna-knee-ckpt-v05g` (5 folds), `timm-coatnet-rmlp-1-rw-224`, `timm-coatnet-rmlp-2-rw-384` — all private, all verified complete. `rsna-knee-train` / `-folds` may now be pushed freely; infer metadata mounts the pins + `rsna-knee-train` only |
| Local checks | `src/cache_selftest.py` (both schemes bit-identical), `src/window_head_test.py`, `src/kaggle_log.py` (log reader) — all green |
| Repo | pushed through `26f070c` + this handoff; `crazy_good_rsna.ipynb` still untracked by design; committed `src/` = `FORCE_SMOKE = True`, `MODE = "auto"`, `ARMS = [v08w, v09h]`, `PRIMARY_ARM = "v08w"`, `INFER_MEMBERS = ["v05a","v05b"]`, `INFER_OVERRIDES = {}` |

### What we talked about and decided

- Tian asked what the 0.936 notebook does better and how to reach/surpass it. Answer (this file's
  12:00 diagnosis, research.md §2.7.1): ensembling *is* the mechanism, but only of members that differ in
  **input representation**, and their fusion sits on a **0.924 single model** we do not have. Same-geometry
  backbones (v06c) buy head-like diversity — measured, not assumed.
- Plan approved (plan file `i-now-want-you-goofy-cake.md`): rebuild the cache first (zero GPU), implement
  table rows #2/#3/#4 (window head, hybrid probe, CoAtNet-2@384), TTA included, docs via `/update`;
  verify locally before any GPU. Tian's choices: **our 6 slots with ragged budgets** (not the notebook's 5),
  **include TTA**, **RunPod** for the heavy members (paid per hour, chosen over free Colab), Kaggle's ~6 h
  → smokes + `v08w` fold 0.
- Deviations from the notebook, on purpose: slot embedding + windows kept inside a slot (theirs cross slot
  boundaries with no slot identity), 130 mm crop / [1, 99] / laterality kept (theirs: 140 mm / [2, 98] /
  **no laterality**) so one test-time builder serves both caches and our measured +0.015 laterality gain
  stays; calibrator, clinical residual and gold-tuned weights **not** copied (Rejected table).
- P-08's K=12 arm (the previous handoff's next step) **withdrawn** — superseded by c02 + random windows.
- Design review changes adopted: header-offset blob reads (no mmap on the FUSE mount), uint8 + indices
  through the DataLoader (not float windows), `eval_windows` cap (42 for CoAtNet-2; same value in
  `oof_eval` and infer), per-arm cache indexing, CACHE vs MEMBER key split, defaults for old checkpoints.
- Rule kept: **no real run and no submission without Tian's go-ahead** — the session stops here.

### What we figured out

1. **The gap decomposes as 0.899 → 0.920 → 0.935 → 0.936** and their DINO branch equals ours; the missing
   +0.015 is one strong hybrid at 384/64 slices; the transformer stack's +0.021 is two more input
   representations (research.md §2.7.1, CORRECTED block for the cell-level facts).
2. **MCL and Lateral Meniscus are our two worst labels (0.836 / 0.833 OOF)** and ConvNeXt does not move them —
   the band, not the backbone, is the suspect (experiments.md 2026-08-30 per-label table → P-26).
3. **The c01 version string never encoded the band**, and the two preprocessing copies had drifted in two
   places — both latent, both closed (traps 23, 24; `cache_selftest.py`).
4. Four CPU kernels run concurrently on Kaggle; a 36 GB cache rebuild costs ~20 min wall and no GPU
   (experiments.md "Cache v2 built"; brainstorm answered).
5. The refactor left the existing members byte-identical (old-vs-new `submission.csv` diff 0.0) and the
   mixed-geometry infer runs on Kaggle in 0.03 h for 9 checkpoints (experiments.md Infrastructure entry).

### ⏭ Next action, in order

1. **Go-ahead needed → real `v08w` fold 0 on Kaggle (~2 h of the ~6 h):**
   ```bash
   export PYTHONUTF8=1 PYTHONPATH=src
   sed -e 's/^FORCE_SMOKE = True/FORCE_SMOKE = False/' -e '/^    ("v09h",/d' src/kaggle_pipeline.py > artifacts/train_real.py
   python src/nbgen.py artifacts/train_real.py kaggle/rsna-knee-train/rsna-knee-train.ipynb
   kaggle kernels push -p kaggle/rsna-knee-train        # -> v17
   ```
   (the `v09h` line is dropped so only `v08w` runs on the T4; `v09h` belongs to RunPod). Read:
   `kaggle kernels output tiankljucanin/rsna-knee-train -p artifacts/kaggle_out/v17 --file-pattern "(oof|no_match)"`,
   `python src/kaggle_log.py artifacts/kaggle_out/v17/rsna-knee-train.log "s/study" "epoch" "EMA score"`.
   First numbers to check: `s/study` on the blob loader (c01 was 0.19; > 0.35 means the FUSE reads bound us),
   the epoch-0 OOF (v05a's was ~0.78). Then
   `python src/blend_check.py --base v05a=artifacts/kaggle_out/v13/v05a_fold0_oof.csv v05b=artifacts/kaggle_out/v13/v05b_fold0_oof.csv v05g=artifacts/kaggle_out/folds_v4/v05g_fold0_oof.csv v06c=artifacts/kaggle_out/v15/v06c_fold0_oof.csv --cand v08w=artifacts/kaggle_out/v17/v08w_fold0_oof.csv`.
   Verdict rule (P-23/P-25/P-26): own OOF ≥ 0.8574 − 0.02, ρ < 0.80, gain > 0.008; **and** per label MCL /
   Lateral Meniscus vs v05a (0.795 / 0.818 for v05a alone; +0.03 each is the P-26 claim). Afterwards
   regenerate the committed train notebook: `python src/nbgen.py src/kaggle_pipeline.py kaggle/rsna-knee-train/rsna-knee-train.ipynb`.
2. **P-12 measurement (~0.5 h GPU), needs a kernel that mounts the caches AND the pins** —
   `kaggle/rsna-knee-eval/kernel-metadata.json` exists (train's sources + the three ckpt pins; never pushed
   yet): `sed -e 's/^FORCE_SMOKE = True/FORCE_SMOKE = False/' -e 's/^MODE = "auto"/MODE = "oof_eval"/' -e 's/^INFER_MEMBERS = \[.*\]/INFER_MEMBERS = ["v05a", "v05b", "v05g", "v06c"]/' -e 's/^INFER_OVERRIDES = {}/INFER_OVERRIDES = {v: {"tta_offsets": (-1, 0, 1), "tta_pool": "mean"} for v in ["v05a", "v05b", "v05g", "v06c"]}/' src/kaggle_pipeline.py > artifacts/eval.py`,
   nbgen → push; a second push with `"tta_pool": "focal"`. Pull `*_tta_oof.csv`, run `blend_check.py` with
   the four TTA files as `--base` vs the four originals; adopt TTA per member only if the 4-version blend
   gains > 0.008 (then set `INFER_OVERRIDES` in the infer kernel). Do **not** push `oof_eval` on the
   `rsna-knee-train` slug — it would repoint `v08w`'s mount.
3. **RunPod (Tian executes; ~$ per hour):** pod with ≥ 24 GB (4090/A5000) or A100, ≥ 60 GB local disk, a
   CUDA PyTorch image; `git clone`, `export KAGGLE_USERNAME KAGGLE_KEY`, `bash scripts/runpod_bootstrap.sh setup`
   (~40 min: CSVs, labels, weights, four c02 shards — check the printed file counts/GB), then
   `bash scripts/runpod_bootstrap.sh train v09h` (~4 h), `... ship v09h`; then `train v10c` (6–8 h; set
   `grad_checkpoint` is already on) and `ship v10c`. Add each `rsna-knee-ckpt-<arm>` to
   `kaggle/rsna-knee-infer/kernel-metadata.json`, pull the `_oof.csv`, `blend_check.py` as above.
4. Only after a member passes the rule: infer push with the new `INFER_MEMBERS`, then submission on
   Tian's explicit call.

### Open decisions for Tian

- Go-ahead for `v08w` fold 0 now (≈ 2 h of the ≈ 6 h left this week), and for the P-12 `oof_eval` pass (~0.5 h).
- RunPod pod size/price; when to run `v09h` (probe) vs going straight to `v10c`.
- Make `convnext-tiny-224-hf`, the `timm-*` and `rsna-knee-ckpt-*` datasets **public** before any *final*
  submission relies on them (competition rule).
- Browser items unchanged: rules text (data off-platform for the RunPod cache; winner licence),
  radimagenet.com T&C, Kaggle output cap / Dataset size cap / max attached sources.
- `crazy_good_rsna.ipynb` keep/delete (still untracked).

### Things that will bite if forgotten

- **Do not regenerate `kaggle/rsna-knee-cache-a/-b` notebooks** — `cache_pipeline.py` now defaults to c02
  (traps 27; done and reverted once today). c02 kernels are `rsna-knee-cache2-{a,b,c,d}`.
- `python src/cache_selftest.py` before any push touching preprocessing (traps 24); `window_head_test.py`
  for the window/timm path.
- The committed `rsna-knee-infer` notebook is the standard `["v05a","v05b"]` variant; the smoke pushed as
  infer v11 used a sed'd copy — infer v10 remains the scored one.
- `MODE="oof_eval"` needs the caches mounted; the infer kernel does not mount them (item 2 above).
- `INFER_OVERRIDES` accepts MEMBER keys only; `eval_windows` must be equal in `oof_eval` and infer.
- Kaggle `kernels output` of the c02 shards: verify by file count (17–19 blobs + csvs) and GB, never by exit
  status (traps 14) — `runpod_bootstrap.sh setup` does this.
- Local clock vs my notes: the docs' "12:20 / 12:45 / 13:15" stamps were written before checking the clock;
  the real local times were ≈ 11:50 (cache push), ≈ 12:10 (caches done), ≈ 12:15 (train v16), ≈ 12:28 (infer v11).

## 2026-08-30 (10:45) — Session closed: docs reconciled through `8ca03ea`; P-24 (free compute) written; next GPU spend is P-23 #2a (K=12) on the ~6 h left

Closes the overnight session. The 10:30 entry below holds the results narrative (`v06c` → LB 0.900,
`v07s` dead); this one records the final `/update` pass and the decision state Tian left with.

### ⏳ Still in flight — nothing

`rsna-knee-train` v15, `rsna-knee-stack` v2, `rsna-knee-infer` v10 all `COMPLETE`; submission #8
scored **0.900**. No kernels running, no background watchers, 4 submissions left today. Kaggle token
valid until ~22:30 (`kaggle auth login --force` at 10:28).

### Where things stand

| | Status |
|---|---|
| Best LB / default blend | **0.900** — infer **v10**, by-version `v05a`+`v05b`+`v05g`+`v06c` (🔁 +0.004 over 0.896, under the 0.005 floor). Submit nothing else without a new member |
| Backlog head | **P-23** multi-family fusion (#1 `v06c` done, #3 `v07s` ❌, **#2a K=12 next**, #2b 336–384 hybrid needs P-24 compute, #4 RadImageNet licence-gated) · **P-24** compute expansion 💡 (2×T4 + Colab-free runner) |
| GPU quota | ≈ **6 h** left this week (resets ~2026-09-05) |
| Pins | `v05a`/`v05b` → Dataset `rsna-knee-ckpt-v05`; `v05g` only in `rsna-knee-folds` v4; `v06c` only in `rsna-knee-train` v15 — **pin `v06c` before the next train push** |
| Docs | experiments/proposals/traps/brainstorm/CLAUDE.md all reconciled and pushed (`8ca03ea`); `crazy_good_rsna.ipynb` untracked by design |

### What we talked about and decided

- Tian asked for off-Kaggle options; **will not pay for GPUs**; realistic supplement is **free Colab**.
  Verified by web search: a Google One storage plan carries no Colab compute units; Colab Pro is a separate
  $9.99/100-unit product; the free US-student Colab Pro is closed. → P-24 card, brainstorm open question.
- Key design fact recorded in CLAUDE.md: **training needs only the 21 GB cache + CSVs + weights, never
  the DICOMs**; inference stays a Kaggle notebook; checkpoints return as a private Dataset (proved by the
  v05 pin). The pipeline already resumes per epoch, which is what makes Colab's disconnects survivable.
- P-08's K sweep, previously downgraded, is **re-raised as P-23 candidate #2a**: after `v06c` (strong,
  head-like) and `v07s` (different, weak), the missing diversity is in the *input*; K=12 from the existing
  16-slice cache is the one input change that needs no rebuild and fits the remaining ~6 h.

### What we figured out

Nothing new was measured in this closing pass; the findings are in the 10:30 entry and experiments.md.
One operational: Bash heredocs with long Markdown payloads fail to parse in this tool environment —
write the patch to a `.py` file and run it (used for every docs patch today).

### ⏭ Next action, in order

1. **Pin `v06c`** (no GPU): `kaggle kernels output tiankljucanin/rsna-knee-train -p artifacts/kaggle_out/v15_ckpt --file-pattern "v06c_fold0_best"`,
   copy into `artifacts/ckpt_pin/rsna-knee-ckpt-v06/` with a `dataset-metadata.json` (title ≤ 50 chars),
   `cd` into it, `kaggle datasets create -p .` (traps 21), add `tiankljucanin/rsna-knee-ckpt-v06` to
   `kaggle/rsna-knee-infer/kernel-metadata.json`. Then `rsna-knee-train` may be pushed again.
2. **P-23 #2a — `v08k`** (~3.5 h of the ~6 h): in `src/kaggle_pipeline.py` set
   `ARMS = [("v08k", {"slices_per_slot": 12, "cache_jitter": True, "epochs": 8})]`, `PRIMARY_ARM = "v08k"`;
   `FORCE_SMOKE = True` push first, check `cache: 4407 studies indexed` and `slices/slot 12` in the log; then the
   sed'd `FORCE_SMOKE = False` push. Read: `kaggle kernels output tiankljucanin/rsna-knee-train -p artifacts/kaggle_out/v16 --file-pattern "(oof|no_match)"`,
   then `python src/blend_check.py --base v05a=artifacts/kaggle_out/v13/v05a_fold0_oof.csv v05b=artifacts/kaggle_out/v13/v05b_fold0_oof.csv v05g=artifacts/kaggle_out/folds_v4/v05g_fold0_oof.csv v06c=artifacts/kaggle_out/v15/v06c_fold0_oof.csv --cand v08k=artifacts/kaggle_out/v16/v08k_fold0_oof.csv`.
   Accept on ρ < 0.80 vs the 4-version blend **and** gain > 0.008; expect ~0.38 s/study (2× the triplet count).
3. **P-24, no GPU, whenever Tian picks the path**: (a) 2×T4 `DataParallel` over `batch_studies=2` — smoke,
   then one fold-0 arm vs its single-T4 twin (same seed): adopt if ≥ 1.5× faster with OOF within 0.008;
   (b) the Colab runner (env-var paths, CUDA requirements, `colab_bootstrap.ipynb`) after turning the two
   cache outputs into one **private** Dataset in the browser.
4. Log every result with `/update`; a member is submitted only if it passes the P-23 rule, or on Tian's
   explicit call as with #8.

### Open decisions for Tian

- Compute path (P-24): Colab free + 2×T4, or Kaggle-only.
- Spend the ~6 h on `v08k` now, or hold them until the compute question is settled.
- Make `convnext-tiny-224-hf` and the `rsna-knee-ckpt-*` datasets public before any *final* submission.
- Browser-only items unchanged (rules text, radimagenet.com T&C, public-checkpoint dataset licences,
  `crazy_good_rsna.ipynb` keep/delete).

### Things that will bite if forgotten

- All items of the 10:30 entry (mount repointing, `--force` login, member weights in infer, Monitor vs
  10-min Bash waits, committed `src/` is the smoke/`["v05a","v05b"]` configuration, never mount `v07s`).
- `slices_per_slot` is a **geometry key**: a `v08k` member cannot share a decode-once blend with the K=6
  members as the infer code stands (`INFER_GEOM_KEYS`); if `v08k` is accepted, the infer loop must apply
  `slices_per_slot` per member the way it now applies `stack_mode` — the cached array is the same.

## 2026-08-30 (10:30) — Overnight P-23 read: `v06c` ConvNeXt-T → **LB 0.900** (#8, +0.004 🔁), `v07s` 16-channel dead; nothing running

The overnight session died after launching both runs (the laptop went down), so the chain in the
00:50 entry was finished by hand at 09:13–10:30. Findings are logged: experiments.md ("ConvNeXt-Tiny
member `v06c`", "16-slices-as-channels … `v07s`", Submissions #8, Scoreboard), proposals.md (P-10 →
pointer, P-15, P-23), traps.md (20 addendum, 22), CLAUDE.md state line. This entry is state only.

### ⏳ Still in flight — nothing

`rsna-knee-train` v15, `rsna-knee-stack` v2, `rsna-knee-infer` v10 all `COMPLETE`; #8 scored. No
background watchers alive (the previous session's Monitors died with it).

### Where things stand

| | Status |
|---|---|
| Best LB | **0.900** — #8, `rsna-knee-infer` **v10**: by-version blend `v05a` attn + `v05b` concat + `v05g` 5-fold + `v06c` ConvNeXt-T (8 checkpoints, 4 votes). **+0.004 over 0.896 is 🔁 under the 0.005 floor**, but it is the default blend now. 4 submissions left today |
| P-23 candidates | #1 `v06c` ✅ strong (own OOF 0.8562 = parity with `v05a`) but head-like (ρ 0.83, blend +0.006, 10/12 labels up) · #3 `v07s` ❌ dead as built (OOF 0.74 on all 5 folds, 4.8 h) · #2 high-res/many-slice hybrid 💡 · #4 RadImageNet 💡 (licence) |
| OOF→LB offset | held again: fold-0 proxy 0.8722 → 0.900 (+0.028; n=5) |
| Checkpoint mounts | `v05a`/`v05b` pinned in Dataset `tiankljucanin/rsna-knee-ckpt-v05` (private); `v05g` still only in `rsna-knee-folds` v4 output (**do not push that slug**); `v06c` only in `rsna-knee-train` v15 output (**do not push that slug without pinning `v06c` first**); infer metadata lists both datasets + `convnext-tiny-224-hf` |
| GPU quota | ≈ **6 h** left this week (resets ~2026-09-05): v15 1.86 h + stack 4.79 h + smokes since the 00:00 entry's ~13 h |
| Kaggle token | re-logged 10:28 with `kaggle auth login --force`; expires ~22:30 |
| Repo | pushed through `7f1b01f`; `crazy_good_rsna.ipynb` still untracked by design |

### What we talked about and decided

- **Read `crazy_good_rsna.ipynb` (0.936) cell by cell** → its DINOv2 branch ≈ 0.899 equals our 0.896;
  the gap is three more families. Tian made **P-23 multi-family fusion the #1 card** (research.md §2.7.1).
- Tian chose the overnight plan "5 folds + auto-blend + auto-submit" for the 16-channel member, knowing
  it needed the laptop on; the laptop did not stay on, training finished anyway, evaluation was done at 09:13.
- **`v06c` failed the pre-registered rule narrowly (ρ 0.831 vs 0.80, +0.0059 vs 0.008); Tian chose to
  submit and let the LB arbitrate.** It came back +0.004: inside the noise floor, so recorded 🔁 — a real
  but head-sized gain, not a family-sized one.
- Compute: Tian does not want to pay for GPUs; Colab free is the realistic supplement (a Google One
  storage plan includes **no** Colab compute units — verified by search). Off-Kaggle training needs only
  the 21 GB cache + CSVs + weights, never the DICOMs; inference stays on Kaggle. Not built yet.

### What we figured out

1. **A second family at 224/6 slots buys head-like diversity, not family-like.** ConvNeXt-T matches
   DINOv2-S alone (0.8562 vs 0.8574) but ρ 0.77–0.83 against the heads — the same band as attn-vs-concat
   — so the blend gain is +0.006 OOF / +0.004 LB. Diversity has to come from the *input* (resolution,
   slice count/band, stack representation), as the 0.936 notebook's own +0.001 three-backbone blend also
   said. → experiments.md "ConvNeXt-Tiny member `v06c`".
2. **The 16-channel stack as a linear patch embed does not learn** (0.74 on every fold, still climbing
   at epoch 8, no overfit). The representation is not disproven; the stem is. A retry needs a non-linear
   stem (`DepthCompress`-style) at ≥ 1e-3. → experiments.md "`v07s`".
3. `best_oof` earned its keep again: `v06c` peaked at epoch 3 (0.8562) and decayed to 0.8416 by epoch 7.
4. Two operational traps: a member family's *weights* must be mounted in the infer kernel (infer v9
   died on the 8th member; now pre-resolved) → traps 22; an expired token still says "already
   logged-in" → traps 20 addendum.
5. Data loading, not the GPU, bounds our epoch time: 6× fewer forwards gave only 1.7× (0.19 → 0.11
   s/study). Any off-Kaggle box needs local NVMe + more workers to be worth it.

### ⏭ Next action, in order

1. **Decide the compute path** (open decision below). If Colab: build the off-Kaggle runner — env-var
   paths (`RSNA_CACHE_DIR`, `RSNA_DATA_DIR`, `RSNA_MODELS_DIR`) replacing `artifacts/cache_local` /
   `data` / `models` in `src/kaggle_pipeline.py`, CUDA requirements, `FORCE_SMOKE` honoured off-Kaggle,
   `num_workers` 8, and a `colab_bootstrap.ipynb` (mount Drive → copy cache to `/content` → clone →
   run → `kaggle datasets version` the checkpoints). First turn the two cache outputs into one **private**
   Dataset (browser: "create dataset from notebook output" on `rsna-knee-cache-a`/`-b`). No GPU needed.
2. **Cheap Kaggle win regardless:** 2×T4 sessions count once against the 30 h — add `DataParallel`
   over `batch_studies=2` (~30 lines), smoke, measure s/study against 0.19. Rule: adopt if ≥ 1.5× faster
   at identical OOF (same seed; within the 0.008 floor).
3. **P-23 candidate #2 on the remaining ~6 h** — the only lever that changes the *input* without a cache
   rebuild is more slices from the existing 16: `ARMS = [("v08k", {"slices_per_slot": 12, "cache_jitter": True, "epochs": 8})]`
   on `rsna-knee-train` **after pinning `v06c`** (pull `v06c_fold0_best.pt` with
   `kaggle kernels output tiankljucanin/rsna-knee-train -p artifacts/kaggle_out/v15_ckpt --file-pattern "v06c_fold0_best"`,
   then a Dataset like the v05 pin). Cost ~2× a normal arm ≈ 3.5 h. Read with
   `python src/blend_check.py --base v05a=… v05b=… v05g=… v06c=artifacts/kaggle_out/v15/v06c_fold0_oof.csv --cand v08k=…`;
   accept on the P-23 rule (ρ < 0.80 vs the 4-version blend, gain > 0.008).
4. If none of the above: infer v10 (0.900) is the standing submission; do not spend submissions on
   weight tuning.

### Open decisions for Tian

- Compute path: Colab free (+ the 2×T4 change) vs staying Kaggle-only with ~6 h/week.
- Whether the ~6 h go to candidate #2 (K=12) now or wait for the Colab runner.
- Make `convnext-tiny-224-hf` and `rsna-knee-ckpt-v05` public before any *final* submission relies on them.
- Browser items still open: rules text (off-platform data use, winner licence), radimagenet.com T&C,
  licence fields of the `raptor-knee-*` / tonylica datasets (only if the "mount their checkpoints" path
  is ever wanted), `crazy_good_rsna.ipynb` keep/delete.

### Things that will bite if forgotten

- **Pushing `rsna-knee-train` or `rsna-knee-folds` repoints infer v10's mounts** (`v06c`, `v05g` live
  only there). Pin before pushing.
- **`kaggle auth login` lies when the token is expired — use `--force`** (traps 20 addendum).
- **Every member family needs its weights dataset in the infer kernel** (traps 22).
- Background Monitors/waits die with the session; Bash `run_in_background` waits are capped at 10 min —
  use Monitor (persistent) for kernel waits.
- `src/kaggle_pipeline.py` is committed with `FORCE_SMOKE = True`, `STACK_RUN = False`,
  `INFER_MEMBERS = ["v05a","v05b"]`; the real kernels are sed'd copies (infer v10 = `["v05a","v05b","v05g","v06c"]`).
- `rsna-knee-stack` v2 output holds five `v07s` checkpoints — ❌, never mount them.

## 2026-08-30 (00:50) — Overnight P-23 chain: `v07s` 16-channel member (5 folds) IN FLIGHT as `rsna-knee-stack` v2, `v06c` still running

Tian's go-ahead at ~00:35: "5 fold + auto-blend + auto-submit". The session was told to leave the
laptop on; if it went down, **training still completes on Kaggle** and only the local steps below
remain. Supersedes the in-flight table of the 00:00 entry; the rest of that entry still holds.

### ⏳ In flight

| In flight | What it is | Started | How to check | How to read it |
|---|---|---|---|---|
| **`rsna-knee-stack` v2** — arm `v07s` | **P-23 candidate #3**: DINOv2-S with the 16 cached slices as input channels (`stack_mode="channels"`, patch embed widened 3→16, `lr_stem` 2e-4), concat head, jitter, 8 ep, `best_oof`, **folds 0–4** | 00:46 | `kaggle kernels status tiankljucanin/rsna-knee-stack`; expect `COMPLETE` ~03:00–04:30 (unknown throughput; guard 8.3 h). Then `kaggle kernels output tiankljucanin/rsna-knee-stack -p artifacts/kaggle_out/stack_v2 --file-pattern "(oof|no_match)"` | `cache: 4407 studies indexed`; `patch embedding widened 3 -> 16`; per-fold `EMA score … -> checkpoint` lines; `s/study` should be well under the triplet members' 0.18 |
| **`rsna-knee-train` v15** — arm `v06c` | P-10 / P-23 candidate #1 (see 00:00 entry) | 23:57 | as in the 00:00 entry | as in the 00:00 entry |

### What is already done tonight

- `tiankljucanin/rsna-knee-ckpt-v05` (private Dataset) holds `v05a_fold0_best.pt` + `v05b_fold0_best.pt`
  — the two-head blend is re-submittable; `kaggle/rsna-knee-infer/kernel-metadata.json` already lists it
  and `rsna-knee-stack` as sources (not pushed yet).
- `src/blend_check.py --base v05a=… v05b=… --cand v06c=… v07s=…` applies the P-23 rule and prints the
  per-label table; verified to reproduce 0.8574 / 0.8471 / blend 0.8670 / ρ 0.773.
- Local + Kaggle smoke of the channels member green; code committed (`e67cc36`).

### ⏭ Remaining chain (the session runs it; if the machine went down, do it by hand)

1. v15 done → `kaggle kernels output tiankljucanin/rsna-knee-train -p artifacts/kaggle_out/v15 --file-pattern "(oof|no_match)"`,
   then `blend_check.py --cand v06c=artifacts/kaggle_out/v15/v06c_fold0_oof.csv`.
2. stack v2 done → pull `--file-pattern "(oof|no_match)"` into `artifacts/kaggle_out/stack_v2`, then
   `blend_check.py` with `--cand v07s=…/v07s_fold0_oof.csv` (and `v06c` if it passed).
3. Accepted set → sed `INFER_MEMBERS = […]` (always includes `v05a`,`v05b`,`v05g`), `MODE="infer"`,
   `FORCE_SMOKE=False` into the infer notebook, push `kaggle/rsna-knee-infer`, wait for `COMPLETE`, read
   the log (`infer members`, `rank correlation`, `blend: by_version`, `wrote …submission.csv`), then
   `kaggle competitions submit rsna-knee-abnormality-detection -k tiankljucanin/rsna-knee-infer -v <ver> -f submission.csv -m "…"`.
   **Rule vs 0.896: < 0.005 is 🔁.** Nothing rejected gets submitted.
4. `/update` (experiments.md ⏳ entries → verdicts; P-23 statuses) and `/handoff`.

---

## 2026-08-30 (00:00) — `v06c` (ConvNeXt-Tiny, P-10) real fold-0 run IN FLIGHT as `rsna-knee-train` v15

Tian approved the launch at 23:55. This supersedes the "awaits go-ahead" entry below (same
session, minutes apart); everything else there still holds. **Read this entry, then the 23:35 one
for the blend results.**

### ⏳ Still in flight as this was written (00:00)

| In flight | What it is | Started | How to check | How to read it |
|---|---|---|---|---|
| **`rsna-knee-train` v15** — arm `v06c` | **P-10**: ConvNeXt-Tiny backbone (HF `facebook/convnext-tiny-224`), concat head, `cache_jitter`, **8 epochs**, `ckpt_policy=best_oof`, `lr_backbone=1e-4` (LLRD 0.75 per stage), fold 0 only, from the cache. The first non-DINOv2 member | 2026-08-29 **23:57** | `kaggle kernels status tiankljucanin/rsna-knee-train` — expect `COMPLETE` around **02:00–02:15** (if the token has expired: `kaggle auth login`, traps 20). Then `kaggle kernels output tiankljucanin/rsna-knee-train -p artifacts/kaggle_out/v15 --file-pattern "(oof|no_match)"` | **First**: `cache: 4407 studies indexed` and the arm banner `v06c … backbone convnext_tiny … folds (0,) epochs 8`. **Throughput**: `N studies in Ns = X s/study` should read ~0.17–0.25 (the smoke's 3.51 on 4 studies was cuDNN warm-up); ~1.0+ means the cache path was not taken — stop and investigate. Eight `fold 0 epoch k:` lines; `EMA score … -> checkpoint = epoch k` / `not taken` lines show `best_oof` at work — the checkpointed epoch is whichever peaked. **Own OOF** (`auc_soft` at the checkpointed epoch) must be ≥ ~0.83 to be usable; DINOv2 concat sits at 0.847–0.851. If the run is `ERROR`, pull the log with `--file-pattern no_match` and read the tail |

### Where things stand

| | Status |
|---|---|
| Best LB | **0.896** (infer v5 = two heads; infer v8 = two heads + five concat folds; prefer v8). 1 submission was left for 2026-08-29; quota resets 02:00 local |
| P-10 | 🔧 code shipped (`c6ed3cd`), smoke green (train v14), **real run = v15 in flight** |
| ⚠️ Infer mounts | **v15 repoints infer v5/v8**: `rsna-knee-infer` mounts the *latest* `rsna-knee-train` output, which will now hold only `v06c_fold0_*`. Scores on the board are unaffected, but a **re-run/re-submit of the two-head blend would fail loudly** until the DINOv2 checkpoints (`v05a`/`v05b` from train v13, `v05g` from folds v4) are pinned to a Kaggle Dataset. Local copies of `v05a_fold0_best.pt` / `v05b_fold0_best.pt` are in `artifacts/kaggle_out/v13/` (88 MB each) — a Dataset can be built from them |
| GPU quota | ~13 h at 23:30 minus v15 (~2 h) → **~11 h** for the week (resets ~2026-09-05) |
| Repo | pushed; `crazy_good_rsna.ipynb` in the root is Tian's, untracked, not committed |

### What we talked about and decided

- Tian chose P-10 (a second architecture family) over the 9 h attention 5-fold after #6/#7 showed
  folds add nothing on top of head diversity. Go-ahead for the real run given at 23:55.
- HF ConvNeXt-**Tiny** via our own private Dataset (no official Kaggle Model exists; Tiny costs
  the same as ViT-S; LayerNorm-only so batch-of-1 is safe); one change per arm (head stays concat).
- Publishing hit two Windows CLI traps → traps 21.

### ⏭ Next action, in order

1. **Read v15** (table above). Then, on fold 0, the three-way check — same Python as the 23:35
   blend check, now with `c = artifacts/kaggle_out/v15/v06c_fold0_oof.csv`:
   own OOF; **ρ(c, v05a) and ρ(c, v05b)**; rank-mean a+b+c (equal votes) vs a+b **0.8670**.
   **Adopt** only if ρ < 0.77 against both **and** the 3-way blend clears **+0.008**, with own OOF
   ≥ ~0.83. Log with `/update` (P-10 card → pointer; experiments.md entry with the ρ table).
2. **If adopted**: pin the DINOv2 checkpoints to a Dataset (see the ⚠️ row), add
   `tiankljucanin/rsna-knee-train` (now holding v06c) + that Dataset + `rsna-knee-folds` to the
   infer kernel's sources, set `INFER_MEMBERS = ["v05a","v05b","v05g","v06c"]` via the sed, push,
   read the log (4 versions, `by_version` blend), submit once. Rule vs 0.896: < 0.005 is 🔁.
3. **If ρ ≈ 0.84** (another concat-like profile): the family bet fails at this size; next
   diversity bets are input geometry (P-11 336 px, P-08 more slices) or DINOv3, not more of this.
4. Regardless: make `convnext-tiny-224-hf` public before any final submission uses it.

### Open decisions for Tian

- Dataset pin for the DINOv2 weights (needed for any new two-head submission).
- `crazy_good_rsna.ipynb`: read for the 0.056 gap, ignore, or delete.
- Delete `artifacts/kaggle_out/v9smoke/` (1.8 GB).

### Things that will bite if forgotten

- **Token expires ~12 h after `kaggle auth login`** (last login 22:01 on 08-29 → expect failure
  ~10:00 on 08-30; the error text blames the slug — traps 20).
- **Mid-run logs are browser-only**; `kernels output`/`logs` are blank until the run ends.
- **Infer v5/v8 now point at v06c's output** (⚠️ row). **Infer v7 never** (flat 7-member).
- `src/kaggle_pipeline.py` is committed with `FORCE_SMOKE = True`; v15 ran the sed'd `False`.
- Everything in the 23:35 / 22:35 lists still applies.

---

## 2026-08-30 (00:05) — P-10 ConvNeXt-Tiny member implemented and smoke-green; real fold-0 run awaits go-ahead

Code through the "P-10: ConvNeXt-Tiny as a second backbone family" commit. Nothing new is
measured; this entry is state. The 23:35 and 22:35 entries below still hold.

### ⏳ Still in flight — nothing running

`rsna-knee-train` v14 (smoke, `COMPLETE`), no submission pending. **The real `v06c` run has NOT
been launched** — the try-out rule requires Tian's explicit go-ahead in a fresh message.

### Where things stand

| | Status |
|---|---|
| Best LB | **0.896** (infer v5 = two heads; infer v8 = two heads + five concat folds — prefer v8 as the default blend) |
| P-10 code | ✅ `Config.backbone` ∈ {`dinov2`, `convnext_tiny`}; `BACKBONES` resolves weights per family; `KneeNet` loads either; LLRD per ConvNeXt stage; infer members carry their family |
| P-10 weights | ✅ private Kaggle Dataset `tiankljucanin/convnext-tiny-224-hf` (HF `facebook/convnext-tiny-224`, Apache-2.0, 111 MB); mounted at `/kaggle/input/convnext-tiny-224-hf` in v14. **Must become public (or be replaced by an official Model) before a final submission relies on it** |
| P-10 smoke | ✅ local CPU + Kaggle v14 green: cache indexed, arm banner `v06c … backbone convnext_tiny`, `backbone LR range 2.37e-05 .. 1.00e-04 over 4 blocks`, inference `v06c/fold0 (convnext_tiny, concat)` 8 s/100 studies |
| GPU quota | ~13 h left this week (v14 smoke ≈ 0.1 h) |
| Repo | pushed; **`crazy_good_rsna.ipynb` in the root is Tian's, untracked, deliberately not committed** |

### What we talked about and decided

- After #6/#7 (folds +0.009 alone, +0.000 on top of heads) Tian chose the "third source of diverse
  errors" — P-10 — over the 9 h attention 5-fold.
- **HF ConvNeXt-Tiny over timm**: no official HF/timm ConvNeXt exists on Kaggle Models; publishing
  the Apache-2.0 HF checkpoint as our own Dataset keeps the code path identical to DINOv2
  (`from_pretrained(dir)`), needs no new dependency, and avoids the BatchNorm-at-batch-1 problem
  the card warned about (ConvNeXt is LayerNorm-only).
- **Tiny, not Small**: 4.5 GFLOPs ≈ ViT-S/14's 4.6, so the arm costs the same ~0.17–0.2 s/study;
  Small would double it for an unmeasured gain.
- **8 epochs under `best_oof`**, concat head, jitter, `lr_backbone = 1e-4` (card value; an
  ImageNet-supervised CNN tolerates ~5× the LR DINOv2's SSL features need). One change per arm:
  the head stays concat so the comparison to `v05b`/`v05g` isolates the family.
- Kaggle CLI dataset upload has two Windows traps (title ≤ 50 chars; run `kaggle datasets create
  -p .` from *inside* the directory or it builds a bad temp path) — noted below, not yet in traps.md.

### ⏭ Next action, in order

1. **On Tian's go-ahead, launch the real `v06c` run** (~1.8–2.0 h; the smoke's 3.51 s/study on 4
   studies is CUDA/cuDNN warm-up, not throughput — inference ran at the ViT members' speed):
   ```bash
   sed 's/^FORCE_SMOKE = True/FORCE_SMOKE = False/' src/kaggle_pipeline.py > /tmp/train_real.py
   python src/nbgen.py /tmp/train_real.py kaggle/rsna-knee-train/rsna-knee-train.ipynb
   kaggle kernels push -p kaggle/rsna-knee-train        # -> v15
   # 10-min gate (browser, the CLI is blind mid-run): s/study ~0.17-0.25; ~1.0+ = something is wrong
   ```
   ⚠️ **Pushing `rsna-knee-train` changes what infer v5/v8 mount** (`kernel_sources` = latest
   output). v15's output will hold only `v06c_fold0_*`, so **infer v5/v8 would fail loudly**
   (`no v05a_fold*_best.pt is mounted`) if re-run — the 0.896 scores already on the board are
   unaffected, but a *new* submission of the two-head blend needs the weights pinned first
   (Dataset) or `rsna-knee-train` re-pushed with the old arms. Do the Dataset pin **before** or
   right after this launch.
2. **Read `v06c`** when it finishes: pull `--file-pattern "(oof|no_match)"` into
   `artifacts/kaggle_out/v15/`, then on fold 0 compute (the same Python as tonight's blend check):
   own OOF at the checkpointed epoch; ρ against `v05a` and `v05b`; rank-mean a+b+c (by-version,
   equal votes) vs a+b 0.8670. **Adopt** only if ρ < 0.77 against both **and** the 3-way blend
   clears +0.008; own OOF should be ≥ ~0.83. Log via `/update`; if adopted, add `v06c` to
   `INFER_MEMBERS`, push infer (with the train mount holding v06c and the folds mount holding
   v05g — the v05a/v05b checkpoints then need the Dataset pin), submit once.
3. If it fails on ρ (≈0.84 like the concat variants): the family is not diverse enough at this
   size; the next diversity bets are a different input geometry (P-11 336 px, P-08 more slices)
   or DINOv3, not more of the same.

### Open decisions for Tian

- Go / no-go on the ~2 h `v06c` run (item 1) and, with it, the Dataset pin for the DINOv2 weights.
- Make `convnext-tiny-224-hf` public (needed for any final submission that uses it).
- What to do with `crazy_good_rsna.ipynb` — read it for the 0.056 gap, ignore it, or delete.

### Things that will bite if forgotten

- **The 10-minute throughput gate is browser-only** (CLI `output`/`logs` are blank mid-run).
- **Pushing `rsna-knee-train` repoints infer v5/v8** — see item 1.
- `kaggle datasets create`: title ≤ 50 chars; run from inside the directory with `-p .`.
- All of the 23:35 / 22:35 lists (infer v7 never; token expires ~12 h; `v05f` never mount).

---

## 2026-08-29 (23:35) — #6/#7 scored: folds +0.009 alone, +0.000 on top of heads; attn 5-fold NOT launched

Supersedes the in-flight table of the 22:35 entry below; everything else there still holds.
Findings logged in experiments.md (Submissions #6/#7, Scoreboard, RESOLVED note on the 5-fold entry),
proposals.md (P-10 raised, P-13 supported), CLAUDE.md state line.

### ⏳ Still in flight — nothing

No kernel is running and no submission is pending. Background watchers from this session are dead.

### Where things stand

| | Status |
|---|---|
| Submissions | **seven**; best **0.896** twice (#5 infer v5 = two heads; #7 infer v8 = two heads + five concat folds). #6 (five folds alone) 0.886. 1 submission left today (resets 02:00) |
| Decision taken | **The 9 h attention 5-fold is not launched** — its gate (#7 > #5 by ≥ 0.005) failed by exactly 0.000 |
| GPU quota | ~13 h left this week (resets ~2026-09-05) |
| Repo | pushed through this entry |

### What we figured out (tonight's two numbers)

1. **Five folds of one model: +0.009 LB** (0.877 → 0.886), 1.8× the floor. Real, but half the
   second head's +0.019. The OOF→LB offset does **not** apply to fold ensembles (+0.039 here) —
   pooled OOF cannot see variance reduction.
2. **Five folds on top of the head blend: +0.000** (0.896 → 0.896). Fold-averaging and
   head-blending remove the same variance; once both heads are in, folds of one head are redundant.
   Caveat: the attention vote also fell 1/2 → 1/3 in #7, so a small gain may have cancelled a small
   dilution — deliberately *not* chased with more submissions (public-LB weight tuning is the trap).

### ⏭ Next action, in order

1. **A third diverse member, fold 0 first (~1 h), not more folds.** P-10 with a licence-clean timm
   ConvNeXt (Tiny or Small, 224, ImageNet weights) as an `ARMS` arm on `rsna-knee-train`, same
   cache/slots/jitter/EMA, 4 epochs first. Decision rule from P-21's own template: report the
   rank correlation of its fold-0 OOF against `v05a` and `v05b` (the two heads sit at 0.77) and the
   fold-0 rank-mean gain over 0.8670 (a+b). Adopt only if the 3-way blend clears +0.008 OOF **and**
   ρ < 0.77 against both — a member that is merely another concat-like error profile adds nothing,
   as #7 just showed. Smoke first; `FORCE_SMOKE = True`; check the cache line.
   Cheaper alternatives if ConvNeXt is more than an evening of code: an attention head at a
   different schedule (4 epochs, ~0.9 h) or a different seed of `v05a` — each is a weaker diversity
   bet (same backbone, ρ likely ≥ 0.84 like `v05b`–`v05g`).
2. **Pin the submission weights to a Kaggle Dataset** before any further push to `rsna-knee-train`
   or `rsna-knee-folds`: infer v5/v8 mount the *latest* outputs of both slugs.
3. **Final-submission hygiene**: v5 and v8 both score 0.896; v8 contains more models and is the
   safer private-LB bet (five folds average site variance even if the public LB cannot see it) —
   prefer v8's blend as the default going forward, and add any new member via `INFER_MEMBERS`.

### Open decisions for Tian

- ConvNeXt (P-10) vs cheaper same-backbone diversity for the remaining ~13 h this week.
- Kaggle Dataset pinning (browser or `kaggle datasets create`).
- Delete `artifacts/kaggle_out/v9smoke/` (1.8 GB of smoke checkpoints with real names).

### Things that will bite if forgotten

- All of the 22:35 entry's list, plus: **infer v7 must never be submitted** (flat 7-member; fold 0
  says 0.8611); the two 0.896 kernels are **v5** and **v8**.

---

## 2026-08-29 (22:35) — LB 0.896 from the two-head blend; valid 5-fold done; two blends submitted and pending

Findings logged through `cad29d7`. This entry is state only.

### ⏳ Still in flight as this was written (22:35)

| In flight | What it is | Started | How to check | How to read it |
|---|---|---|---|---|
| **Submission #6** — ref `55874877` | `rsna-knee-infer` **v6**: `INFER_MEMBERS=["v05g"]` — the five concat folds alone, rank-meaned | 22:30 | `kaggle competitions submissions rsna-knee-abnormality-detection --csv \| head -4` (a background watcher was polling; it dies with the session) | **This is the fold-ensemble gain**, read against `v04d`'s one-fold **0.877** (same recipe). ≥ 0.882 = folds are worth having on their own; 0.877–0.882 = 🔁 (inside the LB floor); < 0.877 would mean the 4-epoch fold models do not average well and needs a look at the per-fold OOF csvs in `artifacts/kaggle_out/folds_v4/`. Expect the +0.02–0.03 OOF→LB offset on pooled 0.8467 → roughly 0.87–0.88 *if* five folds add nothing, higher if they do |
| **Submission #7** — ref `55874878` | `rsna-knee-infer` **v8**: `INFER_MEMBERS=["v05a","v05b","v05g"]`, `INFER_BLEND="by_version"` — attn + concat-8ep + 5-fold concat, one vote per version | 22:30 | same command | Read against **#5 = 0.896**. The fold-0 proxy predicts only +0.001 OOF, so a move under 0.005 either way is 🔁 and the two-head blend stays the reference. A clear gain says folds add on top of heads → the attn 5-fold run is the next spend. A clear loss says the 4-epoch concat folds dilute → drop `v05g` from the blend or weight it down. **Infer v7 (flat 7-member) exists and must not be submitted** — fold 0 says it would score below #5 |

Both submissions run the same notebook, so a scoring error on one is a scoring error on both; a
missing score with status `ERROR` means the rerun crashed — pull `kaggle kernels output
tiankljucanin/rsna-knee-infer -p artifacts/kaggle_out/x --file-pattern no_match` and read the tail.

### Where things stand

| | Status |
|---|---|
| Submissions | ✅ **seven**: 0.500 → 0.841 → 0.871 → 0.877 → **0.896** (#5, P-21 two-head blend) → #6 ⏳ → #7 ⏳. 1 submission left today (resets 02:00 local) |
| Best single model | ✅ `v05a` attn + jitter, 8 ep: OOF 0.8574 |
| Best OOF blend (fold 0) | ✅ a+b 0.8670 → LB 0.896; a+b+g by-version 0.8680 (submitted as #7) |
| 5-fold ensemble | ✅ **`v05g` valid** (`rsna-knee-folds` v4, 4.27 h): per-fold 0.843–0.851, pooled 0.8467, gold 0.8476 on all 58. `v05f` (v2) = invalid, never mount |
| Checkpoint policy | ✅ switched to `ckpt_policy="best_oof"` (P-22); inert at 4 epochs, matters for concat past epoch 4 |
| Infer path | ✅ `INFER_MEMBERS` × `INFER_BLEND="by_version"`, decode-once with in-kernel equality check; infer kernel mounts `rsna-knee-train` **and** `rsna-knee-folds` |
| Kaggle layout | ✅ observed and printed at startup: kernel outputs at `/kaggle/input/notebooks/<owner>/<slug>/` (platform-wide since today) |
| GPU quota | ~19 h at 17:30 → **~13 h left** for the week after `v05g` (4.3 h) + three infer runs + smokes |
| Local env | ✅ `.venv` (torch 2.13.0+cpu, scikit-learn), `requirements.txt` pinned |
| Repo | ✅ clean, pushed through `cad29d7` |

### What we talked about and decided

- **Verified the traps-6f diagnosis before spending anything**: the folds smoke log (local copy +
  Tian's paste) and then v2's own log (`no cache is mounted`, 1.17 s/study, fold-0 OOF 0.8198 =
  the v02 number). Tian stopped v2 at ~10 h wall-clock to save quota; the log's own clock read 8.0 h
  — it had queued ~2 h, so "hours since push" overestimates execution.
- **Reordered the handoff's plan**: P-22 (local) before P-21 before the folds re-run, because P-22
  could change the checkpoint policy of the run about to launch (it did) and the slug was busy.
- **Decode-once was included in the P-21 change** (Tian's call), verified by array equality in the
  kernel, and the arm was renamed `v05g` so the invalid `v05f` files can never be confused.
- **GPU strategy with 19 h/week**: spend ~4.5 h on the concat 5-fold now, hold the ~9 h attn 5-fold
  until the fold-ensemble number is in. Still the plan; #6/#7 decide it.
- **Did not submit the flat 7-member blend (v7)** after the fold-0 weighting check showed it below
  the two-head blend; built the by-version rule instead and submitted that (v8) plus the 5-fold-alone
  kernel (v6), both approved by Tian.
- **Rejected tuning blend weights on fold 0** (attn 2:1:1 = 0.8688 vs 0.8680 by-version — inside
  the floor); by-version is the principled, unfitted rule.

### What we figured out

1. **P-21 transferred: LB 0.896 (+0.019, 3.8× the floor)** from two heads on one backbone, one
   fold, and the OOF→LB offset (+0.029) held for a blend — n=4 now. → experiments.md Submissions.
2. **P-22: best-OOF checkpointing is +0.0128 split-half for the concat head, ~0 for attn, no
   teacher-chasing; P-09 becomes a tie (−0.0024 at best epochs)** — the +0.0103 was concat's late
   decay. Policy switched. → experiments.md "P-22", `src/oof_epoch_analysis.py`.
3. **The valid 5-fold run**: fold spread 0.843–0.851 (one floor), fold 0 is representative, `v04d`
   reproduced within 0.005, pooled 0.8467. → experiments.md "First valid 5-fold run".
4. **Vote weighting beats member count**: a third same-head member adds +0.001 on fold 0; giving
   the concat side 6/7 of the vote *loses* 0.007 vs the two-head blend. Hence `INFER_BLEND`.
5. **Kaggle changed `/kaggle/input` for everyone today** (`notebooks/<owner>/<slug>/`), not just new
   slugs; traps 6f corrected, layout printed in every log. → traps 6f.
6. **Per-version kernel output is not retrievable** (`<slug>/11` returned v13's files) and
   `kernels logs` is blank mid-run; the v11 OOF csvs are gone. → traps 12e.
7. **A local smoke can "resume" from a stale `_last.pt` and train nothing**; smoke mode no longer
   resumes. → traps 19.
8. **The Kaggle OAuth token expires after ~12 h** with a misleading "wrong slug" error. → traps 20.
9. From v2's wreckage: a second seed-pair of the v02 recipe differs by 0.002–0.009 per epoch
   (corroborates the 0.008 floor), and fold 1 ≈ fold 0 at epoch 0. → experiments.md (invalid-run
   entry is unchanged; noted here only).

### ⏭ Next action, in order

1. **Read #6 and #7** (command in the in-flight table) and log both in experiments.md's
   Submissions table + Scoreboard via `/update`, using the rules written in those rows. Then decide:
   - #7 > #5 by ≥ 0.005 **and** #6 ≥ 0.882 → **launch the attn 5-fold** (`ARMS` arm
     `("v05h", {"head_type": "attn", "cache_jitter": True, "folds": (0,1,2,3,4), "epochs": 8})`
     via the `FIVE_FOLD` block — 5 × 8 × ~12.8 min ≈ 8.5 h, so raise `runtime_limit_hours` to 8.8
     for that run or plan a resume; ~9 h of the ~13 h quota). Smoke first (`FORCE_SMOKE = True`,
     gate on `cache: 4407 studies indexed`), then real.
   - #7 ≈ #5 (within 0.005) → folds add nothing on top of heads; spend the quota on a second
     *head* or *schedule* member instead (attn 4-ep fold 0 is cheap: ~0.9 h), not on more folds.
   - #7 < #5 → drop `v05g` from `INFER_MEMBERS` (or weight it down) before any further submission;
     #5's kernel (infer v5) remains the best submitted.
2. **Pin the submission's weights to a Kaggle Dataset** (open since two sessions): `rsna-knee-infer`
   mounts the *latest* `rsna-knee-train` and `rsna-knee-folds` outputs; any new push to either slug
   changes what a re-run of the infer kernel loads. Do this before the next training push to
   `rsna-knee-train`.
3. Only then P-10 / P-14 / P-15 / P-16.

### Open decisions for Tian

- The attn 5-fold (~9 h of the remaining ~13 h this week) — gated on #6/#7 as above.
- Kaggle Dataset pinning (item 2) — a browser or `kaggle datasets create` job.
- `artifacts/kaggle_out/v9smoke/` (1.8 GB of smoke checkpoints with real-run filenames) is still
  safe to delete; also `artifacts/kaggle_out/v11oof/` was removed this session (it held v13 copies).
- Browser-only questions still block P-10/P-16/P-18: rules text, Efficiency Prize formula,
  radimagenet.com T&C, hidden test size.

### Things that will bite if forgotten

- **`kaggle auth login` every ~12 h** (traps 20); the failure looks like a wrong slug.
- **Kernel outputs live under `/kaggle/input/notebooks/<owner>/<slug>/`** now; every resolver
  searches depth 4 and the layout is printed at the top of each log — read it.
- **Infer kernel versions**: v5 = a+b (0.896), v6 = 5-fold alone, v7 = flat 7 (**do not submit**),
  v8 = by-version 7. The committed `.ipynb` is v8's; `INFER_MEMBERS` in `src/` still defaults to
  `["v05a","v05b"]` and is sed'd per push like `MODE`.
- `kaggle kernels output <slug>/<version>` returns the **latest** version; download small results
  with the log at run time (traps 12e).
- `src/kaggle_pipeline.py` is committed with `FORCE_SMOKE = True`; the real kernels had `False`.
- Everything from the entries below still applies (`PYTHONUTF8=1`, never sort DICOMs by filename,
  never edit the `.ipynb`, `experiments.md` is append-only).

---

## 2026-08-29 (17:11) — The 5-fold run was invalid, not slow: the cache never mounted

Findings logged as of `f50e6a4`. This entry is state only. **Read traps 6f first** — it is the
most transferable thing this session produced.

### ⏳ Still in flight as this was written (17:11)

| In flight | What it is | Started | How to check | How to read it |
|---|---|---|---|---|
| **`rsna-knee-folds` v2 — ABANDON IT** | Was meant to be 5 folds × 4 epochs of the `v04d` recipe. It is **training the wrong recipe** (v02 decode path — no crop, no laterality, no per-series norm) because the cache never mounted | 2026-08-29 ~08:00 | `kaggle kernels status tiankljucanin/rsna-knee-folds` | **Do not use its output for anything.** ~9.2 h in; the 8.3 h guard has been reached, and the tail is a post-guard validation pass at ~1 s/study over 882 studies (~15 min) plus checkpoint writes. When it stops, `all folds complete: False`, inference skipped by design. `v05f_fold*` checkpoints are v02-recipe models — **delete them mentally; do not mount them.** Nothing needs to be salvaged |

### Where things stand

| | Status |
|---|---|
| Submissions | ✅ **four**: 0.500 → 0.841 → 0.871 → **0.877** (`v04d`) |
| Best single model | ✅ `v05a` attn + jitter, 8 ep: OOF **0.8574** — not yet submitted |
| Best OOF of any kind | ✅ **0.8670** — rank-mean of `v05a` + `v05b` on fold 0 (P-21), not yet submitted |
| Noise floor | ✅ measured: 0.008 macro, ~0.03 per label |
| P-09 / P-04 / P-05 / P-08-jitter | ✅ all closed — see experiments.md |
| **5-fold ensemble** | ❌ **wasted run, must be redone** (traps 6f) |
| Cache loader | ✅ fixed in `src/` (depth 4, fallback now fatal) — **not yet exercised on Kaggle** |
| Repo | ✅ clean, pushed through `f50e6a4` |

### What we talked about and decided

- **Ran all five prior suggestions** (submit `v04d`; 8-epoch P-09 retest; 5 folds; the direct
  worker-RNG check; P-04 as the retest's control), two kernels concurrently.
- **The 8-epoch retest shipped a matched control** (`v05b`) rather than comparing back to
  `v04d`, so head and schedule could not confound each other again.
- **P-10 de-prioritised** behind the new P-21: head diversity is free, a CNN member costs a
  session, RadImageNet's licence is unresolved.
- **Decided to ask rather than auto-launch the re-run.** ~9 h of GPU was just burned; the
  re-run is another ~4.5 h and the quota is shared.

### What we figured out

1. **THE BIG ONE — a new kernel slug does not mount inputs at the same depth as an old one.**
   `load_cache_manifests` globbed at `max_depth=2`; `rsna-knee-train` mounts the cache at
   `/kaggle/input/rsna-knee-cache-a/…` (depth 2) but the newly created `rsna-knee-folds` mounts
   it type-prefixed at depth 4, like the datasets. It was **the only glob in the file capped at
   2** — every other resolver uses 3 or 4. → **traps 6f**.
   - **Cost:** `cfg.use_cache` flipped to `False`, the dataset silently took the **v02 decode
     branch**, and ~9 h of GPU trained a superseded recipe at 0.99 s/study. Five folds at that
     rate is ~19.6 h — it could never have finished.
   - **How it was found:** Tian opened the kernel page and pasted the **smoke log**, which says
     `! use_cache=True but no cache is mounted` in plain text at line 61. That log had already
     been read once, for the arm banners and the RNG check, and the cache line was skipped.
   - **Why it went unnoticed for six hours:** the overrun *was* the evidence and it was
     explained away. At 14:38 the previous handoff recorded "~6.6 h in against a ~4.5 h estimate
     — the estimate was wrong, the run is not." A 1.7× miss against a throughput figure measured
     three separate times that day should have been treated as a symptom, not as estimator
     error. → experiments.md, "The first 5-fold run was invalid, not slow".
   - **Fixed:** glob depth 2 → 4, and a missing cache in train mode is now a `SystemExit` unless
     `ALLOW_DECODE_FALLBACK = True`.
   - **Second incident in one day from the same root cause:** `use_cache` means both *which
     preprocessing* and *is the file present*. traps 6d was the inference-side twin.
2. **Submission #4 = 0.877, a new best**, and a third OOF→LB point at +0.024 (three for three in
   the +0.02–0.03 band). The LB delta is only 1.2× its floor; jitter is carried by its
   11-of-12 per-label sign pattern. → experiments.md.
3. **P-09 ✅ KEEP (+0.0103 at matched 8 epochs) but the card's reasoning was wrong** — it loses
   on the plane-specific labels it was predicted to help (MCL −0.040, Lateral Meniscus −0.032);
   what it buys is overfit resistance. → experiments.md.
4. **The two heads rank-blend to OOF 0.8670** at ρ = 0.773 — free error diversity, no second
   backbone. → new card **P-21**, now the highest-value untested item.
5. **P-04: 8 epochs does not beat 4** for the concat head; the epoch count is **head-specific**.
6. **traps 6e was wrong and is corrected** — PyTorch already seeds numpy/`random` per worker, so
   the fork pathology does not exist here and `v04d`'s jitter was never confounded.
7. **traps 12e was wrong and is corrected** — per-version kernel output *is* retrievable via
   `<slug>/<version>`; a **run in flight on that slug** is what blocks it.
8. **New trap 4b** — the cosine schedule spans `cfg.epochs`, so epoch N of a 4-epoch run and
   epoch N of an 8-epoch run are at different LRs and are not comparable.

### ⏭ Next action, in order

1. **Do P-21 first — it needs no GPU.** In the inference block, `ckpt_paths` maps
   `fold -> path` for one `cfg.version`; make it accept (version, fold) pairs so `rank_mean`
   blends across *versions*. Then submit the `v05a` + `v05b` blend from `rsna-knee-infer`.
   Decision rule: expect a **small** LB move; a sub-0.005 change is **not** confirmation, and
   the +0.02–0.03 OOF→LB offset was calibrated on single models so ~0.891 is **not** a
   prediction. Push the infer kernel **before** any new `rsna-knee-train` push.
2. **Re-run the 5 folds with the fixed loader.** `sed 's/^FIVE_FOLD = False/FIVE_FOLD = True/'`,
   `FORCE_SMOKE = True` first, push, and **before promoting to real, grep the smoke log for the
   cache line**:
   ```bash
   kaggle kernels output tiankljucanin/rsna-knee-folds -p artifacts/kaggle_out/x --file-pattern "no_match"
   # REQUIRED in the log:  cache: 4407 studies indexed (c01_p224_s16_crop130_lat20)
   # If instead you see "no cache is mounted", it now raises - but check anyway.
   ```
   Then `FORCE_SMOKE = False` and push. **Sanity gate at ~10 min in: `s/study` must read
   ~0.17–0.18.** If it reads ~1.0, kill it immediately — that is the decode path again.
3. **Run P-22** (~0.1 session, analysis only, no training): best-epoch vs last-epoch OOF from
   the `_ep{e}_oof.csv` files already in `artifacts/kaggle_out/v13/`, with the gold curve
   alongside to detect teacher-chasing. `v05b` ships 0.013 below its own peak, and this choice
   silently decides P-09's verdict.
4. Only then: P-10 / P-14 / P-15 / P-16.

### Open decisions for Tian

- **Re-run 5 folds now, or do P-21 first, or both?** Asked and not yet answered. P-21 costs no
  GPU and ships something; the folds re-run is another ~4.5 h after ~9 h already burned today.
- **Pin the final submission's weights to a Kaggle Dataset instead of `kernel_sources`.**
  `rsna-knee-infer` mounts *the latest version* of `rsna-knee-train`, which forced
  submit-before-push ordering twice today. A dataset is immutable and kills the whole failure
  class before the deadline. **Traps 6f raises the priority of this** — mount layout is
  evidently not stable across slugs either.
- **`artifacts/kaggle_out/v9smoke/` is 1.8 GB of smoke checkpoints** whose filenames are
  identical to the real ones (`v9smoke/v04d_fold0_best.pt` is a 4-study toy, not the 0.877
  model). Safe to delete.
- Browser-only questions still block P-10/P-16/P-18: rules text, Efficiency Prize formula,
  radimagenet.com T&C, hidden test size.

### Things that will bite if forgotten

- **Read the cache line in every smoke log before promoting a run to real.** Scan the log for
  the *known* failure modes, not only for the new thing being tested (traps 6f).
- **Treat a large runtime overrun as a symptom, not as a bad estimate** — especially when the
  throughput figure it contradicts was measured the same day.
- A run in flight on a slug makes `kernels output` return nothing for that whole slug, for every
  version form (traps 12e). Past versions are retrievable as `<slug>/<version>` once idle.
- `--file-pattern` is a **regex, not a glob**; `"(oof|manifest)"` pulls results + log,
  `"no_match"` pulls the log alone.
- `src/kaggle_pipeline.py` is committed with `FORCE_SMOKE = True` on purpose; the kernels that
  ran had `False`.
- Everything from the entries below still applies (`PYTHONUTF8=1`, never sort DICOMs by
  filename, never edit the `.ipynb`, `experiments.md` is append-only).

---

## 2026-08-29 (14:50) — LB 0.877, P-09 and P-04 closed, head-blend found; 5-fold still running

Findings are all logged as of `0a131ea`. This entry is state only. Continues the 00:13 entry
below, which described the five-arm batch while it was still in flight.

### ⏳ Still in flight as this was written (14:50)

| In flight | What it is | Started | How to check | How to read it |
|---|---|---|---|---|
| **`rsna-knee-folds` v2** | First real ensemble: **5 folds × 4 epochs** of the confirmed `v04d` recipe (concat head + `cache_jitter`), arm `v05f`. New kernel slug so it could run beside `rsna-knee-train` | 2026-08-29 ~08:00 | `kaggle kernels status tiankljucanin/rsna-knee-folds`, then **`kaggle kernels output tiankljucanin/rsna-knee-folds -p artifacts/kaggle_out/folds --file-pattern "(oof\|manifest)"`** — pulls log **and** the OOF csvs together (traps 12e) | **~6.9 h in at 14:50 against a ~4.5 h estimate — the estimate was wrong, the run is not.** The 8.3 h guard fires ~16:20. Log has 20 `fold N epoch M:` lines (5 folds × 4 epochs) and a `=== v05f fold K ===` banner per fold. Per-fold OOF ~0.85 is normal; the number that matters is the **rank-mean across folds**, printed by the inference block only if *all five* folds completed. If the guard fired mid-fold, `all folds complete: False`, inference is skipped by design, and each fold has `v05f_fold{k}_last.pt` — attach this run's output as an input to a new run and it resumes |

**Compare the 5-fold ensemble gain against P-21's head-blend gain (+0.0096 on fold 0).** That
comparison is the point of the run: it decides whether the final submission spends its budget on
folds or on head diversity.

### Where things stand

| | Status |
|---|---|
| Submissions | ✅ **four**: 0.500 → 0.841 → 0.871 → **0.877** (`v04d`, new best) |
| Best single model | ✅ `v05a` attn + jitter, 8 ep: OOF **0.8574**, gold 0.9266 (n=11) — **not yet submitted** |
| Best OOF of any kind | ✅ **0.8670** — rank-mean of `v05a` + `v05b` on fold 0 (P-21, not yet submitted) |
| Noise floor | ✅ measured: **0.008 macro, ~0.03 per label** |
| P-09 attention head | ✅ KEEP (+0.0103 at matched 8 ep) |
| P-04 8 epochs | 🔁 does not beat 4 for concat; epoch count is **head-specific** |
| P-05 laterality | ✅ confirmed, ≈ +0.015 of v03's +0.022 |
| 5-fold ensemble | ⏳ running |
| Repo | ✅ clean, pushed through `0a131ea` |

### What we talked about and decided

- **Ran all five of the previous turn's suggestions rather than picking.** Submitted `v04d`; the
  8-epoch P-09 retest; the 5-fold run; the direct worker-RNG check; and P-04 folded into the same
  kernel as the retest's control arm.
- **The 8-epoch retest ships a matched control (`v05b`) rather than comparing back to `v04d`.**
  Comparing a new head against a differently-scheduled run would have confounded head with
  schedule — the exact mistake that made v03 ambiguous for a day.
- **The 5-fold run uses today's confirmed winner (`v04d`), not "whatever wins".** #3 was
  contingent on #2, but sequencing them wastes a sitting, and Kaggle runs two GPU sessions at
  once. The ensemble and the first trustworthy multi-fold number are worth having either way.
- **Deliberately did not launch a third GPU run while both were busy** — quota is shared, and the
  right next arm depended on results an hour away.
- **P-10 (second architecture family) de-prioritised** behind the new P-21: head diversity is free,
  a CNN member costs a session, and RadImageNet's licence is still unresolved.

### What we figured out

1. **Submission #4 = 0.877, a new best**, and the **third** OOF→LB point at a +0.024 offset —
   three for three inside +0.02–0.03. Honest caveat recorded: +0.006 LB is only 1.2× the LB
   floor, so jitter is carried by its 11-of-12 per-label sign pattern, not the leaderboard.
   → experiments.md, Submissions.
2. **P-09 ✅ KEEP at a matched schedule (+0.0103), but the card's reasoning was wrong.** It
   predicted gains on plane-specific findings; those are where it *loses* (MCL −0.040, Lateral
   Meniscus −0.032). What it buys is **overfit resistance** — it plateaus at 0.857 while concat
   peaks at epoch 4 and decays to 0.8471. → experiments.md.
3. **That verdict is policy-dependent.** At each head's own best epoch they are indistinguishable
   (0.8576 vs 0.8600); attn wins only under fixed-last-epoch checkpointing. → new card **P-22**.
4. **The two heads rank-blend to OOF 0.8670** (+0.0096 over the best single arm) at mean rank
   correlation **0.773**, despite sharing backbone, data, fold, schedule and seed. Free error
   diversity. → new card **P-21**, the highest-value untested item now.
5. **P-04: 8 epochs does not beat 4 for the concat head** (0.8471 vs 0.8528). The epoch count is
   **head-specific**, not a project setting. → experiments.md.
6. **traps 6e was wrong and is corrected**: the on-Kaggle check shows PyTorch already seeds numpy
   and `random` per worker, so the fork pathology does not exist here. `v04d`'s jitter was never
   confounded. → traps 6e.
7. **traps 12e was also wrong and is corrected**: per-version output *is* retrievable
   (`<slug>/<version>`); what actually blocks it is a **run in flight on that slug**. → traps 12e.
8. **New trap 4b — the cosine schedule spans `cfg.epochs`**, so epoch N of a 4-epoch run and
   epoch N of an 8-epoch run sit at different learning rates and are not comparable. Compare
   final-to-final across budgets. → traps 4b.

### ⏭ Next action, in order

1. **Read the 5-fold run** (command in the in-flight table). Take the **rank-mean across folds**
   from the inference block, not the per-fold numbers. Decision rule: compare the ensemble's gain
   over a single fold (`v04d` 0.8528) against **P-21's +0.0096 head-blend gain**. If 5 folds buys
   less than ~0.01 over one fold, head diversity is the better use of the budget and P-21 comes
   first. If it buys more, folds win and P-21 becomes an addition rather than a substitute.
2. **Ship P-21** (~0.2 session, no training): make the infer path rank-mean across *versions*, not
   just folds — `ckpt_paths` currently maps `fold -> path` for one `cfg.version`; it needs
   (version, fold) pairs. Then submit the `v05a` + `v05b` blend. Expect a *small* LB move; do not
   read a sub-0.005 change as confirmation.
3. **Run P-22** (~0.1 session, analysis only, no training): for every arm, compare best-epoch vs
   last-epoch OOF from the `_ep{e}_oof.csv` files already downloaded in
   `artifacts/kaggle_out/v13/`, with the gold curve alongside to detect teacher-chasing. If
   best-epoch beats fixed-epoch by more than 0.008 **and** gold does not diverge, switch the
   policy and re-read P-09 and P-04 under it.
4. **Only then** consider more architecture (P-10/P-14/P-15) or P-16's re-labelling.

### Open decisions for Tian

- **Pin the final submission's weights to a Kaggle Dataset instead of `kernel_sources`.** Right
  now `rsna-knee-infer` mounts *the latest version* of `rsna-knee-train`, so pushing a training
  run changes what a submission would load. It forced submit-before-push ordering twice today. A
  dataset is immutable and removes the whole failure class before the deadline.
- **`artifacts/kaggle_out/v9smoke/` is 1.8 GB of smoke checkpoints** (4-study models) with
  **filenames identical to the real ones** — `v9smoke/v04d_fold0_best.pt` is *not* the 0.877
  model. Safe to delete; flagged rather than deleted.
- Browser-only questions still block P-10/P-16/P-18: rules text (hosted-LLM clause, winner
  licence), Efficiency Prize formula, radimagenet.com T&C, hidden test size.
- **The remaining 0.075 to the public top is still unexplained.** We use the same public label
  tables the leaders use; ensembling plausibly accounts for 0.01–0.02.

### Things that will bite if forgotten

- **A run in flight on a slug makes `kernels output` return nothing for that whole slug**, for
  every version form. Wait for it to finish (traps 12e).
- Pull the log **and** the small result files in one command: `--file-pattern "(oof|manifest)"`.
  `--file-pattern` is a **regex, not a glob**; `"no_match"` is the log-only trick.
- `src/kaggle_pipeline.py` is committed with `FORCE_SMOKE = True` on purpose; the kernels that ran
  had `False`. Regenerating a notebook locally produces a *smoke* notebook.
- The 5-fold kernel is generated by `sed 's/^FIVE_FOLD = False/FIVE_FOLD = True/'` — never edit
  the `.ipynb`.
- Push `rsna-knee-infer` **before** re-pushing `rsna-knee-train` (see the Kaggle Dataset item).
- Everything from the entries below still applies (`PYTHONUTF8=1`, never sort DICOMs by filename,
  do not download the 21 GB cache, `experiments.md` is append-only).

---

## 2026-08-29 (00:13) — LB 0.841 → 0.871, four silent bugs fixed, five-arm batch launched

All findings are logged in experiments.md / traps.md / proposals.md / CLAUDE.md as of commit
`531a923`. This entry is state only.

### ⏳ Still in flight as this was written (00:13)

| In flight | What it is | Started | How to check | How to read it |
|---|---|---|---|---|
| **Kernel v11** — `rsna-knee-train` v11 | Five fold-0 arms, 4 epochs each, from the cache, in run order: `v04base` (reference, seed 42) → `v04a` (seed 43) → `v04c` (`head_type="attn"`, P-09) → `v04b` (`lat_undo`, P-05) → `v04d` (`cache_jitter`, P-08) | 2026-08-28 ~23:35 | `kaggle kernels status tiankljucanin/rsna-knee-train`, then `kaggle kernels output tiankljucanin/rsna-knee-train -p artifacts/kaggle_out/v11 --file-pattern "no_match"` for the log **alone** (a plain `output` pulls ~1 GB of checkpoints) | ~0.9 h/arm, ~4.6 h total against an 8.3 h guard. Each arm prints a `##########` banner with its resolved folds/epochs/seed, then a per-label table per epoch. **Read `v04a` vs `v04base` FIRST** — that difference *is* the noise floor and nothing else can be judged without it. An arm that raised is logged `!! arm <v> FAILED` and the remaining arms still ran |

**The machine can be shut down without affecting this.** The kernel runs on Kaggle's servers; the
local polling loops watching it die with the session and matter to nothing. Every arm writes
`v04*_fold0_best.pt` as it completes, so even a guard-stopped run keeps whatever finished.

### Where things stand

| | Status |
|---|---|
| Submissions | ✅ **three**: 0.500 (constant, #1) → **0.841** (v02, #2) → **0.871** (v03, #3) |
| Best model | ✅ v03 fold 0: OOF-vs-teacher 0.843, gold 0.906 (n=11), **public LB 0.871** — one fold, one backbone, no ensemble, no TTA |
| Infer path | ✅ fixed and verified — predictions byte-identical to the training kernel's on the same studies (traps 6d) |
| Cache (P-01) | ✅ closed as measured: 5.4× end to end, and the largest scoring gain so far |
| Noise floor (P-02 step 1) | ⏳ being measured for the first time by v11 — still **asserted** at 0.01 until it reports |
| P-09 attention head | 🔧 shipped, ⏳ running |
| Repo | ✅ clean, pushed through `531a923` |

### What we talked about and decided

- **Read the previous session's two in-flight items before proposing anything.** That is what
  surfaced both the 0.871 and the confound; going straight to new ideas would have missed both.
- **Chose P-09 over P-16, P-08 and "just run 5 folds"** as the impact card. P-16 has the highest
  ceiling but its own cited result (Qwen3-14B-AWQ, 0.881 gold) sits *below* our current blend's
  0.8948, so it is not obviously a ceiling-raiser and costs 1–2 sessions. P-08 turned out to buy
  token granularity rather than coverage (card corrected). P-09 is free in GPU time, *removes*
  head parameters in a regime that is measurably overfitting, and targets the plane-specific weak
  labels. Five folds is the reliable +0.01–0.02 and stays queued as breadth.
- **Ablated laterality from the existing cache instead of rebuilding it.** The cache's transforms
  are involutions, so right knees can be de-canonicalised at load time — free, and a *cleaner*
  test than v03-vs-v02 because the crop stays constant. Rejected the no-crop rebuild: two CPU
  kernels, another 21 GB, 42 GB mounted, and it does not change what we do next.
- **Added a fifth arm (`v04base`) mid-build.** Once `seed_worker()` changed how augmentation is
  randomised, kernel v8 stopped being a same-code baseline, so arms b/c/d had nothing to differ
  from in exactly one thing. One extra hour buys single-variable attribution for all three.
- **Submitted v03 before touching `rsna-knee-train`.** `kernel_sources` mounts the *latest*
  version's output, so the infer push had to happen while that was still v8.

### What we figured out

1. **The v03 gain is real and transferred: LB 0.841 → 0.871, +0.030 against a 0.005 floor.** The
   +0.022 OOF was not a teacher-agreement artefact. → experiments.md, Submissions.
2. **OOF-vs-teacher predicts the LB and under-reads by +0.02–0.03** (two points: 0.821→0.841,
   0.843→0.871) — expected when the teacher's own gold is 0.8948. An OOF gain is now evidence
   rather than a hope. n=2, so treat it as an offset, not a law. → experiments.md.
3. **P-01 was mis-specified, not merely under-measured.** Its "OOF within ±0.01 = faithful
   speed-up" rule assumed v03 replayed v02's inputs; v6's config has no `crop_mm` and no
   `lat_dead_zone_mm` at all. Four things moved at once. `v04b` splits them. → experiments.md.
4. **The cache is a 5.4× speed-up end to end, not the ~60× the decode arithmetic implied** — the
   T4 is the bottleneck now, at ~0.18 s/study for ~29 ViT forwards. **This corrects finding #9 of
   the 2026-08-28 entry below, which says 60×.** Consequence: I/O work is now worthless, and
   extra slices cost linearly in GPU time. → experiments.md.
5. **Inference was one line away from scoring a v03 model on v02 pixels.** `use_cache` meant both
   "which preprocessing" and "read a .npy"; with no cache mounted the infer kernel took the v02
   decode branch — no crop, no laterality. Nothing would have raised. → traps 6d.
6. **A real-mode `MODE="auto"` infer kernel would have re-trained at rerun**, because fold
   narrowing was gated on `cfg.smoke` and only fold 0 has a checkpoint. → traps 12c/12d.
7. **Smoke mode structurally cannot reveal real-mode defaults.** Every arm inherited
   `folds=(0,1,2,3,4)` — 25 folds, ≈18 h — and four green smokes said nothing about it, because
   `__post_init__` forces `folds=(0,)`. → traps 12d.
8. **The pipeline's augmentation may never have been random.** numpy and `random` are
   fork-inherited per DataLoader worker, so jitter and noise repeat identically every epoch.
   Fixed with `seed_worker`, **but not empirically verified** — Windows spawns workers, so the
   pathology cannot be reproduced locally. → traps 6e.
9. **v8's train loss falls monotonically while OOF turns over at epoch 2.** That is the
   overfitting signature, and it is the argument against P-04's 8 epochs and for augmentation and
   fewer head parameters. → experiments.md.

### ⏭ Next action, in order

1. **Read v11's log** (command in the in-flight table). Take the epoch-3 `auc_soft` of each arm,
   then in this order:
   - `floor = |v04a − v04base|`. **This replaces the asserted 0.01 everywhere.** If it exceeds
     ~0.02, the v03 +0.022 was never established and that is the session's headline finding.
   - `v04c − v04base` (P-09). Adopt only if it clears the floor. Report the macro **and**
     Effusion~Synovitis, Medial OA~Medial Meniscus, Contusion~Fracture — research.md's stated
     risk is that a per-label head hurts exactly those pairs.
   - `v04b − v04base` (P-05). A **drop** attributes the v03 gain to laterality; flat means the
     130 mm crop or per-series normalisation carried it. Read the five side-specific labels, not
     the macro.
   - `v04d − v04base` (P-08 jitter). Judge on the *shape* of the epoch-2→3 turn, not the peak. If
     flat, "the augmentation still is not random" remains live (finding 8) — settle it by logging
     epoch-0 vs epoch-1 jitter offsets in one Kaggle run before calling it a dead end.
2. **Log the four verdicts with `/update`**, each stating its delta and the floor it beat.
3. **Submit the winning arm** if any clears the floor: set `cfg.version` to that arm, regenerate
   `kaggle/rsna-knee-infer` with the sed recipe in CLAUDE.md, push, submit. Expect roughly
   LB ≈ OOF + 0.02–0.03 (finding 2).
4. **Then five folds of the best recipe** — ~4.6 h now, and the cheapest standing claim on part of
   the 0.081 gap to the public top.

### Open decisions for Tian

- **Depth or breadth next.** Five folds of the current recipe (a real ensemble, a trustworthy LB
  number, +0.01–0.02) versus more fold-0 arms. With ~0.9 h arms, depth-then-breadth still looks
  right, but the 2026-10-22 deadline makes it a judgement call.
- Browser-only questions still block P-10/P-16/P-18: rules text (hosted-LLM clause, winner
  licence), Efficiency Prize formula, radimagenet.com T&C, hidden test size.
- **Where the remaining 0.081 comes from is genuinely unexplained.** We use the same public label
  tables the leaders use, and ensembling accounts for maybe 0.01–0.02. Worth reading the top
  notebooks' configs before assuming more of the same recipe closes it.

### Things that will bite if forgotten

- `kaggle kernels output` with no `--file-pattern` pulls **every** checkpoint — ~1 GB for a
  five-arm run. Use `--file-pattern "no_match"` when you only want the log.
- The infer notebook is generated from a **sed'd copy** (`MODE="infer"`, `FORCE_SMOKE=False`);
  `MODE="auto"` is wrong there, and `FORCE_SMOKE=True` sets a 0.4 h runtime guard (traps 12c/12d).
- `src/kaggle_pipeline.py` is committed with `FORCE_SMOKE = True` on purpose. Kernel v11 on Kaggle
  has `False`; regenerating the notebook locally produces a *smoke* notebook.
- Push the infer kernel **before** re-pushing `rsna-knee-train`: `kernel_sources` mounts the
  latest version's output.
- Everything from the entry below still applies (`PYTHONUTF8=1`, never edit the `.ipynb`, do not
  download the 21 GB cache, `experiments.md` is append-only).

---

## 2026-08-28 (evening) — Research, 2 submissions, v02 baseline, cache built, v03 training from cache

**Two things are still in flight as this was written (22:35). Check them first:**

| In flight | What it is | Started | How to check | How to read it |
|---|---|---|---|---|
| **Submission #2** — ref `55852527` | The v02 fold-0 model, submitted from `rsna-knee-infer` v1 (which mounts `rsna-knee-train` v6's output and runs `MODE=infer`) | 22:00 local (19:59:50 UTC) | `kaggle competitions submissions rsna-knee-abnormality-detection --csv` | Its rerun decodes ~1,300 hidden studies at ~150–320 s/100 studies, so ~40–70 min is normal. **Anything except exactly 0.500 means the submission path finally works.** Expect roughly 0.80–0.88 if the OOF transfers (public DINOv2-S/224 baselines sit at 0.809). Log it in the experiments.md Submissions table next to OOF 0.821 / gold 0.847 |
| **Kernel v8** — `rsna-knee-train` v8 | v03: the same recipe as v6 but **training from the mounted cache** instead of decoding DICOMs, fold 0, 4 epochs | 22:04 local | `kaggle kernels status tiankljucanin/rsna-knee-train`; logs only become readable when it finishes | **The runtime is itself the measurement.** v6 took 5.0 h (58 min/epoch). If v8 finishes in well under an hour the cache works; if it takes ~5 h it silently fell back to per-epoch decode (the notebook prints `! use_cache=True but no cache is mounted` in that case) |

### Where things stand

| | Status |
|---|---|
| Research | ✅ 18-agent workflow (8 researchers → 8 skeptics → synthesis → critic, 817 tool calls) → [research.md](research.md); 21 cards → [proposals.md](proposals.md) |
| Submission #1 | ✅ scored **0.500 exactly** (kernel v2, smoke) = constant output at rerun; root-cause class found and fixed |
| Submission #2 | ⏳ **PENDING** (see table above) |
| Pipeline v02 | ✅ green (Kaggle v3–v5): prob targets, LR 2e-5 + LLRD 0.75, EMA 0.998, fixed-epoch `best.pt`, per-label + OOF logging every epoch, `MODE=infer`, loud-failure submission, resume fixed |
| Real fold-0 run (v02, decode path) | ✅ kernel v6, 5.0 h: **0.99 s/study**, OOF-vs-teacher **0.821**, gold 0.847 (n=11, CI 0.72–0.94) |
| Cache | ✅ built: `rsna-knee-cache-a` (2,115 studies, 10.2 GB) + `-b` (2,292, 11.0 GB), 25 min each, **0 decode failures**, 4,407/4,407 studies |
| Cache loader (v03) | ✅ shipped and smoke-verified on Kaggle (v7: 4,407 studies indexed, no header scan needed) |
| Label audit | ✅ `src/label_audit.py` → `artifacts/label_audit.md`, findings in experiments.md |
| Repo | ✅ 4 commits pushed today (`3bf7085..8b63927`) |

### What we figured out (the findings that changed what we do)

**1. Our training targets were on the wrong scale (P-00, fixed).** The soft targets were
rank percentiles. `rank(pct=True)` gives tied values their *average* rank, so on labels where
most reports say 0, every confident negative got a target of 0.28–0.39 while the 58 gold rows
sat at a hard 0/1 with 8× weight — **no study on any label had a target below 0.1**. AUC is
rank-invariant, which is why this looked principled, but BCE fits *values*. Now the mean of
the sources' probabilities (teacher gold 0.8948 vs 0.8934; the Δ is noise — the fix is about
scale, not AUC).

**2. A submission can score exactly 0.500 and tell you nothing.** Submission #1 completed and
scored 0.500 to three decimals. A near-random model on ~1,300 studies scores 0.47–0.53, never
0.500 — that is a *constant* submission, i.e. our own `fillna(0.5)` fallback fired because the
hidden test tree did not match the assumed layout. Code-competition rerun logs are invisible,
so nothing reported it. v02 probes the test root by glob, tolerates non-`.dcm` names, writes
**no placeholder**, and **raises** when < 90% of test studies have an image slot or > 6 labels
are constant. A visible scoring error beats an invisible 0.500.

**3. The first real model works and its weak labels are visible per label.** v6 fold 0:
OOF-vs-teacher 0.821 over 882 val studies, plateauing at epoch 2–3; `pred_std` 0.12 → 0.23
(no base-rate collapse). Weakest against the teacher: **Lateral Meniscus 0.72, MCL 0.75,
Lateral OA 0.78, PF OA 0.79** — three of the four are side-specific or small focal findings,
which is exactly what P-05 (laterality), P-08 (more slices) and P-11 (resolution) target.

**4. Synovitis is a teacher-ceiling problem, not a model problem.** The student reproduces
the teacher almost perfectly where the teacher is confident (0.94 on confidently-labelled
rows) while gold sits at chance (0.50) — it has faithfully learned that "not mentioned" means
negative, and Synovitis is unaddressed in **84%** of reports. Only better targets move this
(P-07/P-16), never a better backbone.

**5. Our three LLM label sources are about 1.5 sources.** `hans_v4` and `sol56` make identical
decisions at the 0.5 cut on 99.45% of studies (error-φ = 1.000 on gold, every label). Adding
more public label tables is therefore pointless, and the `agreement` term in
`confidence_weights` is inflated by a duplicate.

**6. Silence is not down-weighted the way the code claims.** A report that never mentions a
finding blends to ~0.18 — a confident-looking negative — so its weight is 0.69 versus 0.80–0.89
on addressed rows. The docstring's "silent reports pull far less" was wrong; gating the weight
on pilkwang's `UNK` verdict is an open card.

**7. The published Synovitis←Effusion back-fill does not reproduce on our blend.** Gold AUC
0.788 → 0.729, paired-bootstrap Δ −0.059, 95% CI [−0.164, +0.042] → 🔁 INCONCLUSIVE, not
adopted. The public card's +0.11 came from a 0.678 baseline we are already above.

**8. Laterality can be recovered from geometry, on our own data.** Over all 4,407 studies:
`Laterality` tag present on 49.6%, geometry (image-centre x, 20 mm dead zone) resolves 96.9%,
**tag-vs-geometry agreement 0.988** (n = 2,116), 26 conflicts left unmirrored, 2.1%
unresolved. Previously community-sourced, now measured here.

**9. The cache is a 60× epoch speed-up and is bit-exact.** 4,407 studies in 25 min per shard,
0 decode failures, 21.2 GB in two shards (needed: a single kernel output caps around 20 GB).
The training notebook carries the builder's exact functions, so a *test* study is built on the
fly by the same code — verified bit-identical to the cached array on the local sample for both
a left and a right knee.

**10. Inference cost, not training cost, will bound the ensemble.** Measured 150–320 s per 100
test studies for **one** fold. Five folds each decoding the test set again would spend hours on
inference alone in the rerun, so decode-once-predict-all-folds is a prerequisite for any
multi-fold submission (P-18), not an optimisation.

**11. Two silent bugs the code review caught before they cost a session.** Resume never
resumed (checkpoints were looked for in `WORK`, but a mounted previous run is read-only under
`/kaggle/input`) — every fold would have restarted at epoch 0. And a smoke-mode `MODE=infer`
would have fed 2 slices/slot to a model trained on 6 while passing every assert; the infer path
now reads the input geometry from the checkpoint's saved config.

### ⏭ Next action, in order

1. **Read the two in-flight results** (table at the top). Log submission #2's score in the
   experiments.md Submissions table; for v8 pull `kaggle kernels output tiankljucanin/rsna-knee-train
   -p artifacts/kaggle_out/v8 --file-pattern "(oof|log)"` and compare **epoch time** and
   **OOF-vs-teacher** with v6's 0.821. P-01's verdict rule: within ±0.01 = the cache is a
   faithful speed-up; a real drop means the crop / per-series normalisation / laterality changed
   the inputs, and that must be understood before adopting it.
2. **Submission #3 = v03** if its OOF holds: re-push `kaggle/rsna-knee-infer` (it mounts
   `rsna-knee-train`'s latest output and reads the cache config from the checkpoint), then
   `kaggle competitions submit rsna-knee-abnormality-detection -k tiankljucanin/rsna-knee-infer
   -v <ver> -f submission.csv -m "..."`.
3. **Then the cache-era cards, cheapest first** — they only make sense now that an epoch is
   minutes: P-02 seed-noise baseline (2 seeds of fold 0 — this *measures* the 0.01 OOF floor we
   have only asserted), P-04 8 epochs, P-08 16 slices/slot, P-03b LLRD vs uniform, P-02 proper
   (site-grouped folds), then 5 folds for a real ensemble.

### Open decisions for Tian

- Browser-only questions still block P-10/P-18: rules text (hosted-LLM clause, winner licence),
  Efficiency Prize formula, radimagenet.com T&C, hidden test size.
- After v03 lands, the next GPU session is a choice between **breadth** (5 folds of the current
  recipe → a real ensemble and a trustworthy LB number) and **depth** (P-04/P-08 arms on fold 0
  → a better single model first). With cheap epochs, depth-then-breadth is the better order,
  but it is a judgement call about the 2026-10-22 deadline.

### Things that will bite if forgotten

- `PYTHONUTF8=1` for **any** Python that prints non-ASCII, not just the Kaggle CLI (traps 15b).
- The submitted notebook must be the **infer-mode** one (`rsna-knee-infer`), never the training
  notebook — a code competition re-runs what you submit (traps 12c).
- A public LB of exactly **0.500** is a constant submission, not a bad model (traps 12b).
- `kaggle/rsna-knee-cache-b` and the real-run notebooks are generated by `sed`ing
  `src/*_pipeline.py` before `nbgen.py` — never edit an `.ipynb`.
- Do **not** download the cache locally (21 GB); mount `rsna-knee-cache-a`/`-b` as
  `kernel_sources` and read `manifest_shard{0,1}.csv` + `<study>.npy` on Kaggle.
- `experiments.md` is append-only: superseded rows are marked, not deleted.

---

## 2026-08-28 — Pipeline built and verified green on Kaggle

### Where things stand

| | Status |
|---|---|
| Competition entry | ✅ rules accepted, `userHasEntered: True` |
| Kaggle CLI | ✅ authenticated as `tiankljucanin` (OAuth) |
| Data locally | ✅ CSVs + LLM labels + models. ⚠️ sample DICOMs **459/557** (rate-limited) |
| Targets & folds | ✅ built, verified, reproducible across machines |
| Pipeline | ✅ runs green end-to-end on a Kaggle T4 (smoke mode) |
| Real training run | ❌ **not started** |
| Submissions | ❌ **zero** |
| Repo | ✅ git initialised, remote set, pushed |

### What got done

- Verified every competition fact against the API and the CSVs (see
  [../CLAUDE.md](../CLAUDE.md)). Corrected an earlier wrong hypothesis: reports are
  **not** available at inference time.
- Scored all five public LLM report-label sources against the 58 gold studies. Best blend:
  **teacher gold macro-AUC 0.8934**.
- Built leak-safe grouped folds: **882/882/881/881/881**, gold **11/12/12/12/11**, no report
  group spanning a fold. Byte-identical on Kaggle and locally.
- Built `src/kaggle_pipeline.py` — the whole pipeline in percent format, runnable as a
  script *and* convertible to the notebook via `src/nbgen.py`.
- Pushed and ran `tiankljucanin/rsna-knee-train` (private). **Kernel v2 completed green** in
  9.5 min in smoke mode: targets → folds → header scan → slot selection → triplets → DINOv2
  → training → checkpoints → rank-mean inference → validated `submission.csv`.
- Found and fixed three silent bugs, all recorded in [traps.md](traps.md): a fold could
  finish with **no `best.pt`** and vanish from the ensemble; teacher AUC was being graded
  against itself (printed 1.0000); an empty header scan was being cached and would poison
  resumes.
- Split the docs into [traps.md](traps.md), [experiments.md](experiments.md),
  [brainstorm.md](brainstorm.md), [setup.md](setup.md), and this file.

### ⏭ Next action, in order

1. **Benchmark real throughput on Kaggle.** Everything else depends on this number. Fold 0
   took 36 s for 8 study-passes at 2 slices/slot with `num_workers=0` (~4.5 s/pass), which
   naively extrapolates to **6–8 h per fold** — five folds ≈ five sessions. That is an
   extrapolation from 8 studies, *not a measurement*. Get the real figure with
   `num_workers=2` at the production slice count.
2. **Build the preprocessing cache kernel** (backlog #1). Almost certainly the bottleneck is
   DICOM decode: ~90 file reads per study, every epoch. Decode/resize to uint8 once in a CPU
   kernel, mount it from training.
3. **Add site-grouped folds** (backlog #2). Largest correctness gap in our validation.
   Expect OOF numbers to drop — that is the point.
4. Only then: flip `FORCE_SMOKE = False` and launch a real multi-fold run.

**Do not skip to step 4.** A 5-fold run at current throughput burns several sessions on work
the cache makes ~10× cheaper, and its OOF would be inflated by site leakage anyway.

### Open decisions for Tian

- **Whether to train a single fold now** anyway, to get a first real LB number for
  orientation, accepting that it costs a session and the OOF is not yet trustworthy. There
  is a real argument for it — zero submissions means zero feedback — but it is a judgement
  call about spending a session.
- Several questions need a **browser** and cannot be answered from the CLI: the exact
  runtime limit, the Efficiency Prize scoring formula, and the winner-licence clause (which
  gates using RadImageNet in a final submission). See the open-questions table in
  [brainstorm.md](brainstorm.md).

### Things that will bite if forgotten

- `export PYTHONUTF8=1` before any `kaggle` CLI call on Windows.
- Run scripts from the **repo root** with `PYTHONPATH=src`.
- Edit `src/kaggle_pipeline.py`, **never** the generated `.ipynb`.
- `FORCE_SMOKE = True` for the first push after any edit.
- Never select the **P100**.
- The 98 missing sample DICOMs are a **rate limit**, not a bug, and block nothing.

---

## Template for the next entry

```markdown
## YYYY-MM-DD — one-line summary

### What got done
-

### Results (also log in experiments.md with a verdict)
-

### ⏭ Next action
1.

### Open decisions / blockers
-
```
