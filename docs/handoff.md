# Handoff — where we left off

**Update this at the end of every working session.** Newest entry on top. Keep it short and
concrete: what changed, what state things are in, what the next action is. This is the file
to read first after a break.

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
