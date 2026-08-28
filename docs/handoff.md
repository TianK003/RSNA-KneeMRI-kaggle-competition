# Handoff — where we left off

**Update this at the end of every working session.** Newest entry on top. Keep it short and
concrete: what changed, what state things are in, what the next action is. This is the file
to read first after a break.

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
