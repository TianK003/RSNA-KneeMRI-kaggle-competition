# Handoff — where we left off

**Update this at the end of every working session.** Newest entry on top. Keep it short and
concrete: what changed, what state things are in, what the next action is. This is the file
to read first after a break.

---

## 2026-08-28 (evening) — Research, first submission, v02 pipeline, cache kernel, label audit

### Where things stand

| | Status |
|---|---|
| Research | ✅ 18-agent workflow (8 researchers → 8 skeptics → synthesis → critic) → [research.md](research.md); cards → [proposals.md](proposals.md) |
| Submission #1 | ✅ made (kernel v2, smoke) — **public 0.500 exactly** = constant output at rerun; root cause class found and fixed |
| Pipeline v02 | ✅ local + Kaggle smoke (kernel v3; kernel v4 = post-review fixes): prob targets, LR 2e-5 + LLRD 0.75, EMA, fixed-epoch `best.pt` (last-epoch EMA), per-label/OOF logging every epoch, `MODE=infer` reading model inputs from the checkpoint config, no-placeholder loud-failure submission, **resume bug fixed** (mounted `_last.pt`/`_best.pt` now copied into WORK — previously resume silently restarted at epoch 0, traps 8b) |
| Cache kernel | ✅ **built**: `rsna-knee-cache-a` v3 (2,115 studies, 10.2 GB) + `-b` v2 (2,292, 11.0 GB), 25 min each, 0 decode failures; laterality resolved 96.9%, tag/geo agreement 0.988 |
| Label audit | ✅ `src/label_audit.py` → `artifacts/label_audit.md`; findings in experiments.md |
| Real fold-0 run | ⏳ **launched 16:13 as kernel `rsna-knee-train` v6** (v02, fold 0, 4 epochs, 6 slices/slot, `num_workers=2`) — expect 6–8 h; guard at 8.3 h |
| Submission #2 | ❌ |

### What got done

- **P-00 target-scale bug** (found by the research critic, verified on our data): rank-percentile
  targets put confident negatives at 0.28–0.39. Fixed to mean-of-probabilities in both
  `build_targets.py` and the notebook; teacher gold macro-AUC 0.8948 (was 0.8934).
- **Submission #1** scored exactly 0.500 → constant predictions at rerun (image-root/`.dcm`
  assumption). v02 probes the root by glob, tolerates non-`.dcm`, writes **no placeholder** (loud
  failure) and **refuses to submit** when < 90% of test studies are imaged with ≥ 1 slot or > 6
  labels are constant.
- **v02 recipe** (strong-evidence changes only): backbone LR 2e-5 with layer-wise decay 0.75,
  wd 0.02 (none on bias/LN), EMA 0.998 validated + saved; warmup/cosine/clip were already there.
- **Instrumentation**: per-label AUC/pred_std table each epoch, `{version}_fold{k}_oof.csv`,
  bootstrap CI on gold, s/study logging, `MODE=infer` (verified locally to reproduce predictions).
- **Cache kernel** `src/cache_pipeline.py`: 16 slices/slot, 224 px, 130 mm crop, per-series 1/99,
  laterality canonicalised to left from geometry (tag/geometry agreement 1.000 on 13 tagged
  smoke studies), site-proxy manifest, two md5 shards.
- **Label audit**: hans_v4 ≡ sol56 (error φ 1.000 on all labels); Synovitis unaddressed 84%;
  Synovitis←Effusion back-fill does **not** reproduce on our blend (0.788 → 0.729, CI spans 0);
  Spanish coverage worst. All in experiments.md with verdicts.
- Docs: `proposals.md` (P-00…P-20), `research.md`, traps 6b/6c/12b/12c/15b, brainstorm reduced to
  open questions, CLAUDE.md doc map + commands + RadImageNet licence correction.

### ⏭ Next action, in order

1. **Check kernel v6** (`kaggle kernels status tiankljucanin/rsna-knee-train`). When COMPLETE:
   `kaggle kernels output ... -p artifacts/kaggle_out`, read the per-label table, s/study and
   `v02_fold0_oof.csv`; write the experiments.md entry. If the guard fired, push v6's notebook
   again with its own output attached as `kernel_sources` (resume now works — traps 8b).
2. ~~Pull cache shards~~ done — numbers in experiments.md (P-01 ✅). Do **not** download the
   arrays locally (21 GB); mount `tiankljucanin/rsna-knee-cache-a` and `-b` as `kernel_sources`
   of the training kernel and read `manifest_shard{0,1}.csv` + `<study>.npy` there.
3. **Submission #2**: attach the fold-0 output as `kernel_sources`, push with `MODE="infer"`
   (or leave `auto`), then `kaggle competitions submit -k ... -v <ver> -f submission.csv`.
   Log OOF per-label table + LB in experiments.md.
4. **Training-side cache loader** (reads `[6,16,224,224]` uint8 + manifest, forms triplets from
   neighbouring cached slices, same normalisation at inference) → then P-02 seed-noise baseline,
   P-04 8 epochs, P-08 slices.

### Open decisions for Tian

- Browser-only questions still block P-10/P-18: rules text (hosted-LLM clause, winner licence),
  Efficiency Prize formula, radimagenet.com T&C, hidden test size.
- Whether to spend the next GPU session on fold 0 **without** the cache (a real LB number today,
  ~6–8 h) or wait ~1 h for the cache and write the loader first (cheaper epochs forever). The plan
  as approved does the former; both are defensible.

### Things that will bite if forgotten

- `PYTHONUTF8=1` for **any** Python that prints non-ASCII, not just the Kaggle CLI (traps 15b).
- The submitted notebook must be the **infer-mode** one (traps 12c). A 0.500 LB = constant output.
- `kaggle/rsna-knee-cache-b` is generated by sed'ing `SHARD = 1` — never edit the `.ipynb`.
- `experiments.md` scoreboard: the v01 teacher row is superseded, not deleted.

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
