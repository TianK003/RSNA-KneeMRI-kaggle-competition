# Setting up on a new machine

**Short answer: do not copy `data/`. Re-fetch it.** Everything in `data/`, `models/`, and
`artifacts/` is either downloadable in a few minutes or regenerable from the repo, and all
three are gitignored on purpose. Copying `data/` by hand risks a partial tree that fails
silently later (we already got burned by exactly that — see [traps.md](traps.md)).

The one thing genuinely worth copying is the **sample DICOM tree**, and only because Kaggle
rate-limits it. More on that below.

---

## 1. Clone

```bash
git clone https://github.com/TianK003/RSNA-KneeMRI-kaggle-competition.git
cd RSNA-KneeMRI-kaggle-competition
```

## 2. Python environment

Python 3.11+ (this workspace was verified on 3.13). Install torch from the **CPU index** or
pip pulls a ~2.5 GB CUDA build you will never use — no training happens locally.

```bash
python -m venv .venv
# Windows (Git Bash):  source .venv/Scripts/activate
# Linux / macOS:       source .venv/bin/activate

pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements.txt
```

## 3. Kaggle authentication

```bash
kaggle auth login          # browser OAuth, caches credentials — recommended
```

Then **accept the competition rules in the browser**, or every download returns
`403 Forbidden` even though auth succeeded:
https://www.kaggle.com/competitions/rsna-knee-abnormality-detection/rules

Verify — `userHasEntered` must be `True`:

```bash
export PYTHONUTF8=1
kaggle competitions list -s "rsna-knee-abnormality" --csv
```

> `PYTHONUTF8=1` is needed on Windows for any `kaggle ... --csv` call: team names and
> reports contain non-ASCII and `charmap` otherwise throws.

## 4. Competition CSVs (~9 MB, seconds)

```bash
mkdir -p data
for f in train.csv train_series.csv test.csv test_series.csv sample_submission.csv; do
  kaggle competitions download -c rsna-knee-abnormality-detection -f "$f" -p data --force
done
```

Sanity check — `train.csv` must be 4,407 records with 58 fully labelled:

```bash
python -c "
import pandas as pd
d=pd.read_csv('data/train.csv'); L=[c for c in d.columns if c not in ('StudyInstanceUID','Report')]
print('records', len(d), '| gold', int(d[L].notna().all(axis=1).sum()), '| labels', len(L))"
```

Expect `records 4407 | gold 58 | labels 12`.

## 5. LLM report labels (~1 MB, seconds)

These are the actual training signal — without them the pipeline falls back to prior-only
targets and trains nothing useful (it says so loudly, but still).

```bash
mkdir -p data/llm_labels
for d in pilkwang/rsna-knee-llm-labels \
         stevenleehans/rsna-knee-llm-report-labels \
         lixin73/rsna-knee-llm-report-labels-sol56; do
  kaggle datasets download "$d" -p "data/llm_labels/$(basename $d)" --unzip -q
done
```

## 6. Model weights (~180 MB)

```bash
mkdir -p models
kaggle models instances versions download metaresearch/dinov2/PyTorch/small/1 \
  -p models/dinov2_small --untar
kaggle datasets download marwanmath/resnet-50-radimagenet-marwan \
  -p models/radimagenet_r50 --unzip
```

## 7. Verify the install

```bash
export PYTHONUTF8=1 PYTHONPATH=src
python src/build_targets.py         # should print teacher gold macro-AUC 0.8948 (blend)
python src/label_audit.py           # label audit -> artifacts/label_audit.md (langdetect optional)
python src/kaggle_pipeline.py       # local CPU smoke run
python src/cache_pipeline.py        # local smoke of the cache kernel (serial on Windows)
```

`build_targets.py` is the strongest check: it must report **teacher gold macro-AUC 0.8948**
(and 0.8934 for the diagnostic rank blend)
and fold sizes **882/882/881/881/881** with gold **11/12/12/12/11**. Those are
deterministic and machine-independent — if they differ, a label source is missing or a
different version got downloaded. `kaggle_pipeline.py` additionally needs the sample DICOMs
(next step) to do anything interesting.

---

## The sample DICOMs — the one thing worth copying by hand

`data/sample_dicom/` is the public test tree: **3 studies, 15 series, 557 files, 572 MB**.
It is the only image data that exists locally, and it is what `src/dicom_probe.py` and the
local smoke run operate on.

**Kaggle rate-limits these downloads hard (HTTP 429)** — one file at a time, no folder
download, and the CLI flattens nested paths so the `study/series/` tree has to be rebuilt
manually. Our own attempt stalled at 459/557 and never recovered within the quota window.

So for a second machine, **copy `data/sample_dicom/` directly** (USB, syncing folder,
whatever). It is competition data, so keep it off any public location — the repo already
gitignores it.

If you must re-download it, go sequential with backoff and **verify by file count, not exit
status** — the API returns exit code 0 on a 429:

```bash
find data/sample_dicom -name '*.dcm' -size +0 | wc -l    # want 557
```

You can work without it: `build_targets.py`, all report/CSV analysis, and notebook authoring
need no images at all. Only the local *smoke* run and `dicom_probe.py` do, and both are
convenience checks — the authoritative verification is a `FORCE_SMOKE = True` push to Kaggle,
which runs against the full dataset anyway.

---

## What is intentionally NOT in the repo

| Path | Why | How to get it |
|---|---|---|
| `data/*.csv` | competition data | step 4 |
| `data/llm_labels/` | competition-derived | step 5 |
| `data/sample_dicom/` | 572 MB of competition images | copy, or step in §"sample DICOMs" |
| `models/` | 180 MB of third-party weights | step 6 |
| `artifacts/` | regenerable; **embeds StudyInstanceUIDs** | re-run the scripts |
| `.venv/` | machine-specific | step 2 |

Never commit any of these. `artifacts/` in particular contains `StudyInstanceUID`s and
report-derived targets, which are competition data.

## Where the real work happens

Nothing above trains anything. The local machine is for CSV/report analysis, label tooling,
DICOM header work, notebook authoring, and CLI orchestration. **All DICOM decoding and
training happens on Kaggle**, because the image set is ~570 GB. See the Kaggle-to-Kaggle
workflow in [../CLAUDE.md](../CLAUDE.md).

So "working from the laptop" mostly means: clone, install, authenticate, then
`python src/nbgen.py ...` and `kaggle kernels push`. The heavy compute is identical from any
machine — it is not on your machine at all.
