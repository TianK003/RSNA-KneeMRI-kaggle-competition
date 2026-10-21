# %% [markdown]
# # RSNA Knee V-JEPA solo (candidates.md C21: a solo read of a new public family)
#
# nurikw3's public V-JEPA 2.1 ViT-B 5-fold checkpoints (Dataset `nurikw3/rsna-knee-vjepa21-checkpoints`, CC0; the vendored
# `facebookresearch/vjepa2` code is MIT), scored alone with the author's own `infer.py`, read from the Dataset and patched in memory:
# - fp16 autocast instead of bf16 (a T4 has no bf16 tensor cores);
# - the competition directory found by a depth-limited glob (Kaggle mounts competitions under `/kaggle/input/competitions/`);
# - the predictions written as `submission.csv` in the sample's column order.
# On the placeholder test (≤ 10 studies) the cell also scores fp32 and prints the fp16 / fp32 difference and the time per study.
# Built from `src/vjepa_solo.py` by `src/nbgen.py`; experiments.md 2026-10-21 "C21: V-JEPA solo".

# %%
import contextlib, glob, sys, time, types
from pathlib import Path
import numpy as np
import pandas as pd
import torch

T0 = time.time()


def find_one(tail):
    # never hard-code /kaggle/input (CLAUDE.md constraint 4): depth-limited globs, shallowest first
    for depth in range(1, 6):
        hits = sorted(glob.glob("/kaggle/input/" + "*/" * depth + tail))
        if hits:
            return Path(hits[0])
    raise SystemExit(f"{tail} not found under /kaggle/input")


VJ = find_one("f0_best.pt").parent                      # infer.py, finetune_model.py, vjepa2/, einops/, f*_best.pt
COMP = find_one("test_series.csv").parent               # test.csv, test_series.csv, sample_submission.csv, test_series/
assert (VJ / "infer.py").is_file() and (COMP / "test.csv").is_file(), (VJ, COMP)
print("V-JEPA dir:", VJ, "| competition dir:", COMP, "| folds:", len(list(VJ.glob("f*_best.pt"))), flush=True)
sys.path[:0] = [str(VJ), str(VJ / "vjepa2")]

AMP = "torch.autocast('cuda', dtype=torch.bfloat16)"
SRC = (VJ / "infer.py").read_text(encoding="utf-8")
assert SRC.count(AMP) == 1, "infer.py changed: the bf16 autocast line is not there exactly once"
SRC = SRC.replace(AMP, "AMP_CTX()")


def load_infer(amp_ctx):
    mod = types.ModuleType("vj_infer")
    mod.__file__ = str(VJ / "infer.py")                  # its THIS_DIR = the Dataset, so its own sys.path inserts hold
    mod.AMP_CTX = amp_ctx
    exec(compile(SRC, str(VJ / "infer.py"), "exec"), mod.__dict__)
    return mod


test = pd.read_csv(COMP / "test.csv", dtype={"StudyInstanceUID": str})
sample = pd.read_csv(COMP / "sample_submission.csv")
labels = [c for c in sample.columns if c != "StudyInstanceUID"]
fp16 = load_infer(lambda: torch.autocast("cuda", dtype=torch.float16))
t = time.time()
pred = fp16.run_inference(competition_dir=COMP, output_csv=Path("/kaggle/working/vjepa_fp16.csv"), checkpoint_dir=VJ)
sec = time.time() - t
print(f"fp16: {len(pred)} studies in {sec:.0f} s ({sec / max(len(pred), 1):.1f} s / study, model load included)", flush=True)

pred = pred.set_index("StudyInstanceUID")
sub = pd.DataFrame({"StudyInstanceUID": test.StudyInstanceUID})
for c in labels:
    sub[c] = sub.StudyInstanceUID.map(pred[c]).astype(float)
nan = int(sub[labels].isna().sum().sum())
sub[labels] = sub[labels].fillna(0.5)
assert np.isfinite(sub[labels].values).all() and ((sub[labels].values >= 0) & (sub[labels].values <= 1)).all()
sub.to_csv("/kaggle/working/submission.csv", index=False)
print("submission.csv:", sub.shape, "| filled NaN:", nan, flush=True)

if len(test) <= 10:                                      # placeholder only: fp16 vs fp32 on the same studies
    fp32 = load_infer(contextlib.nullcontext)
    ref = fp32.run_inference(competition_dir=COMP, output_csv=Path("/kaggle/working/vjepa_fp32.csv"),
                             checkpoint_dir=VJ).set_index("StudyInstanceUID")
    d = np.abs(ref.loc[pred.index, labels].values - pred[labels].values)
    print(f"fp16 vs fp32: max |d| {d.max():.2e}, mean {d.mean():.2e}", flush=True)
print(f"VJEPA_SOLO_DONE {time.time() - T0:.0f} s", flush=True)
