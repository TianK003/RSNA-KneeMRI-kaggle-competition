"""End-to-end baseline inference smoke test on the sample DICOM studies.

Runs the whole submission path locally on the 3 public test studies so the shape of
every stage is verified before anything is pushed to Kaggle:

  series selection -> spatial slice ordering -> 2.5D triplets -> DINOv2 encoder
  -> per-slot mean pool -> concat with a presence mask -> 12-logit head
  -> submission.csv

The head is randomly initialised, so the *numbers* are meaningless. What this checks
is that the plumbing is right: slot coverage, tensor shapes, missing-slot handling,
and that the emitted CSV matches sample_submission.csv exactly.

Usage:
  python src/baseline_infer.py                       # all sample studies
  python src/baseline_infer.py --limit 1 --slices 2  # fast smoke
"""

from __future__ import annotations

import argparse
import os

import numpy as np
import pandas as pd
import pydicom
import torch
import torch.nn as nn

from dicom_probe import (
    classify_weighting,
    has_token,
    plane_from_orientation,
    spatial_positions,
    to_hu_like,
    FATSAT_TOKENS,
    FLUID_TOKENS,
)

LABELS = [
    "ACL", "MCL", "Medial Meniscus", "Lateral Meniscus", "Medial OA", "Lateral OA",
    "PF OA", "Effusion", "Synovitis", "Baker's", "Contusion", "Fracture",
]

# Plane x acquisition slots, chosen so every finding has at least one sequence
# that shows it well. A study rarely has all six.
SLOTS = [
    "SAG_FLUID_FS", "COR_FLUID_FS", "AX_FLUID_FS",
    "SAG_FLUID_NOFS", "COR_T1", "SAG_T1",
]

IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


# ---------------------------------------------------------------- series read

def scan_series(root: str) -> pd.DataFrame:
    """One row per series, with header-derived acquisition properties."""
    rows = []
    for dirpath, _, files in os.walk(root):
        dcms = sorted(f for f in files if f.endswith(".dcm"))
        if not dcms:
            continue
        paths = [os.path.join(dirpath, f) for f in dcms]
        h = pydicom.dcmread(paths[0], stop_before_pixels=True)
        desc = " ".join(
            str(getattr(h, k, "") or "")
            for k in ("SeriesDescription", "SequenceName", "ScanOptions", "ProtocolName")
        )
        tr = getattr(h, "RepetitionTime", None)
        te = getattr(h, "EchoTime", None)
        w = classify_weighting(
            float(tr) if tr is not None else None,
            float(te) if te is not None else None,
            str(getattr(h, "ScanningSequence", "") or ""),
            desc,
        )
        fat = has_token(desc, FATSAT_TOKENS)
        rows.append({
            "study": os.path.basename(os.path.dirname(dirpath)),
            "series": os.path.basename(dirpath),
            "paths": paths,
            "n": len(paths),
            "plane": plane_from_orientation(getattr(h, "ImageOrientationPatient", None)),
            "weighting": w,
            "fat_sat": bool(fat),
            "fluid": w in ("T2", "PD") or has_token(desc, FLUID_TOKENS),
        })
    return pd.DataFrame(rows)


# Each slot: (plane, strict predicate, relaxed predicate). The relaxed tier exists
# because real studies routinely carry, say, an axial fluid series with no fat sat --
# strict-only matching silently discards usable sequences (measured: 2 of 12 series on
# the sample studies matched nothing under strict rules alone).
SLOT_SPEC: dict[str, tuple[str, object, object]] = {
    "SAG_FLUID_FS":   ("Sagittal", lambda r: r.fluid and r.fat_sat, lambda r: r.fluid),
    "COR_FLUID_FS":   ("Coronal",  lambda r: r.fluid and r.fat_sat, lambda r: r.fluid),
    "AX_FLUID_FS":    ("Axial",    lambda r: r.fluid and r.fat_sat, lambda r: r.fluid),
    "SAG_FLUID_NOFS": ("Sagittal", lambda r: r.fluid and not r.fat_sat, lambda r: r.fluid),
    "COR_T1":         ("Coronal",  lambda r: r.weighting == "T1", lambda r: not r.fluid),
    "SAG_T1":         ("Sagittal", lambda r: r.weighting == "T1", lambda r: not r.fluid),
}


def assign_slot(r) -> str | None:
    """Strict slot for reporting/EDA only; selection uses the tiered logic below."""
    for slot, (plane, strict, _) in SLOT_SPEC.items():
        if r.plane == plane and strict(r):
            return slot
    return None


def _pick_near_32(cand: pd.DataFrame) -> pd.Series:
    """Prefer a slice count near 32, avoiding unusually long 3D / high-res acquisitions."""
    return cand.iloc[(cand.n - 32).abs().to_numpy().argmin()]


def select_slots(df: pd.DataFrame, allow_reuse: bool = False) -> dict[str, pd.Series]:
    """One series per slot, strict tier first across all slots, then relaxed.

    Greedy so that a series claimed by its strict slot is not stolen by another slot's
    relaxed tier. Returns only the slots that could be filled; absences are carried
    into the head by the presence mask.
    """
    out: dict[str, pd.Series] = {}
    used: set[str] = set()

    for tier in (1, 2):
        for slot, (plane, strict, relaxed) in SLOT_SPEC.items():
            if slot in out:
                continue
            pred = strict if tier == 1 else relaxed
            cand = df[(df.plane == plane) & df.apply(pred, axis=1)]
            if not allow_reuse:
                cand = cand[~cand.series.isin(used)]
            if len(cand) == 0:
                continue
            chosen = _pick_near_32(cand)
            out[slot] = chosen
            used.add(chosen.series)
    return out


# ---------------------------------------------------------- pixel -> tensor

def ordered_paths(paths: list[str]) -> list[str]:
    """Sort by projecting ImagePositionPatient onto the slice normal.

    Falls back to InstanceNumber, then filename. NEVER trust filename order: it is the
    SOP Instance UID, uncorrelated with anatomy (measured mean |rho| ~= 0.01).
    """
    heads = [pydicom.dcmread(p, stop_before_pixels=True) for p in paths]
    pos = spatial_positions(heads)
    if pos is not None:
        return [p for _, p in sorted(zip(pos, paths), key=lambda t: t[0])]
    inst = [getattr(h, "InstanceNumber", None) for h in heads]
    if all(i is not None for i in inst):
        return [p for _, p in sorted(zip(inst, paths), key=lambda t: t[0])]
    return sorted(paths)


def make_triplets(paths: list[str], n_samples: int, gap: int, size: int) -> torch.Tensor:
    """-> (n_samples, 3, size, size); channels are slices [i-gap, i, i+gap]."""
    ordered = ordered_paths(paths)
    n = len(ordered)
    centers = np.linspace(gap, n - 1 - gap, n_samples).round().astype(int)
    centers = np.clip(centers, 0, n - 1)

    triplets = []
    for c in centers:
        idx = [max(0, c - gap), c, min(n - 1, c + gap)]
        planes = [to_hu_like(pydicom.dcmread(ordered[i])) for i in idx]
        h = min(p.shape[0] for p in planes)
        w = min(p.shape[1] for p in planes)
        stack = np.stack([p[:h, :w] for p in planes], axis=-1).astype(np.float32)

        # Clip the triplet jointly so the three channels stay comparable.
        lo, hi = np.percentile(stack, [1, 99])
        stack = np.clip(stack, lo, hi)
        stack = (stack - lo) / max(hi - lo, 1e-6)

        t = torch.from_numpy(stack).permute(2, 0, 1).unsqueeze(0)
        t = torch.nn.functional.interpolate(
            t, size=(size, size), mode="bilinear", align_corners=False
        ).squeeze(0)
        t = (t - torch.tensor(IMAGENET_MEAN)[:, None, None]) / \
            torch.tensor(IMAGENET_STD)[:, None, None]
        triplets.append(t)
    return torch.stack(triplets)


# ------------------------------------------------------------------- model

class SlotHead(nn.Module):
    """Shared encoder over all slots; concat pooled slot features + presence mask."""

    def __init__(self, encoder, dim: int, n_labels: int = 12):
        super().__init__()
        self.encoder = encoder
        self.dim = dim
        self.head = nn.Linear(dim * len(SLOTS) + len(SLOTS), n_labels)

    def forward(self, slot_batches: dict[str, torch.Tensor]) -> torch.Tensor:
        feats, mask = [], []
        for slot in SLOTS:
            if slot in slot_batches:
                out = self.encoder(pixel_values=slot_batches[slot])
                # CLS token; mean-pool across sampled slices.
                f = out.last_hidden_state[:, 0].mean(dim=0)
                feats.append(f)
                mask.append(1.0)
            else:
                feats.append(torch.zeros(self.dim))
                mask.append(0.0)
        x = torch.cat(feats + [torch.tensor(mask)])
        return self.head(x)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="data/sample_dicom")
    ap.add_argument("--weights", default="models/dinov2_small")
    ap.add_argument("--out", default="artifacts/submission.csv")
    ap.add_argument("--sample-sub", default="data/sample_submission.csv")
    ap.add_argument("--size", type=int, default=224)
    ap.add_argument("--slices", type=int, default=4)
    ap.add_argument("--gap", type=int, default=2)
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()

    torch.manual_seed(0)

    from transformers import Dinov2Model
    print(f"loading encoder from {args.weights} ...")
    enc = Dinov2Model.from_pretrained(args.weights)
    enc.eval()
    dim = enc.config.hidden_size
    print(f"  hidden_size={dim}  params={sum(p.numel() for p in enc.parameters())/1e6:.1f}M")

    model = SlotHead(enc, dim).eval()

    df = scan_series(args.root)
    df["slot"] = df.apply(assign_slot, axis=1)
    print(f"\n{len(df)} series over {df.study.nunique()} studies")
    print(df.groupby(["plane", "weighting", "fat_sat"]).size().to_string())
    unassigned = df[df.slot.isna()]
    if len(unassigned):
        print(f"\n{len(unassigned)} series matched no slot:")
        print(unassigned[["plane", "weighting", "fat_sat", "fluid"]].to_string(index=False))

    studies = sorted(df.study.unique())
    if args.limit:
        studies = studies[: args.limit]

    rows = []
    for si, study in enumerate(studies, 1):
        sdf = df[df.study == study]
        slots = select_slots(sdf)
        print(f"\n[{si}/{len(studies)}] {study[-12:]}  "
              f"{len(slots)}/{len(SLOTS)} slots: {sorted(slots)}")
        batches = {}
        for slot, row in slots.items():
            t = make_triplets(row.paths, args.slices, args.gap, args.size)
            batches[slot] = t
            print(f"    {slot:<15} {tuple(t.shape)}  from {row.n} slices "
                  f"({row.plane}/{row.weighting}/fs={row.fat_sat})")
        with torch.no_grad():
            logits = model(batches)
        probs = torch.sigmoid(logits).numpy()
        rows.append({"StudyInstanceUID": study, **dict(zip(LABELS, probs))})

    sub = pd.DataFrame(rows)
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    sub.to_csv(args.out, index=False)
    print(f"\nwrote {args.out}")

    # ---- validate against sample_submission.csv --------------------------
    ok = True
    try:
        ref = pd.read_csv(args.sample_sub)
    except FileNotFoundError:
        print(f"! {args.sample_sub} missing, cannot validate columns")
        return

    if list(sub.columns) != list(ref.columns):
        ok = False
        print("! COLUMN MISMATCH")
        print(f"  expected: {list(ref.columns)}")
        print(f"  got:      {list(sub.columns)}")
    else:
        print(f"columns match sample_submission ({len(ref.columns)} cols)")

    if args.limit == 0:
        missing = set(ref.StudyInstanceUID) - set(sub.StudyInstanceUID)
        extra = set(sub.StudyInstanceUID) - set(ref.StudyInstanceUID)
        if missing or extra:
            ok = False
            print(f"! study mismatch: {len(missing)} missing, {len(extra)} extra")
        else:
            print(f"all {len(ref)} sample-submission studies covered")

    vals = sub[LABELS].to_numpy()
    if not np.isfinite(vals).all():
        ok = False
        print("! non-finite predictions")
    elif vals.min() < 0 or vals.max() > 1:
        ok = False
        print(f"! predictions outside [0,1]: {vals.min():.3f}..{vals.max():.3f}")
    else:
        print(f"predictions finite, in [{vals.min():.3f}, {vals.max():.3f}]")

    print(f"\n{'SMOKE TEST PASSED' if ok else 'SMOKE TEST FAILED'} "
          f"(head is random -- values carry no signal)")


if __name__ == "__main__":
    main()
