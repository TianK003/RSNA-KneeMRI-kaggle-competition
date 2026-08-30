"""Unit checks for the P-25 window path and the timm backbone family (CPU, no GPU, ~2 min).

Executes src/kaggle_pipeline.py up to its run section (RSNA_DEFS_ONLY=1) and exercises:
  * window enumeration counts for both cache schemes (c02: 60 with all six slots present; c01: 84)
  * stratified training sampling never touches an absent slot and never repeats a window
  * equidistant eval subset
  * WindowAttnHead: finite output in fp16 and fp32, with 1 and 60 windows, with padding masks
  * build_model for every BACKBONES family present under models/ (offline timm load with the
    strict-load report), forward shapes at the family's img_size, param_groups covering every
    parameter exactly once with the right LR depths
  * forward_windows on a real c02 array read from the local blob (DINOv2, 4 windows)
Exit status is the verdict.

    export PYTHONUTF8=1
    .venv/Scripts/python.exe src/window_head_test.py
"""
import os
import sys

import numpy as np
import torch

os.environ["RSNA_DEFS_ONLY"] = "1"
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(ROOT)


def load_defs(path):
    ns = {"__name__": "defs_kaggle_pipeline", "__file__": path}
    with open(path, encoding="utf-8") as f:
        code = compile(f.read(), path, "exec")
    try:
        exec(code, ns)
    except SystemExit as e:
        if e.code not in (0, None):
            raise
    return ns


def main():
    fails = []

    def check(cond, msg):
        print(("  ok   " if cond else "  FAIL ") + msg)
        if not cond:
            fails.append(msg)

    K = load_defs(os.path.join("src", "kaggle_pipeline.py"))
    Config, SLOTS, LABELS = K["Config"], K["SLOTS"], K["LABELS"]

    print("\n== windows")
    c02 = Config(cache_scheme="c02")
    c01 = Config(cache_scheme="c01")
    full = np.ones(len(SLOTS), np.float32)
    cen, sid = K["valid_windows"](full, c02)
    check(len(cen) == 60 and sid.max() == 5, f"c02 all slots -> {len(cen)} windows")
    cen1, sid1 = K["valid_windows"](full, c01)
    check(len(cen1) == 84, f"c01 all slots -> {len(cen1)} windows")
    part = np.array([1, 1, 0, 1, 0, 0], np.float32)
    cen, sid = K["valid_windows"](part, c02)
    check(len(cen) == (18 - 2) + (12 - 2) + (14 - 2) and set(sid) == {0, 1, 3},
          f"c02 slots 0,1,3 -> {len(cen)} windows from slots {sorted(set(sid))}")
    np.random.seed(0)
    for n in (4, 24, 100):
        tc, ts = K["sample_train_windows"](cen, sid, n)
        pairs = list(zip(tc.tolist(), ts.tolist()))
        check(len(pairs) == min(n, len(cen)) and len(set(pairs)) == len(pairs) and set(ts) <= {0, 1, 3}
              and (n < 6 or all(int((ts == s).sum()) >= 2 for s in (0, 1, 3))),
              f"train sample n={n}: {len(pairs)} unique windows, per-slot {[int((ts == s).sum()) for s in (0, 1, 3)]}")
    ec, es = K["eval_windows_subset"](cen, sid, 10)
    check(len(ec) == 10 and ec[0] == cen[0] and ec[-1] == cen[-1], f"eval subset 10 spans the list")
    ec, es = K["eval_windows_subset"](cen, sid, 0)
    check(len(ec) == len(cen), "eval subset 0 = all")

    print("\n== WindowAttnHead")
    head = K["WindowAttnHead"](384, len(LABELS))
    for W in (1, 60):
        feats = torch.randn(1, W, 384)
        slot_id = torch.randint(0, 6, (1, W))
        out = head(feats, slot_id)
        check(out.shape == (1, 12) and torch.isfinite(out).all(), f"fp32 W={W} -> {tuple(out.shape)} finite")
        valid = torch.ones(1, W, dtype=torch.bool)
        valid[0, W // 2:] = False
        out_m = head(feats.half(), slot_id, valid) if W > 1 else head(feats.half(), slot_id, valid.fill_(True))
        check(torch.isfinite(out_m.float()).all(), f"fp16 W={W} masked -> finite")
    dead = torch.zeros(1, 3, dtype=torch.bool)
    check(torch.isfinite(head(torch.randn(1, 3, 384), torch.zeros(1, 3, dtype=torch.long), dead)).all(),
          "all-masked row -> finite (uniform fallback)")

    print("\n== backbones / build_model / param_groups")
    device = torch.device("cpu")
    for bb, img in (("dinov2", 224), ("convnext_tiny", 224),
                    ("timm:coatnet_rmlp_1_rw_224", 224), ("timm:coatnet_rmlp_2_rw_384", 384)):
        try:
            K["resolve_backbone_dir"](bb)
        except SystemExit as e:
            print(f"  skip {bb}: {e}")
            continue
        cfg = Config(cache_scheme="c02", backbone=bb, img_size=img, head_type="window_attn",
                     window_mode="random", lr_backbone=1e-4)
        model = K["build_model"](cfg, device)
        n_all = sum(p.numel() for p in model.parameters() if p.requires_grad)
        groups = K["param_groups"](model, cfg)
        n_grp = sum(p.numel() for g in groups for p in g["params"])
        ids = [id(p) for g in groups for p in g["params"]]
        check(n_grp == n_all and len(ids) == len(set(ids)), f"{bb}: param_groups cover {n_grp:,} == {n_all:,} params once")
        lrs = sorted({g["lr"] for g in groups})
        check(lrs[-1] == cfg.lr_head and any(abs(l - cfg.lr_backbone) < 1e-12 for l in lrs),
              f"{bb}: LR set {['%.1e' % l for l in lrs]}")
        model.eval()
        with torch.no_grad():
            x = torch.randn(2, 3, img, img)
            f = model.encode(x)
            check(f.shape == (2, model.dim), f"{bb}: encode {tuple(x.shape)} -> {tuple(f.shape)}")

    print("\n== forward_windows on the local c02 blob")
    local = os.path.join("artifacts", "cache_local", K["cache_version_for"](c02))
    side = [f for f in sorted(os.listdir(local)) if f.endswith(".csv")] if os.path.isdir(local) else []
    if not side:
        check(False, f"no local c02 blob under {local}")
    else:
        import pandas as pd
        sc = pd.read_csv(os.path.join(local, side[0]), dtype={"mask": str})
        r = sc[sc.cached == 1].iloc[0]
        arr = K["read_cached"]((os.path.join(local, r.blob), int(r.row)))
        mask = np.array([float(c) for c in r["mask"]], np.float32)
        cen, sid = K["valid_windows"](mask, c02)
        np.random.seed(1)
        tc, ts = K["sample_train_windows"](cen, sid, 4)
        cfg = Config(cache_scheme="c02", backbone="dinov2", img_size=224, head_type="window_attn",
                     window_mode="random")
        model = K["build_model"](cfg, device).eval()
        starts, total = K["slot_offsets"](K["cache_geom"](cfg)[2])
        check(arr.shape == (total, 336, 336), f"blob study shape {arr.shape}")
        with torch.no_grad():
            out = model.forward_windows(torch.from_numpy(arr), torch.from_numpy(tc), torch.from_numpy(ts), starts)
        check(out.shape == (1, 12) and torch.isfinite(out).all(), f"forward_windows 4 windows -> {tuple(out.shape)}")
        # the gathered triplet must be the slot's own slices: compare one window by hand
        s0, c0 = int(ts[0]), int(tc[0])
        rows = starts[s0] + np.array([c0 - 1, c0, c0 + 1])
        check(np.array_equal(arr[rows], K["slot_stacks"](arr, cfg)[s0][c0 - 1:c0 + 2]),
              f"window (slot {s0}, centre {c0}) gathers rows {rows.tolist()} of its own slot")
        # the Dataset's window item and forward_batch agree with a direct call
        b = {"arr": torch.from_numpy(arr)[None], "centres": torch.from_numpy(tc)[None],
             "slot_id": torch.from_numpy(ts)[None], "mask": torch.from_numpy(mask)[None]}
        with torch.no_grad():
            out2 = K["forward_batch"](model, b, device, cfg)
        check(torch.allclose(out, out2), "forward_batch == forward_windows")
        # fixed-mode array_to_tensor works on the flat array too, and TTA offsets differ from 0
        cfg_fixed = Config(cache_scheme="c02", slices_per_slot=6)     # smoke clamps K to 2
        Kf = cfg_fixed.slices_per_slot
        x0, m0 = K["array_to_tensor"](arr, mask, cfg_fixed, False, centre_offset=0)
        x1, _ = K["array_to_tensor"](arr, mask, cfg_fixed, False, centre_offset=1)
        check(x0.shape == (6, Kf, 3, 224, 224) and not torch.equal(x0, x1),
              f"fixed-mode on flat array {tuple(x0.shape)} (K={Kf}), offset view differs")
        probs = K["pool_views"](torch.rand(3, 2, 12), "focal")
        check(probs.shape == (2, 12), "focal pooling shape")

    print("\n" + ("UNIT CHECKS PASSED" if not fails else f"UNIT CHECKS FAILED ({len(fails)}):\n  - " + "\n  - ".join(fails)))
    sys.exit(1 if fails else 0)


if __name__ == "__main__":
    main()
