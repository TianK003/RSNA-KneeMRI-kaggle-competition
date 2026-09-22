"""Unit checks for the P-25 window path, the timm backbone family, and the P-31/P-32/P-33 additions (CPU, no GPU, ~2 min).

Executes src/kaggle_pipeline.py up to its run section (RSNA_DEFS_ONLY=1) and exercises:
  * window enumeration counts for both cache schemes (c02: 60 with all six slots present; c01: 84)
  * stratified training sampling never touches an absent slot and never repeats a window
  * equidistant eval subset
  * WindowAttnHead: finite output in fp16 and fp32, with 1 and 60 windows, with padding masks
  * build_model for every BACKBONES family present under models/ (offline timm load with the
    strict-load report), forward shapes at the family's img_size, param_groups covering every
    parameter exactly once with the right LR depths
  * forward_windows on a real c02 array read from the local blob (DINOv2, 4 windows)
  * P-32: collate_windows over two studies with different window counts; the batched forward equals
    the two single-study forwards in eval mode (the padded head path smoke never exercises); forward_batch
    refuses a default-collated window batch; weighted_bce normalises per study; c01 + batch > 1 refuses
  * P-33: Config.aug guard, affine_theta geometry (zoom-in = 1/z, 5 % shift = 0.10), augment_light range
  * P-31: nbgen embeds the pipeline source (zlib + base64 + sha256) only when PARALLEL_ARMS is set
Exit status is the verdict.

    export PYTHONUTF8=1
    .venv/Scripts/python.exe src/window_head_test.py
"""
import base64
import hashlib
import json
import os
import sys
import tempfile
import zlib

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

    print("\n== train_all / swa (P-28)")
    import pandas as pd
    a = {"w": torch.tensor([0.0, 2.0]), "n": torch.tensor(3, dtype=torch.int64), "h": torch.tensor([1.0], dtype=torch.float16)}
    b = {"w": torch.tensor([2.0, 4.0]), "n": torch.tensor(7, dtype=torch.int64), "h": torch.tensor([3.0], dtype=torch.float16)}
    avg = K["average_state_dicts"]([a, b])
    check(torch.equal(avg["w"], torch.tensor([1.0, 3.0])) and int(avg["n"]) == 7
          and avg["h"].dtype == torch.float16 and float(avg["h"]) == 2.0,
          f"average_state_dicts: floats averaged {avg['w'].tolist()}, int copied from last {int(avg['n'])}, fp16 kept")
    fake = pd.DataFrame({"StudyInstanceUID": [f"s{i}" for i in range(10)],
                         "fold": [0, 1, 2, 3, 4, 0, 1, 2, 3, 4],
                         "is_gold": [1, 0, 0, 1, 0, 0, 0, 1, 0, 0]})
    cfg_all = Config(train_all=True, ckpt_policy="last", swa_last=3)
    tr, va = K["split_studies"](fake, 0, cfg_all)
    check(len(tr) == 7 and set(va) == {"s0", "s3", "s7"} and not set(tr) & set(va) and set(tr) | set(va) == set(fake.StudyInstanceUID),
          f"split_studies train_all: train {len(tr)} non-gold, val {sorted(va)} gold, disjoint, complete")
    tr0, va0 = K["split_studies"](fake, 0, Config())
    check(set(va0) == {"s0", "s5"} and len(tr0) == 8, "split_studies fold 0 keeps the old fold split")
    check(cfg_all.folds == (0,) and cfg_all.swa_last == 1,
          f"Config(train_all, swa_last=3) -> folds {cfg_all.folds}, swa_last clamped to epochs -> {cfg_all.swa_last} (smoke)")
    try:
        Config(train_all=True)
        check(False, "Config(train_all=True) with best_oof must raise")
    except SystemExit:
        check(True, "Config(train_all=True) with ckpt_policy=best_oof raises SystemExit")

    print("\n== Config: aug guard, smoke window clamp (P-32 / P-33)")
    check(Config().aug == "none", "Config.aug defaults to 'none'")
    try:
        Config(aug="light")                                  # window_mode defaults to "fixed"
        check(False, "Config(aug='light') with window_mode='fixed' must raise")
    except SystemExit:
        check(True, "Config(aug='light', window_mode='fixed') raises SystemExit (the aug only runs in window mode)")
    c_aug = Config(cache_scheme="c02", window_mode="random", head_type="window_attn", aug="light")
    check(c_aug.aug == "light", "Config(aug='light', window_mode='random') accepted")
    os.environ["RSNA_SMOKE_FULL_WINDOWS"] = "1"
    tw_full = Config(cache_scheme="c02", window_mode="random", head_type="window_attn").train_windows
    del os.environ["RSNA_SMOKE_FULL_WINDOWS"]
    tw_smoke = Config(cache_scheme="c02", window_mode="random", head_type="window_attn").train_windows
    check(tw_full == 24 and tw_smoke == 4,
          f"smoke clamps train_windows to {tw_smoke}; RSNA_SMOKE_FULL_WINDOWS=1 keeps {tw_full}")

    print("\n== light augmentation (P-33)")
    th = K["affine_theta"](torch.zeros(3), torch.full((3,), 1.08), torch.zeros(3), torch.zeros(3))
    check(th.shape == (3, 2, 3) and torch.allclose(th[:, 0, 0], torch.full((3,), 1 / 1.08))
          and torch.allclose(th[:, 1, 1], torch.full((3,), 1 / 1.08)) and float(th[:, :, 2].abs().max()) == 0.0,
          "affine_theta: zoom-in 1.08 -> grid scale 1/1.08 (a smaller source patch fills the output)")
    th = K["affine_theta"](torch.zeros(1), torch.ones(1), torch.tensor([0.05]), torch.zeros(1))
    check(torch.allclose(th[0, 0, 2], torch.tensor(0.10)) and float(th[0, 1, 2]) == 0.0,
          "affine_theta: a 5 % shift is 0.10 in normalised coordinates")
    th = K["affine_theta"](torch.tensor([90.0]), torch.ones(1), torch.zeros(1), torch.zeros(1))
    check(torch.allclose(th[0, :, :2].abs(), torch.tensor([[0.0, 1.0], [1.0, 0.0]]), atol=1e-6),
          "affine_theta: a 90 degree rotation swaps the axes")
    x = torch.rand(8, 3, 64, 64)
    torch.manual_seed(0)
    y = K["augment_light"](x)
    check(y.shape == x.shape and y.dtype == x.dtype and float(y.min()) >= 0.0 and float(y.max()) <= 1.0
          and not torch.equal(x, y) and torch.isfinite(y).all(),
          "augment_light: same shape / dtype, stays in [0, 1], changes the windows")
    check(torch.equal(K["augment_light"](x, p=0.0), x), "augment_light p=0 is the identity")
    xh = torch.rand(4, 3, 32, 32).half()
    check(K["augment_light"](xh).dtype == torch.float16, "augment_light keeps a half input half (theta built in fp32)")

    print("\n== weighted_bce per-study normalisation (P-32)")
    logits, yt = torch.randn(2, 12), torch.rand(2, 12)
    w = torch.stack([torch.full((12,), 0.5), torch.full((12,), 8.0)])
    per = torch.stack([K["weighted_bce"](logits[i:i + 1], yt[i:i + 1], w[i:i + 1]) for i in range(2)])
    check(torch.allclose(K["weighted_bce"](logits, yt, w), per.mean(), atol=1e-6),
          "weighted_bce on a 2-study batch = mean of the per-study losses (the 8x gold study does not swallow its partner)")
    w1 = torch.rand(1, 12) + 0.1
    check(torch.allclose(K["weighted_bce"](logits[:1], yt[:1], w1),
                         (torch.nn.functional.binary_cross_entropy_with_logits(logits[:1], yt[:1], reduction="none") * w1).sum() / w1.sum()),
          "weighted_bce at B=1 is unchanged")

    print("\n== forward_windows on the local c02 blob, batched (P-32)")
    local = os.path.join("artifacts", "cache_local", K["cache_version_for"](c02))
    side = [f for f in sorted(os.listdir(local)) if f.endswith(".csv")] if os.path.isdir(local) else []
    if not side:
        check(False, f"no local c02 blob under {local}")
    else:
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
        t_arr = torch.from_numpy(np.ascontiguousarray(arr))
        with torch.no_grad():
            out = model.forward_windows(t_arr[None], torch.from_numpy(tc), torch.from_numpy(ts),
                                        torch.zeros(4, dtype=torch.long), torch.arange(4), starts)
        check(out.shape == (1, 12) and torch.isfinite(out).all(), f"forward_windows 4 windows -> {tuple(out.shape)}")
        # the gathered triplet must be the slot's own slices: compare one window by hand
        s0, c0 = int(ts[0]), int(tc[0])
        rows = starts[s0] + np.array([c0 - 1, c0, c0 + 1])
        check(np.array_equal(arr[rows], K["slot_stacks"](arr, cfg)[s0][c0 - 1:c0 + 2]),
              f"window (slot {s0}, centre {c0}) gathers rows {rows.tolist()} of its own slot")

        def item(tcx, tsx, gold):
            return {"study": f"s{gold}", "arr": t_arr, "centres": torch.from_numpy(tcx.astype(np.int64)),
                    "slot_id": torch.from_numpy(tsx.astype(np.int64)), "mask": torch.from_numpy(mask),
                    "y": torch.rand(12), "w": torch.full((12,), 8.0 if gold else 0.5),
                    "is_gold": torch.tensor(float(gold))}

        np.random.seed(2)
        tcB, tsB = K["sample_train_windows"](cen, sid, 2)
        batch = K["collate_windows"]([item(tc, ts, 0), item(tcB, tsB, 1)])
        check(tuple(batch["arr"].shape) == (2, total, 336, 336) and tuple(batch["centres"].shape) == (6,)
              and batch["study_ix"].tolist() == [0, 0, 0, 0, 1, 1] and batch["pos"].tolist() == [0, 1, 2, 3, 0, 1]
              and tuple(batch["y"].shape) == (2, 12) and tuple(batch["mask"].shape) == (2, 6)
              and batch["study"] == ["s0", "s1"] and batch["is_gold"].tolist() == [0.0, 1.0],
              "collate_windows: arr stacked, windows concatenated with study_ix / pos, targets stacked")
        with torch.no_grad():
            both = K["forward_batch"](model, batch, device, cfg)
            single_a = K["forward_batch"](model, K["collate_windows"]([item(tc, ts, 0)]), device, cfg)
            single_b = K["forward_batch"](model, K["collate_windows"]([item(tcB, tsB, 1)]), device, cfg)
        check(both.shape == (2, 12) and torch.allclose(both[0], single_a[0], atol=1e-4)
              and torch.allclose(both[1], single_b[0], atol=1e-4) and torch.allclose(single_a, out, atol=1e-5),
              "batched forward (4 + 2 windows, padded head) == the two single-study forwards, eval mode")
        try:
            K["forward_batch"](model, {"arr": t_arr[None], "centres": torch.from_numpy(tc)[None],
                                       "slot_id": torch.from_numpy(ts)[None], "mask": torch.from_numpy(mask)[None]},
                               device, cfg)
            check(False, "forward_batch without the window collate must raise")
        except SystemExit:
            check(True, "forward_batch refuses a default-collated window batch (no study_ix)")
        # c01 dense arrays: batch 1 still works through the flat branch; batch > 1 is refused, not misread
        cfg01 = Config(cache_scheme="c01", backbone="dinov2", img_size=224, head_type="window_attn", window_mode="random")
        arr01 = torch.from_numpy((np.random.rand(6, 16, 224, 224) * 255).astype(np.uint8))
        with torch.no_grad():
            o01 = model.forward_windows(arr01[None], torch.tensor([3, 5]), torch.tensor([0, 2]),
                                        torch.zeros(2, dtype=torch.long), torch.arange(2), None)
        check(o01.shape == (1, 12) and torch.isfinite(o01).all(), "c01 dense array, batch 1 -> (1, 12)")
        try:
            with torch.no_grad():
                model.forward_windows(torch.stack([arr01, arr01]), torch.tensor([3, 5]), torch.tensor([0, 2]),
                                      torch.tensor([0, 1]), torch.tensor([0, 0]), None)
            check(False, "c01 dense array with batch 2 must raise")
        except SystemExit:
            check(True, "c01 dense array with batch 2 raises SystemExit (no c01 window member exists)")
        # aug="light" leaves eval untouched; in train mode (no dropout) it changes the output
        cfg_l = Config(cache_scheme="c02", backbone="dinov2", img_size=224, head_type="window_attn",
                       window_mode="random", aug="light", dropout=0.0)
        model_l = K["build_model"](cfg_l, device)
        model_l.load_state_dict(model.state_dict())
        model_l.eval()
        with torch.no_grad():
            out_l = K["forward_batch"](model_l, K["collate_windows"]([item(tc, ts, 0)]), device, cfg_l)
        check(torch.allclose(out_l, out, atol=1e-5), "aug='light' in eval mode == aug='none' (inference never augments)")
        check(getattr(model_l, "aug", None) == "light" and getattr(model, "aug", None) == "none",
              "build_model passes cfg.aug to the model")
        # fixed-mode array_to_tensor works on the flat array too, and TTA offsets differ from 0
        cfg_fixed = Config(cache_scheme="c02", slices_per_slot=6)     # smoke clamps K to 2
        Kf = cfg_fixed.slices_per_slot
        x0, m0 = K["array_to_tensor"](arr, mask, cfg_fixed, False, centre_offset=0)
        x1, _ = K["array_to_tensor"](arr, mask, cfg_fixed, False, centre_offset=1)
        check(x0.shape == (6, Kf, 3, 224, 224) and not torch.equal(x0, x1),
              f"fixed-mode on flat array {tuple(x0.shape)} (K={Kf}), offset view differs")
        probs = K["pool_views"](torch.rand(3, 2, 12), "focal")
        check(probs.shape == (2, 12), "focal pooling shape")

    print("\n== nbgen self-source payload (P-31)")
    sys.path.insert(0, os.path.join(ROOT, "src"))
    import nbgen
    for arms, want in (('PARALLEL_ARMS = ("a", "b")', True), ("PARALLEL_ARMS = ()", False)):
        src_text = (f"# %%\nx = 1\n{arms}\nSELF_SOURCE_SHA256 = None\nSELF_SOURCE_B64 = None  # nbgen: filled at build\n"
                    f"# %%\nprint(x)\n")
        with tempfile.TemporaryDirectory() as td:
            p, q = os.path.join(td, "t.py"), os.path.join(td, "t.ipynb")
            with open(p, "w", encoding="utf-8") as fh:
                fh.write(src_text)
            nbgen.build(p, q)
            with open(q, encoding="utf-8") as fh:
                nb = json.load(fh)
        code = "\n".join("".join(c["source"]) for c in nb["cells"] if c["cell_type"] == "code")
        if want:
            ns = {}
            exec(compile(code.replace("print(x)", ""), "t", "exec"), ns)
            payload = ns.get("SELF_SOURCE_B64")
            decoded = zlib.decompress(base64.b64decode(payload)).decode("utf-8") if isinstance(payload, str) else None
            check(decoded == src_text and ns.get("SELF_SOURCE_SHA256") == hashlib.sha256(src_text.encode("utf-8")).hexdigest()
                  and max(len(ln) for ln in code.splitlines()) < 140,
                  "nbgen embeds the .py (zlib + base64, chunked) with its sha256 when PARALLEL_ARMS is set")
        else:
            check("SELF_SOURCE_B64 = None" in code and "SELF_SOURCE_SHA256 = None" in code,
                  "nbgen leaves the markers as None when PARALLEL_ARMS is empty")

    print("\n" + ("UNIT CHECKS PASSED" if not fails else f"UNIT CHECKS FAILED ({len(fails)}):\n  - " + "\n  - ".join(fails)))
    sys.exit(1 if fails else 0)


if __name__ == "__main__":
    main()
