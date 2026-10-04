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
  * teacher tables (2026-09-23): kernel quantile_match / mix_teacher; yt through both Dataset target
    blocks and collate_windows; build_targets fatal on an unknown or unmounted TEACHER_TABLES name and,
    with selfdistill_v1 mounted, yt__ appended while every plain column stays identical
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
                    ("timm:coatnet_rmlp_1_rw_224", 224), ("timm:coatnet_rmlp_1_rw_224", 320),   # P-43 v09x
                    ("timm:coatnet_rmlp_2_rw_384", 384), ("timm:resnet34", 224),           # P-57 v13a
                    ("timm:resnet50", 224), ("timm:efficientnet_b0", 224),                  # P-64 follow-ups
                    ("timm:efficientnet_b3", 288)):                                        # v13b3 (staged 10-04)
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
        # a ResNet has no final norm above its last stage, so its top group is layer4 at lr_backbone * llrd_decay -- the
        # rate CoAtNet's last stage gets too (only CoAtNet's `norm.*` sits at lr_backbone itself)
        top = cfg.lr_backbone * (cfg.llrd_decay if bb in ("timm:resnet34", "timm:resnet50", "timm:efficientnet_b0",
                                                       "timm:efficientnet_b3") else 1.0)
        check(lrs[-1] == cfg.lr_head and any(abs(l - top) < 1e-12 for l in lrs),
              f"{bb}: LR set {['%.1e' % l for l in lrs]}")
        model.eval()
        with torch.no_grad():
            x = torch.randn(2, 3, img, img)
            f = model.encode(x)
            check(f.shape == (2, model.dim), f"{bb}: encode {tuple(x.shape)} -> {tuple(f.shape)}")
        if bb == "timm:resnet34":
            # P-57: the stem and layer1..4 are decayed per stage, not all lumped into the top LR
            check(len(lrs) >= 6, f"{bb}: {len(lrs)} distinct LRs (stem + 4 layers + head)")

    print("\n== P-63 spatial reader + slot-count correction")
    torch.manual_seed(0)
    sp = K["SpatialFindingPool"](64, 12)
    fm = torch.randn(3, 64, 7, 7)
    v = sp(fm)
    check(v.shape == (3, 12, 64) and torch.allclose(v, fm.mean((2, 3)).unsqueeze(1).expand(-1, 12, -1), atol=1e-6),
          "zero query -> every label's vector is the average pool")
    with torch.no_grad():
        sp.spatial_query.normal_()
    v2 = sp(fm)
    check(torch.isfinite(v2).all() and not torch.allclose(v2[:, 0], v2[:, 1]), "a trained query -> per-label vectors differ")
    hd = K["WindowAttnHead"](64, 12).eval()
    f3 = torch.randn(2, 5, 64)
    sid5 = torch.randint(0, 6, (2, 5))
    val5 = torch.ones(2, 5, dtype=torch.bool)
    val5[1, 3:] = False
    o3 = hd(f3, sid5, val5)
    o4 = hd(f3.unsqueeze(2).expand(-1, -1, 12, -1).contiguous(), sid5, val5)
    check(torch.allclose(o3, o4, atol=1e-5), "head: 12 identical per-label tokens == the (B, W, dim) path")
    o4h = hd(v2[:2, None].expand(-1, 5, -1, -1).half().contiguous(), sid5, val5)
    check(o4h.shape == (2, 12) and torch.isfinite(o4h.float()).all(), "head: (B, W, L, dim) fp16 masked -> finite")
    # count correction: gate output weights zeroed -> equal logits; slot 0 has 4 valid windows (+1 padded), slot 1 has 1
    hn = K["WindowAttnHead"](64, 12).eval()
    hc = K["WindowAttnHead"](64, 12, count_norm=True).eval()
    hc.load_state_dict(hn.state_dict())
    with torch.no_grad():
        for m in (hn, hc):
            m.gate[3].weight.zero_()
    ta, tb = torch.randn(64), torch.randn(64)
    fc = torch.stack([ta] * 4 + [tb] + [torch.randn(64)])[None]
    sc_ = torch.tensor([[0, 0, 0, 0, 1, 0]])
    vc = torch.tensor([[True] * 5 + [False]])
    with torch.no_grad():
        sa, sb = [(hn.norm(t) * hn.w).sum(-1) for t in (ta, tb)]
        on, oc = hn(fc, sc_, vc)[0], hc(fc, sc_, vc)[0]
    check(torch.allclose(on, 0.8 * sa + 0.2 * sb + hn.b, atol=1e-5) and torch.allclose(oc, 0.5 * sa + 0.5 * sb + hc.b, atol=1e-5),
          "slot_count_norm: equal logits give each SLOT equal mass (0.5 / 0.5, padded window not counted); without: 0.8 / 0.2")
    try:
        K["resolve_backbone_dir"]("timm:coatnet_rmlp_1_rw_224")
        have_coat = True
    except SystemExit as e:
        print(f"  skip model-level P-63 checks: {e}")
        have_coat = False
    if have_coat:
        kw = dict(cache_scheme="c02", backbone="timm:coatnet_rmlp_1_rw_224", img_size=224, head_type="window_attn",
                  window_mode="random", lr_backbone=1e-4)
        base = K["build_model"](Config(**kw), device).eval()
        cfg_sp = Config(**kw, spatial_reader=True, slot_count_norm=True)
        spm = K["build_model"](cfg_sp, device).eval()
        res = spm.load_state_dict(base.state_dict(), strict=False)
        check(res.missing_keys == ["spatial_pool.spatial_query"] and not res.unexpected_keys,
              f"spatial model = parent + one new tensor (missing {res.missing_keys}, unexpected {res.unexpected_keys})")
        starts6 = [0, 10, 20, 30, 40, 50]
        arr_s = torch.from_numpy((np.random.RandomState(3).rand(1, 60, 64, 64) * 255).astype(np.uint8))
        cen_s, sid_s = torch.tensor([2, 5, 3, 7, 4, 2]), torch.tensor([0, 0, 1, 2, 3, 5])
        args = (arr_s, cen_s, sid_s, torch.zeros(6, dtype=torch.long), torch.arange(6), starts6)
        spm.window_head.count_norm = False
        with torch.no_grad():
            ob, osp = base.forward_windows(*args), spm.forward_windows(*args)
        check(torch.allclose(ob, osp, atol=1e-4), f"untrained reader == parent model (max |diff| {float((ob - osp).abs().max()):.2e})")
        spm.window_head.count_norm = True
        groups = K["param_groups"](spm, cfg_sp)
        q = spm.spatial_pool.spatial_query
        check(any(any(p is q for p in g["params"]) and g["lr"] == cfg_sp.lr_head for g in groups)
              and sum(p.numel() for g in groups for p in g["params"]) == sum(p.numel() for p in spm.parameters()),
              "param_groups: spatial_query at lr_head, every parameter covered")
        spm.train()
        loss = spm.forward_windows(*args).float().pow(2).mean()
        loss.backward()
        check(q.grad is not None and torch.isfinite(q.grad).all() and float(q.grad.abs().sum()) > 0,
              "train mode: the zero-initialised query receives a finite, non-zero gradient")
        for bad in (dict(kw, backbone="dinov2", spatial_reader=True), dict(kw, head_type="attn", window_mode="fixed",
                                                                             slot_count_norm=True)):
            try:
                K["build_model"](Config(**bad), device)
                check(False, f"P-63 must refuse {bad['backbone']} / {bad['head_type']}")
            except SystemExit:
                check(True, f"P-63 refuses backbone {bad['backbone']} / head {bad['head_type']}")

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

    print("\n== heavy augmentation (P-60)")
    torch.manual_seed(0)
    xa = torch.rand(64, 3, 64, 64) * 0.8 + 0.1
    y = K["augment_heavy"](xa)
    check(y.shape == xa.shape and y.dtype == xa.dtype and float(y.min()) >= 0.0 and float(y.max()) <= 1.0
          and torch.isfinite(y).all() and (y != xa).float().mean() > 0.5,
          "augment_heavy: same shape / dtype, stays in [0, 1], changes most windows")
    check(torch.equal(K["augment_heavy"](xa, p=0.0), xa), "augment_heavy p=0 is the identity")
    check(K["augment_heavy"](xh).dtype == torch.float16, "augment_heavy keeps a half input half")
    torch.manual_seed(1)
    yc = K["augment_heavy"](torch.full((200, 3, 32, 32), 0.5), p=1.0, cutout_p=1.0)
    zero_frac = (yc[:, 0] == 0).float().mean(dim=(1, 2))
    check(bool(((zero_frac >= 0.03) & (zero_frac <= 0.40)).float().mean() > 0.9)
          and bool(torch.equal(yc[:, 0] == 0, yc[:, 1] == 0)),
          f"augment_heavy cutout: each window loses 4-25 % (+ zoom-out padding) to one box, the same box in every "
          f"channel (median {float(zero_frac.median()):.3f})")
    torch.manual_seed(2)
    yn = K["augment_heavy"](torch.full((200, 3, 32, 32), 0.5), p=1.0, cutout_p=0.0)
    check(float((yn[:, :, 8:24, 8:24] == 0).float().mean()) == 0.0,
          "augment_heavy cutout_p=0: no zeroed centre pixels (the warp only pads the border)")
    ok_cfg = K["Config"](aug="heavy", window_mode="random", head_type="window_attn", cache_scheme="c02",
                         backbone="timm:coatnet_rmlp_1_rw_224")
    check(ok_cfg.aug == "heavy", "Config accepts aug='heavy' in window mode")
    try:
        K["Config"](drop_path=0.1, backbone="dinov2")
        check(False, "Config refuses drop_path on a non-timm backbone")
    except SystemExit:
        check(True, "Config refuses drop_path on a non-timm backbone")

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

    print("\n== teacher tables (2026-09-23)")
    # ---- teacher tables (2026-09-23, spec section 2) ------------------------------------------------
    qm = K["quantile_match"]
    pred = np.array([0.9, 0.1, np.nan, 0.5]); ref = np.r_[np.zeros(80), np.ones(20)]
    out = qm(pred, ref)
    check(np.isnan(out[2]) and np.isfinite(out[[0, 1, 3]]).all() and out[0] >= out[3] >= out[1],
          "kernel quantile_match: rank-preserving, NaN passthrough")
    import pandas as pd
    idx = pd.Index([f"s{i}" for i in range(6)])
    soft = pd.DataFrame({l: np.linspace(0.1, 0.9, 6) for l in LABELS}, index=idx)
    table = pd.DataFrame({l: np.linspace(0.9, 0.1, 6) for l in LABELS}, index=idx)     # reversed ranks
    yt = K["mix_teacher"](soft, {"t": table}, 0.5, np.array([True] + [False] * 5))
    check(np.allclose(yt.iloc[0].to_numpy(), soft.iloc[0].to_numpy()), "kernel mix_teacher: gold row keeps the LLM value")
    check(not np.allclose(yt.iloc[1:].to_numpy(), soft.iloc[1:].to_numpy()), "kernel mix_teacher: covered rows move")
    # collate carries yt when present, and the loss reads it
    items = [{"study": "a", "arr": torch.zeros(2, 4, 4, dtype=torch.uint8), "centres": torch.zeros(3, dtype=torch.long),
              "slot_id": torch.zeros(3, dtype=torch.long), "mask": torch.ones(6, dtype=torch.bool),
              "y": torch.full((12,), 0.2), "yt": torch.full((12,), 0.8), "w": torch.ones(12), "is_gold": torch.tensor(0.)}]
    b = K["collate_windows"](items)
    check("yt" in b and b["yt"].shape == (1, 12), "collate_windows: stacks yt")
    logits = torch.zeros(1, 12)
    l_y = K["weighted_bce"](logits, b["y"], b["w"]); l_yt = K["weighted_bce"](logits, b["yt"], b["w"])
    check(abs(float(l_y) - float(l_yt)) < 1e-6, "weighted_bce at logit 0 is symmetric in the target (sanity)")
    check(K["TEACHER_TABLES"] == () and K["Config"]().teacher_tables == (), "TEACHER_TABLES default () (no teacher mixing unless sed'd)")
    check("selfdistill_v1" in K["TEACHER_PATHS"] and "raptor_teacher" in K["TEACHER_PATHS"], "TEACHER_PATHS lists both tables")
    # Drift guard: the kernel is one file, so quantile_match / mix_teacher are COPIES of src/build_targets.py's.
    # A fix applied to one copy would pass both suites while the training and reference targets diverge.
    import ast
    import inspect
    import textwrap
    import build_targets as BT

    def fn_ast(fn):
        return ast.dump(ast.parse(textwrap.dedent(inspect.getsource(fn))))

    for fname in ("quantile_match", "mix_teacher", "silence_mask"):
        check(fn_ast(K[fname]) == fn_ast(getattr(BT, fname)),
              f"kernel {fname} is AST-identical (docstring included) to src/build_targets.py's")

    # Dataset: both target blocks ship `yt` only when the targets frame carries yt__ columns; `y` is untouched
    tgt = pd.DataFrame({"StudyInstanceUID": ["a"], "is_gold": [0], **{l: [0.2] for l in LABELS},
                        **{f"w__{l}": [1.0] for l in LABELS}})
    tgt_t = tgt.assign(**{f"yt__{l}": [0.8] for l in LABELS})
    y02, y08 = torch.full((12,), 0.2), torch.full((12,), 0.8)
    man = pd.DataFrame({"StudyInstanceUID": ["a"], **{s: [""] for s in SLOTS}})
    cfg_fx = Config(use_cache=False)                     # fixed path, no series on disk -> zero slots
    it0 = K["KneeStudyDataset"](man, tgt, "nowhere", cfg_fx, False)[0]
    it1 = K["KneeStudyDataset"](man, tgt_t, "nowhere", cfg_fx, False)[0]
    check("yt" not in it0 and torch.allclose(it1["yt"], y08) and torch.allclose(it1["y"], y02)
          and torch.allclose(it0["y"], y02), "Dataset (fixed path): yt from yt__ columns only when present, y unchanged")
    blob_dir = os.path.join("artifacts", "cache_local", K["cache_version_for"](c02))
    blob_csv = [f for f in sorted(os.listdir(blob_dir)) if f.endswith(".csv")] if os.path.isdir(blob_dir) else []
    if not blob_csv:
        check(False, f"no local c02 blob under {blob_dir} (Dataset window-path yt check)")
    else:
        rw = pd.read_csv(os.path.join(blob_dir, blob_csv[0]), dtype={"mask": str}).query("cached == 1").iloc[0]
        man_w = pd.DataFrame({"StudyInstanceUID": ["a"], "mask": [rw["mask"]], "side": ["L"]})
        cfg_w = Config(cache_scheme="c02", window_mode="random", head_type="window_attn")
        ver = K["cache_version_for"](cfg_w)
        saved_index = K["CACHE_INDEX"].get(ver)
        K["CACHE_INDEX"][ver] = {"a": (os.path.join(blob_dir, rw.blob), int(rw.row))}
        try:
            iw0 = K["KneeStudyDataset"](man_w, tgt, "nowhere", cfg_w, True)[0]
            iw1 = K["KneeStudyDataset"](man_w, tgt_t, "nowhere", cfg_w, True)[0]
        finally:
            if saved_index is None:
                K["CACHE_INDEX"].pop(ver, None)
            else:
                K["CACHE_INDEX"][ver] = saved_index
        bw = K["collate_windows"]([iw1, iw1])
        check("arr" in iw1 and "yt" not in iw0 and torch.allclose(iw1["yt"], y08) and torch.allclose(iw1["y"], y02)
              and "yt" not in K["collate_windows"]([iw0]) and tuple(bw["yt"].shape) == (2, 12),
              "Dataset (window path): yt from yt__ columns only when present; collate_windows stacks it")

    # build_targets: a listed table that is unknown or not mounted is FATAL; a mounted one appends yt__ only
    import contextlib
    import io
    train_csv = os.path.join(K["COMP"], "train.csv")
    plain = K["targets"]
    check(not any(c.startswith("yt__") for c in plain.columns), "plain targets (TEACHER_TABLES = ()) carry no yt__ columns")
    paths0 = dict(K["TEACHER_PATHS"])
    ghost = os.path.join(tempfile.gettempdir(), "rsna_no_such_teacher_table.csv")
    for names, want, label in ((("no_such_table",), "unknown teacher table", "unknown teacher table is fatal"),
                               (("ghost",), "not mounted", "unmounted teacher table is fatal")):
        K["TEACHER_TABLES"] = names
        if names == ("ghost",):
            K["TEACHER_PATHS"]["ghost"] = [ghost]         # a known name whose CSV is not on disk
        msg = None
        try:
            with contextlib.redirect_stdout(io.StringIO()):
                K["build_targets"](train_csv)
        except SystemExit as e:
            msg = str(e)
        finally:
            K["TEACHER_TABLES"] = ()
            K["TEACHER_PATHS"].clear()
            K["TEACHER_PATHS"].update(paths0)
        check(msg is not None and want in msg, f"{label} (SystemExit: {msg})")
    sd_path = K["first_existing"](K["TEACHER_PATHS"]["selfdistill_v1"])
    if sd_path is None:
        print("  skip teacher build_targets end-to-end: selfdistill_v1.csv not present (src/build_distill_table.py)")
    else:
        K["TEACHER_TABLES"] = ("selfdistill_v1",)
        buf = io.StringIO()
        try:
            with contextlib.redirect_stdout(buf):
                mixed = K["build_targets"](train_csv)
        finally:
            K["TEACHER_TABLES"] = ()
        ytc = [f"yt__{l}" for l in LABELS]
        g = (mixed.is_gold == 1).to_numpy()
        ytv = mixed[ytc].to_numpy()
        check(list(mixed.columns) == list(plain.columns) + ytc and mixed[list(plain.columns)].equals(plain),
              "teacher build_targets: every plain column (y, w__, is_gold, fold) identical, yt__ appended")
        check(np.isfinite(ytv).all() and ytv.min() >= 0 and ytv.max() <= 1, "teacher build_targets: yt__ finite, in [0, 1]")
        check(np.array_equal(ytv[g], mixed.loc[g, LABELS].to_numpy()) and set(np.unique(ytv[g])) <= {0.0, 1.0},
              f"teacher build_targets: the {int(g.sum())} gold rows keep hard 0/1 in yt__")
        moved = float((ytv[~g] != mixed.loc[~g, LABELS].to_numpy()).mean())
        check(moved > 0.5, f"teacher build_targets: {moved:.0%} of report-only yt__ cells differ from y")
        check("teacher table selfdistill_v1:" in buf.getvalue() and "training targets = (1 - 0.5)" in buf.getvalue(),
              "teacher build_targets logs the table and the mix")

    # ---- pos_weight (P-37) --------------------------------------------------------------------------
    lpw = K["label_pos_weight"]
    tg = pd.DataFrame({"StudyInstanceUID": [f"s{i}" for i in range(10)]})
    for l in LABELS:
        tg[l] = 0.0
    tg.loc[:0, "ACL"] = 1.0            # 10 % positive -> (1-p)/p = 9
    tg["MCL"] = 1.0                    # all positive -> clip to 1
    tg["Fracture"] = 0.0               # no positive -> clip to max
    pw = lpw(tg, tg.StudyInstanceUID.tolist(), 10.0)
    check(pw.shape == (12,) and np.isfinite(pw).all(), "label_pos_weight: 12 finite weights")
    check(abs(pw[LABELS.index("ACL")] - 9.0) < 1e-6, "label_pos_weight: 10 % positive -> 9")
    check(pw[LABELS.index("MCL")] == 1.0 and pw[LABELS.index("Fracture")] == 10.0, "pos_weight extremes clip to [1, max]")
    try:
        lpw(tg, [], 10.0)
        empty_raised = False
    except SystemExit:
        empty_raised = True
    check(empty_raised, "label_pos_weight: empty study list is fatal, never NaN")
    lg = torch.randn(2, 12); yy = torch.rand(2, 12); ww = torch.ones(2, 12)
    check(torch.allclose(K["weighted_bce"](lg, yy, ww), K["weighted_bce"](lg, yy, ww, pos_weight=None)), "weighted_bce: pos_weight=None is the old loss")
    check(float(K["weighted_bce"](lg, yy, ww, pos_weight=torch.full((12,), 3.0))) > float(K["weighted_bce"](lg, yy, ww)), "weighted_bce: pos_weight > 1 raises the loss")
    check(K["Config"]().pos_weight_max == 0.0, "Config.pos_weight_max defaults to 0 (off)")

    # ---- 2026-09-28 arms (P-54 cross-fit, P-56 c03, P-55 student, P-57 ResNet) ------------------------
    print("\n== 2026-09-28 arms")
    arms = dict(K["SHIPPED_ARMS"])
    Config = K["Config"]
    for k in range(5):
        c = Config(smoke=False, version=f"v09k{k}", **arms[f"v09k{k}"])
        check(c.folds == (k,) and not c.train_all and c.eval_final_only and c.swa_last == 3 and c.ckpt_policy == "last"
              and c.backbone == "timm:coatnet_rmlp_1_rw_224" and c.batch_studies == 2 and c.aug == "light",
              f"v09k{k}: fold {c.folds}, train_all {c.train_all}, eval_final_only, SWA 3, the v09r model keys")
    c_sm = Config(smoke=True, version="v09k3", **arms["v09k3"])
    check(c_sm.folds == (0,), "a SMOKE cross-fit arm is forced to fold 0 (so folds 1-4 are only checked here, not by a smoke)")
    try:
        Config(smoke=False, eval_final_only=True, ckpt_policy="best_oof")
        efo_raised = False
    except SystemExit:
        efo_raised = True
    check(efo_raised, "eval_final_only with ckpt_policy='best_oof' is refused")
    tg = pd.DataFrame({"StudyInstanceUID": [f"u{i}" for i in range(50)], "fold": [i % 5 for i in range(50)],
                       "is_gold": [1 if i % 7 == 0 else 0 for i in range(50)]})
    ti = tg.set_index("StudyInstanceUID")
    for k in range(5):
        tr, va = K["split_studies"](tg, k, Config(smoke=False, **arms[f"v09k{k}"]))
        ftr, fva = set(ti.loc[tr, "fold"]), set(ti.loc[va, "fold"])
        gold_va = int(ti.loc[va, "is_gold"].sum())
        check(k not in ftr and fva == {k} and not (set(tr) & set(va)) and len(tr) + len(va) == 50 and gold_va > 0,
              f"split fold {k}: train folds {sorted(ftr)}, val = fold {k} incl. {gold_va} gold rows, disjoint")
    c3 = Config(smoke=False, version="v11a", **arms["v11a"])
    check(K["cache_version_for"](c3) == "c02_p336_b24-24-24-14-8-8_band2-98_crop150_lat20" and c3.train_windows == 34,
          f"v11a reads c03 = {K['cache_version_for'](c3)}, train_windows {c3.train_windows}")
    check(Config(smoke=False, **arms["v11b"]).seed == 43 and Config(smoke=False, **arms["v09o2"]).seed == 43,
          "v11b / v09o2 are the seed-43 twins")
    c13 = Config(smoke=False, **arms["v13a"])
    check(c13.backbone == "timm:resnet34" and c13.train_all, "v13a = the production recipe on timm:resnet34")
    dm, da = K["DISTILLED_MIX"], K["DISTILLED_ARMS"]
    check(da["v09o"] == ("raptor_teacher", "xfit_v09k") and dm.get("v09o") == 0.75 and dm.get("v09o2") == 0.75
          and all(da[a] == ("raptor_teacher",) and a not in dm for a in ("v09k0", "v09k4", "v11a", "v13a")),
          "DISTILLED_ARMS / DISTILLED_MIX: the student trains on Raptor + xfit at 0.75, the rest on Raptor at 0.5")
    check("xfit_v09k" in K["TEACHER_PATHS"] and K["PROBE_CONST_LABELS"] == (),
          "TEACHER_PATHS has xfit_v09k; PROBE_CONST_LABELS off by default")

    print("\n== P-59 / P-60 arms (2026-09-29)")
    cb, cc = Config(smoke=False, **arms["v13b"]), Config(smoke=False, **arms["v13c"])
    check(cb.backbone == cc.backbone == "timm:resnet34" and cb.lr_backbone == cc.lr_backbone == 3e-4
          and cb.llrd_decay == cc.llrd_decay == 1.0 and cb.epochs == cc.epochs == 12 and not cb.freeze_bn and cc.freeze_bn
          and K["cache_version_for"](cb) == K["cache_version_for"](Config(smoke=False, **arms["v13a"])),
          "v13b = v13a + uniform 3e-4 x 12 epochs on v13a's c02 input; v13c = v13b + freeze_bn")
    cn, cn2 = Config(smoke=False, **arms["v11n"]), Config(smoke=False, **arms["v11n2"])
    check(cn.drop_path == cn2.drop_path == 0.1 and cn.aug == cn2.aug == "heavy" and cn.epochs == 12 and cn.swa_last == 3
          and cn2.seed == 43 and cn.seed == 42 and K["cache_version_for"](cn) == K["cache_version_for"](c3)
          and cn.train_windows == 34 and cn.lr_backbone == 1e-4 and cn.llrd_decay == 0.75,
          "v11n / v11n2 = v11a (c03) + drop_path 0.1 + aug heavy + 12 epochs, seeds 42 / 43")
    check(all(da[a] == ("raptor_teacher",) and a not in dm for a in ("v13b", "v13c", "v11n", "v11n2")),
          "the four new arms train on Raptor at mix 0.5")
    for bb in ("timm:resnet34", "timm:coatnet_rmlp_1_rw_224"):
        try:
            bdir = K["resolve_backbone_dir"](bb)
        except SystemExit as e:
            print(f"  skip drop_path check for {bb}: {e}")
            continue
        enc0 = K["load_timm_backbone"](bb.split(":", 1)[1], bdir, img_size=224)
        enc1 = K["load_timm_backbone"](bb.split(":", 1)[1], bdir, img_size=224, drop_path=0.1)
        dp0 = [float(getattr(m, "drop_prob", 0.0)) for m in enc0.modules() if type(m).__name__ == "DropPath"]
        dp1 = [float(getattr(m, "drop_prob", 0.0)) for m in enc1.modules() if type(m).__name__ == "DropPath"]
        check(max(dp0, default=0.0) == 0.0 and dp1 and 0.0 < max(dp1) <= 0.1 + 1e-6,
              f"{bb}: drop_path 0 -> no active DropPath; 0.1 -> {len(dp1)} DropPath modules, max rate {max(dp1, default=0):.3f}")
        enc1.eval()
        with torch.no_grad():
            xx = torch.randn(2, 3, 224, 224)
            check(torch.allclose(enc1(xx), enc1(xx)), f"{bb}: drop_path is inert in eval mode (deterministic)")

    print("\n== P-45 / P-63 arms (2026-10-03)")
    c11a = Config(smoke=False, **arms["v11a"])
    for a, extra in (("v11d", {}), ("v11d2", {"seed": 43}), ("v11p", {"spatial_reader": True, "slot_count_norm": True}),
                     ("v11p2", {"spatial_reader": True, "slot_count_norm": True, "seed": 43})):
        ca = Config(smoke=False, **arms[a])
        diff = {k for k in K["asdict"](ca) if getattr(ca, k) != getattr(c11a, k)} - {"version"}
        check(diff == set(extra) and all(getattr(ca, k) == v for k, v in extra.items()),
              f"{a} = v11a + {extra or 'nothing'} (fields that differ: {sorted(diff)})")
    for a in ("v11nd", "v11nd2"):
        cnd, cn_ = Config(smoke=False, **arms[a]), Config(smoke=False, **arms[a.replace("nd", "n")])
        diff = {k for k in K["asdict"](cnd) if getattr(cnd, k) != getattr(cn_, k)} - {"version"}
        check(not diff, f"{a} = {a.replace('nd', 'n')} exactly (only the table set differs; fields that differ: {sorted(diff)})")
    check(all(da[a] == ("raptor_teacher", "d4_teacher") and a not in dm for a in ("v11d", "v11d2", "v11nd", "v11nd2"))
          and all(da[a] == ("raptor_teacher",) for a in ("v11p", "v11p2")) and "d4_teacher" in K["TEACHER_PATHS"],
          "v11d* / v11nd* train on Raptor + D4 at mix 0.5 (0.5 LLM + 0.25 + 0.25); v11p* on Raptor alone")
    cdl, cd_ = Config(smoke=False, **arms["v11dl"]), Config(smoke=False, **arms["v11d"])
    diff = {k for k in K["asdict"](cdl) if getattr(cdl, k) != getattr(cd_, k)} - {"version"}
    check(diff == {"lr_backbone", "llrd_decay"} and cdl.lr_backbone == 2e-4 and cdl.llrd_decay == 0.85
          and da["v11dl"] == da["v11d"], f"v11dl = v11d + lr_backbone 2e-4 / llrd_decay 0.85 on the same tables ({sorted(diff)})")
    ch, cc_ = Config(smoke=False, **arms["v13h"]), Config(smoke=False, **arms["v13c"])
    diff = {k for k in K["asdict"](ch) if getattr(ch, k) != getattr(cc_, k)} - {"version"}
    check(diff == {"cache_slot_slices", "crop_mm", "train_windows", "aug", "drop_path", "epochs"}
          and ch.backbone == "timm:resnet34" and ch.freeze_bn and ch.lr_backbone == 3e-4 and ch.llrd_decay == 1.0
          and ch.aug == "heavy" and ch.drop_path == 0.1 and ch.epochs == 30 and ch.swa_last == 3
          and K["cache_version_for"](ch) == K["cache_version_for"](c11a) and da["v13h"] == ("raptor_teacher",),
          f"v13h = v13c + c03 input + heavy aug + drop_path 0.1 + 30 epochs ({sorted(diff)})")
    for a, bb in (("v13r", "timm:resnet50"), ("v13e", "timm:efficientnet_b0")):
        cx = Config(smoke=False, **arms[a])
        diff = {k for k in K["asdict"](cx) if getattr(cx, k) != getattr(ch, k)} - {"version"}
        check(diff == {"backbone"} and cx.backbone == bb and da[a] == ("raptor_teacher",),
              f"{a} = v13h with backbone {bb} ({sorted(diff)})")
    # 2026-10-04: P-62 / P-65 arms are their parents unchanged (the session seds carry the difference); v13b3 = B3 @ 288
    for a, par, tabs in (("v13es", "v13e", ("raptor_teacher",)), ("v13rs", "v13r", ("raptor_teacher",)),
                         ("v13ec", "v13e", ("claude_rap_v1",)), ("v13rc", "v13r", ("claude_rap_v1",))):
        cx, cp = Config(smoke=False, **arms[a]), Config(smoke=False, **arms[par])
        diff = {k for k in K["asdict"](cx) if getattr(cx, k) != getattr(cp, k)} - {"version"}
        check(not diff and da[a] == tabs, f"{a} = {par} (recipe identical; tables {tabs}) ({sorted(diff)})")
    cx = Config(smoke=False, **arms["v13b3"])
    diff = {k for k in K["asdict"](cx) if getattr(cx, k) != getattr(ch, k)} - {"version"}
    check(diff == {"backbone", "img_size"} and cx.img_size == 288 and da["v13b3"] == ("raptor_teacher",),
          f"v13b3 = v13h with backbone efficientnet_b3 @ 288 ({sorted(diff)})")

    print("\n" + ("UNIT CHECKS PASSED" if not fails else f"UNIT CHECKS FAILED ({len(fails)}):\n  - " + "\n  - ".join(fails)))
    sys.exit(1 if fails else 0)


if __name__ == "__main__":
    main()
