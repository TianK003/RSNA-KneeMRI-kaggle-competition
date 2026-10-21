"""Build kaggle/rsna-knee-anchor<tag>/: a public anchor notebook submitted as it is, with none of our models in it
(candidates.md C13; Tian 10-17: "send the C13 today to see how we fare in comparison to others" and "also run the new
heliosli notebook").

Every fork of ours on heliosli's 0.958 notebook read 0.957, under the anchor itself (C10w #94 at beta 0.20, C10w-b05 #95 at
beta 0.05; experiments.md 2026-10-17 "Submission #95"). A final pick must be a submitted entry, so the anchor alone needs its
own submission to be selectable. The notebook's cells are copied byte for byte, apart from the patches an anchor declares;
the metadata mounts what the author's own kernel mounts and pins the author's Docker image.

    0958   heliosli `blend-mine-in-private-lb-luck-to-all-upvoters` (public LB 0.958), verbatim       -> kaggle/rsna-knee-anchor958/
    09586  heliosli `this-is-best-public-dataset-new-version-0-9586` (10-17; 0.958 by the LB): the 0.957 notebook + the 0.958's
           four per-finding weights each pushed 0.05 further + goodpjw2008's 2.5D reader at <= 0.035.  ONE patched line:
           C_SLUG -> the -w1 copy of R6, mounted instead of the bare slug, whose latest version holds R7 (traps 72; the same
           patch C10 / build_fork954.py --anchor 0957 carries)                                     -> kaggle/rsna-knee-anchor9586/

    09586s6f  C14 (Tian 10-18: C13-V2 becomes pick 1, "boost its score"): the 09586 copy above, every cell unchanged, plus nartaa's
           s6fall (0.950 public alone, S6 architecture on JEV labels; C11's patched stage, component A not run) inside V2's R6
           half: ours = 0.6 rank(R6) + 0.4 rank(s6fall) per finding, then V2's own stage-3 and Part B formulas re-applied. Before
           s6fall starts, the run cell rebuilds V2's own final file from the stage-3 / Part B inputs in memory and keeps V2's file
           unless that matches (SELF_CHECK); s6fall starts only before 6.0 h of notebook time and is killed at 8.4 h; on any skip,
           timeout or failure the submission is V2's own file (#100)                         -> kaggle/rsna-knee-anchor9586s6f/
    09586s6fr7 C20 (Tian 10-20: "R7 as a third R6-half model"): C14 + heliosli's R7 run through V2's own cells 6-7 (a patched copy in
           its own namespace and directories, after the s6fall stage), then ours = 0.36 rank(R6) + 0.24 rank(R7) + 0.40 rank(s6fall);
           SELF_CHECK rebuilds C14's file first; R7 starts before 6.3 h only; any failure = C14's file -> kaggle/rsna-knee-anchor9586s6fr7/
    09586s6fr7w C20 on C19 (Tian 10-21): the same with cell 8's four Raptor-half weights at the 0.958 notebook's (C19, #106), so the
           SELF_CHECK and every fallback are C19's file                                       -> kaggle/rsna-knee-anchor9586s6fr7w/
    09586s6fwvj / 09586s6fr7wvj  C21m (Tian 10-21, if the V-JEPA solo reads >= 0.945): C19 / C20 on C19 + nurikw3's V-JEPA 2.1 as a
           third component of the final file at a fixed 15 % (VJX_CFG), start-gated; any failure = the base's file
                                                                                  -> kaggle/rsna-knee-anchor9586s6fwvj/ (-s6fr7wvj/)
    09586s6fx C17 (Tian 10-18: a second C14-family ticket): C14 + the 2026-10-18 audit's fixes (traps 74: s6fall under its own
           scratch root so V2's submission.csv is never absent; its hang guard = the notebook's time left, a deadline keeps the scored
           studies, up to 50 % may fail; exact-duplicate Raptor rows count as failed; a rank-corr floor R6 vs s6fall) + s6fall's
           training-matched crop TTA (C17_CROPS5, probability mean)                          -> kaggle/rsna-knee-anchor9586s6fx/

    export PYTHONUTF8=1
    .venv/Scripts/python.exe src/build_anchor_copy.py --anchor 0958
    .venv/Scripts/python.exe src/build_anchor_copy.py --anchor 09586 [--check]
    .venv/Scripts/python.exe src/build_anchor_copy.py --anchor 09586s6f [--check]
"""
import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from build_fork import cell_text, dump_json, load_notebook, sha256_file  # noqa: E402
from build_fork954 import (ANCHORS as F954_ANCHORS, DOCKER_IMAGE, S6FALL_DATASET, check_no_competition_ids,  # noqa: E402
                           module_level_names, patch_s6fall)

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
COMPETITION = "rsna-knee-abnormality-detection"
_R6 = "rsna-knee-r6-frozen-fe055406"

ANCHORS = {
    "0958": {
        "notebook": "notebook_score_0.958.ipynb",
        "sha256": "ad0b97aca3361f16e32b0836655dd68bc2a37ff0cc5a44f6d6f4de07880d986d",   # = heliosli's version on 10-17 (re-pulled)
        "slug": "rsna-knee-anchor958",
        "title": "RSNA Knee Anchor958",
        "dataset_sources": ["heliosli/" + _R6 + "-w1", "nartaa/rsna-knee-publication-swa-weights-20261007",
                            "heliosli/rsna-orthofoundation-fold0-epoch7-20261004"],   # = the author's kernel-metadata.json
        "kernel_sources": [],
        "needles": {6: ("C_SLUG = '%s-w1'" % _R6, "BUNDLE_ID = 'fe055406fad6a12f'"),
                    8: ("W_RAPTOR_C = {'Medial OA': 0.5, 'Lateral OA': 0.2, 'PF OA': 0.6, 'Synovitis': 0.8, \"Baker's\": 0.35}",)},
        "patches": {},
    },
    "09586": {
        "notebook": "notebook_score_0.9586.ipynb",
        "sha256": "cb0e1eac33a00f96b7d1bcc0e9914f7eb045af2bc9cd01bc290b2213e7f17ddd",   # as pulled 2026-10-17 (run 04:21 UTC)
        "slug": "rsna-knee-anchor9586",
        "title": "RSNA Knee Anchor9586",
        # the author's list, with the bare R6 slug (latest version = R7, traps 72) replaced by its byte-identical -w1 copy
        "dataset_sources": ["goodpjw2008/rsna-knee-2-5d-convnext-reader", "heliosli/" + _R6 + "-w1",
                            "nartaa/rsna-knee-publication-swa-weights-20261007",
                            "heliosli/rsna-orthofoundation-fold0-epoch7-20261004"],
        "kernel_sources": ["goodpjw2008/rsna-knee-coatnet-r6-2-5d-mil-lb-0-958"],
        "needles": {6: ("C_SLUG = '%s'   # [d957]" % _R6, "BUNDLE_ID = 'fe055406fad6a12f'", "FLIP_TTA = True"),
                    8: ("W_RAPTOR_C = {'Medial OA': 0.5, 'Lateral OA': 0.15, 'PF OA': 0.65, 'Synovitis': 0.85, \"Baker's\": 0.30}",),
                    9: ("goodpjw2008's runner cell, unchanged", "OWN_W_PER_TARGET = {")},
        "patches": {6: [("C_SLUG = '%s'   # [d957]" % _R6,
                         "C_SLUG = '%s-w1'   # [anchor9586: the -w1 copy = v1 = R6; this slug's latest version holds R7, traps 72] [d957]"
                         % _R6)]},
    },
}
ANCHORS["09586s6f"] = dict(ANCHORS["09586"], slug="rsna-knee-anchor9586s6f", title="RSNA Knee Anchor9586s6f",
                           dataset_sources=ANCHORS["09586"]["dataset_sources"] + [S6FALL_DATASET],
                           s6fall={"share": 0.4, "latest_start_h": 6.0, "end_h": 8.4})

# C15 (10-18): the 384 px stage's anatomical-mirror TTA (cell 2: _mirror_tri + infer_pair_probs, probability mean of plain + mirror)
# ported to the 224 px stage (cell 4), which runs one view. nartaa's RUNS.md: the 224 run "differs only in --arch, --res, --tag and
# RAPTOR_CROP_FRAC=0.8" from the 384 run, whose 0.949 was scored with this mirror; both trained with NO flips and no laterality
# normalisation, on the same 96-slice slot layout (asserted below against the cell's own SLOTS).
_MIRROR224_HELPERS = '''SLOT_SIZES = [26, 22, 18, 12, 18]   # [C15] cell 2's mirror TTA, verbatim, for the 224 stage
SAGITTAL_SLOTS = (0, 1)
assert SLOT_SIZES == [s_[2] for s_ in SLOTS] and SAGITTAL_SLOTS == tuple(i_ for i_, s_ in enumerate(SLOTS) if s_[0] == "Sagittal")

def _slot_index(D):
    b = np.cumsum([0] + SLOT_SIZES)
    if b[-1] != D: raise RuntimeError(f'slot sizes sum to {b[-1]}, volume has {D} slices')
    s_ = np.searchsorted(b, np.arange(D), side='right') - 1
    rel = (np.arange(D) - b[s_]) / np.maximum(1, np.array(SLOT_SIZES)[s_] - 1)
    return s_, rel

def _mirror_tri(tri, cs, D):
    """final-g1-bundle MIRROR_TTA: true L/R mirror of (k,3,H,W) uint8 eval triplets (research/rnd4/mirror/mirror_f0.py). The slot of each
    window's CENTRE slice decides the axis: sagittal (slots 0,1) -> reverse the 3 slice-channels; coronal/axial (2,3,4) -> flip the width.
    Done on uint8 BEFORE the per-channel LUT, i.e. 'mirror the knee, then normalise'."""
    s_, _ = _slot_index(D)
    sag = torch.from_numpy(np.isin(s_[np.asarray(cs, dtype=np.int64)], SAGITTAL_SLOTS)).to(tri.device).view(-1, 1, 1, 1)
    return torch.where(sag, tri.flip(1), tri.flip(-1))

'''
MIRROR224_PATCHES = [
    ("_WLUT = {}\ndef gpu_windows(vol, mask, k, res, dev, norm=NORM):\n",
     _MIRROR224_HELPERS + "_WLUT = {}\ndef gpu_windows(vol, mask, k, res, dev, norm=NORM, mirror=False):   # [C15] mirror\n"),
    ("    tri = torch.from_numpy(np.ascontiguousarray(vol[np.stack([cs - 1, cs, cs + 1], 1)])).to(dev)   # (k,3,IMG,IMG) uint8\n",
     "    tri = torch.from_numpy(np.ascontiguousarray(vol[np.stack([cs - 1, cs, cs + 1], 1)])).to(dev)   # (k,3,IMG,IMG) uint8\n"
     "    if mirror:   # [C15] as cell 2: mirror the uint8 knee before the LUT, crop and resize\n"
     "        tri = _mirror_tri(tri, cs, D)\n"),
    ("                        arm_probs[a][i] = infer_probs(model, xw, d)\n",
     "                        xm = gpu_windows(vol, mask, K_EVAL, res, d, norm=NORM, mirror=True)   # [C15] cell 2's infer_pair_probs:\n"
     "                        arm_probs[a][i] = 0.5 * (infer_probs(model, xw, d) + infer_probs(model, xm, d))   # plain + mirror\n"
     "                        del xm\n"),
]
for _tag, _base in (("09586m224", "09586"), ("09586s6fm224", "09586s6f")):
    ANCHORS[_tag] = dict(ANCHORS[_base], slug=ANCHORS[_base]["slug"].replace("9586", "9586m") if _tag == "09586m224"
                         else "rsna-knee-anchor9586s6fm",
                         title="RSNA Knee Anchor9586m" if _tag == "09586m224" else "RSNA Knee Anchor9586s6fm",   # title -> slug
                         patches={**ANCHORS["09586"]["patches"], 4: MIRROR224_PATCHES}, mirror224=True)

S6X_INTRO = """# Anchor9586s6f: nartaa's s6fall inside V2's R6 half (candidates.md C14)

Built by `src/build_anchor_copy.py --anchor __TAG__`. Every cell above is heliosli's V2 notebook as `rsna-knee-anchor9586` runs it
(#100)__MIRROR__. The next cell writes nartaa's public s6fall script (heliosli's S6 architecture retrained on JEV labels; component A, his
OAI-trained CoAtNet, not run; the public test ids removed; as C11 / C12 run it). The cell after it keeps V2's final file aside,
rebuilds that file from V2's stage-3 and Part B inputs (and keeps V2's file unless the rebuild matches it), then runs s6fall as its
own process under a time limit and re-blends: ours = __KEEP__ rank(R6) + __SHARE__ rank(s6fall) per finding, then V2's stage-3
weights and Part B weights unchanged. On a skip, a timeout or a failure the submission is V2's own file.
Credits: heliosli (V2, R6, S6 architecture, inference code, SKM-TEA anatomy model), nartaa (CoAtNets; s6fall weights, CC BY-NC 4.0),
goodpjw2008 (2.5D reader).
"""

S6X_RUN_CELL = r"""# Anchor9586s6f: stage S -- nartaa's s6fall inside V2's R6 half (added by src/build_anchor_copy.py --anchor 09586s6f).
# Reads V2's stage-3 globals (raw, raw224, ours, r384_fail, r224_fail, rap_fail, our_fail, labels, cols, test_ids, W_RAPTOR_C,
# W_RAPTOR, A224, pct_on) and Part B's files; binds only _s6x_* names. Any skip / timeout / failure: V2's own file is submitted.
import json as _s6x_json, os as _s6x_os, shutil as _s6x_sh, signal as _s6x_sig, subprocess as _s6x_sp, sys as _s6x_sys
import time as _s6x_time
import numpy as _s6x_np
import pandas as _s6x_pd
_S6X_SHARE = __SHARE__
_S6X_LATEST_START_H = __LATEST__
_S6X_END_H = __END__
_s6x_sub = '/kaggle/working/submission.csv'
_s6x_v2_path = '/kaggle/working/_v2_final.csv'
_s6x_own_path = '/kaggle/working/_own.csv'
_s6x_sh.copyfile(_s6x_sub, _s6x_v2_path)                   # s6fall deletes submission.csv when it starts
_s6x_info = {'share': _S6X_SHARE, 'rows': len(test_ids)}
_s6x_status = 'not_run'


def _s6x_log(msg):
    print('[s6x %5.2fh] %s' % ((_s6x_time.time() - NB_T0) / 3600, msg), flush=True)


def _s6x_stage3(ro_by_c, o_fail):
    # V2's cell-8 loop, verbatim but for the ours rank (ro_by_c) and its failure mask (o_fail)
    out = _s6x_np.zeros((len(test_ids), len(labels)))
    for j, c in enumerate(labels):
        r384, r224 = pct_on(raw[c].values, ~r384_fail), pct_on(raw224[c].values, ~r224_fail)
        rr = _s6x_np.where(r384_fail, r224, _s6x_np.where(r224_fail, r384, (1.0 - A224) * r384 + A224 * r224))
        ro = ro_by_c[c]
        wr = W_RAPTOR_C.get(c, W_RAPTOR)
        b = wr * rr + (1.0 - wr) * ro
        b[rap_fail] = ro[rap_fail]
        b[o_fail] = rr[o_fail]
        b[rap_fail & o_fail] = 0.5
        out[:, j] = b
    sub_ = _s6x_pd.DataFrame(out.astype(_s6x_np.float32), columns=labels)   # V2 writes stage 3 as float32
    sub_.insert(0, 'StudyInstanceUID', test_ids)
    return sub_[cols]


def _s6x_part_b(stack, own):
    # goodpjw2008's runner, mode "end" with per-finding weights, verbatim
    own = own.set_index('StudyInstanceUID').loc[stack.StudyInstanceUID].reset_index()
    lab = [c for c in stack.columns if c != 'StudyInstanceUID']
    rs, ro = stack[lab].rank(method='average', pct=True), own[lab].rank(method='average', pct=True)
    fin = stack.copy()
    for c in lab:
        w = float(OWN_W_PER_TARGET.get(c, OWN_W)) if OWN_W_PER_TARGET else OWN_W
        fin[c] = ((1 - w) * rs[c] + w * ro[c]).to_numpy()
    fin[lab] = fin[lab].rank(method='average', pct=True)
    return fin


def _s6x_kill(p):
    for _s6x_k in (lambda: _s6x_os.killpg(p.pid, _s6x_sig.SIGKILL), lambda: p.kill(), lambda: p.wait(timeout=60),
                   lambda: _s6x_sp.run(['pkill', '-9', '-f', '/tmp/k1'], check=False, timeout=60)):  # its own-session workers
        try:
            _s6x_k()
        except Exception as _s6x_e:
            _s6x_log('kill step: %s' % type(_s6x_e).__name__)


# 1. SELF_CHECK: rebuild V2's final file (#100) from the inputs in memory; s6fall runs only if the rebuild matches it
try:
    _s6x_v2 = _s6x_pd.read_csv(_s6x_v2_path, dtype={'StudyInstanceUID': str})
    _s6x_ro0 = {c: pct_on(ours[c].values, ~our_fail) for c in labels}
    _s6x_s3 = _s6x_stage3(_s6x_ro0, our_fail)
    _s6x_own = None
    if 'OWN_W_PER_TARGET' in globals() and _s6x_os.path.isfile(_s6x_own_path):
        _s6x_own = _s6x_pd.read_csv(_s6x_own_path, dtype={'StudyInstanceUID': str})
    _s6x_diff = {}
    for _s6x_nm, _s6x_cand in (('stage3+partB', None if _s6x_own is None else _s6x_part_b(_s6x_s3, _s6x_own)),
                               ('stage3', _s6x_s3)):
        if _s6x_cand is not None and _s6x_cand['StudyInstanceUID'].tolist() == _s6x_v2['StudyInstanceUID'].tolist():
            _s6x_diff[_s6x_nm] = float(_s6x_np.abs(_s6x_cand[labels].values.astype(_s6x_np.float64)
                                                   - _s6x_v2[labels].values.astype(_s6x_np.float64)).max())
    _s6x_info['selfcheck_max_abs_diff'] = _s6x_diff
    _s6x_path_b = [k for k, v in _s6x_diff.items() if v < 1e-6]
    if not _s6x_path_b:
        raise RuntimeError('V2 rebuild does not match its own file: %s' % _s6x_diff)
    _s6x_use_b = _s6x_path_b[0] == 'stage3+partB'
    _s6x_info['part_b_applied'] = _s6x_use_b
    _s6x_status = 'selfcheck_ok'
    _s6x_log('SELF_CHECK ok %s' % _s6x_json.dumps(_s6x_diff))
except Exception as _s6x_e:
    _s6x_status = 'selfcheck_failed'
    _s6x_info['error'] = repr(_s6x_e)[:500]
    _s6x_log('SELF_CHECK FAILED: %r' % (_s6x_e,))

# 2. s6fall as its own process, under the notebook-time limit
if _s6x_status == 'selfcheck_ok':
    _s6x_elapsed_h = (_s6x_time.time() - NB_T0) / 3600
    _s6x_status = 'skipped_late'
    if _s6x_elapsed_h <= _S6X_LATEST_START_H:
        _s6x_limit = (_S6X_END_H - _s6x_elapsed_h) * 3600
        _s6x_log('s6fall stage: start, limit %.0f min' % (_s6x_limit / 60))
        _s6x_t = _s6x_time.time()
        _s6x_rc = 'not started'
        try:
            _s6x_p = _s6x_sp.Popen([_s6x_sys.executable, '-u', '/tmp/s6fall_run.py'], start_new_session=True)
            try:
                _s6x_rc = _s6x_p.wait(timeout=_s6x_limit)
            except _s6x_sp.TimeoutExpired:
                _s6x_rc = None
                _s6x_kill(_s6x_p)
        except Exception as _s6x_e:
            _s6x_info['error'] = repr(_s6x_e)[:500]
        _s6x_info['s6f_seconds'] = round(_s6x_time.time() - _s6x_t, 1)
        _s6x_info['s6f_rc'] = _s6x_rc
        _s6x_status = 'ran' if (_s6x_rc == 0 and _s6x_os.path.isfile(_s6x_sub)) else ('timeout' if _s6x_rc is None else 'failed')

# 3. re-blend: ours = (1 - share) rank(R6) + share rank(s6fall), then V2's stage 3 and Part B unchanged
if _s6x_status == 'ran':
    try:
        _s6x_s6 = _s6x_pd.read_csv(_s6x_sub, dtype={'StudyInstanceUID': str})
        assert sorted(_s6x_s6['StudyInstanceUID']) == sorted(test_ids), 's6fall: rows differ from test.csv'
        assert set(labels) <= set(_s6x_s6.columns), 's6fall: columns differ'
        _s6x_sh.copyfile(_s6x_sub, '/kaggle/working/_s6f.csv')
        _s6x_s6 = _s6x_s6.set_index('StudyInstanceUID').loc[test_ids, labels]
        try:
            with open('/kaggle/working/k1/blend_receipt.json', encoding='utf-8') as _s6x_fh:
                _s6x_bf = set(_s6x_json.load(_s6x_fh).get('b_failed_rank_A_only') or {})
        except Exception as _s6x_e:
            _s6x_bf = set()
            _s6x_log('blend_receipt.json unread (%s): every s6fall row counts as scored' % type(_s6x_e).__name__)
        _s6x_ok = _s6x_np.array([u not in _s6x_bf for u in test_ids])
        _s6x_r6_ok = ~our_fail
        _s6x_ro, _s6x_rho = {}, {}
        for _s6x_c in labels:
            _s6x_a, _s6x_b = pct_on(ours[_s6x_c].values, _s6x_r6_ok), pct_on(_s6x_s6[_s6x_c].values, _s6x_ok)
            _s6x_ro[_s6x_c] = _s6x_np.where(_s6x_r6_ok & _s6x_ok, (1.0 - _S6X_SHARE) * _s6x_a + _S6X_SHARE * _s6x_b,
                                       _s6x_np.where(_s6x_r6_ok, _s6x_a, _s6x_np.where(_s6x_ok, _s6x_b, _s6x_np.nan)))
            _s6x_both = _s6x_r6_ok & _s6x_ok
            if _s6x_both.sum() >= 3:
                _s6x_rho[_s6x_c] = round(float(_s6x_pd.Series(_s6x_a[_s6x_both]).corr(_s6x_pd.Series(_s6x_b[_s6x_both]),
                                                                                 method='spearman')), 4)
        _s6x_fail = our_fail & ~_s6x_ok
        _s6x_new = _s6x_stage3(_s6x_ro, _s6x_fail)
        if _s6x_use_b:
            _s6x_new = _s6x_part_b(_s6x_new, _s6x_own)
        _s6x_vals = _s6x_new[labels].values.astype(_s6x_np.float64)
        assert _s6x_new['StudyInstanceUID'].tolist() == _s6x_v2['StudyInstanceUID'].tolist(), 'row order changed'
        assert _s6x_np.isfinite(_s6x_vals).all() and ((_s6x_vals >= 0) & (_s6x_vals <= 1)).all(), 'final out of range'
        _s6x_new[_s6x_v2.columns.tolist()].to_csv(_s6x_sub, index=False)
        _s6x_info.update(s6f_failed=len(_s6x_bf), r6_failed=int(our_fail.sum()), both_failed=int(_s6x_fail.sum()),
                         rho_r6_s6f=_s6x_rho,
                         rho_mean=round(float(_s6x_np.mean(list(_s6x_rho.values()))), 4) if _s6x_rho else None,
                         max_abs_change_vs_v2=float(_s6x_np.abs(_s6x_vals - _s6x_v2[labels].values).max()))
        _s6x_status = 'ok'
    except Exception as _s6x_e:
        _s6x_status = 'combine_failed'
        _s6x_info['error'] = repr(_s6x_e)[:500]
if _s6x_status != 'ok':
    _s6x_sh.copyfile(_s6x_v2_path, _s6x_sub)                 # V2's own file (#100)
_s6x_info['status'] = _s6x_status
_s6x_info['total_hours'] = round((_s6x_time.time() - NB_T0) / 3600, 3)
with open('/kaggle/working/s6x_receipt.json', 'w', encoding='utf-8') as _s6x_fh:
    _s6x_json.dump(_s6x_info, _s6x_fh, indent=1)
_s6x_log('S6X_DONE ' + _s6x_json.dumps(_s6x_info))
"""


# C17 (10-19): C14 + the 2026-10-18 audit's fixes (traps 74) + s6fall's training-matched crop TTA.  s6fall trained on a random 288
# crop of its 320 render (compact_train.augment: one offset in 0..32 px per axis per batch); compact_infer scores the centre crop
# only.  Crops are (oy, ox) offsets into the 320 render, (16, 16) = the centre; probabilities are averaged.
C17_CROPS5 = ((16, 16), (0, 0), (0, 32), (32, 0), (32, 32))
ANCHORS["09586s6fx"] = dict(ANCHORS["09586s6f"], slug="rsna-knee-anchor9586s6fx", title="RSNA Knee Anchor9586s6fx",
                            s6fall=dict(ANCHORS["09586s6f"]["s6fall"], crops=C17_CROPS5))

# C18 (10-19): nartaa's 384 px CoAtNet scored on the field it trained on.  RUNS.md (nartaa/rsna-knee-s6fall-weights, training_code/
# coatnet): the 0.949 checkpoint trained with --res 384 and crop 1.0 (the full 140 mm field, no random crop, no affine) and its SWA
# epochs were picked by gold-58 AUC evaluated at 384 (train_knee.py _resize); it was "scored 320 crop + mirror TTA".  V2's cell 2
# scores the 320 px centre crop; INPUT_RES drives the backbone img_size, the crop offset (IMG - res) // 2 and the window asserts,
# but infer_pair_probs (the mirror TTA) hard-codes the window shape (94, 3, 320, 320): the first C18 placeholders (10-19 v1) failed
# that bare assert on 3 of 3 studies, so it is patched too (mirror TTA kept).  Cost: (384 / 320)^2 = 1.44x on that stage.
C18_PATCHES = [("INPUT_RES = 320\n",
                "INPUT_RES = 384   # [C18] the full 384 px field this checkpoint trained and was SWA-selected on (nartaa's RUNS.md: "
                "res 384, crop 1.0); V2 scored a 320 px centre crop\n"),
               ("    assert plain.shape==mirrored.shape==(94,3,320,320)\n",
                "    assert plain.shape==mirrored.shape==(94,3,INPUT_RES,INPUT_RES)   # [C18] V2 hard-coded its 320 crop here\n")]
C18_NOTE = (", except that the 384 px CoAtNet stage (cell 2) scores its full 384 px field, the one it trained on, instead of a "
            "320 px centre crop (C18)")
# C19 (10-19, Tian: pick 2): the four per-finding Raptor-half weights back to the 0.958 notebook's values.  V2 pushed each a further
# 0.05 from 0.5 on public reads (heliosli's PROBE_CLASS); shrinking a public fit is the private-robust direction (research.md 2.7.12),
# and the pair (pick 1 on V2's weights, pick 2 on these) covers both "the pushes were real" and "the pushes were public overfit".
C19_PATCHES = [("W_RAPTOR_C = {'Medial OA': 0.5, 'Lateral OA': 0.15, 'PF OA': 0.65, 'Synovitis': 0.85, \"Baker's\": 0.30}",
                "W_RAPTOR_C = {'Medial OA': 0.5, 'Lateral OA': 0.2, 'PF OA': 0.6, 'Synovitis': 0.8, \"Baker's\": 0.35}   # [C19] the 0.958 "
                "notebook's weights (V2 pushed each 0.05 further on public reads; shrunk back)")]
C19_NOTE = ", except that the four per-finding Raptor-half weights in cell 8 are the 0.958 notebook's, before V2's push (C19)"


def _variant(base, suffix, slug_suffix, patches, note):
    b = ANCHORS[base]
    ANCHORS[base + suffix] = dict(b, slug=b["slug"] + slug_suffix, title=b["title"] + slug_suffix,   # title -> slug
                                  patches={**b["patches"], **patches}, extra_note=b.get("extra_note", "") + note)


for _b in ("09586s6f", "09586s6fx"):        # C18 on C14 / on C17; C19 on either, with or without C18
    _variant(_b, "f", "f", {2: C18_PATCHES}, C18_NOTE)
for _b in ("09586s6f", "09586s6fx", "09586s6ff", "09586s6fxf"):
    _variant(_b, "w", "w", {8: C19_PATCHES}, C19_NOTE)

# C20 (10-20, Tian: "R7 as a third R6-half model"): C14 + heliosli's R7 (bundle 8c7b666e27dc0e8f, the 0.956 notebook's compact model)
# inside V2's R6 half next to R6 and s6fall.  R7's manifest pins exactly V2's seven embedded sources (compact_infer / probe_dual /
# the trainer md5 8622009e ...) and the same anatomy.pt, so V2's own cells 6-7 run it with four constants changed; they run as a
# patched copy in their own namespace and directories, after the s6fall stage.  Weights: s6fall keeps its 0.4; R6 : R7 = 0.6 : 0.4 in
# the rest (goodpjw2008's "R7 at 0.4 next to R6", read as no change on the 0.958 recipe), so 0.36 / 0.24 / 0.40.  R7 runs without
# flip TTA (its manifest: flip_tta false; ~9.8 GPU-s / study, ~1.75-2 h on the hidden test).
R7X_SLUG = "heliosli/rsna-knee-compact-r7cnxfsss"
R7X_CELL6_PATCHES = [
    ("C_SLUG = '%s-w1'   # [anchor9586: the -w1 copy = v1 = R6; this slug's latest version holds R7, traps 72] [d957] the frozen "
     "copy of the R6 bundle dataset (manifest sha256 6508ce6921eda986.. = 57068142's)\n" % _R6,
     "C_SLUG = 'rsna-knee-compact-r7cnxfsss'   # [C20] R7's bundle Dataset (the 0.956 notebook's compact model, as C8 / C12 ran it)\n"),
    ("BUNDLE_ID = 'fe055406fad6a12f'\n", "BUNDLE_ID = '8c7b666e27dc0e8f'   # [C20] R7\n"),
    ("FLIP_TTA = True   # [d957] package E: v4c's flip TTA on R6\n",
     "FLIP_TTA = False   # [C20] R7's manifest: flip_tta false (R7 trained with mirror 0.5, but its own notebook scored no flip)\n"),
    ("TMP = TMP / 'probe'\n", "TMP = TMP / 'probe_r7'   # [C20] R7's own scratch dir: R6's stays as V2 left it\n"),
    ("WORK = Path('/kaggle/working')\n",
     "WORK = Path('/kaggle/working/r7')   # [C20] R7's ours_sub.csv, receipt and worker logs beside V2's, never over them\n"
     "WORK.mkdir(parents=True, exist_ok=True)\n"),
]
ANCHORS["09586s6fr7"] = dict(ANCHORS["09586s6f"], slug="rsna-knee-anchor9586s6fr7", title="RSNA Knee Anchor9586s6fr7",
                             dataset_sources=ANCHORS["09586s6f"]["dataset_sources"] + [R7X_SLUG],
                             r7x={"w_r6": 0.36, "w_r7": 0.24, "w_s6f": 0.40, "latest_start_h": 6.3, "min_frac": 0.5,
                                  "rho_floor": 0.5, "rho_min_rows": 30})
# C20 on C19 (10-21, Tian: "Send C20 on C19 first"): #106 C19 read above C14 inside the tie, so R7 is read on C19's base -- the C20
# render with cell 8's four Raptor-half weights at the 0.958 notebook's.  Stage R's _s6x_stage3 reads W_RAPTOR_C at call time, so its
# SELF_CHECK, its re-blend and every fallback use C19's weights: the fallback file is C19's (#106), not C14's.
_variant("09586s6fr7", "w", "w", {8: C19_PATCHES}, C19_NOTE)
# C21m on either base (C19 or C20 on C19, whichever #107 favours): + V-JEPA as a third component of the final file (VJX_* below).
# Runtime: ~4.5 s / study on one T4 (10-21 placeholder) -> ~0.8 h with one worker per GPU on ~1,284 studies; the start gate leaves
# 1.3 h + margin before the 8.5 h stop, and a late finish keeps the scored studies (>= 50 % coverage required).
VJX_SLUG = "nurikw3/rsna-knee-vjepa21-checkpoints"
VJX_CFG = {"gamma": 0.15, "latest_start_h": 7.0, "end_h": 8.5, "min_frac": 0.5, "rho_floor": 0.35, "rho_min_rows": 30, "nshard": 2}
ANCHORS["09586s6fr7w"]["r7x_note"] = (
    "**C20 on C19** (`--anchor 09586s6fr7w`, Tian 10-21): cell 8's four per-finding Raptor-half weights are the 0.958 notebook's "
    "(C19, `rsna-knee-anchor9586s6fw`, #106), not V2's. Wherever this cell and the next say C14, read C19: the SELF_CHECK rebuilds "
    "C19's file and every fallback is C19's own file.")
for _b, _name in (("09586s6fw", "C19 (`rsna-knee-anchor9586s6fw`, #106)"),
                  ("09586s6fr7w", "C20 on C19 (`rsna-knee-anchor9586s6fr7w`, #107)")):
    ANCHORS[_b + "vj"] = dict(ANCHORS[_b], slug=ANCHORS[_b]["slug"] + "vj", title=ANCHORS[_b]["title"] + "vj",
                              dataset_sources=ANCHORS[_b]["dataset_sources"] + [VJX_SLUG], vjx=dict(VJX_CFG, base=_name))

R7X_INTRO = """# Anchor9586s6fr7: heliosli's R7 as a third member of C14's R6 half (candidates.md C20)

Built by `src/build_anchor_copy.py --anchor 09586s6fr7`. Every cell above is C14 (`rsna-knee-anchor9586s6f`, #101). The next cell
runs heliosli's R7 (bundle 8c7b666e27dc0e8f: 4 ConvNeXt-S folds, no flip TTA, as its manifest says) through V2's own compact
controller: a copy of cells 6-7 with R7's Dataset, bundle id, no flip and its own scratch and output directories, run in its own
namespace. It then re-blends: ours = __W_R6__ rank(R6) + __W_R7__ rank(R7) + __W_S6F__ rank(s6fall) per finding, each over the studies it
scored, then V2's stage-3 and Part B weights unchanged. Before R7 starts, the cell rebuilds C14's file with R7's weight at 0
(SELF_CHECK). R7 starts only before __LATEST__ h of notebook time; its controller stops at V2's 8.3 h and keeps the studies it scored. On a
skip, a failure, under __MINPCT__ % coverage or a low R6 / R7 rank correlation, the submission is C14's own file.
Credits: heliosli (V2, R6, R7, S6 architecture, inference code, SKM-TEA anatomy model), nartaa (CoAtNets; s6fall weights,
CC BY-NC 4.0), goodpjw2008 (2.5D reader).
"""

R7X_RUN_CELL = r"""# Anchor9586s6fr7 (C20): stage R -- heliosli's R7 through V2's own compact controller (cells 6-7, patched copy, own namespace and
# directories), then ours = w_r6 rank(R6) + w_r7 rank(R7) + w_s6f rank(s6fall) per finding, V2's stage 3 and Part B unchanged.
# Reads V2's globals (ours, our_fail, labels, test_ids, pct_on, NB_T0) and the s6fall cell's (_s6x_*, _S6X_SHARE); binds only _r7x_*
# names.  Any skip / failure / low coverage: C14's own file (#101) is the submission.
import base64 as _r7x_b64, csv as _r7x_csv, hashlib as _r7x_hl, json as _r7x_json, os as _r7x_os, shutil as _r7x_sh
import time as _r7x_time, zlib as _r7x_zlib
import numpy as _r7x_np
import pandas as _r7x_pd
_R7X_W_R6, _R7X_W_R7, _R7X_W_S6F = __W_R6__, __W_R7__, __W_S6F__
_R7X_LATEST_START_H = __LATEST__
_R7X_MIN_FRAC = __MIN_FRAC__
_R7X_RHO_FLOOR, _R7X_RHO_MIN_ROWS = __RHO_FLOOR__, __RHO_MIN_ROWS__
_R7X_CODE = (('cell6', '__C6_B64__', '__C6_SHA__'),
             ('cell7', '__C7_B64__', '__C7_SHA__'))
_r7x_sub = '/kaggle/working/submission.csv'
_r7x_c14_path = '/kaggle/working/_c14_final.csv'
_r7x_out = '/kaggle/working/r7/ours_sub.csv'
_r7x_sh.copyfile(_r7x_sub, _r7x_c14_path)            # C14's final file: the submission unless every step below succeeds
_r7x_info = {'weights': {'r6': _R7X_W_R6, 'r7': _R7X_W_R7, 's6f': _R7X_W_S6F}, 'rows': len(test_ids)}
_r7x_status = 'not_run'


def _r7x_log(msg):
    print('[r7x %5.2fh] %s' % ((_r7x_time.time() - NB_T0) / 3600, msg), flush=True)


def _r7x_final(members):
    # ours per finding = the weighted mean of each member's percentile rank over its own scored rows, renormalised over the members
    # that scored the study (none: NaN, which stage 3 replaces by the Raptor rank, as V2 does); then V2's stage 3 and Part B
    ro = {}
    for c in labels:
        num, den = _r7x_np.zeros(len(test_ids)), _r7x_np.zeros(len(test_ids))
        for w, vals, ok in members:
            r = pct_on(vals[c].values, ok)
            num = num + _r7x_np.where(ok, w * r, 0.0)
            den = den + _r7x_np.where(ok, w, 0.0)
        ro[c] = _r7x_np.where(den > 0, num / _r7x_np.where(den > 0, den, 1.0), _r7x_np.nan)
    fail = _r7x_np.ones(len(test_ids), dtype=bool)
    for _, _, ok in members:
        fail &= ~ok
    new = _s6x_stage3(ro, fail)
    return _s6x_part_b(new, _s6x_own) if _s6x_use_b else new


# 1. SELF_CHECK: C14 must be complete, and the three-member blend with R7's weight at 0 must rebuild C14's file
try:
    if globals().get('_s6x_status') != 'ok':
        raise RuntimeError('the s6fall stage ended %r: C20 adds R7 to a complete C14 only' % globals().get('_s6x_status'))
    _r7x_c14 = _r7x_pd.read_csv(_r7x_c14_path, dtype={'StudyInstanceUID': str})
    _r7x_re = _r7x_final([(1.0 - _S6X_SHARE, ours, ~our_fail), (_S6X_SHARE, _s6x_s6, _s6x_ok)])
    assert _r7x_re['StudyInstanceUID'].tolist() == _r7x_c14['StudyInstanceUID'].tolist(), 'row order differs from C14'
    _r7x_d = float(_r7x_np.abs(_r7x_re[labels].values.astype(_r7x_np.float64)
                               - _r7x_c14[labels].values.astype(_r7x_np.float64)).max())
    _r7x_info['selfcheck_max_abs_diff'] = _r7x_d
    if not _r7x_d < 1e-6:
        raise RuntimeError('the C14 rebuild differs from its file by %g' % _r7x_d)
    _r7x_status = 'selfcheck_ok'
    _r7x_log('SELF_CHECK ok %g' % _r7x_d)
except Exception as _r7x_e:
    _r7x_status = 'selfcheck_failed'
    _r7x_info['error'] = repr(_r7x_e)[:500]
    _r7x_log('SELF_CHECK FAILED: %r' % (_r7x_e,))

# 2. R7: V2's cells 6-7, patched, in their own namespace (the controller's deadline is V2's own: 8.3 h of notebook time)
if _r7x_status == 'selfcheck_ok':
    _r7x_status = 'skipped_late'
    _r7x_elapsed_h = (_r7x_time.time() - NB_T0) / 3600
    _r7x_info['start_h'] = round(_r7x_elapsed_h, 3)
    if _r7x_elapsed_h <= _R7X_LATEST_START_H:
        _r7x_t = _r7x_time.time()
        _r7x_g = {'__name__': '__r7x__', 'NB_T0': NB_T0, 'display': globals().get('display', print)}
        try:
            for _r7x_k, _r7x_b, _r7x_h in _R7X_CODE:
                _r7x_txt = _r7x_zlib.decompress(_r7x_b64.b64decode(_r7x_b)).decode('utf-8')
                if _r7x_hl.sha256(_r7x_txt.encode('utf-8')).hexdigest() != _r7x_h:
                    raise RuntimeError(_r7x_k + ': payload sha256 differs')
                _r7x_log('R7 %s: start' % _r7x_k)
                exec(compile(_r7x_txt, 'anchor9586s6fr7-' + _r7x_k, 'exec'), _r7x_g)
            _r7x_status = 'ran'
        except Exception as _r7x_e:
            _r7x_status = 'r7_failed'
            _r7x_info['error'] = repr(_r7x_e)[:500]
            _r7x_log('R7 FAILED: %r' % (_r7x_e,))
        _r7x_info['r7_seconds'] = round(_r7x_time.time() - _r7x_t, 1)

# 3. R7's scored studies (V2's cell-8 rule for R6), the guards, and the three-member re-blend
if _r7x_status == 'ran':
    try:
        _r7x_scored = set()
        for _r7x_f in sorted(_r7x_g['pred'].glob('worker*/predictions.csv')):
            with open(_r7x_f, encoding='utf-8') as _r7x_fh:
                for _r7x_row in _r7x_csv.DictReader(_r7x_fh):
                    try:
                        _r7x_v = [float(_r7x_row[_r7x_c]) for _r7x_c in labels]
                    except (KeyError, ValueError, TypeError):
                        continue
                    if all(0. <= _r7x_x <= 1. for _r7x_x in _r7x_v):
                        _r7x_scored.add(_r7x_row['StudyInstanceUID'])
        _r7x_ok = _r7x_np.array([u in _r7x_scored for u in test_ids])
        _r7x_info['r7_scored'] = int(_r7x_ok.sum())
        if _r7x_ok.sum() < _R7X_MIN_FRAC * len(test_ids):
            raise RuntimeError('R7 scored %d of %d studies' % (_r7x_ok.sum(), len(test_ids)))
        _r7x_r7 = _r7x_pd.read_csv(_r7x_out, dtype={'StudyInstanceUID': str})
        assert sorted(_r7x_r7['StudyInstanceUID']) == sorted(test_ids), 'R7: rows differ from test.csv'
        assert set(labels) <= set(_r7x_r7.columns), 'R7: columns differ'
        _r7x_r7 = _r7x_r7.set_index('StudyInstanceUID').loc[test_ids, labels]
        _r7x_both = ~our_fail & _r7x_ok
        _r7x_rho = {}
        if _r7x_both.sum() >= 3:
            for _r7x_c in labels:
                _r7x_rho[_r7x_c] = round(float(_r7x_pd.Series(ours[_r7x_c].values[_r7x_both]).corr(
                    _r7x_pd.Series(_r7x_r7[_r7x_c].values[_r7x_both]), method='spearman')), 4)
        _r7x_info['rho_r6_r7'] = _r7x_rho
        _r7x_info['rho_mean'] = round(float(_r7x_np.nanmean(list(_r7x_rho.values()))), 4) if _r7x_rho else None
        if _r7x_both.sum() >= _R7X_RHO_MIN_ROWS and not (_r7x_info['rho_mean'] or 0.0) >= _R7X_RHO_FLOOR:
            raise RuntimeError('R6 / R7 mean rank correlation %r under %r' % (_r7x_info['rho_mean'], _R7X_RHO_FLOOR))
        _r7x_new = _r7x_final([(_R7X_W_R6, ours, ~our_fail), (_R7X_W_S6F, _s6x_s6, _s6x_ok), (_R7X_W_R7, _r7x_r7, _r7x_ok)])
        _r7x_vals = _r7x_new[labels].values.astype(_r7x_np.float64)
        assert _r7x_new['StudyInstanceUID'].tolist() == _r7x_c14['StudyInstanceUID'].tolist(), 'row order changed'
        assert _r7x_np.isfinite(_r7x_vals).all() and ((_r7x_vals >= 0) & (_r7x_vals <= 1)).all(), 'final out of range'
        _r7x_new[_r7x_c14.columns.tolist()].to_csv(_r7x_sub + '.r7x.tmp', index=False)
        _r7x_os.replace(_r7x_sub + '.r7x.tmp', _r7x_sub)
        _r7x_info['max_abs_change_vs_c14'] = float(_r7x_np.abs(_r7x_vals - _r7x_c14[labels].values).max())
        _r7x_status = 'ok'
    except Exception as _r7x_e:
        _r7x_status = 'combine_failed'
        _r7x_info['error'] = repr(_r7x_e)[:500]
        _r7x_log('R7 combine FAILED: %r' % (_r7x_e,))
if _r7x_status != 'ok':
    _r7x_sh.copyfile(_r7x_c14_path, _r7x_sub)            # C14's own file (#101)
_r7x_info['status'] = _r7x_status
_r7x_info['total_hours'] = round((_r7x_time.time() - NB_T0) / 3600, 3)
with open('/kaggle/working/r7x_receipt.json', 'w', encoding='utf-8') as _r7x_fh:
    _r7x_json.dump(_r7x_info, _r7x_fh, indent=1)
_r7x_log('R7X_DONE ' + _r7x_json.dumps(_r7x_info))
"""


def _free_names(src):
    """Names a module reads but never binds anywhere, minus builtins (what an exec'd copy needs from its globals)."""
    import builtins
    import symtable
    free, bound = set(), set()

    def walk(t, top):
        for s in t.get_symbols():
            if top and (s.is_assigned() or s.is_imported()):
                bound.add(s.get_name())
            elif (top and s.is_referenced()) or s.is_global():
                free.add(s.get_name())
        for ch in t.get_children():
            walk(ch, False)
    walk(symtable.symtable(src, "cell", "exec"), True)
    return {n for n in free - bound if not hasattr(builtins, n)}


def r7x_cells(cells, cfg, note=""):
    """C20's two added cells: an intro (plus `note`, if any) and the stage-R run cell, which carries V2's cells 6-7 (as this render
    holds them, plus R7X_CELL6_PATCHES) as a zlib + base64 payload with its sha256."""
    import base64
    import hashlib
    import zlib
    c6, c7 = cell_text(cells[6]), cell_text(cells[7])
    if not c7.lstrip().startswith("deadline = T0 + DEADLINE_HOURS * 3600") or "probe_dual.py" not in c7:
        raise SystemExit("C20: cell 7 is not V2's compact-controller run cell")
    if "DEADLINE_HOURS = 8.3\n" not in c6:
        raise SystemExit("C20: cell 6 no longer sets V2's 8.3 h controller deadline")
    for old, new in R7X_CELL6_PATCHES:
        if c6.count(old) != 1:
            raise SystemExit(f"C20: cell-6 patch target found {c6.count(old)} times: {old!r}")
        c6 = c6.replace(old, new)
    for name, src in (("cell6", c6), ("cell7", c7)):
        compile(src, "c20-" + name, "exec")              # plain Python: no IPython magic may survive in the payload
    need = (_free_names(c6) | _free_names(c7)) - module_level_names(c6) - module_level_names(c7)
    missing = sorted(need - {"NB_T0", "display"})       # the two names stage R seeds into the payload's namespace
    if missing:
        raise SystemExit(f"C20: cells 6-7 read {missing}, which the stage-R namespace does not provide")
    if not (0 < cfg["w_r7"] < 1 and abs(cfg["w_r6"] + cfg["w_r7"] + cfg["w_s6f"] - 1.0) < 1e-12
            and cfg["latest_start_h"] < 8.3 and 0 < cfg["min_frac"] <= 1):
        raise SystemExit("C20: weights must sum to 1 with 0 < w_r7 < 1; latest start < 8.3 h; 0 < min_frac <= 1")
    payload = {}
    for name, src in (("cell6", c6), ("cell7", c7)):
        payload[name] = (base64.b64encode(zlib.compress(src.encode("utf-8"), 9)).decode("ascii"),
                         hashlib.sha256(src.encode("utf-8")).hexdigest())
    run = R7X_RUN_CELL
    for k, v in (("__W_R6__", repr(cfg["w_r6"])), ("__W_R7__", repr(cfg["w_r7"])), ("__W_S6F__", repr(cfg["w_s6f"])),
                 ("__LATEST__", repr(cfg["latest_start_h"])), ("__MIN_FRAC__", repr(cfg["min_frac"])),
                 ("__RHO_FLOOR__", repr(cfg["rho_floor"])), ("__RHO_MIN_ROWS__", repr(cfg["rho_min_rows"])),
                 ("__C6_B64__", payload["cell6"][0]), ("__C6_SHA__", payload["cell6"][1]),
                 ("__C7_B64__", payload["cell7"][0]), ("__C7_SHA__", payload["cell7"][1])):
        if run.count(k) != 1:
            raise SystemExit(f"C20: template placeholder {k} found {run.count(k)} times")
        run = run.replace(k, v)
    compile(run, "anchor9586s6fr7-run", "exec")
    bad = sorted(n for n in module_level_names(run) if not n.startswith("_r7x_") and not n.startswith("_R7X_"))
    if bad:
        raise SystemExit(f"the R7 run cell binds {bad} at notebook scope (only _r7x_* names are allowed)")
    intro = (R7X_INTRO.replace("__W_R6__", "%.2f" % cfg["w_r6"]).replace("__W_R7__", "%.2f" % cfg["w_r7"])
             .replace("__W_S6F__", "%.2f" % cfg["w_s6f"]).replace("__LATEST__", "%.1f" % cfg["latest_start_h"])
             .replace("__MINPCT__", "%d" % round(100 * cfg["min_frac"])))
    if note:
        intro = intro.rstrip("\n") + "\n\n" + note + "\n"
    return [{"id": "anchor9586s6fr7-intro", "cell_type": "markdown", "metadata": {}, "source": intro},
            {"id": "anchor9586s6fr7-run", "cell_type": "code", "metadata": {}, "outputs": [], "execution_count": None,
             "source": run}], {"cell6_sha256": payload["cell6"][1], "cell7_sha256": payload["cell7"][1],
                                "cell6_patches": [n for _, n in R7X_CELL6_PATCHES]}


# C21m (10-21, Tian: "build C21 if the V-JEPA solo reads >= 0.945"): nurikw3's public V-JEPA 2.1 ViT-B (5 folds; Dataset CC0, the
# vendored facebookresearch/vjepa2 code MIT) as a third component of the FINAL file, after every other stage: per finding,
# rank((1 - gamma) pct(base) + gamma pct(V-JEPA)) over the studies V-JEPA scored, the base's rank elsewhere.  gamma is fixed a
# priori (Tian: 15 %), never tuned on the LB.  V-JEPA runs as one worker process per GPU over interleaved halves of the studies, each
# writing a partial CSV every 10 studies, so the deadline keeps what was scored.  Any skip / failure / low coverage / low rank
# correlation with the base: the base's own file.  The worker is the author's infer.py functions in fp16 (the 10-21 placeholder:
# fp16 vs fp32 max |d| 1.1e-3; ~4.5 s / study on one T4 with decoding).
VJX_WORKER = r'''"""C21m worker: one shard of the test studies through nurikw3's V-JEPA 2.1 5-fold models in fp16; a partial CSV every 10."""
import glob, os, sys, time, types
from pathlib import Path
import numpy as np
import pandas as pd
import torch

SHARD, NSHARD, OUT = int(sys.argv[1]), int(sys.argv[2]), sys.argv[3]


def find_one(tail):
    for depth in range(1, 6):
        hits = sorted(glob.glob("/kaggle/input/" + "*/" * depth + tail))
        if hits:
            return Path(hits[0])
    raise SystemExit(tail + " not found under /kaggle/input")


VJ = find_one("f0_best.pt").parent
COMP = find_one("test_series.csv").parent
sys.path[:0] = [str(VJ), str(VJ / "vjepa2")]
mod = types.ModuleType("vj_infer")
mod.__file__ = str(VJ / "infer.py")
exec(compile((VJ / "infer.py").read_text(encoding="utf-8"), mod.__file__, "exec"), mod.__dict__)
test = pd.read_csv(COMP / "test.csv", dtype={"StudyInstanceUID": str})
series = pd.read_csv(COMP / "test_series.csv", dtype={"StudyInstanceUID": str})
by = {str(u): g for u, g in series.groupby("StudyInstanceUID", sort=False)}
uids = test.StudyInstanceUID.tolist()[SHARD::NSHARD]
t0 = time.time()
models = [mod.load_model(cp) for cp in sorted(VJ.glob("f*_best.pt"))]
print("VJX shard %d: %d folds loaded in %.0f s, %d studies" % (SHARD, len(models), time.time() - t0, len(uids)), flush=True)
rows, failed = [], 0


def flush():
    pd.DataFrame(rows, columns=["StudyInstanceUID"] + list(mod.TARGETS)).to_csv(OUT + ".tmp", index=False)
    os.replace(OUT + ".tmp", OUT)


with torch.inference_mode():
    for i, uid in enumerate(uids):
        try:
            gray, present = mod.load_study_tensor(COMP, uid, by.get(uid))
            if not gray.any():
                raise ValueError("no pixels")
            x = torch.from_numpy(gray).unsqueeze(0).cuda()
            p = torch.from_numpy(present).unsqueeze(0).cuda()
            with torch.autocast("cuda", dtype=torch.float16):
                probs = [torch.sigmoid(m(x, p)).float().squeeze(0).cpu().numpy() for m in models]
            rows.append([uid] + [float(v) for v in np.mean(probs, axis=0)])
        except Exception as e:
            failed += 1
            print("VJX shard %d: study %d failed: %s" % (SHARD, i, type(e).__name__), flush=True)
        if (i + 1) % 10 == 0 or i + 1 == len(uids):
            flush()
        if (i + 1) % 100 == 0:
            print("VJX shard %d: %d / %d in %.0f s" % (SHARD, i + 1, len(uids), time.time() - t0), flush=True)
flush()
print("VJX_SHARD_DONE %d scored %d failed %d in %.0f s" % (SHARD, len(rows), failed, time.time() - t0), flush=True)
'''

VJX_INTRO = """# C21m: nurikw3's V-JEPA 2.1 as a third component of the final file (candidates.md C21)

Built by `src/build_anchor_copy.py --anchor __TAG__`. Every cell above is __BASE__. The next cell runs nurikw3's public V-JEPA 2.1
ViT-B (5 folds, Dataset `nurikw3/rsna-knee-vjepa21-checkpoints`, CC0; vendored `facebookresearch/vjepa2` code, MIT) in fp16, one worker
per GPU, and re-ranks: final = rank((1 - __GAMMA__) pct(base) + __GAMMA__ pct(V-JEPA)) per finding over the studies V-JEPA scored. It
starts only before __LATEST__ h of notebook time and stops at __END__ h, keeping what it scored. On a skip, a failure, under __MINPCT__ %
coverage or a mean rank correlation with the base under __RHO__, the submission is the base's own file.
Credits: heliosli (V2, R6, R7, inference code, SKM-TEA anatomy model), nartaa (CoAtNets; s6fall weights, CC BY-NC 4.0), goodpjw2008
(2.5D reader), nurikw3 (V-JEPA 2.1 fine-tune).
"""

VJX_RUN_CELL = r"""# C21m: stage V -- nurikw3's V-JEPA 2.1 (5 folds, fp16, one worker per GPU) as a third component of the final file:
# final = rank((1 - gamma) pct(base) + gamma pct(V-JEPA)) per finding over the studies V-JEPA scored, the base's rank elsewhere.
# Reads the notebook's test_ids, labels and NB_T0; binds only _vjx_* names.  Any skip / failure / low coverage: the base's own file.
import base64 as _vjx_b64, hashlib as _vjx_hl, json as _vjx_json, os as _vjx_os, shutil as _vjx_sh, signal as _vjx_sig
import subprocess as _vjx_sp, sys as _vjx_sys, time as _vjx_time, zlib as _vjx_zlib
import numpy as _vjx_np
import pandas as _vjx_pd
import torch as _vjx_torch
_VJX_GAMMA = __GAMMA__
_VJX_LATEST_START_H, _VJX_END_H = __LATEST__, __END__
_VJX_MIN_FRAC, _VJX_RHO_FLOOR, _VJX_RHO_MIN_ROWS = __MIN_FRAC__, __RHO_FLOOR__, __RHO_MIN_ROWS__
_VJX_NSHARD = max(1, min(__NSHARD__, _vjx_torch.cuda.device_count()))
_VJX_WORKER = ('__W_B64__', '__W_SHA__')
_vjx_sub = '/kaggle/working/submission.csv'
_vjx_base_path = '/kaggle/working/_base_final.csv'
_vjx_dir = '/kaggle/working/vjx'
_vjx_sh.copyfile(_vjx_sub, _vjx_base_path)            # the base's final file: the submission unless every step below succeeds
_vjx_info = {'gamma': _VJX_GAMMA, 'rows': len(test_ids), 'nshard': _VJX_NSHARD}
_vjx_status = 'not_run'
_vjx_ps = []


def _vjx_log(msg):
    print('[vjx %5.2fh] %s' % ((_vjx_time.time() - NB_T0) / 3600, msg), flush=True)


def _vjx_kill(p):
    for step in (lambda: _vjx_os.killpg(p.pid, _vjx_sig.SIGKILL), lambda: p.kill(), lambda: p.wait(timeout=60)):
        try:
            step()
        except Exception as e:
            _vjx_log('kill step: %s' % type(e).__name__)


# 1. the base file must be a complete submission over test_ids
try:
    _vjx_base = _vjx_pd.read_csv(_vjx_base_path, dtype={'StudyInstanceUID': str})
    assert sorted(_vjx_base['StudyInstanceUID']) == sorted(test_ids), 'base rows differ from test_ids'
    assert set(labels) <= set(_vjx_base.columns), 'base columns'
    _vjx_status = 'base_ok'
except Exception as _vjx_e:
    _vjx_status = 'base_failed'
    _vjx_info['error'] = repr(_vjx_e)[:500]
    _vjx_log('base check FAILED: %r' % (_vjx_e,))

# 2. V-JEPA workers under the notebook-time limit; their partial CSVs survive a timeout
if _vjx_status == 'base_ok':
    _vjx_elapsed_h = (_vjx_time.time() - NB_T0) / 3600
    _vjx_info['start_h'] = round(_vjx_elapsed_h, 3)
    _vjx_status = 'skipped_late'
    if _vjx_elapsed_h <= _VJX_LATEST_START_H:
        _vjx_t = _vjx_time.time()
        try:
            _vjx_txt = _vjx_zlib.decompress(_vjx_b64.b64decode(_VJX_WORKER[0])).decode('utf-8')
            if _vjx_hl.sha256(_vjx_txt.encode('utf-8')).hexdigest() != _VJX_WORKER[1]:
                raise RuntimeError('worker payload sha256 differs')
            _vjx_os.makedirs(_vjx_dir, exist_ok=True)
            with open('/tmp/vjx_run.py', 'w', encoding='utf-8') as _vjx_fh:
                _vjx_fh.write(_vjx_txt)
            _vjx_limit = (_VJX_END_H - _vjx_elapsed_h) * 3600
            _vjx_log('V-JEPA stage: %d worker(s), limit %.0f min' % (_VJX_NSHARD, _vjx_limit / 60))
            for _vjx_k in range(_VJX_NSHARD):
                _vjx_ps.append(_vjx_sp.Popen([_vjx_sys.executable, '-u', '/tmp/vjx_run.py', str(_vjx_k), str(_VJX_NSHARD),
                                              '%s/shard%d.csv' % (_vjx_dir, _vjx_k)],
                                             env=dict(_vjx_os.environ, CUDA_VISIBLE_DEVICES=str(_vjx_k)), start_new_session=True))
            _vjx_rcs = []
            for _vjx_p in _vjx_ps:
                try:
                    _vjx_rcs.append(_vjx_p.wait(timeout=max(1.0, _vjx_limit - (_vjx_time.time() - _vjx_t))))
                except _vjx_sp.TimeoutExpired:
                    _vjx_rcs.append(None)
                    _vjx_kill(_vjx_p)
            _vjx_info['worker_rc'] = _vjx_rcs
            _vjx_status = 'ran'
        except Exception as _vjx_e:
            _vjx_status = 'vjx_failed'
            _vjx_info['error'] = repr(_vjx_e)[:500]
            _vjx_log('V-JEPA FAILED: %r' % (_vjx_e,))
            for _vjx_p in _vjx_ps:
                _vjx_kill(_vjx_p)
        _vjx_info['vjx_seconds'] = round(_vjx_time.time() - _vjx_t, 1)

# 3. gather, guard, blend
if _vjx_status == 'ran':
    try:
        _vjx_parts = [_vjx_pd.read_csv(f, dtype={'StudyInstanceUID': str})
                      for f in ('%s/shard%d.csv' % (_vjx_dir, k) for k in range(_VJX_NSHARD)) if _vjx_os.path.isfile(f)]
        if not _vjx_parts:
            raise RuntimeError('no V-JEPA output')
        _vjx_b = _vjx_base.set_index('StudyInstanceUID')
        _vjx_v = _vjx_pd.concat(_vjx_parts).drop_duplicates('StudyInstanceUID').set_index('StudyInstanceUID').reindex(_vjx_b.index)
        _vjx_vv = _vjx_v[labels].values.astype(_vjx_np.float64)
        _vjx_ok = _vjx_np.isfinite(_vjx_vv).all(axis=1) & ((_vjx_vv >= 0) & (_vjx_vv <= 1)).all(axis=1)
        _vjx_info['scored'] = int(_vjx_ok.sum())
        if _vjx_ok.sum() < _VJX_MIN_FRAC * len(_vjx_b):
            raise RuntimeError('V-JEPA scored %d of %d studies' % (_vjx_ok.sum(), len(_vjx_b)))
        _vjx_rho = {}
        if _vjx_ok.sum() >= 3:
            for _vjx_c in labels:
                _vjx_rho[_vjx_c] = round(float(_vjx_pd.Series(_vjx_b[_vjx_c].values[_vjx_ok]).corr(
                    _vjx_pd.Series(_vjx_v[_vjx_c].values[_vjx_ok]), method='spearman')), 4)
        _vjx_info['rho_base_vjx'] = _vjx_rho
        _vjx_info['rho_mean'] = round(float(_vjx_np.nanmean(list(_vjx_rho.values()))), 4) if _vjx_rho else None
        if _vjx_ok.sum() >= _VJX_RHO_MIN_ROWS and not (_vjx_info['rho_mean'] or 0.0) >= _VJX_RHO_FLOOR:
            raise RuntimeError('base / V-JEPA mean rank correlation %r under %r' % (_vjx_info['rho_mean'], _VJX_RHO_FLOOR))
        _vjx_new = _vjx_b.copy()
        for _vjx_c in labels:
            _vjx_a = _vjx_b[_vjx_c].rank(method='average', pct=True).values
            _vjx_r = _vjx_np.full(len(_vjx_b), _vjx_np.nan)
            _vjx_r[_vjx_ok] = _vjx_pd.Series(_vjx_v[_vjx_c].values[_vjx_ok]).rank(method='average', pct=True).values
            _vjx_m = _vjx_np.where(_vjx_ok, (1.0 - _VJX_GAMMA) * _vjx_a + _VJX_GAMMA * _vjx_r, _vjx_a)
            _vjx_new[_vjx_c] = _vjx_pd.Series(_vjx_m).rank(method='average', pct=True).values
        _vjx_vals = _vjx_new[labels].values.astype(_vjx_np.float64)
        assert _vjx_np.isfinite(_vjx_vals).all() and ((_vjx_vals >= 0) & (_vjx_vals <= 1)).all(), 'final out of range'
        _vjx_new.reset_index()[_vjx_base.columns.tolist()].to_csv(_vjx_sub + '.vjx.tmp', index=False)
        _vjx_os.replace(_vjx_sub + '.vjx.tmp', _vjx_sub)
        _vjx_info['max_abs_change_vs_base'] = float(_vjx_np.abs(_vjx_vals - _vjx_b[labels].values.astype(_vjx_np.float64)).max())
        _vjx_status = 'ok'
    except Exception as _vjx_e:
        _vjx_status = 'combine_failed'
        _vjx_info['error'] = repr(_vjx_e)[:500]
        _vjx_log('V-JEPA combine FAILED: %r' % (_vjx_e,))
if _vjx_status != 'ok':
    _vjx_sh.copyfile(_vjx_base_path, _vjx_sub)            # the base's own file
_vjx_info['status'] = _vjx_status
_vjx_info['total_hours'] = round((_vjx_time.time() - NB_T0) / 3600, 3)
with open('/kaggle/working/vjx_receipt.json', 'w', encoding='utf-8') as _vjx_fh:
    _vjx_json.dump(_vjx_info, _vjx_fh, indent=1)
_vjx_log('VJX_DONE ' + _vjx_json.dumps(_vjx_info))
"""


def vjx_cells(tag, base_name, cfg):
    """C21m's two added cells: an intro and the stage-V run cell, which carries VJX_WORKER as a zlib + base64 payload with its sha256."""
    import base64
    import hashlib
    import zlib
    if not (0 < cfg["gamma"] < 1 and cfg["latest_start_h"] < cfg["end_h"] <= 8.6 and 0 < cfg["min_frac"] <= 1):
        raise SystemExit("C21m: 0 < gamma < 1, latest start < end <= 8.6 h, 0 < min_frac <= 1")
    compile(VJX_WORKER, "c21m-worker", "exec")
    b64 = base64.b64encode(zlib.compress(VJX_WORKER.encode("utf-8"), 9)).decode("ascii")
    sha = hashlib.sha256(VJX_WORKER.encode("utf-8")).hexdigest()
    run = VJX_RUN_CELL
    for k, v in (("__GAMMA__", repr(cfg["gamma"])), ("__LATEST__", repr(cfg["latest_start_h"])), ("__END__", repr(cfg["end_h"])),
                 ("__MIN_FRAC__", repr(cfg["min_frac"])), ("__RHO_FLOOR__", repr(cfg["rho_floor"])),
                 ("__RHO_MIN_ROWS__", repr(cfg["rho_min_rows"])), ("__NSHARD__", repr(cfg["nshard"])),
                 ("__W_B64__", b64), ("__W_SHA__", sha)):
        if run.count(k) != 1:
            raise SystemExit(f"C21m: template placeholder {k} found {run.count(k)} times")
        run = run.replace(k, v)
    compile(run, "c21m-run", "exec")
    bad = sorted(n for n in module_level_names(run) if not n.startswith("_vjx_") and not n.startswith("_VJX_"))
    if bad:
        raise SystemExit(f"the V-JEPA run cell binds {bad} at notebook scope (only _vjx_* names are allowed)")
    intro = VJX_INTRO
    for k, v in (("__TAG__", tag), ("__BASE__", base_name), ("__GAMMA__", "%.2f" % cfg["gamma"]),
                 ("__LATEST__", "%.1f" % cfg["latest_start_h"]), ("__END__", "%.1f" % cfg["end_h"]),
                 ("__MINPCT__", "%d" % round(100 * cfg["min_frac"])), ("__RHO__", "%.2f" % cfg["rho_floor"])):
        intro = intro.replace(k, v)
    return [{"id": "c21m-intro", "cell_type": "markdown", "metadata": {}, "source": intro},
            {"id": "c21m-run", "cell_type": "code", "metadata": {}, "outputs": [], "execution_count": None, "source": run}], \
        {"worker_sha256": sha, "dataset": VJX_SLUG}

# nartaa's launcher: compact_infer.main() recompiled from its own source with three replacements (score every K1_CROPS crop, average
# the probabilities).  compact_infer.py on disk is unchanged: its sha256 is checked by the script and echoed in every receipt.  B's
# CPU probe imports the launcher, so a patch that no longer applies fails the probe before any GPU work (s6fall fails -> V2).
_C17_LAUNCHER_BLOCK = r'''

# [C17] crop TTA (src/build_anchor_copy.py --anchor 09586s6fx; experiments.md 2026-10-18 "Endgame research"): this model trained on a
# random 288 crop of its 320 render (one offset in 0..32 px per axis per batch, compact_train.augment); compact_infer scores the
# centre crop only.  K1_CROPS = "oy,ox;oy,ox;..." scores every listed crop and averages the 12 probabilities.  compact_infer.py on
# disk is unchanged (its sha256 is checked); main() is recompiled from its own source with three replacements, each found once.
# K1_CROP_DUMP (the timing kernel only): each study's per-crop probabilities go to <dir>/crops_<pid>.jsonl.
C17_MAIN_PATCHES = [
    ("    def score(images, meta):\n        img = images.to(dev, non_blocking=True)[None, :, :, off:off + a.crop, off:off + a.crop]\n",
     "    def c17_score_at(images, meta, oy, ox):\n        img = images.to(dev, non_blocking=True)[None, :, :, oy:oy + a.crop, ox:ox + a.crop]\n"),
    ("    fh = open(out / \"predictions.csv\", \"w\", encoding=\"utf-8\", newline=\"\")\n",
     "    c17_crops = [tuple(int(v) for v in s.split(\",\")) for s in os.environ[\"K1_CROPS\"].split(\";\")]\n"
     "    if not c17_crops or any(len(c) != 2 or not all(0 <= v <= a.size - a.crop for v in c) for c in c17_crops):\n"
     "        raise SystemExit(\"K1_CROPS outside the %d px render: %r\" % (a.size, c17_crops))\n"
     "    print(\"C17_CROPS\", json.dumps(c17_crops), flush=True)\n"
     "    c17_dump_dir, c17_last = os.environ.get(\"K1_CROP_DUMP\"), []\n"
     "\n"
     "    @torch.inference_mode()\n"
     "    def score(images, meta):\n"
     "        ps = []\n"
     "        for oy, ox in c17_crops:\n"
     "            zm, zs = c17_score_at(images, meta, oy, ox)\n"
     "            if not torch.isfinite(zm).all():\n"
     "                return zm, zs                       # handle() makes it a fallback, as for the centre crop alone\n"
     "            ps.append(torch.sigmoid(zm.double()))\n"
     "        c17_last[:] = ps\n"
     "        p = torch.stack(ps).mean(0).clamp(1e-12, 1 - 1e-12)\n"
     "        z = torch.log(p / (1 - p))\n"
     "        return z, z[None]                           # logits.json keeps its (1, 12) shape; sigmoid(z) = the crop mean\n"
     "\n"
     "    def c17_dump(uid):\n"
     "        if c17_dump_dir and c17_last:\n"
     "            with open(os.path.join(c17_dump_dir, \"crops_%d.jsonl\" % os.getpid()), \"a\", encoding=\"utf-8\") as fh_:\n"
     "                fh_.write(json.dumps({\"uid\": uid, \"crops\": c17_crops, \"p\": [x.float().cpu().tolist() for x in c17_last]}) + \"\\n\")\n"
     "        c17_last[:] = []\n"
     "\n"
     "    fh = open(out / \"predictions.csv\", \"w\", encoding=\"utf-8\", newline=\"\")\n"),
    ("            zmean, zs = score(item[\"images\"], item[\"meta\"])\n",
     "            zmean, zs = score(item[\"images\"], item[\"meta\"])\n            c17_dump(uid)\n"),
]
if os.environ.get("K1_CROPS"):
    import inspect
    _c17_src = inspect.getsource(CI.main)
    for _c17_old, _c17_new in C17_MAIN_PATCHES:
        if _c17_src.count(_c17_old) != 1:
            raise RuntimeError("K1 C17: a compact_infer.main patch target was found %d times" % _c17_src.count(_c17_old))
        _c17_src = _c17_src.replace(_c17_old, _c17_new)
    exec(compile(_c17_src, CI.__file__, "exec"), CI.__dict__)
'''


def c17_s6fall_patches(crops):
    """C17's patches on top of patch_s6fall(): traps 74 P3 (a notebook-time guard, a deadline keeps the scored studies, the failure
    cap 0.25 % -> 50 %) and the crop TTA (a default K1_CROPS + the launcher block above)."""
    crops_s = ";".join("%d,%d" % c for c in crops)
    return [
        ('GUARD_S = GUARD_MAX_S + (PARITY_TIMEOUT_S if MODE == "parity" else 0)\n',
         '# [C17] the notebook passes its own time left (traps 74 P3), so B\'s deadline comes before the notebook\'s kill\n'
         'GUARD_S = min(GUARD_MAX_S, int(os.environ.get("K1_GUARD_S", GUARD_MAX_S))) + (PARITY_TIMEOUT_S if MODE == "parity" else 0)\n'),
        ("B_FAIL_MIN, B_FAIL_FRAC = 3, 0.0025    # hidden test only",
         "B_FAIL_MIN, B_FAIL_FRAC = 3, 0.5    # [C17] was 0.0025: the notebook re-blends partial coverage (traps 74 P3). Hidden test only"),
        ('_RECEIPT = {"mode": MODE, "build_id": K1_BUILD_ID, "run": _RUN, "stages": {}, "gpu": {}}\n',
         '_RECEIPT = {"mode": MODE, "build_id": K1_BUILD_ID, "run": _RUN, "stages": {}, "gpu": {}}\n'
         'C17_CROPS = "%s"   # [C17] crop TTA: (oy, ox) offsets of the 288 crop in the 320 render; 16,16 = the centre\n'
         'os.environ.setdefault("K1_CROPS", C17_CROPS)          # read by the B launcher (b_env copies os.environ)\n'
         '_RECEIPT["c17"] = {"crops": os.environ["K1_CROPS"], "guard_s": GUARD_S, "b_fail_frac": B_FAIL_FRAC}\n' % crops_s),
        ("    if timed or deadline_named(fallback):\n"
         "        # the deadline came first: a retry cannot reach them either (review m4); no allowance ever covers a deadline.  A receipt\n"
         "        # that does not state timed_out False counts as one (collect_part already refuses it; this holds even without it)\n"
         "        raise K1Error(\"%s: compact_infer reached its deadline (budget %ds; timed_out not False in part(s) %s); %d studies have no \"\n"
         "                      \"prediction: %s\" % (tag, budget_s, timed, len(fallback), dict(list(fallback.items())[:5])))\n"
         "    retry = None\n"
         "    if fallback:\n",
         "    c17_cut = bool(timed or deadline_named(fallback))\n"
         "    if c17_cut:\n"
         "        # [C17] (traps 74 P3) the deadline came first: the studies scored so far are kept and every study without a prediction\n"
         "        # becomes a failed study (no retry: the deadline leaves no time for one), up to the raised cap below\n"
         "        log(\"[%s] [C17] compact_infer reached its deadline (budget %ds; part(s) %s): %d studies without a prediction become \"\n"
         "            \"failed studies, no retry\" % (tag, budget_s, timed, len(fallback)))\n"
         "    retry = None\n"
         "    if fallback and not c17_cut:\n"),
        ("        if rec2.get(\"timed_out\") is not False or deadline_named(fb2):\n"
         "            # review C1: a retry that reaches its deadline is a HARD ERROR (PREREG section 5.1), never a capped B failure: the\n"
         "            # allowance below covers only studies that still fail after a retry that completed within its deadline\n"
         "            raise K1Error(\"%s: compact_infer reached its deadline in the retry (%.0fs left at its start; timed_out %r); %d of %d \"\n"
         "                          \"retried studies have no prediction: %s\" % (tag, left, rec2.get(\"timed_out\"), len(fb2), len(rids),\n"
         "                                                                       dict(list(fb2.items())[:5])))\n",
         "        if rec2.get(\"timed_out\") is not False or deadline_named(fb2):\n"
         "            # [C17] (traps 74 P3) a retry that reaches its deadline keeps what it scored; the rest stay failed studies\n"
         "            log(\"[%s] [C17] the retry reached its deadline (%.0fs left at its start): %d of %d retried studies stay failed\" % (\n"
         "                tag, left, len(fb2), len(rids)))\n"),
        ("\n\nCI.CStudies = LocalCStudies\n", _C17_LAUNCHER_BLOCK + "\n\nCI.CStudies = LocalCStudies\n"),
    ]


def patch_c17(stage_src, crops):
    for old, new in c17_s6fall_patches(crops):
        if stage_src.count(old) != 1:
            raise SystemExit(f"s6fall script (C17): patch target found {stage_src.count(old)} times: {old[:70]!r}")
        stage_src = stage_src.replace(old, new)
    compile(stage_src, "s6fall_run.py (C17)", "exec")
    return stage_src


S6X17_INTRO = """# Anchor9586s6fx: C14 with the audit's fixes and s6fall's crop TTA (candidates.md C17)

Built by `src/build_anchor_copy.py --anchor __TAG__`. Every cell above is heliosli's V2 notebook as `rsna-knee-anchor9586` runs it
(#100)__EXTRA__. The next cell writes nartaa's public s6fall script as C14 runs it (heliosli's S6 architecture retrained on JEV labels;
component A, his OAI-trained CoAtNet, not run; the public test ids removed) with four changes: it scores __NCROPS__ crops of its 320 px
render (the 288 crop it trained on at random offsets, __CROPS__) and averages their probabilities; its hang guard takes the time this
notebook has left; a deadline keeps the studies scored so far; and up to half the studies may fail. The cell after it keeps V2's final
file aside and in place, rebuilds that file from V2's stage-3 and Part B inputs (and keeps V2's file unless the rebuild matches it),
runs s6fall as its own process under its own scratch root and a time limit, and re-blends: ours = __KEEP__ rank(R6) + __SHARE__
rank(s6fall) per finding, each over its own scored studies, then V2's stage-3 weights and Part B weights unchanged; a Raptor row that
exactly repeats another study's row counts as failed. On a skip, a timeout, a failure or a rank-correlation floor miss the
submission is V2's own file.
Credits: heliosli (V2, R6, S6 architecture, inference code, SKM-TEA anatomy model), nartaa (CoAtNets; s6fall weights, CC BY-NC 4.0),
goodpjw2008 (2.5D reader).
"""

S6X17_RUN_CELL = r"""# Anchor9586s6fx: stage S -- C17 = C14 + the 2026-10-18 audit's fixes (traps 74) + s6fall's crop TTA (src/build_anchor_copy.py).
# Reads V2's stage-3 globals (raw, raw224, ours, r384_fail, r224_fail, rap_fail, our_fail, labels, cols, test_ids, W_RAPTOR_C,
# W_RAPTOR, A224, pct_on) and Part B's files; binds only _s6x_* names.  V2's submission.csv stays in place throughout: s6fall runs
# under its own scratch root and the final file replaces V2's in one os.replace.  Any skip / timeout / failure: V2's own file.
import json as _s6x_json, os as _s6x_os, shutil as _s6x_sh, signal as _s6x_sig, subprocess as _s6x_sp, sys as _s6x_sys
import time as _s6x_time
import numpy as _s6x_np
import pandas as _s6x_pd
_S6X_SHARE = __SHARE__
_S6X_LATEST_START_H = __LATEST__
_S6X_END_H = __END__
_S6X_GUARD_LEAD_S = 15 * 60          # s6fall's own hang guard fires this long before the kill below; its B deadline comes earlier
_S6X_RHO_FLOOR, _S6X_RHO_MIN_N = 0.5, 100   # mean rank corr(R6, s6fall) over >= 100 studies both scored; below = s6fall unused
_s6x_sub = '/kaggle/working/submission.csv'
_s6x_v2_path = '/kaggle/working/_v2_final.csv'
_s6x_own_path = '/kaggle/working/_own.csv'
_s6x_root = '/tmp/s6xroot'                     # s6fall's K1_KAGGLE_ROOT: input -> /kaggle/input, its own empty working/
_s6x_s6sub = _s6x_root + '/working/submission.csv'
_s6x_k1 = _s6x_root + '/working/k1'
_s6x_sh.copyfile(_s6x_sub, _s6x_v2_path)
_s6x_info = {'share': _S6X_SHARE, 'rows': len(test_ids)}
_s6x_status = 'not_run'


def _s6x_log(msg):
    print('[s6x %5.2fh] %s' % ((_s6x_time.time() - NB_T0) / 3600, msg), flush=True)


def _s6x_stage3(ro_by_c, o_fail, f384=None, f224=None):
    # V2's cell-8 loop, verbatim but for the ours rank (ro_by_c), its failure mask (o_fail) and the Raptor failure masks (default:
    # V2's own; C17 adds exact-duplicate rows)
    f384 = r384_fail if f384 is None else f384
    f224 = r224_fail if f224 is None else f224
    frap = f384 & f224
    out = _s6x_np.zeros((len(test_ids), len(labels)))
    for j, c in enumerate(labels):
        r384, r224 = pct_on(raw[c].values, ~f384), pct_on(raw224[c].values, ~f224)
        rr = _s6x_np.where(f384, r224, _s6x_np.where(f224, r384, (1.0 - A224) * r384 + A224 * r224))
        ro = ro_by_c[c]
        wr = W_RAPTOR_C.get(c, W_RAPTOR)
        b = wr * rr + (1.0 - wr) * ro
        b[frap] = ro[frap]
        b[o_fail] = rr[o_fail]
        b[frap & o_fail] = 0.5
        out[:, j] = b
    sub_ = _s6x_pd.DataFrame(out.astype(_s6x_np.float32), columns=labels)   # V2 writes stage 3 as float32
    sub_.insert(0, 'StudyInstanceUID', test_ids)
    return sub_[cols]


def _s6x_part_b(stack, own):
    # goodpjw2008's runner, mode "end" with per-finding weights, verbatim
    own = own.set_index('StudyInstanceUID').loc[stack.StudyInstanceUID].reset_index()
    lab = [c for c in stack.columns if c != 'StudyInstanceUID']
    rs, ro = stack[lab].rank(method='average', pct=True), own[lab].rank(method='average', pct=True)
    fin = stack.copy()
    for c in lab:
        w = float(OWN_W_PER_TARGET.get(c, OWN_W)) if OWN_W_PER_TARGET else OWN_W
        fin[c] = ((1 - w) * rs[c] + w * ro[c]).to_numpy()
    fin[lab] = fin[lab].rank(method='average', pct=True)
    return fin


def _s6x_kill(p):
    for _s6x_k in (lambda: _s6x_os.killpg(p.pid, _s6x_sig.SIGKILL), lambda: p.kill(), lambda: p.wait(timeout=60),
                   lambda: _s6x_sp.run(['pkill', '-9', '-f', '/tmp/k1'], check=False, timeout=60)):  # its own-session workers
        try:
            _s6x_k()
        except Exception as _s6x_e:
            _s6x_log('kill step: %s' % type(_s6x_e).__name__)


# 1. SELF_CHECK: rebuild V2's final file (#100) from the inputs in memory; s6fall runs only if the rebuild matches it
try:
    _s6x_v2 = _s6x_pd.read_csv(_s6x_v2_path, dtype={'StudyInstanceUID': str})
    _s6x_ro0 = {c: pct_on(ours[c].values, ~our_fail) for c in labels}
    _s6x_s3 = _s6x_stage3(_s6x_ro0, our_fail)
    _s6x_own = None
    if 'OWN_W_PER_TARGET' in globals() and _s6x_os.path.isfile(_s6x_own_path):
        _s6x_own = _s6x_pd.read_csv(_s6x_own_path, dtype={'StudyInstanceUID': str})
    _s6x_diff = {}
    for _s6x_nm, _s6x_cand in (('stage3+partB', None if _s6x_own is None else _s6x_part_b(_s6x_s3, _s6x_own)),
                               ('stage3', _s6x_s3)):
        if _s6x_cand is not None and _s6x_cand['StudyInstanceUID'].tolist() == _s6x_v2['StudyInstanceUID'].tolist():
            _s6x_diff[_s6x_nm] = float(_s6x_np.abs(_s6x_cand[labels].values.astype(_s6x_np.float64)
                                                   - _s6x_v2[labels].values.astype(_s6x_np.float64)).max())
    _s6x_info['selfcheck_max_abs_diff'] = _s6x_diff
    _s6x_path_b = [k for k, v in _s6x_diff.items() if v < 1e-6]
    if not _s6x_path_b:
        raise RuntimeError('V2 rebuild does not match its own file: %s' % _s6x_diff)
    _s6x_use_b = _s6x_path_b[0] == 'stage3+partB'
    _s6x_info['part_b_applied'] = _s6x_use_b
    # traps 74: a Raptor stage scores a blank volume as a real study; a row that exactly repeats another study's row counts as failed
    _s6x_d384 = raw.duplicated(keep=False).values & ~r384_fail
    _s6x_d224 = raw224.duplicated(keep=False).values & ~r224_fail
    _s6x_f384, _s6x_f224 = r384_fail | _s6x_d384, r224_fail | _s6x_d224
    _s6x_info['raptor_duplicate_rows'] = {'384': int(_s6x_d384.sum()), '224': int(_s6x_d224.sum())}
    _s6x_status = 'selfcheck_ok'
    _s6x_log('SELF_CHECK ok %s | Raptor duplicate rows %s' % (_s6x_json.dumps(_s6x_diff), _s6x_info['raptor_duplicate_rows']))
except Exception as _s6x_e:
    _s6x_status = 'selfcheck_failed'
    _s6x_info['error'] = repr(_s6x_e)[:500]
    _s6x_log('SELF_CHECK FAILED: %r' % (_s6x_e,))

# 2. s6fall as its own process under its own scratch root, under the notebook-time limit
if _s6x_status == 'selfcheck_ok':
    _s6x_elapsed_h = (_s6x_time.time() - NB_T0) / 3600
    _s6x_status = 'skipped_late'
    if _s6x_elapsed_h <= _S6X_LATEST_START_H:
        _s6x_limit = (_S6X_END_H - _s6x_elapsed_h) * 3600
        _s6x_guard = int(max(600, _s6x_limit - _S6X_GUARD_LEAD_S))
        _s6x_info['s6f_guard_s'] = _s6x_guard
        _s6x_log('s6fall stage: start, limit %.0f min, its guard %.0f min' % (_s6x_limit / 60, _s6x_guard / 60))
        _s6x_t = _s6x_time.time()
        _s6x_rc = 'not started'
        try:
            _s6x_os.makedirs(_s6x_root + '/working', exist_ok=True)
            if not _s6x_os.path.lexists(_s6x_root + '/input'):
                _s6x_os.symlink('/kaggle/input', _s6x_root + '/input')
            _s6x_env = dict(_s6x_os.environ, K1_KAGGLE_ROOT=_s6x_root, K1_GUARD_S=str(_s6x_guard))
            _s6x_p = _s6x_sp.Popen([_s6x_sys.executable, '-u', '/tmp/s6fall_run.py'], start_new_session=True, env=_s6x_env)
            try:
                _s6x_rc = _s6x_p.wait(timeout=_s6x_limit)
            except _s6x_sp.TimeoutExpired:
                _s6x_rc = None
                _s6x_kill(_s6x_p)
        except Exception as _s6x_e:
            _s6x_info['error'] = repr(_s6x_e)[:500]
        _s6x_info['s6f_seconds'] = round(_s6x_time.time() - _s6x_t, 1)
        _s6x_info['s6f_rc'] = _s6x_rc
        _s6x_status = 'ran' if (_s6x_rc == 0 and _s6x_os.path.isfile(_s6x_s6sub)) else ('timeout' if _s6x_rc is None else 'failed')

# 3. re-blend: ours = (1 - share) rank(R6) + share rank(s6fall), then V2's stage 3 and Part B unchanged
if _s6x_status == 'ran':
    _s6x_tmp = '/kaggle/working/.submission_s6x.tmp'
    try:
        _s6x_s6 = _s6x_pd.read_csv(_s6x_s6sub, dtype={'StudyInstanceUID': str})
        assert sorted(_s6x_s6['StudyInstanceUID']) == sorted(test_ids), 's6fall: rows differ from test.csv'
        assert set(labels) <= set(_s6x_s6.columns), 's6fall: columns differ'
        _s6x_sh.copyfile(_s6x_s6sub, '/kaggle/working/_s6f.csv')
        _s6x_s6 = _s6x_s6.set_index('StudyInstanceUID').loc[test_ids, labels]
        with open(_s6x_k1 + '/blend_receipt.json', encoding='utf-8') as _s6x_fh:   # its failed studies (partial coverage)
            _s6x_bf = set(_s6x_json.load(_s6x_fh)['b_failed_rank_A_only'] or {})
        try:
            with open(_s6x_k1 + '/k1_receipt.json', encoding='utf-8') as _s6x_fh:
                _s6x_info['s6f_c17'] = _s6x_json.load(_s6x_fh).get('c17')
        except Exception as _s6x_e:
            _s6x_log('k1_receipt.json unread (%s)' % type(_s6x_e).__name__)
        _s6x_ok = _s6x_np.array([u not in _s6x_bf for u in test_ids])
        _s6x_r6_ok = ~our_fail
        _s6x_both = _s6x_r6_ok & _s6x_ok
        _s6x_ro, _s6x_rho = {}, {}
        for _s6x_c in labels:
            _s6x_a, _s6x_b = pct_on(ours[_s6x_c].values, _s6x_r6_ok), pct_on(_s6x_s6[_s6x_c].values, _s6x_ok)
            _s6x_ro[_s6x_c] = _s6x_np.where(_s6x_both, (1.0 - _S6X_SHARE) * _s6x_a + _S6X_SHARE * _s6x_b,
                                       _s6x_np.where(_s6x_r6_ok, _s6x_a, _s6x_np.where(_s6x_ok, _s6x_b, _s6x_np.nan)))
            if _s6x_both.sum() >= 3:
                _s6x_rho[_s6x_c] = round(float(_s6x_pd.Series(_s6x_a[_s6x_both]).corr(_s6x_pd.Series(_s6x_b[_s6x_both]),
                                                                                 method='spearman')), 4)
        _s6x_rho_mean = round(float(_s6x_np.mean(list(_s6x_rho.values()))), 4) if _s6x_rho else None
        _s6x_info.update(s6f_failed=len(_s6x_bf), r6_failed=int(our_fail.sum()), rho_r6_s6f=_s6x_rho, rho_mean=_s6x_rho_mean)
        if _s6x_both.sum() >= _S6X_RHO_MIN_N and not (_s6x_rho_mean is not None and _s6x_rho_mean >= _S6X_RHO_FLOOR):
            raise RuntimeError('rank corr(R6, s6fall) mean %s < %.2f over %d studies' % (_s6x_rho_mean, _S6X_RHO_FLOOR,
                                                                                         int(_s6x_both.sum())))
        _s6x_fail = our_fail & ~_s6x_ok
        _s6x_new = _s6x_stage3(_s6x_ro, _s6x_fail, _s6x_f384, _s6x_f224)
        if _s6x_use_b:
            _s6x_new = _s6x_part_b(_s6x_new, _s6x_own)
        _s6x_vals = _s6x_new[labels].values.astype(_s6x_np.float64)
        assert _s6x_new['StudyInstanceUID'].tolist() == _s6x_v2['StudyInstanceUID'].tolist(), 'row order changed'
        assert _s6x_np.isfinite(_s6x_vals).all() and ((_s6x_vals >= 0) & (_s6x_vals <= 1)).all(), 'final out of range'
        _s6x_new[_s6x_v2.columns.tolist()].to_csv(_s6x_tmp, index=False)
        _s6x_os.replace(_s6x_tmp, _s6x_sub)                      # V2's file is replaced in one step, never absent
        _s6x_info.update(both_failed=int(_s6x_fail.sum()),
                         max_abs_change_vs_v2=float(_s6x_np.abs(_s6x_vals - _s6x_v2[labels].values).max()))
        _s6x_status = 'ok'
    except Exception as _s6x_e:
        _s6x_status = 'combine_failed'
        _s6x_info['error'] = repr(_s6x_e)[:500]
        try:
            _s6x_os.remove(_s6x_tmp)
        except OSError:
            pass
if _s6x_status != 'ok':
    _s6x_sh.copyfile(_s6x_v2_path, _s6x_sub)                 # V2's own file (#100)
try:                                                         # s6fall's logs and receipts, for the commit run's output
    if _s6x_os.path.isdir(_s6x_k1):
        _s6x_sh.copytree(_s6x_k1, '/kaggle/working/k1', dirs_exist_ok=True)
except Exception as _s6x_e:
    _s6x_log('k1 copy: %s' % type(_s6x_e).__name__)
_s6x_info['status'] = _s6x_status
_s6x_info['total_hours'] = round((_s6x_time.time() - NB_T0) / 3600, 3)
with open('/kaggle/working/s6x_receipt.json', 'w', encoding='utf-8') as _s6x_fh:
    _s6x_json.dump(_s6x_info, _s6x_fh, indent=1)
_s6x_log('S6X_DONE ' + _s6x_json.dumps(_s6x_info))
"""


def s6fall_cells(cfg, tag="09586s6f", mirror224=False, extra_note=""):
    """C14's two added cells: the patched s6fall script (C11's stage) and the guarded run + re-blend cell.  With cfg["crops"]
    (C17): the script also carries C17's patches, and the run cell is C17's."""
    s6f = F954_ANCHORS["0950s6f"]
    if sha256_file(s6f["notebook"]) != s6f["sha256"]:
        raise SystemExit("s6fall script: sha256 differs from the pull build_fork954.py was written against")
    with open(s6f["notebook"], encoding="utf-8") as fh:
        stage_src = patch_s6fall(fh.read())
    if not 0.0 < cfg["share"] < 1.0 or not cfg["latest_start_h"] < cfg["end_h"] < 8.75:
        raise SystemExit("s6fall: share must be in (0, 1) and latest start < end < 8.75 h")
    c17 = "crops" in cfg
    if c17:
        crops = cfg["crops"]
        if not crops or crops[0] != (16, 16) or len(set(crops)) != len(crops) or any(not 0 <= v <= 32 for c in crops for v in c):
            raise SystemExit("C17 crops: the centre (16, 16) first, distinct, offsets in 0..32")
        stage_src = patch_c17(stage_src, crops)
    run_src = ((S6X17_RUN_CELL if c17 else S6X_RUN_CELL).replace("__SHARE__", repr(cfg["share"]))
               .replace("__LATEST__", repr(cfg["latest_start_h"])).replace("__END__", repr(cfg["end_h"])))
    compile(run_src, "anchor9586s6f-run", "exec")
    bad = sorted(n for n in module_level_names(run_src) if not n.startswith("_s6x_") and not n.startswith("_S6X_"))
    if bad:
        raise SystemExit(f"the s6fall run cell binds {bad} at notebook scope (only _s6x_* names are allowed)")
    if c17:
        intro = (S6X17_INTRO.replace("__KEEP__", "%.1f" % (1.0 - cfg["share"])).replace("__SHARE__", "%.1f" % cfg["share"])
                 .replace("__TAG__", tag).replace("__EXTRA__", extra_note).replace("__NCROPS__", str(len(cfg["crops"])))
                 .replace("__CROPS__", "offsets " + ", ".join("(%d, %d)" % c for c in cfg["crops"])))
    else:
        intro = (S6X_INTRO.replace("__KEEP__", "%.1f" % (1.0 - cfg["share"])).replace("__SHARE__", "%.1f" % cfg["share"])
                 .replace("__TAG__", tag)
                 .replace("__MIRROR__", ", except that the 224 px CoAtNet stage (cell 4) also runs the 384 px stage's anatomical-mirror "
                                        "TTA (C15)" if mirror224 else extra_note))

    def code(cid, src):
        return {"id": cid, "cell_type": "code", "metadata": {}, "outputs": [], "execution_count": None, "source": src}
    pre = "anchor9586s6fx" if c17 else "anchor9586s6f"
    return [{"id": pre + "-intro", "cell_type": "markdown", "metadata": {}, "source": intro},
            code(pre + "-script", "%%writefile /tmp/s6fall_run.py\n" + stage_src),
            code(pre + "-run", run_src)], stage_src


def build(spec):
    if sha256_file(spec["notebook"]) != spec["sha256"]:
        raise SystemExit(f"{spec['notebook']}: sha256 differs from the pull this builder was written against")
    nb = load_notebook(spec["notebook"])
    cells = [json.loads(json.dumps(c)) for c in nb["cells"]]
    txt = [cell_text(c) for c in cells]
    for i, keys in spec["needles"].items():
        for k in keys:
            if k not in txt[i]:
                raise SystemExit(f"anchor cell {i} lacks {k!r}")
    for i, pairs in spec["patches"].items():
        src = txt[i]
        for old, new in pairs:
            if src.count(old) != 1:
                raise SystemExit(f"anchor cell {i}: patch target found {src.count(old)} times: {old!r}")
            src = src.replace(old, new)
        cells[i]["source"] = src
    if spec["patches"].get(2) == C18_PATCHES:        # C18: no 320 px literal may survive in cell 2's code (the v1 placeholders failed on one)
        left = [ln for ln in cell_text(cells[2]).split(chr(10)) if "320" in ln.split("#", 1)[0]]
        if left:
            raise SystemExit(f"C18: cell 2 still holds a 320 literal in code: {left}")
    for i in spec["patches"]:                        # a patched %%writefile cell must still be valid Python
        if txt[i].startswith("%%writefile"):
            src_i = cell_text(cells[i])
            compile(src_i.split(chr(10), 1)[1], f"anchor cell {i}", "exec")
    for c in cells:                                  # a clean copy: no stored outputs from the author's run
        if c["cell_type"] == "code":
            c["outputs"], c["execution_count"] = [], None
    changed = [i for i, (a, b) in enumerate(zip(cells, nb["cells"])) if cell_text(a) != cell_text(b)]
    if sorted(changed) != sorted(spec["patches"]):
        raise SystemExit(f"cells changed {changed}, expected only the patched {sorted(spec['patches'])}")
    provenance = {"builder": "src/build_anchor_copy.py", "source_notebook": spec["notebook"],
                  "source_notebook_sha256": spec["sha256"], "patched_cells": sorted(spec["patches"])}
    if "s6fall" in spec:
        anchor_names = set()
        for c in cells:
            if c["cell_type"] == "code":
                try:
                    anchor_names |= module_level_names(cell_text(c))
                except SyntaxError:                  # a %%writefile / magic cell binds nothing at notebook scope
                    pass
        for need in ("raw", "raw224", "ours", "r384_fail", "r224_fail", "rap_fail", "our_fail", "labels", "cols", "test_ids",
                     "W_RAPTOR_C", "W_RAPTOR", "A224", "pct_on", "NB_T0", "OWN_W_PER_TARGET", "OWN_W"):
            if need not in anchor_names:
                raise SystemExit(f"the s6fall run cell reads {need!r}, which no anchor cell binds")
        extra, stage_src = s6fall_cells(spec["s6fall"], spec.get("tag", "09586s6f"), spec.get("mirror224", False),
                                        spec.get("extra_note", ""))
        cells += extra
        script_patches = ["component A (OAI-trained 0.949 CoAtNet) not run; a B-failed study takes the column-mean rank",
                          "PUBLIC_TEST_IDS removed"]
        if "crops" in spec["s6fall"]:
            script_patches += ["C17: hang guard = the notebook's time left (K1_GUARD_S); a B deadline keeps the scored studies; "
                               "B failure cap 0.25 % -> 50 %", "C17: crop TTA in the B launcher (K1_CROPS, probability mean)"]
        provenance.update(s6fall=dict(spec["s6fall"], script=os.path.basename(F954_ANCHORS["0950s6f"]["notebook"]),
                                      script_sha256=F954_ANCHORS["0950s6f"]["sha256"], script_patches=script_patches))
    if "r7x" in spec:                                # C20: stage R after the s6fall stage
        if "s6fall" not in spec or "crops" in spec["s6fall"]:
            raise SystemExit("C20 builds on C14's s6fall stage (09586s6f-based, no C17 crops)")
        bound = set()
        for c in cells:
            if c["cell_type"] == "code":
                try:
                    bound |= module_level_names(cell_text(c))
                except SyntaxError:
                    pass
        for need in ("ours", "our_fail", "labels", "test_ids", "pct_on", "NB_T0", "_s6x_status", "_s6x_s6", "_s6x_ok",
                     "_s6x_own", "_s6x_use_b", "_s6x_stage3", "_s6x_part_b", "_S6X_SHARE"):
            if need not in bound:
                raise SystemExit(f"the R7 run cell reads {need!r}, which no earlier cell binds")
        extra, r7prov = r7x_cells(cells, spec["r7x"], spec.get("r7x_note", ""))
        cells += extra
        provenance.update(r7x=dict(spec["r7x"], dataset=R7X_SLUG, bundle_id="8c7b666e27dc0e8f", **r7prov))
    if "vjx" in spec:                                # C21m: stage V after every other stage, on the base's final file
        bound = set()
        for c in cells:
            if c["cell_type"] == "code":
                try:
                    bound |= module_level_names(cell_text(c))
                except SyntaxError:
                    pass
        for need in ("test_ids", "labels", "NB_T0"):
            if need not in bound:
                raise SystemExit(f"the V-JEPA run cell reads {need!r}, which no earlier cell binds")
        extra, vjprov = vjx_cells(spec["tag"], spec["vjx"]["base"], spec["vjx"])
        cells += extra
        provenance.update(vjx=dict(spec["vjx"], **vjprov))
    out = {"cells": cells,
           "metadata": {"kernelspec": nb["metadata"].get("kernelspec", {"display_name": "Python 3", "language": "python",
                                                                         "name": "python3"}),
                        "language_info": nb["metadata"].get("language_info", {"name": "python"}),
                        "rsna_anchor_copy": provenance},
           "nbformat": nb.get("nbformat", 4), "nbformat_minor": nb.get("nbformat_minor", 4)}
    meta = {"id": "tiankljucanin/" + spec["slug"], "title": spec["title"], "code_file": spec["slug"] + ".ipynb",
            "language": "python", "kernel_type": "notebook", "is_private": True, "enable_gpu": True, "enable_tpu": False,
            "enable_internet": False, "keywords": [], "dataset_sources": spec["dataset_sources"],
            "kernel_sources": spec["kernel_sources"], "competition_sources": [COMPETITION], "model_sources": [],
            "docker_image": DOCKER_IMAGE, "machine_shape": "NvidiaTeslaT4"}
    return out, meta


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--anchor", choices=sorted(ANCHORS), required=True)
    ap.add_argument("--check", action="store_true", help="build in memory and compare with the files on disk")
    a = ap.parse_args(argv)
    os.chdir(ROOT)
    spec = dict(ANCHORS[a.anchor], tag=a.anchor)
    out, meta = build(spec)
    nb_text, meta_text = dump_json(out), dump_json(meta)
    check_no_competition_ids(nb_text, "the rendered notebook")
    out_dir = os.path.join("kaggle", spec["slug"])
    paths = ((os.path.join(out_dir, spec["slug"] + ".ipynb"), nb_text), (os.path.join(out_dir, "kernel-metadata.json"), meta_text))
    summary = (f"{paths[0][0]}: {len(out['cells'])} cells, patched {sorted(spec['patches']) or 'none'}; "
               f"{len(meta['dataset_sources'])} datasets, {len(meta['kernel_sources'])} kernel sources")
    if a.check:
        ok = all(os.path.exists(p) and open(p, encoding="utf-8").read() == t for p, t in paths)
        print(("check: ok -- " if ok else "check: FAILED -- ") + summary)
        return 0 if ok else 1
    os.makedirs(out_dir, exist_ok=True)
    for p, t in paths:
        with open(p, "w", encoding="utf-8", newline="\n") as f:
            f.write(t)
    print("wrote " + summary)
    return 0


if __name__ == "__main__":
    sys.exit(main())
