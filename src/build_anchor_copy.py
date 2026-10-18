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

S6X_INTRO = """# Anchor9586s6f: nartaa's s6fall inside V2's R6 half (candidates.md C14)

Built by `src/build_anchor_copy.py --anchor 09586s6f`. Every cell above is heliosli's V2 notebook as `rsna-knee-anchor9586` runs it
(#100). The next cell writes nartaa's public s6fall script (heliosli's S6 architecture retrained on JEV labels; component A, his
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


def s6fall_cells(cfg):
    """C14's two added cells: the patched s6fall script (C11's stage) and the guarded run + re-blend cell."""
    s6f = F954_ANCHORS["0950s6f"]
    if sha256_file(s6f["notebook"]) != s6f["sha256"]:
        raise SystemExit("s6fall script: sha256 differs from the pull build_fork954.py was written against")
    with open(s6f["notebook"], encoding="utf-8") as fh:
        stage_src = patch_s6fall(fh.read())
    if not 0.0 < cfg["share"] < 1.0 or not cfg["latest_start_h"] < cfg["end_h"] < 8.75:
        raise SystemExit("s6fall: share must be in (0, 1) and latest start < end < 8.75 h")
    run_src = (S6X_RUN_CELL.replace("__SHARE__", repr(cfg["share"])).replace("__LATEST__", repr(cfg["latest_start_h"]))
               .replace("__END__", repr(cfg["end_h"])))
    compile(run_src, "anchor9586s6f-run", "exec")
    bad = sorted(n for n in module_level_names(run_src) if not n.startswith("_s6x_") and not n.startswith("_S6X_"))
    if bad:
        raise SystemExit(f"the s6fall run cell binds {bad} at notebook scope (only _s6x_* names are allowed)")
    intro = (S6X_INTRO.replace("__KEEP__", "%.1f" % (1.0 - cfg["share"])).replace("__SHARE__", "%.1f" % cfg["share"]))

    def code(cid, src):
        return {"id": cid, "cell_type": "code", "metadata": {}, "outputs": [], "execution_count": None, "source": src}
    return [{"id": "anchor9586s6f-intro", "cell_type": "markdown", "metadata": {}, "source": intro},
            code("anchor9586s6f-script", "%%writefile /tmp/s6fall_run.py\n" + stage_src),
            code("anchor9586s6f-run", run_src)], stage_src


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
        extra, stage_src = s6fall_cells(spec["s6fall"])
        cells += extra
        provenance.update(s6fall=dict(spec["s6fall"], script=os.path.basename(F954_ANCHORS["0950s6f"]["notebook"]),
                                      script_sha256=F954_ANCHORS["0950s6f"]["sha256"],
                                      script_patches=["component A (OAI-trained 0.949 CoAtNet) not run; a B-failed study "
                                                      "takes the column-mean rank", "PUBLIC_TEST_IDS removed"]))
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
    spec = ANCHORS[a.anchor]
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
