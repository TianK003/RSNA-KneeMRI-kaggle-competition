"""Build kaggle/rsna-knee-fork949/ from notebook_score_0.949.ipynb + src/kaggle_pipeline.py (candidates.md C4).

The public "RSNA Knee 0949 Anatomical Mirror" notebook (nartaa, public LB 0.949) is ONE retrained Raptor CoAtNet-2 @ 384
scored on two views; its checkpoint was trained with 2,399 OAI knees as masked external labels (experiments.md 2026-10-08
"The 0.949 / 0.950 public notebooks, read"). Tian accepted the OAI rule risk on 2026-10-08 to read this fork on the public
LB; whether it is ever a final pick is his separate call.

This builder keeps the anchor's five cells byte-identical (three markdown cells, the checkpoint hash check, the scored
source, which writes `submission.csv`), prepends one cell that records the start time, and appends "our arm": our own
inference pipeline (MODE="infer", the members given) run as a subprocess on cuda:0 after the anchor has finished, then
rank-blended per finding exactly as src/build_fork.py does:

    final = rank_pct((1 - beta) * rank_pct(anchor) + beta * rank_pct(ours))

Fail-soft by construction: if our subprocess fails, times out or writes an invalid CSV, `submission.csv` stays
byte-identical to the anchor. beta 0 = the anchor-only control (our arm is not launched).

    export PYTHONUTF8=1
    .venv/Scripts/python.exe src/build_fork949.py                    # B17 leg, beta 0.35
    .venv/Scripts/python.exe src/build_fork949.py --check            # rebuild in memory, diff vs disk
    .venv/Scripts/python.exe src/build_fork949.py --members v15c v13b3 --beta 0.35
"""
import argparse
import json
import os
import re
import sys
from dataclasses import dataclass, field

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from build_fork import (LABELS, cell_text, chunk_literal, dump_json, load_notebook, make_payload,  # noqa: E402
                        prepare_pipeline, selftest_blend, sha256_file, sha256_text)

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

ANCHOR_NOTEBOOK = "notebook_score_0.949.ipynb"
PIPELINE = os.path.join("src", "kaggle_pipeline.py")
OUT_DIR = os.path.join("kaggle", "rsna-knee-fork949")
KERNEL_ID = "tiankljucanin/rsna-knee-fork949"
KERNEL_TITLE = "RSNA Knee Fork949"
COMPETITION = "rsna-knee-abnormality-detection"
ANCHOR_DATASET = "nartaa/rsna-knee-publication-swa-weights-20261007"
ANCHOR_CKPT = "raptor_ft_alldata_t16_blendjev_oai_d96_r384_swa.pt"
ANCHOR_CKPT_SHA256 = "7e5315dad125b99fc65b340b3de41de628e9be51ff5835355dd61c86472244ef"
ANCHOR_CELLS = 5
DINOV2_MODEL = "metaresearch/dinov2/PyTorch/small/1"
# the infer kernel's label tables: our pipeline then sees exactly the environment rsna-knee-infer gives it
LABEL_DATASETS = ("pilkwang/rsna-knee-llm-labels", "stevenleehans/rsna-knee-llm-report-labels",
                  "lixin73/rsna-knee-llm-report-labels-sol56")

# version -> (checkpoint Dataset, backbone weight Dataset)
MEMBER_SOURCES = {
    "v11a": ("tiankljucanin/rsna-knee-ckpt-v11a", "tiankljucanin/timm-coatnet-rmlp-1-rw-224"),
    "v13r": ("tiankljucanin/rsna-knee-ckpt-v13r", "tiankljucanin/timm-resnet50-a1"),
    "v13e": ("tiankljucanin/rsna-knee-ckpt-v13e", "tiankljucanin/timm-efficientnet-b0-ra"),
    "v13e2": ("tiankljucanin/rsna-knee-ckpt-v13e2", "tiankljucanin/timm-efficientnet-b0-ra"),
    "v13b3": ("tiankljucanin/rsna-knee-ckpt-v13b3", "tiankljucanin/timm-efficientnet-b3-ra2"),
    "v15c": ("tiankljucanin/rsna-knee-ckpt-v15c", "tiankljucanin/timm-convnext-tiny-in12k"),
}


@dataclass(frozen=True)
class ForkParams:
    members: tuple = ("v11a", "v13r", "v13e", "v13b3", "v13e2", "v15c")   # B17, read 0.943 (#65)
    beta: float = 0.35          # a priori: our leg reads 0.006 under the anchor; C2's 0.45 was for an equal-strength leg
    betas_diag: tuple = (0.20, 0.50)
    gate_h: float = 6.0         # skip our arm if the anchor alone used more than this
    hard_stop_h: float = 8.4    # our subprocess is killed here
    sec_per_study: float = 5.5
    extra_member_sources: dict = field(default_factory=dict)


T0_CELL = """# Fork949: start clock for our arm's time budget (added by src/build_fork949.py; changes nothing below)
import time as _fork_time0
_FORK_T0 = _fork_time0.time()
"""

FORK_MARKDOWN = """## Fork arm — our members blended onto the public 0.949 single model

The previous cell (the scored 0.949 source, unchanged) has written `submission.csv`. This cell runs our own inference
pipeline (`src/kaggle_pipeline.py`, `MODE="infer"`, `INFER_MEMBERS = __MEMBERS__`) as a subprocess on `cuda:0`, then
rank-blends per finding:

    final = rank_pct((1 - β) · rank_pct(anchor) + β · rank_pct(ours)),   β = __BETA__

Outputs: `submission.csv` (β = __BETA__) · `submission_fork_anchor_0949.csv` (exact anchor) ·
`submission_fork_beta*.csv` (β = __BETAS_DIAG__) · `_ours.csv` · `ours_infer.log` · `fork_diagnostics.json`.

Fail-soft: if the anchor used more than __GATE_H__ h, or our subprocess fails, times out or writes an invalid CSV,
`submission.csv` stays byte-identical to the anchor and the log says so. β = 0 is the anchor-only control.
"""

FORK_PAYLOAD_TEMPLATE = """# Our inference pipeline (src/kaggle_pipeline.py, sed'd to MODE="infer"), zlib + base64.
# Built by src/build_fork949.py; pipeline sha256 __PIPELINE_SHA__
_FORK_PAYLOAD_SHA256 = '__PAYLOAD_SHA__'
_FORK_PAYLOAD = __PAYLOAD__
"""

FORK_ARM_TEMPLATE = r"""# ===================== FORK ARM: ours as a beta-blend on the public 0.949 single model =====================
import base64 as _fork_b64, gc as _fork_gc, hashlib as _fork_hashlib, json as _fork_json, os as _fork_os
import shutil as _fork_shutil, signal as _fork_signal, subprocess as _fork_sp, sys as _fork_sys
import threading as _fork_thr, time as _fork_time, traceback as _fork_tb, zlib as _fork_zlib
from pathlib import Path as _ForkPath
import numpy as _fork_np, pandas as _fork_pd

_FORK_WORK = _ForkPath('/kaggle/working')
_FORK_FINAL = _FORK_WORK / 'submission.csv'
_FORK_ANCHOR = _FORK_WORK / '_anchor_0949.csv'
_FORK_ANCHOR_PUB = _FORK_WORK / 'submission_fork_anchor_0949.csv'
_FORK_OURS = _FORK_WORK / '_ours.csv'
_FORK_SCRIPT = _FORK_WORK / 'ours_infer.py'
_FORK_LOG = _FORK_WORK / 'ours_infer.log'
_FORK_DIAG = _FORK_WORK / 'fork_diagnostics.json'
_FORK_LABELS = ['ACL', 'MCL', 'Medial Meniscus', 'Lateral Meniscus', 'Medial OA', 'Lateral OA',
                'PF OA', 'Effusion', 'Synovitis', "Baker's", 'Contusion', 'Fracture']
_FORK_MEMBERS = __MEMBERS__
_FORK_BETA = __BETA__
_FORK_BETAS_DIAG = __BETAS_DIAG__
_FORK_GATE_H = __GATE_H__
_FORK_HARD_STOP_H = __HARD_STOP_H__
_FORK_SEC_PER_STUDY = __SEC_PER_STUDY__


class _ForkControl(Exception):   # beta 0: the anchor-only control -- our arm is deliberately not run
    pass


def _fork_log(msg):
    print(f'[fork {(_fork_time.time() - _FORK_T0) / 3600:5.2f}h] {msg}', flush=True)


# --- pure ---
def _fork_rankpct(values):
    return _fork_pd.DataFrame(_fork_np.asarray(values, _fork_np.float64)).rank(
        method='average', pct=True).to_numpy(_fork_np.float64)


def _fork_blend(anchor, ours, beta):
    a = _fork_rankpct(anchor[_FORK_LABELS].to_numpy())
    o = _fork_rankpct(ours[_FORK_LABELS].to_numpy())
    out = anchor.copy()
    out[_FORK_LABELS] = _fork_rankpct((1.0 - float(beta)) * a + float(beta) * o)
    return out


def _fork_spearman(anchor, ours):
    return {lab: float(anchor[lab].corr(ours[lab], method='spearman')) for lab in _FORK_LABELS}


def _fork_validate_ours(frame, uids):
    if frame.columns.tolist() != ['StudyInstanceUID', *_FORK_LABELS]:
        raise ValueError(f'ours: schema {frame.columns.tolist()}')
    frame = frame.copy()
    frame['StudyInstanceUID'] = frame['StudyInstanceUID'].astype(str)
    if frame.StudyInstanceUID.duplicated().any() or set(frame.StudyInstanceUID) != set(map(str, uids)):
        raise ValueError('ours: UID set / duplicates differ from the anchor')
    v = frame[_FORK_LABELS].to_numpy(_fork_np.float64)
    if not _fork_np.isfinite(v).all() or v.min() < 0 or v.max() > 1:
        raise ValueError('ours: non-finite or out-of-range values')
    if len(frame) > 3 and int((v.std(0) < 1e-9).sum()) > len(_FORK_LABELS) // 2:
        raise ValueError('ours: mostly constant columns')
    return frame.set_index('StudyInstanceUID').loc[[str(u) for u in uids]].reset_index()
# --- end pure ---


def _fork_sha(path):
    return _fork_hashlib.sha256(_ForkPath(path).read_bytes()).hexdigest()


def _fork_check_anchor(frame):
    if sorted(frame.columns.tolist()) != sorted(['StudyInstanceUID', *_FORK_LABELS]):
        raise ValueError(f'anchor: schema {frame.columns.tolist()}')
    if frame.StudyInstanceUID.duplicated().any():
        raise ValueError('anchor: duplicate UIDs')
    if not _fork_np.isfinite(frame[_FORK_LABELS].to_numpy(_fork_np.float64)).all():
        raise ValueError('anchor: non-finite values')
    return frame


def _fork_write_csv(frame, path):
    tmp = path.with_suffix('.csv.tmp')
    frame.to_csv(tmp, index=False)
    _fork_os.replace(tmp, path)


def _fork_free_gpu():
    _fork_gc.collect()
    try:
        import torch as _fork_torch
        if _fork_torch.cuda.is_available():
            for _i in range(_fork_torch.cuda.device_count()):
                with _fork_torch.cuda.device(_i):
                    _fork_torch.cuda.empty_cache()
            _fork_log('gpu free after the anchor: ' + ', '.join(
                f'cuda:{_i} {_fork_torch.cuda.mem_get_info(_i)[0] / 2**30:.1f} GiB'
                for _i in range(_fork_torch.cuda.device_count())))
    except Exception as _e:
        _fork_log(f'gpu cleanup skipped: {type(_e).__name__}: {_e}')


def _fork_run_ours(timeout_s):
    raw = _fork_zlib.decompress(_fork_b64.b64decode(_FORK_PAYLOAD)).decode('utf-8')
    if _fork_hashlib.sha256(raw.encode('utf-8')).hexdigest() != _FORK_PAYLOAD_SHA256:
        raise RuntimeError('payload sha256 mismatch')
    compile(raw, str(_FORK_SCRIPT), 'exec')
    _FORK_SCRIPT.write_text(raw, encoding='utf-8')
    env = dict(_fork_os.environ)
    env.update(PYTHONUTF8='1', PYTHONUNBUFFERED='1', RSNA_WORKERS='4', CUDA_VISIBLE_DEVICES='0',
               HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1')
    proc = _fork_sp.Popen([_fork_sys.executable, str(_FORK_SCRIPT)], cwd=str(_FORK_WORK), env=env,
                          stdout=_fork_sp.PIPE, stderr=_fork_sp.STDOUT, text=True, errors='replace',
                          bufsize=1, start_new_session=True)
    killed = []

    def _kill():
        killed.append(True)
        for sig, wait in ((_fork_signal.SIGTERM, 30), (_fork_signal.SIGKILL, 10)):
            try:
                _fork_os.killpg(proc.pid, sig)
                proc.wait(timeout=wait)
                return
            except Exception:
                pass

    timer = _fork_thr.Timer(max(1.0, float(timeout_s)), _kill)
    timer.start()
    try:
        with _FORK_LOG.open('w', encoding='utf-8') as handle:
            for line in proc.stdout:
                handle.write(line)
                print('[ours] ' + line.rstrip('\n'), flush=True)
        rc = proc.wait()
    finally:
        timer.cancel()
    if killed:
        raise TimeoutError(f'ours_infer.py killed at the {_FORK_HARD_STOP_H} h hard stop')
    return rc


_fork_status, _fork_reason = 'anchor', ''
_fork_diag = {'anchor': 'public 0.949 single model (nartaa)', 'members': list(_FORK_MEMBERS), 'beta': _FORK_BETA,
              'betas_diag': list(_FORK_BETAS_DIAG)}
_fork_anchor = _fork_check_anchor(_fork_pd.read_csv(_FORK_FINAL, dtype={'StudyInstanceUID': str}))
_fork_shutil.copyfile(_FORK_FINAL, _FORK_ANCHOR)
_fork_shutil.copyfile(_FORK_FINAL, _FORK_ANCHOR_PUB)
_fork_anchor_sha = _fork_sha(_FORK_ANCHOR)
_fork_log(f'anchor saved -> {_FORK_ANCHOR.name} rows={len(_fork_anchor)} sha256={_fork_anchor_sha}')
try:
    if _FORK_BETA <= 0.0:
        raise _ForkControl('beta 0: anchor-only control -- our arm is not run, submission.csv = the exact anchor')
    _fork_free_gpu()
    _fork_elapsed = _fork_time.time() - _FORK_T0
    _fork_est = len(_fork_anchor) * _FORK_SEC_PER_STUDY + 600
    if _fork_elapsed > _FORK_GATE_H * 3600:
        raise TimeoutError(f'anchor took {_fork_elapsed / 3600:.2f} h > gate {_FORK_GATE_H} h')
    if _fork_elapsed + _fork_est > _FORK_HARD_STOP_H * 3600:
        raise TimeoutError(f'estimated ours {_fork_est / 60:.0f} min would pass the {_FORK_HARD_STOP_H} h hard stop')
    _fork_log(f'launching ours ({", ".join(_FORK_MEMBERS)}) on cuda:0; budget '
              f'{(_FORK_HARD_STOP_H * 3600 - _fork_elapsed) / 60:.0f} min')
    _fork_t = _fork_time.time()
    _fork_rc = _fork_run_ours(_FORK_HARD_STOP_H * 3600 - _fork_elapsed)
    _fork_diag['subprocess'] = {'returncode': int(_fork_rc), 'seconds': _fork_time.time() - _fork_t}
    if _fork_rc != 0:
        raise RuntimeError(f'ours_infer.py exited {_fork_rc} (see ours_infer.log)')
    if not _FORK_FINAL.is_file():
        raise FileNotFoundError('ours_infer.py wrote no submission.csv')
    _fork_os.replace(_FORK_FINAL, _FORK_OURS)
    _fork_ours = _fork_validate_ours(_fork_pd.read_csv(_FORK_OURS, dtype={'StudyInstanceUID': str}),
                                     _fork_anchor.StudyInstanceUID.tolist())
    _fork_diag['spearman_ours_vs_anchor'] = _fork_spearman(_fork_anchor, _fork_ours)
    _fork_diag['spearman_mean'] = float(_fork_np.nanmean(list(_fork_diag['spearman_ours_vs_anchor'].values())))
    _fork_log('spearman(ours, anchor) per finding: '
              + str({k: round(v, 3) for k, v in _fork_diag['spearman_ours_vs_anchor'].items()}))
    for _b in _FORK_BETAS_DIAG:
        _fork_write_csv(_fork_blend(_fork_anchor, _fork_ours, _b),
                        _FORK_WORK / f'submission_fork_beta{int(round(_b * 100)):03d}.csv')
    _fork_main = _fork_blend(_fork_anchor, _fork_ours, _FORK_BETA)
    if not _fork_np.isfinite(_fork_main[_FORK_LABELS].to_numpy(_fork_np.float64)).all():
        raise ValueError('blend: non-finite values')
    _fork_write_csv(_fork_main, _FORK_FINAL)
    _fork_status = f'beta{_FORK_BETA:.2f}'
except _ForkControl as _fork_exc:
    _fork_status, _fork_reason = 'anchor_control', str(_fork_exc)
    _fork_log('ANCHOR-ONLY CONTROL -> ' + _fork_reason)
except BaseException as _fork_exc:
    _fork_reason = f'{type(_fork_exc).__name__}: {str(_fork_exc)[:800]}'
    _fork_log('OUR ARM FAILED/SKIPPED -> anchor retained: ' + _fork_reason)
    _fork_tb.print_exc()
finally:
    if _fork_status == 'anchor' or not _FORK_FINAL.is_file():
        _fork_shutil.copyfile(_FORK_ANCHOR, _FORK_FINAL)
        _fork_status = 'anchor'
    _fork_final_sha = _fork_sha(_FORK_FINAL)
    if _fork_status in ('anchor', 'anchor_control') and _fork_final_sha != _fork_anchor_sha:
        _fork_shutil.copyfile(_FORK_ANCHOR, _FORK_FINAL)
        _fork_final_sha = _fork_sha(_FORK_FINAL)
    _fork_diag.update(status=_fork_status, reason=_fork_reason, anchor_sha256=_fork_anchor_sha,
                      submission_sha256=_fork_final_sha, elapsed_h=(_fork_time.time() - _FORK_T0) / 3600)
    _FORK_DIAG.write_text(_fork_json.dumps(_fork_diag, indent=1), encoding='utf-8')
    _fork_log(f'FINAL submission.csv = {_fork_status} sha256={_fork_final_sha}')
"""


def _fill(template, params, extra=None):
    rep = {
        "__MEMBERS__": json.dumps(list(params.members)),
        "__BETA__": f"{params.beta:.2f}",
        "__BETAS_DIAG__": "(" + ", ".join(f"{b:.2f}" for b in params.betas_diag) + ("," if len(params.betas_diag) == 1 else "") + ")",
        "__GATE_H__": f"{params.gate_h:.1f}",
        "__HARD_STOP_H__": f"{params.hard_stop_h:.1f}",
        "__SEC_PER_STUDY__": f"{params.sec_per_study:.1f}",
    }
    rep.update(extra or {})
    out = template
    for k, v in rep.items():
        out = out.replace(k, v)
    if re.search(r"__[A-Z_]+__", out):
        raise SystemExit("unfilled placeholder in a fork cell template")
    return out


def check_anchor(nb):
    """The anchor must be the 0.949 release as read on 2026-10-08: five cells, the hash check, the scored source."""
    cells = nb["cells"]
    if len(cells) != ANCHOR_CELLS:
        raise SystemExit(f"anchor notebook has {len(cells)} cells, expected {ANCHOR_CELLS}")
    if [c["cell_type"] for c in cells] != ["markdown", "markdown", "markdown", "code", "code"]:
        raise SystemExit("anchor cell types changed")
    hash_cell, src_cell = cell_text(cells[3]), cell_text(cells[4])
    for needle in (ANCHOR_CKPT, ANCHOR_CKPT_SHA256, "RUN_INFERENCE = True"):
        if needle not in hash_cell:
            raise SystemExit(f"anchor cell 3 lacks {needle!r}")
    for needle in ("def main():", 'if __name__ == "__main__":', '"/kaggle/working/submission.csv"',
                   ANCHOR_CKPT, ANCHOR_CKPT_SHA256, "K_EVAL = 94", "INPUT_RES = 320"):
        if needle not in src_cell:
            raise SystemExit(f"anchor cell 4 lacks {needle!r}")


def render_fork_cells(payload, payload_sha, pipeline_sha, params):
    md = {"id": "fork-markdown", "cell_type": "markdown", "metadata": {}, "source": _fill(FORK_MARKDOWN, params)}
    pay = {"id": "fork-payload", "cell_type": "code", "metadata": {}, "outputs": [], "execution_count": None,
           "source": _fill(FORK_PAYLOAD_TEMPLATE, params,
                           {"__PAYLOAD__": chunk_literal(payload), "__PAYLOAD_SHA__": payload_sha,
                            "__PIPELINE_SHA__": pipeline_sha})}
    arm = {"id": "fork-arm", "cell_type": "code", "metadata": {}, "outputs": [], "execution_count": None,
           "source": _fill(FORK_ARM_TEMPLATE, params)}
    for c in (pay, arm):
        compile(cell_text(c), c["id"], "exec")
    for rx in (r"rsna_phase", r"rsna_frame", r"rsna_json", r"rsna_sha", r"_RSNA_", r"(?<![A-Za-z0-9_])T0"):
        if re.search(rx, cell_text(arm)):
            raise SystemExit(f"arm cell matches {rx!r}: a helper of the 0.942 notebook that this anchor lacks")
    return [md, pay, arm]


def build_notebook(nb, fork_cells, provenance):
    kept = [json.loads(json.dumps(c)) for c in nb["cells"]]
    for a, b in zip(kept, nb["cells"]):
        if json.dumps(a, sort_keys=True) != json.dumps(b, sort_keys=True):
            raise SystemExit("anchor cell changed during copy")
    t0 = {"id": "fork-t0", "cell_type": "code", "metadata": {}, "outputs": [], "execution_count": None, "source": T0_CELL}
    return {
        "cells": kept[:3] + [t0] + kept[3:] + fork_cells,
        "metadata": {
            "kernelspec": nb["metadata"].get("kernelspec", {"display_name": "Python 3", "language": "python", "name": "python3"}),
            "language_info": nb["metadata"].get("language_info", {"name": "python"}),
            "rsna_fork": provenance,
        },
        "nbformat": nb.get("nbformat", 4),
        "nbformat_minor": nb.get("nbformat_minor", 4),
    }


def build_metadata(params):
    srcs = dict(MEMBER_SOURCES)
    srcs.update(params.extra_member_sources)
    ours = []
    for v in params.members:
        if v not in srcs:
            raise SystemExit(f"no Dataset mapping for member {v!r}; pass --member {v}=<ckpt-ds>:<backbone-ds>")
        for s in srcs[v]:
            if s and s not in ours:
                ours.append(s)
    return {
        "id": KERNEL_ID,
        "title": KERNEL_TITLE,
        "code_file": "rsna-knee-fork949.ipynb",
        "language": "python",
        "kernel_type": "notebook",
        "is_private": True,
        "enable_gpu": True,
        "enable_tpu": False,
        "enable_internet": False,
        "keywords": [],
        "dataset_sources": [ANCHOR_DATASET, *LABEL_DATASETS] + sorted(ours),
        "kernel_sources": [],
        "competition_sources": [COMPETITION],
        "model_sources": [DINOV2_MODEL],
        "machine_shape": "NvidiaTeslaT4",
    }


def build(params, notebook=ANCHOR_NOTEBOOK, pipeline=PIPELINE):
    nb = load_notebook(notebook)
    check_anchor(nb)
    text, pipeline_sha = prepare_pipeline(pipeline, params.members)
    payload = make_payload(text)
    fork_cells = render_fork_cells(payload, sha256_text(text), pipeline_sha, params)
    selftest_blend(fork_cells)
    provenance = {
        "builder": "src/build_fork949.py",
        "source_notebook": os.path.basename(notebook),
        "source_notebook_sha256": sha256_file(notebook),
        "anchor_checkpoint_sha256": ANCHOR_CKPT_SHA256,
        "pipeline_sha256": pipeline_sha,
        "payload_sha256": sha256_text(text),
        "members": list(params.members),
        "beta": params.beta,
        "betas_diag": list(params.betas_diag),
        "gate_h": params.gate_h,
        "hard_stop_h": params.hard_stop_h,
        "sec_per_study": params.sec_per_study,
    }
    return build_notebook(nb, fork_cells, provenance), build_metadata(params), payload


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--notebook", default=ANCHOR_NOTEBOOK)
    ap.add_argument("--pipeline", default=PIPELINE)
    ap.add_argument("--out", default=OUT_DIR)
    ap.add_argument("--members", nargs="+", default=list(ForkParams.members))
    ap.add_argument("--member", action="append", default=[],
                    help="version=<ckpt-dataset>:<backbone-dataset> for a member not in MEMBER_SOURCES")
    ap.add_argument("--beta", type=float, default=ForkParams.beta)
    ap.add_argument("--betas-diag", nargs="+", type=float, default=list(ForkParams.betas_diag))
    ap.add_argument("--gate-hours", type=float, default=ForkParams.gate_h)
    ap.add_argument("--hard-stop-hours", type=float, default=ForkParams.hard_stop_h)
    ap.add_argument("--sec-per-study", type=float, default=ForkParams.sec_per_study)
    ap.add_argument("--check", action="store_true", help="build in memory and compare with the files on disk")
    a = ap.parse_args(argv)
    os.chdir(ROOT)
    extra = {}
    for spec in a.member:
        v, _, rest = spec.partition("=")
        ckpt, _, bb = rest.partition(":")
        if not v or not ckpt:
            raise SystemExit(f"--member expects version=<ckpt-dataset>:<backbone-dataset>, got {spec!r}")
        extra[v] = (ckpt, bb or None)
    params = ForkParams(members=tuple(a.members), beta=a.beta, betas_diag=tuple(a.betas_diag), gate_h=a.gate_hours,
                        hard_stop_h=a.hard_stop_hours, sec_per_study=a.sec_per_study, extra_member_sources=extra)
    out_nb, meta, payload = build(params, a.notebook, a.pipeline)
    nb_text, meta_text = dump_json(out_nb), dump_json(meta)
    nb_path = os.path.join(a.out, "rsna-knee-fork949.ipynb")
    meta_path = os.path.join(a.out, "kernel-metadata.json")
    summary = (f"{nb_path}: {len(out_nb['cells'])} cells (5 anchor + 1 clock + 3 fork); "
               f"{len(meta['dataset_sources'])} datasets; payload {len(payload) / 1024:.0f} KB; "
               f"members {list(params.members)} beta {params.beta}")
    if a.check:
        ok = True
        for path, text in ((nb_path, nb_text), (meta_path, meta_text)):
            same = os.path.exists(path) and open(path, encoding="utf-8").read() == text
            print(f"  {'same    ' if same else 'DIFFERS '}{path}")
            ok = ok and same
        print(("check: " + ("ok" if ok else "FAILED")) + " -- " + summary)
        return 0 if ok else 1
    os.makedirs(a.out, exist_ok=True)
    with open(nb_path, "w", encoding="utf-8", newline="\n") as f:
        f.write(nb_text)
    with open(meta_path, "w", encoding="utf-8", newline="\n") as f:
        f.write(meta_text)
    print("wrote " + summary)
    return 0


if __name__ == "__main__":
    sys.exit(main())
