"""Build kaggle/rsna-knee-fork/ from notebook_score_0.942.ipynb + src/kaggle_pipeline.py (P-27).

The public "DINOsaur V5" notebook (public LB 0.942) trains nothing: cells 0-49 rank-fuse ~35 public
checkpoints and write the scored `submission.csv`; cells 50-51 are a "FineSpacing" residual stage
whose dataset was NOT attached in the scored run (its own log branch says "exact 0.942 anchor
retained"). This builder keeps cells 0-49 byte-identical, drops the FineSpacing stage, and appends
"our arm": our own inference pipeline (MODE="infer", the c02 members) run as a subprocess on cuda:0,
then rank-blended onto the anchor per finding:

    final = rank_pct((1 - beta) * rank_pct(anchor) + beta * rank_pct(ours))

Fail-soft by construction: if the anchor graph used too much of the 8 h budget, or our subprocess
fails, times out or writes an invalid CSV, `submission.csv` stays byte-identical to the anchor.

Deterministic: the same inputs give byte-identical .ipynb and kernel-metadata.json (`--check`).

    export PYTHONUTF8=1
    .venv/Scripts/python.exe src/build_fork.py                       # v08w + v09h, beta 0.20
    .venv/Scripts/python.exe src/build_fork.py --check               # rebuild in memory, diff vs disk
    .venv/Scripts/python.exe src/build_fork.py --beta 0.0             # anchor-only control: our arm is NOT run
    .venv/Scripts/python.exe src/build_fork.py --members v08w v09h v09a \
        --member v09a=tiankljucanin/rsna-knee-ckpt-v09a:tiankljucanin/timm-coatnet-rmlp-1-rw-224
    .venv/Scripts/python.exe src/build_fork.py --anchor-preset parent --beta 0.0   # the flat-0.60 hedge (2026-09-22)

--anchor-preset patches ONE token of the anchor's config cell: its `PRESET = os.environ.get("RSNA_PRESET", "speedy")`
default. "parent" flattens the per-label outer CoAtNet map (LatMen 1.00, ACL / LatOA / Fracture 0.75, MedMen 0.80) to
0.60 -- the anchor author's own hedge preset, the map being "the likeliest place to give back points privately". With
"parent" the kept cells are no longer byte-identical to the 0.942 run (provenance records the preset).
"""
import argparse
import base64
import hashlib
import json
import os
import re
import sys
import zlib
from dataclasses import dataclass, field

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

ANCHOR_NOTEBOOK = "notebook_score_0.942.ipynb"
PIPELINE = os.path.join("src", "kaggle_pipeline.py")
OUT_DIR = os.path.join("kaggle", "rsna-knee-fork")
KERNEL_ID = "tiankljucanin/rsna-knee-fork"
KERNEL_TITLE = "RSNA Knee Fork"
DROPPED_SOURCES = {"dreaddevelopment/raptor-knee-finespacing"}
KEEP_CELLS = range(0, 50)          # the anchor graph, byte-identical
REPLACED_CELLS = (50, 51)          # FineSpacing markdown + code
TAIL_CELLS = (52,)                 # credits
EXPECTED_ANCHOR_COUNTS = (13, 2, 1)   # datasets, kernels, models -- the scored run's 17 minus the competition
COMPETITION = "rsna-knee-abnormality-detection"
ANCHOR_PRESETS = ("speedy", "parent", "halfway", "sparse")   # the anchor's own RSNA_PRESET values; speedy = the 0.942 run
ANCHOR_PRESET_CELL = 2
_PRESET_LINE_RX = re.compile(r'^PRESET = os\.environ\.get\("RSNA_PRESET", "speedy"\)$', re.M)
DINOV2_MODEL = "metaresearch/dinov2/PyTorch/small/1"   # the spelling rsna-knee-infer already mounts

LABELS = ["ACL", "MCL", "Medial Meniscus", "Lateral Meniscus", "Medial OA", "Lateral OA", "PF OA",
          "Effusion", "Synovitis", "Baker's", "Contusion", "Fracture"]

# version -> (checkpoint Dataset, backbone Dataset or None when the backbone is the DINOv2 Model)
MEMBER_SOURCES = {
    "v08w": ("tiankljucanin/rsna-knee-ckpt-v08w", None),
    "v09h": ("tiankljucanin/rsna-knee-ckpt-v09h", "tiankljucanin/timm-coatnet-rmlp-1-rw-224"),
    "v10c": ("tiankljucanin/rsna-knee-ckpt-v10c", "tiankljucanin/timm-coatnet-rmlp-2-rw-384"),
    "v06c": ("tiankljucanin/rsna-knee-ckpt-v06", "tiankljucanin/convnext-tiny-224-hf"),
    "v05a": ("tiankljucanin/rsna-knee-ckpt-v05", None),
    "v05b": ("tiankljucanin/rsna-knee-ckpt-v05", None),
    "v05g": ("tiankljucanin/rsna-knee-ckpt-v05g", None),
}


@dataclass(frozen=True)
class ForkParams:
    members: tuple = ("v08w", "v09h")
    beta: float = 0.20
    anchor_preset: str = "speedy"   # "parent" = the flat-0.60 outer map (hedge build, 2026-09-22)
    betas_diag: tuple = (0.10, 0.30)
    gate_h: float = 7.0            # skip our arm if the anchor graph alone used more than this
    hard_stop_h: float = 8.4       # our subprocess is killed here (their guard raises at 8.0 h from T0)
    sec_per_study: float = 5.5     # our arm's cost on one T4 (v08w + 5 x v09h + c02 decode; #9/#10/#11 reruns)
    extra_member_sources: dict = field(default_factory=dict)


@dataclass(frozen=True)
class Sources:
    datasets: tuple
    kernels: tuple
    models: tuple


def sha256_text(text):
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(8 << 20), b""):
            h.update(block)
    return h.hexdigest()


def load_notebook(path):
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def cell_text(cell):
    src = cell.get("source", "")
    return src if isinstance(src, str) else "".join(src)


def apply_anchor_preset(nb, preset):
    """Patch the anchor's `PRESET` default (cell 2) to one of its own presets. "speedy" leaves the notebook
    untouched. The regex must match exactly once, so a drifted anchor fails loudly instead of running the
    0.942 map under a hedge label."""
    if preset not in ANCHOR_PRESETS:
        raise SystemExit(f"--anchor-preset {preset!r}: choose from {ANCHOR_PRESETS}")
    if preset == "speedy":
        return nb
    cell = nb["cells"][ANCHOR_PRESET_CELL]
    text = cell_text(cell)
    new_text, n = _PRESET_LINE_RX.subn(f'PRESET = os.environ.get("RSNA_PRESET", "{preset}")', text)
    if n != 1:
        raise SystemExit(f"anchor cell {ANCHOR_PRESET_CELL}: the PRESET default line matched {n} times (expected 1)")
    if f'if PRESET == "{preset}":' not in text and preset != "sparse":
        raise SystemExit(f"anchor cell {ANCHOR_PRESET_CELL} has no branch for preset {preset!r}")
    cell["source"] = new_text.splitlines(keepends=True)
    return nb


def assert_finespacing_absent(nb):
    for i in KEEP_CELLS:
        c = nb["cells"][i]
        if c["cell_type"] != "code":
            continue
        if re.search(r"finespacing|v9_full", cell_text(c), flags=re.I):
            raise SystemExit(f"cell {i} references the FineSpacing stage; the anchor graph changed")


_SOURCE_RX = re.compile(r"/kaggle/input/(datasets|notebooks|models)/([A-Za-z0-9_.\-]+(?:/[A-Za-z0-9_.\-]+)*)")


def derive_anchor_sources(nb):
    """Dataset / kernel / model slugs the anchor graph reads, in order of first appearance."""
    datasets, kernels, models = [], [], []
    for i in KEEP_CELLS:
        c = nb["cells"][i]
        if c["cell_type"] != "code":
            continue
        for kind, rest in _SOURCE_RX.findall(cell_text(c)):
            parts = rest.split("/")
            if kind in ("datasets", "notebooks"):
                if len(parts) < 2:
                    continue
                slug = "/".join(parts[:2])
                target = datasets if kind == "datasets" else kernels
            else:
                # /kaggle/input/models/metaresearch/dinov2/pytorch/small/1 (either case of "PyTorch")
                if len(parts) < 5 or parts[0] != "metaresearch" or parts[1] != "dinov2":
                    raise SystemExit(f"unexpected model path in cell {i}: {rest}")
                slug = DINOV2_MODEL
                target = models
            if slug in DROPPED_SOURCES:
                continue
            if slug not in target:
                target.append(slug)
    got = (len(datasets), len(kernels), len(models))
    if got != EXPECTED_ANCHOR_COUNTS:
        raise SystemExit(f"anchor sources {got} != expected {EXPECTED_ANCHOR_COUNTS}: "
                         f"{datasets} {kernels} {models}")
    return Sources(tuple(datasets), tuple(kernels), tuple(models))


def prepare_pipeline(pipeline_path, members):
    """Our pipeline as the infer kernel runs it, with INFER_MEMBERS set for the fork arm."""
    with open(pipeline_path, encoding="utf-8") as f:
        text = f.read()
    subs = [
        (r'^FORCE_SMOKE = True$', 'FORCE_SMOKE = False'),
        (r'^MODE = "auto"$', 'MODE = "infer"'),
        (r'^INFER_MEMBERS = \[.*\]$', 'INFER_MEMBERS = ' + json.dumps(list(members))),
    ]
    for pat, rep in subs:
        text, n = re.subn(pat, rep, text, flags=re.M)
        if n != 1:
            raise SystemExit(f"prepare_pipeline: pattern {pat!r} matched {n} times (expected 1); "
                             f"the config cell of {pipeline_path} drifted")
    compile(text, "ours_infer.py", "exec")
    return text, sha256_text(text)


def make_payload(text):
    return base64.b64encode(zlib.compress(text.encode("utf-8"), 9)).decode("ascii")


def chunk_literal(payload, width=120):
    lines = [payload[i:i + width] for i in range(0, len(payload), width)]
    return "(\n" + "\n".join(f"    '{ln}'" for ln in lines) + "\n)"


FORK_MARKDOWN = """## Fork arm — our c02 members blended onto the 0.942 anchor

The previous cell has written the exact 0.942 `submission.csv`. This cell replaces the original
FineSpacing residual (its dataset was not attached in the scored run, so that stage was a no-op)
with our own inference pipeline (`src/kaggle_pipeline.py`, `MODE="infer"`,
`INFER_MEMBERS = __MEMBERS__`), run as a subprocess on `cuda:0`, then rank-blended per finding:

    final = rank_pct((1 - β) · rank_pct(anchor) + β · rank_pct(ours)),   β = __BETA__

Outputs: `submission.csv` (β = __BETA__) · `submission_fork_anchor_0942.csv` (exact anchor) ·
`submission_fork_beta*.csv` (β = __BETAS_DIAG__) · `_ours.csv` · `ours_infer.log` ·
`fork_diagnostics.json` (per-finding Spearman ρ of ours vs the anchor).

Fail-soft: if the anchor graph used more than __GATE_H__ h, or our subprocess fails, times out or
writes an invalid CSV, `submission.csv` stays byte-identical to the anchor and the log says so.
__PRESET_NOTE__
**β = 0 is the anchor-only control**: our arm is not launched at all and `submission.csv` is the exact
anchor (`fork_diagnostics.json` → `status: anchor_control`) -- it measures what the 0.942 graph scores
from this account, which every β read is relative to.
No DINO / A5 / RadImageNet / Raptor / CoAt arithmetic above is touched.
"""

FORK_PAYLOAD_TEMPLATE = """# Our inference pipeline (src/kaggle_pipeline.py, sed'd to MODE="infer"), zlib + base64.
# Built by src/build_fork.py; pipeline sha256 __PIPELINE_SHA__
_FORK_PAYLOAD_SHA256 = '__PAYLOAD_SHA__'
_FORK_PAYLOAD = __PAYLOAD__
"""

FORK_ARM_TEMPLATE = r"""# ===================== FORK ARM: ours as a beta-blend on the 0.942 anchor =====================
import base64 as _fork_b64, hashlib as _fork_hashlib, os as _fork_os, shutil as _fork_shutil
import signal as _fork_signal, subprocess as _fork_sp, sys as _fork_sys, threading as _fork_thr
import time as _fork_time, traceback as _fork_tb, zlib as _fork_zlib
from pathlib import Path as _ForkPath
import numpy as _fork_np, pandas as _fork_pd

_FORK_WORK = _ForkPath('/kaggle/working')
_FORK_FINAL = _FORK_WORK / 'submission.csv'
_FORK_ANCHOR = _FORK_WORK / '_anchor_0942.csv'
_FORK_ANCHOR_PUB = _FORK_WORK / 'submission_fork_anchor_0942.csv'
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
    print(f'[fork {(_fork_time.time() - T0) / 3600:5.2f}h] {msg}', flush=True)


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


def _fork_write_csv(frame, path):
    tmp = path.with_suffix('.csv.tmp')
    frame.to_csv(tmp, index=False)
    _fork_os.replace(tmp, path)


def _fork_run_ours(timeout_s):
    raw = _fork_zlib.decompress(_fork_b64.b64decode(_FORK_PAYLOAD)).decode('utf-8')
    if _fork_hashlib.sha256(raw.encode('utf-8')).hexdigest() != _FORK_PAYLOAD_SHA256:
        raise RuntimeError('payload sha256 mismatch')
    compile(raw, str(_FORK_SCRIPT), 'exec')
    _FORK_SCRIPT.write_text(raw, encoding='utf-8')
    env = dict(_fork_os.environ)
    env.update(PYTHONUTF8='1', PYTHONUNBUFFERED='1', RSNA_WORKERS='4', CUDA_VISIBLE_DEVICES='0',
               HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1')
    env.pop('CUDNN_CONV_WSCAP_DBG', None)          # theirs; never validated with our models
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


rsna_phase('fork_ours', 'START')
_fork_status, _fork_reason = 'anchor', ''
_fork_diag = {'members': list(_FORK_MEMBERS), 'beta': _FORK_BETA, 'betas_diag': list(_FORK_BETAS_DIAG)}
_fork_anchor = rsna_frame(_fork_pd.read_csv(_FORK_FINAL, dtype={'StudyInstanceUID': str}),
                          _RSNA_TEST_IDS, _RSNA_LABELS, 'fork anchor')
_fork_shutil.copyfile(_FORK_FINAL, _FORK_ANCHOR)
_fork_shutil.copyfile(_FORK_FINAL, _FORK_ANCHOR_PUB)
_fork_anchor_sha = rsna_sha(_FORK_ANCHOR)
_fork_log(f'anchor saved -> {_FORK_ANCHOR.name} rows={len(_fork_anchor)} sha256={_fork_anchor_sha}')
try:
    if _FORK_BETA <= 0.0:
        raise _ForkControl('beta 0: anchor-only control -- our arm is not run, submission.csv = the exact anchor')
    _fork_elapsed = _fork_time.time() - T0
    _fork_est = len(_fork_anchor) * _FORK_SEC_PER_STUDY + 600
    if _fork_elapsed > _FORK_GATE_H * 3600:
        raise TimeoutError(f'anchor graph took {_fork_elapsed / 3600:.2f} h > gate {_FORK_GATE_H} h')
    if _fork_elapsed + _fork_est > _FORK_HARD_STOP_H * 3600:
        raise TimeoutError(f'estimated ours {_fork_est / 60:.0f} min would pass the '
                           f'{_FORK_HARD_STOP_H} h hard stop')
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
    _fork_main = rsna_frame(_fork_blend(_fork_anchor, _fork_ours, _FORK_BETA),
                            _RSNA_TEST_IDS, _RSNA_LABELS, 'fork blend')
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
    _fork_final_sha = rsna_sha(_FORK_FINAL)
    if _fork_status in ('anchor', 'anchor_control') and _fork_final_sha != _fork_anchor_sha:
        _fork_shutil.copyfile(_FORK_ANCHOR, _FORK_FINAL)
        _fork_final_sha = rsna_sha(_FORK_FINAL)
    _fork_diag.update(status=_fork_status, reason=_fork_reason, anchor_sha256=_fork_anchor_sha,
                      submission_sha256=_fork_final_sha, elapsed_h=(_fork_time.time() - T0) / 3600)
    rsna_json(_FORK_DIAG, _fork_diag)
    # The anchor's phase logger takes the status as its second positional argument, so the outcome
    # travels under another keyword (fork v1 died here with "multiple values for 'status'").
    rsna_phase('fork_ours', 'COMPLETE', outcome=_fork_status)
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
        "__PRESET_NOTE__": ("" if params.anchor_preset == "speedy" else
                            f"**Anchor preset `{params.anchor_preset}`** (2026-09-22 hedge build): the anchor's own "
                            f"`PRESET` default is patched from `speedy` to `{params.anchor_preset}` in its config cell, "
                            "which flattens the per-label outer CoAtNet map (LatMen 1.00, ACL / LatOA / Fracture 0.75, "
                            "MedMen 0.80) to 0.60 -- its author's hedge against the map having been tuned on the public "
                            "split. This is therefore NOT the 0.942 graph; it is the candidate for the second final slot."),
    }
    rep.update(extra or {})
    out = template
    for k, v in rep.items():
        out = out.replace(k, v)
    if re.search(r"__[A-Z_]+__", out):
        raise SystemExit("unfilled placeholder in a fork cell template")
    return out


def render_fork_cells(payload, payload_sha, pipeline_sha, params):
    md = {"id": "fork-markdown", "cell_type": "markdown", "metadata": {},
          "source": _fill(FORK_MARKDOWN, params)}
    pay = {"id": "fork-payload", "cell_type": "code", "metadata": {}, "outputs": [], "execution_count": None,
           "source": _fill(FORK_PAYLOAD_TEMPLATE, params,
                           {"__PAYLOAD__": chunk_literal(payload), "__PAYLOAD_SHA__": payload_sha,
                            "__PIPELINE_SHA__": pipeline_sha})}
    arm = {"id": "fork-arm", "cell_type": "code", "metadata": {}, "outputs": [], "execution_count": None,
           "source": _fill(FORK_ARM_TEMPLATE, params)}
    compile(cell_text(pay), "fork-payload", "exec")
    compile(cell_text(arm), "fork-arm", "exec")
    # The anchor's phase logger takes (stage, status, **extra); a `status=` keyword collides with the
    # positional name and raises TypeError inside `finally` -- the one place that must not fail.
    if re.search(r"rsna_phase\([^)]*\bstatus\s*=", cell_text(arm)):
        raise SystemExit("arm cell passes status= to rsna_phase (positional collision, fork v1 failure)")
    return [md, pay, arm]


def selftest_blend(fork_cells, payload_text_sha=None):
    """Execute the arm cell's pure block on synthetic data; exit on any violation."""
    import numpy as np
    import pandas as pd
    src = cell_text(fork_cells[2])
    pure = src[src.index("# --- pure ---"):src.index("# --- end pure ---")]
    ns = {"_fork_np": np, "_fork_pd": pd, "_FORK_LABELS": list(LABELS)}
    exec(compile(pure, "fork-pure", "exec"), ns)
    rng = np.random.default_rng(0)
    n = 50
    uids = [f"u{i:03d}" for i in range(n)]
    anchor = pd.DataFrame(rng.random((n, 12)), columns=LABELS)
    anchor.insert(0, "StudyInstanceUID", uids)
    ours = pd.DataFrame(rng.random((n, 12)), columns=LABELS)
    ours.insert(0, "StudyInstanceUID", list(reversed(uids)))       # shuffled order on purpose
    ours_v = ns["_fork_validate_ours"](ours, uids)
    assert ours_v.StudyInstanceUID.tolist() == uids, "validator must reorder to the anchor order"
    ra = ns["_fork_rankpct"](anchor[LABELS].to_numpy())
    ro = ns["_fork_rankpct"](ours_v[LABELS].to_numpy())
    b0 = ns["_fork_blend"](anchor, ours_v, 0.0)[LABELS].to_numpy()
    b1 = ns["_fork_blend"](anchor, ours_v, 1.0)[LABELS].to_numpy()
    assert np.allclose(b0, ra) and np.allclose(b1, ro), "beta 0 / 1 must reproduce the ranked inputs"
    bm = ns["_fork_blend"](anchor, ours_v, 0.2)[LABELS].to_numpy()
    assert np.isfinite(bm).all() and bm.min() > 0 and bm.max() <= 1, "blend values in (0, 1]"
    # monotone in beta: moving beta toward 1 moves the blend toward ours on average
    d02 = np.abs(bm - ro).mean()
    d05 = np.abs(ns["_fork_blend"](anchor, ours_v, 0.5)[LABELS].to_numpy() - ro).mean()
    assert d05 < d02, "blend must move toward ours as beta grows"
    for bad, why in ((ours.rename(columns={"ACL": "acl"}), "schema"),
                     (ours.iloc[:-1], "UID set"),
                     (ours.assign(MCL=np.nan), "non-finite")):
        try:
            ns["_fork_validate_ours"](bad, uids)
            raise SystemExit(f"validator accepted a bad frame ({why})")
        except ValueError:
            pass
    sp = ns["_fork_spearman"](anchor, ours_v)
    assert set(sp) == set(LABELS) and all(np.isfinite(v) for v in sp.values())
    print("  selftest_blend: ok (rank identities, monotone in beta, validator rejects 3 bad frames)")


def build_notebook(nb, fork_cells, provenance):
    cells = nb["cells"]
    if len(cells) != 53:
        raise SystemExit(f"anchor notebook has {len(cells)} cells, expected 53")
    kept = [json.loads(json.dumps(cells[i])) for i in KEEP_CELLS]
    for i, c in zip(KEEP_CELLS, kept):
        if json.dumps(c, sort_keys=True) != json.dumps(cells[i], sort_keys=True):
            raise SystemExit(f"cell {i} changed during copy")
    tail = [json.loads(json.dumps(cells[i])) for i in TAIL_CELLS]
    out = {
        "cells": kept + fork_cells + tail,
        "metadata": {
            "kernelspec": nb["metadata"].get("kernelspec", {"display_name": "Python 3", "language": "python", "name": "python3"}),
            "language_info": nb["metadata"].get("language_info", {"name": "python"}),
            "rsna_fork": provenance,
        },
        "nbformat": nb.get("nbformat", 4),
        "nbformat_minor": nb.get("nbformat_minor", 4),
    }
    return out


def build_metadata(anchor, params):
    extra = dict(MEMBER_SOURCES)
    extra.update(params.extra_member_sources)
    ours = []
    for v in params.members:
        if v not in extra:
            raise SystemExit(f"no Dataset mapping for member {v!r}; pass --member {v}=<ckpt-ds>[:<backbone-ds>]")
        ckpt, bb = extra[v]
        for s in (ckpt, bb):
            if s and s not in ours and s not in anchor.datasets:
                ours.append(s)
    return {
        "id": KERNEL_ID,
        "title": KERNEL_TITLE,
        "code_file": "rsna-knee-fork.ipynb",
        "language": "python",
        "kernel_type": "notebook",
        "is_private": True,
        "enable_gpu": True,
        "enable_tpu": False,
        "enable_internet": False,
        "keywords": [],
        "dataset_sources": list(anchor.datasets) + sorted(ours),
        "kernel_sources": list(anchor.kernels),
        "competition_sources": [COMPETITION],
        "model_sources": list(anchor.models),
        "machine_shape": "NvidiaTeslaT4",
    }


def dump_json(obj):
    return json.dumps(obj, indent=1, ensure_ascii=False) + "\n"


def build(params, notebook=ANCHOR_NOTEBOOK, pipeline=PIPELINE):
    nb = apply_anchor_preset(load_notebook(notebook), params.anchor_preset)
    assert_finespacing_absent(nb)
    anchor = derive_anchor_sources(nb)
    text, pipeline_sha = prepare_pipeline(pipeline, params.members)
    payload = make_payload(text)
    fork_cells = render_fork_cells(payload, sha256_text(text), pipeline_sha, params)
    selftest_blend(fork_cells)
    provenance = {
        "builder": "src/build_fork.py",
        "source_notebook": os.path.basename(notebook),
        "source_notebook_sha256": sha256_file(notebook),
        "pipeline_sha256": pipeline_sha,
        "payload_sha256": sha256_text(text),
        "members": list(params.members),
        "beta": params.beta,
        "anchor_preset": params.anchor_preset,
        "betas_diag": list(params.betas_diag),
        "gate_h": params.gate_h,
        "hard_stop_h": params.hard_stop_h,
        "sec_per_study": params.sec_per_study,
        "kept_cells": [KEEP_CELLS.start, KEEP_CELLS.stop - 1],
        "replaced_cells": list(REPLACED_CELLS),
    }
    out_nb = build_notebook(nb, fork_cells, provenance)
    meta = build_metadata(anchor, params)
    return out_nb, meta, payload


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--notebook", default=ANCHOR_NOTEBOOK)
    ap.add_argument("--pipeline", default=PIPELINE)
    ap.add_argument("--out", default=OUT_DIR)
    ap.add_argument("--members", nargs="+", default=list(ForkParams.members))
    ap.add_argument("--member", action="append", default=[],
                    help="version=<ckpt-dataset>[:<backbone-dataset>] for a member not in MEMBER_SOURCES")
    ap.add_argument("--beta", type=float, default=ForkParams.beta)
    ap.add_argument("--anchor-preset", choices=ANCHOR_PRESETS, default=ForkParams.anchor_preset,
                    help="the anchor's RSNA_PRESET default; 'parent' = flat 0.60 outer map (hedge build)")
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
            raise SystemExit(f"--member expects version=<ckpt-dataset>[:<backbone-dataset>], got {spec!r}")
        extra[v] = (ckpt, bb or None)
    params = ForkParams(members=tuple(a.members), beta=a.beta, anchor_preset=a.anchor_preset,
                        betas_diag=tuple(a.betas_diag), gate_h=a.gate_hours, hard_stop_h=a.hard_stop_hours,
                        sec_per_study=a.sec_per_study, extra_member_sources=extra)
    out_nb, meta, payload = build(params, a.notebook, a.pipeline)
    nb_text, meta_text = dump_json(out_nb), dump_json(meta)
    nb_path = os.path.join(a.out, "rsna-knee-fork.ipynb")
    meta_path = os.path.join(a.out, "kernel-metadata.json")
    n_src = len(meta["dataset_sources"]) + len(meta["kernel_sources"]) + len(meta["model_sources"]) + 1
    summary = (f"{nb_path}: {len(out_nb['cells'])} cells ({len(KEEP_CELLS)} anchor + 3 fork + {len(TAIL_CELLS)} credits); "
               f"{n_src} sources ({len(meta['dataset_sources'])} datasets); payload {len(payload) / 1024:.0f} KB; "
               f"members {list(params.members)} beta {params.beta} anchor_preset {params.anchor_preset}")
    if a.check:
        ok = True
        for path, text in ((nb_path, nb_text), (meta_path, meta_text)):
            if not os.path.exists(path):
                print(f"  MISSING {path}")
                ok = False
                continue
            with open(path, encoding="utf-8") as f:
                same = f.read() == text
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
