"""Build kaggle/rsna-knee-fork954/ from notebook_score_0.954.ipynb + src/kaggle_pipeline.py (candidates.md C5, A14).

The public "rsna-knee-blend-gold (Dataset restored)" notebook (heliosli, public LB 0.954; pulled 2026-10-11) is a rank blend
of two public models: stage 1 runs nartaa's 0.949 CoAtNet-2 (OAI-trained; the C4 / C4b anchor) as a subprocess, stage 2
runs heliosli's compact model S6 (ConvNeXt-small, 4 folds, joint-anchored views, a side classifier; it pip-installs
timm 1.0.25 + pydicom 3.0.1 from its code Dataset first), stage 3 writes `submission.csv` =
`0.7 * rank_pct(raptor) + 0.3 * rank_pct(S6)` per finding (experiments.md 2026-10-11 "The public frontier moved to 0.954").

This builder keeps the anchor's seven cells byte-identical, inserts one clock cell after its two markdown cells, and
appends "our arm" exactly as src/build_fork949.py does (same cells, same fail-soft rules, same blend):

    final = rank_pct((1 - beta) * rank_pct(anchor) + beta * rank_pct(ours))

`--s6-only` (A14) changes ONE line of the anchor, stage 3's `W_RAPTOR, W_OURS = 0.7, 0.3` -> `0.0, 1.0`, and forces
beta 0: `submission.csv` is then S6's per-finding ranks (a study S6 failed on takes the CoAtNet's rank, the anchor's
own fallback). The CoAtNet stage still runs; only its weight is zero.

The kernel pins the anchor's Docker image (`docker_image` in kernel-metadata.json): the image heliosli's run used, and the
one `rsna-knee-fork949` has carried since its creation (traps 54).

    export PYTHONUTF8=1
    .venv/Scripts/python.exe src/build_fork954.py --members v15co v13b3 --beta 0.20      # C5 (B21 leg)
    .venv/Scripts/python.exe src/build_fork954.py --s6-only                              # A14 (S6 alone)
    .venv/Scripts/python.exe src/build_fork954.py --members v15co v13b3 --beta 0.20 --check
"""
import argparse
import ast
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import build_fork949 as f949  # noqa: E402
from build_fork import (cell_text, chunk_literal, dump_json, load_notebook, make_payload,  # noqa: E402
                        prepare_pipeline, selftest_blend, sha256_file, sha256_text)

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

ANCHOR_NOTEBOOK = "notebook_score_0.954.ipynb"
ANCHOR_NOTEBOOK_SHA256 = "1dc9a5df79e2ed03848a89cf5ecdc139dfb76e1079bef1cc428a014f684d9d33"   # as pulled 2026-10-11
PIPELINE = os.path.join("src", "kaggle_pipeline.py")
OUT_DIR = os.path.join("kaggle", "rsna-knee-fork954")
KERNEL_ID = "tiankljucanin/rsna-knee-fork954"
KERNEL_TITLE = "RSNA Knee Fork954"
COMPETITION = "rsna-knee-abnormality-detection"
ANCHOR_DATASETS = ("heliosli/rsna-knee-compact-v1", "nartaa/rsna-knee-publication-swa-weights-20261007",
                   "heliosli/rsna-orthofoundation-fold0-epoch7-20261004")
DOCKER_IMAGE = ("gcr.io/kaggle-private-byod/python@sha256:"
                "2757e0c7d1e0a9cb43da657b97e223c321a98f5014bdf64f44f2f6b083ad2b2f")
ANCHOR_CELL_TYPES = ["markdown", "markdown", "code", "code", "code", "code", "code"]
WEIGHT_LINE = "W_RAPTOR, W_OURS = 0.7, 0.3"
WEIGHT_LINE_S6 = "W_RAPTOR, W_OURS = 0.0, 1.0   # [fork954 --s6-only] A14: S6 alone (the CoAtNet still runs, weight 0)"

MEMBER_SOURCES = dict(f949.MEMBER_SOURCES)
MEMBER_SOURCES["v15co"] = ("tiankljucanin/rsna-knee-ckpt-v15co", "tiankljucanin/timm-convnext-tiny-in12k")


def _swap(text, pairs):
    for old, new in pairs:
        if text.count(old) != 1:
            raise SystemExit(f"fork954: expected exactly one {old!r} in the fork949 template, found {text.count(old)}")
        text = text.replace(old, new)
    return text


FORK_MARKDOWN = _swap(f949.FORK_MARKDOWN, [
    ("## Fork arm — our members blended onto the public 0.949 single model",
     "## Fork arm — our members blended onto the public 0.954 blend (nartaa's CoAtNet 0.7 + heliosli's S6 0.3)"),
    ("The previous cell (the scored 0.949 source, unchanged) has written `submission.csv`.",
     "The previous cell (stage 3 of the 0.954 notebook, unchanged) has written `submission.csv`."),
    ("`submission_fork_anchor_0949.csv`", "`submission_fork_anchor_0954.csv`"),
])

FORK_ARM_TEMPLATE = _swap(f949.FORK_ARM_TEMPLATE, [
    ("# ===================== FORK ARM: ours as a beta-blend on the public 0.949 single model =====================",
     "# ===================== FORK ARM: ours as a beta-blend on the public 0.954 blend =====================      "),
    ("_FORK_ANCHOR = _FORK_WORK / '_anchor_0949.csv'", "_FORK_ANCHOR = _FORK_WORK / '_anchor_0954.csv'"),
    ("_FORK_ANCHOR_PUB = _FORK_WORK / 'submission_fork_anchor_0949.csv'",
     "_FORK_ANCHOR_PUB = _FORK_WORK / 'submission_fork_anchor_0954.csv'"),
    ("'anchor': 'public 0.949 single model (nartaa)'",
     "'anchor': 'public 0.954 blend (heliosli: nartaa CoAtNet 0.7 + S6 0.3)'"),
])

T0_CELL = """# Fork954: start clock for our arm's time budget (added by src/build_fork954.py; changes nothing below)
import time as _fork_time0
_FORK_T0 = _fork_time0.time()
"""

# names the anchor's own cells bind at notebook scope: the arm must not rebind any of them
ANCHOR_GLOBALS = ("T0", "NB_T0", "WORK", "DATA", "TMP", "pred", "env", "rc", "deadline", "receipt", "sub", "blend",
                  "ours", "raw", "rsub", "labels", "cols", "test_ids", "out", "pct", "n")


def check_anchor(nb, s6_only):
    """The anchor must be heliosli's notebook as pulled on 2026-10-11: seven cells, three stages, the 0.7 / 0.3 blend."""
    cells = nb["cells"]
    if [c["cell_type"] for c in cells] != ANCHOR_CELL_TYPES:
        raise SystemExit(f"anchor cell types changed: {[c['cell_type'] for c in cells]}")
    txt = [cell_text(c) for c in cells]
    needles = {
        2: ("%%writefile /tmp/raptor_run.py", f949.ANCHOR_CKPT, f949.ANCHOR_CKPT_SHA256, "RUN_INFERENCE = True"),
        3: ("NB_T0 = time.time()", "/tmp/raptor_run.py", "raptor_sub.csv", "raptor_raw.csv"),
        4: ("ARM = 'E_lrprior'", "BUNDLE_ID = '209a32c8b50a6b1c'", "DEADLINE_HOURS = 8.3", "T0 = NB_T0"),
        5: ("probe_dual.py", "ours_sub.csv", "OURS_STAGE_COMPLETE"),
        6: (WEIGHT_LINE, "sub.to_csv(WORK / 'submission.csv', index=False)", "NOTEBOOK_COMPLETE"),
    }
    for i, keys in needles.items():
        for k in keys:
            if k not in txt[i]:
                raise SystemExit(f"anchor cell {i} lacks {k!r}")
    if txt[6].count(WEIGHT_LINE) != 1:
        raise SystemExit("anchor stage 3: the weight line is not unique")
    # our arm writes these names in /kaggle/working: the anchor must not use any of them
    for name in ("_ours.csv", "ours_infer.py", "ours_infer.log", "fork_diagnostics.json", "_anchor_0954.csv",
                 "submission_fork_"):
        if any(name in t for t in txt):
            raise SystemExit(f"anchor already uses {name!r}: our arm would collide with it")


def render_fork_cells(payload, payload_sha, pipeline_sha, params):
    md = {"id": "fork-markdown", "cell_type": "markdown", "metadata": {}, "source": f949._fill(FORK_MARKDOWN, params)}
    pay = {"id": "fork-payload", "cell_type": "code", "metadata": {}, "outputs": [], "execution_count": None,
           "source": f949._fill(f949.FORK_PAYLOAD_TEMPLATE, params,
                                {"__PAYLOAD__": chunk_literal(payload), "__PAYLOAD_SHA__": payload_sha,
                                 "__PIPELINE_SHA__": pipeline_sha})}
    arm = {"id": "fork-arm", "cell_type": "code", "metadata": {}, "outputs": [], "execution_count": None,
           "source": f949._fill(FORK_ARM_TEMPLATE, params)}
    for c in (pay, arm):
        compile(cell_text(c), c["id"], "exec")
        clash = sorted(module_level_names(cell_text(c)) & set(ANCHOR_GLOBALS))
        if clash:
            raise SystemExit(f"{c['id']} binds {clash} at notebook scope, names the anchor's cells use")
    return [md, pay, arm]


def module_level_names(src):
    """Names a cell binds at notebook scope (assignments, imports, loop / with / except targets, defs, classes)."""
    names = set()

    def targets(node):
        for n in ast.walk(node):
            if isinstance(n, ast.Name):
                names.add(n.id)

    def visit(stmts):
        for s in stmts:
            if isinstance(s, (ast.Assign, ast.AugAssign, ast.AnnAssign)):
                for t in (s.targets if isinstance(s, ast.Assign) else [s.target]):
                    targets(t)
            elif isinstance(s, (ast.Import, ast.ImportFrom)):
                names.update((a.asname or a.name).split(".")[0] for a in s.names)
            elif isinstance(s, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                names.add(s.name)
            elif isinstance(s, (ast.For, ast.AsyncFor)):
                targets(s.target)
                visit(s.body + s.orelse)
            elif isinstance(s, (ast.With, ast.AsyncWith)):
                for it in s.items:
                    if it.optional_vars is not None:
                        targets(it.optional_vars)
                visit(s.body)
            elif isinstance(s, ast.Try):
                for h in s.handlers:
                    if h.name:
                        names.add(h.name)
                    visit(h.body)
                visit(s.body + s.orelse + s.finalbody)
            elif isinstance(s, (ast.If, ast.While)):
                visit(s.body + s.orelse)

    visit(ast.parse(src).body)
    return names


def build_notebook(nb, fork_cells, provenance, s6_only):
    kept = [json.loads(json.dumps(c)) for c in nb["cells"]]
    for a, b in zip(kept, nb["cells"]):
        if json.dumps(a, sort_keys=True) != json.dumps(b, sort_keys=True):
            raise SystemExit("anchor cell changed during copy")
    if s6_only:
        src = cell_text(kept[6]).replace(WEIGHT_LINE, WEIGHT_LINE_S6)
        if WEIGHT_LINE_S6 not in src or WEIGHT_LINE in src:
            raise SystemExit("--s6-only: the weight patch did not apply")
        kept[6]["source"] = src
    t0 = {"id": "fork-t0", "cell_type": "code", "metadata": {}, "outputs": [], "execution_count": None, "source": T0_CELL}
    return {
        "cells": kept[:2] + [t0] + kept[2:] + fork_cells,
        "metadata": {
            "kernelspec": nb["metadata"].get("kernelspec", {"display_name": "Python 3", "language": "python", "name": "python3"}),
            "language_info": nb["metadata"].get("language_info", {"name": "python"}),
            "rsna_fork": provenance,
        },
        "nbformat": nb.get("nbformat", 4),
        "nbformat_minor": nb.get("nbformat_minor", 4),
    }


def build_metadata(params):
    ours = []
    for v in params.members:
        if v not in MEMBER_SOURCES:
            raise SystemExit(f"no Dataset mapping for member {v!r}")
        for s in MEMBER_SOURCES[v]:
            if s and s not in ours:
                ours.append(s)
    return {
        "id": KERNEL_ID,
        "title": KERNEL_TITLE,
        "code_file": "rsna-knee-fork954.ipynb",
        "language": "python",
        "kernel_type": "notebook",
        "is_private": True,
        "enable_gpu": True,
        "enable_tpu": False,
        "enable_internet": False,
        "keywords": [],
        "dataset_sources": [*ANCHOR_DATASETS, *f949.LABEL_DATASETS] + sorted(ours),
        "kernel_sources": [],
        "competition_sources": [COMPETITION],
        "model_sources": [f949.DINOV2_MODEL],
        "docker_image": DOCKER_IMAGE,
        "machine_shape": "NvidiaTeslaT4",
    }


def build(params, s6_only, notebook=ANCHOR_NOTEBOOK, pipeline=PIPELINE):
    if sha256_file(notebook) != ANCHOR_NOTEBOOK_SHA256:
        raise SystemExit(f"{notebook}: sha256 differs from the 2026-10-11 pull")
    nb = load_notebook(notebook)
    check_anchor(nb, s6_only)
    text, pipeline_sha = prepare_pipeline(pipeline, params.members)
    payload = make_payload(text)
    fork_cells = render_fork_cells(payload, sha256_text(text), pipeline_sha, params)
    selftest_blend(fork_cells)
    provenance = {
        "builder": "src/build_fork954.py",
        "source_notebook": os.path.basename(notebook),
        "source_notebook_sha256": ANCHOR_NOTEBOOK_SHA256,
        "s6_only": s6_only,
        "pipeline_sha256": pipeline_sha,
        "payload_sha256": sha256_text(text),
        "members": list(params.members),
        "beta": params.beta,
        "betas_diag": list(params.betas_diag),
        "gate_h": params.gate_h,
        "hard_stop_h": params.hard_stop_h,
        "sec_per_study": params.sec_per_study,
    }
    return build_notebook(nb, fork_cells, provenance, s6_only), build_metadata(params), payload


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--notebook", default=ANCHOR_NOTEBOOK)
    ap.add_argument("--pipeline", default=PIPELINE)
    ap.add_argument("--out", default=OUT_DIR)
    ap.add_argument("--members", nargs="+", default=["v15co", "v13b3"])          # B21, read 0.946 (#79)
    ap.add_argument("--beta", type=float, default=0.20)
    ap.add_argument("--betas-diag", nargs="+", type=float, default=[0.10, 0.30])
    ap.add_argument("--gate-hours", type=float, default=6.0)
    ap.add_argument("--hard-stop-hours", type=float, default=8.4)
    ap.add_argument("--sec-per-study", type=float, default=5.5)
    ap.add_argument("--s6-only", action="store_true", help="A14: stage-3 weights 0.0 / 1.0 and beta 0 (S6 alone)")
    ap.add_argument("--check", action="store_true", help="build in memory and compare with the files on disk")
    a = ap.parse_args(argv)
    os.chdir(ROOT)
    beta = 0.0 if a.s6_only else a.beta
    params = f949.ForkParams(members=tuple(a.members), beta=beta, betas_diag=tuple(a.betas_diag), gate_h=a.gate_hours,
                             hard_stop_h=a.hard_stop_hours, sec_per_study=a.sec_per_study)
    out_nb, meta, payload = build(params, a.s6_only, a.notebook, a.pipeline)
    nb_text, meta_text = dump_json(out_nb), dump_json(meta)
    nb_path = os.path.join(a.out, "rsna-knee-fork954.ipynb")
    meta_path = os.path.join(a.out, "kernel-metadata.json")
    summary = (f"{nb_path}: {len(out_nb['cells'])} cells (7 anchor + 1 clock + 3 fork); "
               f"{len(meta['dataset_sources'])} datasets; payload {len(payload) / 1024:.0f} KB; "
               f"members {list(params.members)} beta {params.beta}" + (" [S6 ONLY: stage-3 weights 0.0 / 1.0]" if a.s6_only else ""))
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
