"""Build kaggle/rsna-knee-fork954/ (or -fork956/) from heliosli's public blend notebook + src/kaggle_pipeline.py
(candidates.md C5, A14, C6 on the 0.954 notebook; C7, C8 on the 0.956 notebook).

heliosli's public blend notebooks are rank blends of two public models: stage 1 runs nartaa's 0.949 CoAtNet-2 (OAI-trained;
the C4 / C4b anchor) as a subprocess, stage 2 runs one of heliosli's compact models (ConvNeXt-small, 4 folds, knee-centred
views, a side classifier; it pip-installs timm 1.0.25 + pydicom 3.0.1 from its code Dataset first, traps 68), stage 3 writes
`submission.csv` = `w * rank_pct(raptor) + (1 - w) * rank_pct(compact)` per finding. `--anchor` picks the notebook:

    0954  "rsna-knee-blend-gold (Dataset restored)", public LB 0.954, pulled 2026-10-11: CoAtNet 0.7 + S6 0.3
          (experiments.md 2026-10-11 "The public frontier moved to 0.954")            -> kaggle/rsna-knee-fork954/
    0956  "≈0.957 | Latest dataset| score in one click!", public LB 0.956, pulled 2026-10-13: CoAtNet 0.5 + R7 0.5
          (R7 = S6's recipe + a fastMRI auxiliary loss + stackers; experiments.md 2026-10-13
          "The public frontier moved to 0.956")                                       -> kaggle/rsna-knee-fork956/
    0956v "RSNA Knee: One Unused Checkpoint Lifts the 0.956" (varshithakolupuri), public LB 0.956, pulled 2026-10-14:
          the 0956 notebook's stage 1 / 2 cells byte for byte, plus a gated stage 2b running nartaa's second checkpoint
          (224 px, 0.8 centre crop, OAI-trained like the 384 one) and a stage 3 that mixes it at 35 % of the Raptor half
          (experiments.md 2026-10-14 "Submissions #89 / #90"; candidates.md C9)            -> kaggle/rsna-knee-fork956/

The two notebooks share cells 1-3 byte for byte (nartaa's stage) and the stage contract (`raptor_sub.csv` / `raptor_raw.csv`,
`ours_sub.csv`, the stage-3 blend).

This builder keeps the anchor's seven cells byte-identical, inserts one clock cell after its two markdown cells, and
appends "our arm" exactly as src/build_fork949.py does (same cells, same fail-soft rules, same blend):

    final = rank_pct((1 - beta) * rank_pct(anchor) + beta * rank_pct(ours))

`--s6-only` (A14; 0954 only) changes ONE line of the anchor, stage 3's `W_RAPTOR, W_OURS = 0.7, 0.3` -> `0.0, 1.0`, and
forces beta 0: `submission.csv` is then S6's per-finding ranks (a study S6 failed on takes the CoAtNet's rank, the anchor's
own fallback). The CoAtNet stage still runs; only its weight is zero. The 0956 notebook adds per-finding overrides
(`W_RAPTOR_C`) that one line would not reach, so the mode refuses it.

`--compact-anchor` (alias `--s6-anchor`; C6 on 0954, C8 on 0956: OAI-free pick-1 candidates) drops stage 1 (the
OAI-trained CoAtNet) and stage 3 altogether, does not mount nartaa's Dataset, defines `NB_T0` in the clock cell (stage 2
reads it), and replaces stage 3 with one cell that writes the compact model's own `ours_sub.csv` as `submission.csv`; our
arm then blends onto it at beta. Stage 2's two cells stay byte-identical. On 0956, stage 2 runs with
`--max-fallback-fraction 1.0` (stage 3 used to give an unscored study the CoAtNet's rank) and mean-fills such studies, so the
replacement cell counts them, prints `R7_FALLBACK`, and stops above 50 % (stage 3's own guard).

The kernel pins the anchor's Docker image (`docker_image` in kernel-metadata.json): the image heliosli's runs used, and the
one `rsna-knee-fork949` has carried since its creation (traps 54).

    export PYTHONUTF8=1
    .venv/Scripts/python.exe src/build_fork954.py --members v15co v13b3 --beta 0.20      # C5 (B21 leg)
    .venv/Scripts/python.exe src/build_fork954.py --s6-only                              # A14 (S6 alone)
    .venv/Scripts/python.exe src/build_fork954.py --s6-anchor --members v15c v13b3 --beta 0.45   # C6 (S6 + B18, OAI-free)
    .venv/Scripts/python.exe src/build_fork954.py --anchor 0956 --members v15co v13b3 --beta 0.20                   # C7
    .venv/Scripts/python.exe src/build_fork954.py --anchor 0956 --compact-anchor --members v15c v13b3 --beta 0.45  # C8
    .venv/Scripts/python.exe src/build_fork954.py --anchor 0956v --members v15co v13b3 --beta 0.20 \
        --gate-hours 7.0 --sec-per-study 3.0                                                                     # C9
    .venv/Scripts/python.exe src/build_fork954.py --anchor 0957 --members v15co v13b3 --beta 0.20 \
        --gate-hours 7.5 --sec-per-study 3.0                                                                     # C10
    .venv/Scripts/python.exe src/build_fork954.py --anchor 0958 --members v15co v13b3 --beta 0.20 \
        --gate-hours 7.5 --sec-per-study 3.0                                                                     # C10w
    .venv/Scripts/python.exe src/build_fork954.py --anchor 0950s6f --members v15c v13b3 --beta 0.45              # C11

0957 / 0958 (heliosli, 10-14 / 10-15; experiments.md 2026-10-16 "Submission #91 and the 0.957 / 0.958 frontier") are nine-cell
notebooks: stage 1 (384) and stage 1b (224) before stage 2 (R6 with flip TTA), which runs longer than R7, so C10 / C10w raise the
gate to 7.5 h. 0957 gets one patched line (C_SLUG -> the -w1 copy of R6, traps 72); 0958 is verbatim. Each has its own kernel slug
(rsna-knee-fork957 / -fork958), so their placeholders can run at once. 0950s6f is nartaa's s6fall script (C11, OAI-free): see
ANCHORS["0950s6f"] and patch_s6fall(); kernel rsna-knee-forks6f.

On 0956v our arm still runs after the anchor's stage 3 (it blends onto its submission.csv), so the 224 stage now runs
before it: C9 raises the gate to 7 h and lowers the runtime estimate to 3 s / study (the 8.4 h hard stop is unchanged; a
late or killed arm keeps the anchor, as always). --compact-anchor and --s6-only are refused there (C8 is the 0956 build).
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

PIPELINE = os.path.join("src", "kaggle_pipeline.py")
COMPETITION = "rsna-knee-abnormality-detection"
NARTAA_DATASET = "nartaa/rsna-knee-publication-swa-weights-20261007"     # the OAI-trained CoAtNet: never in --compact-anchor
ORTHO_DATASET = "heliosli/rsna-orthofoundation-fold0-epoch7-20261004"    # heliosli's code + wheels + SKM-TEA anatomy model
DOCKER_IMAGE = ("gcr.io/kaggle-private-byod/python@sha256:"
                "2757e0c7d1e0a9cb43da657b97e223c321a98f5014bdf64f44f2f6b083ad2b2f")
ANCHOR_CELL_TYPES = ["markdown", "markdown", "code", "code", "code", "code", "code"]
# the anchor cell that holds each stage (0954 / 0956: seven cells; 0956v interleaves markdown and adds stage 2b)
STAGE_CELLS = {"raptor_write": 2, "raptor_run": 3, "compact_setup": 4, "compact_run": 5, "blend": 6}
R224_CKPT = "raptor_ft_alldata_t16_blendjev_oai_d96_c08_r224_swa.pt"
R224_CKPT_SHA256 = "3394fdd7e668286956e85af2554bef0386df5cd79d1c1af223b095c828b815d6"

ANCHORS = {
    "0954": {
        "notebook": "notebook_score_0.954.ipynb",
        "sha256": "1dc9a5df79e2ed03848a89cf5ecdc139dfb76e1079bef1cc428a014f684d9d33",   # as pulled 2026-10-11
        "score": "0.954",
        "compact": "S6",
        "compact_dataset": "heliosli/rsna-knee-compact-v1",
        "bundle_id": "209a32c8b50a6b1c",
        "weights": ("0.7", "0.3"),
        "slug": "rsna-knee-fork954",
        "title": "RSNA Knee Fork954",
        "extra_needles": {},
    },
    "0956": {
        "notebook": "notebook_score_0.956.ipynb",
        "sha256": "cb2a0f57962ca9b6668661e34461aad772ed9ed10d576d32772e7c2898f2e315",   # as pulled 2026-10-13
        "score": "0.956",
        "compact": "R7",
        "compact_dataset": "heliosli/rsna-knee-compact-r7cnxfsss",
        "bundle_id": "8c7b666e27dc0e8f",
        "weights": ("0.5", "0.5"),
        "slug": "rsna-knee-fork956",
        "title": "RSNA Knee Fork956",
        # the R7-anchor cell relies on these: stage 2 mean-fills unscored studies and leaves its workers' rows in `pred`
        "extra_needles": {5: ("'--max-fallback-fraction', '1.0'", "pred = TMP / 'prediction'"),
                          6: ("W_RAPTOR_C = {'Medial OA': 0.5, 'Lateral OA': 0.5, 'PF OA': 0.5}",
                              "for f in pred.glob('worker*/predictions.csv'):")},
    },
    "0956v": {
        "notebook": "notebook_score_0.956v.ipynb",
        "sha256": "6406e92300011e48f0534ce74a91f5b7a0629b3771168ab7275f5584f884f36d",   # as pulled 2026-10-14 (= scored v1)
        "score": "0.956",
        "compact": "R7",
        "compact_dataset": "heliosli/rsna-knee-compact-r7cnxfsss",
        "bundle_id": "8c7b666e27dc0e8f",
        "weights": ("0.5", "0.5"),
        "slug": "rsna-knee-fork956",
        "title": "RSNA Knee Fork956",
        # no quote marks here: it lands inside a quoted string in the arm
        "desc": "+ the nartaa 224 px checkpoint at 0.35 of the Raptor half (varshithakolupuri)",
        "cell_types": ["markdown", "markdown", "markdown", "code", "code", "markdown", "code", "code", "markdown",
                       "code", "code", "markdown", "code"],
        "stage_cells": {"raptor_write": 3, "raptor_run": 4, "compact_setup": 6, "compact_run": 7, "blend": 12},
        "extra_needles": {7: ("'--max-fallback-fraction', '1.0'", "pred = TMP / 'prediction'"),
                          9: ("%%writefile /tmp/raptor224_run.py", R224_CKPT, R224_CKPT_SHA256,
                              'out = "/kaggle/working/raptor224_sub.csv"'),
                          10: ("_left = 9 * 3600 - (time.time() - NB_T0) - 40 * 60", "timeout=min(_left, 2.5 * 3600)",
                               "R224_OK = _rc == 0 and os.path.isfile('/kaggle/working/raptor224_sub.csv')"),
                          12: ("W_RAPTOR_C = {'Medial OA': 0.5, 'Lateral OA': 0.5, 'PF OA': 0.5}",
                               "for f in pred.glob('worker*/predictions.csv'):", "A224 = 0.35",
                               "rr = (1.0 - A224) * rr + A224 * pct_on(r224[c].values, ~rap_fail)")},
    },
}
# heliosli's two notebooks of 10-14 / 10-15 (experiments.md 2026-10-16 "Submission #91 and the 0.957 / 0.958 frontier"): nine cells,
# stage 1 (384) and stage 1b (224, ungated: a failed 224 stage stops the notebook) before stage 2, which runs R6 (bundle
# fe055406fad6a12f) with flip TTA; stage 3 mixes the 224 rank at 0.35 of the Raptor half.  0958 = 0957 + five per-finding Raptor
# weights (W_RAPTOR_C) and the R6 Dataset's -w1 copy.  0957 mounts `heliosli/rsna-knee-r6-frozen-fe055406`, whose latest version
# holds R7 (traps 72): the build patches its C_SLUG to the byte-identical -w1 copy, the one line it changes.
_NINE_CELLS = ["markdown", "markdown", "code", "code", "code", "code", "code", "code", "code"]
_NINE_STAGES = {"raptor_write": 2, "raptor_run": 3, "compact_setup": 6, "compact_run": 7, "blend": 8}
_R6_SLUG = "rsna-knee-r6-frozen-fe055406"


def _nine_needles(w_raptor_c):
    return {4: ("%%writefile /tmp/raptor224_run.py", R224_CKPT, R224_CKPT_SHA256, 'out = "/kaggle/working/raptor224_sub.csv"'),
            5: ("/tmp/raptor224_run.py", "RAPTOR224_STAGE_DONE", "raise RuntimeError('Raptor-224 stage failed"),
            6: ("FLIP_TTA = True", "if manifest.get('flip_tta') or FLIP_TTA:"),
            7: ("'--max-fallback-fraction', '1.0'", "pred = TMP / 'prediction'"),
            8: (w_raptor_c, "A224 = 0.35", "for f in pred.glob('worker*/predictions.csv'):",
                "rr = np.where(r384_fail, r224, np.where(r224_fail, r384, (1.0 - A224) * r384 + A224 * r224))")}


ANCHORS["0957"] = {
    "notebook": "notebook_score_0.957.ipynb",
    "sha256": "16a9d953972d9d82717f279f75e57e798d3902f93d1071c30f3caa67326e3601",   # as pulled 2026-10-16
    "score": "0.957",
    "compact": "R6",
    "compact_dataset": "heliosli/" + _R6_SLUG + "-w1",                              # the mount (v1 of the slug = R6)
    "c_slug": _R6_SLUG,                                                               # as the notebook names it (patched below)
    "bundle_id": "fe055406fad6a12f",
    "weights": ("0.5", "0.5"),
    "slug": "rsna-knee-fork957",
    "title": "RSNA Knee Fork957",
    "desc": "+ the nartaa 224 px checkpoint at 0.35 of the Raptor half, R6 with flip TTA (heliosli)",
    "cell_types": _NINE_CELLS,
    "stage_cells": _NINE_STAGES,
    "extra_needles": _nine_needles("W_RAPTOR_C = {'Medial OA': 0.5, 'Lateral OA': 0.5, 'PF OA': 0.5}"),
    "patches": {6: [("C_SLUG = '%s'   # [d957]" % _R6_SLUG,
                     "C_SLUG = '%s-w1'   # [fork957: the -w1 copy = v1 = R6; this slug's latest version holds R7, traps 72] [d957]"
                     % _R6_SLUG)]},
}
ANCHORS["0958"] = {
    "notebook": "notebook_score_0.958.ipynb",
    "sha256": "ad0b97aca3361f16e32b0836655dd68bc2a37ff0cc5a44f6d6f4de07880d986d",   # as pulled 2026-10-16
    "score": "0.958",
    "compact": "R6",
    "compact_dataset": "heliosli/" + _R6_SLUG + "-w1",
    "bundle_id": "fe055406fad6a12f",
    "weights": ("0.5", "0.5"),
    "slug": "rsna-knee-fork958",
    "title": "RSNA Knee Fork958",
    "desc": "+ the nartaa 224 px checkpoint, R6 with flip TTA, per-finding Raptor weights (heliosli)",
    "cell_types": _NINE_CELLS,
    "stage_cells": _NINE_STAGES,
    "extra_needles": _nine_needles("W_RAPTOR_C = {'Medial OA': 0.5, 'Lateral OA': 0.2, 'PF OA': 0.6, 'Synovitis': 0.8, "
                                   "\"Baker's\": 0.35}"),
}
# nartaa's s6fall solo (a script kernel, 0.950 public alone; experiments.md 2026-10-16 "Submission #91 …"): heliosli's S6
# architecture retrained on JEV labels, run through heliosli's compact_infer.py unchanged.  The script also runs nartaa's
# OAI-trained 0.949 CoAtNet as component A, at weight 0 (W_A = 0.0) but as the rank of a study B fails on and as a rank-corr
# sanity floor.  C11 is OAI-free: the build skips component A, does not mount its Dataset, gives a failed B study the column-mean
# rank (heliosli's controller rule, as C8 does), and removes the 3 public test ids the script embeds (only A read them; the
# script stays gitignored under artifacts/anchors/, traps 73).  Everything else runs byte for byte as a subprocess.
S6FALL_DATASET = "nartaa/rsna-knee-s6fall-weights"
ANCHORS["0950s6f"] = {
    "kind": "script",
    "notebook": os.path.join("artifacts", "anchors", "notebook_score_0.950s6fall.py"),
    "sha256": "2f2d50d8add7c8d2166577cdc16e511def9fa53620d431e25188be06da0aaa24",   # as pulled 2026-10-16
    "score": "0.950",
    "compact": "s6fall",
    "compact_owner": "nartaa's",
    "compact_prev": ("The previous cell has run nartaa's s6fall script (component A, the OAI-trained CoAtNet, removed by the "
                     "builder) and written its predictions as `submission.csv`."),
    "compact_label": ("'anchor': 's6fall alone (nartaa: S6 architecture on JEV labels; component A, the OAI-trained CoAtNet, "
                      "not run)'"),
    "compact_dataset": S6FALL_DATASET,
    "weights": ("0.0", "1.0"),
    "slug": "rsna-knee-forks6f",
    "title": "RSNA Knee Forks6f",
    "extra_needles": {},
}
S6F_NEEDLES = ('DS_SLUG = "rsna-knee-s6fall-weights"', 'HELIOS_SLUG = "rsna-orthofoundation-fold0-epoch7-20261004"',
               "W_A, W_B = 0.0, 1.0", 'MODE = "submit"', 'B_MODEL = {"file": "E_prior_s42_fall.pt"',
               'KROOT = Path(os.environ.get("K1_KAGGLE_ROOT", "/kaggle"))', "def blend_with_failures(A, B, failed_mask):",
               'if __name__ == "__main__":\n    main()')
S6F_PATCHES = [
    ('        stage("A")\n        A, a_info = run_a(I)\n        _RECEIPT["A"] = a_info\n        stage_done("A")\n',
     '        stage("A")   # [forks6f] component A (nartaa\'s OAI-trained 0.949 CoAtNet) is NOT run; its Dataset is not mounted\n'
     '        _RECEIPT["A"] = {"skipped": "forks6f (src/build_fork954.py --anchor 0950s6f): OAI-free; a B-failed study takes the '
     'column-mean rank"}\n'
     '        stage_done("A")\n'),
    ('        stage("blend")\n        _RECEIPT["blend"] = blend_and_write(I, A, B, b_failed)\n',
     '        # [forks6f] A := B with its failed studies mean-filled (heliosli\'s controller rule): with W_A = 0, blend_and_write then\n'
     '        # writes rank(B) on every scored study and the column-mean rank on a failed one, and its rank-corr floor against A is 1\n'
     '        _forks6f_fmask = np.array([u in (b_failed or {}) for u in I.test_ids])\n'
     '        A = np.array(B, np.float64)\n'
     '        if _forks6f_fmask.any():\n'
     '            A[_forks6f_fmask] = np.nanmean(A[~_forks6f_fmask], axis=0)\n'
     '        stage("blend")\n        _RECEIPT["blend"] = blend_and_write(I, A, B, b_failed)\n'),
]
S6F_IDS_RE = r'^PUBLIC_TEST_IDS = \[[^\]]*\]'
S6F_IDS_NEW = "PUBLIC_TEST_IDS = []   # [forks6f] the 3 public test ids removed (competition data; only component A read them)"
for _k, _s in ANCHORS.items():
    _s["name"] = _k
    _s["weight_line"] = "W_RAPTOR, W_OURS = %s, %s" % _s["weights"]

WEIGHT_LINE_S6 = "W_RAPTOR, W_OURS = 0.0, 1.0   # [fork954 --s6-only] A14: S6 alone (the CoAtNet still runs, weight 0)"

MEMBER_SOURCES = dict(f949.MEMBER_SOURCES)
MEMBER_SOURCES["v15co"] = ("tiankljucanin/rsna-knee-ckpt-v15co", "tiankljucanin/timm-convnext-tiny-in12k")
MEMBER_SOURCES["v15cr"] = ("tiankljucanin/rsna-knee-ckpt-v15cr", "tiankljucanin/timm-convnext-tiny-in12k")


def _swap(text, pairs):
    for old, new in pairs:
        if text.count(old) != 1:
            raise SystemExit(f"fork954: expected exactly one {old!r} in the fork949 template, found {text.count(old)}")
        text = text.replace(old, new)
    return text


def fork_templates(spec):
    """The fork-arm markdown and code for this anchor: (blend markdown, blend arm, compact markdown, compact arm)."""
    s, c, (wr, wo) = spec["score"], spec["compact"], spec["weights"]
    tag = s.replace(".", "")
    desc = (" " + spec["desc"]) if spec.get("desc") else ""
    blend_head = f"## Fork arm — our members blended onto the public {s} blend (nartaa's CoAtNet {wr} + heliosli's {c} {wo}{desc})"
    blend_prev = f"The previous cell (stage 3 of the {s} notebook, unchanged) has written `submission.csv`."
    arm_head = f"# ===================== FORK ARM: ours as a beta-blend on the public {s} blend =====================      "
    arm_label = f"'anchor': 'public {s} blend (heliosli: nartaa CoAtNet {wr} + {c} {wo}{desc})'"
    md = _swap(f949.FORK_MARKDOWN, [
        ("## Fork arm — our members blended onto the public 0.949 single model", blend_head),
        ("The previous cell (the scored 0.949 source, unchanged) has written `submission.csv`.", blend_prev),
        ("`submission_fork_anchor_0949.csv`", f"`submission_fork_anchor_{tag}.csv`"),
    ])
    arm = _swap(f949.FORK_ARM_TEMPLATE, [
        ("# ===================== FORK ARM: ours as a beta-blend on the public 0.949 single model =====================",
         arm_head),
        ("_FORK_ANCHOR = _FORK_WORK / '_anchor_0949.csv'", f"_FORK_ANCHOR = _FORK_WORK / '_anchor_{tag}.csv'"),
        ("_FORK_ANCHOR_PUB = _FORK_WORK / 'submission_fork_anchor_0949.csv'",
         f"_FORK_ANCHOR_PUB = _FORK_WORK / 'submission_fork_anchor_{tag}.csv'"),
        ("'anchor': 'public 0.949 single model (nartaa)'", arm_label),
    ])
    owner = spec.get("compact_owner", "heliosli's")          # a script anchor (0950s6f) names its own author and stage
    md_c = _swap(md, [
        (blend_head, f"## Fork arm — our members blended onto {owner} {c} alone (OAI-free: no CoAtNet stage in this notebook)"),
        (blend_prev, spec.get("compact_prev",
                              f"The previous cell has written {c}'s predictions (stage 2 of the {s} notebook, unchanged) as "
                              f"`submission.csv`.")),
    ])
    arm_c = _swap(arm, [
        (arm_head,
         f"# ===================== FORK ARM: ours as a beta-blend on {owner} {c} alone (OAI-free) =====================  "),
        (arm_label, spec.get("compact_label", f"'anchor': '{c} alone (heliosli compact model; no OAI-trained stage)'")),
    ])
    return md, arm, md_c, arm_c


T0_CELL_S6 = """# Fork954 --s6-anchor: start clock for our arm's time budget (added by src/build_fork954.py). Stage 1 (the OAI-trained
# CoAtNet) is not in this notebook, so its clock NB_T0, which stage 2 reads (T0 = NB_T0), is set here instead.
import time as _fork_time0
_FORK_T0 = _fork_time0.time()
NB_T0 = _FORK_T0
"""

S6_ANCHOR_CELL = """# Fork954 --s6-anchor: S6's own predictions (stage 2, unchanged, wrote ours_sub.csv) become the anchor submission.csv.
# Stage 3 (the 0.7 / 0.3 blend with the CoAtNet) is not in this notebook. A failed S6 stage has already raised above.
import pandas as _fork_pd0
_fork_ids = _fork_pd0.read_csv(DATA / 'test.csv')['StudyInstanceUID'].astype(str).tolist()
_fork_s6 = _fork_pd0.read_csv(WORK / 'ours_sub.csv', dtype={'StudyInstanceUID': str})
_fork_cols = (list(_fork_pd0.read_csv(DATA / 'sample_submission.csv', nrows=1).columns)
              if (DATA / 'sample_submission.csv').is_file() else list(_fork_s6.columns))
assert sorted(_fork_s6['StudyInstanceUID']) == sorted(_fork_ids), 'S6: rows differ from test.csv'
assert set(_fork_cols) <= set(_fork_s6.columns), 'S6: columns differ'
_fork_s6.set_index('StudyInstanceUID').loc[_fork_ids].reset_index()[_fork_cols].to_csv(WORK / 'submission.csv', index=False)
print('S6_ANCHOR_WRITTEN rows %d' % len(_fork_ids), flush=True)
"""

R7_ANCHOR_CELL = """# Fork956 --compact-anchor: R7's own predictions (stage 2, unchanged, wrote ours_sub.csv) become the anchor submission.csv.
# Stage 3 (the 0.5 / 0.5 blend with the CoAtNet) is not in this notebook. Stage 2 runs with --max-fallback-fraction 1.0 (stage 3
# gave a study R7 did not score the CoAtNet's rank) and mean-fills such a study; with no stage 3 here, count them with stage 3's
# own rule (a study with no valid row in pred/worker*/predictions.csv), print the count, and stop above 50 % (stage 3's guard).
import csv as _fork_csv0
import pandas as _fork_pd0
_fork_ids = _fork_pd0.read_csv(DATA / 'test.csv')['StudyInstanceUID'].astype(str).tolist()
_fork_s6 = _fork_pd0.read_csv(WORK / 'ours_sub.csv', dtype={'StudyInstanceUID': str})
_fork_cols = (list(_fork_pd0.read_csv(DATA / 'sample_submission.csv', nrows=1).columns)
              if (DATA / 'sample_submission.csv').is_file() else list(_fork_s6.columns))
assert sorted(_fork_s6['StudyInstanceUID']) == sorted(_fork_ids), 'R7: rows differ from test.csv'
assert set(_fork_cols) <= set(_fork_s6.columns), 'R7: columns differ'
_fork_scored = set()
for _fork_f in pred.glob('worker*/predictions.csv'):
    with open(_fork_f, encoding='utf-8') as _fork_fh:
        for _fork_r in _fork_csv0.DictReader(_fork_fh):
            try:
                _fork_v = [float(_fork_r[_fork_c]) for _fork_c in _fork_cols[1:]]
            except (KeyError, ValueError, TypeError):
                continue
            if all(0. <= _fork_x <= 1. for _fork_x in _fork_v):
                _fork_scored.add(_fork_r['StudyInstanceUID'])
_fork_unscored = sum(_fork_u not in _fork_scored for _fork_u in _fork_ids)
print('R7_FALLBACK unscored %d of %d (mean-filled by stage 2)' % (_fork_unscored, len(_fork_ids)), flush=True)
if _fork_unscored > 0.5 * len(_fork_ids):
    raise RuntimeError('R7 did not score %d of %d studies (more than 50 %%)' % (_fork_unscored, len(_fork_ids)))
_fork_s6.set_index('StudyInstanceUID').loc[_fork_ids].reset_index()[_fork_cols].to_csv(WORK / 'submission.csv', index=False)
print('R7_ANCHOR_WRITTEN rows %d' % len(_fork_ids), flush=True)
"""

T0_CELL = """# Fork954: start clock for our arm's time budget (added by src/build_fork954.py; changes nothing below)
import time as _fork_time0
_FORK_T0 = _fork_time0.time()
"""


def clock_cell(spec, compact):
    """The clock cell: 0954's text unchanged, other anchors renamed (the compact one also sets NB_T0)."""
    text = T0_CELL_S6 if compact else T0_CELL
    if spec["name"] != "0954":
        text = text.replace("Fork954 --s6-anchor", "Fork%s --compact-anchor" % spec["name"][1:])
        text = text.replace("Fork954:", "Fork%s:" % spec["name"][1:])
    return text


def compact_anchor_cell(spec):
    return {"0954": S6_ANCHOR_CELL, "0956": R7_ANCHOR_CELL}[spec["name"]]


# names the anchor's own cells bind at notebook scope: the arm must not rebind any of them (the 0956 stage 3 adds the last row)
ANCHOR_GLOBALS = ("T0", "NB_T0", "WORK", "DATA", "TMP", "pred", "env", "rc", "deadline", "receipt", "sub", "blend",
                  "ours", "raw", "rsub", "labels", "cols", "test_ids", "out", "pct", "n",
                  "W_RAPTOR_C", "pct_on", "rap_fail", "our_fail", "scored", "rho_rr", "rho_ro",
                  "R224_OK", "A224", "r224",
                  # 0957 / 0958: stage 1b, the R6 flip switch and the 224-aware stage 3
                  "RAPTOR_SECONDS", "RAPTOR224_SECONDS", "FLIP_TTA", "raw224", "rsub224", "r384", "r384_fail", "r224_fail",
                  "rho_rr224", "rho_224_384")


def check_anchor(nb, spec):
    """The anchor must be heliosli's notebook as pulled: seven cells, three stages, its stage-3 weights."""
    cells = nb["cells"]
    if [c["cell_type"] for c in cells] != spec.get("cell_types", ANCHOR_CELL_TYPES):
        raise SystemExit(f"anchor cell types changed: {[c['cell_type'] for c in cells]}")
    txt = [cell_text(c) for c in cells]
    st = spec.get("stage_cells", STAGE_CELLS)
    needles = {
        st["raptor_write"]: ("%%writefile /tmp/raptor_run.py", f949.ANCHOR_CKPT, f949.ANCHOR_CKPT_SHA256, "RUN_INFERENCE = True"),
        st["raptor_run"]: ("NB_T0 = time.time()", "/tmp/raptor_run.py", "raptor_sub.csv", "raptor_raw.csv"),
        st["compact_setup"]: ("ARM = 'E_lrprior'", f"BUNDLE_ID = '{spec['bundle_id']}'", "DEADLINE_HOURS = 8.3", "T0 = NB_T0",
                              f"C_SLUG = '{spec.get('c_slug', spec['compact_dataset'].split('/')[1])}'"),
        st["compact_run"]: ("probe_dual.py", "ours_sub.csv", "OURS_STAGE_COMPLETE"),
        st["blend"]: (spec["weight_line"], "sub.to_csv(WORK / 'submission.csv', index=False)", "NOTEBOOK_COMPLETE"),
    }
    for i, keys in list(needles.items()) + list(spec["extra_needles"].items()):
        for k in keys:
            if k not in txt[i]:
                raise SystemExit(f"anchor cell {i} lacks {k!r}")
    if txt[st["blend"]].count(spec["weight_line"]) != 1:
        raise SystemExit("anchor stage 3: the weight line is not unique")
    # our arm writes these names in /kaggle/working: the anchor must not use any of them
    for name in ("_ours.csv", "ours_infer.py", "ours_infer.log", "fork_diagnostics.json",
                 "_anchor_%s.csv" % spec["score"].replace(".", ""), "submission_fork_"):
        if any(name in t for t in txt):
            raise SystemExit(f"anchor already uses {name!r}: our arm would collide with it")


def render_fork_cells(payload, payload_sha, pipeline_sha, params, spec, mode="blend"):
    md_b, arm_b, md_c, arm_c = fork_templates(spec)
    md_t, arm_t = (md_c, arm_c) if mode == "compact_anchor" else (md_b, arm_b)
    md = {"id": "fork-markdown", "cell_type": "markdown", "metadata": {}, "source": f949._fill(md_t, params)}
    pay = {"id": "fork-payload", "cell_type": "code", "metadata": {}, "outputs": [], "execution_count": None,
           "source": f949._fill(f949.FORK_PAYLOAD_TEMPLATE, params,
                                {"__PAYLOAD__": chunk_literal(payload), "__PAYLOAD_SHA__": payload_sha,
                                 "__PIPELINE_SHA__": pipeline_sha})}
    arm = {"id": "fork-arm", "cell_type": "code", "metadata": {}, "outputs": [], "execution_count": None,
           "source": f949._fill(arm_t, params)}
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


def build_notebook(nb, fork_cells, provenance, mode, spec):
    kept = [json.loads(json.dumps(c)) for c in nb["cells"]]
    for a, b in zip(kept, nb["cells"]):
        if json.dumps(a, sort_keys=True) != json.dumps(b, sort_keys=True):
            raise SystemExit("anchor cell changed during copy")
    if mode == "s6_only":
        src = cell_text(kept[6]).replace(spec["weight_line"], WEIGHT_LINE_S6)
        if WEIGHT_LINE_S6 not in src or spec["weight_line"] in src:
            raise SystemExit("--s6-only: the weight patch did not apply")
        kept[6]["source"] = src
    for i, pairs in spec.get("patches", {}).items():     # 0957: the one line it changes (C_SLUG -> the -w1 copy of R6)
        src = cell_text(kept[i])
        for old, new in pairs:
            if src.count(old) != 1:
                raise SystemExit(f"anchor cell {i}: patch target found {src.count(old)} times: {old!r}")
            src = src.replace(old, new)
        kept[i]["source"] = src
    if mode == "compact_anchor":
        anchor_src = compact_anchor_cell(spec)
        t0 = {"id": "fork-t0", "cell_type": "code", "metadata": {}, "outputs": [], "execution_count": None,
              "source": clock_cell(spec, compact=True)}
        cpa = {"id": "fork-s6-anchor", "cell_type": "code", "metadata": {}, "outputs": [], "execution_count": None,
               "source": anchor_src}
        compile(anchor_src, "fork-s6-anchor", "exec")
        clash = sorted(module_level_names(anchor_src) & set(ANCHOR_GLOBALS))
        if clash:
            raise SystemExit(f"fork-s6-anchor binds {clash}")
        # stage 2 mentions the CoAtNet only in comments ("the Raptor stage ran first"); it must not use its files
        for needle in (NARTAA_DATASET.split("/")[1], "raptor_", "RAPTOR_SECONDS", f949.ANCHOR_CKPT):
            if any(needle in cell_text(c) for c in kept[4:6]):
                raise SystemExit(f"--compact-anchor: stage 2 uses {needle!r} from the CoAtNet stage")
        cells = kept[:2] + [t0] + kept[4:6] + [cpa]
    else:
        t0 = {"id": "fork-t0", "cell_type": "code", "metadata": {}, "outputs": [], "execution_count": None,
              "source": clock_cell(spec, compact=False)}
        cells = kept[:2] + [t0] + kept[2:]
    return {
        "cells": cells + fork_cells,
        "metadata": {
            "kernelspec": nb["metadata"].get("kernelspec", {"display_name": "Python 3", "language": "python", "name": "python3"}),
            "language_info": nb["metadata"].get("language_info", {"name": "python"}),
            "rsna_fork": provenance,
        },
        "nbformat": nb.get("nbformat", 4),
        "nbformat_minor": nb.get("nbformat_minor", 4),
    }


def build_metadata(params, spec, mode="blend"):
    ours = []
    for v in params.members:
        if v not in MEMBER_SOURCES:
            raise SystemExit(f"no Dataset mapping for member {v!r}")
        for s in MEMBER_SOURCES[v]:
            if s and s not in ours:
                ours.append(s)
    anchor_datasets = (spec["compact_dataset"], NARTAA_DATASET, ORTHO_DATASET)
    return {
        "id": "tiankljucanin/" + spec["slug"],
        "title": spec["title"],
        "code_file": spec["slug"] + ".ipynb",
        "language": "python",
        "kernel_type": "notebook",
        "is_private": True,
        "enable_gpu": True,
        "enable_tpu": False,
        "enable_internet": False,
        "keywords": [],
        "dataset_sources": [d for d in anchor_datasets if not (mode == "compact_anchor" and d == NARTAA_DATASET)]
                           + [*f949.LABEL_DATASETS] + sorted(ours),
        "kernel_sources": [],
        "competition_sources": [COMPETITION],
        "model_sources": [f949.DINOV2_MODEL],
        "docker_image": DOCKER_IMAGE,
        "machine_shape": "NvidiaTeslaT4",
    }


def build(params, mode, spec, notebook=None, pipeline=PIPELINE):
    notebook = notebook or spec["notebook"]
    if sha256_file(notebook) != spec["sha256"]:
        raise SystemExit(f"{notebook}: sha256 differs from the pull this builder was written against")
    nb = load_notebook(notebook)
    check_anchor(nb, spec)
    text, pipeline_sha = prepare_pipeline(pipeline, params.members)
    payload = make_payload(text)
    fork_cells = render_fork_cells(payload, sha256_text(text), pipeline_sha, params, spec, mode)
    selftest_blend(fork_cells)
    provenance = {
        "builder": "src/build_fork954.py",
        "source_notebook": os.path.basename(notebook),
        "source_notebook_sha256": spec["sha256"],
        "s6_only": mode == "s6_only",
        "pipeline_sha256": pipeline_sha,
        "payload_sha256": sha256_text(text),
        "members": list(params.members),
        "beta": params.beta,
        "betas_diag": list(params.betas_diag),
        "gate_h": params.gate_h,
        "hard_stop_h": params.hard_stop_h,
        "sec_per_study": params.sec_per_study,
    }
    if mode == "compact_anchor":
        provenance["s6_anchor" if spec["name"] == "0954" else "compact_anchor"] = True
    if spec.get("patches"):
        provenance["anchor_patches"] = {str(i): [new for _, new in pairs] for i, pairs in spec["patches"].items()}
    return build_notebook(nb, fork_cells, provenance, mode, spec), build_metadata(params, spec, mode), payload


S6F_INTRO = """# RSNA Knee Forks6f: nartaa's s6fall (0.950 public alone) + our members (candidates.md C11)

Built by `src/build_fork954.py --anchor 0950s6f`. The next cells run nartaa's public `rsna-knee-s6fall-solo` script (heliosli's S6
architecture retrained on JEV labels, through heliosli's `compact_infer.py`) as its own process. The builder changed exactly two things:
component A, nartaa's OAI-trained 0.949 CoAtNet, is not run and its Dataset is not mounted (a study the ConvNeXt fails on takes the
column-mean rank instead of A's); and the three public test ids the script embedded for A are removed. Our arm then blends onto it.
Credits: nartaa (s6fall weights, CC BY-NC 4.0), heliosli (S6 architecture, inference code, SKM-TEA anatomy model).
"""

S6F_RUN_CELL = """# Forks6f: run nartaa's s6fall script (patched by src/build_fork954.py: component A not run, the public test ids removed) as its own
# process, as heliosli's notebooks run nartaa's Raptor script. It writes /kaggle/working/submission.csv or raises (its own HARD ERROR).
import os as _forks6f_os, subprocess as _forks6f_sp, sys as _forks6f_sys, time as _forks6f_time
_forks6f_t = _forks6f_time.time()
_forks6f_rc = _forks6f_sp.run([_forks6f_sys.executable, '-u', '/tmp/s6fall_run.py'], timeout=8.25 * 3600).returncode
if _forks6f_rc != 0:
    raise RuntimeError('s6fall stage failed with exit code %d' % _forks6f_rc)
assert _forks6f_os.path.isfile('/kaggle/working/submission.csv'), 'submission.csv missing after the s6fall stage'
print('S6FALL_STAGE_DONE seconds %.0f' % (_forks6f_time.time() - _forks6f_t), flush=True)
"""


def patch_s6fall(text):
    """nartaa's s6fall script -> the C11 stage: component A skipped, B's failures mean-filled, the public test ids removed."""
    import re
    for k in S6F_NEEDLES:
        if text.count(k) != 1:
            raise SystemExit(f"s6fall script: expected exactly one {k!r}, found {text.count(k)}")
    for old, new in S6F_PATCHES:
        if text.count(old) != 1:
            raise SystemExit(f"s6fall script: patch target found {text.count(old)} times: {old[:60]!r}")
        text = text.replace(old, new)
    ids = re.findall(S6F_IDS_RE, text, flags=re.M | re.S)
    if len(ids) != 1 or ids[0].count('"') != 6:
        raise SystemExit("s6fall script: PUBLIC_TEST_IDS is not one list of three ids")
    text = re.sub(S6F_IDS_RE, S6F_IDS_NEW, text, count=1, flags=re.M | re.S)
    if "= run_a(I)" in text or "PUBLIC_TEST_IDS)" not in text:     # A's call is gone; A's own (dead) comparison still names it
        raise SystemExit("s6fall script: component A is still called, or the id list is no longer where A reads it")
    compile(text, "s6fall_run.py", "exec")
    return text


def check_no_competition_ids(text, what):
    """The repo is public (traps 73): no train / test StudyInstanceUID may appear in a rendered kernel."""
    import csv
    for split in ("test", "train"):
        p = os.path.join("data", split + ".csv")
        if not os.path.isfile(p):
            continue
        with open(p, encoding="utf-8", newline="") as fh:
            hits = [r["StudyInstanceUID"] for r in csv.DictReader(fh) if r["StudyInstanceUID"] and r["StudyInstanceUID"] in text]
        if hits:
            raise SystemExit(f"{what} contains {len(hits)} {split}.csv StudyInstanceUID(s): refusing to write it (traps 73)")


def build_script(params, spec, notebook=None, pipeline=PIPELINE):
    """C11: a script anchor (0950s6f) wrapped as %%writefile + subprocess cells, then our arm in compact-anchor form."""
    notebook = notebook or spec["notebook"]
    if sha256_file(notebook) != spec["sha256"]:
        raise SystemExit(f"{notebook}: sha256 differs from the pull this builder was written against")
    with open(notebook, encoding="utf-8") as fh:
        stage_src = patch_s6fall(fh.read())
    text, pipeline_sha = prepare_pipeline(pipeline, params.members)
    payload = make_payload(text)
    fork_cells = render_fork_cells(payload, sha256_text(text), pipeline_sha, params, spec, "compact_anchor")
    selftest_blend(fork_cells)
    for src, cid in ((S6F_RUN_CELL, "forks6f-run"),):
        compile(src, cid, "exec")
        clash = sorted(module_level_names(src) & set(ANCHOR_GLOBALS))
        if clash:
            raise SystemExit(f"{cid} binds {clash}")

    def code(cid, src):
        return {"id": cid, "cell_type": "code", "metadata": {}, "outputs": [], "execution_count": None, "source": src}
    cells = [{"id": "forks6f-intro", "cell_type": "markdown", "metadata": {}, "source": S6F_INTRO},
             code("fork-t0", T0_CELL.replace("Fork954:", "Forks6f:")),
             code("forks6f-script", "%%writefile /tmp/s6fall_run.py\n" + stage_src),
             code("forks6f-run", S6F_RUN_CELL)] + fork_cells
    provenance = {
        "builder": "src/build_fork954.py",
        "source_script": os.path.basename(notebook),
        "source_script_sha256": spec["sha256"],
        "stage_sha256": sha256_text(stage_src),
        "script_patches": ["component A (OAI-trained 0.949 CoAtNet) not run; a B-failed study takes the column-mean rank",
                           "PUBLIC_TEST_IDS removed"],
        "pipeline_sha256": pipeline_sha,
        "payload_sha256": sha256_text(text),
        "members": list(params.members),
        "beta": params.beta,
        "betas_diag": list(params.betas_diag),
        "gate_h": params.gate_h,
        "hard_stop_h": params.hard_stop_h,
        "sec_per_study": params.sec_per_study,
        "compact_anchor": True,
    }
    out_nb = {"cells": cells,
              "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
                           "language_info": {"name": "python"}, "rsna_fork": provenance},
              "nbformat": 4, "nbformat_minor": 4}
    meta = build_metadata(params, spec, "compact_anchor")      # mounts s6fall + heliosli's code / anatomy Dataset + ours
    if NARTAA_DATASET in meta["dataset_sources"]:
        raise SystemExit("C11 must not mount nartaa's OAI-trained CoAtNet Dataset")
    return out_nb, meta, payload


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--anchor", choices=sorted(ANCHORS), default="0954")
    ap.add_argument("--notebook", default=None, help="default: the --anchor's notebook")
    ap.add_argument("--pipeline", default=PIPELINE)
    ap.add_argument("--out", default=None, help="default: kaggle/<the --anchor's kernel slug>")
    ap.add_argument("--members", nargs="+", default=["v15co", "v13b3"])          # B21, read 0.946 (#79)
    ap.add_argument("--beta", type=float, default=0.20)
    ap.add_argument("--betas-diag", nargs="+", type=float, default=[0.10, 0.30])
    ap.add_argument("--gate-hours", type=float, default=6.0)
    ap.add_argument("--hard-stop-hours", type=float, default=8.4)
    ap.add_argument("--sec-per-study", type=float, default=5.5)
    ap.add_argument("--s6-only", action="store_true", help="A14: stage-3 weights 0.0 / 1.0 and beta 0 (S6 alone; 0954 only)")
    ap.add_argument("--compact-anchor", "--s6-anchor", dest="compact_anchor", action="store_true",
                    help="C6 / C8: no CoAtNet stage; our arm blends onto the compact model alone (OAI-free)")
    ap.add_argument("--check", action="store_true", help="build in memory and compare with the files on disk")
    a = ap.parse_args(argv)
    os.chdir(ROOT)
    spec = ANCHORS[a.anchor]
    if a.s6_only and a.compact_anchor:
        raise SystemExit("--s6-only and --compact-anchor are exclusive")
    if a.s6_only and a.anchor != "0954":
        raise SystemExit("--s6-only patches one weight line; the 0956 stage 3 also has per-finding W_RAPTOR_C overrides")
    if a.compact_anchor and a.anchor == "0956v":
        raise SystemExit("--compact-anchor drops the CoAtNet stages; on 0956v that is the 0956 build (C8): use --anchor 0956")
    if a.compact_anchor and a.anchor in ("0957", "0958"):
        raise SystemExit("--compact-anchor is not written for the 0957 / 0958 notebooks (R6 alone would need its own anchor cell)")
    script = spec.get("kind") == "script"
    if script and a.s6_only:
        raise SystemExit("--s6-only has no meaning on a script anchor")
    mode = "s6_only" if a.s6_only else ("compact_anchor" if (a.compact_anchor or script) else "blend")
    beta = 0.0 if a.s6_only else a.beta
    params = f949.ForkParams(members=tuple(a.members), beta=beta, betas_diag=tuple(a.betas_diag), gate_h=a.gate_hours,
                             hard_stop_h=a.hard_stop_hours, sec_per_study=a.sec_per_study)
    if script:
        out_nb, meta, payload = build_script(params, spec, a.notebook, a.pipeline)
    else:
        out_nb, meta, payload = build(params, mode, spec, a.notebook, a.pipeline)
    nb_text, meta_text = dump_json(out_nb), dump_json(meta)
    check_no_competition_ids(nb_text, "the rendered notebook")
    out_dir = a.out or os.path.join("kaggle", spec["slug"])
    nb_path = os.path.join(out_dir, spec["slug"] + ".ipynb")
    meta_path = os.path.join(out_dir, "kernel-metadata.json")
    c = spec["compact"]
    summary = (f"{nb_path}: {len(out_nb['cells'])} cells ("
               + ("1 intro + 1 clock + 1 script + 1 run + 3 fork" if script
                  else f"4 anchor + 1 clock + 1 {c}-anchor + 3 fork" if mode == "compact_anchor"
                  else f"{len(out_nb['cells']) - 4} anchor + 1 clock + 3 fork") + "); "
               f"{len(meta['dataset_sources'])} datasets; payload {len(payload) / 1024:.0f} KB; "
               f"members {list(params.members)} beta {params.beta}" + (" [S6 ONLY: stage-3 weights 0.0 / 1.0]" if a.s6_only else "")
               + (f" [{c} ANCHOR: no CoAtNet stage, OAI-free]" if a.compact_anchor else ""))
    if a.check:
        ok = True
        for path, text in ((nb_path, nb_text), (meta_path, meta_text)):
            same = os.path.exists(path) and open(path, encoding="utf-8").read() == text
            print(f"  {'same    ' if same else 'DIFFERS '}{path}")
            ok = ok and same
        print(("check: " + ("ok" if ok else "FAILED")) + " -- " + summary)
        return 0 if ok else 1
    os.makedirs(out_dir, exist_ok=True)
    with open(nb_path, "w", encoding="utf-8", newline="\n") as f:
        f.write(nb_text)
    with open(meta_path, "w", encoding="utf-8", newline="\n") as f:
        f.write(meta_text)
    print("wrote " + summary)
    return 0


if __name__ == "__main__":
    sys.exit(main())
