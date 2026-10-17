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

    export PYTHONUTF8=1
    .venv/Scripts/python.exe src/build_anchor_copy.py --anchor 0958
    .venv/Scripts/python.exe src/build_anchor_copy.py --anchor 09586 [--check]
"""
import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from build_fork import cell_text, dump_json, load_notebook, sha256_file  # noqa: E402
from build_fork954 import DOCKER_IMAGE, check_no_competition_ids  # noqa: E402

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
    out = {"cells": cells,
           "metadata": {"kernelspec": nb["metadata"].get("kernelspec", {"display_name": "Python 3", "language": "python",
                                                                         "name": "python3"}),
                        "language_info": nb["metadata"].get("language_info", {"name": "python"}),
                        "rsna_anchor_copy": {"builder": "src/build_anchor_copy.py", "source_notebook": spec["notebook"],
                                             "source_notebook_sha256": spec["sha256"],
                                             "patched_cells": sorted(spec["patches"])}},
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
