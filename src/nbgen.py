"""Convert a percent-format Python file into a Jupyter notebook.

Percent format keeps the pipeline authorable and runnable as a plain .py (so it can be
smoke-tested without Jupyter) while still producing the .ipynb that Kaggle needs.

Cell markers, each on its own line:
    # %%              -> code cell
    # %% [markdown]   -> markdown cell (subsequent '# ' comments become the text)

Usage:  python src/nbgen.py src/kaggle_pipeline.py kaggle/rsna-knee-train/rsna-knee-train.ipynb
"""

from __future__ import annotations

import base64
import hashlib
import json
import os
import re
import sys
import zlib

# P-31: the pipeline declares these two lines; when its PARALLEL_ARMS is a non-empty tuple, the generated
# notebook carries the whole .py as a zlib + base64 payload (chunked literals) plus its sha256, so the
# notebook can hand its own source to the per-GPU child processes. Files without the marker (the cache
# pipeline) and pipelines with PARALLEL_ARMS = () are left untouched, so the committed single-arm
# notebooks never churn a 50 KB blob.
_MARKER = re.compile(r"^SELF_SOURCE_B64 = None  # nbgen: filled at build$", re.M)
_SHA_LINE = re.compile(r"^SELF_SOURCE_SHA256 = None$", re.M)
_PARALLEL = re.compile(r"^PARALLEL_ARMS = \((?!\s*\)).*\)\s*$", re.M)


def embed_self_source(text: str) -> str:
    if not _MARKER.search(text) or not _PARALLEL.search(text):
        return text
    if len(_MARKER.findall(text)) != 1 or len(_SHA_LINE.findall(text)) != 1:
        raise SystemExit("nbgen: SELF_SOURCE_B64 / SELF_SOURCE_SHA256 markers must each appear exactly once")
    compile(text, "pipeline", "exec")
    payload = base64.b64encode(zlib.compress(text.encode("utf-8"), 9)).decode("ascii")
    sha = hashlib.sha256(text.encode("utf-8")).hexdigest()
    chunks = [payload[i:i + 120] for i in range(0, len(payload), 120)]
    literal = "(\n" + "\n".join(f"    '{c}'" for c in chunks) + "\n)"
    text = _SHA_LINE.sub(lambda m: f"SELF_SOURCE_SHA256 = '{sha}'", text)
    text = _MARKER.sub(lambda m: "SELF_SOURCE_B64 = " + literal, text)
    print(f"embedded self-source payload: {len(payload) / 1024:.0f} KB, sha256 {sha[:12]}…")
    return text


def split_cells(text: str) -> list[tuple[str, list[str]]]:
    cells: list[tuple[str, list[str]]] = []
    kind = "code"
    buf: list[str] = []
    for line in text.splitlines():
        stripped = line.strip()
        if stripped.startswith("# %%"):
            if buf:
                cells.append((kind, buf))
            kind = "markdown" if "[markdown]" in stripped else "code"
            buf = []
            continue
        buf.append(line)
    if buf:
        cells.append((kind, buf))
    return cells


def clean(kind: str, lines: list[str]) -> list[str]:
    # Drop leading/trailing blank lines.
    while lines and not lines[0].strip():
        lines.pop(0)
    while lines and not lines[-1].strip():
        lines.pop()
    if kind == "markdown":
        out = []
        for line in lines:
            s = line.lstrip()
            if s.startswith("# "):
                out.append(s[2:])
            elif s == "#":
                out.append("")
            else:
                out.append(line)
        lines = out
    return [line + "\n" for line in lines[:-1]] + (lines[-1:] if lines else [])


def build(src: str, dst: str) -> None:
    with open(src, encoding="utf-8") as fh:
        text = fh.read()
    text = embed_self_source(text)

    cells = []
    for kind, lines in split_cells(text):
        source = clean(kind, lines)
        if not source:
            continue
        if kind == "markdown":
            cells.append({"cell_type": "markdown", "metadata": {}, "source": source})
        else:
            cells.append({
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": source,
            })

    nb = {
        "cells": cells,
        "metadata": {
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3",
            },
            "language_info": {"name": "python", "version": "3.11"},
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }
    os.makedirs(os.path.dirname(dst) or ".", exist_ok=True)
    with open(dst, "w", encoding="utf-8") as fh:
        json.dump(nb, fh, indent=1)
        fh.write("\n")
    n_md = sum(1 for c in cells if c["cell_type"] == "markdown")
    print(f"wrote {dst}: {len(cells)} cells ({n_md} markdown, {len(cells)-n_md} code)")


if __name__ == "__main__":
    if len(sys.argv) != 3:
        raise SystemExit(__doc__)
    build(sys.argv[1], sys.argv[2])
