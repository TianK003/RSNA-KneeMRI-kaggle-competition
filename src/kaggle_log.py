"""Print a Kaggle kernel log (the JSON the CLI downloads) as plain lines, optionally filtered.

    python src/kaggle_log.py artifacts/kaggle_out/v16smoke/rsna-knee-train.log [substring ...]

With no filter every stdout/stderr line is printed; with filters, only lines containing at least
one of the substrings (case-sensitive). Saves re-inventing a JSON one-liner in every session.
"""
import json
import sys


def main():
    if len(sys.argv) < 2:
        raise SystemExit(__doc__)
    path, keys = sys.argv[1], sys.argv[2:]
    with open(path, encoding="utf-8") as f:
        raw = f.read()
    try:
        events = json.loads(raw)
    except json.JSONDecodeError:
        # some CLI versions write one JSON object per line, preceded by "," -- fall back
        events = []
        for line in raw.splitlines():
            line = line.strip().lstrip(",").rstrip(",")
            if line.startswith("{"):
                try:
                    events.append(json.loads(line))
                except json.JSONDecodeError:
                    pass
    for ev in events:
        data = ev.get("data", "") if isinstance(ev, dict) else ""
        for line in str(data).rstrip("\n").split("\n"):
            if not keys or any(k in line for k in keys):
                print(line[:300])


if __name__ == "__main__":
    main()
