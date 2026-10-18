"""Poll several pending submissions; the moment each one scores, save a full public-LB snapshot (artifacts/lb_snap_<ref>/).

Only a team's best entry shows on the LB, so several related entries in flight can still be read apart when they finish at
different times: our rank inside the tie, read right after one entry scores, moves only if that entry beat our previous best.
Prints one line per event and appends it to artifacts/lb_snapshots.csv (ref, first seen scored UTC, status, public score, our
rank, our LB score, teams).

    python src/watch_lb_snapshots.py 57353067 57353764 57353769 [--every 180] [--max-h 12] [--team "Tian Kljucanin"]
"""
import argparse
import csv
import glob
import io
import os
import subprocess
import sys
import time
import zipfile
from datetime import datetime, timezone

COMP = "rsna-knee-abnormality-detection"
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def kaggle(*args):
    env = dict(os.environ, PYTHONUTF8="1")
    return subprocess.run(["kaggle", *args], capture_output=True, text=True, encoding="utf-8", env=env, timeout=300)


def submissions():
    r = kaggle("competitions", "submissions", COMP, "--csv")
    rows = {}
    for row in csv.DictReader(io.StringIO(r.stdout.strip())):
        if row.get("ref"):
            rows[row["ref"]] = row
    return rows


def snapshot(ref, team):
    out = os.path.join(ROOT, "artifacts", "lb_snap_%s" % ref)
    os.makedirs(out, exist_ok=True)
    kaggle("competitions", "leaderboard", COMP, "-d", "-p", out)
    for z in glob.glob(os.path.join(out, "*.zip")):
        with zipfile.ZipFile(z) as zf:
            zf.extractall(out)
    files = sorted(glob.glob(os.path.join(out, "*.csv")))
    if not files:
        return None, None, None
    with open(files[-1], encoding="utf-8-sig", newline="") as fh:
        rows = list(csv.DictReader(fh))
    mine = [r for r in rows if r["TeamName"] == team]
    return (mine[0]["Rank"], mine[0]["Score"], len(rows)) if mine else (None, None, len(rows))


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("refs", nargs="+")
    ap.add_argument("--every", type=int, default=180)
    ap.add_argument("--max-h", type=float, default=12.0)
    ap.add_argument("--team", default="Tian Kljucanin")
    a = ap.parse_args(argv)
    pending, t0 = set(a.refs), time.time()
    log = os.path.join(ROOT, "artifacts", "lb_snapshots.csv")
    while pending and time.time() - t0 < a.max_h * 3600:
        try:
            subs = submissions()
        except Exception as e:                    # token window / 429: retry next poll
            print("poll error: %r" % (e,), flush=True)
            subs = {}
        for ref in sorted(pending):
            row = subs.get(ref)
            if not row or "PENDING" in row.get("status", ""):
                continue
            now = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S")
            rank, lb_score, teams = snapshot(ref, a.team)
            line = [ref, now, row["status"], row.get("publicScore", ""), rank, lb_score, teams]
            print("SCORED " + " | ".join(str(x) for x in line), flush=True)
            new = not os.path.exists(log)
            with open(log, "a", encoding="utf-8", newline="") as fh:
                w = csv.writer(fh)
                if new:
                    w.writerow(["ref", "first_seen_scored_utc", "status", "public_score", "our_rank", "our_lb_score", "teams"])
                w.writerow(line)
            pending.discard(ref)
        if pending:
            time.sleep(a.every)
    print("DONE, still pending: %s" % sorted(pending), flush=True)
    return 0 if not pending else 3


if __name__ == "__main__":
    sys.exit(main())
