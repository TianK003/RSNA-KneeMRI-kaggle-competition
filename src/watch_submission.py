"""Poll one competition submission until it is scored, and record how long scoring really took (P-41).

Every scoring time in docs/ so far is an upper bound ("sent 11:54, read 20:00") because nobody was
polling when the score landed. This script polls `competition_submissions`, prints one line per status
change, and on COMPLETE / ERROR appends a row to artifacts/submission_timing.csv:
sent (the submission's own UTC `date`), first poll that saw it scored, and the bound that implies
(scored within [elapsed - every, elapsed]). Run it in the background right after a submit; its exit is
the notification.

    export PYTHONUTF8=1
    .venv/Scripts/python.exe src/watch_submission.py                  # the newest submission
    .venv/Scripts/python.exe src/watch_submission.py --ref 56590282   # a specific one
    .venv/Scripts/python.exe src/watch_submission.py --every 120 --max-h 12

Exit 0 = COMPLETE (score printed), 2 = ERROR, 3 = still pending at --max-h, 4 = ref not found.
API failures (the ~30 min token-expiry window, traps 20; 429s) are printed and retried, never fatal.
"""
import argparse
import csv
import datetime as dt
import os
import sys
import time

COMPETITION = "rsna-knee-abnormality-detection"
TIMING_CSV = os.path.join("artifacts", "submission_timing.csv")
FIELDS = ["ref", "sent_utc", "scored_seen_utc", "elapsed_min", "bound_lo_min", "status",
          "public_score", "description"]


def utcnow():
    return dt.datetime.now(dt.timezone.utc).replace(tzinfo=None)


def status_name(s):
    """'SubmissionStatus.COMPLETE' / <SubmissionStatus.COMPLETE: 1> -> 'COMPLETE'."""
    return str(getattr(s, "name", s)).split(".")[-1].upper()


def fetch(api, ref):
    subs = api.competition_submissions(COMPETITION)
    if not subs:
        return None
    if ref is None:
        return subs[0]
    return next((s for s in subs if str(s.ref) == str(ref)), None)


def record(row):
    os.makedirs(os.path.dirname(TIMING_CSV), exist_ok=True)
    new = not os.path.exists(TIMING_CSV)
    with open(TIMING_CSV, "a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS)
        if new:
            w.writeheader()
        w.writerow(row)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ref", default=None, help="submission ref (default: the newest)")
    ap.add_argument("--every", type=int, default=120, help="poll interval, seconds")
    ap.add_argument("--max-h", type=float, default=12.0, help="give up after this many hours")
    a = ap.parse_args()

    from kaggle.api.kaggle_api_extended import KaggleApi
    api = KaggleApi()
    api.authenticate()

    t_end = time.time() + a.max_h * 3600
    last, first_status, sub = None, None, None
    while time.time() < t_end:
        try:
            sub = fetch(api, a.ref)
        except Exception as e:                       # token window / rate limit: retry, never die
            print(f"{utcnow():%H:%M:%S}Z  api error ({type(e).__name__}: {str(e)[:120]}) -- retrying", flush=True)
            time.sleep(a.every)
            try:                                     # the token is fixed at authenticate(); re-run it to refresh
                api.authenticate()
            except Exception:
                pass
            continue
        if sub is None:
            print(f"submission {a.ref} not found in {COMPETITION}", flush=True)
            return 4
        st = status_name(sub.status)
        first_status = first_status or st
        sent = sub.date.replace(tzinfo=None) if sub.date.tzinfo is None else \
            sub.date.astimezone(dt.timezone.utc).replace(tzinfo=None)
        now = utcnow()
        el = (now - sent).total_seconds() / 60
        if st != last:
            print(f"{now:%H:%M:%S}Z  ref {sub.ref}  {st}  {el:.1f} min after sending", flush=True)
            last = st
        if st in ("COMPLETE", "ERROR"):
            if first_status == st:
                # scored before we started watching: only the upper bound is known
                print(f"  was already {st} at the first poll -- scoring took <= {el:.0f} min (not recorded)",
                      flush=True)
            else:
                record({"ref": sub.ref, "sent_utc": f"{sent:%Y-%m-%d %H:%M:%S}",
                        "scored_seen_utc": f"{now:%Y-%m-%d %H:%M:%S}", "elapsed_min": f"{el:.1f}",
                        "bound_lo_min": f"{max(0.0, el - a.every / 60):.1f}", "status": st,
                        "public_score": sub.public_score or "", "description": (sub.description or "")[:200]})
                print(f"  scored within [{max(0.0, el - a.every / 60):.1f}, {el:.1f}] min of sending -> {TIMING_CSV}",
                      flush=True)
            if st == "COMPLETE":
                print(f"  public score {sub.public_score}  ({(sub.description or '')[:100]})", flush=True)
                return 0
            print(f"  ERROR: {getattr(sub, 'error_description', '') or '(no description)'}", flush=True)
            return 2
        time.sleep(a.every)
    print(f"still {last} after {a.max_h} h -- giving up", flush=True)
    return 3


if __name__ == "__main__":
    sys.exit(main())
