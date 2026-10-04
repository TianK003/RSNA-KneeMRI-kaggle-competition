"""Send a planned list of code-competition submissions at a set UTC time, unattended, then watch them score.

Written 2026-10-04 for the 10-05 lineup (Tian: "auto-push at reset"). The submission count resets at 00:00 UTC, so the
plan waits until --at, then submits each entry in order (put the slow public-stack fork first), records each ref, starts
`src/watch_submission.py` per ref (timing -> artifacts/submission_timing.csv) and prints a score summary when all are scored.

    .venv/Scripts/python.exe src/auto_submit.py --plan artifacts/submit_plan_1005.json --at 2026-10-05T00:00:30Z
    .venv/Scripts/python.exe src/auto_submit.py --plan artifacts/submit_plan_1005.json --dry-run    # checks only

Plan file: a JSON list of {"slug": "tiankljucanin/rsna-knee-infer", "version": 47, "message": "..."}.

Safety:
- a pre-flight API call (with retries) runs before the first submit; on this laptop's OAuth credentials the first call
  >= 30 min after expiry refreshes the token (traps 20);
- a submit is retried only after checking that no submission with the same description exists since the start, so a
  network error never double-spends a slot;
- while it runs, Windows is asked not to sleep (SetThreadExecutionState); closing the lid can still force sleep.
"""
import argparse
import datetime as dt
import json
import os
import subprocess
import sys
import time

COMPETITION = "rsna-knee-abnormality-detection"
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
VENV = os.path.join(ROOT, ".venv", "Scripts")
KAGGLE = os.path.join(VENV, "kaggle.exe") if os.name == "nt" else "kaggle"
PY = os.path.join(VENV, "python.exe") if os.name == "nt" else sys.executable


def now():
    return dt.datetime.now(dt.timezone.utc)


def log(msg):
    print(f"{now():%Y-%m-%d %H:%M:%S}Z  {msg}", flush=True)


def keep_awake():
    if os.name == "nt":
        import ctypes
        ES_CONTINUOUS, ES_SYSTEM_REQUIRED = 0x80000000, 0x00000001
        ctypes.windll.kernel32.SetThreadExecutionState(ES_CONTINUOUS | ES_SYSTEM_REQUIRED)
        log("Windows sleep blocked while this process runs (lid close can still force sleep)")


def api_client():
    from kaggle.api.kaggle_api_extended import KaggleApi
    api = KaggleApi()
    api.authenticate()
    return api


def list_subs(api, tries=8, wait=300):
    for i in range(tries):
        try:
            return api.competition_submissions(COMPETITION) or []
        except Exception as e:                       # token window / 429: wait and retry
            log(f"api error ({type(e).__name__}: {str(e)[:120]}) -- retry {i + 1}/{tries} in {wait} s")
            time.sleep(wait)
    raise SystemExit("API unreachable after retries -- nothing submitted past this point")


def sub_date(s):
    d = s.date
    return d.replace(tzinfo=dt.timezone.utc) if d.tzinfo is None else d.astimezone(dt.timezone.utc)


def find(api, message, since):
    for s in list_subs(api):
        if (s.description or "").strip() == message.strip() and sub_date(s) >= since - dt.timedelta(minutes=2):
            return s
    return None


def submit(api, entry, since):
    cmd = [KAGGLE, "competitions", "submit", COMPETITION, "-k", entry["slug"], "-v", str(entry["version"]),
           "-f", "submission.csv", "-m", entry["message"]]
    for attempt in range(1, 4):
        hit = find(api, entry["message"], since)
        if hit is not None:
            log(f"already submitted (ref {hit.ref}) -- not resubmitting")
            return hit
        log(f"submit {entry['slug']} v{entry['version']} (attempt {attempt})")
        try:
            r = subprocess.run(cmd, capture_output=True, text=True, timeout=900, cwd=ROOT)
            out = (r.stdout + r.stderr).strip().replace("\n", " | ")
            log(f"  rc={r.returncode}: {out[:300]}")
        except Exception as e:
            log(f"  submit raised {type(e).__name__}: {str(e)[:200]}")
        time.sleep(20)
        hit = find(api, entry["message"], since)
        if hit is not None:
            log(f"  -> ref {hit.ref}, status {hit.status}")
            return hit
        time.sleep(120 * attempt)
    log("  FAILED after 3 attempts -- moving on")
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--plan", required=True)
    ap.add_argument("--at", default=None, help="UTC start, e.g. 2026-10-05T00:00:30Z (default: now)")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--watch-every", type=int, default=90)
    ap.add_argument("--watch-max-h", type=float, default=14.0)
    a = ap.parse_args()
    os.chdir(ROOT)

    plan = json.load(open(a.plan, encoding="utf-8"))
    for e in plan:
        assert {"slug", "version", "message"} <= set(e), e
    log(f"plan {a.plan}: {len(plan)} submissions")
    for i, e in enumerate(plan, 1):
        log(f"  {i}. {e['slug']} v{e['version']}: {e['message'][:110]}")
    if len(plan) > 5:
        raise SystemExit("more than 5 entries: the daily limit is 5")

    keep_awake()
    api = api_client()
    if a.dry_run:
        subs = list_subs(api, tries=2, wait=30)
        log(f"dry run: API ok, {len(subs)} submissions listed, newest ref {subs[0].ref if subs else None}; nothing sent")
        return 0

    if a.at:
        start = dt.datetime.fromisoformat(a.at.replace("Z", "+00:00"))
        while now() < start:
            left = (start - now()).total_seconds()
            if left > 600 and int(left) % 3600 < 60:
                log(f"waiting for {start:%Y-%m-%d %H:%M:%S}Z ({left / 3600:.1f} h)")
            time.sleep(min(60, max(1, left)))
    since = now()
    subs = list_subs(api)                            # pre-flight: also refreshes an expired token
    log(f"pre-flight ok: {len(subs)} submissions listed")

    sent, watchers = [], []
    for e in plan:
        s = submit(api, e, since)
        if s is None:
            continue
        sent.append((e, s.ref))
        wlog = open(os.path.join("artifacts", f"watch_{s.ref}.log"), "w", encoding="utf-8")
        watchers.append(subprocess.Popen([PY, os.path.join("src", "watch_submission.py"), "--ref", str(s.ref),
                                          "--every", str(a.watch_every), "--max-h", str(a.watch_max_h)],
                                         stdout=wlog, stderr=subprocess.STDOUT, cwd=ROOT))
        log(f"  watcher started -> artifacts/watch_{s.ref}.log")
        time.sleep(20)
    log(f"sent {len(sent)} / {len(plan)}; waiting for the watchers")
    for p in watchers:
        p.wait()

    log("SUMMARY")
    final = {str(s.ref): s for s in list_subs(api)}
    for e, ref in sent:
        s = final.get(str(ref))
        st = str(getattr(s, "status", "?")).split(".")[-1]
        log(f"  ref {ref}  {e['slug'].split('/')[-1]} v{e['version']}  {st}  public {getattr(s, 'public_score', None)}  "
            f"| {e['message'][:80]}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
