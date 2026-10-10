"""Watch one Kaggle kernel run until it ends: a status timeline, any mid-run log lines, then its small outputs.

    python src/watch_kernel.py tiankljucanin/rsna-knee-train --out artifacts/kaggle_out/sA_1010 \
        [--every 600] [--max-hours 10] [--pattern "\\.log$|_oof\\.csv$"] [--key "fold results" ...]

Run it in the background with stdout redirected to `artifacts/watch_<name>.log`; the timeline is that file.
- Every poll is a fresh `kaggle` CLI process. A long-lived KaggleApi never refreshes its token (traps 20 addendum), and a
  CLI call inside the dead half hour after expiry fails once and refreshes on the next poll, so a failed poll is logged and
  retried, never fatal.
- `kaggle kernels logs <slug>` is tried on every poll. It is blank mid-run for our notebook kernels (10-10 check), so the
  timeline is usually status only; any lines it does return are appended.
- When the status leaves QUEUED / RUNNING, the outputs matching --pattern are downloaded (never the checkpoints by default:
  `_best.pt` is hundreds of MB for some arms) and the lines containing a --key are printed from every downloaded .log.

Exit code: 0 = COMPLETE, 1 = ERROR / CANCEL / anything else, 2 = still running after --max-hours.
"""
import argparse
import glob
import json
import os
import re
import subprocess
import sys
import time
from datetime import datetime, timezone

KAGGLE = os.path.join(os.path.dirname(sys.executable), "kaggle.exe" if os.name == "nt" else "kaggle")
DEFAULT_KEYS = ("fold results", "ok  arm", "FAIL", "Traceback", "Error", "-> ", "gold macro", "runtime guard",
                "reseeded", "teacher table", "out of time", "completed")


def now():
    return datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%SZ")


def say(msg):
    print(f"{now()}  {msg}", flush=True)


def cli(*args, timeout=180):
    env = {**os.environ, "PYTHONUTF8": "1"}
    try:
        r = subprocess.run([KAGGLE, *args], capture_output=True, text=True, encoding="utf-8", errors="replace",
                           timeout=timeout, env=env)
        return r.returncode, (r.stdout or "") + (r.stderr or "")
    except subprocess.TimeoutExpired:
        return -1, "timeout"


def status(slug):
    rc, out = cli("kernels", "status", slug)
    m = re.search(r'status "(?:KernelWorkerStatus\.)?([A-Z_]+)"', out)
    return (m.group(1) if m else None), out.strip()


def log_lines(path):
    """The CLI's log file is JSON (a list of {stream_name, data}) or one JSON object per line; plain text otherwise."""
    with open(path, encoding="utf-8", errors="replace") as f:
        raw = f.read()
    try:
        events = json.loads(raw)
        return [ln for ev in events if isinstance(ev, dict) for ln in str(ev.get("data", "")).rstrip("\n").split("\n")]
    except json.JSONDecodeError:
        pass
    lines = []
    for ln in raw.splitlines():
        s = ln.strip().lstrip(",").rstrip(",")
        if s.startswith("{"):
            try:
                lines.extend(str(json.loads(s).get("data", "")).rstrip("\n").split("\n"))
                continue
            except json.JSONDecodeError:
                pass
        lines.append(ln)
    return lines


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("slug")
    ap.add_argument("--out", required=True, help="directory for the downloaded outputs")
    ap.add_argument("--every", type=int, default=600, help="seconds between polls")
    ap.add_argument("--max-hours", type=float, default=10.0)
    ap.add_argument("--pattern", default=r"\.log$|_oof\.csv$", help="regex of output files to download at the end")
    ap.add_argument("--key", action="append", help="substring to print from the downloaded logs (repeatable)")
    a = ap.parse_args()
    keys = tuple(a.key) if a.key else DEFAULT_KEYS
    os.makedirs(a.out, exist_ok=True)

    t0, last, seen_log = time.time(), None, 0
    say(f"watching {a.slug} every {a.every} s (max {a.max_hours} h); outputs -> {a.out}")
    while True:
        st, raw = status(a.slug)
        if st is None:
            say(f"status call failed (retrying next poll): {raw[:160]!r}")
        elif st != last:
            say(f"status {st}  (+{(time.time() - t0) / 60:.1f} min)")
            last = st
        rc, out = cli("kernels", "logs", a.slug, timeout=120)
        new = [ln for ln in out.splitlines() if ln.strip()]
        if rc == 0 and len(new) > seen_log:
            for ln in new[seen_log:][-30:]:
                say(f"  log| {ln[:240]}")
            seen_log = len(new)
        if st is not None and st not in ("QUEUED", "RUNNING"):
            break
        if time.time() - t0 > a.max_hours * 3600:
            say(f"still {last} after {a.max_hours} h -- giving up (the kernel keeps running)")
            sys.exit(2)
        time.sleep(a.every)

    say(f"ended {st} after {(time.time() - t0) / 3600:.2f} h of watching; downloading /{a.pattern}/ ...")
    rc, out = cli("kernels", "output", a.slug, "-p", a.out, "--file-pattern", a.pattern, timeout=1800)
    say(f"output rc {rc}: {out.strip()[-300:]!r}")
    files = sorted(glob.glob(os.path.join(a.out, "**", "*"), recursive=True))
    say(f"{len(files)} files: {[os.path.basename(f) for f in files][:40]}")
    for f in files:
        if f.endswith(".log"):
            hits = [ln for ln in log_lines(f) if any(k in ln for k in keys)]
            say(f"--- {os.path.basename(f)}: {len(hits)} key lines (last 40)")
            for ln in hits[-40:]:
                print(f"    {ln[:240]}", flush=True)
    sys.exit(0 if st == "COMPLETE" else 1)


if __name__ == "__main__":
    main()
