"""Pull the final leaderboards and our submissions after the deadline -> artifacts/lb_final/ (or --out).

After the deadline `kaggle competitions leaderboard -d` still downloads the PUBLIC board; only the paged
`get_leaderboard` view (the CLI's `-s`) shows the private one (traps 79). This script pages that view, downloads the
public CSV beside it, joins the two by TeamId, and lists every submission of ours with both scores.

Writes:
  private_lb.csv             team_id, team_name, submission_date, score  (private board, in rank order)
  private_public_merged.csv  + prank, urank (public rank), pub, SubmissionCount, TeamMemberUserNames
  submissions_clean.json     every submission of ours, raw API fields
  our_entries_pub_priv.csv   no (our #, ERRORs skipped), ref, date, pub, priv, desc, gap

Run from the repo root:  PYTHONUTF8=1 .venv/Scripts/python.exe src/pull_private_lb.py
Then rebuild the write-up:  .venv/Scripts/python.exe docs/writeup/build.py --data
"""
import argparse, csv, glob, json, os, subprocess, zipfile

import pandas as pd
from kaggle.api.kaggle_api_extended import KaggleApi
from kagglesdk.competitions.types.competition_api_service import ApiGetLeaderboardRequest

COMP = "rsna-knee-abnormality-detection"
OUR_TEAM_ID = 16783421


def page_private(api):
    rows, token = [], None
    while True:
        with api.build_kaggle_client() as k:
            req = ApiGetLeaderboardRequest()
            req.competition_name = COMP
            api._set_paging(req, 200, token)
            resp = k.competitions.competition_api_client.get_leaderboard(req)
        subs = resp.submissions or []
        rows += [dict(team_id=s.team_id, team_name=s.team_name, submission_date=s.submission_date, score=s.score)
                 for s in subs]
        token = resp.next_page_token
        if not token or not subs:
            return rows


def our_submissions(out):
    # the CLI prints a version warning on stdout before the JSON
    txt = subprocess.run(["kaggle", "competitions", "submissions", COMP, "--format", "json", "--page-size", "200"],
                         capture_output=True, text=True, encoding="utf-8", check=True).stdout
    subs = json.loads(txt[txt.index("["):])
    json.dump(subs, open(os.path.join(out, "submissions_clean.json"), "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    rows, n = [], 0
    for r in sorted(subs, key=lambda r: r["date"]):
        if "ERROR" in r["status"] or r["publicScore"] in (None, ""):
            continue
        n += 1
        rows.append(dict(no=n, ref=r["ref"], date=r["date"][:10], pub=float(r["publicScore"]),
                         priv=float(r["privateScore"]), desc=r["description"][:60]))
    t = pd.DataFrame(rows)
    t["gap"] = (t.pub - t.priv).round(3)
    t.to_csv(os.path.join(out, "our_entries_pub_priv.csv"), index=False)
    return t


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="artifacts/lb_final")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    api = KaggleApi()
    api.authenticate()

    priv = page_private(api)
    with open(os.path.join(a.out, "private_lb.csv"), "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(priv[0]))
        w.writeheader()
        w.writerows(priv)

    api.competition_leaderboard_cli(COMP, path=a.out, download=True, quiet=True)
    for z in glob.glob(os.path.join(a.out, "*.zip")):
        zipfile.ZipFile(z).extractall(a.out)
    pub_csv = max(glob.glob(os.path.join(a.out, "*publicleaderboard*.csv")), key=os.path.getmtime)
    pv = pd.read_csv(os.path.join(a.out, "private_lb.csv"))
    pv["prank"] = range(1, len(pv) + 1)
    pb = pd.read_csv(pub_csv, encoding="utf-8-sig").rename(columns={"Rank": "urank", "TeamId": "team_id", "Score": "pub"})
    m = pv.merge(pb[["team_id", "urank", "pub", "SubmissionCount", "TeamMemberUserNames"]], on="team_id", how="left")
    m.to_csv(os.path.join(a.out, "private_public_merged.csv"), index=False)

    t = our_submissions(a.out)
    me = m[m.team_id == OUR_TEAM_ID].iloc[0]
    print(f"{len(pv)} teams; us: private {me.score:.3f} rank {me.prank}, public {me.pub:.3f} rank {me.urank}; "
          f"{len(t)} scored entries, best private {t.priv.max():.3f}")


if __name__ == "__main__":
    main()
