"""Run nda-tools' downloadcmd with the NDA credentials from a .env file, held in memory only (P-81, OAI).

Usage: python scripts/nda_run.py <path/to/.env> <downloadcmd args...>     (needs `pip install nda-tools`)
  list a package without downloading:  ... -dp 1249779 --verify -d <dir>
  only the small metadata files:       ... -dp 1249779 --file-regex '^[a-z_0-9]+\.(txt|pdf)$' -d data/oai/nda_pkg_meta
  a chosen subset:                     ... -dp 1249779 -t <s3-links.txt> -d <dir> -wt 16
The .env needs NDA_USERNAME (the NDA account username from nda.nih.gov/user/dashboard/profile, not the Login.gov email)
and NDA_PASSWORD (set there with "Update Password"; the Login.gov password is rejected with a 401).
The password is served by an in-memory keyring backend: never printed, never prompted for, never written to the
OS credential store (nda-tools tries to save it after login; the backend's set_password is a no-op).
"""
import os
import sys

import keyring
from keyring.backend import KeyringBackend


def load_env(path):
    vals = {}
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            k, v = line.split("=", 1)
            vals[k.strip()] = v.strip().strip('"').strip("'")
    return vals


class EnvKeyring(KeyringBackend):
    priority = 100

    def __init__(self, user, pw):
        super().__init__()
        self._user, self._pw = user, pw

    def get_password(self, service, username):
        return self._pw if username.lower() == self._user.lower() else None

    def set_password(self, service, username, password):
        pass

    def delete_password(self, service, username):
        pass


env = load_env(sys.argv[1])
user, pw = env["NDA_USERNAME"], env["NDA_PASSWORD"]
keyring.set_keyring(EnvKeyring(user, pw))
args = sys.argv[2:]
if "-u" not in args and "--username" not in args:
    args = ["-u", user] + args
sys.argv = ["downloadcmd"] + args
from NDATools.clientscripts.downloadcmd import main  # noqa: E402

main()
