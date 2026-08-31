"""Pre-publish repository hygiene scanner (read-only).

Three independent scans, none of which mutate the repository:

1. ``scan_paths``   -- local filesystem paths / personal usernames leaked into text files.
2. ``scan_secrets`` -- API-key / token / credential shaped strings.
3. ``scan_bundled_data`` -- data files that ship *in* the repo rather than being
   downloaded at runtime, so redistribution rights can be checked by hand.

Run with no arguments to print all three reports:

    venv/Scripts/python.exe scripts/prepublish_hygiene_scan.py

Nothing here deletes, rewrites or stages anything; it prints findings only.
"""

from __future__ import annotations

import getpass
import os
import re
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

# Directories never worth scanning: third-party code, our own timestamped backups,
# git internals and caches. ``backups/`` is excluded from *content* scans but is
# reported separately by the size/ignore audit, because its contents are copies of
# files that are themselves scanned.
SKIP_DIRS = {
    "venv",
    ".venv",
    "backups",
    ".git",
    "__pycache__",
    ".pytest_cache",
    "node_modules",
    "rai.egg-info",
    ".eggs",
    ".mypy_cache",
}

TEXT_SUFFIXES = {
    ".md",
    ".py",
    ".toml",
    ".txt",
    ".cfg",
    ".ini",
    ".yaml",
    ".yml",
    ".json",
    ".ipynb",
    ".sh",
    ".bat",
}

DATA_SUFFIXES = {
    ".csv",
    ".parquet",
    ".json",
    ".jsonl",
    ".npy",
    ".npz",
    ".pkl",
    ".h5",
    ".xlsx",
    ".zip",
    ".pt",
    ".pth",
}


def iter_text_files():
    for path in sorted(ROOT.rglob("*")):
        if not path.is_file():
            continue
        rel = path.relative_to(ROOT)
        if any(part in SKIP_DIRS for part in rel.parts):
            continue
        if path.suffix.lower() not in TEXT_SUFFIXES:
            continue
        yield rel, path


def read(path: Path) -> str:
    try:
        return path.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return ""


# --------------------------------------------------------------------------- #
# 1. Local filesystem paths and personal identifiers
# --------------------------------------------------------------------------- #

BACKSLASH = chr(92)

PATH_PATTERNS = {
    # C:\Users\someone  /  C:/Users/someone
    "win_user_dir": re.compile(
        r"[A-Za-z]:[" + BACKSLASH + r"/]+Users[" + BACKSLASH + r"/]+[^" + BACKSLASH + r"/\s\"')\]]+",
        re.I,
    ),
    # /home/someone  or  /Users/someone
    "unix_home_dir": re.compile(r"/(?:home|Users)/[A-Za-z0-9._-]+"),
    # Any other absolute drive-letter path, e.g. A:\RAI\...
    "drive_abs_path": re.compile(
        r"\b[A-Za-z]:[" + BACKSLASH + r"/]{1,2}(?!Users)[A-Za-z0-9._-]+", re.I
    ),
    "appdata": re.compile(r"AppData[" + BACKSLASH + r"/]", re.I),
    # Kaggle notebook-only absolute paths -- expected in kaggle_*.py, not a leak,
    # but worth listing so the reviewer can confirm they are guarded.
    "kaggle_abs_path": re.compile(r"/kaggle/(?:input|working)[^\s\"')]*"),
}

# The account name of whoever runs the scan, so a workstation username that
# leaked into a docstring or a hard-coded path is caught. Taken from the
# environment rather than hard-coded, both for portability and so this file does
# not itself publish an account name. Override with RAI_SCAN_USERNAMES (a
# comma-separated list) to check for additional names, e.g. a co-author's.
_names = [
    n.strip()
    for n in os.environ.get("RAI_SCAN_USERNAMES", "").split(",")
    if n.strip()
]
try:
    _names.append(getpass.getuser())
except Exception:  # no password database entry / no env vars set
    pass
_names = [n for n in dict.fromkeys(_names) if len(n) >= 3]
if _names:
    PATH_PATTERNS["local_username"] = re.compile(
        "|".join(re.escape(n) for n in _names), re.I
    )


def scan_paths():
    hits = []
    for rel, path in iter_text_files():
        for lineno, line in enumerate(read(path).splitlines(), 1):
            for name, rx in PATH_PATTERNS.items():
                for match in rx.finditer(line):
                    hits.append((name, str(rel).replace(BACKSLASH, "/"), lineno, match.group(0), line.strip()))
    return hits


# --------------------------------------------------------------------------- #
# 2. Credential-shaped strings
# --------------------------------------------------------------------------- #

SECRET_PATTERNS = {
    "aws_access_key_id": re.compile(r"\b(?:AKIA|ASIA)[0-9A-Z]{16}\b"),
    "github_token": re.compile(r"\bgh[pousr]_[A-Za-z0-9]{36,}\b"),
    "anthropic_key": re.compile(r"\bsk-ant-[A-Za-z0-9_-]{20,}"),
    "openai_key": re.compile(r"\bsk-[A-Za-z0-9]{32,}\b"),
    "slack_token": re.compile(r"\bxox[abprs]-[A-Za-z0-9-]{10,}"),
    "google_api_key": re.compile(r"\bAIza[0-9A-Za-z_-]{35}\b"),
    "private_key_block": re.compile(r"-----BEGIN [A-Z ]*PRIVATE KEY-----"),
    "jwt": re.compile(r"\beyJ[A-Za-z0-9_-]{10,}\.[A-Za-z0-9_-]{10,}\.[A-Za-z0-9_-]{10,}\b"),
    "bearer_literal": re.compile(r"Bearer\s+[A-Za-z0-9._-]{20,}"),
    # Assignment of a non-placeholder-looking literal to a secret-ish name.
    "assigned_secret": re.compile(
        r"(?i)\b(?:api[_-]?key|secret[_-]?key|access[_-]?token|auth[_-]?token|password|passwd|credential)"
        r"\s*[:=]\s*[\"'][^\"']{8,}[\"']"
    ),
    # kaggle.json shape.
    "kaggle_credentials": re.compile(r"(?i)[\"']?(?:username|key)[\"']?\s*:\s*[\"'][A-Za-z0-9]{16,}[\"']"),
}

# Obvious non-secrets that the assignment pattern will otherwise flag.
PLACEHOLDER_HINTS = (
    "your_",
    "xxx",
    "placeholder",
    "example",
    "changeme",
    "<",
    "dummy",
    "todo",
    "redacted",
    "os.environ",
    "getenv",
)


def scan_secrets():
    hits = []
    for rel, path in iter_text_files():
        for lineno, line in enumerate(read(path).splitlines(), 1):
            for name, rx in SECRET_PATTERNS.items():
                for match in rx.finditer(line):
                    token = match.group(0)
                    low = token.lower()
                    placeholder = any(hint in low for hint in PLACEHOLDER_HINTS)
                    hits.append(
                        (
                            name,
                            str(rel).replace(BACKSLASH, "/"),
                            lineno,
                            token[:80],
                            "LIKELY-PLACEHOLDER" if placeholder else "REVIEW",
                        )
                    )
    return hits


# --------------------------------------------------------------------------- #
# 3. Bundled (not runtime-downloaded) data files
# --------------------------------------------------------------------------- #


def scan_bundled_data():
    rows = []
    for path in sorted(ROOT.rglob("*")):
        if not path.is_file():
            continue
        rel = path.relative_to(ROOT)
        if any(part in SKIP_DIRS for part in rel.parts):
            continue
        if path.suffix.lower() not in DATA_SUFFIXES:
            continue
        try:
            size = path.stat().st_size
        except OSError:
            size = -1
        rows.append((str(rel).replace(BACKSLASH, "/"), size, path.suffix.lower()))
    return rows


def main() -> int:
    print("=" * 100)
    print("1. LOCAL FILESYSTEM PATHS / PERSONAL IDENTIFIERS  (report only -- nothing removed)")
    print("=" * 100)
    path_hits = scan_paths()
    print("total hits: %d" % len(path_hits))
    print(dict(Counter(h[0] for h in path_hits)))
    print()
    seen = set()
    for name, rel, lineno, token, line in path_hits:
        key = (name, rel, token)
        if key in seen:
            continue
        seen.add(key)
        print("[%-15s] %s:%d" % (name, rel, lineno))
        print("    token: %s" % token)
        print("    line : %s" % line[:150])
    print()

    print("=" * 100)
    print("2. CREDENTIAL-SHAPED STRINGS")
    print("=" * 100)
    secret_hits = scan_secrets()
    print("total hits: %d" % len(secret_hits))
    if not secret_hits:
        print("no matches for any of: %s" % ", ".join(sorted(SECRET_PATTERNS)))
    for name, rel, lineno, token, verdict in secret_hits:
        print("[%-20s] %-12s %s:%d  %s" % (name, verdict, rel, lineno, token))
    print()

    print("=" * 100)
    print("3. BUNDLED DATA FILES (present on disk, not runtime-only)")
    print("=" * 100)
    rows = scan_bundled_data()
    total = sum(size for _, size, _ in rows if size > 0)
    print("total: %d files, %.1f MB" % (len(rows), total / 1048576))
    by_dir = Counter()
    size_by_dir = Counter()
    for rel, size, _ in rows:
        top = rel.split("/")[0] if "/" in rel else "(root)"
        by_dir[top] += 1
        size_by_dir[top] += max(size, 0)
    print()
    print("%-28s %6s  %10s" % ("directory", "files", "MB"))
    for top, count in by_dir.most_common():
        print("%-28s %6d  %10.2f" % (top, count, size_by_dir[top] / 1048576))
    print()
    print("largest 30 bundled data files:")
    for rel, size, _ in sorted(rows, key=lambda r: -r[1])[:30]:
        print("  %10.3f MB  %s" % (size / 1048576, rel))
    return 0


if __name__ == "__main__":
    sys.exit(main())
