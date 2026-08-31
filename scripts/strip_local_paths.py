"""Rewrite absolute ``file:///a:/RAI/...`` links into repo-relative markdown links.

The comprehensive summary was authored in an editor that inserted absolute
``file://`` URIs pointing at the authoring workstation's ``A:`` drive. Those
resolve nowhere else, so they are rewritten to paths relative to the file's own
location in the destination tree before the document is published.

Usage:
    python scripts/strip_local_paths.py SOURCE DEST --rel-prefix ../..

``--rel-prefix`` is the path from the destination file back to the repository
root (e.g. ``../..`` for a file living in ``docs/history/``).
"""

from __future__ import annotations

import argparse
import re
import sys

# file:///a:/RAI/scripts/foo.py  ->  scripts/foo.py   (case-insensitive drive)
FILE_URI = re.compile(r"file:/{2,3}[A-Za-z]:/RAI/([^\s)\"'>\]]+)", re.I)
# Bare A:\RAI\... or a:/RAI/... outside a URI.
BARE_ABS = re.compile(r"\b[A-Za-z]:[\\/]{1,2}RAI[\\/]+([^\s)\"'>\]]+)", re.I)


def rewrite(text: str, rel_prefix: str) -> tuple[str, int]:
    prefix = rel_prefix.rstrip("/") + "/" if rel_prefix else ""
    count = 0

    def sub(match: re.Match) -> str:
        nonlocal count
        count += 1
        inner = match.group(1).replace("\\", "/")
        return prefix + inner

    text = FILE_URI.sub(sub, text)
    text = BARE_ABS.sub(sub, text)
    return text, count


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("source")
    ap.add_argument("dest")
    ap.add_argument("--rel-prefix", default="", help="path from dest file to repo root")
    args = ap.parse_args(argv)

    with open(args.source, "r", encoding="utf-8") as fh:
        text = fh.read()
    out, count = rewrite(text, args.rel_prefix)

    leftover = FILE_URI.findall(out) + BARE_ABS.findall(out)
    with open(args.dest, "w", encoding="utf-8", newline="\n") as fh:
        fh.write(out)

    print("rewrote %d absolute path(s) -> %r-relative" % (count, args.rel_prefix or "."))
    print("residual absolute paths: %d" % len(leftover))
    return 1 if leftover else 0


if __name__ == "__main__":
    sys.exit(main())
