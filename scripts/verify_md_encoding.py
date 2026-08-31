"""Encoding verification for every Markdown file in the repository.

Checks each ``*.md`` for the two failure modes a UTF-8 document picks up when it
has been round-tripped through a single-byte codepage:

1. **A UTF-8 BOM** (``EF BB BF``). Legal but unwanted -- it leaks into the first
   heading on some renderers and breaks ``grep '^#'``.
2. **Mojibake markers** -- the ``Â Ã â Å Ð Ÿ`` prefixes and any C1 control
   character (U+0080..U+009F). These appear when UTF-8 bytes were decoded as
   cp1252 and re-encoded, turning ``—`` into ``â€"`` and similar.

It also reports the count of the intended non-ASCII typography per file, so a
"clean" result can be distinguished from "the characters were stripped
entirely" -- a repair that removes every em dash also removes every marker.

    python scripts/verify_md_encoding.py

Exit status is 0 when no file carries a BOM or a residual marker, 1 otherwise.
Read-only: nothing is rewritten. The repair counterpart is
``scripts/fix_md_encoding.py``.
"""

from __future__ import annotations

import os
import re
import sys

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

SKIP_DIRS = {".git", "venv", ".venv", "__pycache__", "backups", "node_modules",
             ".pytest_cache", "rai.egg-info"}

MARKERS = "ÂÃâÅÐŸ"
C1 = "".join(chr(c) for c in range(0x80, 0xA0))
PAT = re.compile("[" + MARKERS + C1 + "]")

# Intended typography -- counted, not flagged.
GLYPHS = [("check", "✅"), ("cross", "❌"), ("emdash", "—"),
          ("endash", "–"), ("arrow", "→"), ("times", "×"),
          ("pm", "±"), ("minus", "−")]


def markdown_files() -> list:
    found = []
    for dirpath, dirnames, filenames in os.walk(PROJECT_ROOT):
        dirnames[:] = [d for d in dirnames if d not in SKIP_DIRS]
        for name in filenames:
            if name.lower().endswith(".md"):
                full = os.path.join(dirpath, name)
                found.append(os.path.relpath(full, PROJECT_ROOT).replace(os.sep, "/"))
    return sorted(found)


def main() -> int:
    files = markdown_files()
    print("scanning %d markdown file(s) under %s" % (len(files), PROJECT_ROOT))
    print()

    bad = 0
    for rel in files:
        raw = open(os.path.join(PROJECT_ROOT, rel), "rb").read()
        bom = raw.startswith(b"\xef\xbb\xbf")
        try:
            text = raw.decode("utf-8")
        except UnicodeDecodeError as exc:
            bad += 1
            print("  %-52s NOT VALID UTF-8: %s" % (rel, exc))
            continue
        hits = PAT.findall(text)
        if bom or hits:
            bad += 1
            print("  %-52s %s residual=%d %s"
                  % (rel, "BOM!" if bom else "    ", len(hits),
                     [hex(ord(c)) for c in hits[:8]]))
        else:
            print("  %-52s clean" % rel)

    print()
    print("  files with a BOM or residual mojibake markers: %d" % bad)
    print()
    print("  intended non-ASCII typography (a zero row would mean it was stripped, "
          "not repaired):")
    for rel in files:
        text = open(os.path.join(PROJECT_ROOT, rel), encoding="utf-8").read()
        counts = " ".join("%s=%d" % (label, text.count(glyph)) for label, glyph in GLYPHS)
        print("    %-52s %s" % (rel, counts))
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
