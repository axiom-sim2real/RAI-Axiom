"""
================================================================================
  MARKDOWN ENCODING REPAIR  (TASK 4)
================================================================================
  Repairs UTF-8 text that was decoded as cp1252 and re-encoded as UTF-8
  ("mojibake": em-dash -> a-circumflex + euro + quote, etc.), and rewrites every
  .md file as UTF-8 WITHOUT a BOM.

  Method: scan each maximal run of non-ASCII characters and greedily re-decode
  the longest prefix that round-trips through cp1252 -> utf-8. Text that is
  already correct fails that round-trip (e.g. U+2014 -> byte 0x97, which is not
  valid UTF-8 on its own) and is therefore left untouched. This makes the repair
  safe to run on files that mix correct and corrupted characters, and idempotent.

  Usage:
    python scripts/fix_md_encoding.py --check     # report only
    python scripts/fix_md_encoding.py             # repair in place
    python scripts/fix_md_encoding.py --include-backups
================================================================================
"""

import os
import re
import sys
import argparse

if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SKIP_DIRS = {".git", "venv", "node_modules", "__pycache__", ".pytest_cache", "rai.egg-info"}

NON_ASCII_RUN = re.compile(r"[^\x00-\x7F]+")

# Characters that only ever appear in mojibake, used as a detection signal.
MOJI_MARKERS = ("Â", "Ã", "â", "Å", "Ð", "Ÿ")


def to_bytes(s):
    """cp1252 encode, falling back to latin-1 for the five cp1252-undefined
    positions (0x81, 0x8D, 0x8F, 0x90, 0x9D). Mojibake that passed through a
    lenient decoder keeps those bytes as raw C1 control characters, so e.g.
    "❌" (E2 9D 8C) arrives as U+00E2 U+009D U+0152 and a pure cp1252 encode
    would raise on U+009D."""
    out = bytearray()
    for ch in s:
        try:
            out += ch.encode("cp1252")
        except UnicodeEncodeError:
            out += ch.encode("latin-1")
    return bytes(out)


def fix_run(run):
    """Greedily re-decode the longest prefixes of a non-ASCII run."""
    out = []
    i = 0
    n = len(run)
    while i < n:
        best = None
        for j in range(n, i, -1):
            chunk = run[i:j]
            try:
                decoded = to_bytes(chunk).decode("utf-8")
            except (UnicodeEncodeError, UnicodeDecodeError):
                continue
            if decoded != chunk:
                best = (j, decoded)
                break
        if best is None:
            out.append(run[i])
            i += 1
        else:
            out.append(best[1])
            i = best[0]
    return "".join(out)


def fix_text(text):
    return NON_ASCII_RUN.sub(lambda m: fix_run(m.group(0)), text)


def count_markers(text):
    return sum(text.count(m) for m in MOJI_MARKERS)


def iter_md(include_backups):
    for root, dirs, files in os.walk(PROJECT_ROOT):
        dirs[:] = [d for d in dirs if d not in SKIP_DIRS]
        if not include_backups:
            dirs[:] = [d for d in dirs if d != "backups"]
        for fn in files:
            if fn.lower().endswith(".md"):
                yield os.path.join(root, fn)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true", help="report only, do not write")
    ap.add_argument("--include-backups", action="store_true")
    args = ap.parse_args()

    n_files = n_bom = n_moji = 0
    for path in sorted(iter_md(args.include_backups)):
        raw = open(path, "rb").read()
        bom = raw.startswith(b"\xef\xbb\xbf")
        body = raw[3:] if bom else raw
        try:
            text = body.decode("utf-8")
            enc = "utf-8"
        except UnicodeDecodeError:
            text = body.decode("cp1252", errors="replace")
            enc = "cp1252-fallback"
        before = count_markers(text)
        fixed = fix_text(text)
        after = count_markers(fixed)
        rel = os.path.relpath(path, PROJECT_ROOT)
        n_files += 1
        changed = bom or fixed != text or enc != "utf-8"
        if not changed:
            continue
        if bom:
            n_bom += 1
        if before:
            n_moji += 1
        print("  %-52s bom=%d enc=%-15s mojibake markers %d -> %d"
              % (rel, int(bom), enc, before, after))
        if not args.check:
            with open(path, "wb") as f:
                f.write(fixed.encode("utf-8"))
    verb = "would fix" if args.check else "fixed"
    print("\n  scanned %d .md files; %s %d with mojibake, %d with a BOM"
          % (n_files, verb, n_moji, n_bom))


if __name__ == "__main__":
    main()
