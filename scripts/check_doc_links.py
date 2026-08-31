"""Check every relative link and internal anchor in the repository's Markdown.

Three classes of link are resolved:

* **File links** -- ``[text](path/to/file.md)``, ``(scripts/foo.py)``, image
  ``![alt](data/x.png)``. Resolved relative to the linking file's own directory
  and reported as ``MISSING`` if nothing is there.
* **Anchor-only links** -- ``[text](#some-heading)``. Checked against the
  headings of the linking file, using GitHub's slug rules.
* **File + anchor** -- ``[text](docs/other.md#heading)``. Both halves checked.

Absolute URLs (``http:``, ``https:``, ``mailto:``) are skipped, except that a
``file://`` URL is always an error: it can only resolve on the machine that
wrote it.

GitHub's heading slug rules, as implemented here: lower-case, strip everything
that is not a word character / space / hyphen (so ``&``, ``.``, backticks, em
dashes and ``/`` all vanish), then spaces to hyphens. Note that stripping
punctuation *before* replacing spaces is what produces the double hyphens in
slugs like ``#1-the-model--what-it-is``.

    python scripts/check_doc_links.py

Exit status is 0 when every link resolves, 1 otherwise. Read-only.
"""

from __future__ import annotations

import os
import re
import sys
from urllib.parse import unquote

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

SKIP_DIRS = {".git", "venv", ".venv", "__pycache__", "backups", "node_modules",
             ".pytest_cache", "rai.egg-info"}

# [text](target) -- target captured up to the closing paren. Titles ("...") are
# not used anywhere in this repo, so they are not parsed.
LINK = re.compile(r"!?\[(?P<text>[^\]]*)\]\((?P<target>[^)\s]+)\)")
ATX_HEADING = re.compile(r"^(#{1,6})\s+(.*?)\s*#*\s*$")
FENCE = re.compile(r"^\s*(```|~~~)")

SKIP_SCHEMES = ("http://", "https://", "mailto:", "tel:", "data:")


def slugify(heading: str) -> str:
    """GitHub's anchor slug for a heading's rendered text."""
    text = heading.strip()
    text = re.sub(r"`([^`]*)`", r"\1", text)          # inline code -> contents
    text = re.sub(r"!?\[([^\]]*)\]\([^)]*\)", r"\1", text)  # links -> link text
    text = text.replace("*", "").replace("_", "")     # emphasis markers
    text = text.lower()
    text = re.sub(r"[^\w\s-]", "", text, flags=re.UNICODE)
    return text.strip().replace(" ", "-")


def headings_of(path: str) -> set:
    """All anchor slugs a Markdown file exposes, with GitHub's -1/-2 suffixes."""
    slugs = set()
    seen = {}
    in_fence = False
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            if FENCE.match(line):
                in_fence = not in_fence
                continue
            if in_fence:
                continue
            match = ATX_HEADING.match(line.rstrip("\n"))
            if not match:
                continue
            base = slugify(match.group(2))
            if not base:
                continue
            count = seen.get(base, 0)
            seen[base] = count + 1
            slugs.add(base if count == 0 else "%s-%d" % (base, count))
    return slugs


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
    anchors = {rel: headings_of(os.path.join(PROJECT_ROOT, rel)) for rel in files}

    checked = 0
    problems = []
    per_file = {}

    for rel in files:
        base_dir = os.path.dirname(os.path.join(PROJECT_ROOT, rel))
        ok_here = 0
        in_fence = False
        with open(os.path.join(PROJECT_ROOT, rel), encoding="utf-8") as fh:
            for lineno, line in enumerate(fh, 1):
                if FENCE.match(line):
                    in_fence = not in_fence
                    continue
                if in_fence:
                    continue
                for match in LINK.finditer(line):
                    target = match.group("target").strip()
                    if target.startswith("file://"):
                        problems.append((rel, lineno, target, "file:// URL -- machine-local"))
                        continue
                    if target.lower().startswith(SKIP_SCHEMES):
                        continue
                    checked += 1
                    path_part, _, anchor = target.partition("#")
                    path_part = unquote(path_part)

                    if not path_part:                     # same-file anchor
                        if anchor and anchor.lower() not in anchors[rel]:
                            problems.append((rel, lineno, target, "anchor not found in this file"))
                        else:
                            ok_here += 1
                        continue

                    resolved = os.path.normpath(os.path.join(base_dir, path_part))
                    if not os.path.exists(resolved):
                        problems.append((rel, lineno, target, "MISSING target"))
                        continue
                    if anchor:
                        target_rel = os.path.relpath(resolved, PROJECT_ROOT).replace(os.sep, "/")
                        if target_rel in anchors:
                            if anchor.lower() not in anchors[target_rel]:
                                problems.append((rel, lineno, target,
                                                 "anchor not found in %s" % target_rel))
                                continue
                        # Anchor into a non-markdown file (e.g. #L42) -- not checkable.
                    ok_here += 1
        per_file[rel] = ok_here

    print("checked %d relative link(s) across %d markdown file(s)" % (checked, len(files)))
    print()
    for rel in files:
        print("  %-52s %3d ok" % (rel, per_file[rel]))
    print()
    if problems:
        print("  UNRESOLVED (%d):" % len(problems))
        for rel, lineno, target, why in problems:
            print("    %s:%d  %s  -- %s" % (rel, lineno, target, why))
    else:
        print("  every relative link and internal anchor resolves.")
    return 1 if problems else 0


if __name__ == "__main__":
    sys.exit(main())
