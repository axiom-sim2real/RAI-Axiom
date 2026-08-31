"""Fetch -- or verify -- the pinned real-market price windows behind ``data/``.

This repository deliberately ships **no** Yahoo Finance price data: ``yfinance``
carries no redistribution licence for the underlying quotes. Every committed
result file in ``data/`` was nevertheless produced from a fixed, explicitly dated
price window, so the inputs can be regenerated and the evaluation re-run.

    python scripts/fetch_pinned_universes.py            # download what is missing
    python scripts/fetch_pinned_universes.py --verify   # check against the manifest
    python scripts/fetch_pinned_universes.py --force    # re-download everything

What is pinned, and why it matters
----------------------------------
* Window: ``start`` = 2016-08-20 (10-year universes) / 2021-08-20 (5-year
  universes), ``end`` = 2026-08-20, ``auto_adjust=True``. These reproduce the
  relative ``period="10y"`` / ``"5y"`` downloads of the original Kaggle run.
* **Column order is load-bearing.** Columns are left exactly as ``yfinance``
  returns them -- alphabetical by ticker, which is *not* the order the ticker
  lists are written in. The policy's per-asset logits are position-dependent, so
  sorting columns differently changes every number in ``data/``.
* Rows with any missing value are dropped (``dropna()``) before caching.

``--verify`` checks each file against ``data/pinned_universes_manifest.json`` on
shape, date range, column order, and a float fingerprint (every value rounded to
6 decimal places, hashed). The fingerprint is newline- and platform-independent.
The raw byte checksum is recorded too, but it differs between CRLF and LF
systems, so a raw-checksum mismatch on its own is not a failure.

Yahoo revises history. A fingerprint mismatch means the vendor's data for that
window has changed since the manifest was written (2026-08-30) -- not
necessarily that anything here is wrong. ``--verify`` reports which universes
drifted so the effect can be judged rather than guessed at.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

DEFAULT_CACHE_DIR = os.path.join(PROJECT_ROOT, "data", "pinned_universes")
MANIFEST_PATH = os.path.join(PROJECT_ROOT, "data", "pinned_universes_manifest.json")

# Kept in step with scripts/baseline_multiseed.py (PIN_END / PIN_START) and
# scripts/kaggle_axiom_10seed.py (UNIVERSES); verify_constants() asserts it.
PIN_END = "2026-08-20"
PIN_START = {"10y": "2016-08-20", "5y": "2021-08-20"}

UNIVERSES = {
    "US_ETFs": {
        "tickers": ["SPY", "QQQ", "EEM", "VNQ", "HYG", "TLT", "GLD", "USO", "UUP", "IWM"],
        "period": "10y",
    },
    "US_MegaCap_PIT": {
        "tickers": ["AAPL", "XOM", "MSFT", "GOOGL", "GE", "JNJ", "PG", "WFC", "JPM", "CVX"],
        "period": "10y",
    },
    "Global_Indices": {
        "tickers": ["SPY", "EWJ", "EWG", "EWU", "MCHI", "INDA", "EWZ", "EFA", "EEM", "FXI"],
        "period": "10y",
    },
    "India_Nifty_50": {
        "tickers": [
            "RELIANCE.NS", "TCS.NS", "HDFCBANK.NS", "INFY.NS", "ICICIBANK.NS",
            "BHARTIARTL.NS", "ITC.NS", "SBIN.NS", "LT.NS", "AXISBANK.NS",
        ],
        "period": "5y",
    },
    "Forex_Commodities": {
        "tickers": [
            "EURUSD=X", "GBPUSD=X", "USDJPY=X", "AUDUSD=X", "USDCAD=X",
            "GC=F", "CL=F", "SI=F", "HG=F", "NG=F",
        ],
        "period": "5y",
    },
    "Crypto_PIT": {
        "tickers": [
            "BTC-USD", "ETH-USD", "XRP-USD", "BCH-USD", "LTC-USD",
            "EOS-USD", "BNB-USD", "XTZ-USD", "LINK-USD", "TRX-USD",
        ],
        "period": "5y",
    },
}

# The SPY fixed-reference arm is downloaded on its own, over a slightly wider
# window than any universe, because it must cover every universe's calendar.
# Mirrors SPY_REFERENCE_* in scripts/canonical_evaluation.py.
SPY_REFERENCE = {
    "filename": "_spy_reference.csv",
    "ticker": "SPY",
    "start": "2016-07-01",
    "end": "2026-08-21",
}

def verify_constants() -> bool:
    """Assert the pinned constants here still match their source of truth.

    Importing the source modules pulls in torch, which is heavy and not needed to
    download prices, so a failed import is a warning rather than an error.
    """
    try:
        from scripts.baseline_multiseed import PIN_END as SRC_END, PIN_START as SRC_START
        from scripts.kaggle_axiom_10seed import UNIVERSES as SRC_UNIVERSES
    except Exception as exc:  # torch / yfinance absent, or import-time failure
        print("  [warn] could not import source modules to cross-check constants: %s" % exc)
        return False
    assert SRC_END == PIN_END, "PIN_END drifted from scripts/baseline_multiseed.py"
    assert SRC_START == PIN_START, "PIN_START drifted from scripts/baseline_multiseed.py"
    assert set(SRC_UNIVERSES) == set(UNIVERSES), "universe names drifted"
    for name, info in UNIVERSES.items():
        src = SRC_UNIVERSES[name]
        assert list(src["tickers"]) == info["tickers"], "%s tickers drifted" % name
        assert src["period"] == info["period"], "%s period drifted" % name
    print("  constants cross-checked against scripts/baseline_multiseed.py and "
          "scripts/kaggle_axiom_10seed.py: OK")
    return True


def fingerprint(df) -> str:
    """Platform-independent hash of a price frame's contents.

    Dates plus every value at 6 decimal places, in column order. Independent of
    line endings, float repr and index dtype, which the raw byte checksum is not.
    """
    h = hashlib.sha256()
    h.update(("|".join(str(c) for c in df.columns) + "\n").encode("utf-8"))
    for idx, row in zip(df.index, df.to_numpy()):
        h.update(str(getattr(idx, "date", lambda: idx)()).encode("utf-8"))
        for value in row:
            h.update(b"\t")
            h.update(("%.6f" % float(value)).encode("utf-8"))
        h.update(b"\n")
    return h.hexdigest()


def describe(path: str, df) -> dict:
    raw = open(path, "rb").read()
    return {
        "sha256_raw": hashlib.sha256(raw).hexdigest(),
        "sha256_lf": hashlib.sha256(raw.replace(b"\r\n", b"\n")).hexdigest(),
        "fingerprint": fingerprint(df),
        "bytes": len(raw),
        "rows": int(df.shape[0]),
        "cols": int(df.shape[1]),
        "first_date": str(df.index[0].date()),
        "last_date": str(df.index[-1].date()),
        "column_order": [str(c) for c in df.columns],
    }

def download_universe(name: str, info: dict, cache_dir: str, force: bool) -> str:
    """Download one universe's pinned window to ``cache_dir``; return its path."""
    import pandas as pd

    os.makedirs(cache_dir, exist_ok=True)
    path = os.path.join(cache_dir, "%s.csv" % name)
    if os.path.exists(path) and not force:
        print("  %-18s cached" % name)
        return path
    import yfinance as yf

    start = PIN_START[info["period"]]
    df = yf.download(info["tickers"], start=start, end=PIN_END,
                     auto_adjust=True, progress=False)
    if isinstance(df.columns, pd.MultiIndex):
        df = df["Close"]
    # Column order left exactly as returned -- see the module docstring.
    df = df.dropna()
    df.to_csv(path, encoding="utf-8")
    print("  %-18s downloaded  %s -> %s  %d rows x %d cols"
          % (name, start, PIN_END, df.shape[0], df.shape[1]))
    return path


def download_spy_reference(cache_dir: str, force: bool) -> str:
    import pandas as pd

    os.makedirs(cache_dir, exist_ok=True)
    path = os.path.join(cache_dir, SPY_REFERENCE["filename"])
    if os.path.exists(path) and not force:
        print("  %-18s cached" % "SPY reference")
        return path
    import yfinance as yf

    df = yf.download(SPY_REFERENCE["ticker"], start=SPY_REFERENCE["start"],
                     end=SPY_REFERENCE["end"], auto_adjust=True, progress=False)
    if isinstance(df.columns, pd.MultiIndex):
        df = df["Close"]
    elif "Close" in df.columns:
        df = df[["Close"]].rename(columns={"Close": SPY_REFERENCE["ticker"]})
    df = df.dropna()
    df.to_csv(path, encoding="utf-8")
    print("  %-18s downloaded  %s -> %s  %d rows"
          % ("SPY reference", SPY_REFERENCE["start"], SPY_REFERENCE["end"], df.shape[0]))
    return path


def load(path: str):
    import pandas as pd

    return pd.read_csv(path, index_col=0, parse_dates=True)

def all_filenames() -> list:
    return ["%s.csv" % n for n in UNIVERSES] + [SPY_REFERENCE["filename"]]


def write_manifest(cache_dir: str, manifest_path: str) -> int:
    entries = {}
    for filename in all_filenames():
        path = os.path.join(cache_dir, filename)
        if not os.path.exists(path):
            print("  [missing] %s" % filename)
            continue
        entries[filename] = describe(path, load(path))
    payload = {
        "note": ("Fingerprints of the pinned price windows the committed results in "
                 "data/ were computed from. The CSVs themselves are not "
                 "redistributed; regenerate them with "
                 "scripts/fetch_pinned_universes.py and check with --verify."),
        "pin_end": PIN_END,
        "pin_start": PIN_START,
        "spy_reference": {k: v for k, v in SPY_REFERENCE.items() if k != "filename"},
        "auto_adjust": True,
        "dropna": True,
        "column_order": "as returned by yfinance (alphabetical by ticker) -- load-bearing",
        "fingerprint_definition": ("sha256 over the column names, then per row the ISO "
                                   "date and every value formatted '%.6f', tab-separated"),
        "files": entries,
    }
    with open(manifest_path, "w", encoding="utf-8", newline="\n") as fh:
        json.dump(payload, fh, indent=2)
        fh.write("\n")
    print("  wrote %s (%d files)" % (os.path.relpath(manifest_path, PROJECT_ROOT), len(entries)))
    return 0


def verify(cache_dir: str, manifest_path: str) -> int:
    with open(manifest_path, "r", encoding="utf-8") as fh:
        manifest = json.load(fh)
    expected = manifest["files"]
    missing, drifted, exact = [], [], []
    for filename, want in expected.items():
        path = os.path.join(cache_dir, filename)
        if not os.path.exists(path):
            missing.append(filename)
            print("  [MISSING ] %s" % filename)
            continue
        got = describe(path, load(path))
        if got["fingerprint"] == want["fingerprint"]:
            exact.append(filename)
            print("  [EXACT   ] %-24s %s rows, %s -> %s"
                  % (filename, got["rows"], got["first_date"], got["last_date"]))
            continue
        drifted.append(filename)
        print("  [DRIFTED ] %s" % filename)
        for key in ("rows", "cols", "first_date", "last_date"):
            if got[key] != want[key]:
                print("      %-12s expected %-12s got %s" % (key, want[key], got[key]))
        if got["column_order"] != want["column_order"]:
            print("      column_order CHANGED -- results will not reproduce")
            print("        expected %s" % want["column_order"])
            print("        got      %s" % got["column_order"])
        else:
            print("      shape and column order match; values differ "
                  "(Yahoo revised this window)")
    print()
    print("  exact: %d    drifted: %d    missing: %d"
          % (len(exact), len(drifted), len(missing)))
    if drifted:
        print("  Drift is a vendor-side revision, not necessarily an error here. "
              "Re-running the\n  evaluation on drifted inputs will not reproduce "
              "data/ to the last decimal.")
    return 1 if (missing or drifted) else 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--verify", action="store_true",
                    help="compare existing caches against the committed manifest")
    ap.add_argument("--write-manifest", action="store_true",
                    help="regenerate the manifest from existing caches")
    ap.add_argument("--force", action="store_true", help="re-download even if cached")
    ap.add_argument("--cache-dir", default=DEFAULT_CACHE_DIR)
    ap.add_argument("--manifest", default=MANIFEST_PATH)
    args = ap.parse_args(argv)

    if args.write_manifest:
        print("WRITE MANIFEST")
        return write_manifest(args.cache_dir, args.manifest)

    if args.verify:
        print("VERIFY AGAINST MANIFEST")
        return verify(args.cache_dir, args.manifest)

    print("FETCH PINNED UNIVERSES")
    verify_constants()
    print("  window: %s / %s -> %s   auto_adjust=True"
          % (PIN_START["10y"], PIN_START["5y"], PIN_END))
    for name, info in UNIVERSES.items():
        download_universe(name, info, args.cache_dir, args.force)
    download_spy_reference(args.cache_dir, args.force)
    print()
    if os.path.exists(args.manifest):
        print("VERIFY AGAINST MANIFEST")
        return verify(args.cache_dir, args.manifest)
    return 0


if __name__ == "__main__":
    sys.exit(main())
