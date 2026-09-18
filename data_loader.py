"""
data_loader.py
==============
Loads all STUDY-042 domain CSVs from the data directory.

Implementation constraints
--------------------------
* Uses ONLY Python standard-library modules: csv, pathlib, datetime
* No pandas, no numpy, no external dependencies
* All values are kept as strings — callers convert to float/int/date as needed
* Missing fields are stored as "" (empty string), never None
* The data directory is located automatically from __file__

Data model
----------
Each domain is a list of dicts:
    [
        {"USUBJID": "042-S01-001", "AESEQ": "1", "AETERM": "Headache", ...},
        ...
    ]

Key types
---------
    DomainTable  = list[dict[str, str]]
    StudyDB      = dict[str, DomainTable]   # keyed by domain name: "DM", "AE", …

Cut filtering
-------------
Keep only rows where int(cut_available) <= requested_cut.
Rows with a blank cut_available are kept (defensive default).

Corrections
-----------
corrections.csv lists values that were superseded at a later cut.
For each correction issued at cut <= requested_cut, find the matching row
in the domain table by (usubjid, seq) and overwrite the named field in-place.
"""

from __future__ import annotations

import csv
from datetime import date, datetime
from pathlib import Path
from typing import Optional


# ---------------------------------------------------------------------------
# Data directory — auto-located relative to this file
# ---------------------------------------------------------------------------

def _find_data_dir() -> Path:
    """
    Walk up from this file and find the 'data' folder that contains DM.csv.
    Handles the double-nested hackathon-data/hackathon-data layout.
    """
    base = Path(__file__).resolve().parent
    # Try known relative path first (most common layout)
    candidate = base / "hackathon-data" / "hackathon-data" / "data"
    if (candidate / "DM.csv").exists():
        return candidate
    # Fallback: walk subdirectories up to 4 levels deep
    for depth in range(1, 5):
        for sub in base.rglob("DM.csv"):
            return sub.parent
    raise FileNotFoundError(
        f"Cannot find data directory containing DM.csv under {base}"
    )


DATA_DIR: Path = _find_data_dir()

# ---------------------------------------------------------------------------
# Domain file map
# ---------------------------------------------------------------------------

DOMAIN_FILES: dict[str, str] = {
    "DM": "DM.csv",
    "AE": "AE.csv",
    "LB": "LB.csv",
    "VS": "VS.csv",
    "EX": "EX.csv",
    "CM": "CM.csv",
    "DS": "DS.csv",
    "MH": "MH.csv",
    "EG": "EG.csv",
}

META_FILES: dict[str, str] = {
    "CUTS":       "cuts.csv",
    "CORRECTIONS":"corrections.csv",
    "REF_RANGES": "reference_ranges.csv",
}

# Type alias
DomainTable = list[dict[str, str]]


# ---------------------------------------------------------------------------
# Low-level CSV reader
# ---------------------------------------------------------------------------

def load_csv(filename: str) -> DomainTable:
    """
    Read a single CSV file from DATA_DIR and return it as a list of dicts.

    Rules
    -----
    * Column names are preserved exactly as written in the header row.
    * Every value is a string. Missing/empty cells become "".
    * The function never raises on empty fields.
    * The file is read as UTF-8 with BOM stripping.

    Parameters
    ----------
    filename : str
        Bare filename, e.g. "DM.csv".  The DATA_DIR prefix is added automatically.

    Returns
    -------
    list[dict[str, str]]
        One dict per data row, keyed by column name.
    """
    path = DATA_DIR / filename
    rows: DomainTable = []
    with open(path, newline="", encoding="utf-8-sig") as fh:
        reader = csv.DictReader(fh)
        for raw in reader:
            # DictReader gives OrderedDict; convert to plain dict, strip whitespace,
            # and replace None (csv module uses None for missing trailing columns) with ""
            row: dict[str, str] = {
                k.strip(): ("" if v is None else v.strip())
                for k, v in raw.items()
            }
            rows.append(row)
    return rows


# ---------------------------------------------------------------------------
# Cut-aware filtering
# ---------------------------------------------------------------------------

def _to_int(value: str, default: int = 0) -> int:
    """Safely convert a string to int; return default on failure."""
    try:
        return int(value)
    except (ValueError, TypeError):
        return default


def filter_by_cut(rows: DomainTable, cut: int) -> DomainTable:
    """
    Keep only rows where cut_available <= cut.
    Rows with a blank or non-numeric cut_available are retained (safe default).
    """
    result: DomainTable = []
    for row in rows:
        raw_cut = row.get("cut_available", "")
        if raw_cut == "":
            result.append(row)          # no cut tag → always visible
        elif _to_int(raw_cut, -1) <= cut:
            result.append(row)
    return result


# ---------------------------------------------------------------------------
# Corrections
# ---------------------------------------------------------------------------

def apply_corrections(
    tables: dict[str, DomainTable],
    corrections: DomainTable,
    cut: int,
) -> None:
    """
    Apply corrections issued at or before `cut` to the domain tables in-place.

    corrections.csv schema:
        cut, domain, usubjid, seq, field, old_value, new_value, reason

    Logic
    -----
    For each correction where int(cut) <= requested_cut:
      find the row in tables[domain] where
          row[USUBJID]   == correction[usubjid]
          row[{DOMAIN}SEQ] == correction[seq]
      and overwrite row[field] with correction[new_value].
    """
    for corr in corrections:
        corr_cut = _to_int(corr.get("cut", ""), 0)
        if corr_cut > cut:
            continue

        domain = corr.get("domain", "").upper()
        if domain not in tables:
            continue

        usubjid   = corr.get("usubjid", "")
        seq_str   = corr.get("seq", "")
        field     = corr.get("field", "")
        new_value = corr.get("new_value", "")
        seq_col   = f"{domain}SEQ"

        table = tables[domain]
        for row in table:
            if row.get("USUBJID") == usubjid and row.get(seq_col) == seq_str:
                if field in row:
                    row[field] = new_value
                break   # (USUBJID, SEQ) is unique per domain


# ---------------------------------------------------------------------------
# Date parsing (used by callers, not by the loader itself)
# ---------------------------------------------------------------------------

_DATE_FMTS = (
    "%Y-%m-%d",   # ISO  : 2026-03-21
    "%d-%b-%Y",   # Long : 03-FEB-2026
    "%d-%b-%y",   # Short: 03-FEB-26
)


def parse_date(value: str) -> Optional[date]:
    """
    Parse a date string in any format seen in the study data.
    Returns a date object, or None if blank / unparseable.
    Values like '<5', 'ND', or '' are returned as None.
    """
    s = (value or "").strip()
    if not s or s.startswith("<") or s.upper() == "ND":
        return None
    for fmt in _DATE_FMTS:
        try:
            return datetime.strptime(s, fmt).date()
        except ValueError:
            continue
    return None


def to_float(value: str) -> Optional[float]:
    """
    Convert a lab/vital value string to float.
    Returns None for blank, '<N', 'ND', or non-numeric strings.

    Handles:
    - European comma decimals: "0,32" -> 0.32  (seen in LB.csv LBORRES)
    - Standard dot decimals:   "40.4"
    - Below-detection:         "<5"  -> None
    - Not done:                "ND"  -> None
    - Empty:                   ""    -> None
    """
    s = (value or "").strip()
    if not s or s.startswith("<") or s.upper() == "ND":
        return None
    # Replace comma used as decimal separator (European locale)
    s = s.replace(",", ".")
    try:
        return float(s)
    except ValueError:
        return None


# ---------------------------------------------------------------------------
# Top-level: load everything
# ---------------------------------------------------------------------------

def load_all_data(cut: int = 12) -> "StudyData":
    """
    Load all domain CSVs, filter to `cut`, apply corrections, and return
    a StudyData container.

    Parameters
    ----------
    cut : int
        The data cut to evaluate (1–12).

    Returns
    -------
    StudyData
        Holds all domain tables and meta tables.
    """
    return StudyData(cut=cut)


# ---------------------------------------------------------------------------
# StudyData container
# ---------------------------------------------------------------------------

class StudyData:
    """
    Container for all domain tables, cut-filtered and corrections-applied.

    All tables are list[dict[str, str]].

    Usage
    -----
    >>> sd = StudyData(cut=12)
    >>> sd.dm[0]
    {'USUBJID': '042-S01-001', 'SITEID': 'S01', ...}
    >>> len(sd.lb)
    14160
    """

    def __init__(self, cut: int = 12):
        self.cut = cut
        self._tables: dict[str, DomainTable] = {}
        self._meta:   dict[str, DomainTable] = {}
        self._load()

    # ------------------------------------------------------------------
    # Private loading
    # ------------------------------------------------------------------

    def _load(self) -> None:
        # 1. Load meta files (no cut_available; keep all rows)
        for key, fname in META_FILES.items():
            self._meta[key] = load_csv(fname)

        # 2. Load domain files and apply cut filter
        for domain, fname in DOMAIN_FILES.items():
            raw = load_csv(fname)
            self._tables[domain] = filter_by_cut(raw, self.cut)

        # 3. Apply corrections that exist up to this cut
        apply_corrections(
            tables=self._tables,
            corrections=self._meta["CORRECTIONS"],
            cut=self.cut,
        )

    # ------------------------------------------------------------------
    # Domain accessors (read-only by convention)
    # ------------------------------------------------------------------

    @property
    def dm(self) -> DomainTable:
        return self._tables["DM"]

    @property
    def ae(self) -> DomainTable:
        return self._tables["AE"]

    @property
    def lb(self) -> DomainTable:
        return self._tables["LB"]

    @property
    def vs(self) -> DomainTable:
        return self._tables["VS"]

    @property
    def ex(self) -> DomainTable:
        return self._tables["EX"]

    @property
    def cm(self) -> DomainTable:
        return self._tables["CM"]

    @property
    def ds(self) -> DomainTable:
        return self._tables["DS"]

    @property
    def mh(self) -> DomainTable:
        return self._tables["MH"]

    @property
    def eg(self) -> DomainTable:
        return self._tables["EG"]

    # ------------------------------------------------------------------
    # Meta accessors
    # ------------------------------------------------------------------

    @property
    def cuts(self) -> DomainTable:
        return self._meta["CUTS"]

    @property
    def corrections(self) -> DomainTable:
        return self._meta["CORRECTIONS"]

    @property
    def reference_ranges(self) -> DomainTable:
        return self._meta["REF_RANGES"]

    # ------------------------------------------------------------------
    # Utility helpers
    # ------------------------------------------------------------------

    def protocol_version(self) -> int:
        """Return the protocol version in force at self.cut."""
        # Look for exact cut match first, then most recent prior cut
        best_cut = 0
        best_pv  = 1
        for row in self.cuts:
            c = _to_int(row.get("cut", ""), -1)
            if c < 0:
                continue
            if c <= self.cut and c >= best_cut:
                best_cut = c
                best_pv  = _to_int(row.get("protocol_version", "1"), 1)
        return best_pv

    def subjects(self) -> list[str]:
        """Sorted list of all USUBJID values in DM."""
        seen: set[str] = set()
        for row in self.dm:
            uid = row.get("USUBJID", "")
            if uid:
                seen.add(uid)
        return sorted(seen)

    def get_domain(self, domain: str) -> DomainTable:
        """Generic domain lookup by name (e.g. 'LB')."""
        return self._tables.get(domain.upper(), [])

    def row_counts(self) -> dict[str, int]:
        """Return a dict of domain → number of rows (post cut-filter)."""
        counts = {k: len(v) for k, v in self._tables.items()}
        counts["CUTS"]        = len(self._meta["CUTS"])
        counts["CORRECTIONS"] = len(self._meta["CORRECTIONS"])
        counts["REF_RANGES"]  = len(self._meta["REF_RANGES"])
        return counts
