"""
rules.py
========
Protocol-version-aware rule constants and helper functions.

All clinical decision logic lives here (or in checks.py which calls these).
rules.py is pure data/logic — no file I/O, no DataFrames.

Protocol version timeline (from cuts.csv):
  v1  →  cuts 1–4  : visit window ±7 days, prohibited = {SYSTEMIC_GLUCOCORTICOID}
  v2  →  cuts 5–8  : visit window ±3 days, added creatinine exclusion
  v3  →  cuts 9–12 : adds SULFONYLUREA to prohibited medications
"""

from __future__ import annotations

from typing import FrozenSet

# ---------------------------------------------------------------------------
# Visit schedule (nominal study days relative to Day 0 = BASELINE)
# ---------------------------------------------------------------------------

VISIT_DAYS: dict[str, int] = {
    "SCREENING":  -14,
    "BASELINE":     0,
    "WEEK2":       14,
    "WEEK4":       28,
    "WEEK8":       56,
    "WEEK12":      84,
    "WEEK16":     112,
    "WEEK20":     140,
    "WEEK24":     168,
    "EOS":        182,   # End of Study
}

# ---------------------------------------------------------------------------
# Visit windows per protocol version (±days)
# ---------------------------------------------------------------------------

VISIT_WINDOW: dict[int, int] = {
    1: 7,   # ±7 days
    2: 3,   # ±3 days
    3: 3,   # same as v2
}

# ---------------------------------------------------------------------------
# Prohibited concomitant medication classes (CMCLAS values) per protocol version
# ---------------------------------------------------------------------------

PROHIBITED_CLASSES: dict[int, FrozenSet[str]] = {
    1: frozenset({"SYSTEMIC_GLUCOCORTICOID"}),
    2: frozenset({"SYSTEMIC_GLUCOCORTICOID"}),
    3: frozenset({"SYSTEMIC_GLUCOCORTICOID", "SULFONYLUREA"}),
}

# ---------------------------------------------------------------------------
# Exclusion criteria — lab thresholds at screening
# ---------------------------------------------------------------------------

# Creatinine exclusion added in protocol v2 (amendment 2)
CREATININE_EXCLUSION_MG_DL: float = 1.5   # > 1.5 mg/dL at screening → excluded

# Hepatic disease exclusion (all versions): ALT or AST > 2×ULN at screening
HEPATIC_ALT_AST_ULN_MULTIPLIER: float = 2.0

# HbA1c inclusion window at screening
HBA1C_LOWER: float = 7.0
HBA1C_UPPER: float = 10.5

# Age inclusion window
AGE_LOWER: int = 18
AGE_UPPER: int = 75

# ---------------------------------------------------------------------------
# Hy's law thresholds (all protocol versions)
# ---------------------------------------------------------------------------

HYS_ALT_AST_ULN_MULTIPLIER: float = 3.0   # ALT or AST > 3×ULN
HYS_BILI_ULN_MULTIPLIER: float = 2.0       # AND total bilirubin > 2×ULN
HYS_WINDOW_DAYS: int = 14                  # within 14 days of each other

# ---------------------------------------------------------------------------
# Dosing rules
# ---------------------------------------------------------------------------

EXPECTED_DOSE_DRUG: float = 10.0    # mg — DRUG arm
EXPECTED_DOSE_PLACEBO: float = 0.0  # mg — PLACEBO arm
DOSE_UNIT: str = "mg"

# ---------------------------------------------------------------------------
# Safety reporting
# ---------------------------------------------------------------------------

# An event with AESHOSP == "Y" is serious regardless of AESER coding
HOSPITALIZATION_MAKES_SERIOUS: bool = True

# ---------------------------------------------------------------------------
# Lab unit conversion
# ---------------------------------------------------------------------------

# Site S07 reports ALT/AST in µkat/L; multiply by this to get U/L
UKAT_TO_UL: float = 60.0
S07_LOCAL_LAB_SITE: str = "S07"
S07_LOCAL_TESTS: FrozenSet[str] = frozenset({"ALT", "AST"})

# ---------------------------------------------------------------------------
# Helper: get protocol version for a given cut number
# ---------------------------------------------------------------------------

_CUT_TO_PROTOCOL: dict[int, int] = {}   # populated lazily from cuts.csv by StudyData


def get_protocol_version(cut: int, cuts_rows: list = None) -> int:
    """
    Return the protocol version in force at the given cut.

    Parameters
    ----------
    cut       : data cut number (1–12)
    cuts_rows : optional list of dicts from cuts.csv (uses hardcoded fallback if None)
    """
    if cuts_rows:
        best_pv = 1
        best_cut = 0
        for row in cuts_rows:
            try:
                c = int(row.get("cut", "0"))
            except (ValueError, TypeError):
                continue
            if c <= cut and c >= best_cut:
                best_cut = c
                best_pv  = int(row.get("protocol_version", "1"))
        return best_pv
    # Hardcoded fallback (matches the supplied dataset)
    if cut <= 4:
        return 1
    if cut <= 8:
        return 2
    return 3


def prohibited_classes(protocol_version: int) -> FrozenSet[str]:
    """Return the set of prohibited CMCLAS values for this protocol version."""
    return PROHIBITED_CLASSES.get(protocol_version, PROHIBITED_CLASSES[1])


def visit_window(protocol_version: int) -> int:
    """Return the allowed visit window (±days) for this protocol version."""
    return VISIT_WINDOW.get(protocol_version, 7)
