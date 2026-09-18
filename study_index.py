"""
study_index.py
==============
Builds a single in-memory index from a loaded StudyData object.

All CSV data is read ONCE (by data_loader.StudyData) and then indexed
into nested dictionaries so every query is O(1) or O(k) where k is the
number of records for that subject — no disk reads after startup.

Data model
----------
All records are kept as plain dict[str, str] — exactly as returned by
data_loader.load_csv().  Nothing is converted here; callers use
data_loader.to_float() and data_loader.parse_date() when they need numbers
or dates.

Index layout
------------
_by_subject : dict[USUBJID -> SubjectBundle]

SubjectBundle holds:
    dm          : dict[str, str]           (the single DM row)
    ae          : list[dict[str, str]]     (adverse events, ordered by AESEQ)
    lb          : list[dict[str, str]]     (labs, ordered by LBSEQ)
    vs          : list[dict[str, str]]     (vitals, ordered by VSSEQ)
    ex          : list[dict[str, str]]     (exposure, ordered by EXSEQ)
    cm          : list[dict[str, str]]     (concomitant meds, ordered by CMSEQ)
    mh          : list[dict[str, str]]     (medical history, ordered by MHSEQ)
    ds          : list[dict[str, str]]     (disposition, ordered by DSSEQ)
    eg          : list[dict[str, str]]     (ECG, ordered by EGSEQ)

Secondary indexes
-----------------
_ref_ranges  : list[dict]                 (reference_ranges.csv — all 8 rows)
_cuts        : list[dict]                 (cuts.csv — all 12 rows)
_corrections : list[dict]                 (corrections.csv — all 200 rows)

Reference-range lookup
----------------------
ref_range(lbtestcd, siteid) -> dict with LOW, HIGH, UNIT
Uses the LAB column: siteid == "S07"  -> LAB="S07", else LAB="CENTRAL"

Visit index
-----------
get_visits(uid) -> dict[visit_name -> {
    "lb": [...], "vs": [...], "ex": [...], "eg": [...]
}]
Aggregates all visit-tagged domains per visit name.
"""

from __future__ import annotations

from typing import Optional

# ---------------------------------------------------------------------------
# SubjectBundle — everything we know about one subject
# ---------------------------------------------------------------------------

class SubjectBundle:
    """
    Holds all domain records for a single subject.
    All records are original dict[str, str] rows — values unchanged.
    """

    __slots__ = ("dm", "ae", "lb", "vs", "ex", "cm", "mh", "ds", "eg")

    def __init__(self, dm_row: dict) -> None:
        self.dm: dict             = dm_row         # single DM row
        self.ae: list[dict]       = []
        self.lb: list[dict]       = []
        self.vs: list[dict]       = []
        self.ex: list[dict]       = []
        self.cm: list[dict]       = []
        self.mh: list[dict]       = []
        self.ds: list[dict]       = []
        self.eg: list[dict]       = []

    def record_counts(self) -> dict[str, int]:
        return {
            "DM": 1,
            "AE": len(self.ae),
            "LB": len(self.lb),
            "VS": len(self.vs),
            "EX": len(self.ex),
            "CM": len(self.cm),
            "MH": len(self.mh),
            "DS": len(self.ds),
            "EG": len(self.eg),
        }


# ---------------------------------------------------------------------------
# StudyIndex
# ---------------------------------------------------------------------------

class StudyIndex:
    """
    In-memory index of all study data, keyed by USUBJID.

    Build with:
        index = build_study_index(study_data)

    Then query with:
        index.get_subject("042-S01-001")
        index.get_labs("042-S01-001")
        index.get_adverse_events("042-S01-001")
        ...
    """

    def __init__(self) -> None:
        self._by_subject: dict[str, SubjectBundle] = {}
        self._ref_ranges: list[dict] = []
        self._cuts:       list[dict] = []
        self._corrections: list[dict] = []
        # Secondary: ref-range lookup cache
        # key: (lbtestcd, lab)  -> dict row
        self._rr_cache: dict[tuple[str, str], dict] = {}

    # ------------------------------------------------------------------
    # Subject-level accessors
    # ------------------------------------------------------------------

    def _bundle(self, uid: str) -> Optional[SubjectBundle]:
        return self._by_subject.get(uid)

    def get_subject(self, uid: str) -> Optional[dict]:
        """Return the DM row for a subject, or None if not found."""
        b = self._bundle(uid)
        return b.dm if b else None

    def get_adverse_events(self, uid: str) -> list[dict]:
        """All AE records for the subject, ordered by AESEQ."""
        b = self._bundle(uid)
        return b.ae if b else []

    def get_labs(self, uid: str) -> list[dict]:
        """All LB records for the subject, ordered by LBSEQ."""
        b = self._bundle(uid)
        return b.lb if b else []

    def get_vitals(self, uid: str) -> list[dict]:
        """All VS records for the subject, ordered by VSSEQ."""
        b = self._bundle(uid)
        return b.vs if b else []

    def get_exposure(self, uid: str) -> list[dict]:
        """All EX records for the subject, ordered by EXSEQ."""
        b = self._bundle(uid)
        return b.ex if b else []

    def get_medications(self, uid: str) -> list[dict]:
        """All CM records for the subject, ordered by CMSEQ."""
        b = self._bundle(uid)
        return b.cm if b else []

    def get_medical_history(self, uid: str) -> list[dict]:
        """All MH records for the subject, ordered by MHSEQ."""
        b = self._bundle(uid)
        return b.mh if b else []

    def get_disposition(self, uid: str) -> list[dict]:
        """All DS records for the subject, ordered by DSSEQ."""
        b = self._bundle(uid)
        return b.ds if b else []

    def get_ecg(self, uid: str) -> list[dict]:
        """All EG records for the subject, ordered by EGSEQ."""
        b = self._bundle(uid)
        return b.eg if b else []

    def get_visits(self, uid: str) -> dict[str, dict[str, list[dict]]]:
        """
        Return a visit-keyed dict aggregating all visit-tagged domains.

        Example:
            {
              "SCREENING": {"lb": [...], "vs": [...], "ex": [], "eg": []},
              "BASELINE":  {"lb": [...], "vs": [...], "ex": [...], "eg": []},
              ...
            }
        """
        b = self._bundle(uid)
        if not b:
            return {}

        visits: dict[str, dict[str, list[dict]]] = {}

        def _add(domain_key: str, records: list[dict], visit_col: str) -> None:
            for rec in records:
                vname = rec.get(visit_col, "")
                if not vname:
                    continue
                if vname not in visits:
                    visits[vname] = {"lb": [], "vs": [], "ex": [], "eg": []}
                visits[vname][domain_key].append(rec)

        _add("lb", b.lb, "VISIT")
        _add("vs", b.vs, "VISIT")
        _add("ex", b.ex, "VISIT")
        _add("eg", b.eg, "VISIT")

        return visits

    # ------------------------------------------------------------------
    # Reference-range lookup
    # ------------------------------------------------------------------

    def ref_range(
        self, lbtestcd: str, siteid: str
    ) -> Optional[dict]:
        """
        Return the reference-range row for a given test and site.

        Site S07 has its own ranges (ukat/L); every other site uses CENTRAL (U/L).
        Returns a dict with keys: LBTESTCD, UNIT, LOW, HIGH, LAB
        Returns None if no range is found.
        """
        lab = "S07" if siteid.upper() == "S07" else "CENTRAL"
        return self._rr_cache.get((lbtestcd.upper(), lab))

    # ------------------------------------------------------------------
    # Study-level accessors
    # ------------------------------------------------------------------

    def all_subject_ids(self) -> list[str]:
        """Sorted list of all USUBJIDs."""
        return sorted(self._by_subject.keys())

    def subject_count(self) -> int:
        return len(self._by_subject)

    def protocol_version_at(self, cut: int) -> int:
        """Return the protocol version in force at a given cut number."""
        best_cut = 0
        best_pv  = 1
        for row in self._cuts:
            try:
                c = int(row.get("cut", "0"))
            except ValueError:
                continue
            if c <= cut and c >= best_cut:
                best_cut = c
                best_pv  = int(row.get("protocol_version", "1"))
        return best_pv

    def cuts(self) -> list[dict]:
        return self._cuts

    def corrections(self) -> list[dict]:
        return self._corrections

    def reference_ranges(self) -> list[dict]:
        return self._ref_ranges


# ---------------------------------------------------------------------------
# Sorting helpers
# ---------------------------------------------------------------------------

def _seq_key(row: dict, seq_col: str) -> int:
    """Return int value of a sequence column, defaulting to 0."""
    try:
        return int(row.get(seq_col, "0"))
    except ValueError:
        return 0


# ---------------------------------------------------------------------------
# build_study_index  — the one public function to call
# ---------------------------------------------------------------------------

def build_study_index(data) -> StudyIndex:
    """
    Build the StudyIndex from a StudyData object (from data_loader).

    Parameters
    ----------
    data : data_loader.StudyData
        Loaded, cut-filtered, and corrections-applied dataset.

    Returns
    -------
    StudyIndex
        All data indexed by USUBJID.  Build once, query many times.
    """
    idx = StudyIndex()

    # ── 1. Seed from DM (one bundle per subject) ──────────────────────
    for row in data.dm:
        uid = row.get("USUBJID", "")
        if uid:
            idx._by_subject[uid] = SubjectBundle(dm_row=row)

    # ── 2. Index each domain ───────────────────────────────────────────
    _index_domain(idx, data.ae, "USUBJID", "ae", "AESEQ")
    _index_domain(idx, data.lb, "USUBJID", "lb", "LBSEQ")
    _index_domain(idx, data.vs, "USUBJID", "vs", "VSSEQ")
    _index_domain(idx, data.ex, "USUBJID", "ex", "EXSEQ")
    _index_domain(idx, data.cm, "USUBJID", "cm", "CMSEQ")
    _index_domain(idx, data.mh, "USUBJID", "mh", "MHSEQ")
    _index_domain(idx, data.ds, "USUBJID", "ds", "DSSEQ")
    _index_domain(idx, data.eg, "USUBJID", "eg", "EGSEQ")

    # ── 3. Copy meta tables ────────────────────────────────────────────
    idx._ref_ranges  = data.reference_ranges
    idx._cuts        = data.cuts
    idx._corrections = data.corrections

    # ── 4. Build reference-range cache ────────────────────────────────
    for rr in data.reference_ranges:
        key = (rr.get("LBTESTCD", "").upper(), rr.get("LAB", "").upper())
        idx._rr_cache[key] = rr

    return idx


def _index_domain(
    idx:     StudyIndex,
    records: list[dict],
    uid_col: str,
    attr:    str,
    seq_col: str,
) -> None:
    """
    Append each record to the correct SubjectBundle list, then sort by seq.

    Parameters
    ----------
    idx     : StudyIndex being built
    records : flat list of dict rows from a domain CSV
    uid_col : column name for the subject key (always "USUBJID")
    attr    : attribute name on SubjectBundle ("ae", "lb", etc.)
    seq_col : sequence column name used for ordering
    """
    # Distribute
    for row in records:
        uid = row.get(uid_col, "")
        if not uid:
            continue
        bundle = idx._by_subject.get(uid)
        if bundle is None:
            # Record for a subject not in DM — skip (orphan)
            continue
        getattr(bundle, attr).append(row)

    # Sort each subject's list by sequence number
    for bundle in idx._by_subject.values():
        lst = getattr(bundle, attr)
        lst.sort(key=lambda r: _seq_key(r, seq_col))
