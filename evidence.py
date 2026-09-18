"""
evidence.py
===========
Evidence system for the ATLAS solver.

Design
------
RecordRef
    Identifies a single source row in the study dataset.
    Built from the actual CSV identifying fields — no invented keys.
    The canonical citation string is:  DOMAIN|USUBJID|SEQ
    e.g.  LB|042-S07-011|1

    Additional context fields (date, test code, value) are stored
    for human readability and to support "does the record actually
    support the claim?" verification, but they are NOT part of the
    unique identity — identity is (domain, usubjid, seq).

make_record_ref(domain, record)
    Factory: creates a RecordRef from a raw dict row returned by
    data_loader.load_csv().  Reads real field names from the row.

record_ref_exists(ref, index)
    Verifier: confirms the (domain, usubjid, seq) triple actually
    exists in the loaded StudyIndex.  Raises ValueError if not found.

Evidence
    A claim string paired with one or more verified RecordRefs.
    Cannot be constructed with unverified refs — call
    Evidence.build(claim, refs, index) which runs existence checks.

Finding / CheckResult
    Higher-level wrappers used by checks.py to accumulate results
    per check category.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from study_index import StudyIndex


# ---------------------------------------------------------------------------
# Sequence-column map — tells make_record_ref which column holds the
# sequence number for each domain.  Based on actual CSV schemas.
# ---------------------------------------------------------------------------

_SEQ_COL: dict[str, str] = {
    "DM": "DMSEQ",    # DM has no explicit seq col; we use "1" as default
    "AE": "AESEQ",
    "LB": "LBSEQ",
    "VS": "VSSEQ",
    "EX": "EXSEQ",
    "CM": "CMSEQ",
    "MH": "MHSEQ",
    "DS": "DSSEQ",
    "EG": "EGSEQ",
}

# Date column per domain (for human-readable context in the ref)
_DATE_COL: dict[str, str] = {
    "AE": "AESTDTC",
    "LB": "LBDTC",
    "VS": "VSDTC",
    "EX": "EXSTDTC",
    "CM": "CMSTDTC",
    "DS": "DSSTDTC",
    "EG": "EGDTC",
}

# Domain accessor name on StudyIndex (and SubjectBundle)
_DOMAIN_ATTR: dict[str, str] = {
    "AE": "ae",
    "LB": "lb",
    "VS": "vs",
    "EX": "ex",
    "CM": "cm",
    "MH": "mh",
    "DS": "ds",
    "EG": "eg",
    "DM": "dm",
}


# ---------------------------------------------------------------------------
# RecordRef
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class RecordRef:
    """
    An immutable reference to a single row in a domain CSV.

    Identity fields (used for existence checking)
    -----------------------------------------------
    domain  : str   e.g. "LB"
    usubjid : str   e.g. "042-S07-011"
    seq     : int   e.g. 1            (value of {DOMAIN}SEQ column)

    Context fields (human-readable; not part of identity)
    -----------------------------------------------
    visit   : str   e.g. "SCREENING"  (VISIT column, if present)
    date    : str   e.g. "2026-01-15" (date column, if present)
    testcd  : str   e.g. "ALT"        (LBTESTCD / VSTESTCD / etc.)
    value   : str   e.g. "0.61"       (result as reported)
    unit    : str   e.g. "ukat/L"
    extra   : dict  any other fields the caller wants to preserve
    """
    domain:  str
    usubjid: str
    seq:     int

    # Optional human-readable context — preserved from the original row
    visit:   str = ""
    date:    str = ""
    testcd:  str = ""
    value:   str = ""
    unit:    str = ""
    extra:   dict = field(default_factory=dict, hash=False, compare=False)

    # ------------------------------------------------------------------
    # Citation
    # ------------------------------------------------------------------

    @property
    def cite(self) -> str:
        """
        Canonical citation string used by the grader:  DOMAIN|USUBJID|SEQ
        e.g.  LB|042-S07-011|1
        """
        return f"{self.domain}|{self.usubjid}|{self.seq}"

    def __str__(self) -> str:
        return self.cite

    def __repr__(self) -> str:
        parts = [f"cite={self.cite!r}"]
        if self.testcd: parts.append(f"test={self.testcd!r}")
        if self.value:  parts.append(f"value={self.value!r}")
        if self.unit:   parts.append(f"unit={self.unit!r}")
        if self.visit:  parts.append(f"visit={self.visit!r}")
        if self.date:   parts.append(f"date={self.date!r}")
        return f"RecordRef({', '.join(parts)})"

    def to_dict(self) -> dict:
        """Serialisable representation."""
        d: dict[str, Any] = {"cite": self.cite, "domain": self.domain,
                              "usubjid": self.usubjid, "seq": self.seq}
        if self.visit:  d["visit"]  = self.visit
        if self.date:   d["date"]   = self.date
        if self.testcd: d["testcd"] = self.testcd
        if self.value:  d["value"]  = self.value
        if self.unit:   d["unit"]   = self.unit
        return d


# ---------------------------------------------------------------------------
# make_record_ref — factory from a raw dict row
# ---------------------------------------------------------------------------

def make_record_ref(domain: str, record: dict[str, str]) -> RecordRef:
    """
    Create a RecordRef from a raw CSV row dict.

    Parameters
    ----------
    domain : str
        Domain name in uppercase, e.g. "LB", "AE", "EX".
    record : dict[str, str]
        A row returned by data_loader.load_csv() or from StudyIndex.
        All values are strings.

    Returns
    -------
    RecordRef
        Identity: (domain, USUBJID, {DOMAIN}SEQ).
        Context:  visit, date, test code, value, unit extracted from
                  the actual column names of that domain.

    Raises
    ------
    ValueError
        If the row is missing USUBJID or the sequence column.
    """
    domain = domain.upper()

    usubjid = record.get("USUBJID", "").strip()
    if not usubjid:
        raise ValueError(
            f"make_record_ref: record in domain {domain!r} has no USUBJID: {record}"
        )

    # Sequence
    seq_col = _SEQ_COL.get(domain, f"{domain}SEQ")
    raw_seq = record.get(seq_col, "").strip()
    if not raw_seq:
        # DM has no explicit seq column — use 1
        if domain == "DM":
            raw_seq = "1"
        else:
            raise ValueError(
                f"make_record_ref: domain {domain!r} row missing seq col "
                f"{seq_col!r}: {record}"
            )
    try:
        seq = int(raw_seq)
    except ValueError:
        raise ValueError(
            f"make_record_ref: seq value {raw_seq!r} in {domain!r} is not an integer"
        )

    # Optional context — domain-specific column names
    visit  = record.get("VISIT", "")
    date   = record.get(_DATE_COL.get(domain, ""), "")

    # Test code (LB, VS, EG each have a different column name)
    testcd = (
        record.get("LBTESTCD") or
        record.get("VSTESTCD") or
        record.get("EGTESTCD") or
        record.get("EGTESTCD") or
        ""
    )

    # Result value and unit
    value  = (
        record.get("LBORRES") or
        record.get("VSORRES") or
        record.get("EGORRES") or
        record.get("EXDOSE")  or   # EX: dose as the "value"
        ""
    )
    unit   = (
        record.get("LBORRESU") or
        record.get("VSORRESU") or
        record.get("EGORRESU") or
        record.get("EXDOSU")   or
        ""
    )

    # Preserve the full row as extra context (for later checks)
    extra = dict(record)

    return RecordRef(
        domain=domain,
        usubjid=usubjid,
        seq=seq,
        visit=visit,
        date=date,
        testcd=testcd,
        value=value,
        unit=unit,
        extra=extra,
    )


# ---------------------------------------------------------------------------
# record_ref_exists — verifier
# ---------------------------------------------------------------------------

def record_ref_exists(ref: RecordRef | str | dict, index: "StudyIndex") -> bool:
    """
    Return True if the (domain, usubjid, seq) triple actually exists in
    the loaded StudyIndex. Accepts RecordRef instance, cite string (e.g. "LB|042-S07-011|1"),
    or dict.
    """
    if isinstance(ref, str):
        parts = ref.strip().split("|")
        if len(parts) != 3:
            return False
        try:
            seq_val = int(parts[2])
        except ValueError:
            return False
        ref = RecordRef(domain=parts[0], usubjid=parts[1], seq=seq_val)
    elif isinstance(ref, dict):
        try:
            seq_val = int(ref.get("seq", 0))
        except (ValueError, TypeError):
            return False
        ref = RecordRef(
            domain=str(ref.get("domain", "")),
            usubjid=str(ref.get("usubjid", "")),
            seq=seq_val,
        )

    domain  = ref.domain.upper()
    uid     = ref.usubjid
    seq_col = _SEQ_COL.get(domain, f"{domain}SEQ")
    attr    = _DOMAIN_ATTR.get(domain)

    if attr is None:
        return False  # unknown domain

    bundle = index._by_subject.get(uid)
    if bundle is None:
        return False  # subject not in index

    # DM is a single dict, not a list
    if domain == "DM":
        return True   # subject exists → DM row exists (seq always 1)

    records: list[dict] = getattr(bundle, attr, [])
    target_seq = str(ref.seq)
    for row in records:
        if row.get(seq_col, "").strip() == target_seq:
            return True
    return False


def assert_record_ref_exists(ref: RecordRef, index: "StudyIndex") -> None:
    """
    Raise ValueError if the record referenced by `ref` does not exist
    in the loaded index.  Used by Evidence.build() to enforce integrity.
    """
    if not record_ref_exists(ref, index):
        raise ValueError(
            f"Evidence integrity check FAILED: record {ref.cite!r} does not "
            f"exist in the loaded study index.  "
            f"domain={ref.domain!r}  usubjid={ref.usubjid!r}  seq={ref.seq}"
        )


# ---------------------------------------------------------------------------
# Evidence — a verified claim + record refs
# ---------------------------------------------------------------------------

@dataclass
class Evidence:
    """
    A clinical claim supported by one or more verified source records.

    Use Evidence.build() rather than the constructor directly — it
    runs existence checks on all refs before returning.

    Attributes
    ----------
    claim       : human-readable statement, e.g. "ALT was 0.61 ukat/L at SCREENING"
    record_refs : list of RecordRef objects, each verified to exist
    """
    claim:       str
    record_refs: list[RecordRef] = field(default_factory=list)

    # ------------------------------------------------------------------
    # Factory — the only safe way to create an Evidence object
    # ------------------------------------------------------------------

    @classmethod
    def build(
        cls,
        claim: str,
        refs:  list[RecordRef],
        index: "StudyIndex",
    ) -> "Evidence":
        """
        Create an Evidence object after verifying all refs exist.

        Parameters
        ----------
        claim : str
            Human-readable claim this evidence supports.
        refs  : list[RecordRef]
            Source records that support the claim.
        index : StudyIndex
            The loaded in-memory index used for existence checking.

        Returns
        -------
        Evidence

        Raises
        ------
        ValueError
            If any ref does not exist in the index, or refs is empty.
        """
        if not refs:
            raise ValueError("Evidence.build: refs list must not be empty")
        for ref in refs:
            assert_record_ref_exists(ref, index)
        return cls(claim=claim, record_refs=list(refs))

    # ------------------------------------------------------------------
    # Output helpers
    # ------------------------------------------------------------------

    def citations(self) -> list[str]:
        """Return the canonical cite string for every record ref."""
        return [r.cite for r in self.record_refs]

    def to_dict(self) -> dict:
        return {
            "claim":       self.claim,
            "record_refs": [r.to_dict() for r in self.record_refs],
            "citations":   self.citations(),
        }

    def __str__(self) -> str:
        cites = ", ".join(self.citations())
        return f'Evidence("{self.claim}" | refs=[{cites}])'


# ---------------------------------------------------------------------------
# Finding — one clinical issue with evidence
# ---------------------------------------------------------------------------

@dataclass
class Finding:
    """
    A single clinical finding raised by a check function.

    Attributes
    ----------
    category    : machine label, e.g. "HYS_LAW_CANDIDATE", "SAE_MISCODED"
    usubjid     : subject involved
    description : human-readable explanation
    evidence    : list[Evidence] objects (each already verified)
    severity    : "INFO" | "WARNING" | "CRITICAL"
    """
    category:    str
    usubjid:     str
    description: str
    evidence:    list[Evidence] = field(default_factory=list)
    severity:    str = "WARNING"

    def all_citations(self) -> list[str]:
        """Flat list of all citation strings across all evidence objects."""
        out = []
        for ev in self.evidence:
            out.extend(ev.citations())
        return out

    def to_dict(self) -> dict:
        return {
            "category":    self.category,
            "usubjid":     self.usubjid,
            "severity":    self.severity,
            "description": self.description,
            "evidence":    [ev.to_dict() for ev in self.evidence],
            "citations":   self.all_citations(),
        }


# ---------------------------------------------------------------------------
# CheckResult — output of one check function
# ---------------------------------------------------------------------------

@dataclass
class CheckResult:
    """
    Output of a single check function.
    If findings is empty, the answer to 'anything to report?' is 'none'.
    """
    check_name: str
    findings:   list[Finding] = field(default_factory=list)
    notes:      list[str]     = field(default_factory=list)

    @property
    def is_clean(self) -> bool:
        return len(self.findings) == 0

    def summary(self) -> str:
        if self.is_clean:
            return f"{self.check_name}: none"
        return f"{self.check_name}: {len(self.findings)} finding(s)"
