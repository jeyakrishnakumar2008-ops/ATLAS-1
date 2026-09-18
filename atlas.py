"""
atlas.py
========
Central facade for the ATLAS solver.

StudySentinel
-------------
Wraps StudyData (the loader) and exposes high-level query methods
that the main.py entry point will call.

This file is intentionally thin at this stage.  Concrete check logic
will be added in checks.py; atlas.py just wires everything together.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from data_loader import StudyData, load_all_data, DomainTable
from evidence import CheckResult, Finding
from study_index import StudyIndex, build_study_index
from rules import get_protocol_version

# ---------------------------------------------------------------------------
# Response file helpers
# ---------------------------------------------------------------------------

RESPONSES_DIR = (
    Path(__file__).resolve().parent
    / "hackathon-data"
    / "hackathon-data"
    / "responses"
)


def _load_json(filename: str) -> dict:
    path = RESPONSES_DIR / filename
    with open(path, encoding="utf-8") as f:
        return json.load(f)


class SiteReplies:
    """
    Look up what the site said when queried about a record.
    Key format: "DOMAIN|USUBJID|SEQ"
    A miss returns the _default reply.
    """

    def __init__(self):
        data = _load_json("site_replies.json")
        self._replies: dict[str, list[str]] = data.get("replies", {})
        self._default: list[str] = data.get(
            "_default",
            ["ANSWERED", "Data verified against source documents. No change."]
        )

    def query(self, domain: str, usubjid: str, seq: int | str) -> tuple[str, str]:
        key = f"{domain}|{usubjid}|{seq}"
        result = self._replies.get(key, self._default)
        return result[0], result[1]   # (status, text)


class MonitorDecisions:
    """
    Look up the medical monitor's decision on a finding.
    Key format: "FINDING_CODE|USUBJID"  or  "FINDING_CODE|SITEID"
    Returns: (decision, reason) where decision in {APPROVED, REJECTED, CLARIFY}

    On CLARIFY: answer the clarifying question from the data and resubmit.
    The monitor's response on resubmission is always APPROVED.
    """

    def __init__(self):
        data = _load_json("monitor_decisions.json")
        self._decisions: dict[str, list[str]] = data.get("decisions", {})

    def query(self, code: str, subject_or_site: str) -> tuple[str, str]:
        key = f"{code}|{subject_or_site}"
        result = self._decisions.get(key)
        if result is None:
            return ("APPROVED", "No specific decision on record; finding stands.")
        return result[0], result[1]   # (decision, reason)


# ---------------------------------------------------------------------------
from question_engine import QuestionEngine

# ---------------------------------------------------------------------------
# StudySentinel — the top-level object callers interact with
# ---------------------------------------------------------------------------

class StudySentinel:
    """
    Main interface for the ATLAS solver.

    Parameters
    ----------
    cut : int
        The data cut to evaluate (1–12).  All data and protocol logic
        will be scoped to this cut.
    """

    def __init__(self, cut: int = 12):
        self.cut = cut
        self.data = load_all_data(cut=cut)
        self.protocol_version = self.data.protocol_version()
        self.index = build_study_index(self.data)   # in-memory index (loaded once)
        self.engine = QuestionEngine(self.index)
        self.site_replies = SiteReplies()
        self.monitor = MonitorDecisions()

    def reload(self, cut: int) -> None:
        """
        Reload and reindex the study under a new cut/protocol version.
        Rebuilds index and question engine in memory.
        """
        self.cut = cut
        self.data = load_all_data(cut=cut)
        self.protocol_version = self.data.protocol_version()
        self.index = build_study_index(self.data)
        self.engine = QuestionEngine(self.index)

    # ------------------------------------------------------------------
    # Query delegates
    # ------------------------------------------------------------------

    def count(self, target: str, **kwargs) -> dict:
        return self.engine.count(target, **kwargs)

    def lookup(self, domain: str, subject: str, **kwargs) -> dict:
        return self.engine.lookup(domain, subject, **kwargs)

    def finding(self, rule: str, **kwargs) -> dict:
        return self.engine.finding(rule, **kwargs)

    def trap(self, trap_type: str, **kwargs) -> dict:
        return self.engine.trap(trap_type, **kwargs)

    # ------------------------------------------------------------------
    # Convenience passthrough
    # ------------------------------------------------------------------

    def subjects(self) -> list[str]:
        return self.data.subjects()

    def summary(self) -> dict[str, Any]:
        counts = self.data.row_counts()
        return {
            "cut": self.cut,
            "protocol_version": self.protocol_version,
            "subjects": len(self.data.subjects()),
            "ae_rows":  counts.get("AE", 0),
            "lb_rows":  counts.get("LB", 0),
            "vs_rows":  counts.get("VS", 0),
            "ex_rows":  counts.get("EX", 0),
            "cm_rows":  counts.get("CM", 0),
        }
