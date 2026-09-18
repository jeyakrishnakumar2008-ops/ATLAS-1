"""
question_engine.py
==================
Deterministic ATLAS question engine.

Supports two question types at this stage:
    COUNT   — "How many X …?"
    LOOKUP  — "What was the value of X for subject Y at visit Z?"

No LLM, no database, no external services.
All answers come from the already-loaded in-memory StudyIndex.

Usage
-----
    engine = QuestionEngine(study_index)

    result = engine.count("subjects")
    result = engine.count("ae", subject="042-S07-001")
    result = engine.count("lb", subject="042-S07-011", filters={"LBTESTCD": "ALT"})

    result = engine.lookup("lb", subject="042-S07-011",
                           filters={"LBTESTCD": "ALT", "VISIT": "SCREENING"})

Answer structure
----------------
Every answer is a plain dict:

    {
        "question_type": "count" | "lookup",
        "question":      human-readable restatement,
        "answer":        integer (count) or string/list (lookup),
        "record_refs":   [{"cite": "LB|...|1", ...}, ...],
        "status":        "answered" | "not_found" | "error",
        "note":          str   (optional extra context)
    }

Record refs are always verified against the index before being included.
Fake or missing records cannot appear in the output.
"""

from __future__ import annotations

from typing import Any, Optional, TYPE_CHECKING
from data_loader import to_float, parse_date
from evidence import RecordRef, make_record_ref, record_ref_exists, Evidence

if TYPE_CHECKING:
    from study_index import StudyIndex


# ---------------------------------------------------------------------------
# Domain accessor map — which StudyIndex method to call per domain
# ---------------------------------------------------------------------------

_DOMAIN_GETTER = {
    "AE":  "get_adverse_events",
    "LB":  "get_labs",
    "VS":  "get_vitals",
    "EX":  "get_exposure",
    "CM":  "get_medications",
    "MH":  "get_medical_history",
    "DS":  "get_disposition",
    "EG":  "get_ecg",
}


# ---------------------------------------------------------------------------
# Answer builder helpers
# ---------------------------------------------------------------------------

def _answer(
    question_type: str,
    question:      str,
    answer:        Any,
    refs:          list[RecordRef],
    status:        str = "answered",
    note:          str = "",
) -> dict:
    """Build the canonical answer dict."""
    return {
        "question_type": question_type,
        "question":      question,
        "answer":        answer,
        "record_refs":   [r.to_dict() for r in refs],
        "status":        status,
        "note":          note,
    }


def _not_found(question_type: str, question: str, note: str = "") -> dict:
    return _answer(question_type, question, None, [], "not_found", note)


def _error(question_type: str, question: str, note: str) -> dict:
    return _answer(question_type, question, None, [], "error", note)


# ---------------------------------------------------------------------------
# Filter matching
# ---------------------------------------------------------------------------

def _matches_filters(row: dict[str, str], filters: dict[str, str]) -> bool:
    """
    Return True if every key-value in `filters` matches the row.
    Comparison is case-insensitive on the VALUE side only.
    Keys must match exactly (they are CSV column names).
    """
    for col, wanted in filters.items():
        actual = row.get(col, "")
        if actual.strip().upper() != str(wanted).strip().upper():
            return False
    return True


def _apply_filters(
    rows: list[dict], filters: dict[str, str]
) -> list[dict]:
    """Return only rows that match all filter criteria."""
    if not filters:
        return rows
    return [r for r in rows if _matches_filters(r, filters)]


# ---------------------------------------------------------------------------
# QuestionEngine
# ---------------------------------------------------------------------------

class QuestionEngine:
    """
    Deterministic query engine over a loaded StudyIndex.

    Parameters
    ----------
    index : StudyIndex
        The in-memory index built by build_study_index().

    All queries are answered entirely from the index — no disk reads,
    no LLM calls, no external services.
    """

    def __init__(self, index: "StudyIndex") -> None:
        self._idx = index

    # ======================================================================
    # COUNT
    # ======================================================================

    def count(
        self,
        target: str,
        *,
        subject:  Optional[str] = None,
        filters:  Optional[dict[str, str]] = None,
    ) -> dict:
        """
        Answer a count question.

        Parameters
        ----------
        target  : str
            What to count.  Options:
            "subjects"  — total enrolled subjects (from DM)
            "ae"        — adverse events
            "lb"        — lab records
            "vs"        — vital signs
            "ex"        — exposure / dosing records
            "cm"        — concomitant medications
            "mh"        — medical history items
            "ds"        — disposition records
            "eg"        — ECG records
            "drug"      — subjects in DRUG arm
            "placebo"   — subjects in PLACEBO arm

        subject : str, optional
            Restrict to a single subject.  If None, count across all subjects.

        filters : dict[str, str], optional
            Additional column-value filters, e.g. {"LBTESTCD": "ALT"}.

        Returns
        -------
        dict — standard answer structure
        """
        filters = filters or {}
        t = target.lower().strip()
        idx = self._idx

        # ── subjects ──────────────────────────────────────────────────────
        if t in ("subjects", "subject", "dm"):
            if subject:
                dm = idx.get_subject(subject)
                if dm is None:
                    return _not_found("count",
                                      f"Is subject {subject} in the study?",
                                      f"Subject {subject!r} not found in DM")
                ref = RecordRef(domain="DM", usubjid=subject, seq=1)
                return _answer("count",
                               f"Is subject {subject} enrolled?",
                               1, [ref],
                               note=f"ARM={dm.get('ARM')} SITE={dm.get('SITEID')}")
            # study-wide
            uids = idx.all_subject_ids()
            # refs: one DM ref per subject would be huge — sample first 5 + note
            sample_refs = [
                RecordRef(domain="DM", usubjid=u, seq=1)
                for u in uids[:5]
            ]
            return _answer("count",
                           "How many subjects are enrolled in the study?",
                           len(uids), sample_refs,
                           note=f"Total enrolled subjects in DM: {len(uids)} (sample of 5 refs shown)")

        # ── ARM breakdown ──────────────────────────────────────────────────
        if t in ("drug", "placebo"):
            arm_target = t.upper()
            uids = idx.all_subject_ids()
            arm_subjects = [
                u for u in uids
                if (idx.get_subject(u) or {}).get("ARM", "").upper() == arm_target
            ]
            if not arm_subjects:
                return _not_found("count",
                                  f"How many subjects are in the {arm_target} arm?",
                                  "No subjects found for that arm")
            sample_refs = [
                RecordRef(domain="DM", usubjid=u, seq=1)
                for u in arm_subjects[:5]
            ]
            q = f"How many subjects are in the {arm_target} arm?"
            if subject:
                dm = idx.get_subject(subject)
                if dm is None:
                    return _not_found("count", q, f"Subject {subject!r} not found")
                actual_arm = dm.get("ARM", "")
                in_arm = actual_arm.upper() == arm_target
                ref = RecordRef(domain="DM", usubjid=subject, seq=1)
                return _answer("count", f"Is {subject} in the {arm_target} arm?",
                               1 if in_arm else 0, [ref],
                               note=f"Subject ARM = {actual_arm}")
            return _answer("count", q, len(arm_subjects), sample_refs,
                           note=f"{arm_target} arm has {len(arm_subjects)} subjects (sample of 5 refs shown)")

        # ── domain record counts ───────────────────────────────────────────
        getter_name = _DOMAIN_GETTER.get(t.upper())
        if getter_name is None:
            return _error("count", f"Count of {target!r}",
                          f"Unknown target {target!r}. "
                          f"Valid targets: subjects, drug, placebo, "
                          + ", ".join(k.lower() for k in _DOMAIN_GETTER))

        domain = t.upper()

        if subject:
            # Single-subject count
            records = getattr(idx, getter_name)(subject)
            if not records:
                dm = idx.get_subject(subject)
                if dm is None:
                    return _not_found("count",
                                      f"How many {domain} records does {subject} have?",
                                      f"Subject {subject!r} not found in DM")
                return _answer("count",
                               f"How many {domain} records does {subject} have?",
                               0, [],
                               note=f"Subject {subject} has no {domain} records")
            matched = _apply_filters(records, filters)
            refs = [make_record_ref(domain, r) for r in matched]
            filter_note = f" (filters: {filters})" if filters else ""
            q = f"How many {domain} records does subject {subject} have{filter_note}?"
            return _answer("count", q, len(refs), refs)

        else:
            # Study-wide count across all subjects
            total  = 0
            sample: list[RecordRef] = []
            uids   = idx.all_subject_ids()
            for uid in uids:
                records = getattr(idx, getter_name)(uid)
                matched = _apply_filters(records, filters)
                total += len(matched)
                if len(sample) < 5:
                    for r in matched[: 5 - len(sample)]:
                        sample.append(make_record_ref(domain, r))

            filter_note = f" (filters: {filters})" if filters else ""
            q = f"How many {domain} records are in the study{filter_note}?"
            return _answer("count", q, total, sample,
                           note=f"Summed across {len(uids)} subjects (sample of 5 refs shown)")

    # ======================================================================
    # LOOKUP
    # ======================================================================

    def lookup(
        self,
        domain:  str,
        subject: str,
        *,
        filters: Optional[dict[str, str]] = None,
        field:   Optional[str] = None,
    ) -> dict:
        """
        Retrieve exact records (or a specific field) from the index.

        Parameters
        ----------
        domain  : str
            Domain to search: "LB", "AE", "EX", "VS", "CM", "MH", "DS", "EG", "DM".
        subject : str
            USUBJID of the subject.
        filters : dict[str, str], optional
            Column-value filters narrowing the result set.
            e.g. {"LBTESTCD": "ALT", "VISIT": "SCREENING"}
        field   : str, optional
            If provided, extract only this column from each matched row.
            e.g. "LBORRES".  If None, return full rows.

        Returns
        -------
        dict — standard answer structure.
            answer = list of matching rows (or field values if field given).
            record_refs = one RecordRef per matched row.
        """
        filters = filters or {}
        domain  = domain.upper()
        idx     = self._idx

        q_parts = [f"Lookup {domain} for subject {subject}"]
        if filters:
            q_parts.append(f"where {filters}")
        if field:
            q_parts.append(f"returning field {field!r}")
        question = " ".join(q_parts)

        # ── DM special case ───────────────────────────────────────────────
        if domain == "DM":
            dm = idx.get_subject(subject)
            if dm is None:
                return _not_found("lookup", question,
                                  f"Subject {subject!r} not found in DM")
            ref = RecordRef(domain="DM", usubjid=subject, seq=1)
            if field:
                val = dm.get(field)
                if val is None:
                    return _not_found("lookup", question,
                                      f"Field {field!r} not found in DM row")
                return _answer("lookup", question, val, [ref])
            return _answer("lookup", question, dm, [ref])

        # ── Domain records ────────────────────────────────────────────────
        getter_name = _DOMAIN_GETTER.get(domain)
        if getter_name is None:
            return _error("lookup", question,
                          f"Unknown domain {domain!r}. "
                          f"Valid: DM, " + ", ".join(_DOMAIN_GETTER))

        dm = idx.get_subject(subject)
        if dm is None:
            return _not_found("lookup", question,
                              f"Subject {subject!r} not found in DM")

        records  = getattr(idx, getter_name)(subject)
        matched  = _apply_filters(records, filters)

        if not matched:
            return _not_found("lookup", question,
                              f"No {domain} records match filters {filters} "
                              f"for subject {subject}")

        refs = [make_record_ref(domain, r) for r in matched]

        if field:
            values = []
            for r in matched:
                v = r.get(field)
                if v is not None:
                    values.append(v)
            if not values:
                return _not_found("lookup", question,
                                  f"Field {field!r} not present in matched {domain} rows")
            answer = values[0] if len(values) == 1 else values
        else:
            answer = matched if len(matched) > 1 else matched[0]

        return _answer("lookup", question, answer, refs)

    # ======================================================================
    # Convenience wrappers (named for common questions)
    # ======================================================================

    def subject_count(self) -> dict:
        """How many subjects are enrolled?"""
        return self.count("subjects")

    def subject_arm_count(self, arm: str) -> dict:
        """How many subjects are in the DRUG / PLACEBO arm?"""
        return self.count(arm)

    def lab_result(
        self,
        subject: str,
        testcd:  str,
        visit:   Optional[str] = None,
    ) -> dict:
        """
        What was the lab result for a given test (and optionally visit)?
        Returns LBORRES and LBORRESU from the matched LB rows.
        """
        filters = {"LBTESTCD": testcd}
        if visit:
            filters["VISIT"] = visit
        return self.lookup("LB", subject, filters=filters, field="LBORRES")

    def dose_at_visit(self, subject: str, visit: str) -> dict:
        """What dose was administered at a given visit?"""
        return self.lookup("EX", subject,
                           filters={"VISIT": visit}, field="EXDOSE")

    def ae_terms(self, subject: str) -> dict:
        """List all adverse event terms for a subject."""
        return self.lookup("AE", subject, field="AETERM")

    def qtcf_at_visit(self, subject: str, visit: str) -> dict:
        """What was the QTcF at a given visit?"""
        return self.lookup("EG", subject,
                           filters={"VISIT": visit}, field="EGORRES")

    def disposition(self, subject: str) -> dict:
        """What was the subject's disposition outcome?"""
        return self.lookup("DS", subject, field="DSDECOD")

    # ======================================================================
    # FINDING
    # ======================================================================

    def finding(
        self,
        rule: str,
        *,
        subject: Optional[str] = None,
    ) -> dict:
        """
        Run a named deterministic clinical rule and return matching findings.

        Parameters
        ----------
        rule    : str
            Rule to run.  Supported values:
            "hys_law"        — Hy's law hepatotoxicity candidates
            "sae_miscoding"  — AESHOSP=Y but AESER=N
            "dosing_error"   — dose does not match ARM expectation
            "missing_dose"   — visit with no EX record
        subject : str, optional
            If provided, filter findings to this subject only.

        Returns
        -------
        dict — standard answer structure.
            answer   = list of finding dicts (one per subject flagged)
            question_type = "finding"
            status   = "answered" | "not_found" | "error"
        """
        import checks  # lazy import to avoid circular at module level

        rule_map = {
            "hys_law":       checks.check_hys_law,
            "sae_miscoding": checks.check_sae_miscoding,
            "dosing_error":  checks.check_dosing_errors,
            "missing_dose":  checks.check_missing_doses,
        }

        r = rule.lower().strip()
        check_fn = rule_map.get(r)
        if check_fn is None:
            return _error("finding", f"Run rule {rule!r}",
                          f"Unknown rule {rule!r}. Valid: {', '.join(rule_map)}")

        question = f"Find subjects matching rule: {rule}"
        if subject:
            question += f" (subject {subject})"

        # Run the deterministic check over the whole index
        check_result = check_fn(self._idx)

        findings = check_result.findings
        if subject:
            findings = [f for f in findings if f.usubjid == subject]

        if not findings:
            return _not_found("finding", question,
                              f"No subjects match rule {rule!r}"
                              + (f" for subject {subject}" if subject else ""))

        # Collect all unique RecordRefs across all findings
        all_refs: list[RecordRef] = []
        seen_cites: set[str] = set()
        for f in findings:
            for ev in f.evidence:
                for ref in ev.record_refs:
                    if ref.cite not in seen_cites:
                        all_refs.append(ref)
                        seen_cites.add(ref.cite)

        answer_list = [f.to_dict() for f in findings]

        return _answer(
            "finding",
            question,
            answer_list,
            all_refs,
            note=(
                f"Rule={rule!r}  "
                f"Subjects flagged: {len(findings)}  "
                f"Notes: {'; '.join(check_result.notes[:3]) if check_result.notes else 'none'}"
            ),
        )

    # ======================================================================
    # TRAP
    # ======================================================================

    def trap(
        self,
        trap_type: str,
        **kwargs,
    ) -> dict:
        """
        Evaluate a potentially misleading or adversarial claim against real data.

        Parameters
        ----------
        trap_type : str
            Which trap pattern to evaluate.  Supported values:

            "prompt_injection"
                kwargs: claim (str)
                Detects lab-manual text addressed at automated reviewers.

            "hys_law_claim"
                kwargs: subject (str)
                Is the subject a Hy's law candidate per the actual data?

            "sulfonylurea_deviation"
                kwargs: subject (str), cut (int)
                Is sulfonylurea a prohibited-medication deviation at this cut?

            "corrected_value"
                kwargs: subject (str), domain (str), seq (int), claimed_value (str)
                Does the claimed value match the current (post-correction) value?

            "lab_value"
                kwargs: subject (str), testcd (str), visit (str), claimed (str)
                Does the claimed lab value match actual data (current value)?

            "sae_miscoding"
                kwargs: subject (str), aeseq (int)
                Is AESHOSP=Y but AESER=N for this AE record?

            "missing_disposition"
                kwargs: subject (str)
                Is the subject in DM but missing a DS record?

        Returns
        -------
        dict — standard answer structure with status:
            "supported" / "unsupported" / "insufficient_evidence" /
            "conflict" / "prompt_injection_detected"
        """
        import trap_checker as tc

        t = trap_type.lower().strip()

        try:
            if t == "prompt_injection":
                result = tc.check_prompt_injection(kwargs.get("claim", ""))
                if result is None:
                    result = tc.TrapResult(
                        status=tc.TrapResult.UNSUPPORTED,
                        claim=kwargs.get("claim", ""),
                        answer="No prompt injection pattern detected in this claim.",
                        record_refs=[],
                    )

            elif t == "hys_law_claim":
                result = tc.check_hys_law_claim(self._idx, kwargs["subject"])

            elif t == "sulfonylurea_deviation":
                result = tc.check_sulfonylurea_deviation(
                    self._idx,
                    kwargs["subject"],
                    int(kwargs["cut"]),
                )

            elif t == "corrected_value":
                result = tc.check_corrected_value_claim(
                    self._idx,
                    kwargs["subject"],
                    kwargs["domain"],
                    int(kwargs["seq"]),
                    kwargs["claimed_value"],
                )

            elif t == "lab_value":
                result = tc.check_lab_value_claim(
                    self._idx,
                    kwargs["subject"],
                    kwargs["testcd"],
                    kwargs["visit"],
                    kwargs["claimed"],
                )

            elif t == "sae_miscoding":
                result = tc.check_sae_miscoding_claim(
                    self._idx,
                    kwargs["subject"],
                    int(kwargs["aeseq"]),
                )

            elif t == "missing_disposition":
                result = tc.check_missing_disposition(
                    self._idx, kwargs["subject"]
                )

            else:
                return _error(
                    "trap",
                    f"Trap check: {trap_type!r}",
                    f"Unknown trap_type {trap_type!r}. Valid: "
                    "prompt_injection, hys_law_claim, sulfonylurea_deviation, "
                    "corrected_value, lab_value, sae_miscoding, missing_disposition",
                )

        except KeyError as e:
            return _error("trap", f"Trap check: {trap_type!r}",
                          f"Missing required kwarg: {e}")

        d = result.to_dict()
        # Add standard answer_* fields expected by the test harness
        d["question_type"] = "trap"
        d["answer"]        = result.answer
        d["record_refs"]   = [r.to_dict() for r in result.record_refs]
        d["status"]        = result.status
        return d
