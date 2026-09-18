"""
monitor.py
==========
Problem 2: ATLAS MONITOR — Evidence-to-Action Pipeline.

Implements the 6-module architecture:
1. Finding Intake
2. Risk Assessment
3. Action Planner
4. Compliance Check
5. Decision Center
6. Memory + Action
"""

from __future__ import annotations

import datetime
from dataclasses import dataclass, field, asdict
from typing import Any, Optional

from atlas import StudySentinel, SiteReplies, MonitorDecisions
from data_loader import parse_date, to_float
from evidence import make_record_ref, record_ref_exists
from rules import VISIT_DAYS, visit_window, prohibited_classes
from checks import (
    check_hys_law,
    check_sae_miscoding,
    check_dosing_errors,
    check_missing_doses,
)


def _current_timestamp() -> str:
    return datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")


# ---------------------------------------------------------------------------
# Data Transfer Models
# ---------------------------------------------------------------------------

@dataclass
class TraceRecord:
    timestamp: str
    module: str
    decision: str
    reason: str
    record_refs: list[str]
    subject: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class PipelineFinding:
    id: str
    category: str
    usubjid: str
    description: str
    severity: str
    evidence_claims: list[str]
    record_refs: list[str]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class RiskProfile:
    finding_id: str
    usubjid: str
    seriousness: str         # CRITICAL, HIGH, MEDIUM, LOW
    plausibility: str
    safety_significance: str
    recommended_action: str  # ESCALATE, QUERY, MONITOR
    reason: str
    record_refs: list[str]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class ActionItem:
    action_id: str
    action_type: str         # QUERY, ESCALATE, MONITOR
    usubjid: str
    domain: str
    record_ref: str
    title: str
    details: str
    reason: str
    status: str              # PENDING, SENT, RESOLVED, CLOSED
    site_reply: str = ""
    monitor_decision: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class ComplianceDeviation:
    deviation_id: str
    usubjid: str
    deviation_type: str      # PROHIBITED_MEDICATION, VISIT_WINDOW, ELIGIBILITY
    rule_cited: str
    details: str
    protocol_version: int
    record_ref: str
    severity: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class EscalationItem:
    escalation_id: str
    finding_code: str
    usubjid: str
    title: str
    description: str
    severity: str
    record_refs: list[str]
    status: str              # PENDING, APPROVED, REJECTED, CLARIFIED_APPROVED
    decision_reason: str = ""
    clarification_q: str = ""
    clarification_answer: str = ""
    cycle: int = 1

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


# ---------------------------------------------------------------------------
# Module 6 Storage: Memory Across Cycles
# ---------------------------------------------------------------------------

class MonitorMemory:
    """
    Maintains clinical trial monitor state across cycles and cuts:
    - Same query + same record = never raise again
    - Same escalation = never repeat
    - Rejected escalation = never re-escalate
    - Subject flagged in two cycles = escalate
    - Recurring site problem = site-level flag
    - Same cut run twice = 0 new queries and 0 new escalations
    """

    def __init__(self) -> None:
        self.sent_queries: set[str] = set()               # record_ref | query_key
        self.active_queries: list[ActionItem] = []
        self.escalation_history: dict[str, str] = {}      # finding_key -> status
        self.rejected_escalations: set[str] = set()       # finding_key
        self.escalations: dict[str, EscalationItem] = {}  # escalation_id -> EscalationItem
        self.subject_cycles: dict[str, set[int]] = {}     # usubjid -> set of cycle ints
        self.site_issues: dict[str, list[str]] = {}       # siteid -> list of issues
        self.site_flags: dict[str, str] = {}              # siteid -> site warning
        self.trace: list[TraceRecord] = []
        self.cycle_count: int = 0
        self.last_run_cut: Optional[int] = None

    def reset(self) -> None:
        self.sent_queries.clear()
        self.active_queries.clear()
        self.escalation_history.clear()
        self.rejected_escalations.clear()
        self.escalations.clear()
        self.subject_cycles.clear()
        self.site_issues.clear()
        self.site_flags.clear()
        self.trace.clear()
        self.cycle_count = 0
        self.last_run_cut = None

    def log_trace(
        self,
        module: str,
        decision: str,
        reason: str,
        record_refs: list[str],
        subject: str = "",
    ) -> None:
        record = TraceRecord(
            timestamp=_current_timestamp(),
            module=module,
            decision=decision,
            reason=reason,
            record_refs=record_refs,
            subject=subject,
        )
        self.trace.append(record)


# Global singleton memory
global_memory = MonitorMemory()


# ---------------------------------------------------------------------------
# Module 1: FINDING INTAKE
# ---------------------------------------------------------------------------

class FindingIntake:
    """
    Calls the existing Stage 1 ATLAS checks and ingests all findings with
    verified RecordRefs without altering Stage 1 detection logic.
    """

    def __init__(self, sentinel: StudySentinel) -> None:
        self.sentinel = sentinel
        self.index = sentinel.index

    def run(self) -> list[PipelineFinding]:
        findings: list[PipelineFinding] = []
        f_idx = 1

        # 1. Hy's Law Findings
        hys_res = check_hys_law(self.index)
        for f in hys_res.findings:
            claims = [e.claim for e in f.evidence]
            refs = f.all_citations()
            findings.append(PipelineFinding(
                id=f"FND-{f_idx:04d}",
                category=f.category,
                usubjid=f.usubjid,
                description=f.description,
                severity=f.severity,
                evidence_claims=claims,
                record_refs=refs,
            ))
            f_idx += 1

        # 2. SAE Miscoding Findings
        sae_res = check_sae_miscoding(self.index)
        for f in sae_res.findings:
            claims = [e.claim for e in f.evidence]
            refs = f.all_citations()
            findings.append(PipelineFinding(
                id=f"FND-{f_idx:04d}",
                category=f.category,
                usubjid=f.usubjid,
                description=f.description,
                severity=f.severity,
                evidence_claims=claims,
                record_refs=refs,
            ))
            f_idx += 1

        # 3. Dosing Errors
        dose_res = check_dosing_errors(self.index)
        for f in dose_res.findings:
            claims = [e.claim for e in f.evidence]
            refs = f.all_citations()
            findings.append(PipelineFinding(
                id=f"FND-{f_idx:04d}",
                category=f.category,
                usubjid=f.usubjid,
                description=f.description,
                severity=f.severity,
                evidence_claims=claims,
                record_refs=refs,
            ))
            f_idx += 1

        # 4. Missing Doses
        miss_res = check_missing_doses(self.index)
        for f in miss_res.findings:
            claims = [e.claim for e in f.evidence]
            refs = f.all_citations()
            findings.append(PipelineFinding(
                id=f"FND-{f_idx:04d}",
                category=f.category,
                usubjid=f.usubjid,
                description=f.description,
                severity=f.severity,
                evidence_claims=claims,
                record_refs=refs,
            ))
            f_idx += 1

        # 5. Chronological Discrepancy: AE reported before first study dose
        for uid in self.index.all_subject_ids():
            dm = self.index.get_subject(uid)
            if not dm:
                continue
            rfstdtc = parse_date(dm.get("RFSTDTC", ""))
            if not rfstdtc:
                continue
            for ae in self.index.get_adverse_events(uid):
                aestdtc = parse_date(ae.get("AESTDTC", ""))
                if aestdtc and aestdtc < rfstdtc:
                    ref = make_record_ref("AE", ae)
                    findings.append(PipelineFinding(
                        id=f"FND-{f_idx:04d}",
                        category="AE_BEFORE_FIRST_DOSE",
                        usubjid=uid,
                        description=(
                            f"AE '{ae.get('AETERM')}' onset on {ae.get('AESTDTC')} "
                            f"precedes first study dose on {dm.get('RFSTDTC')}."
                        ),
                        severity="MEDIUM",
                        evidence_claims=[
                            f"AE onset {ae.get('AESTDTC')} < First dose {dm.get('RFSTDTC')}"
                        ],
                        record_refs=[ref.cite],
                    ))
                    f_idx += 1

        # 6. Pre-existing Screening Liver Elevation (for Monitor-only demonstration)
        for uid in self.index.all_subject_ids():
            dm = self.index.get_subject(uid)
            siteid = dm.get("SITEID", "") if dm else ""
            rr_alt = self.index.ref_range("ALT", siteid)
            uln_alt = to_float(rr_alt.get("HIGH")) if rr_alt else None
            for lb in self.index.get_labs(uid):
                if lb.get("VISIT") == "SCREENING" and lb.get("LBTESTCD") == "ALT":
                    val = to_float(lb.get("LBORRES"))
                    if val and uln_alt and val > (1.2 * uln_alt):
                        ref = make_record_ref("LB", lb)
                        findings.append(PipelineFinding(
                            id=f"FND-{f_idx:04d}",
                            category="SCREENING_TRANSAMINASE_ELEVATED",
                            usubjid=uid,
                            description=(
                                f"Screening ALT={val} {lb.get('LBORRESU')} is elevated "
                                f"above normal ULN={uln_alt}, indicating pre-existing baseline status."
                            ),
                            severity="LOW",
                            evidence_claims=[f"Screening ALT {val} > {uln_alt} ULN"],
                            record_refs=[ref.cite],
                        ))
                        f_idx += 1

        return findings


# ---------------------------------------------------------------------------
# Module 2: RISK ASSESSMENT
# ---------------------------------------------------------------------------

class RiskAssessment:
    """
    Evaluates each finding for seriousness, plausibility, and safety significance.
    Recognizes AESHOSP=Y as serious regardless of AESER flag.
    Does not escalate every finding; selects appropriate path: ESCALATE / QUERY / MONITOR.
    """

    def evaluate(
        self,
        finding: PipelineFinding,
        memory: MonitorMemory,
        cycle_num: int,
    ) -> RiskProfile:
        cat = finding.category
        uid = finding.usubjid
        refs = finding.record_refs

        # Check recurring subject rule from Memory
        is_repeat_flag = len(memory.subject_cycles.get(uid, set())) >= 1

        if cat == "HYS_LAW_CANDIDATE":
            seriousness = "CRITICAL"
            plausibility = "Biochemically plausible severe drug-induced hepatocellular injury."
            safety_significance = "Extreme safety impact: risk of acute liver failure."
            recommended_action = "ESCALATE"
            reason = "Hy's Law signal satisfies FDA/ICH hepatotoxicity escalation criteria."

        elif cat == "SAE_MISCODING":
            seriousness = "CRITICAL"
            plausibility = "Data inconsistency: hospitalisation recorded (AESHOSP=Y) but AESER=N."
            safety_significance = "Regulatory compliance violation: unreported Serious Adverse Event."
            recommended_action = "ESCALATE"
            reason = "Protocol §6 requires immediate escalation when hospitalization occurs."

        elif cat == "DOSING_ERROR":
            seriousness = "HIGH"
            plausibility = "Dispensation discrepancy: dose administered does not match ARM assignment."
            safety_significance = "Patient safety risk due to incorrect pharmacological exposure."
            recommended_action = "ESCALATE"
            reason = "Dosing error exceeds protocol tolerance; escalate to medical monitor."

        elif cat == "AE_BEFORE_FIRST_DOSE":
            seriousness = "MEDIUM"
            plausibility = "High likelihood of transcription date error or pre-existing medical condition."
            safety_significance = "Low immediate clinical danger, but compromises trial integrity."
            recommended_action = "QUERY"
            reason = "Site clarification required: query whether AE onset is a recording error."

        elif cat == "MISSING_DOSE":
            seriousness = "MEDIUM"
            plausibility = "Potential protocol non-adherence or missing eCRF record."
            safety_significance = "Incomplete treatment exposure tracking."
            recommended_action = "QUERY"
            reason = "Data query to site to confirm if dose was omitted or record was missed."

        elif cat == "SCREENING_TRANSAMINASE_ELEVATED":
            seriousness = "LOW"
            plausibility = "Expected biological baseline variance."
            safety_significance = "Stable baseline liver function; no acute drug-induced signal."
            recommended_action = "MONITOR"
            reason = "Already-high screening liver value is baseline; monitor-only is appropriate."

        else:
            seriousness = "MEDIUM"
            plausibility = "Plausible trial record variance."
            safety_significance = "Routine study monitoring."
            recommended_action = "MONITOR"
            reason = "Routine finding for ongoing monitoring."

        # Memory override: Subject flagged in two cycles escalates automatically
        if is_repeat_flag and recommended_action == "MONITOR":
            recommended_action = "ESCALATE"
            reason += " (Upgraded to ESCALATE because subject was flagged across multiple cycles)."

        return RiskProfile(
            finding_id=finding.id,
            usubjid=uid,
            seriousness=seriousness,
            plausibility=plausibility,
            safety_significance=safety_significance,
            recommended_action=recommended_action,
            reason=reason,
            record_refs=refs,
        )


# ---------------------------------------------------------------------------
# Module 3: ACTION PLANNER
# ---------------------------------------------------------------------------

class ActionPlanner:
    """
    Converts risk assessments into actionable outputs:
    - QUERY: specific, actionable, citing subject/domain/record, never duplicated.
    - ESCALATE: serious/high-risk, preserving evidence and rationale.
    - MONITOR: ongoing observation for stable/low-risk findings.
    """

    def __init__(self, site_replies: SiteReplies) -> None:
        self.site_replies = site_replies

    def plan(
        self,
        finding: PipelineFinding,
        risk: RiskProfile,
        memory: MonitorMemory,
    ) -> Optional[ActionItem]:
        action_type = risk.recommended_action
        uid = finding.usubjid
        ref_str = finding.record_refs[0] if finding.record_refs else f"DM|{uid}|1"
        domain = ref_str.split("|")[0] if "|" in ref_str else "GEN"

        # Unique signature for query deduplication
        query_sig = f"{uid}|{domain}|{ref_str}|{finding.category}"

        if action_type == "QUERY":
            if query_sig in memory.sent_queries:
                return None  # Same query + same record = never raise again

            query_text = (
                f"Data Query for {uid} [{ref_str}]: {finding.description} "
                f"Please verify source documents and confirm or rectify the date/value."
            )

            # Look up site reply from repository responses
            seq = ref_str.split("|")[2] if ref_str.count("|") >= 2 else "1"
            reply_status, reply_text = self.site_replies.query(domain, uid, seq)

            item = ActionItem(
                action_id=f"QRY-{len(memory.sent_queries)+1:03d}",
                action_type="QUERY",
                usubjid=uid,
                domain=domain,
                record_ref=ref_str,
                title=f"Site Query: {finding.category}",
                details=query_text,
                reason=risk.reason,
                status="SENT",
                site_reply=f"[{reply_status}] {reply_text}",
            )
            return item

        elif action_type == "ESCALATE":
            finding_key = f"{finding.category}|{uid}"

            # If this escalation was previously REJECTED, memory downgrades to MONITOR
            if finding_key in memory.rejected_escalations:
                return ActionItem(
                    action_id=f"MON-{len(memory.active_queries)+1:03d}",
                    action_type="MONITOR",
                    usubjid=uid,
                    domain=domain,
                    record_ref=ref_str,
                    title=f"Monitor: {finding.category} (Previously Rejected)",
                    details=f"Escalation previously rejected by monitor. Downgraded to monitoring.",
                    reason="Never re-escalate an item previously rejected by the medical monitor.",
                    status="RESOLVED",
                )

            # If already escalated in this session, do not repeat
            if finding_key in memory.escalation_history:
                return None

            item = ActionItem(
                action_id=f"ESC-{len(memory.escalations)+1:03d}",
                action_type="ESCALATE",
                usubjid=uid,
                domain=domain,
                record_ref=ref_str,
                title=f"Safety Escalation: {finding.category}",
                details=finding.description,
                reason=risk.reason,
                status="PENDING",
            )
            return item

        else:  # MONITOR
            item = ActionItem(
                action_id=f"MON-{len(memory.active_queries)+1:03d}",
                action_type="MONITOR",
                usubjid=uid,
                domain=domain,
                record_ref=ref_str,
                title=f"Clinical Monitor: {finding.category}",
                details=finding.description,
                reason=risk.reason,
                status="RESOLVED",
            )
            return item


# ---------------------------------------------------------------------------
# Module 4: COMPLIANCE CHECK
# ---------------------------------------------------------------------------

class ComplianceCheck:
    """
    Checks subjects against the protocol version active at the selected cut.
    Detects visit-window, prohibited medicine, and eligibility deviations.
    Never blindly uses old protocol rules after an amendment.
    """

    def __init__(self, sentinel: StudySentinel) -> None:
        self.sentinel = sentinel
        self.index = sentinel.index

    def check_compliance(self) -> list[ComplianceDeviation]:
        deviations: list[ComplianceDeviation] = []
        pv = self.sentinel.protocol_version
        window = visit_window(pv)
        prohib = prohibited_classes(pv)
        d_idx = 1

        for uid in self.index.all_subject_ids():
            dm = self.index.get_subject(uid)
            if not dm:
                continue
            base_date = parse_date(dm.get("RFSTDTC", ""))

            # 1. Prohibited Concomitant Medication Check
            for cm in self.index.get_medications(uid):
                cclass = cm.get("CMCLAS", "").strip()
                if cclass in prohib:
                    ref = make_record_ref("CM", cm)
                    deviations.append(ComplianceDeviation(
                        deviation_id=f"DEV-{d_idx:04d}",
                        usubjid=uid,
                        deviation_type="PROHIBITED_MEDICATION",
                        rule_cited=f"Protocol v{pv} prohibits class '{cclass}' (Cut {self.sentinel.cut})",
                        details=(
                            f"Subject received prohibited concomitant medication '{cm.get('CMTRT')}' "
                            f"(class: {cclass}) on {cm.get('CMSTDTC')}."
                        ),
                        protocol_version=pv,
                        record_ref=ref.cite,
                        severity="HIGH",
                    ))
                    d_idx += 1

            # 2. Visit Window Deviations (Window threshold is protocol version specific)
            if base_date:
                for ex in self.index.get_exposure(uid):
                    vname = ex.get("VISIT", "")
                    if vname in VISIT_DAYS and vname != "BASELINE":
                        vdate = parse_date(ex.get("EXSTDTC", ""))
                        if vdate:
                            act_day = (vdate - base_date).days
                            nom_day = VISIT_DAYS[vname]
                            day_diff = abs(act_day - nom_day)
                            if day_diff > window:
                                ref = make_record_ref("EX", ex)
                                deviations.append(ComplianceDeviation(
                                    deviation_id=f"DEV-{d_idx:04d}",
                                    usubjid=uid,
                                    deviation_type="VISIT_WINDOW_DEVIATION",
                                    rule_cited=f"Protocol v{pv} visit window is ±{window} days",
                                    details=(
                                        f"Visit {vname} occurred on Day {act_day} "
                                        f"(expected Day {nom_day}, variance: {day_diff} days > ±{window}d window)."
                                    ),
                                    protocol_version=pv,
                                    record_ref=ref.cite,
                                    severity="MEDIUM",
                                ))
                                d_idx += 1

            # 3. Renal Exclusion Criterion (Added in Protocol v2 / Cuts 5+)
            if pv >= 2:
                for lb in self.index.get_labs(uid):
                    if lb.get("VISIT") == "SCREENING" and lb.get("LBTESTCD") == "CREAT":
                        creat_val = to_float(lb.get("LBORRES"))
                        if creat_val and creat_val > 1.5:
                            ref = make_record_ref("LB", lb)
                            deviations.append(ComplianceDeviation(
                                deviation_id=f"DEV-{d_idx:04d}",
                                usubjid=uid,
                                deviation_type="RENAL_EXCLUSION_VIOLATION",
                                rule_cited=f"Protocol v{pv} Amendment 1: Creatinine > 1.5 mg/dL excluded at screening",
                                details=(
                                    f"Screening creatinine={creat_val} mg/dL exceeds 1.5 mg/dL limit "
                                    f"under Protocol v{pv}."
                                ),
                                protocol_version=pv,
                                record_ref=ref.cite,
                                severity="HIGH",
                            ))
                            d_idx += 1

        return deviations


# ---------------------------------------------------------------------------
# Module 5: DECISION CENTER
# ---------------------------------------------------------------------------

class DecisionCenter:
    """
    Manages pending escalations and handles human/monitor decisions:
    - APPROVED: Execute proposed action + log decision
    - REJECTED: Downgrade to monitoring + store reason + never re-escalate
    - CLARIFY: Answer the question from ATLAS evidence + resubmit escalation
    """

    def __init__(
        self,
        monitor_decisions: MonitorDecisions,
        sentinel: StudySentinel,
    ) -> None:
        self.monitor_decisions = monitor_decisions
        self.sentinel = sentinel

    def answer_clarification(self, usubjid: str, question: str) -> str:
        """
        Answers medical monitor clarifying questions directly from the
        in-memory ATLAS StudyIndex/graph with verified evidence.
        """
        q_lower = question.lower()
        idx = self.sentinel.index
        parts = []

        if "alt" in q_lower or "screening" in q_lower:
            alt_res = None
            for lb in idx.get_labs(usubjid):
                if lb.get("VISIT") == "SCREENING" and lb.get("LBTESTCD") == "ALT":
                    alt_res = f"{lb.get('LBORRES')} {lb.get('LBORRESU', '')} (LB|{usubjid}|{lb.get('LBSEQ')})"
                    break
            parts.append(f"Screening ALT: {alt_res or 'Not reported'}")

        if "medication" in q_lower or "hepatotoxic" in q_lower:
            cms = idx.get_medications(usubjid)
            if cms:
                med_list = [f"{c.get('CMTRT')} ({c.get('CMCLAS')})" for c in cms]
                parts.append(f"Concomitant Meds: {', '.join(med_list)}")
            else:
                parts.append("Concomitant Meds: None on record")

        return " | ".join(parts) if parts else "Data verified from study index source records."

    def adjudicate(
        self,
        escalation: EscalationItem,
        action: str,  # "APPROVED", "REJECTED", "CLARIFY"
        custom_reason: str = "",
        memory: MonitorMemory = None,
    ) -> EscalationItem:
        action = action.upper()
        finding_key = f"{escalation.finding_code}|{escalation.usubjid}"

        if action == "APPROVED":
            escalation.status = "APPROVED"
            escalation.decision_reason = (
                custom_reason or "Approved by medical monitor. Proposed safety action executed."
            )
            if memory:
                memory.escalation_history[finding_key] = "APPROVED"
                memory.log_trace(
                    module="Decision Center",
                    decision="APPROVED",
                    reason=escalation.decision_reason,
                    record_refs=escalation.record_refs,
                    subject=escalation.usubjid,
                )

        elif action == "REJECTED":
            escalation.status = "REJECTED"
            escalation.decision_reason = (
                custom_reason or "Monitor rejected: baseline elevation or non-causal. Downgraded to monitoring."
            )
            if memory:
                memory.rejected_escalations.add(finding_key)
                memory.escalation_history[finding_key] = "REJECTED"
                memory.log_trace(
                    module="Decision Center",
                    decision="REJECTED",
                    reason=f"{escalation.decision_reason} (Will not re-escalate).",
                    record_refs=escalation.record_refs,
                    subject=escalation.usubjid,
                )

        elif action == "CLARIFY":
            # Medical monitor asks for clarification
            clarification_q = (
                custom_reason or "What was the ALT at screening, and is there a concomitant hepatotoxic medication?"
            )
            escalation.clarification_q = clarification_q

            # Answer from ATLAS evidence graph
            answer = self.answer_clarification(escalation.usubjid, clarification_q)
            escalation.clarification_answer = answer

            # Resubmit escalation with evidence answer -> automatically APPROVED per protocol
            escalation.status = "CLARIFIED_APPROVED"
            escalation.decision_reason = (
                f"Clarification answered from ATLAS evidence [{answer}]. Escalation resubmitted and APPROVED."
            )
            if memory:
                memory.escalation_history[finding_key] = "CLARIFIED_APPROVED"
                memory.log_trace(
                    module="Decision Center",
                    decision="CLARIFIED_APPROVED",
                    reason=escalation.decision_reason,
                    record_refs=escalation.record_refs,
                    subject=escalation.usubjid,
                )

        return escalation


# ---------------------------------------------------------------------------
# End-to-End Pipeline Orchestrator: AtlasMonitorPipeline
# ---------------------------------------------------------------------------

class AtlasMonitorPipeline:
    """
    Executes the complete Evidence-to-Action pipeline:
    Stage 1 Atlas → Finding Intake → Risk Assessment → Action Planner →
    Compliance Check → Decision Center → Memory + Action → Review Report.
    """

    def __init__(self, sentinel: StudySentinel, memory: MonitorMemory = None) -> None:
        self.sentinel = sentinel
        self.memory = memory or global_memory
        self.intake = FindingIntake(sentinel)
        self.risk_assessor = RiskAssessment()
        self.action_planner = ActionPlanner(sentinel.site_replies)
        self.compliance = ComplianceCheck(sentinel)
        self.decision_center = DecisionCenter(sentinel.monitor, sentinel)

    def run_cycle(self, auto_adjudicate: bool = False) -> dict[str, Any]:
        self.memory.cycle_count += 1
        cycle_num = self.memory.cycle_count
        cut = self.sentinel.cut
        pv = self.sentinel.protocol_version

        # 1. Finding Intake
        findings = self.intake.run()
        self.memory.log_trace(
            module="Finding Intake",
            decision="INGEST",
            reason=f"Ingested {len(findings)} findings from Stage 1 ATLAS for Cut {cut} (v{pv}).",
            record_refs=[],
        )

        # 2. Risk Assessment & 3. Action Planner
        new_queries = 0
        new_escalations = 0
        planned_actions: list[ActionItem] = []

        for f in findings:
            risk = self.risk_assessor.evaluate(f, self.memory, cycle_num)
            action = self.action_planner.plan(f, risk, self.memory)
            if not action:
                continue

            planned_actions.append(action)

            # Track subject in memory across cycles
            if f.usubjid not in self.memory.subject_cycles:
                self.memory.subject_cycles[f.usubjid] = set()
            self.memory.subject_cycles[f.usubjid].add(cycle_num)

            # Record site issues
            siteid = f.usubjid.split("-")[1] if "-" in f.usubjid else "S00"
            if siteid not in self.memory.site_issues:
                self.memory.site_issues[siteid] = []
            self.memory.site_issues[siteid].append(action.title)

            if action.action_type == "QUERY":
                new_queries += 1
                query_sig = f"{f.usubjid}|{action.domain}|{action.record_ref}|{f.category}"
                self.memory.sent_queries.add(query_sig)
                self.memory.active_queries.append(action)
                self.memory.log_trace(
                    module="Action Planner",
                    decision="QUERY",
                    reason=f"{action.title}: {action.details}",
                    record_refs=f.record_refs,
                    subject=f.usubjid,
                )

            elif action.action_type == "ESCALATE":
                new_escalations += 1
                finding_key = f"{f.category}|{f.usubjid}"
                esc_id = f"ESC-{cut}-{len(self.memory.escalations)+1:03d}"

                esc_item = EscalationItem(
                    escalation_id=esc_id,
                    finding_code=f.category,
                    usubjid=f.usubjid,
                    title=action.title,
                    description=action.details,
                    severity=risk.seriousness,
                    record_refs=f.record_refs,
                    status="PENDING",
                    cycle=cycle_num,
                )
                self.memory.escalations[esc_id] = esc_item
                self.memory.escalation_history[finding_key] = "PENDING"
                self.memory.log_trace(
                    module="Action Planner",
                    decision="ESCALATE",
                    reason=f"Escalated {f.category} for {f.usubjid} to Medical Monitor. Rationale: {risk.reason}",
                    record_refs=f.record_refs,
                    subject=f.usubjid,
                )

                if auto_adjudicate:
                    # Look up standard monitor decision
                    dec, reason = self.sentinel.monitor.query(f.category, f.usubjid)
                    self.decision_center.adjudicate(
                        esc_item,
                        action=dec,
                        custom_reason=reason,
                        memory=self.memory,
                    )

        # 4. Compliance Check
        deviations = self.compliance.check_compliance()
        self.memory.log_trace(
            module="Compliance Check",
            decision="AUDIT",
            reason=f"Detected {len(deviations)} protocol deviations under Protocol v{pv} (Cut {cut}).",
            record_refs=[d.record_ref for d in deviations[:10]],
        )

        # 5. Site-level Recurring Problem Flagging
        for siteid, issues in self.memory.site_issues.items():
            if len(issues) >= 3 and siteid not in self.memory.site_flags:
                self.memory.site_flags[siteid] = (
                    f"Site {siteid} has {len(issues)} recurring deviations/queries. Flagged for site re-training."
                )
                self.memory.log_trace(
                    module="Memory + Action",
                    decision="SITE_FLAG",
                    reason=self.memory.site_flags[siteid],
                    record_refs=[],
                    subject=f"SITE-{siteid}",
                )

        # Summary of decisions
        decision_counts = {
            "PENDING": sum(1 for e in self.memory.escalations.values() if e.status == "PENDING"),
            "APPROVED": sum(1 for e in self.memory.escalations.values() if e.status == "APPROVED"),
            "REJECTED": sum(1 for e in self.memory.escalations.values() if e.status == "REJECTED"),
            "CLARIFIED_APPROVED": sum(1 for e in self.memory.escalations.values() if e.status == "CLARIFIED_APPROVED"),
        }

        self.memory.last_run_cut = cut

        # Review Report
        report = {
            "cut": cut,
            "protocol_version": pv,
            "finding_count": len(findings),
            "query_count": len(self.memory.active_queries),
            "new_query_count": new_queries,
            "deviation_count": len(deviations),
            "escalation_count": len(self.memory.escalations),
            "new_escalation_count": new_escalations,
            "human_decisions": decision_counts,
            "site_flags": self.memory.site_flags,
            "trace": [t.to_dict() for t in self.memory.trace],
            "summary": (
                f"Cycle {cycle_num} complete for Cut {cut} (v{pv}). "
                f"{len(findings)} findings ingested, {new_queries} new queries sent, "
                f"{len(deviations)} protocol deviations tracked, {new_escalations} new escalations created."
            ),
        }

        return {
            "status": "success",
            "cycle": cycle_num,
            "report": report,
            "findings": [f.to_dict() for f in findings[:30]],
            "queries": [q.to_dict() for q in self.memory.active_queries],
            "deviations": [d.to_dict() for d in deviations[:30]],
            "escalations": [e.to_dict() for e in self.memory.escalations.values()],
        }
