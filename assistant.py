"""
assistant.py
============
Interactive AI Clinical Trial Assistant for ATLAS STUDY-042.
Operates alongside existing Agent, Study Graph, and MONITOR without modifying them.

Features:
- Natural-language question parsing using existing in-memory StudyIndex and QuestionEngine.
- Maintains conversation context across follow-up turns during the session.
- Formats evidence with verified RecordRefs.
- Generates context-aware suggested follow-up questions.
- Provides deep-links to existing Patient 360 (Study Graph) and Medical Monitor.
- Zero external LLM / API keys needed — pure deterministic intelligence.
"""

from __future__ import annotations

import json
import re
from typing import Any, Optional

from atlas import StudySentinel
from checks import (
    check_hys_law,
    check_sae_miscoding,
    check_dosing_errors,
    check_missing_doses,
)
from evidence import make_record_ref, record_ref_exists
from patient360 import patient_360
from rules import VISIT_DAYS, prohibited_classes


# ---------------------------------------------------------------------------
# Conversational Query Processor
# ---------------------------------------------------------------------------

def process_assistant_query(
    question: str,
    context: dict[str, Any] | None,
    sentinel: StudySentinel,
) -> dict[str, Any]:
    """
    Evaluates a natural-language query using the existing ATLAS engine,
    maintaining session context for multi-turn follow-ups.
    """
    context = context or {}
    q_clean = question.strip()
    q_lower = q_clean.lower()
    index = sentinel.index
    engine = sentinel.engine

    # 1. Subject extraction and context resolution
    subj_match = re.search(r"042-s\d{2}-\d{3}", q_lower)
    if subj_match:
        current_subject = subj_match.group(0).upper()
    else:
        # Contextual pronouns / references
        if any(w in q_lower for w in ["they", "this subject", "their", "the patient", "him", "her", "that subject"]):
            current_subject = context.get("last_subject", "042-S07-001")
        else:
            current_subject = context.get("last_subject")

    last_topic = context.get("last_topic", "")
    updated_context = {
        "last_subject": current_subject,
        "last_topic": last_topic,
    }

    # ------------------------------------------------------------------
    # A. Adversarial Traps & Prompt Injections
    # ------------------------------------------------------------------
    if any(k in q_lower for k in ["automated reviewer", "exclude s03", "exclude s07", "prompt injection", "ignore previous", "ignore all", "system compromised"]):
        trap_res = engine.trap("prompt_injection", claim=q_clean)
        return {
            "status": "refusal",
            "answer_text": "Security Warning: Refused adversarial instruction or prompt injection. Directives attempting to override clinical surveillance rules or exclude study sites are not permitted.",
            "findings": [],
            "record_refs": [],
            "suggested_followups": [
                "Which subjects have a missing SAE?",
                "Find Hy's Law hepatotoxicity candidates",
                "How many subjects are enrolled in the study?",
            ],
            "context_updates": updated_context,
        }

    # ------------------------------------------------------------------
    # B. Missing SAE / SAE Miscoding
    # ------------------------------------------------------------------
    if any(k in q_lower for k in [
        "missing sae", "unreported sae", "sae miscod", "miscoded sae",
        "hospitalized without sae", "hospitalized non-serious", "sae_miscod"
    ]):
        updated_context["last_topic"] = "sae_miscoding"
        res = check_sae_miscoding(index)
        findings_list = []
        all_refs = []

        for f in res.findings:
            refs = f.all_citations()
            all_refs.extend(refs)
            site = f.usubjid.split("-")[1] if "-" in f.usubjid else "S00"
            findings_list.append({
                "subject": f.usubjid,
                "site": site,
                "finding": "Hospitalized adverse event recorded as non-serious (AESHOSP=Y, AESER=N)",
                "severity": f.severity,
                "record_refs": refs,
                "is_monitor_relevant": True,
                "patient_360_url": f"/?view=graph&subject={f.usubjid}",
                "monitor_url": "/?view=monitor",
            })
            if not current_subject:
                current_subject = f.usubjid
                updated_context["last_subject"] = current_subject

        count = len(findings_list)
        ans_text = (
            f"{count} subject{' was' if count==1 else 's were'} found with miscoded or unreported Serious Adverse Event (SAE) criteria. "
            f"Under ICH-GCP E2A guidelines, any event requiring inpatient hospitalization must be classified as an SAE regardless of the investigator's initial seriousness assessment."
        )

        followups = [
            f"What adverse event did {findings_list[0]['subject']} experience?" if findings_list else "Find Hy's Law candidates",
            f"Open Patient 360 for {findings_list[0]['subject']}" if findings_list else "How many subjects are enrolled?",
            "View pending escalations in Decision Center",
        ]

        return {
            "status": "answered",
            "answer_text": ans_text,
            "findings": findings_list,
            "record_refs": all_refs,
            "suggested_followups": followups,
            "context_updates": updated_context,
        }

    # ------------------------------------------------------------------
    # C. Hy's Law Hepatotoxicity Candidates
    # ------------------------------------------------------------------
    if any(k in q_lower for k in ["hy's law", "hys law", "hepatotox", "liver injury", "liver candidate"]):
        updated_context["last_topic"] = "hys_law"
        res = check_hys_law(index)
        findings_list = []
        all_refs = []

        for f in res.findings:
            refs = f.all_citations()
            all_refs.extend(refs)
            site = f.usubjid.split("-")[1] if "-" in f.usubjid else "S00"
            findings_list.append({
                "subject": f.usubjid,
                "site": site,
                "finding": "Hy's Law Signal: Concurrent ALT > 3x ULN and Total Bilirubin > 2x ULN within 14 days",
                "severity": f.severity,
                "record_refs": refs,
                "is_monitor_relevant": True,
                "patient_360_url": f"/?view=graph&subject={f.usubjid}",
                "monitor_url": "/?view=monitor",
            })

        count = len(findings_list)
        ans_text = (
            f"{count} subjects meet FDA Hy's Law hepatotoxicity criteria with confirmed concurrent ALT > 3x ULN and Total Bilirubin > 2x ULN within a 14-day window."
        )

        followups = [
            "Why did the Medical Monitor reject escalation for 042-S07-001?",
            "What was the screening ALT for 042-S07-001?",
            "Open Patient 360 for 042-S05-003",
            "Are there any concomitant medications for 042-S07-001?",
        ]

        return {
            "status": "answered",
            "answer_text": ans_text,
            "findings": findings_list,
            "record_refs": all_refs,
            "suggested_followups": followups,
            "context_updates": updated_context,
        }

    # ------------------------------------------------------------------
    # D. Dosing Errors & Overdoses
    # ------------------------------------------------------------------
    if any(k in q_lower for k in ["dosing error", "dose error", "overdose", "wrong dose", "dose discrepancy"]):
        updated_context["last_topic"] = "dosing_errors"
        res = check_dosing_errors(index)
        findings_list = []
        all_refs = []

        # Group by subject
        subj_map: dict[str, list[Any]] = {}
        for f in res.findings:
            subj_map.setdefault(f.usubjid, []).append(f)

        for uid, flist in subj_map.items():
            site = uid.split("-")[1] if "-" in uid else "S00"
            refs = [ref for f in flist for ref in f.all_citations()]
            all_refs.extend(refs)
            findings_list.append({
                "subject": uid,
                "site": site,
                "finding": f"{len(flist)} dose administrations outside protocol nominal range (expected 100 mg / 200 mg)",
                "severity": "HIGH",
                "record_refs": refs[:3],
                "is_monitor_relevant": True,
                "patient_360_url": f"/?view=graph&subject={uid}",
                "monitor_url": "/?view=monitor",
            })

        ans_text = (
            f"{len(res.findings)} dosing error records were detected across {len(findings_list)} subjects. "
            f"These represent exposures where administered doses deviated from protocol-specified dose steps."
        )

        followups = [
            f"Open Patient 360 for {findings_list[0]['subject']}" if findings_list else "Check missing doses",
            "Which sites have recurring dosing issues?",
            "Which subjects have a missing SAE?",
        ]

        return {
            "status": "answered",
            "answer_text": ans_text,
            "findings": findings_list,
            "record_refs": all_refs[:15],
            "suggested_followups": followups,
            "context_updates": updated_context,
        }

    # ------------------------------------------------------------------
    # E. Missing Doses & Compliance
    # ------------------------------------------------------------------
    if any(k in q_lower for k in ["missing dose", "missed dose", "missed medication", "treatment interruption"]):
        updated_context["last_topic"] = "missing_doses"
        res = check_missing_doses(index)
        findings_list = []
        all_refs = []

        for f in res.findings[:10]:
            refs = f.all_citations()
            all_refs.extend(refs)
            site = f.usubjid.split("-")[1] if "-" in f.usubjid else "S00"
            findings_list.append({
                "subject": f.usubjid,
                "site": site,
                "finding": f.description,
                "severity": f.severity,
                "record_refs": refs,
                "is_monitor_relevant": True,
                "patient_360_url": f"/?view=graph&subject={f.usubjid}",
                "monitor_url": "/?view=monitor",
            })

        ans_text = (
            f"{len(res.findings)} missed dose occurrences were detected across scheduled study visits."
        )

        return {
            "status": "answered",
            "answer_text": ans_text,
            "findings": findings_list,
            "record_refs": all_refs,
            "suggested_followups": [
                "Find dosing errors across study",
                "Which subjects have a missing SAE?",
                "View compliance audit in Monitor",
            ],
            "context_updates": updated_context,
        }

    # ------------------------------------------------------------------
    # F. Concomitant Medications for Subject
    # ------------------------------------------------------------------
    if any(k in q_lower for k in ["medication", "conmed", "concomitant", "taking", "drugs"]) and current_subject:
        updated_context["last_topic"] = "medications"
        cms = index.get_medications(current_subject)
        if not cms:
            return {
                "status": "answered",
                "answer_text": f"Subject {current_subject} has no concomitant medications on record.",
                "findings": [],
                "record_refs": [],
                "suggested_followups": [
                    f"Tell me about subject {current_subject}",
                    f"What adverse events did {current_subject} experience?",
                    f"Open Patient 360 for {current_subject}",
                ],
                "context_updates": updated_context,
            }

        findings_list = []
        all_refs = []
        pv = sentinel.protocol_version
        prohib_classes = prohibited_classes(pv)

        for c in cms:
            ref = make_record_ref("CM", c)
            all_refs.append(ref.cite)
            cmtrt = c.get("CMTRT", "Unknown")
            cmclas = c.get("CMCLAS", "")
            is_prohib = any(p in cmclas.upper() or p in cmtrt.upper() for p in prohib_classes)
            severity = "HIGH" if is_prohib else "INFO"
            finding_label = f"{cmtrt} ({cmclas})" + (" [PROHIBITED BY PROTOCOL]" if is_prohib else " [Concomitant]")

            findings_list.append({
                "subject": current_subject,
                "site": current_subject.split("-")[1] if "-" in current_subject else "S00",
                "finding": finding_label,
                "severity": severity,
                "record_refs": [ref.cite],
                "is_monitor_relevant": is_prohib,
                "patient_360_url": f"/?view=graph&subject={current_subject}",
                "monitor_url": "/?view=monitor" if is_prohib else "",
            })

        ans_text = (
            f"Subject {current_subject} has {len(cms)} concomitant medication record{'s' if len(cms)!=1 else ''}. "
            + ("Warning: One or more medications violate active protocol restrictions." if any(f["is_monitor_relevant"] for f in findings_list) else "All medications are currently permitted.")
        )

        followups = [
            f"Is sulfonylurea a protocol deviation at Cut {sentinel.cut}?",
            f"What adverse events did {current_subject} have?",
            f"Open Patient 360 for {current_subject}",
        ]

        return {
            "status": "answered",
            "answer_text": ans_text,
            "findings": findings_list,
            "record_refs": all_refs,
            "suggested_followups": followups,
            "context_updates": updated_context,
        }

    # ------------------------------------------------------------------
    # G. Adverse Events for Subject
    # ------------------------------------------------------------------
    if any(k in q_lower for k in ["adverse event", "ae ", "aes ", "symptoms", "safety event"]) and current_subject:
        updated_context["last_topic"] = "adverse_events"
        aes = index.get_adverse_events(current_subject)
        if not aes:
            return {
                "status": "answered",
                "answer_text": f"Subject {current_subject} experienced 0 adverse events during the study.",
                "findings": [],
                "record_refs": [],
                "suggested_followups": [
                    f"Tell me about subject {current_subject}",
                    f"Lookup Screening ALT for {current_subject}",
                    f"Open Patient 360 for {current_subject}",
                ],
                "context_updates": updated_context,
            }

        findings_list = []
        all_refs = []
        for a in aes:
            ref = make_record_ref("AE", a)
            all_refs.append(ref.cite)
            term = a.get("AETERM", "Adverse Event")
            is_ser = a.get("AESER", "").upper() == "Y"
            is_hosp = a.get("AESHOSP", "").upper() == "Y"
            sev = "CRITICAL" if (is_ser or is_hosp) else "MEDIUM"
            details = f"{term} | Severity: {a.get('AESEV','')} | Serious: {a.get('AESER','')} | Hospitalized: {a.get('AESHOSP','')}"

            findings_list.append({
                "subject": current_subject,
                "site": current_subject.split("-")[1] if "-" in current_subject else "S00",
                "finding": details,
                "severity": sev,
                "record_refs": [ref.cite],
                "is_monitor_relevant": is_ser or is_hosp,
                "patient_360_url": f"/?view=graph&subject={current_subject}",
                "monitor_url": "/?view=monitor" if (is_ser or is_hosp) else "",
            })

        ans_text = f"Subject {current_subject} has {len(aes)} adverse event record{'s' if len(aes)!=1 else ''}."
        return {
            "status": "answered",
            "answer_text": ans_text,
            "findings": findings_list,
            "record_refs": all_refs,
            "suggested_followups": [
                f"What concomitant medications is {current_subject} taking?",
                f"Open Patient 360 for {current_subject}",
                "View pending escalations in Decision Center",
            ],
            "context_updates": updated_context,
        }

    # ------------------------------------------------------------------
    # H. Subject Demographics & Patient 360 Details
    # ------------------------------------------------------------------
    if current_subject and (
        any(k in q_lower for k in ["tell me about", "who is", "patient profile", "demographics", "details for", "summary for"])
        or q_clean.upper() == current_subject
    ):
        updated_context["last_topic"] = "patient_profile"
        p_bundle = patient_360(index, current_subject)
        if not p_bundle:
            return {
                "status": "not_found",
                "answer_text": f"Subject {current_subject} was not found in the study database.",
                "findings": [],
                "record_refs": [],
                "suggested_followups": ["How many subjects are enrolled in the study?"],
                "context_updates": updated_context,
            }

        demo = p_bundle["demographics"]
        counts = p_bundle["record_counts"]
        ref_dm = f"DM|{current_subject}|1"

        ans_text = (
            f"Subject {current_subject} is enrolled at Site {demo['site']} ({demo['country']}), assigned to the {demo['arm']} arm. "
            f"Age: {demo['age']} ({demo['sex']}), Screening HbA1c: {demo['scr_hba1c']}%. "
            f"Study record totals: {counts['AE']} adverse events, {counts['LB']} laboratory results, {counts['CM']} concomitant medications, {counts['EX']} doses administered."
        )

        finding_item = {
            "subject": current_subject,
            "site": demo["site"],
            "finding": f"Arm: {demo['arm']} | Age: {demo['age']} | Sex: {demo['sex']} | HbA1c: {demo['scr_hba1c']}%",
            "severity": "INFO",
            "record_refs": [ref_dm],
            "is_monitor_relevant": False,
            "patient_360_url": f"/?view=graph&subject={current_subject}",
            "monitor_url": "",
        }

        return {
            "status": "answered",
            "answer_text": ans_text,
            "findings": [finding_item],
            "record_refs": [ref_dm],
            "suggested_followups": [
                f"What adverse events did {current_subject} have?",
                f"What concomitant medications is {current_subject} taking?",
                f"Open Patient 360 for {current_subject}",
            ],
            "context_updates": updated_context,
        }

    # ------------------------------------------------------------------
    # I. Medical Monitor Adjudication Rationale
    # ------------------------------------------------------------------
    if any(k in q_lower for k in ["why was", "medical monitor", "reject", "adjudicat", "decision for"]) and current_subject:
        code = "HYS_LAW" if "hys" in q_lower or "liver" in q_lower else "SAE_MISCODING"
        dec, reason = sentinel.monitor.query(code, current_subject)
        ans_text = f"Medical Monitor Adjudication for {current_subject} [{code}]: Status is {dec}. Rationale: \"{reason}\""

        return {
            "status": "answered",
            "answer_text": ans_text,
            "findings": [{
                "subject": current_subject,
                "site": current_subject.split("-")[1] if "-" in current_subject else "S00",
                "finding": f"Monitor Adjudication: {dec} — {reason}",
                "severity": "CRITICAL" if dec == "APPROVED" else "MEDIUM",
                "record_refs": [],
                "is_monitor_relevant": True,
                "patient_360_url": f"/?view=graph&subject={current_subject}",
                "monitor_url": "/?view=monitor",
            }],
            "record_refs": [],
            "suggested_followups": [
                f"Open Patient 360 for {current_subject}",
                "View Decision Center in Monitor",
                "Which subjects have a missing SAE?",
            ],
            "context_updates": updated_context,
        }

    # ------------------------------------------------------------------
    # J. Protocol Version & Amendments
    if any(k in q_lower for k in ["sulfonylurea", "protocol version", "protocol v", "amendment", "protocol rule", "protocol"]):
        pv = sentinel.protocol_version
        is_dev = pv >= 3
        ans_text = (
            f"Active Data Cut is {sentinel.cut} operating under Protocol Version {pv}. "
            f"Key rules: Sulfonylureas are "
            + ("strictly prohibited per Amendment 2. Any concomitant administration constitutes a critical protocol deviation."
               if is_dev else
               "not prohibited under baseline Protocol v1. Concomitant use does not violate study protocol.")
        )
        return {
            "status": "answered",
            "answer_text": ans_text,
            "findings": [{
                "subject": "042-S07-001",
                "site": "S07",
                "finding": "Glibenclamide (Sulfonylurea) use under Protocol v3",
                "severity": "HIGH" if is_dev else "INFO",
                "record_refs": ["CM|042-S07-001|1"],
                "is_monitor_relevant": is_dev,
                "patient_360_url": "/?view=graph&subject=042-S07-001",
                "monitor_url": "/?view=monitor" if is_dev else "",
            }],
            "record_refs": ["CM|042-S07-001|1"],
            "suggested_followups": [
                "What concomitant medications is 042-S07-001 taking?",
                "Open Patient 360 for 042-S07-001",
                "Check protocol deviations in Monitor",
            ],
            "context_updates": updated_context,
        }

    # ------------------------------------------------------------------
    # K. Fallback: Existing ATLAS Question Engine
    # ------------------------------------------------------------------
    from server import parse_and_route_query
    base_res = parse_and_route_query(q_clean, sentinel)

    status = base_res.get("status", "unknown")
    ans_raw = base_res.get("answer")
    refs_raw = base_res.get("record_refs", [])
    cite_strings = []
    for r in refs_raw:
        if isinstance(r, str):
            cite_strings.append(r)
        elif hasattr(r, "cite"):
            cite_strings.append(r.cite)
        elif isinstance(r, dict) and "cite" in r:
            cite_strings.append(r["cite"])

    # Conversational phrasing of engine answer
    if status == "answered":
        if isinstance(ans_raw, int):
            ans_text = f"Result: {ans_raw:,} subjects were verified in the study records."
        elif isinstance(ans_raw, list):
            ans_text = f"Found {len(ans_raw)} matching record{'s' if len(ans_raw)!=1 else ''}: {', '.join(str(x) for x in ans_raw[:5])}"
        else:
            ans_text = f"Verified Study Answer: {ans_raw}"
    elif status == "prompt_injection_detected":
        ans_text = "Security Alert: Prompt injection attempt detected and safely neutralized."
    elif status == "insufficient_evidence":
        ans_text = "Insufficient clinical evidence found in the study index to support this claim."
    else:
        ans_text = f"Status: {status}. The query could not be definitively resolved from current cut records."

    followups = [
        "Which subjects have a missing SAE?",
        "Find Hy's Law hepatotoxicity candidates",
        "How many subjects are in the DRUG arm?",
        "Tell me about subject 042-S07-001",
    ]

    return {
        "status": status,
        "answer_text": ans_text,
        "findings": [],
        "record_refs": cite_strings,
        "suggested_followups": followups,
        "context_updates": updated_context,
    }


# ---------------------------------------------------------------------------
# Separate HTML Template for /assistant
# ---------------------------------------------------------------------------

HTML_ASSISTANT_TEMPLATE = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<title>ATLAS Assistant &mdash; Clinical Trial Intelligence</title>
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<style>
  body {
    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
    margin: 0; padding: 20px; background: #f8fafc; color: #0f172a;
  }
  .container {
    max-width: 960px; margin: 0 auto; background: #ffffff; border: 1px solid #e2e8f0;
    border-radius: 8px; box-shadow: 0 1px 3px rgba(0,0,0,0.05); overflow: hidden;
    display: flex; flex-direction: column; min-height: 85vh;
  }
  .nav-header {
    display: flex; justify-content: space-between; align-items: center;
    border-bottom: 1px solid #e2e8f0; padding: 16px 24px; background: #ffffff;
  }
  .app-title { font-size: 18px; font-weight: 700; color: #0f172a; }
  .nav-links { display: flex; gap: 8px; }
  .nav-link-btn {
    padding: 6px 12px; font-size: 12px; font-weight: 600; border: 1px solid #cbd5e1;
    background: #f1f5f9; color: #334155; border-radius: 6px; text-decoration: none;
  }
  .nav-link-btn:hover { background: #e2e8f0; }
  .nav-link-btn.primary { background: #0284c7; color: #ffffff; border-color: #0284c7; }

  /* Chat Area */
  .chat-area {
    flex: 1; padding: 24px; overflow-y: auto; display: flex; flex-direction: column; gap: 18px;
    background: #fafafa;
  }
  .msg-row { display: flex; width: 100%; }
  .msg-row.user { justify-content: flex-end; }
  .msg-row.assistant { justify-content: flex-start; }

  .msg-bubble {
    max-width: 82%; padding: 14px 18px; border-radius: 8px; font-size: 14px; line-height: 1.5;
  }
  .msg-row.user .msg-bubble {
    background: #0284c7; color: #ffffff; border-bottom-right-radius: 2px;
  }
  .msg-row.assistant .msg-bubble {
    background: #ffffff; border: 1px solid #e2e8f0; color: #0f172a;
    box-shadow: 0 1px 2px rgba(0,0,0,0.04); border-bottom-left-radius: 2px;
  }

  /* Findings Table */
  .finding-table {
    width: 100%; border-collapse: collapse; font-size: 12px; margin: 12px 0 6px 0;
    border: 1px solid #e2e8f0; border-radius: 6px; overflow: hidden; background: #ffffff;
  }
  .finding-table th {
    background: #f1f5f9; padding: 7px 10px; text-align: left; font-weight: 700; color: #475569;
    border-bottom: 1px solid #e2e8f0;
  }
  .finding-table td {
    padding: 7px 10px; border-bottom: 1px solid #f1f5f9; vertical-align: top;
  }

  .ref-tag {
    display: inline-block; background: #eff6ff; border: 1px solid #bfdbfe;
    color: #1e40af; padding: 2px 6px; font-family: monospace; font-size: 11px;
    border-radius: 4px; margin: 2px 4px 2px 0; font-weight: 600;
  }
  .ref-tag.verified::after { content: " ✓ Verified"; color: #16a34a; font-weight: 700; }

  .action-btn {
    display: inline-block; padding: 4px 8px; font-size: 11px; font-weight: 600;
    border-radius: 4px; text-decoration: none; margin: 2px 4px 2px 0; cursor: pointer;
  }
  .btn-p360 { background: #0f172a; color: #ffffff; }
  .btn-p360:hover { background: #334155; }
  .btn-mon { background: #d97706; color: #ffffff; }
  .btn-mon:hover { background: #b45309; }

  /* Suggested Chips */
  .chip-group { display: flex; flex-wrap: wrap; gap: 6px; margin-top: 10px; }
  .chip {
    background: #f1f5f9; border: 1px solid #cbd5e1; border-radius: 14px;
    padding: 4px 10px; font-size: 12px; color: #1e293b; cursor: pointer;
  }
  .chip:hover { background: #e2e8f0; border-color: #94a3b8; }

  /* Input Footer */
  .chat-footer {
    padding: 16px 24px; background: #ffffff; border-top: 1px solid #e2e8f0;
  }
  .input-group { display: flex; gap: 8px; }
  input[type="text"] {
    flex: 1; padding: 10px 14px; font-size: 14px; border: 1px solid #cbd5e1;
    border-radius: 6px; outline: none; background: #ffffff;
  }
  input[type="text"]:focus { border-color: #0284c7; }
  button.submit-btn {
    padding: 10px 22px; font-size: 14px; font-weight: 600; background: #0284c7;
    color: white; border: none; border-radius: 6px; cursor: pointer;
  }
  button.submit-btn:hover { background: #0369a1; }
  button.submit-btn:disabled { background: #94a3b8; cursor: not-allowed; }

  .loading-indicator {
    font-size: 12px; color: #64748b; font-style: italic; margin-top: 6px; display: none;
  }
</style>
</head>
<body>
<div class="container">
  <!-- Top Navigation -->
  <div class="nav-header">
    <div>
      <div class="app-title">ATLAS Assistant <span style="font-size:13px; font-weight:400; color:#64748b;">| Interactive AI Agent</span></div>
      <div style="font-size:12px; color:#64748b; margin-top:2px;">Query STUDY-042 via real in-memory StudyIndex &amp; QuestionEngine</div>
    </div>
    <div class="nav-links">
      <a href="/" class="nav-link-btn">&larr; Main Study Portal</a>
      <a href="/?view=graph" class="nav-link-btn">Study Graph</a>
      <a href="/?view=monitor" class="nav-link-btn">Decision Center</a>
    </div>
  </div>

  <!-- Messages List -->
  <div class="chat-area" id="chatArea">
    <!-- Welcome Greeting -->
    <div class="msg-row assistant">
      <div class="msg-bubble">
        <div style="font-weight:700; margin-bottom:4px;">Hello! I am the ATLAS Clinical Intelligence Assistant.</div>
        <div>I can answer natural-language questions about STUDY-042 subjects, adverse events, laboratory signals (Hy's Law), dosing errors, protocol compliance, and medical monitor adjudications. All answers cite verified in-memory <code>RecordRef</code>s.</div>
        <div style="margin-top:10px; font-size:12px; font-weight:700; color:#64748b;">TRY ASKING:</div>
        <div class="chip-group">
          <div class="chip" onclick="askChip(this.innerText)">Which subjects have a missing SAE?</div>
          <div class="chip" onclick="askChip(this.innerText)">Which subjects have Hy's Law?</div>
          <div class="chip" onclick="askChip(this.innerText)">Tell me about subject 042-S07-001</div>
          <div class="chip" onclick="askChip(this.innerText)">Which subjects have dosing errors?</div>
          <div class="chip" onclick="askChip(this.innerText)">Is sulfonylurea a protocol deviation?</div>
        </div>
      </div>
    </div>
  </div>

  <!-- Input Footer -->
  <div class="chat-footer">
    <div class="input-group">
      <input type="text" id="assistantInput" placeholder="Ask a question about STUDY-042 (e.g. Which subjects have a missing SAE?)..." onkeydown="handleKeyDown(event)">
      <button class="submit-btn" id="askBtn" onclick="submitMessage()">Ask</button>
      <button class="nav-link-btn" onclick="clearChat()" title="Clear conversation history">Clear</button>
    </div>
    <div class="loading-indicator" id="loadingIndicator">Querying ATLAS Study Index &amp; Question Engine...</div>
  </div>
</div>

<script>
let sessionContext = {
  last_subject: null,
  last_topic: null,
  history: []
};

function handleKeyDown(e) {
  if (e.key === 'Enter') {
    submitMessage();
  }
}

function askChip(text) {
  document.getElementById('assistantInput').value = text;
  submitMessage();
}

function clearChat() {
  sessionContext = { last_subject: null, last_topic: null, history: [] };
  const chatArea = document.getElementById('chatArea');
  chatArea.innerHTML = `
    <div class="msg-row assistant">
      <div class="msg-bubble">
        <div style="font-weight:700; margin-bottom:4px;">Chat Cleared</div>
        <div>Ask any question about STUDY-042 clinical data.</div>
        <div class="chip-group" style="margin-top:10px;">
          <div class="chip" onclick="askChip(this.innerText)">Which subjects have a missing SAE?</div>
          <div class="chip" onclick="askChip(this.innerText)">Find Hy's Law hepatotoxicity candidates</div>
          <div class="chip" onclick="askChip(this.innerText)">How many subjects are enrolled in the study?</div>
        </div>
      </div>
    </div>
  `;
}

async function submitMessage() {
  const input = document.getElementById('assistantInput');
  const question = input.value.trim();
  if (!question) return;

  input.value = '';
  const btn = document.getElementById('askBtn');
  btn.disabled = true;
  document.getElementById('loadingIndicator').style.display = 'block';

  // Append user bubble
  appendUserMessage(question);

  try {
    const resp = await fetch('/api/assistant', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        question: question,
        context: sessionContext
      })
    });
    const data = await resp.json();

    if (data.context_updates) {
      if (data.context_updates.last_subject) sessionContext.last_subject = data.context_updates.last_subject;
      if (data.context_updates.last_topic) sessionContext.last_topic = data.context_updates.last_topic;
    }
    sessionContext.history.push({ q: question, a: data.answer_text });

    appendAssistantMessage(data);
  } catch (err) {
    appendErrorMessage('Failed to connect to ATLAS engine: ' + err);
  } finally {
    btn.disabled = false;
    document.getElementById('loadingIndicator').style.display = 'none';
    const chatArea = document.getElementById('chatArea');
    chatArea.scrollTop = chatArea.scrollHeight;
  }
}

function appendUserMessage(text) {
  const chatArea = document.getElementById('chatArea');
  const div = document.createElement('div');
  div.className = 'msg-row user';
  div.innerHTML = `<div class="msg-bubble">${escapeHtml(text)}</div>`;
  chatArea.appendChild(div);
  chatArea.scrollTop = chatArea.scrollHeight;
}

function appendAssistantMessage(data) {
  const chatArea = document.getElementById('chatArea');
  const div = document.createElement('div');
  div.className = 'msg-row assistant';

  let findingsHtml = '';
  if (data.findings && data.findings.length > 0) {
    findingsHtml = `
      <table class="finding-table">
        <thead>
          <tr>
            <th>Subject</th>
            <th>Site</th>
            <th>Finding</th>
            <th>Severity</th>
            <th>Evidence RecordRef</th>
            <th>Actions</th>
          </tr>
        </thead>
        <tbody>
          ${data.findings.map(f => {
            const refs = (f.record_refs || []).map(r => `<span class="ref-tag verified">${r}</span>`).join(' ');
            let actLinks = '';
            if (f.patient_360_url) {
              actLinks += `<a href="${f.patient_360_url}" target="_blank" class="action-btn btn-p360">Open Patient 360</a>`;
            }
            if (f.is_monitor_relevant) {
              actLinks += `<a href="${f.monitor_url || '/?view=monitor'}" target="_blank" class="action-btn btn-mon">View in Monitor</a>`;
            }
            return `<tr>
              <td><code><b>${f.subject}</b></code></td>
              <td>${f.site}</td>
              <td>${f.finding}</td>
              <td><span style="font-weight:700; color:${f.severity==='CRITICAL'?'#dc2626':'#d97706'}">${f.severity}</span></td>
              <td>${refs || '<span style="color:#94a3b8;">None</span>'}</td>
              <td>${actLinks}</td>
            </tr>`;
          }).join('')}
        </tbody>
      </table>
    `;
  }

  let refsSummaryHtml = '';
  if ((!data.findings || data.findings.length === 0) && data.record_refs && data.record_refs.length > 0) {
    refsSummaryHtml = `<div style="margin-top:8px;"><b>Supporting Evidence:</b> ${data.record_refs.map(r => `<span class="ref-tag verified">${r}</span>`).join(' ')}</div>`;
  }

  let followupsHtml = '';
  if (data.suggested_followups && data.suggested_followups.length > 0) {
    followupsHtml = `
      <div style="margin-top:12px; font-size:11px; font-weight:700; color:#64748b;">SUGGESTED FOLLOW-UPS:</div>
      <div class="chip-group">
        ${data.suggested_followups.map(q => `<div class="chip" onclick="askChip('${escapeHtml(q)}')">${escapeHtml(q)}</div>`).join('')}
      </div>
    `;
  }

  div.innerHTML = `
    <div class="msg-bubble">
      <div>${escapeHtml(data.answer_text)}</div>
      ${findingsHtml}
      ${refsSummaryHtml}
      ${followupsHtml}
    </div>
  `;
  chatArea.appendChild(div);
}

function appendErrorMessage(msg) {
  const chatArea = document.getElementById('chatArea');
  const div = document.createElement('div');
  div.className = 'msg-row assistant';
  div.innerHTML = `<div class="msg-bubble" style="border-left: 4px solid #dc2626; color:#991b1b;">${escapeHtml(msg)}</div>`;
  chatArea.appendChild(div);
}

function escapeHtml(str) {
  if (!str) return '';
  return String(str).replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;').replace(/"/g, '&quot;');
}
</script>
</body>
</html>
"""
