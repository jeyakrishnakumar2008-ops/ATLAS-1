"""
test_finding.py
===============
Runs the FINDING question type against the actual dataset.
Prints full Hy's law analysis showing:
  - subject, site, arm
  - ALT/AST evidence (raw value, unit, converted value, xULN)
  - BILI evidence (value, xULN)
  - reference ranges used
  - unit conversion arithmetic
  - date window calculation
  - final deterministic result
  - RecordRefs

Also runs SAE miscoding and dosing error checks.
"""
import json
from data_loader import StudyData
from study_index import build_study_index
from question_engine import QuestionEngine
import rules

# ── Setup ─────────────────────────────────────────────────────────────────
print("Loading and indexing study (cut=12)...")
data = StudyData(cut=12)
idx  = build_study_index(data)
eng  = QuestionEngine(idx)
print(f"  {idx.subject_count()} subjects, protocol v{idx.protocol_version_at(12)}\n")

SEP = "=" * 66
PASS = 0
FAIL = 0

def check(label, condition, detail=""):
    global PASS, FAIL
    ok = bool(condition)
    PASS += ok; FAIL += (not ok)
    print(f"  {'[PASS]' if ok else '[FAIL]'} {label}" +
          (f"  ({detail})" if detail else ""))


# ==========================================================================
print(SEP)
print("TEST 1 — Hy's law: full deterministic run across all 241 subjects")
print(SEP)
print()
print("  Reference ranges:")
for test, site in [("ALT","S01"),("AST","S01"),("BILI","S01"),("ALT","S07"),("AST","S07")]:
    rr = idx.ref_range(test, site)
    print(f"    {test} @ {'CENTRAL' if site!='S07' else 'S07  '}  "
          f"ULN={rr['HIGH']} {rr['UNIT']} (LAB={rr['LAB']})")
print()
print(f"  Thresholds applied:")
print(f"    ALT/AST spike  > {rules.HYS_ALT_AST_ULN_MULTIPLIER}× ULN")
print(f"    BILI elevation > {rules.HYS_BILI_ULN_MULTIPLIER}× ULN")
print(f"    Time window      within {rules.HYS_WINDOW_DAYS} days")
print(f"    Unit conversion  S07 ukat/L × {rules.UKAT_TO_UL} = U/L")
print()

result = eng.finding("hys_law")

print(f"  Status    : {result['status']}")
print(f"  Note      : {result['note']}")
print(f"  Candidates: {len(result['answer']) if result['answer'] else 0}")
print()

check("finding question_type = 'finding'",  result["question_type"] == "finding")
check("status = answered or not_found",      result["status"] in ("answered","not_found"))

if result["answer"]:
    for i, finding in enumerate(result["answer"]):
        uid   = finding["usubjid"]
        dm    = idx.get_subject(uid)
        print(f"  {'-'*62}")
        print(f"  CANDIDATE {i+1}: {uid}")
        print(f"    Site: {dm.get('SITEID')}  ARM: {dm.get('ARM')}  AGE: {dm.get('AGE')}")
        print(f"    Severity  : {finding['severity']}")
        print(f"    Category  : {finding['category']}")
        print(f"    Description:")
        # Wrap at 70 chars for readability
        desc = finding["description"]
        parts = desc.split("  |  ")
        for part in parts:
            print(f"      {part.strip()}")
        print(f"    Citations : {finding['citations']}")
        print(f"    Evidence  :")
        for ev in finding["evidence"]:
            print(f"      Claim  : {ev['claim']}")
            for ref in ev["record_refs"]:
                print(f"      RecordRef: {ref['cite']}  "
                      f"test={ref.get('testcd','-')}  "
                      f"val={ref.get('value','-')}  "
                      f"unit={ref.get('unit','-')}  "
                      f"visit={ref.get('visit','-')}  "
                      f"date={ref.get('date','-')}")
        print()

    check("at least 1 Hy's law candidate found",  len(result["answer"]) >= 1,
          f"found {len(result['answer'])}")
    check("all finding citations non-empty",
          all(len(f["citations"]) >= 2 for f in result["answer"]))
    check("all findings have 2 evidence objects",
          all(len(f["evidence"]) == 2 for f in result["answer"]))

    # Verify all RecordRefs in the result exist in the index
    from evidence import RecordRef, record_ref_exists
    all_exist = True
    for ref_dict in result["record_refs"]:
        ref = RecordRef(domain=ref_dict["domain"],
                        usubjid=ref_dict["usubjid"],
                        seq=ref_dict["seq"])
        if not record_ref_exists(ref, idx):
            all_exist = False
            print(f"  [FAIL] RecordRef {ref.cite} does not exist in index!")
    check("all Hy's law RecordRefs exist in index", all_exist)
else:
    print("  No Hy's law candidates found in this dataset.")
    check("Hy's law ran without error", result["status"] != "error")

print()


# ==========================================================================
print(SEP)
print("TEST 2 — Hy's law: subject-specific filter")
print(SEP)

# If we found candidates, pick the first one and verify single-subject filter works
if result["answer"]:
    target_uid = result["answer"][0]["usubjid"]
    r_single = eng.finding("hys_law", subject=target_uid)
    print(f"  Filtered to subject: {target_uid}")
    print(f"  Status: {r_single['status']}  Candidates: {len(r_single['answer'])}")
    check("single-subject filter returns 1 finding",
          len(r_single["answer"]) == 1, f"got {len(r_single['answer'])}")

    # Subject NOT a candidate should return not_found
    # Find a subject not in the findings list
    candidate_uids = {f["usubjid"] for f in result["answer"]}
    clean_uid = next(u for u in idx.all_subject_ids() if u not in candidate_uids)
    r_clean = eng.finding("hys_law", subject=clean_uid)
    print(f"  Clean subject {clean_uid}: status={r_clean['status']}")
    check("clean subject returns not_found",  r_clean["status"] == "not_found")
else:
    print("  (Skipped: no candidates to filter)")
print()


# ==========================================================================
print(SEP)
print("TEST 3 — SAE miscoding check")
print(SEP)

r_sae = eng.finding("sae_miscoding")
print(f"  Status     : {r_sae['status']}")
print(f"  Subjects   : {len(r_sae['answer']) if r_sae['answer'] else 0}")
if r_sae["answer"]:
    for f in r_sae["answer"]:
        print(f"  {f['usubjid']}:  {f['description']}")
        print(f"    citations={f['citations']}")

check("SAE miscoding ran without error", r_sae["status"] != "error")
check("SAE miscoding found exactly 1 record",
      r_sae["answer"] is not None and len(r_sae["answer"]) == 1,
      f"found {len(r_sae['answer']) if r_sae['answer'] else 0}")
print()


# ==========================================================================
print(SEP)
print("TEST 4 — Dosing error check")
print(SEP)

r_dose = eng.finding("dosing_error")
print(f"  Status  : {r_dose['status']}")
print(f"  Findings: {len(r_dose['answer']) if r_dose['answer'] else 0}")
if r_dose["answer"]:
    for f in r_dose["answer"]:
        print(f"  {f['usubjid']}: {f['description']}")
        print(f"    citations={f['citations']}")

check("Dosing error ran without error",   r_dose["status"] != "error")
# 6 subjects had wrong doses; the error spans WEEK2/WEEK4/WEEK8 (3 visits each) = 18 rows
dosing_count = len(r_dose["answer"]) if r_dose["answer"] else 0
check("Dosing errors found (expect 18 rows across 6 subjects)",
      dosing_count == 18,
      f"found {dosing_count}")
print()


# ==========================================================================
print(SEP)
print("TEST 5 — Unknown rule returns error")
print(SEP)

r_bad = eng.finding("no_such_rule")
print(f"  Status: {r_bad['status']}")
print(f"  Note  : {r_bad['note']}")
check("unknown rule returns error status", r_bad["status"] == "error")
print()


# ==========================================================================
print(SEP)
print("FINAL REPORT")
print(SEP)

print(f"\n  {'Test':<35}  Result")
print(f"  {'-'*45}")
tests = [
    ("Hy's law (study-wide)", PASS >= 1),
    ("Hy's law (subject filter)", True),   # tracked in check() calls above
    ("SAE miscoding", True),
    ("Dosing error", True),
    ("Unknown rule -> error", True),
    ("All RecordRefs verified", True),
]
print(f"\n  Individual checks: {PASS} passed, {FAIL} failed")
if FAIL == 0:
    print("\n  ALL TESTS PASSED — FINDING engine ready")
else:
    print("\n  SOME TESTS FAILED — see output above")
print(SEP)
