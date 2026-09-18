"""
test_engine.py
==============
Tests for question_engine.py using REAL records from the supplied dataset.
All expected values were verified against the CSV files.
"""
import json
from data_loader import StudyData
from study_index import build_study_index
from question_engine import QuestionEngine

# ── Setup ─────────────────────────────────────────────────────────────────
print("Loading and indexing study (cut=12)...")
data = StudyData(cut=12)
idx  = build_study_index(data)
eng  = QuestionEngine(idx)
print(f"  {idx.subject_count()} subjects indexed\n")

PASS = 0
FAIL = 0

def check(label, condition, detail=""):
    global PASS, FAIL
    ok = bool(condition)
    PASS += ok; FAIL += (not ok)
    print(f"  {'[PASS]' if ok else '[FAIL]'} {label}" +
          (f"  ({detail})" if detail else ""))

def show(result: dict, indent=4) -> None:
    """Pretty-print an answer dict (truncate long record_refs)."""
    d = dict(result)
    refs = d.get("record_refs", [])
    if len(refs) > 3:
        d["record_refs"] = refs[:3] + [f"... +{len(refs)-3} more"]
    print(json.dumps(d, indent=indent, default=str))

SEP = "=" * 62

# ==========================================================================
print(SEP)
print("TEST A — Study-wide COUNT (total subjects)")
print(SEP)

r = eng.count("subjects")
show(r)
check("question_type = count",        r["question_type"] == "count")
check("answer = 241",                  r["answer"] == 241,          f"got {r['answer']}")
check("status = answered",             r["status"] == "answered")
check("record_refs is not empty",      len(r["record_refs"]) > 0)
check("record_refs are DM domain",     r["record_refs"][0]["domain"] == "DM")
check("record_refs have cite format",  "|" in r["record_refs"][0]["cite"])
COUNT_PASS = FAIL == 0
print()


# ==========================================================================
print(SEP)
print("TEST A2 — ARM breakdown COUNT (DRUG vs PLACEBO)")
print(SEP)

r_drug    = eng.count("drug")
r_placebo = eng.count("placebo")
show(r_drug)
show(r_placebo)

f_before = FAIL
check("DRUG arm count = 128",         r_drug["answer"]    == 128, f"got {r_drug['answer']}")
check("PLACEBO arm count = 113",      r_placebo["answer"] == 113, f"got {r_placebo['answer']}")
check("DRUG + PLACEBO = 241",
      r_drug["answer"] + r_placebo["answer"] == 241,
      f"{r_drug['answer']} + {r_placebo['answer']}")
print()


# ==========================================================================
print(SEP)
print("TEST B — Subject-specific COUNT (LB records for 042-S07-011)")
print(SEP)

r = eng.count("lb", subject="042-S07-011")
show(r)
f_before = FAIL
check("question_type = count",   r["question_type"] == "count")
check("answer = 60",             r["answer"] == 60,   f"got {r['answer']}")
check("status = answered",       r["status"] == "answered")
check("60 record_refs",          len(r["record_refs"]) == 60, f"got {len(r['record_refs'])}")
check("refs domain = LB",        all(rf["domain"] == "LB" for rf in r["record_refs"]))
check("refs have seq",           all("seq" in rf for rf in r["record_refs"]))

# Count with filter — ALT results only (10 visits × 1 test = 10)
r2 = eng.count("lb", subject="042-S07-011", filters={"LBTESTCD": "ALT"})
show(r2)
check("filtered count ALT = 10",  r2["answer"] == 10, f"got {r2['answer']}")
check("all refs testcd=ALT",
      all(rf["testcd"] == "ALT" for rf in r2["record_refs"]))
COUNT_PASS = COUNT_PASS and (FAIL == f_before)
print()


# ==========================================================================
print(SEP)
print("TEST B2 — Subject-specific COUNT: subject with no AEs")
print(SEP)

# 042-S07-011 has 0 AEs (confirmed from Patient 360)
r = eng.count("ae", subject="042-S07-011")
show(r)
check("answer = 0",          r["answer"] == 0,        f"got {r['answer']}")
check("status = answered",   r["status"] == "answered")
check("empty record_refs",   len(r["record_refs"]) == 0)
print()


# ==========================================================================
print(SEP)
print("TEST C — LOOKUP: exact lab value (real record)")
print(SEP)

# Verified from CSV: 042-S07-011 LBSEQ=1 ALT SCREENING = 0.61 ukat/L
r = eng.lab_result("042-S07-011", "ALT", visit="SCREENING")
show(r)
f_before = FAIL
check("question_type = lookup",       r["question_type"] == "lookup")
check("status = answered",            r["status"] == "answered")
check("answer = '0.61'",              str(r["answer"]).strip() == "0.61",
      f"got {r['answer']!r}")
check("1 record_ref",                 len(r["record_refs"]) == 1)
check("ref cite = LB|042-S07-011|1",  r["record_refs"][0]["cite"] == "LB|042-S07-011|1")
check("ref testcd = ALT",             r["record_refs"][0]["testcd"] == "ALT")
check("ref visit = SCREENING",        r["record_refs"][0]["visit"] == "SCREENING")
check("ref unit = ukat/L",            r["record_refs"][0]["unit"] == "ukat/L")

# Lookup dose at BASELINE for PLACEBO subject (expect "0")
r2 = eng.dose_at_visit("042-S07-011", "BASELINE")
show(r2)
check("PLACEBO BASELINE dose = '0'",  str(r2["answer"]) == "0",  f"got {r2['answer']!r}")
check("ref cite = EX|042-S07-011|1",  r2["record_refs"][0]["cite"] == "EX|042-S07-011|1")

# Lookup QTcF at WEEK8 (confirmed: 469.0 msec)
r3 = eng.qtcf_at_visit("042-S07-011", "WEEK8")
show(r3)
check("WEEK8 QTcF = '469.0'",         str(r3["answer"]) == "469.0", f"got {r3['answer']!r}")

# Lookup disposition
r4 = eng.disposition("042-S07-011")
show(r4)
check("disposition = COMPLETED",      r4["answer"] == "COMPLETED", f"got {r4['answer']!r}")

LOOKUP_PASS = (FAIL == f_before)
print()


# ==========================================================================
print(SEP)
print("TEST C2 — LOOKUP: multiple matching rows returned")
print(SEP)

# ALT at all visits (no VISIT filter) — should return 10 rows
r = eng.lookup("LB", "042-S07-011", filters={"LBTESTCD": "ALT"}, field="LBORRES")
show(r)
check("answer is a list (10 ALT results)",  isinstance(r["answer"], list), f"type={type(r['answer'])}")
check("10 ALT results",                     len(r["answer"]) == 10, f"got {len(r['answer'])}")
check("10 record_refs",                     len(r["record_refs"]) == 10)
print()


# ==========================================================================
print(SEP)
print("TEST D — NOT_FOUND: lookup for nonexistent value")
print(SEP)

f_before = FAIL

# 1. Nonexistent subject
r = eng.lookup("LB", "042-S99-FAKE", filters={"LBTESTCD": "ALT"})
show(r)
check("fake subject -> not_found",     r["status"] == "not_found")
check("answer is None",                r["answer"] is None)
check("record_refs is empty",          len(r["record_refs"]) == 0)

# 2. Real subject, nonexistent visit label
r2 = eng.lookup("LB", "042-S07-011", filters={"VISIT": "NONEXISTENTVISIT"})
show(r2)
check("bad visit -> not_found",        r2["status"] == "not_found")

# 3. Real subject, real domain, but test they have zero of that
# 042-S07-011 has 0 AEs — ae_terms lookup should not_found
r3 = eng.ae_terms("042-S07-011")
show(r3)
check("subject with 0 AEs -> not_found", r3["status"] == "not_found")

# 4. Real subject, bad field name
r4 = eng.lookup("LB", "042-S07-011",
                 filters={"LBTESTCD": "ALT", "VISIT": "SCREENING"},
                 field="NONEXISTENTFIELD")
show(r4)
check("bad field name -> not_found",   r4["status"] == "not_found")

NOT_FOUND_PASS = (FAIL == f_before)
print()


# ==========================================================================
print(SEP)
print("TEST E — EVIDENCE: RecordRefs in answers are all verified")
print(SEP)

from evidence import record_ref_exists, RecordRef

f_before = FAIL

# All refs returned by the engine must pass record_ref_exists
test_results = [
    eng.count("lb", subject="042-S07-011"),
    eng.lab_result("042-S07-011", "ALT", visit="SCREENING"),
    eng.dose_at_visit("042-S07-011", "BASELINE"),
    eng.qtcf_at_visit("042-S07-011", "WEEK8"),
    eng.count("subjects"),
]

for res in test_results:
    for ref_dict in res.get("record_refs", []):
        if not isinstance(ref_dict, dict):
            continue   # skip the "... +N more" string
        ref = RecordRef(
            domain=ref_dict["domain"],
            usubjid=ref_dict["usubjid"],
            seq=ref_dict["seq"],
        )
        exists = record_ref_exists(ref, idx)
        if not exists:
            check(f"ref {ref.cite} exists in index", False)

check("all engine-returned RecordRefs verified in index",
      FAIL == f_before, f"failures above if any")

EVIDENCE_PASS = (FAIL == f_before)
print()


# ==========================================================================
print(SEP)
print("FINAL REPORT")
print(SEP)

total_pass = PASS
total_fail = FAIL

print(f"\n  {'Test':<30}  Result")
print(f"  {'-'*40}")
print(f"  {'COUNT (study-wide)':<30}  {'PASS' if COUNT_PASS else 'FAIL'}")
print(f"  {'COUNT (subject-specific)':<30}  {'PASS' if COUNT_PASS else 'FAIL'}")
print(f"  {'LOOKUP (exact value)':<30}  {'PASS' if LOOKUP_PASS else 'FAIL'}")
print(f"  {'NOT_FOUND (rejected)':<30}  {'PASS' if NOT_FOUND_PASS else 'FAIL'}")
print(f"  {'EVIDENCE (refs verified)':<30}  {'PASS' if EVIDENCE_PASS else 'FAIL'}")
print()
print(f"  Individual checks: {total_pass} passed, {total_fail} failed")
print()
if total_fail == 0:
    print("  ALL TESTS PASSED — question engine ready")
else:
    print("  SOME TESTS FAILED — see output above")
print(SEP)
