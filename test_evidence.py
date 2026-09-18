"""
test_evidence.py
================
Tests for the evidence system: RecordRef, make_record_ref,
record_ref_exists, and Evidence.build().

All records used here are real rows from LB.csv and AE.csv.
No values are invented.
"""
import json
from data_loader import StudyData
from study_index import build_study_index
from evidence import (
    RecordRef,
    make_record_ref,
    record_ref_exists,
    assert_record_ref_exists,
    Evidence,
    Finding,
)

# ── Setup (load once) ─────────────────────────────────────────────────────
print("Loading study and building index...")
data = StudyData(cut=12)
idx  = build_study_index(data)
print(f"  {idx.subject_count()} subjects indexed\n")

PASS = 0
FAIL = 0

def check(label: str, condition: bool, detail: str = "") -> None:
    global PASS, FAIL
    status = "PASS" if condition else "FAIL"
    if condition:
        PASS += 1
    else:
        FAIL += 1
    suffix = f"  ({detail})" if detail else ""
    print(f"  [{status}] {label}{suffix}")


# ==========================================================================
print("=" * 62)
print("TEST 1 — make_record_ref from a real LB row")
print("=" * 62)

# Pull the first SCREENING ALT for subject 042-S07-011 (site S07, ukat/L)
uid    = "042-S07-011"
lb_rows = idx.get_labs(uid)
# LBSEQ=1 is ALT at SCREENING for this subject (confirmed from previous tests)
real_lb = next((r for r in lb_rows if r["LBSEQ"] == "1"), None)

assert real_lb is not None, "Precondition: LBSEQ=1 must exist for 042-S07-011"
print(f"  Real row: {real_lb}")
print()

ref = make_record_ref("LB", real_lb)

check("domain is LB",              ref.domain  == "LB")
check("usubjid matches",           ref.usubjid == "042-S07-011")
check("seq is 1",                  ref.seq     == 1)
check("cite format correct",       ref.cite    == "LB|042-S07-011|1")
check("visit extracted",           ref.visit   == "SCREENING")
check("date extracted",            ref.date    == "2026-01-15")
check("testcd extracted (ALT)",    ref.testcd  == "ALT")
check("value extracted",           ref.value   != "")
check("unit extracted (ukat/L)",   ref.unit    == "ukat/L")
check("extra contains full row",   ref.extra.get("LBTESTCD") == "ALT")

print(f"\n  ref       : {ref}")
print(f"  repr      : {repr(ref)}")
print(f"  to_dict() : {json.dumps(ref.to_dict(), indent=4)}")
print()


# ==========================================================================
print("=" * 62)
print("TEST 2 — record_ref_exists: real record -> True")
print("=" * 62)

exists = record_ref_exists(ref, idx)
check("real LB|042-S07-011|1 exists", exists)

# Also check a real AE record
ae_rows = idx.get_adverse_events("042-S07-001")
if ae_rows:
    real_ae  = ae_rows[0]
    ref_ae   = make_record_ref("AE", real_ae)
    exists_ae = record_ref_exists(ref_ae, idx)
    check(f"real {ref_ae.cite} exists", exists_ae,
          f"term={real_ae.get('AETERM','?')}")
else:
    print("  (042-S07-001 has no AE records — skipping AE existence test)")

# Check several real LB records across different seq numbers
for seq_str in ("1", "30", "60"):
    row = next((r for r in lb_rows if r["LBSEQ"] == seq_str), None)
    if row:
        r = make_record_ref("LB", row)
        check(f"real {r.cite} exists", record_ref_exists(r, idx),
              f"test={row.get('LBTESTCD','?')} visit={row.get('VISIT','?')}")
print()


# ==========================================================================
print("=" * 62)
print("TEST 3 — record_ref_exists: fake records -> False")
print("=" * 62)

# 3a. Wrong subject ID (subject does not exist)
fake_subject = RecordRef(domain="LB", usubjid="042-S99-999", seq=1)
check("fake subject ID rejected",
      not record_ref_exists(fake_subject, idx),
      f"cite={fake_subject.cite}")

# 3b. Real subject, wrong SEQ (out of range — LB only has 1..60)
fake_seq = RecordRef(domain="LB", usubjid="042-S07-011", seq=999)
check("fake LBSEQ=999 rejected",
      not record_ref_exists(fake_seq, idx),
      f"cite={fake_seq.cite}")

# 3c. Real subject, real seq, but wrong domain (EX seq=1 exists but AE seq=99 does not)
fake_domain_seq = RecordRef(domain="AE", usubjid="042-S07-011", seq=99)
check("fake AE seq=99 rejected (subject has 0 AEs)",
      not record_ref_exists(fake_domain_seq, idx),
      f"cite={fake_domain_seq.cite}")

# 3d. Empty string subject
fake_empty = RecordRef(domain="LB", usubjid="", seq=1)
check("empty USUBJID rejected",
      not record_ref_exists(fake_empty, idx))

# 3e. Unknown domain
fake_unknown_domain = RecordRef(domain="XX", usubjid="042-S07-011", seq=1)
check("unknown domain XX rejected",
      not record_ref_exists(fake_unknown_domain, idx))

print()


# ==========================================================================
print("=" * 62)
print("TEST 4 — assert_record_ref_exists raises on fake ref")
print("=" * 62)

raised = False
try:
    assert_record_ref_exists(fake_subject, idx)
except ValueError as e:
    raised = True
    print(f"  ValueError raised (expected):")
    print(f"  {e}")
check("ValueError raised for fake ref", raised)
print()


# ==========================================================================
print("=" * 62)
print("TEST 5 — Evidence.build() with real refs")
print("=" * 62)

# Build two refs from consecutive LB rows for this subject
ref1 = make_record_ref("LB", lb_rows[0])   # LBSEQ=1 (ALT, SCREENING)
ref2 = make_record_ref("LB", lb_rows[1])   # LBSEQ=2 (AST, SCREENING)

claim = (
    f"Subject {uid} had ALT={ref1.value} {ref1.unit} "
    f"and AST={ref2.value} {ref2.unit} at SCREENING ({ref1.date})"
)
ev = Evidence.build(claim=claim, refs=[ref1, ref2], index=idx)

check("Evidence created successfully",    ev is not None)
check("claim text preserved",             ev.claim == claim)
check("two record_refs stored",           len(ev.record_refs) == 2)
check("citations correct",                ev.citations() == [ref1.cite, ref2.cite])
check("first cite is LB|042-S07-011|1",  ev.citations()[0] == "LB|042-S07-011|1")

print(f"\n  {ev}")
print(f"\n  to_dict():\n{json.dumps(ev.to_dict(), indent=4)}")
print()


# ==========================================================================
print("=" * 62)
print("TEST 6 — Evidence.build() REJECTS fake ref")
print("=" * 62)

raised_ev = False
try:
    bad_ev = Evidence.build(
        claim="This should fail",
        refs=[RecordRef(domain="LB", usubjid="042-S99-FAKE", seq=1)],
        index=idx,
    )
except ValueError as e:
    raised_ev = True
    print(f"  ValueError raised (expected):")
    print(f"  {e}")
check("Evidence.build raises for fake ref", raised_ev)

empty_raised = False
try:
    empty_ev = Evidence.build(claim="empty refs", refs=[], index=idx)
except ValueError as e:
    empty_raised = True
    print(f"\n  ValueError raised for empty refs (expected):")
    print(f"  {e}")
check("Evidence.build raises for empty refs", empty_raised)
print()


# ==========================================================================
print("=" * 62)
print("TEST 7 — Finding with Evidence")
print("=" * 62)

# Simulate a Hy's law candidate finding
alt_row  = next((r for r in lb_rows if r.get("LBTESTCD") == "ALT" and r.get("VISIT") == "SCREENING"), None)
bili_row = next((r for r in lb_rows if r.get("LBTESTCD") == "BILI" and r.get("VISIT") == "SCREENING"), None)

if alt_row and bili_row:
    ev_alt  = Evidence.build(
        claim=f"ALT={alt_row['LBORRES']} {alt_row['LBORRESU']} at {alt_row['VISIT']}",
        refs=[make_record_ref("LB", alt_row)],
        index=idx,
    )
    ev_bili = Evidence.build(
        claim=f"BILI={bili_row['LBORRES']} {bili_row['LBORRESU']} at {bili_row['VISIT']}",
        refs=[make_record_ref("LB", bili_row)],
        index=idx,
    )
    finding = Finding(
        category="HYS_LAW_CANDIDATE",
        usubjid=uid,
        description="ALT elevated with concurrent bilirubin elevation within 14 days",
        evidence=[ev_alt, ev_bili],
        severity="CRITICAL",
    )
    check("Finding created",                  finding is not None)
    check("Finding has 2 evidence objects",   len(finding.evidence) == 2)
    check("Finding has 2 total citations",    len(finding.all_citations()) == 2)

    d = finding.to_dict()
    check("to_dict has citations list",       "citations" in d)
    print(f"\n  Finding summary:")
    print(f"  category   : {d['category']}")
    print(f"  usubjid    : {d['usubjid']}")
    print(f"  severity   : {d['severity']}")
    print(f"  description: {d['description']}")
    print(f"  citations  : {d['citations']}")
    print(f"\n  Full to_dict():\n{json.dumps(d, indent=4)}")
else:
    print("  (Skipped — could not find ALT and BILI rows at SCREENING)")
print()


# ==========================================================================
print("=" * 62)
print(f"RESULTS: {PASS} passed, {FAIL} failed")
print("=" * 62)
