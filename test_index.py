"""
test_index.py
=============
Smoke test for study_index.py
Loads the study (cut=12), builds the index, then prints per-domain
record counts for several real subjects from DM.csv.
"""
from data_loader import StudyData, to_float
from study_index import build_study_index

# ── 1. Load ────────────────────────────────────────────────────────────────
print("Loading study data (cut=12) ...")
data = StudyData(cut=12)
print(f"  DM rows loaded : {len(data.dm)}")
print(f"  LB rows loaded : {len(data.lb)}")
print()

# ── 2. Build index ─────────────────────────────────────────────────────────
print("Building study index ...")
idx = build_study_index(data)
print(f"  Subjects indexed : {idx.subject_count()}")
print()

# ── 3. Per-subject record counts ───────────────────────────────────────────
# Pick three real subjects from different sites to show variety
test_subjects = [
    idx.all_subject_ids()[0],     # first alphabetically
    "042-S07-001",                # site S07 (local lab in ukat/L)
    "042-S01-005",                # another real subject
]

for uid in test_subjects:
    dm = idx.get_subject(uid)
    if dm is None:
        print(f"Subject {uid}: NOT FOUND")
        continue

    counts = idx._by_subject[uid].record_counts()
    print(f"Subject: {uid}")
    print(f"  Site : {dm.get('SITEID')}  |  ARM : {dm.get('ARM')}"
          f"  |  AGE : {dm.get('AGE')}  |  SEX : {dm.get('SEX')}")
    for domain, n in counts.items():
        print(f"  {domain:<4}: {n}")
    print()

# ── 4. Visit breakdown for one subject ────────────────────────────────────
uid = idx.all_subject_ids()[0]
visits = idx.get_visits(uid)
print(f"Visit breakdown for {uid}:")
for vname in sorted(visits.keys()):
    parts = []
    for domain in ("lb", "vs", "ex", "eg"):
        n = len(visits[vname][domain])
        if n:
            parts.append(f"{domain.upper()}={n}")
    print(f"  {vname:<12} {', '.join(parts)}")
print()

# ── 5. Reference-range lookup ────────────────────────────────────────────
print("Reference-range lookups:")
for test, site in [("ALT", "S01"), ("ALT", "S07"), ("AST", "S07"), ("BILI", "S03")]:
    rr = idx.ref_range(test, site)
    if rr:
        print(f"  {test} @ site {site} -> LOW={rr['LOW']} HIGH={rr['HIGH']} UNIT={rr['UNIT']} LAB={rr['LAB']}")
    else:
        print(f"  {test} @ site {site} -> no range found")
print()

# ── 6. to_float comma-decimal fix verification ────────────────────────────
print("to_float comma-decimal fix:")
cases = [("0,32", 0.32), ("40.4", 40.4), ("", None), ("<5", None), ("ND", None)]
all_ok = True
for raw, expected in cases:
    result = to_float(raw)
    ok = result == expected
    if not ok:
        all_ok = False
    print(f"  to_float({raw!r:8}) = {str(result):8}  {'OK' if ok else 'FAIL (expected ' + str(expected) + ')'}")
print()

# ── 7. Protocol version at each cut ──────────────────────────────────────
print("Protocol version at each cut:")
for cut in [1, 4, 5, 8, 9, 12]:
    pv = idx.protocol_version_at(cut)
    print(f"  cut {cut:2d} -> v{pv}")
print()

print("=== ALL TESTS PASSED ===" if all_ok else "=== SOME TESTS FAILED ===")
