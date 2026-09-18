"""
inspect_schema.py
=================
Introspects every CSV file in the data directory and produces a
relationship / schema report.  Read-only — no files are modified.
"""
import csv, sys
from pathlib import Path
from collections import defaultdict

DATA_DIR = Path(__file__).parent / "hackathon-data" / "hackathon-data" / "data"

FILES = [
    "DM.csv", "AE.csv", "LB.csv", "VS.csv", "EX.csv",
    "CM.csv", "MH.csv", "DS.csv", "EG.csv",
    "reference_ranges.csv", "cuts.csv", "corrections.csv",
]

def load(fname):
    rows = []
    with open(DATA_DIR / fname, newline="", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        for r in reader:
            rows.append({k.strip(): ("" if v is None else v.strip()) for k, v in r.items()})
    return rows

def col_stats(rows, col):
    """Return (n_nonempty, n_unique, sample_values)."""
    vals = [r[col] for r in rows if r.get(col, "") != ""]
    unique = sorted(set(vals))
    samples = unique[:6]
    return len(vals), len(unique), samples

def detect_date_cols(cols):
    date_hints = {"DTC", "DATE", "STDT", "ENDT", "BRTH"}
    return [c for c in cols if any(h in c.upper() for h in date_hints)]

def detect_seq_cols(cols):
    return [c for c in cols if c.upper().endswith("SEQ")]

def detect_visit_cols(cols):
    visit_hints = {"VISIT", "WEEK", "EPOCH"}
    return [c for c in cols if any(h in c.upper() for h in visit_hints)]

SEP = "=" * 72

all_data = {}
for fname in FILES:
    all_data[fname] = load(fname)

# ── per-file analysis ────────────────────────────────────────────────────────
print(SEP)
print("ATLAS SCHEMA INSPECTION REPORT")
print(SEP)

for fname in FILES:
    rows = all_data[fname]
    cols = list(rows[0].keys()) if rows else []
    print(f"\n{'─'*72}")
    print(f"FILE : {fname}   ({len(rows)} rows, {len(cols)} columns)")
    print(f"{'─'*72}")

    # USUBJID
    if "USUBJID" in cols:
        n, u, s = col_stats(rows, "USUBJID")
        print(f"  Subject key  : USUBJID  ({u} unique subjects, e.g. {s[:3]})")
    elif "SITEID" in cols and "USUBJID" not in cols:
        print("  Subject key  : [none – not a subject-level file]")
    else:
        print("  Subject key  : [none]")

    # Sequence cols
    seq_cols = detect_seq_cols(cols)
    for sc in seq_cols:
        n, u, s = col_stats(rows, sc)
        print(f"  Record seq   : {sc}  (range ~1..{max((int(v) for r in rows for v in [r.get(sc,'0')] if v.isdigit()), default='?')})")

    # Visit cols
    visit_cols = detect_visit_cols(cols)
    for vc in visit_cols:
        n, u, s = col_stats(rows, vc)
        print(f"  Visit col    : {vc}  ({u} unique: {s})")

    # Date cols
    date_cols = detect_date_cols(cols)
    for dc in date_cols:
        n, u, s = col_stats(rows, dc)
        print(f"  Date col     : {dc}  ({n} non-empty, samples: {s[:3]})")

    # All columns with sample values
    print(f"  All columns  ({len(cols)}):")
    for c in cols:
        n, u, s = col_stats(rows, c)
        empty = len(rows) - n
        note = f"  [EMPTY in {empty} rows]" if empty > 0 else ""
        print(f"    {c:<25}  unique={u:>6}  sample={s[:4]}{note}")

# ── cross-table key analysis ─────────────────────────────────────────────────
print(f"\n{SEP}")
print("CROSS-TABLE KEY ANALYSIS")
print(SEP)

# USUBJID coverage
domain_subjects = {}
for fname in FILES:
    rows = all_data[fname]
    if rows and "USUBJID" in rows[0]:
        subjects = {r["USUBJID"] for r in rows if r.get("USUBJID")}
        domain_subjects[fname] = subjects

dm_subjects = domain_subjects.get("DM.csv", set())
print(f"\nMaster subject list (DM.csv): {len(dm_subjects)} subjects")
for fname, subjects in sorted(domain_subjects.items()):
    if fname == "DM.csv":
        continue
    in_dm  = subjects & dm_subjects
    not_dm = subjects - dm_subjects
    miss   = dm_subjects - subjects
    print(f"  {fname:<25}  {len(subjects):>4} unique USUBJIDs"
          f"  | in DM={len(in_dm)}  extra={len(not_dm)}  missing_from_file={len(miss)}")

# VISIT values across files
print("\nVISIT values present across files:")
for fname in FILES:
    rows = all_data[fname]
    if rows and "VISIT" in rows[0]:
        visits = sorted({r["VISIT"] for r in rows if r.get("VISIT")})
        print(f"  {fname:<25}  visits={visits}")

# corrections.csv join keys
print("\ncorrections.csv join keys:")
corr_rows = all_data["corrections.csv"]
if corr_rows:
    domains_in_corr = sorted({r.get("domain","") for r in corr_rows})
    fields_in_corr  = sorted({r.get("field","") for r in corr_rows})
    print(f"  domains corrected : {domains_in_corr}")
    print(f"  fields corrected  : {fields_in_corr}")
    print(f"  cuts present      : {sorted({r.get('cut','') for r in corr_rows})}")
    print(f"  join via          : (domain, usubjid, seq) -> domain.{'{DOMAIN}SEQ'}")

# reference_ranges.csv join keys
print("\nreference_ranges.csv join keys:")
rr_rows = all_data["reference_ranges.csv"]
if rr_rows:
    tests = sorted({r.get("LBTESTCD","") for r in rr_rows})
    labs  = sorted({r.get("LAB","") for r in rr_rows})
    units = sorted({r.get("UNIT","") for r in rr_rows})
    print(f"  LBTESTCD values   : {tests}")
    print(f"  LAB values        : {labs}")
    print(f"  UNIT values       : {units}")
    print(f"  join via          : LB.LBTESTCD + DM.SITEID (-> LAB) -> reference_ranges")

# cuts.csv
print("\ncuts.csv join keys:")
for r in all_data["cuts.csv"]:
    print(f"  cut={r['cut']}  protocol_version={r['protocol_version']}  "
          f"new_records={r['new_records']}  corrections={r['corrections']}")

# ── ARM / treatment group ────────────────────────────────────────────────────
print(f"\n{SEP}")
print("TREATMENT GROUP (ARM) in DM.csv")
print(SEP)
arm_counts = defaultdict(int)
for r in all_data["DM.csv"]:
    arm_counts[r.get("ARM","?")] += 1
for arm, n in sorted(arm_counts.items()):
    print(f"  ARM={arm!r:15}  {n} subjects")

# ── CMCLAS values (prohibited-med check) ─────────────────────────────────────
print(f"\n{SEP}")
print("CMCLAS values in CM.csv (concomitant medication classes)")
print(SEP)
cmclas_counts = defaultdict(int)
for r in all_data["CM.csv"]:
    cmclas_counts[r.get("CMCLAS","?")] += 1
for cls, n in sorted(cmclas_counts.items()):
    print(f"  CMCLAS={cls!r:35}  {n} records")

# ── DSDECOD values (disposition outcomes) ────────────────────────────────────
print(f"\n{SEP}")
print("DSDECOD values in DS.csv (disposition)")
print(SEP)
ds_counts = defaultdict(int)
for r in all_data["DS.csv"]:
    ds_counts[r.get("DSDECOD","?")] += 1
for dec, n in sorted(ds_counts.items()):
    print(f"  DSDECOD={dec!r:30}  {n} records")

# ── LBTESTCD and VSTESTCD distinct values ─────────────────────────────────────
print(f"\n{SEP}")
print("LBTESTCD distinct values in LB.csv")
print(SEP)
lbtests = sorted({r.get("LBTESTCD","") for r in all_data["LB.csv"] if r.get("LBTESTCD")})
for t in lbtests:
    units = sorted({r.get("LBORRESU","") for r in all_data["LB.csv"]
                    if r.get("LBTESTCD")==t and r.get("LBORRESU")})
    print(f"  {t:<10}  units={units}")

print(f"\n{SEP}")
print("VSTESTCD distinct values in VS.csv")
print(SEP)
vstests = sorted({r.get("VSTESTCD","") for r in all_data["VS.csv"] if r.get("VSTESTCD")})
for t in vstests:
    units = sorted({r.get("VSORRESU","") for r in all_data["VS.csv"]
                    if r.get("VSTESTCD")==t and r.get("VSORRESU")})
    print(f"  {t:<15}  units={units}")

print(f"\n{SEP}")
print("EGTESTCD distinct values in EG.csv")
print(SEP)
egtests = sorted({r.get("EGTESTCD","") for r in all_data["EG.csv"] if r.get("EGTESTCD")})
for t in egtests:
    units = sorted({r.get("EGORRESU","") for r in all_data["EG.csv"]
                    if r.get("EGTESTCD")==t and r.get("EGORRESU")})
    print(f"  {t:<15}  units={units}")

# ── AESER / AESHOSP mismatch potential ───────────────────────────────────────
print(f"\n{SEP}")
print("AE: AESER x AESHOSP cross-tabulation (mismatch detection)")
print(SEP)
ae_cross = defaultdict(int)
for r in all_data["AE.csv"]:
    key = (r.get("AESER","?"), r.get("AESHOSP","?"))
    ae_cross[key] += 1
print(f"  {'(AESER, AESHOSP)':<25}  count")
for (ser, hosp), n in sorted(ae_cross.items()):
    flag = "  *** POTENTIAL MISMATCH ***" if ser=="N" and hosp=="Y" else ""
    print(f"  ({ser!r}, {hosp!r}){'':<18}  {n:>4}{flag}")

# ── EXDOSE values by ARM ──────────────────────────────────────────────────────
print(f"\n{SEP}")
print("EX: EXDOSE values (dosing errors if != 10mg DRUG or 0mg PLACEBO)")
print(SEP)
# Join EX -> DM on USUBJID
dm_arm = {r["USUBJID"]: r.get("ARM","?") for r in all_data["DM.csv"]}
ex_dose_arm = defaultdict(set)
for r in all_data["EX.csv"]:
    arm  = dm_arm.get(r.get("USUBJID",""), "?")
    dose = r.get("EXDOSE","")
    unit = r.get("EXDOSU","")
    ex_dose_arm[(arm, dose, unit)].add(r.get("USUBJID",""))
print(f"  {'ARM':<10}  {'EXDOSE':<8}  {'EXDOSU':<6}  subjects")
for (arm, dose, unit), subjects in sorted(ex_dose_arm.items()):
    flag = ""
    try:
        d = float(dose)
        if arm=="DRUG"    and d != 10.0: flag = "  *** DOSING ERROR ***"
        if arm=="PLACEBO" and d != 0.0:  flag = "  *** DOSING ERROR ***"
    except ValueError:
        pass
    print(f"  {arm:<10}  {dose:<8}  {unit:<6}  {len(subjects)}{flag}")

print(f"\n{SEP}")
print("END OF SCHEMA INSPECTION REPORT")
print(SEP)
