from data_loader import StudyData, to_float
from study_index import build_study_index
import rules

data = StudyData(cut=12)
idx  = build_study_index(data)

rr_s07  = idx.ref_range('ALT','S07')
uln_s07 = to_float(rr_s07['HIGH'])   # 0.93 ukat/L

print("=== S07-001 ALT trajectory ===")
labs = idx.get_labs('042-S07-001')
dm   = idx.get_subject('042-S07-001')
print(f"  ARM={dm['ARM']}  Site={dm['SITEID']}")
for lb in labs:
    if lb['LBTESTCD'] == 'ALT':
        v = to_float(lb['LBORRES'])
        xULN = v / uln_s07 if v else 0
        print(f"  ALT {lb['VISIT']:12} {lb['LBORRES']:8} ukat/L  {xULN:.2f}xULN  >2xULN={v > 2*uln_s07}")

print()
print("=== CM sulfonylurea records ===")
for uid in idx.all_subject_ids():
    for cm in idx.get_medications(uid):
        if 'SULFONYLUREA' in cm.get('CMCLAS','').upper():
            ca = cm.get('cut_available','?')
            print(f"  {uid}: CMCLAS={cm['CMCLAS']} VISIT={cm.get('VISIT','?')} cut_avail={ca}")

print()
print("=== AE AESHOSP=Y / AESER=N ===")
for uid in idx.all_subject_ids():
    for ae in idx.get_adverse_events(uid):
        if ae.get('AESHOSP','').upper()=='Y' and ae.get('AESER','').upper()=='N':
            print(f"  {uid} AESEQ={ae['AESEQ']} AETERM={ae['AETERM']} AESHOSP=Y AESER=N")

print()
print("=== DM vs DS count ===")
print(f"  DM rows: {len(data.dm)}")
print(f"  DS rows: {len(data.ds)}")
dm_ids = set(r['USUBJID'] for r in data.dm)
ds_ids = set(r['USUBJID'] for r in data.ds)
print(f"  Subjects in DM not in DS: {sorted(dm_ids - ds_ids)}")
