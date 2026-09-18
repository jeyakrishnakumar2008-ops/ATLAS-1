"""Quick smoke test for data_loader.py — stdlib csv only."""
from data_loader import load_csv, DATA_DIR, DOMAIN_FILES, META_FILES, StudyData

print("Data directory:", DATA_DIR)
print()

# Raw file row counts
all_files = list(DOMAIN_FILES.items()) + list(META_FILES.items())
print(f"{'Filename':<25} {'Raw rows':>9}  {'Cols':>5}")
print("-" * 45)
for name, fname in all_files:
    rows = load_csv(fname)
    ncols = len(rows[0].keys()) if rows else 0
    print(f"{fname:<25} {len(rows):>9}  {ncols:>5}")

print()

# Cut-filtered
sd = StudyData(cut=12)
counts = sd.row_counts()
print("=== After cut=12 filter + corrections applied ===")
print(f"{'Domain':<15} {'Rows':>9}")
print("-" * 27)
for key in ["DM","AE","LB","VS","EX","CM","DS","MH","EG","CUTS","CORRECTIONS","REF_RANGES"]:
    print(f"{key:<15} {counts.get(key, 0):>9}")

print()
subjects = sd.subjects()
print(f"Subjects from DM.csv : {len(subjects)}")
print(f"First 5 USUBJIDs     : {subjects[:5]}")
print(f"Protocol version     : v{sd.protocol_version()} (at cut 12)")

# Spot-check DM first row
print()
print("DM first row:")
for k, v in sd.dm[0].items():
    print(f"  {k}: {repr(v)}")

# Spot-check cut->protocol routing
print()
print("Cut -> protocol version:")
for cut in [1, 4, 5, 8, 9, 12]:
    pv = StudyData(cut=cut).protocol_version()
    print(f"  cut {cut:2d} -> v{pv}")

print()
print("=== LOADING SUCCEEDED ===")
