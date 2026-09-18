"""
patient360.py
=============
Patient 360 summary for a real subject from the supplied dataset.
Queries the in-memory StudyIndex — no CSV files are re-read.
"""
from __future__ import annotations

import time
from typing import Any
from data_loader import StudyData, to_float, parse_date
from study_index import StudyIndex, build_study_index
from rules import prohibited_classes


def patient_360(idx: StudyIndex, uid: str) -> dict[str, Any]:
    """
    Generate complete Patient 360 bundle for a subject from the in-memory index.
    """
    dm = idx.get_subject(uid)
    if dm is None:
        return {}

    aes = idx.get_adverse_events(uid)
    labs = idx.get_labs(uid)
    vs = idx.get_vitals(uid)
    ex = idx.get_exposure(uid)
    cm = idx.get_medications(uid)
    mh = idx.get_medical_history(uid)
    ds = idx.get_disposition(uid)
    eg = idx.get_ecg(uid)
    vis = idx.get_visits(uid)

    return {
        "usubjid": uid,
        "demographics": {
            "site": dm.get("SITEID", ""),
            "country": dm.get("COUNTRY", ""),
            "arm": dm.get("ARM", ""),
            "age": dm.get("AGE", ""),
            "sex": dm.get("SEX", ""),
            "dminit": dm.get("DMINIT", ""),
            "brthdtc": dm.get("BRTHDTC", ""),
            "rfstdtc": dm.get("RFSTDTC", ""),
            "scr_hba1c": dm.get("SCR_HBA1C", ""),
        },
        "record_counts": {
            "DM": 1,
            "AE": len(aes),
            "LB": len(labs),
            "VS": len(vs),
            "EX": len(ex),
            "CM": len(cm),
            "MH": len(mh),
            "DS": len(ds),
            "EG": len(eg),
        },
        "visits": vis,
        "dm": dm,
        "ae": aes,
        "lb": labs,
        "vs": vs,
        "ex": ex,
        "cm": cm,
        "mh": mh,
        "ds": ds,
        "eg": eg,
    }


def run_demo() -> None:
    print("=" * 62)
    print("PHASE 1: Load + Build Index")
    print("=" * 62)

    t0 = time.perf_counter()
    data = StudyData(cut=12)
    t1 = time.perf_counter()
    idx = build_study_index(data)
    t2 = time.perf_counter()

    print(f"  CSV load time   : {(t1-t0)*1000:6.1f} ms")
    print(f"  Index build time: {(t2-t1)*1000:6.1f} ms")
    print(f"  Total subjects  : {idx.subject_count()}")
    print()

    TARGET = "042-S07-011"
    print("=" * 62)
    print(f"PHASE 2: Patient 360 — {TARGET}")
    print("=" * 62)

    t3 = time.perf_counter()
    p_data = patient_360(idx, TARGET)
    t4 = time.perf_counter()
    query_us = (t4 - t3) * 1_000_000

    print(f"  [Query time: {query_us:.1f} µs — no CSV reads]")
    print()

    dm = p_data["dm"]
    print("-" * 62)
    print("DEMOGRAPHICS")
    print("-" * 62)
    print(f"  Subject ID  : {dm['USUBJID']}")
    print(f"  Site        : {dm['SITEID']}  (country: {dm['COUNTRY']})")
    print(f"  ARM         : {dm['ARM']}")
    print(f"  Age         : {dm['AGE']}  |  Sex: {dm['SEX']}")
    print(f"  First dose  : {dm['RFSTDTC']}  (RFSTDTC)")
    print(f"  Scr HbA1c   : {dm['SCR_HBA1C']} %  (inclusion: 7.0–10.5)")
    print()

    counts = p_data["record_counts"]
    print("-" * 62)
    print("RECORD COUNTS")
    print("-" * 62)
    print(f"  Visits (unique names) : {len(p_data['visits'])}")
    print(f"  Lab records (LB)      : {counts['LB']}")
    print(f"  Adverse events (AE)   : {counts['AE']}")
    print(f"  Exposure records (EX) : {counts['EX']}")
    print(f"  Concomitant meds (CM) : {counts['CM']}")
    print(f"  Medical history (MH)  : {counts['MH']}")
    print(f"  Disposition (DS)      : {counts['DS']}")
    print(f"  ECG records (EG)      : {counts['EG']}")
    print(f"  Vital signs (VS)      : {counts['VS']}")
    print()


if __name__ == "__main__":
    run_demo()
