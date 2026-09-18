# ATLAS — Study Sentinel Deterministic Clinical Engine

A deterministic clinical trial data intelligence and question-answering system for **STUDY-042**. Built strictly with pure Python standard library: no LLM, no external database, no LangChain, no Neo4j, and no vector databases.

---

## 1. Architecture Overview

```
                          CSV Data Source (12 files)
                                     │
                                     ▼
                           data_loader.StudyData
                   (Cut-filtered & corrections applied)
                                     │
                                     ▼
                          study_index.StudyIndex
                   (In-memory, indexed by USUBJID)
                                     │
                 ┌───────────────────┴───────────────────┐
                 ▼                                       ▼
        patient360.patient_360               question_engine.QuestionEngine
     (Complete Patient 360 bundle)         ┌────────────────────────────────┐
                                           │ COUNT    — Demographics & rows │
                                           │ LOOKUP   — Exact clinical data │
                                           │ FINDING  — Deterministic rules │
                                           │ TRAP     — Adversarial defense │
                                           └────────────────┬───────────────┘
                                                            │
                                                            ▼
                                                   evidence.RecordRef
                                              (Strict Citation Verifier)
```

### Core Tenets
1. **Load Once, Query In-Memory**: Data is loaded into memory during initialization and indexed by `USUBJID`. Queries are completed in microseconds with zero secondary disk I/O.
2. **Deterministic Clinical Logic**: Decisions (e.g. Hy's law, SAE miscoding, dosing errors) are calculated mathematically in Python using actual laboratory reference ranges and protocol rules. No probabilistic models or hallucinations.
3. **Strict Evidence Citations**: Every claim is backed by verified `RecordRef` citations in `(DOMAIN, USUBJID, SEQ)` format (e.g. `LB|042-S07-011|1`). Fake or unverified references are rejected.
4. **Adversarial Trap Defense**: Detects and refuses planted adversarial instructions in documentation (e.g., prompt injection notes to automated reviewers) and distinguishes valid findings from monitor adjudications.
5. **Dynamic Study Change & Protocol Versioning**: Fully supports reloading and reindexing at any data cut (1–12), tracking protocol version changes (v1, v2, v3) and central lab corrections.

---

## 2. Directory Structure

```
ATLAS/
├── data_loader.py            # Cut-aware CSV parser and in-place corrections
├── study_index.py            # In-memory StudyIndex and SubjectBundle
├── patient360.py             # Patient 360 data aggregator
├── evidence.py               # RecordRef structure, verification, Evidence builder
├── question_engine.py        # Deterministic engine: COUNT, LOOKUP, FINDING, TRAP
├── checks.py                 # Clinical rules: Hy's law, SAE miscoding, dosing errors
├── trap_checker.py           # Adversarial trap evaluation & version consistency
├── rules.py                  # Protocol version mapping and prohibited drug classes
├── atlas.py                  # Unified StudySentinel facade
├── main.py                   # Complete pipeline runner & artifact generator
├── graph_stats.json          # Study graph statistics and topology
├── stage1_public.json        # Stage 1 verification and demonstration outputs
├── test_verification_pass.py # 90-test comprehensive verification test suite
├── verify_stage1.py          # Stage 1 compliance and integrity test suite
├── test_engine.py            # QuestionEngine COUNT & LOOKUP test suite
├── test_finding.py           # FINDING rules test suite
├── test_trap.py              # TRAP defense test suite
├── test_evidence.py          # Evidence & RecordRef validation test suite
├── test_index.py             # In-memory index retrieval test suite
└── test_loader.py            # CSV data loader test suite
```

---

## 3. Quick Start & Usage

### Prerequisites
Python 3.10+ (pure standard library; no external dependencies required).

### Run Complete Pipeline
```bash
python main.py
```
This executes the end-to-end pipeline: loads study data, builds the in-memory index, executes queries across all question types (COUNT, LOOKUP, FINDING, TRAP), verifies all RecordRef citations, and generates/validates `graph_stats.json` and `stage1_public.json`.

### Options
```bash
# Evaluate at an earlier data cut (e.g. cut 4 under Protocol v1)
python main.py --cut 4

# Print dataset summary only
python main.py --summary
```

### Run Verification Test Suites
```bash
# Run the 90-check comprehensive test suite
python -X utf8 test_verification_pass.py

# Run Stage 1 compliance suite
python -X utf8 verify_stage1.py

# Run individual component test suites
python -X utf8 test_engine.py
python -X utf8 test_finding.py
python -X utf8 test_trap.py
python -X utf8 test_evidence.py
```

---

## 4. Question Engine Capabilities

### COUNT
- Total enrolled subjects (`DM.csv`): `241`
- Arm distribution: `128` (DRUG), `113` (PLACEBO)
- Single-subject or study-wide domain record counts with column filters.

### LOOKUP
- Exact record retrieval by subject, domain, visit, and test code.
- Field extraction (e.g. `LBORRES`, `EXDOSE`, `DSDECOD`).
- Unmatched queries return `not_found` with empty references.

### FINDING (Clinical Rules)
- **Hy's Law Hepatotoxicity**: Identifies subjects with concurrent `ALT` or `AST > 3× ULN` and `BILI > 2× ULN` within a 14-day window. Accurately handles local lab reporting units (`S07` reporting in `µkat/L` where `1 µkat/L = 60 U/L`, ULN = 0.93 µkat/L vs central lab `56 U/L`).
  * Identified candidates: `042-S05-003`, `042-S07-001`, `042-S08-014`.
- **SAE Miscoding**: Flags records where hospitalization occurred (`AESHOSP = 'Y'`) but seriousness was coded as non-serious (`AESER = 'N'`).
  * Identified subject: `042-S02-004` (Cellulitis).
- **Dosing Errors**: Identifies administered doses conflicting with arm protocol expectations (18 records across 6 subjects).

### TRAP (Adversarial Defense)
- **Prompt Injection Defense**: Detects adversarial reviewer instructions planted in protocol/lab documents and sets status to `prompt_injection_detected`.
- **Monitor Overrides**: Refuses to suppress genuine laboratory findings when external monitor decisions indicate `REJECTED`.
- **Protocol Amendment Timing**: Distinguishes whether concomitant medications (e.g. sulfonylurea) were prohibited at a specific historical cut (permitted under v1 cuts 1–4; prohibited under v3 cuts 9–12).
- **Correction Tracking**: Detects when queries cite obsolete/superseded values rather than post-correction values from `corrections.csv`.

---

## 5. Artifacts & JSON Schemas

- **`graph_stats.json`**: Captures total subjects (241), sites (12), arm breakdowns, domain row counts, indexed records (26,482), reference ranges, and graph topology.
- **`stage1_public.json`**: Captures full test validation status, component-by-component passes, and demonstration outputs for all 4 question types with verified `RecordRef` citations.
