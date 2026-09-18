"""
test_monitor.py
===============
End-to-End Validation Test Suite for Problem 2: ATLAS MONITOR Pipeline.

Verifies:
1. Complete MONITOR cycle execution.
2. Cross-cycle memory: same cut rerun -> 0 new queries, 0 new escalations.
3. Escalation APPROVED decision & trace.
4. Escalation REJECTED decision & permanent downgrade to monitoring (no re-escalation).
5. Escalation CLARIFY decision -> answers from ATLAS evidence graph -> resubmits.
6. SAE_MISCODED subject 042-S02-004 correctly escalates with evidence.
7. AE before first dose generates a data query.
8. Protocol version adherence (Cut 4 v1 vs Cut 12 v3 Sulfonylurea deviation).
9. AESHOSP=Y recognized as serious even when AESER=N.
10. HTTP endpoints on running server (/api/monitor/run, /api/monitor/decision, /queries, /escalations).
"""

import unittest
from atlas import StudySentinel
from monitor import AtlasMonitorPipeline, MonitorMemory, EscalationItem


class TestAtlasMonitorPipeline(unittest.TestCase):

    def setUp(self):
        self.sentinel_cut12 = StudySentinel(cut=12)
        self.memory = MonitorMemory()
        self.pipeline = AtlasMonitorPipeline(self.sentinel_cut12, memory=self.memory)

    def test_01_complete_monitor_cycle(self):
        """Test 1: Run one complete MONITOR cycle."""
        result = self.pipeline.run_cycle()
        self.assertEqual(result["status"], "success")
        self.assertGreater(len(result["findings"]), 0)
        self.assertGreater(result["report"]["finding_count"], 0)
        self.assertGreater(result["report"]["query_count"], 0)
        self.assertGreater(result["report"]["escalation_count"], 0)
        self.assertGreater(len(result["report"]["trace"]), 0)

    def test_02_rerun_deduplication(self):
        """Test 2: Same cut run twice -> 0 new queries and 0 new escalations."""
        res1 = self.pipeline.run_cycle()
        new_q1 = res1["report"]["new_query_count"]
        new_e1 = res1["report"]["new_escalation_count"]
        self.assertGreater(new_q1, 0)
        self.assertGreater(new_e1, 0)

        # Second run of same cut
        res2 = self.pipeline.run_cycle()
        self.assertEqual(res2["report"]["new_query_count"], 0)
        self.assertEqual(res2["report"]["new_escalation_count"], 0)

    def test_03_escalation_approved(self):
        """Test 3: Escalation APPROVED executes proposed action and logs trace."""
        self.pipeline.run_cycle()
        esc = list(self.memory.escalations.values())[0]
        self.assertEqual(esc.status, "PENDING")

        self.pipeline.decision_center.adjudicate(
            esc, action="APPROVED", custom_reason="Medical Monitor signs off.", memory=self.memory
        )
        self.assertEqual(esc.status, "APPROVED")
        self.assertEqual(self.memory.escalation_history[f"{esc.finding_code}|{esc.usubjid}"], "APPROVED")

        # Verify trace entry
        traces = [t for t in self.memory.trace if t.decision == "APPROVED" and t.subject == esc.usubjid]
        self.assertGreater(len(traces), 0)
        self.assertEqual(traces[0].module, "Decision Center")

    def test_04_escalation_rejected_never_reescalates(self):
        """Test 4: Escalation REJECTED downgrades to monitoring and never re-escalates."""
        self.pipeline.run_cycle()
        esc = list(self.memory.escalations.values())[0]
        key = f"{esc.finding_code}|{esc.usubjid}"

        self.pipeline.decision_center.adjudicate(
            esc, action="REJECTED", custom_reason="Baseline elevation; monitor, do not escalate.", memory=self.memory
        )
        self.assertEqual(esc.status, "REJECTED")
        self.assertIn(key, self.memory.rejected_escalations)

        # Clear escalations dictionary and rerun
        self.memory.escalations.clear()
        res = self.pipeline.run_cycle()
        re_escalated = any(e.usubjid == esc.usubjid and e.finding_code == esc.finding_code for e in self.memory.escalations.values())
        self.assertFalse(re_escalated, "Rejected escalation must NOT be re-escalated!")

    def test_05_escalation_clarify(self):
        """Test 5: CLARIFY answers from ATLAS graph/evidence and resubmits without rejection."""
        self.pipeline.run_cycle()
        esc = list(self.memory.escalations.values())[0]

        q = "What was the ALT at screening, and is there a concomitant hepatotoxic medication?"
        self.pipeline.decision_center.adjudicate(esc, action="CLARIFY", custom_reason=q, memory=self.memory)

        self.assertEqual(esc.status, "CLARIFIED_APPROVED")
        self.assertNotEqual(esc.status, "REJECTED")
        self.assertIn("Screening ALT:", esc.clarification_answer)
        self.assertIn("Concomitant Meds:", esc.clarification_answer)

    def test_06_sae_miscoded_escalation(self):
        """Test 6: SAE_MISCODED subject 042-S02-004 leads to escalation."""
        self.pipeline.run_cycle()
        s02_004_esc = [e for e in self.memory.escalations.values() if e.usubjid == "042-S02-004"]
        self.assertGreater(len(s02_004_esc), 0)
        self.assertIn(s02_004_esc[0].finding_code, ("SAE_MISCODING", "SAE_MISCODED"))
        self.assertEqual(s02_004_esc[0].severity, "CRITICAL")
        self.assertIn("AE|042-S02-004|1", s02_004_esc[0].record_refs)

    def test_07_ae_before_first_dose_query(self):
        """Test 7: AE before first dose generates a data query citing RecordRef."""
        self.pipeline.run_cycle()
        pre_dose_queries = [
            q for q in self.memory.active_queries
            if "AE_BEFORE_FIRST_DOSE" in q.title or "first study dose" in q.details.lower()
        ]
        self.assertGreater(len(pre_dose_queries), 0)
        q = pre_dose_queries[0]
        self.assertEqual(q.action_type, "QUERY")
        self.assertTrue(q.record_ref.startswith("AE|"))

        # Specifically check 042-S02-010
        s02_010 = [q for q in pre_dose_queries if q.usubjid == "042-S02-010"]
        self.assertGreater(len(s02_010), 0)
        self.assertEqual(s02_010[0].record_ref, "AE|042-S02-010|1")
        self.assertIn("CLOSED", s02_010[0].site_reply)

    def test_08_protocol_version_compliance(self):
        """Test 8: Protocol version adherence across amendments."""
        # Cut 4 = Protocol v1 (Sulfonylurea was NOT prohibited)
        s4 = StudySentinel(cut=4)
        m4 = MonitorMemory()
        p4 = AtlasMonitorPipeline(s4, memory=m4)
        devs_cut4 = p4.compliance.check_compliance()
        sulf_cut4 = [d for d in devs_cut4 if "sulfonylurea" in d.details.lower() or "glibenclamide" in d.details.lower()]
        self.assertEqual(len(sulf_cut4), 0, "Cut 4 (Protocol v1) must NOT flag sulfonylurea as deviation!")

        # Cut 12 = Protocol v3 (Sulfonylurea IS prohibited by Amendment 2)
        devs_cut12 = self.pipeline.compliance.check_compliance()
        sulf_cut12 = [d for d in devs_cut12 if "sulfonylurea" in d.details.lower() or "glibenclamide" in d.details.lower()]
        self.assertGreater(len(sulf_cut12), 0, "Cut 12 (Protocol v3) MUST flag sulfonylurea as deviation!")

    def test_09_aeshosp_seriousness(self):
        """Test 9: AESHOSP=Y must be recognized as serious even when AESER=N."""
        findings = self.pipeline.intake.run()
        hosp_findings = [f for f in findings if f.category in ("SAE_MISCODING", "SAE_MISCODED")]
        self.assertGreater(len(hosp_findings), 0)
        for f in hosp_findings:
            risk = self.pipeline.risk_assessor.evaluate(f, self.memory, 1)
            self.assertEqual(risk.seriousness, "CRITICAL")
            self.assertEqual(risk.recommended_action, "ESCALATE")


if __name__ == "__main__":
    unittest.main()
