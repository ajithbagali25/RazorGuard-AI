"""Tests for deterministic Guardian explanations."""

import unittest

from src.guardian_explainer import generate_guardian_explanation


class GuardianExplainerTests(unittest.TestCase):
    def test_low_risk_explanation(self):
        result = generate_guardian_explanation(rule_result={"risk_band": "Low Risk", "risk_score": 0})
        self.assertEqual(result["headline"], "Looks normal")
        self.assertIn("significant unusual signals", result["summary"].lower())

    def test_new_recipient_explanation(self):
        result = generate_guardian_explanation(
            behavioral_result={"signals": [{"type": "NEW_RECIPIENT_OR_CUSTOMER", "severity": "informational"}]}
        )
        self.assertEqual(result["signals"][0]["title"], "New recipient")
        self.assertIn("trusted contact", result["signals"][0]["recommended_action"].lower())

    def test_unusual_amount_explanation(self):
        result = generate_guardian_explanation(
            behavioral_result={"signals": [{"type": "UNUSUAL_AMOUNT", "severity": "elevated"}]}
        )
        self.assertEqual(result["signals"][0]["title"], "Unusual amount")

    def test_urgency_explanation(self):
        result = generate_guardian_explanation(
            context_result={"context_signals": [{"type": "URGENCY_DETECTED", "severity": "informational"}]}
        )
        self.assertEqual(result["signals"][0]["title"], "Urgency detected")
        self.assertIn("independently", result["signals"][0]["recommended_action"])

    def test_multiple_signals(self):
        result = generate_guardian_explanation(
            rule_result={"risk_band": "Medium Risk", "signals": [{"type": "COUNTRY_MISMATCH"}]},
            behavioral_result={"signals": [{"type": "RECENT_FAILED_ATTEMPTS"}]},
        )
        self.assertEqual(result["headline"], "Pause and verify")
        self.assertEqual(len(result["signals"]), 2)

    def test_missing_behavioral_data(self):
        result = generate_guardian_explanation(behavioral_result={"recipient_familiarity": "UNKNOWN"})
        self.assertIn("Behavioral history is unavailable for this payment.", result["limitations"])

    def test_compatibility_default_data(self):
        result = generate_guardian_explanation(
            context_result={"limitations": ["Compatibility fallback values are not payment facts and must not be used as behavioral evidence."]}
        )
        self.assertTrue(any("Compatibility fallback" in item for item in result["limitations"]))

    def test_no_false_fraud_probability_language(self):
        result = generate_guardian_explanation(rule_result={"risk_band": "High Risk"})
        combined = " ".join([result["headline"], result["summary"], result["recommendation"]]).lower()
        self.assertNotIn("definitely", combined)
        self.assertNotIn("probability of fraud", combined)

    def test_ml_score_is_not_presented_as_probability(self):
        result = generate_guardian_explanation(ml_result={"ml_risk_score": 92, "decision": "REVIEW"})
        notes = " ".join(result["technical_notes"]).lower()
        self.assertIn("ml risk score: 92/100", notes)
        self.assertIn("not a probability", notes)


if __name__ == "__main__":
    unittest.main()
