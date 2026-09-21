"""Tests for Guardian Engine orchestration without payment-side effects."""

import unittest

from src.guardian_engine import analyze_with_guardian
from src.risk_engine import score_transaction


def transaction(**overrides):
    base = {
        "transaction_id": "TXN-1",
        "customer_id": "CUST-1",
        "amount": 1000,
        "avg_order_value": 1000,
        "payment_method": "UPI",
        "device_type": "Mobile",
        "ip_country": "IN",
        "billing_country": "IN",
        "failed_attempts": 0,
        "previous_order_count": 2,
        "account_age_days": 100,
        "is_new_customer": False,
    }
    base.update(overrides)
    return base


class GuardianEngineTests(unittest.TestCase):
    def test_amount_over_ten_rupees_starts_risk_points(self):
        self.assertEqual(score_transaction(transaction(amount=10))["risk_score"], 0)
        self.assertEqual(score_transaction(transaction(amount=10.01))["risk_score"], 1)

    def test_low_risk_transaction(self):
        result = analyze_with_guardian(transaction(), rule_result={"risk_band": "Low Risk", "signals": []}, ml_result={"ml_risk_score": 10})
        self.assertEqual(result["status"], "LOW")
        self.assertEqual(result["headline"], "Looks normal")

    def test_medium_pause_transaction(self):
        result = analyze_with_guardian(transaction(), rule_result={"risk_band": "Medium Risk", "signals": []})
        self.assertEqual(result["status"], "PAUSE")
        self.assertEqual(result["headline"], "Pause and verify")

    def test_high_risk_transaction(self):
        result = analyze_with_guardian(transaction(), rule_result={"risk_band": "High Risk", "signals": []})
        self.assertEqual(result["status"], "HIGH")
        self.assertEqual(result["headline"], "High-risk payment")

    def test_rule_result_is_preserved(self):
        rule = {"risk_band": "Low Risk", "risk_score": 15, "recommendation": "APPROVE", "signals": []}
        result = analyze_with_guardian(transaction(), rule_result=rule)
        self.assertEqual(result["rule_result"], rule)

    def test_ml_result_is_preserved(self):
        ml = {"ml_risk_score": 72, "decision": "REVIEW"}
        result = analyze_with_guardian(transaction(), ml_result=ml)
        self.assertEqual(result["ml_result"], ml)
        self.assertEqual(result["status"], "PAUSE")

    def test_new_recipient_signal_pauses(self):
        history = [{"transaction_id": "OLD", "customer_id": "CUST-2"}]
        result = analyze_with_guardian(transaction(), historical_transactions=history)
        self.assertEqual(result["status"], "PAUSE")
        self.assertTrue(any(signal["type"] == "NEW_RECIPIENT_OR_CUSTOMER" for signal in result["signals"]))

    def test_unusual_amount_signal_pauses(self):
        result = analyze_with_guardian(transaction(amount=4600, avg_order_value=1000))
        self.assertEqual(result["status"], "PAUSE")
        self.assertTrue(any(signal["type"] == "UNUSUAL_AMOUNT" for signal in result["signals"]))

    def test_multiple_failed_attempts_are_exposed(self):
        result = analyze_with_guardian(transaction(failed_attempts=4))
        self.assertTrue(any(signal["type"] == "RECENT_FAILED_ATTEMPTS" for signal in result["signals"]))

    def test_missing_history_is_limited(self):
        result = analyze_with_guardian(transaction())
        self.assertFalse(result["behavioral_result"]["history_available"])
        self.assertIn("Behavioral history is unavailable for this payment.", result["limitations"])

    def test_missing_ml_result_does_not_crash(self):
        result = analyze_with_guardian(transaction(), rule_result={"risk_band": "Low Risk"})
        self.assertEqual(result["ml_result"], {})

    def test_missing_rule_result_does_not_crash(self):
        result = analyze_with_guardian(transaction(), ml_result={"ml_risk_score": 10})
        self.assertEqual(result["rule_result"], {})

    def test_limited_razorpay_coverage_is_visible(self):
        limited = transaction(
            compatibility_values_are_not_payment_facts=True,
            feature_coverage="Limited",
            feature_sources={"avg_order_value": "unavailable", "customer_id": "unavailable"},
            unavailable_fraud_features=["avg_order_value", "customer_id"],
        )
        result = analyze_with_guardian(limited)
        self.assertEqual(result["context_result"]["availability"], "unavailable")
        self.assertTrue(result["limitations"])

    def test_no_automatic_payment_action(self):
        result = analyze_with_guardian(transaction(), rule_result={"risk_band": "High Risk"})
        serialized = str(result).lower()
        self.assertNotIn("payment_capture", serialized)
        self.assertNotIn("cancel payment", serialized)
        self.assertEqual(result["recommendation"], "Verify this payment before proceeding.")

    def test_explanation_is_included(self):
        result = analyze_with_guardian(transaction())
        self.assertEqual(result["headline"], result["explanation"]["headline"])

    def test_no_fraud_probability_language(self):
        result = analyze_with_guardian(transaction(), ml_result={"ml_risk_score": 92})
        text = " ".join([result["headline"], result["summary"], result["recommendation"]]).lower()
        self.assertNotIn("probability of fraud", text)
        self.assertNotIn("definitely", text)

    def test_duplicate_unusual_amount_is_one_aggregation_concept(self):
        result = analyze_with_guardian(
            transaction(amount=5000, avg_order_value=1000),
            rule_result={"risk_band": "Low Risk", "signals": [{"type": "UNUSUAL_AMOUNT"}]},
        )
        self.assertEqual(result["status"], "PAUSE")
        self.assertGreaterEqual(len(result["signals"]), 2)

    def test_duplicate_failed_attempts_are_one_aggregation_concept(self):
        result = analyze_with_guardian(
            transaction(failed_attempts=4),
            rule_result={"risk_band": "Low Risk", "signals": [{"type": "MULTIPLE_FAILED_ATTEMPTS"}]},
        )
        self.assertEqual(result["status"], "PAUSE")

    def test_new_recipient_and_unusual_amount_are_two_concepts(self):
        history = [{"transaction_id": "OLD", "customer_id": "CUST-2"}]
        result = analyze_with_guardian(
            transaction(amount=5000, avg_order_value=1000),
            historical_transactions=history,
        )
        self.assertEqual(result["status"], "PAUSE")
        types = {signal["type"] for signal in result["signals"]}
        self.assertTrue({"NEW_RECIPIENT_OR_CUSTOMER", "UNUSUAL_AMOUNT"}.issubset(types))

    def test_three_distinct_message_concepts_remain_high(self):
        result = analyze_with_guardian(
            transaction(),
            payment_message="URGENT! Pay immediately or your account will be blocked today. Final warning.",
        )
        self.assertEqual(result["status"], "HIGH")

    def test_ml_alone_still_cannot_be_high(self):
        result = analyze_with_guardian(transaction(), ml_result={"ml_risk_score": 99, "decision": "REVIEW"})
        self.assertEqual(result["status"], "PAUSE")


if __name__ == "__main__":
    unittest.main()
