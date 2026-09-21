"""Tests for deterministic payment-message context analysis."""

import unittest

from src.context_analyzer import analyze_context, analyze_payment_message


class ContextAnalyzerMessageTests(unittest.TestCase):
    def test_no_message(self):
        self.assertEqual(analyze_payment_message(None), [])

    def test_empty_message(self):
        self.assertEqual(analyze_payment_message("   "), [])

    def test_normal_payment_message(self):
        self.assertEqual(analyze_payment_message("Payment for my regular purchase today."), [])

    def test_urgency_message(self):
        signals = analyze_payment_message("URGENT: please act now")
        self.assertEqual([signal["type"] for signal in signals], ["URGENCY_DETECTED"])

    def test_account_threat_message(self):
        signals = analyze_payment_message("Your account will be suspended unless you pay.")
        self.assertTrue(any(signal["type"] == "ACCOUNT_THREAT_DETECTED" for signal in signals))

    def test_pressure_message(self):
        signals = analyze_payment_message("Final warning: you must pay now.")
        self.assertTrue(any(signal["type"] == "PRESSURE_TO_ACT_DETECTED" for signal in signals))

    def test_unusual_payment_instruction(self):
        signals = analyze_payment_message("Do not contact support; send to this new account.")
        self.assertTrue(any(signal["type"] == "UNUSUAL_PAYMENT_INSTRUCTION" for signal in signals))

    def test_multiple_signals(self):
        signals = analyze_payment_message("URGENT! Pay immediately or your account will be blocked today.")
        types = {signal["type"] for signal in signals}
        self.assertTrue({"URGENCY_DETECTED", "ACCOUNT_THREAT_DETECTED", "PRESSURE_TO_ACT_DETECTED"}.issubset(types))

    def test_case_insensitive_detection(self):
        signals = analyze_payment_message("aSaP: ACCOUNT WILL BE LOCKED")
        self.assertTrue(any(signal["type"] == "URGENCY_DETECTED" for signal in signals))
        self.assertTrue(any(signal["type"] == "ACCOUNT_THREAT_DETECTED" for signal in signals))

    def test_no_duplicate_signals(self):
        signals = analyze_payment_message("urgent urgent ASAP act now")
        self.assertEqual(len([signal for signal in signals if signal["type"] == "URGENCY_DETECTED"]), 1)

    def test_message_does_not_declare_fraud(self):
        result = analyze_context({}, "URGENT! Pay immediately or your account will be blocked today.")
        text = " ".join(
            item["description"] + " " + item["why_it_matters"]
            for item in result["context_signals"]
            if "description" in item
        ).lower()
        self.assertNotIn("definitely", text)
        self.assertNotIn("fraudulent", text)


if __name__ == "__main__":
    unittest.main()
