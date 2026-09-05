import os
import unittest
from types import SimpleNamespace
from unittest.mock import patch

from src.razorpay_client import (
    RazorpayIntegrationError,
    fetch_test_payment,
    payment_to_test_transaction,
)


VALID_PAYMENT = {
    "id": "pay_test123",
    "order_id": "order_test123",
    "amount": 10000,
    "currency": "INR",
    "status": "captured",
    "method": "upi",
    "created_at": 1700000000,
    "error_code": None,
}


class FakePaymentAPI:
    def __init__(self, payment=None, error=None):
        self.payment = payment
        self.error = error

    def fetch(self, payment_id):
        if self.error:
            raise self.error
        return self.payment


class FakeRazorpayClient:
    payment_api = None
    utility_api = None

    def __init__(self, auth):
        self.payment = self.payment_api
        self.utility = self.utility_api


class RazorpayClientTests(unittest.TestCase):
    def setUp(self):
        self.credentials = patch.dict(
            os.environ,
            {"RAZORPAY_KEY_ID": "rzp_test_example", "RAZORPAY_KEY_SECRET": "not-printed"},
        )
        self.credentials.start()

    def tearDown(self):
        self.credentials.stop()

    def test_valid_payment_is_fetched_and_mapped(self):
        FakeRazorpayClient.payment_api = FakePaymentAPI(VALID_PAYMENT)
        fake_sdk = SimpleNamespace(Client=FakeRazorpayClient)
        with patch.dict("sys.modules", {"razorpay": fake_sdk}):
            payment = fetch_test_payment("pay_test123", "order_test123")
        transaction = payment_to_test_transaction(payment)
        self.assertEqual(transaction["payment_id"], "pay_test123")
        self.assertEqual(transaction["order_id"], "order_test123")
        self.assertEqual(transaction["amount"], 100.0)
        self.assertEqual(transaction["payment_method"], "upi")
        self.assertEqual(transaction["feature_coverage"], "Limited")
        self.assertEqual(transaction["feature_sources"]["amount"], "razorpay")
        self.assertEqual(transaction["feature_sources"]["avg_order_value"], "unavailable")

    def test_payment_from_wrong_order_is_rejected(self):
        FakeRazorpayClient.payment_api = FakePaymentAPI(VALID_PAYMENT)
        fake_sdk = SimpleNamespace(Client=FakeRazorpayClient)
        with patch.dict("sys.modules", {"razorpay": fake_sdk}):
            with self.assertRaisesRegex(RazorpayIntegrationError, "does not belong"):
                fetch_test_payment("pay_test123", "order_other")

    def test_missing_payment_fields_are_rejected(self):
        FakeRazorpayClient.payment_api = FakePaymentAPI({"id": "pay_test123"})
        fake_sdk = SimpleNamespace(Client=FakeRazorpayClient)
        with patch.dict("sys.modules", {"razorpay": fake_sdk}):
            with self.assertRaisesRegex(RazorpayIntegrationError, "missing required fields"):
                fetch_test_payment("pay_test123", "order_test123")

    def test_api_failure_is_sanitized(self):
        FakeRazorpayClient.payment_api = FakePaymentAPI(error=RuntimeError("secret response"))
        fake_sdk = SimpleNamespace(Client=FakeRazorpayClient)
        with patch.dict("sys.modules", {"razorpay": fake_sdk}):
            with self.assertRaisesRegex(RazorpayIntegrationError, "payment lookup failed") as context:
                fetch_test_payment("pay_test123", "order_test123")
        self.assertNotIn("secret response", str(context.exception))

    def test_valid_checkout_signature_is_verified(self):
        class Utility:
            def verify_payment_signature(self, data):
                self.data = data

        utility = Utility()
        FakeRazorpayClient.utility_api = utility
        fake_sdk = SimpleNamespace(Client=FakeRazorpayClient)
        with patch.dict("sys.modules", {"razorpay": fake_sdk}):
            from src.razorpay_client import verify_test_payment_signature

            verify_test_payment_signature("pay_test123", "order_test123", "signature")
        self.assertEqual(utility.data["razorpay_payment_id"], "pay_test123")
        self.assertEqual(utility.data["razorpay_order_id"], "order_test123")

    def test_invalid_checkout_signature_is_rejected(self):
        class Utility:
            def verify_payment_signature(self, data):
                raise RuntimeError("bad signature")

        FakeRazorpayClient.utility_api = Utility()
        fake_sdk = SimpleNamespace(Client=FakeRazorpayClient)
        with patch.dict("sys.modules", {"razorpay": fake_sdk}):
            from src.razorpay_client import verify_test_payment_signature

            with self.assertRaisesRegex(RazorpayIntegrationError, "signature verification failed"):
                verify_test_payment_signature("pay_test123", "order_test123", "signature")


if __name__ == "__main__":
    unittest.main()
