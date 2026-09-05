"""Minimal Razorpay Test Mode order creation client."""

from decimal import Decimal, InvalidOperation
import os
from typing import Any, Mapping

from dotenv import load_dotenv

load_dotenv()


class RazorpayIntegrationError(Exception):
    """Expected configuration, validation, or API error from Razorpay integration."""


def _amount_in_paise(amount: Any) -> int:
    try:
        value = Decimal(str(amount))
    except (InvalidOperation, ValueError, TypeError) as error:
        raise RazorpayIntegrationError("Amount must be a positive number.") from error

    if not value.is_finite() or value <= 0:
        raise RazorpayIntegrationError("Amount must be a positive number.")

    paise = (value * 100).quantize(Decimal("1"))
    if paise < 100:
        raise RazorpayIntegrationError("Amount must be at least 1.00 INR.")
    return int(paise)


def create_test_order(amount: Any, currency: str, receipt: str) -> dict[str, Any]:
    """Create one Razorpay Test Mode order; amount is specified in major units."""
    amount_in_paise = _amount_in_paise(amount)
    currency = str(currency).strip().upper()
    receipt = str(receipt).strip()
    if not currency or not receipt:
        raise RazorpayIntegrationError("Currency and receipt are required.")

    key_id = os.getenv("RAZORPAY_KEY_ID", "").strip()
    key_secret = os.getenv("RAZORPAY_KEY_SECRET", "").strip()
    if not key_id or not key_secret:
        raise RazorpayIntegrationError(
            "Razorpay Test Mode credentials are missing. Set RAZORPAY_KEY_ID and RAZORPAY_KEY_SECRET."
        )
    if not key_id.startswith("rzp_test_"):
        raise RazorpayIntegrationError("Only Razorpay Test Mode keys are accepted.")

    try:
        import razorpay
    except ImportError as error:
        raise RazorpayIntegrationError("The Razorpay Python SDK is not installed.") from error

    order_data = {
        "amount": amount_in_paise,
        "currency": currency,
        "receipt": receipt,
        "payment_capture": 1,
    }
    try:
        client = razorpay.Client(auth=(key_id, key_secret))
        return client.order.create(order_data)
    except Exception as error:
        raise RazorpayIntegrationError("Razorpay Test Mode order creation failed.") from error


def fetch_test_order(order_id: str) -> dict[str, Any]:
    """Fetch one Razorpay Test Mode order by ID without exposing credentials."""
    order_id = str(order_id).strip()
    if not order_id or not order_id.startswith("order_"):
        raise RazorpayIntegrationError("Enter a valid Razorpay order ID.")

    key_id = os.getenv("RAZORPAY_KEY_ID", "").strip()
    key_secret = os.getenv("RAZORPAY_KEY_SECRET", "").strip()
    if not key_id or not key_secret:
        raise RazorpayIntegrationError(
            "Razorpay Test Mode credentials are missing. Set RAZORPAY_KEY_ID and RAZORPAY_KEY_SECRET."
        )
    if not key_id.startswith("rzp_test_"):
        raise RazorpayIntegrationError("Only Razorpay Test Mode keys are accepted.")

    try:
        import razorpay
    except ImportError as error:
        raise RazorpayIntegrationError("The Razorpay Python SDK is not installed.") from error

    try:
        client = razorpay.Client(auth=(key_id, key_secret))
        order = client.order.fetch(order_id)
    except Exception as error:
        raise RazorpayIntegrationError("Razorpay Test Mode order lookup failed.") from error

    required_fields = {"id", "amount", "currency"}
    if not isinstance(order, Mapping) or not required_fields.issubset(order):
        raise RazorpayIntegrationError("Razorpay order response is missing required fields.")
    return order


def order_to_test_transaction(order: dict[str, Any]) -> dict[str, Any]:
    """Map order-only data to the existing scorer shape with explicit demo fields."""
    required_fields = {"id", "amount", "currency"}
    if not required_fields.issubset(order):
        raise RazorpayIntegrationError("Razorpay order is missing required fields.")
    try:
        amount = float(order["amount"]) / 100
    except (TypeError, ValueError) as error:
        raise RazorpayIntegrationError("Razorpay order amount is invalid.") from error

    return {
        "transaction_id": order["id"],
        "customer_id": "TEST_UNKNOWN",
        "amount": amount,
        "currency": str(order["currency"]).upper(),
        "payment_method": "TEST_UNKNOWN",
        "device_type": "TEST_UNKNOWN",
        "ip_country": "UNKNOWN",
        "billing_country": "UNKNOWN",
        "failed_attempts": 0,
        "account_age_days": 365,
        "previous_order_count": 1,
        "avg_order_value": amount,
        "is_new_customer": False,
        "razorpay_order_status": order.get("status", "UNKNOWN"),
        "demo_default_fields": [
            "customer_id", "payment_method", "device_type", "ip_country",
            "billing_country", "failed_attempts", "account_age_days",
            "previous_order_count", "avg_order_value", "is_new_customer",
        ],
    }


def fetch_test_payment(payment_id: str, expected_order_id: str) -> dict[str, Any]:
    """Fetch a Test Mode payment and verify it belongs to the expected order."""
    payment_id = str(payment_id).strip()
    expected_order_id = str(expected_order_id).strip()
    if not payment_id.startswith("pay_"):
        raise RazorpayIntegrationError("Enter a valid Razorpay payment ID.")
    if not expected_order_id.startswith("order_"):
        raise RazorpayIntegrationError("Enter a valid Razorpay order ID.")

    key_id = os.getenv("RAZORPAY_KEY_ID", "").strip()
    key_secret = os.getenv("RAZORPAY_KEY_SECRET", "").strip()
    if not key_id or not key_secret:
        raise RazorpayIntegrationError(
            "Razorpay Test Mode credentials are missing. Set RAZORPAY_KEY_ID and RAZORPAY_KEY_SECRET."
        )
    if not key_id.startswith("rzp_test_"):
        raise RazorpayIntegrationError("Only Razorpay Test Mode keys are accepted.")

    try:
        import razorpay
    except ImportError as error:
        raise RazorpayIntegrationError("The Razorpay Python SDK is not installed.") from error

    try:
        client = razorpay.Client(auth=(key_id, key_secret))
        payment = client.payment.fetch(payment_id)
    except Exception as error:
        raise RazorpayIntegrationError("Razorpay Test Mode payment lookup failed.") from error

    required_fields = {"id", "order_id", "amount", "currency", "status"}
    if not isinstance(payment, Mapping) or not required_fields.issubset(payment):
        raise RazorpayIntegrationError("Razorpay payment response is missing required fields.")
    if payment["order_id"] != expected_order_id:
        raise RazorpayIntegrationError("Razorpay payment does not belong to the expected order.")
    return payment


def verify_test_payment_signature(payment_id: str, order_id: str, signature: str) -> None:
    """Verify Checkout's signature server-side using the Test Mode secret."""
    if not payment_id.startswith("pay_") or not order_id.startswith("order_") or not signature:
        raise RazorpayIntegrationError("Razorpay Checkout returned invalid payment details.")

    key_id = os.getenv("RAZORPAY_KEY_ID", "").strip()
    key_secret = os.getenv("RAZORPAY_KEY_SECRET", "").strip()
    if not key_id or not key_secret:
        raise RazorpayIntegrationError(
            "Razorpay Test Mode credentials are missing. Set RAZORPAY_KEY_ID and RAZORPAY_KEY_SECRET."
        )
    if not key_id.startswith("rzp_test_"):
        raise RazorpayIntegrationError("Only Razorpay Test Mode keys are accepted.")

    try:
        import razorpay
    except ImportError as error:
        raise RazorpayIntegrationError("The Razorpay Python SDK is not installed.") from error

    try:
        client = razorpay.Client(auth=(key_id, key_secret))
        client.utility.verify_payment_signature({
            "razorpay_payment_id": payment_id,
            "razorpay_order_id": order_id,
            "razorpay_signature": signature,
        })
    except Exception as error:
        raise RazorpayIntegrationError("Razorpay payment signature verification failed.") from error


def payment_to_test_transaction(payment: Mapping[str, Any]) -> dict[str, Any]:
    """Map payment data to the existing scorer shape with provenance metadata."""
    required_fields = {"id", "order_id", "amount", "currency", "status"}
    if not isinstance(payment, Mapping) or not required_fields.issubset(payment):
        raise RazorpayIntegrationError("Razorpay payment is missing required fields.")
    try:
        amount = float(payment["amount"]) / 100
    except (TypeError, ValueError) as error:
        raise RazorpayIntegrationError("Razorpay payment amount is invalid.") from error

    payment_fields = {
        "payment_id": payment["id"],
        "order_id": payment["order_id"],
        "amount": amount,
        "currency": str(payment["currency"]).upper(),
        "status": payment["status"],
        "payment_method": payment.get("method") or "UNAVAILABLE",
        "payment_timestamp": payment.get("created_at"),
        "payment_error_code": payment.get("error_code"),
        "payment_error_description": payment.get("error_description"),
        "payment_error_reason": payment.get("error_reason"),
        "payment_error_source": payment.get("error_source"),
        "payment_error_step": payment.get("error_step"),
        "payment_error_type": payment.get("error_type"),
    }
    feature_sources = {
        "payment_id": "razorpay",
        "order_id": "razorpay",
        "amount": "razorpay",
        "currency": "razorpay",
        "status": "razorpay",
        "payment_method": "razorpay" if payment.get("method") else "unavailable",
        "payment_timestamp": "razorpay" if payment.get("created_at") else "unavailable",
        "payment_failure_information": "razorpay" if payment.get("error_code") else "unavailable",
        "country_mismatch": "unavailable",
        "amount_to_avg_ratio": "unavailable",
        "customer_id": "unavailable",
        "device_type": "unavailable",
        "ip_country": "unavailable",
        "billing_country": "unavailable",
        "failed_attempts": "unavailable",
        "account_age_days": "unavailable",
        "previous_order_count": "unavailable",
        "avg_order_value": "unavailable",
        "is_new_customer": "unavailable",
    }
    compatibility_values = {
        "customer_id": "UNAVAILABLE",
        "device_type": "UNAVAILABLE",
        "ip_country": "UNAVAILABLE",
        "billing_country": "UNAVAILABLE",
        "failed_attempts": 0,
        "account_age_days": 365,
        "previous_order_count": 0,
        "avg_order_value": amount,
        "is_new_customer": False,
    }
    transaction = {
        **payment_fields,
        **compatibility_values,
        "transaction_id": payment["id"],
        "feature_sources": feature_sources,
        "feature_coverage": "Limited",
        "unavailable_fraud_features": [
            name for name, source in feature_sources.items() if source == "unavailable"
        ],
        "compatibility_fallback_sources": {
            name: "merchant/demo" for name in compatibility_values
        },
        "derived_fields": ["amount"],
        "compatibility_values_are_not_payment_facts": True,
    }
    return transaction
