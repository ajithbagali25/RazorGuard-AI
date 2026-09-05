"""Feature preparation and inference helpers for the saved RazorGuard v2 model."""

from typing import Mapping, Any

import pandas as pd


ML_FEATURES = [
    "amount",
    "failed_attempts",
    "account_age_days",
    "previous_order_count",
    "avg_order_value",
    "is_new_customer",
    "country_mismatch",
    "amount_to_avg_ratio",
    "payment_method",
    "device_type",
    "ip_country",
    "billing_country",
]


def _as_binary(value: Any) -> int:
    """Normalize boolean-like CSV and pandas values to the model's 0/1 format."""
    if isinstance(value, str):
        return int(value.strip().lower() in {"true", "1", "yes"})
    return int(bool(value))


def prepare_ml_features(transaction: Mapping[str, Any]) -> pd.DataFrame:
    """Build the exact permitted v2 model inputs for one dashboard transaction."""
    amount = float(transaction["amount"])
    average_order_value = float(transaction["avg_order_value"])
    ip_country = str(transaction["ip_country"]).strip()
    billing_country = str(transaction["billing_country"]).strip()

    # The v2 generator stored this feature rounded to four decimal places.
    amount_to_average_ratio = round(amount / average_order_value, 4) if average_order_value > 0 else 0.0
    row = {
        "amount": amount,
        "failed_attempts": int(transaction["failed_attempts"]),
        "account_age_days": int(transaction["account_age_days"]),
        "previous_order_count": int(transaction["previous_order_count"]),
        "avg_order_value": average_order_value,
        "is_new_customer": _as_binary(transaction["is_new_customer"]),
        "country_mismatch": int(ip_country != billing_country),
        "amount_to_avg_ratio": amount_to_average_ratio,
        "payment_method": str(transaction["payment_method"]).strip(),
        "device_type": str(transaction["device_type"]).strip(),
        "ip_country": ip_country,
        "billing_country": billing_country,
    }
    return pd.DataFrame([row], columns=ML_FEATURES)


def get_ml_risk_score(model, transaction: Mapping[str, Any]) -> int:
    """Return an uncalibrated model risk score on a 0-100 scale."""
    features = prepare_ml_features(transaction)
    return int(round(float(model.predict_proba(features)[0, 1]) * 100))
