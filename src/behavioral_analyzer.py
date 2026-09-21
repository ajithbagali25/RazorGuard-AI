"""Conservative behavioral analysis for Guardian Engine integration.

The analyzer only describes behavior supported by the supplied transaction and
history. It never infers customer history from compatibility defaults.
"""

from collections.abc import Iterable, Mapping
from typing import Any


_HISTORY_FIELDS = ("customer_id", "recipient_id", "recipient", "transaction_id")
_COMPARE_FIELDS = ("payment_method", "device_type", "ip_country", "billing_country")


def _value(row: Mapping[str, Any], name: str) -> Any:
    """Return a usable value, treating empty strings and unavailable markers as absent."""
    value = row.get(name)
    if isinstance(value, str) and value.strip().upper() in {"", "UNKNOWN", "UNAVAILABLE", "TEST_UNKNOWN"}:
        return None
    return value


def _number(row: Mapping[str, Any], name: str) -> float | None:
    """Safely read a positive numeric field from one row."""
    try:
        value = float(_value(row, name))
    except (TypeError, ValueError):
        return None
    return value if value >= 0 else None


def _as_history_rows(rows: Iterable[Mapping[str, Any]] | None) -> list[Mapping[str, Any]]:
    """Normalize optional history without trusting malformed entries."""
    if rows is None or isinstance(rows, (str, bytes, Mapping)):
        return []
    try:
        return [row for row in rows if isinstance(row, Mapping)]
    except TypeError:
        return []


def _same_party(current: Mapping[str, Any], historical: Mapping[str, Any]) -> bool | None:
    """Compare customer/recipient identity when a comparable identity is present."""
    for field in ("recipient_id", "recipient", "customer_id"):
        current_value = _value(current, field)
        historical_value = _value(historical, field)
        if current_value is not None and historical_value is not None:
            return str(current_value) == str(historical_value)
    return None


def analyze_behavior(
    transaction: Mapping[str, Any] | None,
    historical_transactions: Iterable[Mapping[str, Any]] | None = None,
) -> dict[str, Any]:
    """Return structured behavioral evidence for a current transaction.

    ``historical_transactions`` is optional. When it is not supplied, the
    result reports unavailable history rather than guessing that a customer or
    recipient is new. Compatibility/default values marked by the payment
    adapter are excluded from behavioral claims.
    """
    current = transaction if isinstance(transaction, Mapping) else {}
    history = _as_history_rows(historical_transactions)
    signals: list[dict[str, str]] = []
    limitations: list[str] = []
    coverage: dict[str, str] = {
        "amount": "unavailable",
        "average_order_value": "unavailable",
        "transaction_history": "unavailable",
        "payment_method_history": "unavailable",
        "device_history": "unavailable",
        "country_history": "unavailable",
    }

    feature_sources = current.get("feature_sources", {})
    compatibility_defaults = bool(current.get("compatibility_values_are_not_payment_facts", False))
    if not isinstance(feature_sources, Mapping):
        feature_sources = {}

    amount = _number(current, "amount")
    average = _number(current, "avg_order_value")
    amount_source = str(feature_sources.get("amount", "available")).lower()
    average_source = str(feature_sources.get("avg_order_value", "available")).lower()
    amount_is_available = amount is not None and amount_source not in {"unavailable", "merchant/demo"}
    average_is_available = (
        average is not None
        and average > 0
        and average_source not in {"unavailable", "merchant/demo"}
        and not compatibility_defaults
    )
    coverage["amount"] = "available" if amount_is_available else "unavailable"
    coverage["average_order_value"] = "available" if average_is_available else "unavailable"

    ratio: float | None = None
    if amount_is_available and average_is_available:
        ratio = round(amount / average, 2)
        if ratio >= 2.5:
            signals.append({
                "type": "UNUSUAL_AMOUNT",
                "severity": "elevated",
                "reason": f"This payment is approximately {ratio:.1f}× the recent average.",
            })
    else:
        limitations.append("Average-order history is unavailable, so unusual-amount behavior cannot be assessed.")

    comparable_history = [row for row in history if _same_party(current, row) is not None]
    if comparable_history:
        coverage["transaction_history"] = "available"
        matching_history = [row for row in comparable_history if _same_party(current, row)]
        recipient_familiarity = "FAMILIAR" if matching_history else "NEW"
        if recipient_familiarity == "NEW":
            signals.append({
                "type": "NEW_RECIPIENT_OR_CUSTOMER",
                "severity": "informational",
                "reason": "No prior transaction was found for this recipient/customer in the supplied history.",
            })
    else:
        recipient_familiarity = "UNKNOWN"
        limitations.append("No comparable recipient/customer history was supplied.")

    previous_order_count = _number(current, "previous_order_count")
    if previous_order_count is not None and not compatibility_defaults:
        if previous_order_count == 0:
            signals.append({
                "type": "NO_PREVIOUS_ORDERS",
                "severity": "informational",
                "reason": "The available transaction record reports no previous orders.",
            })
    else:
        limitations.append("Previous-order count is unavailable or is a compatibility default.")

    failed_attempts = _number(current, "failed_attempts")
    if failed_attempts is not None and not compatibility_defaults and failed_attempts >= 1:
        signals.append({
            "type": "RECENT_FAILED_ATTEMPTS",
            "severity": "elevated" if failed_attempts >= 3 else "informational",
            "reason": f"{int(failed_attempts)} recent failed payment attempt(s) were reported.",
        })

    for field in _COMPARE_FIELDS:
        current_value = _value(current, field)
        historical_values = {_value(row, field) for row in comparable_history if _value(row, field) is not None}
        coverage_name = {
            "payment_method": "payment_method_history",
            "device_type": "device_history",
            "ip_country": "country_history",
            "billing_country": "country_history",
        }[field]
        if current_value is not None and historical_values:
            coverage[coverage_name] = "available"
            if current_value not in historical_values:
                label = field.replace("_", " ")
                signals.append({
                    "type": f"{field.upper()}_CHANGE",
                    "severity": "informational",
                    "reason": f"The current {label} differs from the supplied transaction history.",
                })

    history_available = coverage["transaction_history"] == "available"
    availability = "available" if history_available and not limitations else ("limited" if any(coverage[name] == "available" for name in coverage) else "unavailable")
    return {
        "signals": signals,
        "recipient_familiarity": recipient_familiarity,
        "amount_to_average_ratio": ratio,
        "history_available": history_available,
        "availability": availability,
        "data_coverage": coverage,
        "limitations": list(dict.fromkeys(limitations)),
    }
