"""Data provenance and payment-context analysis for Guardian Engine integration."""

from collections.abc import Mapping
from typing import Any


_IMPORTANT_BEHAVIORAL_FIELDS = {
    "customer_id",
    "device_type",
    "ip_country",
    "billing_country",
    "failed_attempts",
    "account_age_days",
    "previous_order_count",
    "avg_order_value",
    "is_new_customer",
}

_MESSAGE_PATTERNS = {
    "URGENCY_DETECTED": (
        ("urgent", "immediately", "right now", "act now", "today only", "asap"),
        {
            "title": "Urgency detected",
            "description": "The payment message pressures you to act immediately.",
            "why_it_matters": "Urgency does not prove fraud, but pressure can be a reason to verify the request independently.",
            "recommended_action": "Verify the request independently before paying.",
            "severity": "medium",
        },
    ),
    "ACCOUNT_THREAT_DETECTED": (
        (
            "account will be blocked", "account will be suspended", "account will be closed",
            "account will be locked", "access will be removed", "account may be blocked",
            "account may be suspended", "account may be locked",
        ),
        {
            "title": "Account threat detected",
            "description": "The message claims your account may be blocked, suspended, closed, or locked.",
            "why_it_matters": "Threatening consequences can pressure users into making rushed payments.",
            "recommended_action": "Contact the organization through an official channel before paying.",
            "severity": "medium",
        },
    ),
    "PRESSURE_TO_ACT_DETECTED": (
        ("don't delay", "do not delay", "must pay now", "pay immediately", "final warning", "last chance"),
        {
            "title": "Pressure to act detected",
            "description": "The payment message uses language intended to rush a decision.",
            "why_it_matters": "Pressure does not prove fraud, but it can make independent verification more important.",
            "recommended_action": "Pause and verify the request through a trusted channel.",
            "severity": "medium",
        },
    ),
    "UNUSUAL_PAYMENT_INSTRUCTION": (
        (
            "bypass verification", "do not contact support", "don't contact support",
            "send to this new account", "use this different payment method",
        ),
        {
            "title": "Unusual payment instruction",
            "description": "The message asks you to bypass normal verification or change payment instructions.",
            "why_it_matters": "Requests to bypass normal checks deserve independent verification.",
            "recommended_action": "Use an official, trusted channel to confirm the payment instructions.",
            "severity": "medium",
        },
    ),
}


def _clean_message(value: Any) -> str | None:
    """Preserve optional supplied text for later analysis without interpreting it."""
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def analyze_payment_message(message: str | None) -> list[dict[str, str]]:
    """Return deterministic contextual warnings found in an optional message.

    Matching is case-insensitive and category-based: several phrases in the
    same category result in one signal. A message signal is not a fraud claim.
    """
    cleaned = _clean_message(message)
    if cleaned is None:
        return []
    normalized = cleaned.casefold()
    signals = []
    for signal_type, (patterns, details) in _MESSAGE_PATTERNS.items():
        if any(pattern in normalized for pattern in patterns):
            signals.append({"type": signal_type, **details})
    return signals


def analyze_context(
    transaction: Mapping[str, Any] | None,
    payment_message: str | None = None,
) -> dict[str, Any]:
    """Describe field provenance and preserve optional payment context.

    This function does not perform message/LLM analysis. It distinguishes
    provided payment facts from compatibility defaults and missing data so
    downstream decisions do not mistake placeholder values for low-risk facts.
    """
    current = transaction if isinstance(transaction, Mapping) else {}
    raw_sources = current.get("feature_sources", {})
    sources = dict(raw_sources) if isinstance(raw_sources, Mapping) else {}
    unavailable = current.get("unavailable_fraud_features", [])
    unavailable_fields = {str(field) for field in unavailable} if isinstance(unavailable, (list, tuple, set)) else set()
    compatibility_defaults = bool(current.get("compatibility_values_are_not_payment_facts", False))
    declared_coverage = str(current.get("feature_coverage", "")).strip().lower()

    coverage: dict[str, str] = {}
    for field in sorted(_IMPORTANT_BEHAVIORAL_FIELDS):
        source = str(sources.get(field, "")).strip().lower()
        if field in unavailable_fields or source == "unavailable":
            coverage[field] = "unavailable"
        elif compatibility_defaults or source in {"merchant/demo", "demo", "compatibility"}:
            coverage[field] = "demo_or_compatibility_default"
        elif field in current:
            coverage[field] = "available"
        else:
            coverage[field] = "unavailable"

    limitations: list[str] = []
    context_signals: list[dict[str, str]] = []
    unavailable_important = [field for field, state in coverage.items() if state == "unavailable"]
    defaulted_important = [field for field, state in coverage.items() if state == "demo_or_compatibility_default"]
    if unavailable_important or defaulted_important or declared_coverage == "limited":
        limitations.append("Important behavioral information is limited or unavailable; unavailable/defaulted fields are not evidence of low risk.")
        context_signals.append({
            "type": "LIMITED_BEHAVIORAL_CONTEXT",
            "severity": "informational",
            "reason": "Guardian analysis has limited payment/customer context for this transaction.",
        })
    if compatibility_defaults:
        limitations.append("Compatibility fallback values are not payment facts and must not be used as behavioral evidence.")

    message = _clean_message(payment_message)
    if message is None:
        message = _clean_message(current.get("payment_message") or current.get("context_message"))
    context_signals.extend(analyze_payment_message(message))

    availability = "available" if not unavailable_important and not defaulted_important else ("limited" if any(state == "available" for state in coverage.values()) else "unavailable")
    return {
        "context_signals": context_signals,
        "message": message,
        "message_analysis": "deterministic_keyword_matching" if message else "not_performed",
        "availability": availability,
        "data_coverage": coverage,
        "limitations": limitations,
        "sources": sources,
        "payment_context": {
            "payment_id": current.get("payment_id"),
            "order_id": current.get("order_id"),
            "status": current.get("status") or current.get("razorpay_order_status"),
            "currency": current.get("currency"),
            "feature_coverage": current.get("feature_coverage"),
        },
    }
