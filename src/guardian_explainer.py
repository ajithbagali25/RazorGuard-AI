"""Deterministic, human-readable explanations for Guardian Engine signals."""

from collections.abc import Mapping
from typing import Any


_SIGNAL_COPY = {
    "NEW_RECIPIENT_OR_CUSTOMER": {
        "title": "New recipient",
        "description": "You haven't paid this recipient before in the available history.",
        "why_it_matters": "New recipients deserve additional verification, especially for higher-value payments.",
        "recommended_action": "Verify the recipient using a trusted contact method.",
    },
    "UNUSUAL_AMOUNT": {
        "title": "Unusual amount",
        "description": "This payment is significantly higher than the available recent average.",
        "why_it_matters": "A payment outside normal behavior can merit an extra check.",
        "recommended_action": "Confirm the amount and payment purpose before continuing.",
    },
    "NO_PREVIOUS_ORDERS": {
        "title": "Limited purchase history",
        "description": "The available record reports no previous orders.",
        "why_it_matters": "First-time activity has less history available for comparison.",
        "recommended_action": "Verify the payment purpose before continuing.",
    },
    "RECENT_FAILED_ATTEMPTS": {
        "title": "Multiple failed attempts",
        "description": "Several recent payment attempts were unsuccessful.",
        "why_it_matters": "Repeated failures can indicate a payment or account issue that needs checking.",
        "recommended_action": "Pause and verify why the earlier attempts failed.",
    },
    "IP_COUNTRY_CHANGE": {
        "title": "Location change",
        "description": "The current IP country differs from the supplied transaction history.",
        "why_it_matters": "A location change can be normal, but it is useful context for verification.",
        "recommended_action": "Verify the payment and account details before continuing.",
    },
    "BILLING_COUNTRY_CHANGE": {
        "title": "Billing location change",
        "description": "The current billing country differs from the supplied transaction history.",
        "why_it_matters": "A billing-location change can deserve additional confirmation.",
        "recommended_action": "Verify the payment and account details before continuing.",
    },
    "PAYMENT_METHOD_CHANGE": {
        "title": "Payment method change",
        "description": "The current payment method differs from the supplied transaction history.",
        "why_it_matters": "A new payment method can be legitimate, but it provides a reason to confirm the transaction.",
        "recommended_action": "Confirm the payment method with the account holder.",
    },
    "DEVICE_TYPE_CHANGE": {
        "title": "Device change",
        "description": "The current device type differs from the supplied transaction history.",
        "why_it_matters": "A device change can be normal, but it is useful verification context.",
        "recommended_action": "Confirm that the account holder recognizes this payment.",
    },
    "COUNTRY_MISMATCH": {
        "title": "Location mismatch",
        "description": "The available payment information shows a mismatch between IP and billing country.",
        "why_it_matters": "A location mismatch does not prove fraud, but it can be a reason to verify details.",
        "recommended_action": "Verify the payment and account details before continuing.",
    },
    "URGENCY_DETECTED": {
        "title": "Urgency detected",
        "description": "The supplied payment message pressures you to act immediately.",
        "why_it_matters": "Urgency does not prove fraud, but pressure can be a reason to verify the request independently.",
        "recommended_action": "Do not rely only on the urgent message. Verify the request independently.",
    },
    "ACCOUNT_THREAT_DETECTED": {
        "title": "Account threat detected",
        "description": "The message claims your account may be blocked, suspended, closed, or locked.",
        "why_it_matters": "Threatening consequences can pressure users into making rushed payments.",
        "recommended_action": "Contact the organization through an official channel before paying.",
    },
    "PRESSURE_TO_ACT_DETECTED": {
        "title": "Pressure to act detected",
        "description": "The payment message uses language intended to rush a decision.",
        "why_it_matters": "Pressure does not prove fraud, but it can make independent verification more important.",
        "recommended_action": "Pause and verify the request through a trusted channel.",
    },
    "UNUSUAL_PAYMENT_INSTRUCTION": {
        "title": "Unusual payment instruction",
        "description": "The message asks you to bypass normal verification or change payment instructions.",
        "why_it_matters": "Requests to bypass normal checks deserve independent verification.",
        "recommended_action": "Use an official, trusted channel to confirm the payment instructions.",
    },
}


def _mapping(value: Any) -> Mapping[str, Any]:
    """Return a mapping or an empty mapping for optional result objects."""
    return value if isinstance(value, Mapping) else {}


def _severity(value: Any) -> str:
    """Normalize source severities to the public low/medium/high vocabulary."""
    text = str(value or "").strip().lower()
    if text in {"high", "elevated", "critical"}:
        return "high"
    if text in {"medium", "informational", "warning"}:
        return "medium"
    return "low"


def _risk_state(rule_result: Mapping[str, Any]) -> str:
    """Read an already-established state; use the rule band only as a fallback."""
    state = str(
        rule_result.get("guardian_state")
        or rule_result.get("risk_state")
        or rule_result.get("risk_band")
        or ""
    ).strip().upper()
    if "HIGH" in state or state in {"REJECT", "ESCALATE"}:
        return "HIGH"
    if "MEDIUM" in state or state in {"REVIEW", "PAUSE"}:
        return "MEDIUM"
    return "LOW"


def _copy_signal(signal: Mapping[str, Any]) -> dict[str, str]:
    """Translate a structured signal without calculating a new fraud score."""
    signal_type = str(signal.get("type", "RULE_SIGNAL")).upper()
    template = _SIGNAL_COPY.get(signal_type)
    if template is None:
        reason = str(signal.get("reason") or signal.get("description") or "An available risk signal needs review.")
        template = {
            "title": "Risk signal",
            "description": reason,
            "why_it_matters": "This signal is available for human review.",
            "recommended_action": "Review the available transaction details before proceeding.",
        }
    return {
        "type": signal_type,
        **template,
        "severity": _severity(signal.get("severity")),
    }


def _append_unique(target: list[str], values: Any) -> None:
    """Append non-empty textual limitations or notes while preserving order."""
    if not isinstance(values, (list, tuple, set)):
        return
    for value in values:
        text = str(value).strip()
        if text and text not in target:
            target.append(text)


def generate_guardian_explanation(
    rule_result: Mapping[str, Any] | None = None,
    ml_result: Mapping[str, Any] | None = None,
    behavioral_result: Mapping[str, Any] | None = None,
    context_result: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Convert existing Guardian signals into deterministic user guidance.

    The explainer does not score transactions or override the rule engine. It
    merely converts supplied results into plain language and exposes data gaps.
    """
    rules = _mapping(rule_result)
    ml = _mapping(ml_result)
    behavioral = _mapping(behavioral_result)
    context = _mapping(context_result)
    state = _risk_state(rules)
    signals: list[dict[str, str]] = []
    technical_notes: list[str] = []
    limitations: list[str] = []

    raw_rule_signals = rules.get("signals", [])
    if isinstance(raw_rule_signals, (list, tuple)):
        signals.extend(_copy_signal(item) for item in raw_rule_signals if isinstance(item, Mapping))
    raw_behavioral_signals = behavioral.get("signals", [])
    if isinstance(raw_behavioral_signals, (list, tuple)):
        signals.extend(_copy_signal(item) for item in raw_behavioral_signals if isinstance(item, Mapping))
    raw_context_signals = context.get("context_signals", [])
    if isinstance(raw_context_signals, (list, tuple)):
        signals.extend(_copy_signal(item) for item in raw_context_signals if isinstance(item, Mapping))

    if behavioral.get("recipient_familiarity") == "UNKNOWN":
        _append_unique(limitations, ["Behavioral history is unavailable for this payment."])
    _append_unique(limitations, behavioral.get("limitations"))
    _append_unique(limitations, context.get("limitations"))

    if "risk_score" in rules:
        technical_notes.append(f"Rule score: {rules['risk_score']}.")
    if rules.get("risk_band"):
        technical_notes.append(f"Rule risk band: {rules['risk_band']}.")
    ml_score = ml.get("ml_risk_score", ml.get("risk_score", ml.get("score")))
    if ml_score is not None:
        technical_notes.append(f"ML Risk Score: {ml_score}/100. This is an uncalibrated model score, not a probability.")
    if ml.get("decision"):
        technical_notes.append(f"ML decision: {ml['decision']}.")
    if str(context.get("payment_context", {}).get("feature_coverage", "")).lower() == "limited":
        technical_notes.append("Razorpay Test Mode context has limited fraud-analysis feature coverage.")

    if state == "HIGH":
        headline = "High-risk payment"
        summary = "Several signals indicate that this payment deserves immediate verification."
        recommendation = "Verify this payment before proceeding."
        action_steps = ["Pause the payment review.", "Verify the recipient and payment purpose through a trusted channel."]
    elif state == "MEDIUM":
        headline = "Pause and verify"
        summary = "This payment has some unusual characteristics in the available data."
        recommendation = "Verify this payment before proceeding."
        action_steps = ["Confirm the amount and recipient.", "Review the available risk signals before continuing."]
    else:
        headline = "Looks normal"
        summary = "We didn't find significant unusual signals in the available data."
        recommendation = "Continue with normal payment checks."
        action_steps = ["Use normal payment verification practices."]

    for signal in signals:
        action = signal["recommended_action"]
        if action not in action_steps:
            action_steps.append(action)

    return {
        "headline": headline,
        "summary": summary,
        "signals": signals,
        "recommendation": recommendation,
        "action_steps": action_steps,
        "technical_notes": technical_notes,
        "limitations": limitations,
    }
