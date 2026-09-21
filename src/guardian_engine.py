"""Guardian orchestration layer for existing risk, ML, behavioral, and context results."""

from collections.abc import Iterable, Mapping
from typing import Any

from src.behavioral_analyzer import analyze_behavior
from src.context_analyzer import analyze_context
from src.guardian_explainer import generate_guardian_explanation


def _mapping(value: Any) -> dict[str, Any]:
    """Copy an optional mapping without mutating caller-owned result objects."""
    return dict(value) if isinstance(value, Mapping) else {}


def _ml_is_elevated(ml_result: Mapping[str, Any]) -> bool:
    """Recognize an existing ML review assessment without treating it as certainty."""
    decision = str(ml_result.get("decision", ml_result.get("ml_decision", ""))).upper()
    if decision == "REVIEW":
        return True
    score = ml_result.get("ml_risk_score", ml_result.get("risk_score", ml_result.get("score")))
    try:
        return float(score) >= 70
    except (TypeError, ValueError):
        return False


def _rule_state(rule_result: Mapping[str, Any]) -> str:
    """Map the pre-existing rule band to an aggregation state without changing it."""
    band = str(rule_result.get("risk_band", "")).upper()
    if "HIGH" in band:
        return "HIGH"
    if "MEDIUM" in band:
        return "PAUSE"
    return "LOW"


def _meaningful_signals(*signal_groups: Any) -> list[Mapping[str, Any]]:
    """Return supplied risk signals, excluding informational data-coverage notices."""
    signals: list[Mapping[str, Any]] = []
    for group in signal_groups:
        if not isinstance(group, Iterable) or isinstance(group, (str, bytes, Mapping)):
            continue
        for signal in group:
            if not isinstance(signal, Mapping):
                continue
            if str(signal.get("type", "")).upper() == "LIMITED_BEHAVIORAL_CONTEXT":
                continue
            signals.append(signal)
    return signals


def _signal_concept(signal: Mapping[str, Any]) -> str:
    """Map equivalent source signal types to one aggregation concept.

    This affects status escalation only. Every original source signal is still
    preserved for explanation and UI transparency.
    """
    signal_type = str(signal.get("type", "")).upper()
    concept_map = {
        "UNUSUAL_AMOUNT": "unusual_amount",
        "AMOUNT_UNUSUALLY_HIGH": "unusual_amount",
        "AMOUNT_VS_NORMAL_BEHAVIOR": "unusual_amount",
        "RECENT_FAILED_ATTEMPTS": "failed_attempts",
        "MULTIPLE_FAILED_ATTEMPTS": "failed_attempts",
        "FAILED_ATTEMPTS": "failed_attempts",
        "COUNTRY_MISMATCH": "country_location_change",
        "IP_COUNTRY_CHANGE": "country_location_change",
        "BILLING_COUNTRY_CHANGE": "country_location_change",
        "PAYMENT_METHOD_CHANGE": "payment_method_change",
        "DEVICE_TYPE_CHANGE": "device_change",
    }
    if signal_type in concept_map:
        return concept_map[signal_type]

    # The existing rule engine supplies reason/weight dictionaries without a
    # machine-readable type. Match only its established phrases so equivalent
    # legacy rule signals do not inflate the concept count.
    reason = str(signal.get("reason", "")).casefold()
    if "failed attempt" in reason:
        return "failed_attempts"
    if "billing country" in reason or "ip country" in reason:
        return "country_location_change"
    if "much higher" in reason or "large purchase" in reason or "transaction is large" in reason:
        return "unusual_amount"
    return signal_type.lower() or "unknown_signal"


def _distinct_signal_concepts(*signal_groups: Any) -> set[str]:
    """Return unique semantic concepts for aggregation, not presentation."""
    return {_signal_concept(signal) for signal in _meaningful_signals(*signal_groups)}


def _aggregate_status(
    rule_result: Mapping[str, Any],
    ml_result: Mapping[str, Any],
    behavioral_result: Mapping[str, Any],
    context_result: Mapping[str, Any],
) -> str:
    """Aggregate existing evidence with transparent, non-numeric escalation rules.

    A high rule band remains high. Otherwise, three or more available risk
    signals become HIGH; one or more signals, a medium rule band, or an
    elevated ML review assessment become PAUSE. Limited data alone never raises
    a status, and ML evidence alone never becomes HIGH.
    """
    existing_rule_state = _rule_state(rule_result)
    if existing_rule_state == "HIGH":
        return "HIGH"
    concepts = _distinct_signal_concepts(
        rule_result.get("signals"),
        behavioral_result.get("signals"),
        context_result.get("context_signals"),
    )
    if len(concepts) >= 3:
        return "HIGH"
    if existing_rule_state == "PAUSE" or concepts or _ml_is_elevated(ml_result):
        return "PAUSE"
    return "LOW"


def _merge_coverage(behavioral_result: Mapping[str, Any], context_result: Mapping[str, Any]) -> dict[str, Any]:
    """Expose analyzer coverage under stable namespaces for a future UI."""
    return {
        "behavioral": dict(behavioral_result.get("data_coverage", {})),
        "context": dict(context_result.get("data_coverage", {})),
        "behavioral_availability": behavioral_result.get("availability", "unavailable"),
        "context_availability": context_result.get("availability", "unavailable"),
    }


def _unique_limitations(*groups: Any) -> list[str]:
    """Combine limitation lists while preserving readable order."""
    result: list[str] = []
    for group in groups:
        if not isinstance(group, (list, tuple, set)):
            continue
        for item in group:
            text = str(item).strip()
            if text and text not in result:
                result.append(text)
    return result


def analyze_with_guardian(
    transaction: Mapping[str, Any] | None,
    rule_result: Mapping[str, Any] | None = None,
    ml_result: Mapping[str, Any] | None = None,
    historical_transactions: Iterable[Mapping[str, Any]] | None = None,
    payment_message: str | None = None,
    context: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Orchestrate existing analysis results into one transparent Guardian result.

    Supplied rule and ML results are preserved as-is. This function does not
    invoke training, invent missing data, or make payment approval, rejection,
    capture, blocking, or cancellation decisions.
    """
    current_transaction = _mapping(transaction)
    supplied_context = _mapping(context)
    context_transaction = {**current_transaction, **supplied_context}
    preserved_rule_result = _mapping(rule_result)
    preserved_ml_result = _mapping(ml_result)

    behavioral_result = analyze_behavior(current_transaction, historical_transactions)
    context_result = analyze_context(context_transaction, payment_message)
    status = _aggregate_status(
        preserved_rule_result,
        preserved_ml_result,
        behavioral_result,
        context_result,
    )
    risk_band = {"LOW": "LOW", "PAUSE": "MEDIUM", "HIGH": "HIGH"}[status]

    explanation_rule_result = {
        **preserved_rule_result,
        "guardian_state": risk_band,
    }
    explanation = generate_guardian_explanation(
        rule_result=explanation_rule_result,
        ml_result=preserved_ml_result,
        behavioral_result=behavioral_result,
        context_result=context_result,
    )
    limitations = _unique_limitations(
        behavioral_result.get("limitations"),
        context_result.get("limitations"),
        explanation.get("limitations"),
    )
    return {
        "status": status,
        "risk_band": risk_band,
        "headline": explanation["headline"],
        "summary": explanation["summary"],
        "signals": explanation["signals"],
        "recommendation": explanation["recommendation"],
        "action_steps": explanation["action_steps"],
        "technical_notes": explanation["technical_notes"],
        "rule_result": preserved_rule_result,
        "ml_result": preserved_ml_result,
        "behavioral_result": behavioral_result,
        "context_result": context_result,
        "explanation": explanation,
        "data_coverage": _merge_coverage(behavioral_result, context_result),
        "limitations": limitations,
    }
