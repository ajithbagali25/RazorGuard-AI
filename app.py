from pathlib import Path
import os

import joblib
import pandas as pd
import streamlit as st
import streamlit.components.v1 as components

from src.ml_inference import get_ml_risk_score
from src.guardian_engine import analyze_with_guardian
from src.razorpay_client import (
    RazorpayIntegrationError,
    create_test_order,
    fetch_test_order,
    fetch_test_payment,
    order_to_test_transaction,
    payment_to_test_transaction,
    verify_test_payment_signature,
)
from src.risk_engine import score_transaction, score_transactions

DATA_PATH = "data/mock_transactions.csv"
ML_MODEL_PATH = Path("models/razorguard_model_v2.joblib")
ML_REVIEW_THRESHOLD = 70
GUARDIAN_HIGH_VALUE_AMOUNT = 50000.0
GUARDIAN_UNKNOWN_SUSPICIOUS_AMOUNT = 10.0
GUARDIAN_NEW_SUSPICIOUS_MAX_AMOUNT = 10000.0
GUARDIAN_FAMILIAR_SAFE_MAX_AMOUNT = 2000000.0
RISKORA_COMMUNITY_REVIEW_MIN_AMOUNT = 700.0
RISK_BANDS = ["Low Risk", "Medium Risk", "High Risk"]
RECOMMENDATIONS = {
    "Low Risk": "APPROVE",
    "Medium Risk": "REVIEW",
    "High Risk": "REJECT",
}

DEMO_SCENARIOS = {
    "Normal Payment": {
        "form": {
            "amount": 500.0,
            "recipient_name": "Known Recipient",
            "recipient_identifier": "demo-known-recipient",
            "purpose": "Shopping",
            "familiarity": "Familiar",
            "message": "Payment for my regular purchase.",
        },
        "transaction": {
            "transaction_id": "DEMO-NORMAL-001", "customer_id": "DEMO-CUSTOMER-001",
            "amount": 500.0, "payment_method": "UPI", "device_type": "Mobile",
            "ip_country": "IN", "billing_country": "IN", "failed_attempts": 0,
            "account_age_days": 240, "previous_order_count": 5, "avg_order_value": 520.0,
            "is_new_customer": False, "recipient_id": "demo-known-recipient",
        },
        "history": [
            {"transaction_id": "DEMO-HISTORY-001", "customer_id": "DEMO-CUSTOMER-001", "recipient_id": "demo-known-recipient", "amount": 480.0, "payment_method": "UPI", "device_type": "Mobile", "ip_country": "IN", "billing_country": "IN"},
        ],
    },
    "New Recipient": {
        "form": {
            "amount": 8500.0,
            "recipient_name": "Rahul Enterprises",
            "recipient_identifier": "demo-new-recipient",
            "purpose": "Laptop advance",
            "familiarity": "New",
            "message": "Advance payment for laptop.",
        },
        "transaction": {
            "transaction_id": "DEMO-NEW-001", "customer_id": "DEMO-CUSTOMER-001",
            "amount": 8500.0, "payment_method": "UPI", "device_type": "Mobile",
            "ip_country": "IN", "billing_country": "IN", "failed_attempts": 0,
            "account_age_days": 240, "previous_order_count": 5, "avg_order_value": 650.0,
            "is_new_customer": False, "recipient_id": "demo-new-recipient",
        },
        "history": [
            {"transaction_id": "DEMO-HISTORY-002", "customer_id": "DEMO-CUSTOMER-001", "recipient_id": "demo-known-recipient", "amount": 600.0, "payment_method": "UPI", "device_type": "Mobile", "ip_country": "IN", "billing_country": "IN"},
        ],
    },
    "Suspicious Payment Request": {
        "form": {
            "amount": 8500.0,
            "recipient_name": "Unknown Support",
            "recipient_identifier": "demo-suspicious-recipient",
            "purpose": "Account verification",
            "familiarity": "New",
            "message": "URGENT! Pay immediately or your account will be blocked today.",
        },
        "transaction": {
            "transaction_id": "DEMO-SUSPICIOUS-001", "customer_id": "DEMO-CUSTOMER-001",
            "amount": 8500.0, "payment_method": "CARD", "device_type": "Desktop",
            "ip_country": "US", "billing_country": "IN", "failed_attempts": 1,
            "account_age_days": 240, "previous_order_count": 5, "avg_order_value": 700.0,
            "is_new_customer": False, "recipient_id": "demo-suspicious-recipient",
        },
        "history": [
            {"transaction_id": "DEMO-HISTORY-003", "customer_id": "DEMO-CUSTOMER-001", "recipient_id": "demo-known-recipient", "amount": 700.0, "payment_method": "UPI", "device_type": "Mobile", "ip_country": "IN", "billing_country": "IN"},
        ],
    },
    "Multiple Failed Attempts": {
        "form": {
            "amount": 12000.0,
            "recipient_name": "New Merchant",
            "recipient_identifier": "demo-failed-attempt-recipient",
            "purpose": "Online purchase",
            "familiarity": "New",
            "message": "Please complete the payment again.",
        },
        "transaction": {
            "transaction_id": "DEMO-FAILED-001", "customer_id": "DEMO-CUSTOMER-001",
            "amount": 12000.0, "payment_method": "CARD", "device_type": "Desktop",
            "ip_country": "US", "billing_country": "IN", "failed_attempts": 4,
            "account_age_days": 240, "previous_order_count": 5, "avg_order_value": 850.0,
            "is_new_customer": False, "recipient_id": "demo-failed-attempt-recipient",
        },
        "history": [
            {"transaction_id": "DEMO-HISTORY-004", "customer_id": "DEMO-CUSTOMER-001", "recipient_id": "demo-known-recipient", "amount": 800.0, "payment_method": "UPI", "device_type": "Mobile", "ip_country": "IN", "billing_country": "IN"},
        ],
    },
}


@st.cache_data
def load_transactions():
    """Load and score the mock transaction dataset."""
    df = pd.read_csv(DATA_PATH)
    return score_transactions(df)


@st.cache_resource
def load_ml_model():
    """Load the pre-trained v2 pipeline without retraining it."""
    try:
        return joblib.load(ML_MODEL_PATH), None
    except Exception as error:
        return None, str(error)


def to_native_value(value):
    """Convert pandas/numpy scalar values into JSON-friendly Python values."""
    if isinstance(value, dict):
        return {key: to_native_value(item) for key, item in value.items()}
    if isinstance(value, list):
        return [to_native_value(item) for item in value]
    if hasattr(value, "item"):
        return value.item()
    return value


def get_combined_interpretation(rule_band, rule_recommendation, ml_decision):
    """Summarize the two independent decision-support signals for an investigator."""
    if ml_decision == "REVIEW" and rule_band in {"Medium Risk", "High Risk"}:
        return "Both the rule engine and ML model indicate elevated risk. Prioritize human investigation."
    if ml_decision == "REVIEW":
        return "The ML model recommends review while the rule engine is lower risk. Investigate the transaction before acting."
    if rule_recommendation != "APPROVE":
        return "The rule engine indicates elevated risk while the ML score is below review threshold. Keep the rule-based recommendation and investigate its signals."
    return "Both systems indicate a lower-risk candidate. The existing rule recommendation remains the decision-support baseline."


def build_user_provided_guardian_input(amount, recipient_name, recipient_identifier, purpose, familiarity, message):
    """Create an explicitly user-provided context object without inventing payment history."""
    provided_fields = {
        "amount": "user_provided",
        "recipient_name": "user_provided",
        "recipient_identifier": "user_provided",
        "payment_purpose": "user_provided",
        "recipient_familiarity": "user_provided",
    }
    if message:
        provided_fields["payment_message"] = "user_provided"
    transaction = {
        "transaction_id": "USER_PROVIDED_PAYMENT",
        "amount": float(amount),
        "recipient_name": recipient_name.strip() or None,
        "recipient_id": recipient_identifier.strip() or None,
        "payment_purpose": purpose.strip() or None,
        "feature_coverage": "User-provided context",
        "feature_sources": provided_fields,
        "unavailable_fraud_features": [
            "customer_id", "device_type", "ip_country", "billing_country",
            "failed_attempts", "account_age_days", "previous_order_count",
            "avg_order_value", "is_new_customer",
        ],
    }
    context = {
        "feature_coverage": "User-provided context",
        "feature_sources": provided_fields,
        "unavailable_fraud_features": transaction["unavailable_fraud_features"],
        "recipient_familiarity": familiarity,
        "payment_purpose": transaction["payment_purpose"],
    }
    return transaction, context


def add_manual_high_value_signal(result, amount, purpose):
    """Add a transparent high-value verification signal when no personal baseline is available."""
    if float(amount or 0) < GUARDIAN_HIGH_VALUE_AMOUNT:
        return result

    result = dict(result)
    signals = list(result.get("signals", []))

    purpose_text = purpose.strip() if purpose else "the stated purpose"
    high_value_signal = {
        "type": "HIGH_VALUE_PAYMENT",
        "title": "High-value payment",
        "description": (
            f"₹{float(amount):,.2f} is a high-value payment for this Guardian check. "
            f"The stated purpose is '{purpose_text}'."
        ),
        "why_it_matters": (
            "There is not enough personal payment history in this manual check to "
            "compare the amount with the user's normal spending pattern."
        ),
        "severity": "medium",
        "action": (
            "Verify the recipient and payment purpose independently before proceeding. "
            "A high amount alone does not prove fraud."
        ),
    }

    if not any(signal.get("type") == "HIGH_VALUE_PAYMENT" for signal in signals):
        signals.append(high_value_signal)

    result["signals"] = signals

    if result.get("status") == "LOW":
        result["status"] = "PAUSE"
        result["headline"] = "Pause and verify"
        result["summary"] = (
            "This is a high-value payment with limited personal history available "
            "for comparison. Verify the recipient and payment purpose before paying."
        )

    limitations = list(result.get("limitations", []))
    limitation = (
        "No personal payment baseline was supplied, so the high-value warning is "
        "a verification prompt rather than evidence of fraud."
    )
    if limitation not in limitations:
        limitations.append(limitation)
    result["limitations"] = limitations

    return result


def apply_manual_guardian_thresholds(result, amount, familiarity):
    """Apply the requested demo-friendly amount/familiarity rules to manual Guardian checks.

    These are explicit prototype rules for the manual Guardian experience:
    - Unknown + amount > ₹10 -> Suspicious Payment
    - New + amount <= ₹10,000 -> Suspicious Payment
    - Familiar + amount <= ₹20,00,000 -> Looks normal
    Unspecified combinations retain the existing Guardian result.
    """
    result = dict(result)
    signals = list(result.get("signals", []))
    familiarity_value = str(familiarity or "").strip().casefold()
    amount_value = float(amount)

    classification = None
    if familiarity_value == "unknown" and amount_value > GUARDIAN_UNKNOWN_SUSPICIOUS_AMOUNT:
        classification = (
            "Suspicious Payment",
            "This payment matches the configured Unknown-recipient demo rule. "
            "Please verify the recipient and payment details before proceeding.",
        )
    elif familiarity_value == "new" and amount_value <= GUARDIAN_NEW_SUSPICIOUS_MAX_AMOUNT:
        classification = (
            "Suspicious Payment",
            "This payment matches the configured New-recipient demo rule. "
            "Please verify the recipient and payment details before proceeding.",
        )
    elif familiarity_value == "familiar" and amount_value <= GUARDIAN_FAMILIAR_SAFE_MAX_AMOUNT:
        classification = (
            "Looks normal",
            "The payment matches the configured Familiar-recipient demo range. "
            "Still recheck the amount and recipient before paying.",
        )

    if classification is None:
        return result

    headline, summary = classification
    result["headline"] = headline
    result["summary"] = summary

    if headline == "Suspicious Payment":
        result["status"] = "PAUSE"
        result["recommendation"] = "Pause and verify before paying."
        suspicious_signal = {
            "type": "MANUAL_DEMO_RULE",
            "title": "Suspicious payment pattern",
            "description": (
                f"The selected recipient familiarity ({familiarity}) and amount "
                f"₹{amount_value:,.2f} match a configured Riskora prototype rule."
            ),
            "why_it_matters": "This is a prototype verification rule, not proof of fraud.",
            "severity": "medium",
            "action": "Verify the recipient independently before proceeding.",
        }
        if not any(signal.get("type") == "MANUAL_DEMO_RULE" for signal in signals):
            signals.append(suspicious_signal)
        result["signals"] = signals
    else:
        result["status"] = "LOW"
        result["recommendation"] = "Proceed only after your normal payment checks."

    return result


def get_community_store():
    """Return a lightweight prototype-only community signal store."""
    store = st.session_state.setdefault(
        "riskora_community_store",
        {
            "merchant": {"verified": 0, "unverified": 0},
            "personal": {"verified": 0, "unverified": 0},
        },
    )
    return store


def get_community_signal(payment_type):
    """Summarize community input without treating it as proof."""
    store = get_community_store()
    bucket = store.get(payment_type, {"verified": 0, "unverified": 0})
    total = bucket["verified"] + bucket["unverified"]
    if total == 0:
        return {
            "total": 0,
            "verified": 0,
            "unverified": 0,
            "label": "No community reports yet",
            "direction": "unknown",
        }
    verified = bucket["verified"]
    unverified = bucket["unverified"]
    if unverified > verified:
        direction = "caution"
        label = f"{unverified} unverified · {verified} verified reports"
    elif verified > unverified:
        direction = "positive"
        label = f"{verified} verified · {unverified} unverified reports"
    else:
        direction = "mixed"
        label = f"{verified} verified · {unverified} unverified reports"
    return {
        "total": total,
        "verified": verified,
        "unverified": unverified,
        "label": label,
        "direction": direction,
    }


def record_community_feedback(payment_type, feedback, amount):
    """Record prototype community feedback for payments at/above the review floor."""
    if float(amount or 0) < RISKORA_COMMUNITY_REVIEW_MIN_AMOUNT:
        return False
    store = get_community_store()
    store[payment_type][feedback] += 1
    st.session_state["riskora_community_store"] = store
    return True


def apply_community_signals(result, amount, payment_type):
    """Add community reputation as a transparent supporting signal."""
    result = dict(result)
    signals = list(result.get("signals", []))
    community = get_community_signal(payment_type)

    if float(amount or 0) < RISKORA_COMMUNITY_REVIEW_MIN_AMOUNT or community["total"] == 0:
        return result, community

    if community["unverified"] > community["verified"]:
        signal = {
            "type": "COMMUNITY_UNVERIFIED_REPORTS",
            "title": "Community caution signal",
            "description": community["label"],
            "why_it_matters": "Other users have reported this payment target as unverified more often than verified in this prototype.",
            "severity": "medium",
            "action": "Pause and independently verify the recipient before paying.",
        }
        if not any(s.get("type") == signal["type"] for s in signals):
            signals.append(signal)
        if result.get("status") == "LOW":
            result["status"] = "PAUSE"
            result["headline"] = "Suspicious Payment"
            result["summary"] = "Community reports add a caution signal. Verify the recipient before paying; reports are not proof of fraud."
    elif community["verified"] > community["unverified"] and result.get("status") == "PAUSE":
        # Community confidence is supporting evidence only; it never overrides HIGH.
        result["summary"] = "Community reports currently lean verified, but Riskora still asks you to review the other available warning signals."

    result["signals"] = signals
    return result, community


def apply_final_payment_state(result):
    """Return a concise pre-payment state for the UI."""
    status = result.get("status", "LOW")
    if status == "HIGH":
        return {
            "label": "STOP & VERIFY",
            "tone": "risk",
            "message": "Multiple warning signals were detected. Do not enter your payment PIN until you verify the recipient and request.",
        }
    if status == "PAUSE":
        return {
            "label": "PAUSE & CHECK",
            "tone": "caution",
            "message": "Some warning signals were detected. Take a moment to verify before entering your payment PIN.",
        }
    return {
        "label": "READY TO REVIEW",
        "tone": "safe",
        "message": "No major warning was detected from the available information. Recheck the amount and recipient before entering your payment PIN.",
    }


def apply_demo_scenario():
    """Populate Guardian form state from explicitly synthetic demo data."""
    scenario_name = st.session_state.get("guardian_demo_scenario", "None")
    scenario = DEMO_SCENARIOS.get(scenario_name)
    if not scenario:
        return
    form = scenario["form"]
    st.session_state["guardian_amount"] = form["amount"]
    st.session_state["guardian_recipient_name"] = form["recipient_name"]
    st.session_state["guardian_recipient_identifier"] = form["recipient_identifier"]
    st.session_state["guardian_purpose"] = form["purpose"]
    st.session_state["guardian_familiarity"] = form["familiarity"]
    st.session_state["guardian_message"] = form["message"]
    st.session_state.pop("guardian_user_result", None)


def analyze_demo_scenario(scenario_name, ml_model, ml_model_error):
    """Run a selected synthetic scenario through existing analysis components."""
    scenario = DEMO_SCENARIOS[scenario_name]
    form = scenario["form"]
    transaction = dict(scenario["transaction"])
    transaction["payment_purpose"] = form["purpose"]
    transaction["recipient_name"] = form["recipient_name"]
    rule_result = score_transaction(transaction)
    rule_result["recommendation"] = RECOMMENDATIONS[rule_result["risk_band"]]
    ml_result = {}
    if ml_model is not None and not ml_model_error:
        ml_score = get_ml_risk_score(ml_model, transaction)
        ml_result = {
            "ml_risk_score": ml_score,
            "decision": "REVIEW" if ml_score >= ML_REVIEW_THRESHOLD else "BELOW REVIEW THRESHOLD",
        }
    context = {
        "feature_coverage": "Demo scenario",
        "feature_sources": {"demo_scenario": "synthetic_demo_data"},
        "recipient_familiarity": form["familiarity"],
        "payment_purpose": form["purpose"],
    }
    return analyze_with_guardian(
        transaction,
        rule_result=rule_result,
        ml_result=ml_result,
        historical_transactions=scenario["history"],
        payment_message=form["message"],
        context=context,
    )


def get_signal_concept(signal):
    """Group equivalent source signals for concise Guardian presentation only."""
    signal_type = str(signal.get("type", "")).upper()
    concepts = {
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
    if signal_type in concepts:
        return concepts[signal_type]
    title = str(signal.get("title", "")).casefold()
    if "unusual amount" in title or "large purchase" in title:
        return "unusual_amount"
    if "failed attempt" in title:
        return "failed_attempts"
    if "location mismatch" in title or "location change" in title:
        return "country_location_change"
    return signal_type.lower() or title or "other_signal"


def get_primary_guardian_signals(signals):
    """Return one readable card per risk concept while retaining raw evidence elsewhere."""
    grouped = {}
    for signal in signals:
        if signal.get("type") == "LIMITED_BEHAVIORAL_CONTEXT":
            continue
        grouped.setdefault(get_signal_concept(signal), []).append(signal)
    return list(grouped.values())


def render_guardian_result(result, community=None):
    """Render a concise consumer-facing Guardian result."""
    status = result.get("status", "LOW")
    state = apply_final_payment_state(result)
    st.markdown(f"### {state['label']}")
    if state["tone"] == "risk":
        st.error(state["message"])
    elif state["tone"] == "caution":
        st.warning(state["message"])
    else:
        st.success(state["message"])

    st.write(result.get("summary", "Riskora has completed the available checks."))

    source_signals = result.get("signals", [])
    primary_signal_groups = get_primary_guardian_signals(source_signals)

    st.markdown("### 🔎 Riskora check")
    if primary_signal_groups:
        for evidence_group in primary_signal_groups:
            signal = evidence_group[0]
            with st.container(border=True):
                st.markdown(
                    f"**{signal.get('title', 'Risk signal')}** · "
                    f"{signal.get('severity', 'low').title()}"
                )
                st.write(signal.get("description", ""))
                if signal.get("why_it_matters"):
                    st.caption(signal["why_it_matters"])
    else:
        st.info("No major warning was detected from the information you provided.")

    if community is not None and community.get("total", 0):
        st.caption(f"Community signal: {community['label']} · community input is supporting evidence, not proof.")

    st.markdown("### Next step")
    for action in result.get("action_steps", []):
        st.markdown(f"- {action}")
    if not result.get("action_steps"):
        st.write(result.get("recommendation", "Review the payment details before proceeding."))




st.set_page_config(page_title="Riskora", page_icon="🛡️", layout="wide")

st.markdown(
    """
    <div class="riskora-brand-panel">
        <div style="font-size:.72rem;letter-spacing:.16em;font-weight:800;color:#B8B0FF;">
            RISKORA
        </div>
        <div style="font-size:1.55rem;font-weight:800;margin-top:.15rem;">
            CHECK BEFORE YOU PAY.
        </div>
        <div style="font-size:.88rem;opacity:.82;margin-top:.25rem;">
            A quiet safety layer between you and your payment.
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

# ---------------------------------------------------------------------------
# Riskora visual system
# ---------------------------------------------------------------------------
st.markdown(
    """
    <style>
    /* Riskora Electric Aurora — 80% controlled dark neutral / 20% electric colour */
    :root {
        --riskora-bg: #090C12;
        --riskora-surface: #10151F;
        --riskora-surface-2: #151C29;
        --riskora-surface-3: #1B2433;
        --riskora-text: #F4F7FF;
        --riskora-muted: #8B96AA;
        --riskora-border: #263142;
        --riskora-cyan: #28F0D0;
        --riskora-blue: #5267FF;
        --riskora-coral: #FF5C8A;
        --riskora-yellow: #FFD166;
        --riskora-green: #42E6A4;
    }

    .stApp {
        background:
            radial-gradient(circle at 88% -5%, rgba(82,103,255,.13), transparent 24%),
            radial-gradient(circle at 8% 18%, rgba(40,240,208,.055), transparent 22%),
            #090C12;
        color: var(--riskora-text);
    }
    [data-testid="stHeader"] { background: transparent; }
    .block-container { max-width: 1180px; padding-top: 1.25rem; padding-bottom: 3rem; }

    h1,h2,h3,h4,p,label,span { color: var(--riskora-text); }
    h1,h2,h3 { letter-spacing: -.035em; }
    .riskora-muted { color: var(--riskora-muted) !important; }
    .riskora-eyebrow {
        color: var(--riskora-cyan) !important;
        letter-spacing: .16em !important;
        font-weight: 850 !important;
        font-size: .67rem !important;
    }

    .riskora-card {
        background: var(--riskora-surface) !important;
        border: 1px solid var(--riskora-border) !important;
        box-shadow: 0 18px 48px rgba(0,0,0,.25) !important;
        border-radius: 20px !important;
    }
    .riskora-soft {
        background: var(--riskora-surface-2) !important;
        border: 1px solid #293548 !important;
        border-radius: 16px !important;
    }

    .riskora-brand-panel {
        background: linear-gradient(145deg, #0D121B 0%, #111827 62%, #151B2A 100%);
        color: var(--riskora-text);
        border: 1px solid #263142;
        border-radius: 24px;
        padding: 1.7rem 1.75rem;
        margin-bottom: 1.15rem;
        box-shadow: 0 22px 60px rgba(0,0,0,.30);
        position: relative;
        overflow: hidden;
    }
    .riskora-brand-panel:before {
        content: ""; position:absolute; width:280px; height:280px; border-radius:50%;
        right:-125px; top:-150px; background:rgba(40,240,208,.10);
        box-shadow: -90px 100px 150px rgba(82,103,255,.10);
    }
    .riskora-brand-panel:after {
        content: ""; position:absolute; left:55%; bottom:-90px; width:210px; height:210px;
        border-radius:50%; background:rgba(82,103,255,.075); filter:blur(8px);
    }
    .riskora-brand-panel h1,.riskora-brand-panel h2,.riskora-brand-panel h3,.riskora-brand-panel p { color:#F4F7FF !important; }
    .riskora-brand-panel .riskora-eyebrow { color:var(--riskora-cyan) !important; }

    button[kind="primary"] {
        background: var(--riskora-cyan) !important;
        border: 1px solid var(--riskora-cyan) !important;
        color: #07100F !important;
        box-shadow: 0 0 0 1px rgba(40,240,208,.08), 0 10px 28px rgba(40,240,208,.16) !important;
        border-radius: 12px !important; font-weight: 850 !important;
    }
    button[kind="primary"]:hover {
        background: #7CFFE9 !important; border-color:#7CFFE9 !important; color:#07100F !important;
        box-shadow: 0 0 24px rgba(40,240,208,.22) !important;
    }
    div[data-testid="stButton"] > button {
        border-radius:12px !important; min-height:44px !important; font-weight:750 !important;
        background:#151C29 !important; color:#F4F7FF !important; border:1px solid #2A3547 !important;
    }
    div[data-testid="stButton"] > button:hover { border-color:var(--riskora-cyan) !important; color:var(--riskora-cyan) !important; }

    input,textarea {
        border-radius:12px !important; border:1px solid #2A3547 !important;
        background:#0E141E !important; color:#F4F7FF !important;
    }
    input::placeholder, textarea::placeholder {
        color:#718097 !important; opacity:1 !important;
    }
    input:focus,textarea:focus {
        border-color:var(--riskora-cyan) !important; box-shadow:0 0 0 1px var(--riskora-cyan), 0 0 18px rgba(40,240,208,.08) !important;
    }
    div[data-baseweb="select"] > div {
        border-radius:12px !important; border-color:#2A3547 !important; background:#0E141E !important;
    }
    div[data-testid="stRadio"] label { border-radius:12px !important; }
    div[data-testid="stMetric"] {
        background:#111925 !important; border:1px solid #293548 !important;
        border-radius:14px !important; padding:.8rem !important;
    }

    .riskora-status-safe {
        background:#0D211C !important; border:1px solid #1B5B46 !important;
        border-left:6px solid var(--riskora-green) !important; border-radius:18px !important;
    }
    .riskora-status-caution {
        background:#241E0E !important; border:1px solid #6B5520 !important;
        border-left:6px solid var(--riskora-yellow) !important; border-radius:18px !important;
    }
    .riskora-status-stop {
        background:#28131B !important; border:1px solid #703042 !important;
        border-left:6px solid var(--riskora-coral) !important; border-radius:18px !important;
    }

    [data-testid="stSidebar"] {
        background:#080B10 !important; border-right:1px solid #1D2633 !important;
    }
    [data-testid="stSidebar"] * { color:#C8D0DE !important; }
    hr { border-color:#202A38 !important; }

    @media (max-width:900px) {
        .block-container { padding:.75rem .8rem 2rem !important; max-width:100% !important; }
        [data-testid="stSidebar"],[data-testid="stSidebarCollapsedControl"] { display:none !important; }
        .riskora-brand-panel { border-radius:19px !important; padding:1.2rem 1.15rem !important; }
        .riskora-card { border-radius:17px !important; }
        div[data-testid="stHorizontalBlock"] { flex-wrap:wrap !important; gap:.65rem !important; }
        div[data-testid="stHorizontalBlock"] > div[data-testid="column"] { min-width:100% !important; flex:1 1 100% !important; }
        div[data-testid="stHorizontalBlock"]:has(button) > div[data-testid="column"] { min-width:0 !important; flex:1 1 48% !important; }
        div[data-testid="stButton"] > button { width:100% !important; min-height:48px !important; }
        div[data-testid="stNumberInput"] input,div[data-testid="stTextInput"] input { min-height:48px !important; font-size:1rem !important; }
        div[data-testid="stRadio"] > div { gap:.45rem !important; flex-wrap:wrap !important; }
        div[data-testid="stRadio"] label {
            border:1px solid #2A3547 !important; border-radius:12px !important; padding:.65rem .7rem !important;
            background:#111925 !important; flex:1 1 30% !important; justify-content:center !important;
        }
        iframe { max-width:100% !important; }
    }
    @media (max-width:520px) {
        .block-container { padding:.45rem .6rem 1.5rem !important; }
        .riskora-brand-panel { padding:1rem !important; border-radius:17px !important; }
        .riskora-card { padding:.72rem !important; margin-bottom:.65rem !important; }
        .riskora-card h3 { font-size:1.28rem !important; margin:.28rem 0 .55rem !important; }
        .riskora-eyebrow { font-size:.60rem !important; }
        div[data-testid="stNumberInput"] { margin-bottom:.1rem !important; }
        div[data-testid="stNumberInput"] input { min-height:43px !important; }
        div[data-testid="stRadio"] { margin-bottom:.15rem !important; }
        div[data-testid="stRadio"] label { flex:1 1 30% !important; min-height:40px !important; padding:.42rem .5rem !important; }
        .riskora-card [data-testid="stCaptionContainer"] { margin-top:.1rem !important; }
        .riskora-card [data-testid="stTextInput"] { margin-top:.05rem !important; }
        .riskora-card [data-testid="stExpander"] { margin-top:.45rem !important; }
    }
    </style>
    """,
    unsafe_allow_html=True,
)

def riskora_display_error(error):
    """Keep provider-specific implementation details out of the consumer UI."""
    message = str(error)
    message = message.replace("Razorpay", "payment provider")
    return message

def render_riskora_brand():
    st.markdown(
        """
        <div class="riskora-brand">
            <div class="riskora-logo">🛡️</div>
            <div>
                <div class="riskora-name">Riskora</div>
                <div class="riskora-tag">A calmer way to check a payment before you pay.</div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

render_riskora_brand()

st.sidebar.markdown(
    """
    <div style="padding:.35rem 0 1rem 0;">
      <div style="font-size:1.35rem;font-weight:850;">Riskora</div>
      <div style="font-size:.78rem;opacity:.78;">Payment safety workspace</div>
    </div>
    """,
    unsafe_allow_html=True,
)
mode = st.sidebar.radio(
    "Workspace",
    ["AI Financial Guardian", "Analyst Mode"],
    label_visibility="collapsed",
)

# ---------------------------------------------------------------------------
# Consumer experience
# ---------------------------------------------------------------------------
if mode == "AI Financial Guardian":
    st.markdown(
        """
        <div class="riskora-hero">
            <div class="riskora-eyebrow" style="color:#dcd8ff;">BEFORE YOU PAY</div>
            <h1>Check once. Pay with confidence.</h1>
            <p>Riskora looks at the payment context, unusual signals and optional community feedback — then gives you a simple next step.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.session_state.setdefault("guardian_payment_type", "Merchant")
    st.session_state.setdefault("guardian_feedback", "Verified")
    st.session_state.setdefault("guardian_recipient_identifier", "")
    st.session_state.setdefault("guardian_purpose", "")
    st.session_state.setdefault("guardian_familiarity", "Unknown")
    st.session_state.setdefault("guardian_message", "")

    left, right = st.columns([1.55, 1], gap="large")

    with left:
        st.markdown('<div class="riskora-card">', unsafe_allow_html=True)
        st.markdown('<div class="riskora-eyebrow">STEP 1 · AMOUNT</div>', unsafe_allow_html=True)
        st.markdown("### How much are you paying?")
        guardian_amount = st.number_input(
            "Amount (₹)",
            min_value=0.0,
            step=1.0,
            key="guardian_amount",
            placeholder="0",
            label_visibility="collapsed",
        )
        if guardian_amount > 0:
            st.markdown(
                f'<div style="font-size:1.75rem;font-weight:850;color:#211b50;margin:.35rem 0 1rem 0;">₹{guardian_amount:,.2f}</div>',
                unsafe_allow_html=True,
            )
        else:
            st.caption("Start with the amount. You can add more context only if you want.")
        st.markdown("</div>", unsafe_allow_html=True)

        st.markdown('<div class="riskora-card">', unsafe_allow_html=True)
        st.markdown('<div class="riskora-eyebrow">STEP 2 · PAYMENT TYPE</div>', unsafe_allow_html=True)
        st.markdown("### What kind of payment is this?")
        guardian_payment_type = st.radio(
            "Payment type",
            ["Merchant", "Personal"],
            horizontal=True,
            key="guardian_payment_type",
            label_visibility="collapsed",
        )

        st.markdown('<div class="riskora-eyebrow" style="margin-top:1rem;">RECIPIENT HISTORY</div>', unsafe_allow_html=True)
        st.markdown("### Have you paid them before?")
        guardian_familiarity = st.radio(
            "Recipient familiarity",
            ["Familiar", "New", "Unknown"],
            horizontal=True,
            key="guardian_familiarity",
            label_visibility="collapsed",
        )
        st.markdown("</div>", unsafe_allow_html=True)

        st.markdown('<div class="riskora-card">', unsafe_allow_html=True)
        st.markdown('<div class="riskora-eyebrow">STEP 3 · CONTEXT</div>', unsafe_allow_html=True)
        guardian_purpose = st.text_input(
            "Payment purpose (optional)",
            key="guardian_purpose",
            placeholder="e.g. rent, groceries, school fees",
        )
        with st.expander("Add details only if useful"):
            guardian_recipient_identifier = st.text_input(
                "UPI ID / account / reference (optional)",
                key="guardian_recipient_identifier",
                placeholder="Optional",
            )
            guardian_message = st.text_area(
                "Payment request / message (optional)",
                key="guardian_message",
                placeholder="Paste a message if someone asked you to pay",
            )
        st.markdown("</div>", unsafe_allow_html=True)

    with right:
        st.markdown('<div class="riskora-card">', unsafe_allow_html=True)
        st.markdown('<div class="riskora-eyebrow">COMMUNITY SIGNAL</div>', unsafe_allow_html=True)
        st.markdown("### Has this recipient been verified?")
        community = get_community_signal(guardian_payment_type.casefold())

        if guardian_amount >= RISKORA_COMMUNITY_REVIEW_MIN_AMOUNT:
            st.caption(
                f"Available for payments of ₹{RISKORA_COMMUNITY_REVIEW_MIN_AMOUNT:,.0f}+."
            )
            feedback = st.radio(
                "Community feedback",
                ["Verified", "Unverified"],
                horizontal=True,
                key="guardian_feedback",
                label_visibility="collapsed",
            )
            if st.button("Share this signal", type="primary", use_container_width=True):
                record_community_feedback(
                    guardian_payment_type.casefold(),
                    feedback.casefold(),
                    guardian_amount,
                )
                community = get_community_signal(guardian_payment_type.casefold())
                st.success("Signal shared.")
        else:
            st.caption(
                f"Community feedback starts at ₹{RISKORA_COMMUNITY_REVIEW_MIN_AMOUNT:,.0f}. "
                "Other Riskora checks still apply below that amount."
            )

        if community.get("total", 0):
            st.markdown(
                f"""
                <div class="riskora-soft">
                    <div style="font-weight:800;">Current community tally</div>
                    <div style="font-size:1.25rem;font-weight:850;margin:.25rem 0;">{community['verified']} verified · {community['unverified']} unverified</div>
                    <div class="riskora-muted" style="font-size:.8rem;">Supporting signal only — never proof of fraud.</div>
                </div>
                """,
                unsafe_allow_html=True,
            )
        else:
            st.markdown(
                """
                <div class="riskora-soft">
                    <div style="font-weight:800;">No reports yet</div>
                    <div class="riskora-muted" style="font-size:.82rem;">Be the first to add a signal when eligible.</div>
                </div>
                """,
                unsafe_allow_html=True,
            )
        st.markdown("</div>", unsafe_allow_html=True)

        st.markdown(
            """
            <div class="riskora-card">
              <div class="riskora-eyebrow">WHY RISKORA?</div>
              <div style="font-weight:800;font-size:1.02rem;">Small checks. Clear action.</div>
              <div class="riskora-muted" style="font-size:.84rem;margin-top:.3rem;">
                Riskora does not ask for your PIN. It gives you a pre-payment signal so you can decide what to do next.
              </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    if guardian_amount > 0:
        try:
            ml_model, ml_model_error = load_ml_model()
            guardian_transaction, guardian_context = build_user_provided_guardian_input(
                guardian_amount,
                "",
                guardian_recipient_identifier,
                guardian_purpose,
                guardian_familiarity,
                guardian_message,
            )
            guardian_transaction["payment_type"] = guardian_payment_type

            guardian_result = analyze_with_guardian(
                guardian_transaction,
                historical_transactions=None,
                payment_message=guardian_message or None,
                context=guardian_context,
            )
            guardian_result = add_manual_high_value_signal(
                guardian_result, guardian_amount, guardian_purpose
            )
            guardian_result = apply_manual_guardian_thresholds(
                guardian_result, guardian_amount, guardian_familiarity
            )
            guardian_result, community = apply_community_signals(
                guardian_result,
                guardian_amount,
                guardian_payment_type.casefold(),
            )
            st.session_state["guardian_user_result"] = guardian_result
        except Exception as error:
            st.error(f"Riskora could not complete the safety check: {riskora_display_error(error)}")
            guardian_result = None
    else:
        st.session_state.pop("guardian_user_result", None)
        guardian_result = None

    if guardian_result:
        final_state = apply_final_payment_state(guardian_result)
        tone_class = {
            "risk": "riskora-state-risk",
            "caution": "riskora-state-caution",
            "safe": "riskora-state-safe",
        }[final_state["tone"]]
        icon = {"risk": "🔴", "caution": "🟡", "safe": "🟢"}[final_state["tone"]]

        st.markdown(
            f"""
            <div class="riskora-state {tone_class}">
                <div class="riskora-state-title">{icon} {final_state['label']}</div>
                <div style="color:#475467;">{final_state['message']}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        st.markdown('<div class="riskora-card">', unsafe_allow_html=True)
        st.markdown('<div class="riskora-eyebrow">RISKORA CHECK</div>', unsafe_allow_html=True)
        st.markdown(f"### {guardian_result.get('headline', 'Payment check complete')}")
        st.write(guardian_result.get("summary", "Riskora has completed the available checks."))

        source_signals = guardian_result.get("signals", [])
        primary_signal_groups = get_primary_guardian_signals(source_signals)

        if primary_signal_groups:
            for evidence_group in primary_signal_groups:
                signal = evidence_group[0]
                severity = signal.get("severity", "low").title()
                st.markdown(
                    f"""
                    <div class="riskora-soft" style="margin:.65rem 0;">
                        <div style="font-weight:800;">{signal.get('title', 'Risk signal')}</div>
                        <div style="font-size:.82rem;color:#667085;margin-top:.2rem;">{severity} signal</div>
                        <div style="margin-top:.35rem;">{signal.get('description', '')}</div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
        else:
            st.success("No major warning was detected from the information you provided.")

        if community is not None and community.get("total", 0):
            st.caption(
                f"Community signal: {community['label']} · supporting evidence, not proof."
            )
        st.markdown("</div>", unsafe_allow_html=True)

        st.markdown('<div class="riskora-card">', unsafe_allow_html=True)
        st.markdown('<div class="riskora-eyebrow">NEXT STEP</div>', unsafe_allow_html=True)
        actions = result_actions = guardian_result.get("action_steps", [])
        if actions:
            # Keep the consumer advice short.
            for action in actions[:2]:
                st.markdown(f"**•** {action}")
        else:
            st.write("Recheck the amount and recipient before paying.")
        st.markdown("</div>", unsafe_allow_html=True)

    st.markdown(
        '<div class="riskora-footer">Riskora is a decision-support layer. It does not request, store or enter your payment PIN.</div>',
        unsafe_allow_html=True,
    )
    st.stop()

# ---------------------------------------------------------------------------
# Analyst workspace
# ---------------------------------------------------------------------------
transactions = load_transactions()
ml_model, ml_model_error = load_ml_model()

checkout_payment_id = st.query_params.get("razorpay_payment_id")
checkout_order_id = st.query_params.get("razorpay_order_id")
checkout_signature = st.query_params.get("razorpay_signature")
if checkout_payment_id or checkout_order_id or checkout_signature:
    pending_order_id = st.session_state.get("razorpay_checkout_order_id")
    try:
        if not pending_order_id or checkout_order_id != pending_order_id:
            raise RazorpayIntegrationError("Test Checkout returned an unexpected order.")
        verify_test_payment_signature(
            checkout_payment_id or "",
            checkout_order_id or "",
            checkout_signature or "",
        )
        verified_payment = fetch_test_payment(
            checkout_payment_id or "", pending_order_id
        )
        if verified_payment.get("status") not in {"authorized", "captured"}:
            raise RazorpayIntegrationError(
                "The test payment was not successfully authorized or captured."
            )
        st.session_state["razorpay_verified_payment"] = verified_payment
        st.session_state["razorpay_checkout_result"] = "success"
    except RazorpayIntegrationError as error:
        st.session_state["razorpay_checkout_error"] = str(error)
        st.session_state.pop("razorpay_verified_payment", None)
    finally:
        st.query_params.clear()

if transactions.empty:
    st.warning("No transaction data was found. Please add records to the dataset.")
    st.stop()

st.markdown(
    """
    <div class="riskora-hero">
        <div class="riskora-eyebrow" style="color:#dcd8ff;">ANALYST WORKSPACE</div>
        <h1>Riskora Test & Investigation</h1>
        <p>Explore synthetic transaction risk and run safe payment-provider test workflows without exposing consumer-facing technical details.</p>
    </div>
    """,
    unsafe_allow_html=True,
)

total_transactions = len(transactions)
band_counts = transactions["risk_band"].value_counts()
high_risk_count = int(band_counts.get("High Risk", 0))
medium_risk_count = int(band_counts.get("Medium Risk", 0))
low_risk_count = int(band_counts.get("Low Risk", 0))
overall_risk_percentage = ((high_risk_count + medium_risk_count) / total_transactions) * 100

summary_columns = st.columns(4)
summary_columns[0].metric("Transactions", total_transactions)
summary_columns[1].metric("High risk", high_risk_count)
summary_columns[2].metric("Medium risk", medium_risk_count)
summary_columns[3].metric("Risk rate", f"{overall_risk_percentage:.1f}%")

st.markdown('<div class="riskora-card">', unsafe_allow_html=True)
st.markdown('<div class="riskora-eyebrow">PORTFOLIO</div>', unsafe_allow_html=True)
st.markdown("### Transaction overview")
risk_distribution = (
    transactions["risk_band"]
    .value_counts()
    .reindex(RISK_BANDS, fill_value=0)
    .rename_axis("Risk band")
    .reset_index(name="Transactions")
)
st.bar_chart(risk_distribution, x="Risk band", y="Transactions", width="stretch")
st.markdown("</div>", unsafe_allow_html=True)

st.markdown('<div class="riskora-card">', unsafe_allow_html=True)
st.markdown('<div class="riskora-eyebrow">INVESTIGATION</div>', unsafe_allow_html=True)
st.markdown("### Inspect a transaction")

risk_filter = st.radio(
    "Risk band",
    ["All", *RISK_BANDS],
    horizontal=True,
)
filtered_transactions = transactions
if risk_filter != "All":
    filtered_transactions = transactions[transactions["risk_band"] == risk_filter]

if filtered_transactions.empty:
    st.info("No transactions match this filter.")
else:
    selected_transaction_id = st.selectbox(
        "Transaction",
        filtered_transactions["transaction_id"].tolist(),
    )
    selected_transaction = filtered_transactions[
        filtered_transactions["transaction_id"] == selected_transaction_id
    ].iloc[0]
    investigation = score_transaction(selected_transaction)
    risk_band = investigation["risk_band"]
    recommendation = RECOMMENDATIONS[risk_band]

    investigation_columns = st.columns(3)
    investigation_columns[0].metric("Rule score", investigation["risk_score"])
    investigation_columns[1].metric("Risk band", risk_band)
    investigation_columns[2].metric("Recommendation", recommendation)

    if ml_model_error:
        st.error(f"ML model unavailable: {riskora_display_error(ml_model_error)}")
    else:
        ml_risk_score = get_ml_risk_score(ml_model, selected_transaction)
        ml_decision = "REVIEW" if ml_risk_score >= ML_REVIEW_THRESHOLD else "NO_REVIEW"
        ml_columns = st.columns(2)
        ml_columns[0].metric("ML Risk Score", ml_risk_score)
        ml_columns[1].metric("ML Decision", ml_decision)
        st.info(get_combined_interpretation(risk_band, recommendation, ml_decision))

    st.markdown("#### Risk signals")
    if investigation["signals"]:
        for signal in investigation["signals"]:
            st.warning(f"**+{signal['weight']} risk points** — {signal['reason']}")
    else:
        st.success("No major suspicious signals were detected.")

    st.markdown("#### Risk reasons")
    if investigation["reasons"]:
        for reason in investigation["reasons"]:
            st.markdown(f"- {reason}")
    else:
        st.write("No major suspicious signals detected.")
st.markdown("</div>", unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# Riskora test workflows
# ---------------------------------------------------------------------------
st.markdown(
    """
    <div class="riskora-test-banner">
        <strong>🧪 Riskora Test Environment</strong><br>
        <span style="font-size:.82rem;">Safe test-only workflows. No production payment or real money is involved.</span>
    </div>
    """,
    unsafe_allow_html=True,
)

st.markdown('<div class="riskora-card">', unsafe_allow_html=True)
st.markdown('<div class="riskora-eyebrow">TEST ORDER</div>', unsafe_allow_html=True)
st.markdown("### Create a Riskora test order")
st.caption("Use this to create a provider test order that Riskora can inspect.")

test_amount = st.number_input(
    "Test order amount (₹)",
    min_value=1.0,
    value=100.0,
    step=1.0,
    key="riskora_test_order_amount",
)
if st.button("Create test order", type="primary"):
    try:
        test_order = create_test_order(
            amount=test_amount,
            currency="INR",
            receipt=f"riskora_test_{selected_transaction_id if 'selected_transaction_id' in locals() else 'demo'}",
        )
        st.session_state["razorpay_test_order_id"] = test_order["id"]
        st.success(f"Test order created: {test_order['id']}")
    except RazorpayIntegrationError as error:
        st.error(riskora_display_error(error))
st.markdown("</div>", unsafe_allow_html=True)

st.markdown('<div class="riskora-card">', unsafe_allow_html=True)
st.markdown('<div class="riskora-eyebrow">TEST PAYMENT</div>', unsafe_allow_html=True)
st.markdown("### Run a Riskora checkout test")
st.caption("Open the test checkout, complete the simulated payment, then Riskora verifies the result.")

checkout_amount = st.number_input(
    "Checkout amount (₹)",
    min_value=1.0,
    value=100.0,
    step=1.0,
    key="riskora_checkout_amount",
)
if st.button("Create checkout test", type="primary"):
    try:
        checkout_order = create_test_order(
            amount=checkout_amount,
            currency="INR",
            receipt=f"riskora_checkout_{selected_transaction_id if 'selected_transaction_id' in locals() else 'demo'}",
        )
        st.session_state["razorpay_checkout_order_id"] = checkout_order["id"]
        st.session_state["razorpay_checkout_order"] = checkout_order
        st.session_state.pop("razorpay_checkout_error", None)
        st.rerun()
    except RazorpayIntegrationError as error:
        st.error(riskora_display_error(error))

checkout_order = st.session_state.get("razorpay_checkout_order")
if checkout_order:
    st.markdown(
        f"""
        <div class="riskora-soft">
            <div style="font-weight:800;">Test order ready</div>
            <div class="riskora-muted" style="font-size:.82rem;">Order reference: {checkout_order['id']}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    checkout_key_id = os.getenv("RAZORPAY_KEY_ID", "").strip()
    if not checkout_key_id.startswith("rzp_test_"):
        st.error("Only test-mode keys are accepted for Checkout.")
    else:
        checkout_script = f"""
        <script src="https://checkout.razorpay.com/v1/checkout.js"></script>
        <button id="riskora-checkout" style="
            width:100%;padding:.8rem 1rem;border:0;border-radius:12px;
            cursor:pointer;background:linear-gradient(135deg,#6d5dfc,#0F9D8A);
            color:white;font-weight:750;font-size:15px;">
            Open Riskora Test Checkout
        </button>
        <script>
        document.getElementById("riskora-checkout").onclick = function () {{
          const options = {{
            key: {checkout_key_id!r},
            amount: {int(checkout_order["amount"])},
            currency: {str(checkout_order["currency"])!r},
            name: "Riskora",
            description: "Riskora Test Environment",
            order_id: {str(checkout_order["id"])!r},
            handler: function (response) {{
              const params = new URLSearchParams({{
                razorpay_payment_id: response.razorpay_payment_id,
                razorpay_order_id: response.razorpay_order_id,
                razorpay_signature: response.razorpay_signature
              }});
              window.top.location.href = "?" + params.toString();
            }},
            modal: {{ ondismiss: function () {{ window.alert("Riskora Test Checkout cancelled."); }} }}
          }};
          const checkout = new Razorpay(options);
          checkout.on("payment.failed", function () {{ window.alert("Riskora Test payment failed."); }});
          checkout.open();
        }};
        </script>
        """
        components.html(checkout_script, width=1200, height=720, scrolling=False)

if st.session_state.get("razorpay_checkout_error"):
    st.error(riskora_display_error(st.session_state["razorpay_checkout_error"]))

if st.session_state.get("razorpay_verified_payment"):
    verified_payment = st.session_state["razorpay_verified_payment"]
    checkout_transaction = payment_to_test_transaction(verified_payment)
    checkout_investigation = score_transaction(checkout_transaction)
    checkout_rule_band = checkout_investigation["risk_band"]
    checkout_recommendation = RECOMMENDATIONS[checkout_rule_band]

    # ---------------------------------------------------------------------------
# Riskora investigation console
# ---------------------------------------------------------------------------
latest_verified_payment = st.session_state.get("razorpay_verified_payment") or {}
raw_latest_payment_id = (
    latest_verified_payment.get("id")
    or latest_verified_payment.get("payment_id")
    or ""
)
# Razorpay payment IDs use the pay_ prefix. Never place an order_ ID into
# the Payment ID field; stale session state should not create a misleading UI.
latest_payment_id = (
    raw_latest_payment_id
    if str(raw_latest_payment_id).startswith("pay_")
    else ""
)
latest_checkout_order_id = st.session_state.get("razorpay_checkout_order_id", "")
latest_test_order_id = st.session_state.get("razorpay_test_order_id", "")
latest_order_id = latest_checkout_order_id or latest_test_order_id

# If an older session accidentally contains an order ID in the payment field,
# clear it so the user gets a clean Payment ID input.
existing_payment_lookup = str(st.session_state.get("riskora_payment_id", ""))
if existing_payment_lookup.startswith("order_"):
    st.session_state.pop("riskora_payment_id", None)

st.markdown(
    """
    <div class="riskora-card" style="margin-top:1.25rem;">
        <div class="riskora-eyebrow">🔎 INVESTIGATE</div>
        <h3 style="margin:.25rem 0 .35rem 0;">Inspect a Riskora test transaction</h3>
        <div class="riskora-muted">
            Analyze a payment or order created in the Riskora Test Lab.
            Riskora keeps the technical lookup details here so the main payment experience stays clean.
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

# Show the latest generated references so the user does not need to copy IDs
# between the Test Lab and Investigation manually.
if latest_payment_id or latest_order_id:
    st.markdown(
        f"""
        <div class="riskora-soft" style="margin-top:.75rem;">
            <div style="font-weight:800;margin-bottom:.3rem;">Latest test references</div>
            <div class="riskora-muted" style="font-size:.82rem;">
                {"Payment: " + latest_payment_id if latest_payment_id else "No verified payment yet"}
                &nbsp;&nbsp;•&nbsp;&nbsp;
                {"Order: " + latest_order_id if latest_order_id else "No test order yet"}
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

# ---------------------------------------------------------------------------
# Payment lookup
# ---------------------------------------------------------------------------
st.markdown('<div class="riskora-card">', unsafe_allow_html=True)
st.markdown('<div class="riskora-eyebrow">PAYMENT LOOKUP</div>', unsafe_allow_html=True)
st.markdown("### Analyze a test payment")
st.caption("Enter the payment ID returned after a successful Riskora test checkout. Payment IDs start with pay_.")

if latest_payment_id and not st.session_state.get("riskora_payment_id"):
    st.session_state["riskora_payment_id"] = latest_payment_id

payment_id = st.text_input(
    "Payment ID",
    key="riskora_payment_id",
    placeholder="pay_XXXXXXXXXXXX",
)

if latest_order_id:
    st.caption(f"Validation order: `{latest_order_id}`")
else:
    st.caption("Create a Riskora test order first so Riskora can validate the payment.")

if st.button("Analyze Payment", type="primary", key="riskora_analyze_payment"):
    payment_order_id = (
        st.session_state.get("razorpay_checkout_order_id")
        or st.session_state.get("razorpay_test_order_id")
        or ""
    )
    try:
        if not payment_id.strip():
            raise RazorpayIntegrationError("Enter a Riskora test payment ID first.")
        if not payment_order_id:
            raise RazorpayIntegrationError("Create a Riskora test order first so the payment can be validated.")

        test_payment = fetch_test_payment(payment_id.strip(), payment_order_id)
        test_transaction = payment_to_test_transaction(test_payment)
        test_investigation = score_transaction(test_transaction)
        test_rule_band = test_investigation["risk_band"]
        test_recommendation = RECOMMENDATIONS[test_rule_band]

        st.markdown("#### Payment investigation result")
        payment_details_columns = st.columns(4)
        payment_details_columns[0].metric("Payment ID", test_transaction["payment_id"])
        payment_details_columns[1].metric("Order ID", test_transaction["order_id"])
        payment_details_columns[2].metric("Amount", f"{test_transaction['amount']:.2f}")
        payment_details_columns[3].metric("Status", test_transaction["status"])
        st.write(f"Payment method: {test_transaction['payment_method']}")
        if test_transaction["payment_timestamp"] is not None:
            st.write(f"Payment timestamp: {test_transaction['payment_timestamp']}")

        test_risk_columns = st.columns(3)
        test_risk_columns[0].metric("Rule score", test_investigation["risk_score"])
        test_risk_columns[1].metric("Rule band", test_rule_band)
        test_risk_columns[2].metric("Recommendation", test_recommendation)
        if ml_model_error:
            st.error(f"ML model unavailable: {riskora_display_error(ml_model_error)}")
        else:
            test_ml_score = get_ml_risk_score(ml_model, test_transaction)
            st.metric("ML Risk Score", test_ml_score)
    except RazorpayIntegrationError as error:
        st.error(riskora_display_error(error))
st.markdown("</div>", unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# Order lookup
# ---------------------------------------------------------------------------
st.markdown('<div class="riskora-card">', unsafe_allow_html=True)
st.markdown('<div class="riskora-eyebrow">ORDER LOOKUP</div>', unsafe_allow_html=True)
st.markdown("### Analyze a test order")
st.caption("Use an order ID created by the Riskora Test Lab.")

if latest_order_id and not st.session_state.get("riskora_test_order_lookup"):
    st.session_state["riskora_test_order_lookup"] = latest_order_id

test_order_id = st.text_input(
    "Order ID",
    key="riskora_test_order_lookup",
    placeholder="order_XXXXXXXXXXXX",
)

if st.button("Analyze Order", type="primary", key="riskora_analyze_order"):
    try:
        if not test_order_id.strip():
            raise RazorpayIntegrationError("Enter a Riskora test order ID first.")

        test_order = fetch_test_order(test_order_id.strip())
        test_transaction = order_to_test_transaction(test_order)
        test_investigation = score_transaction(test_transaction)
        test_rule_band = test_investigation["risk_band"]
        test_recommendation = RECOMMENDATIONS[test_rule_band]

        st.markdown("#### Order investigation result")
        order_columns = st.columns(3)
        order_columns[0].metric("Order ID", test_order["id"])
        order_columns[1].metric("Amount", f"{test_transaction['amount']:.2f}")
        order_columns[2].metric("Currency", test_transaction["currency"])
        order_risk_columns = st.columns(3)
        order_risk_columns[0].metric("Rule score", test_investigation["risk_score"])
        order_risk_columns[1].metric("Rule band", test_rule_band)
        order_risk_columns[2].metric("Recommendation", test_recommendation)
        if ml_model_error:
            st.error(f"ML model unavailable: {riskora_display_error(ml_model_error)}")
        else:
            order_ml_score = get_ml_risk_score(ml_model, test_transaction)
            st.metric("ML Risk Score", order_ml_score)
    except RazorpayIntegrationError as error:
        st.error(riskora_display_error(error))
st.markdown("</div>", unsafe_allow_html=True)

st.markdown(
    '<div class="riskora-footer">Riskora Test Environment · Test-only payment workflows · No production payment is initiated by this dashboard.</div>',
    unsafe_allow_html=True,
)
