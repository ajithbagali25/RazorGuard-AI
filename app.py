from pathlib import Path
import os

import joblib
import pandas as pd
import streamlit as st
import streamlit.components.v1 as components

from src.ml_inference import get_ml_risk_score
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
RISK_BANDS = ["Low Risk", "Medium Risk", "High Risk"]
RECOMMENDATIONS = {
    "Low Risk": "APPROVE",
    "Medium Risk": "REVIEW",
    "High Risk": "REJECT",
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


st.set_page_config(page_title="RazorGuard AI", page_icon="\U0001F6E1\ufe0f", layout="wide")

st.markdown(
    """
    <style>
    :root {
        --rg-ink: #18212b;
        --rg-muted: #5b6875;
        --rg-line: #d9e0e6;
        --rg-panel: #f6f8fa;
        --rg-green: #19734a;
        --rg-amber: #9a6500;
        --rg-red: #b42318;
    }
    .block-container { padding-top: 2.5rem; padding-bottom: 3rem; }
    h1, h2, h3, h4 { color: var(--rg-ink); }
    [data-testid="stMetric"] {
        background: var(--rg-panel);
        border: 1px solid var(--rg-line);
        border-radius: 6px;
        padding: 0.8rem 1rem;
    }
    [data-testid="stMetricLabel"] { color: var(--rg-muted); }
    [data-testid="stMetricValue"] { color: var(--rg-ink); }
    </style>
    """,
    unsafe_allow_html=True,
)

st.title("RazorGuard AI")
st.caption("Payment risk investigation with transparent rule-based and ML decision support")

transactions = load_transactions()
ml_model, ml_model_error = load_ml_model()

checkout_payment_id = st.query_params.get("razorpay_payment_id")
checkout_order_id = st.query_params.get("razorpay_order_id")
checkout_signature = st.query_params.get("razorpay_signature")
if checkout_payment_id or checkout_order_id or checkout_signature:
    pending_order_id = st.session_state.get("razorpay_checkout_order_id")
    try:
        if not pending_order_id or checkout_order_id != pending_order_id:
            raise RazorpayIntegrationError("Razorpay Checkout returned an unexpected order.")
        verify_test_payment_signature(checkout_payment_id or "", checkout_order_id or "", checkout_signature or "")
        verified_payment = fetch_test_payment(checkout_payment_id or "", pending_order_id)
        if verified_payment.get("status") not in {"authorized", "captured"}:
            raise RazorpayIntegrationError("Razorpay payment was not successfully authorized or captured.")
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

total_transactions = len(transactions)
band_counts = transactions["risk_band"].value_counts()
high_risk_count = int(band_counts.get("High Risk", 0))
medium_risk_count = int(band_counts.get("Medium Risk", 0))
low_risk_count = int(band_counts.get("Low Risk", 0))
overall_risk_percentage = ((high_risk_count + medium_risk_count) / total_transactions) * 100

summary_columns = st.columns(5)
summary_columns[0].metric("Total transactions", total_transactions)
summary_columns[1].metric("High-risk transactions", high_risk_count)
summary_columns[2].metric("Medium-risk transactions", medium_risk_count)
summary_columns[3].metric("Low-risk transactions", low_risk_count)
summary_columns[4].metric("Overall risk percentage", f"{overall_risk_percentage:.1f}%")

st.caption("Overall risk percentage = high-risk and medium-risk transactions as a share of all transactions.")

st.header("Transaction Analysis")
st.subheader("Portfolio overview")
risk_distribution = (
    transactions["risk_band"]
    .value_counts()
    .reindex(RISK_BANDS, fill_value=0)
    .rename_axis("Risk band")
    .reset_index(name="Transactions")
)
st.bar_chart(risk_distribution, x="Risk band", y="Transactions", width="stretch")

st.subheader("Transactions")
risk_filter = st.selectbox(
    "Filter by risk band",
    ["All", *RISK_BANDS],
)

filtered_transactions = transactions
if risk_filter != "All":
    filtered_transactions = transactions[transactions["risk_band"] == risk_filter]

table_columns = [
    "transaction_id",
    "amount",
    "payment_method",
    "failed_attempts",
    "account_age_days",
    "risk_score",
    "risk_band",
]
st.dataframe(
    filtered_transactions[table_columns],
    column_config={
        "transaction_id": "Transaction ID",
        "amount": st.column_config.NumberColumn("Amount", format="\u20b9%.2f"),
        "payment_method": "Payment method",
        "failed_attempts": "Failed attempts",
        "account_age_days": "Account age (days)",
        "risk_score": st.column_config.NumberColumn("Risk score", format="%d"),
        "risk_band": "Risk band",
    },
    hide_index=True,
    width="stretch",
)

st.divider()
st.subheader("Risk Investigation")
st.caption("Inspect one transaction at a time using the existing rule engine and ML model.")
if filtered_transactions.empty:
    st.info("No transactions match this risk-band filter. Choose another filter to investigate a transaction.")
    st.stop()

selected_transaction_id = st.selectbox(
    "Choose a transaction to inspect",
    filtered_transactions["transaction_id"].tolist(),
)
selected_transaction = filtered_transactions[
    filtered_transactions["transaction_id"] == selected_transaction_id
].iloc[0]
investigation = score_transaction(selected_transaction)
risk_band = investigation["risk_band"]
recommendation = RECOMMENDATIONS[risk_band]

st.markdown("#### Rule-Based Risk")
investigation_columns = st.columns(3)
investigation_columns[0].metric("Risk score", investigation["risk_score"])
investigation_columns[1].metric("Risk band", risk_band)
investigation_columns[2].metric("Recommendation", recommendation)

if ml_model_error:
    st.error(f"ML model unavailable: {ml_model_error}")
else:
    st.markdown("#### ML Risk")
    ml_risk_score = get_ml_risk_score(ml_model, selected_transaction)
    ml_decision = "REVIEW" if ml_risk_score >= ML_REVIEW_THRESHOLD else "NO_REVIEW"
    ml_columns = st.columns(2)
    ml_columns[0].metric("ML Risk Score", ml_risk_score)
    ml_columns[1].metric("ML Decision", ml_decision)
    st.info(get_combined_interpretation(risk_band, recommendation, ml_decision))

st.markdown(f"#### Transaction {selected_transaction['transaction_id']}")
st.write(f"Customer: {selected_transaction['customer_id']}")
st.write(f"Amount: \u20b9{selected_transaction['amount']:.2f}")

st.markdown("#### Suspicious Signals & Explanation")
st.caption("Rule-based signals explain why this transaction was flagged.")
if investigation["signals"]:
    for signal in investigation["signals"]:
        st.warning(f"**+{signal['weight']} risk points** - {signal['reason']}")
else:
    st.success("No major suspicious signals were detected for this transaction.")

st.markdown("#### Risk Reasons")
if investigation["reasons"]:
    for reason in investigation["reasons"]:
        st.markdown(f"- {reason}")
else:
    st.write("No major suspicious signals detected.")

st.markdown("#### Transaction Details")
transaction_details = {
    "transaction_id": selected_transaction["transaction_id"],
    "customer_id": selected_transaction["customer_id"],
    "amount": selected_transaction["amount"],
    "payment_method": selected_transaction["payment_method"],
    "device_type": selected_transaction["device_type"],
    "ip_country": selected_transaction["ip_country"],
    "billing_country": selected_transaction["billing_country"],
    "failed_attempts": selected_transaction["failed_attempts"],
    "account_age_days": selected_transaction["account_age_days"],
    "previous_order_count": selected_transaction["previous_order_count"],
    "avg_order_value": selected_transaction["avg_order_value"],
    "is_new_customer": selected_transaction["is_new_customer"],
}
st.json(to_native_value(transaction_details))

st.divider()
st.header("Razorpay Test Mode Workflows")
st.subheader("Razorpay Test Order")
st.caption("TEST MODE ONLY - No production payments or automatic fraud rejection.")
st.caption("Test Mode - no real payment or real money involved.")
test_amount = st.number_input("Test order amount (INR)", min_value=1.0, value=100.0, step=1.0)
if st.button("Create test order"):
    try:
        test_order = create_test_order(
            amount=test_amount,
            currency="INR",
            receipt=f"razorguard_test_{selected_transaction['transaction_id']}",
        )
        st.session_state["razorpay_test_order_id"] = test_order["id"]
        st.success(f"Test order created: {test_order['id']}")
    except RazorpayIntegrationError as error:
        st.error(str(error))

st.subheader("Razorpay Test Payment")
st.caption("RAZORPAY TEST MODE - SIMULATED PAYMENT")
st.caption("Test Mode - no real payment or real money involved.")
st.caption("Payment data from Razorpay is shown separately from merchant/demo context.")
checkout_amount = st.number_input("Checkout amount (INR)", min_value=1.0, value=100.0, step=1.0)
if st.button("Create Checkout test order"):
    try:
        checkout_order = create_test_order(
            amount=checkout_amount,
            currency="INR",
            receipt=f"razorguard_checkout_{selected_transaction['transaction_id']}",
        )
        st.session_state["razorpay_checkout_order_id"] = checkout_order["id"]
        st.session_state["razorpay_checkout_order"] = checkout_order
        st.session_state.pop("razorpay_checkout_error", None)
        st.rerun()
    except RazorpayIntegrationError as error:
        st.error(str(error))

checkout_order = st.session_state.get("razorpay_checkout_order")
if checkout_order:
    st.write(f"Checkout test order: {checkout_order['id']}")
    checkout_key_id = os.getenv("RAZORPAY_KEY_ID", "").strip()
    if not checkout_key_id.startswith("rzp_test_"):
        st.error("Only Razorpay Test Mode keys are accepted for Checkout.")
    else:
        checkout_script = f"""
        <script src="https://checkout.razorpay.com/v1/checkout.js"></script>
        <button id="razorguard-checkout" style="padding:0.6rem 1rem;cursor:pointer;">Open Razorpay Test Checkout</button>
        <script>
        document.getElementById("razorguard-checkout").onclick = function () {{
          const options = {{
            key: {checkout_key_id!r},
            amount: {int(checkout_order["amount"])},
            currency: {str(checkout_order["currency"])!r},
            name: "RazorGuard AI",
            description: "TEST MODE ONLY",
            order_id: {str(checkout_order["id"])!r},
            handler: function (response) {{
              const params = new URLSearchParams({{
                razorpay_payment_id: response.razorpay_payment_id,
                razorpay_order_id: response.razorpay_order_id,
                razorpay_signature: response.razorpay_signature
              }});
              window.top.location.href = "?" + params.toString();
            }},
            modal: {{ ondismiss: function () {{ window.alert("Razorpay Checkout cancelled."); }} }}
          }};
          const checkout = new Razorpay(options);
          checkout.on("payment.failed", function () {{ window.alert("Razorpay Test payment failed."); }});
          checkout.open();
        }};
        </script>
        """
        components.html(checkout_script, width=1200, height=760, scrolling=False)

if st.session_state.get("razorpay_checkout_error"):
    st.error(st.session_state["razorpay_checkout_error"])
if st.session_state.get("razorpay_verified_payment"):
    verified_payment = st.session_state["razorpay_verified_payment"]
    checkout_transaction = payment_to_test_transaction(verified_payment)
    checkout_investigation = score_transaction(checkout_transaction)
    checkout_rule_band = checkout_investigation["risk_band"]
    checkout_recommendation = RECOMMENDATIONS[checkout_rule_band]
    st.markdown("#### Verified Razorpay Test Transaction")
    st.success("Payment verified server-side and matched to the created order.")
    st.caption("Razorpay Payment Data")
    verified_columns = st.columns(4)
    verified_columns[0].metric("Payment Status", checkout_transaction["status"])
    verified_columns[1].metric("Payment ID", checkout_transaction["payment_id"])
    verified_columns[2].metric("Order ID", checkout_transaction["order_id"])
    verified_columns[3].metric("Amount", f"{checkout_transaction['amount']:.2f} {checkout_transaction['currency']}")
    st.write(f"Payment method: {checkout_transaction['payment_method']}")
    st.write(f"Payment timestamp: {checkout_transaction['payment_timestamp']}")
    st.markdown("##### Rule-Based Risk")
    verified_risk_columns = st.columns(5)
    verified_risk_columns[0].metric("Rule Risk Score", checkout_investigation["risk_score"])
    verified_risk_columns[1].metric("Rule Risk Band", checkout_rule_band)
    verified_risk_columns[2].metric("Rule Recommendation", checkout_recommendation)
    if ml_model_error:
        st.error(f"ML model unavailable: {ml_model_error}")
    else:
        st.markdown("##### ML Risk")
        checkout_ml_score = get_ml_risk_score(ml_model, checkout_transaction)
        checkout_ml_decision = "REVIEW" if checkout_ml_score >= ML_REVIEW_THRESHOLD else "NO_REVIEW"
        verified_risk_columns[3].metric("ML Risk Score", checkout_ml_score)
        verified_risk_columns[4].metric("ML Decision", checkout_ml_decision)
        st.caption("ML Risk Score is a model score, not a calibrated fraud probability.")
    st.write(f"Feature Coverage: {checkout_transaction['feature_coverage']}")
    st.write("Feature source information")
    st.json(checkout_transaction["feature_sources"])
    st.write("Risk reasons/signals")
    if checkout_investigation["signals"]:
        for signal in checkout_investigation["signals"]:
            st.warning(f"**+{signal['weight']} risk points** - {signal['reason']}")
    else:
        st.info("No rule-based suspicious signals were detected from available verified payment data.")

payment_id = st.text_input("Razorpay test payment ID to analyze")
payment_order_id = st.text_input(
    "Expected Razorpay test order ID",
    value=st.session_state.get("razorpay_test_order_id", ""),
)
if st.button("Analyze Razorpay test payment"):
    try:
        razorpay_payment = fetch_test_payment(payment_id, payment_order_id)
        razorpay_transaction = payment_to_test_transaction(razorpay_payment)
        payment_investigation = score_transaction(razorpay_transaction)
        payment_rule_band = payment_investigation["risk_band"]
        payment_recommendation = RECOMMENDATIONS[payment_rule_band]

        st.markdown("#### Razorpay Test Transaction")
        st.caption("Test Mode - no real payment or real money involved.")
        st.caption("Razorpay Payment Data")
        payment_details_columns = st.columns(4)
        payment_details_columns[0].metric("Payment ID", razorpay_transaction["payment_id"])
        payment_details_columns[1].metric("Order ID", razorpay_transaction["order_id"])
        payment_details_columns[2].metric("Amount", f"{razorpay_transaction['amount']:.2f}")
        payment_details_columns[3].metric("Currency", razorpay_transaction["currency"])
        st.write(f"Payment status: {razorpay_transaction['status']}")
        st.write(f"Payment method: {razorpay_transaction['payment_method']}")
        if razorpay_transaction["payment_timestamp"] is not None:
            st.write(f"Payment timestamp: {razorpay_transaction['payment_timestamp']}")
        if razorpay_transaction["payment_error_code"]:
            st.warning(f"Payment failure code: {razorpay_transaction['payment_error_code']}")
        if razorpay_transaction["payment_error_description"]:
            st.warning(f"Payment failure information: {razorpay_transaction['payment_error_description']}")

        st.markdown("##### Rule-Based Risk")
        payment_risk_columns = st.columns(5)
        payment_risk_columns[0].metric("Rule Risk Score", payment_investigation["risk_score"])
        payment_risk_columns[1].metric("Rule Risk Band", payment_rule_band)
        payment_risk_columns[2].metric("Rule Recommendation", payment_recommendation)
        if ml_model_error:
            st.error(f"ML model unavailable: {ml_model_error}")
        else:
            st.markdown("##### ML Risk")
            payment_ml_score = get_ml_risk_score(ml_model, razorpay_transaction)
            payment_ml_decision = "REVIEW" if payment_ml_score >= ML_REVIEW_THRESHOLD else "NO_REVIEW"
            payment_risk_columns[3].metric("ML Risk Score", payment_ml_score)
            payment_risk_columns[4].metric("ML Decision", payment_ml_decision)
            st.caption("MODEL RISK SCORE is a 0-100 model score, not a fraud probability.")

        st.write(f"Feature Coverage: {razorpay_transaction['feature_coverage']}")
        st.write("Feature sources")
        st.json(razorpay_transaction["feature_sources"])
        st.write("Merchant/Demo Context")
        st.write("Compatibility fallback sources")
        st.json(razorpay_transaction["compatibility_fallback_sources"])
        st.write("Derived fields")
        st.write(", ".join(razorpay_transaction["derived_fields"]))
        st.write("Unavailable fraud features")
        st.write(", ".join(razorpay_transaction["unavailable_fraud_features"]))
        st.write("Suspicious Signals & Explanation")
        if payment_investigation["signals"]:
            for signal in payment_investigation["signals"]:
                st.warning(f"**+{signal['weight']} risk points** - {signal['reason']}")
        else:
            st.info("No rule-based suspicious signals were detected from the available payment data.")
        st.caption("Compatibility fallback values are not Razorpay payment facts and must not be treated as evidence of low risk.")
    except RazorpayIntegrationError as error:
        st.error(str(error))

test_order_id = st.text_input(
    "Razorpay test order ID to analyze",
    value=st.session_state.get("razorpay_test_order_id", ""),
)
if st.button("Analyze Razorpay test order"):
    try:
        razorpay_order = fetch_test_order(test_order_id)
        razorpay_transaction = order_to_test_transaction(razorpay_order)
        razorpay_investigation = score_transaction(razorpay_transaction)
        razorpay_rule_band = razorpay_investigation["risk_band"]
        razorpay_recommendation = RECOMMENDATIONS[razorpay_rule_band]
        if ml_model_error:
            st.error(f"ML model unavailable: {ml_model_error}")
        else:
            razorpay_ml_score = get_ml_risk_score(ml_model, razorpay_transaction)
            razorpay_ml_decision = "REVIEW" if razorpay_ml_score >= ML_REVIEW_THRESHOLD else "NO_REVIEW"
            st.markdown("#### Razorpay Test Transaction")
            razorpay_order_columns = st.columns(3)
            razorpay_order_columns[0].metric("Razorpay Test Order ID", razorpay_order["id"])
            razorpay_order_columns[1].metric("Amount", f"{razorpay_transaction['amount']:.2f}")
            razorpay_order_columns[2].metric("Currency", razorpay_transaction["currency"])
            st.markdown("##### Rule-Based Risk")
            razorpay_risk_columns = st.columns(5)
            razorpay_risk_columns[0].metric("Rule Risk Score", razorpay_investigation["risk_score"])
            razorpay_risk_columns[1].metric("Rule Risk Band", razorpay_rule_band)
            razorpay_risk_columns[2].metric("Rule Recommendation", razorpay_recommendation)
            razorpay_risk_columns[3].metric("ML Risk Score", razorpay_ml_score)
            razorpay_risk_columns[4].metric("ML Decision", razorpay_ml_decision)
            st.markdown("##### ML Risk")
            st.caption("MODEL RISK SCORE is a 0-100 model score, not a fraud probability.")
            st.info(get_combined_interpretation(razorpay_rule_band, razorpay_recommendation, razorpay_ml_decision))
            st.write("Risk reasons/signals")
            if razorpay_investigation["signals"]:
                for signal in razorpay_investigation["signals"]:
                    st.warning(f"**+{signal['weight']} risk points** - {signal['reason']}")
            else:
                st.success("No rule-based suspicious signals were detected.")
            st.caption(
                "The order did not provide fraud-analysis fields; these fields use demo defaults: "
                + ", ".join(razorpay_transaction["demo_default_fields"])
            )
    except RazorpayIntegrationError as error:
        st.error(str(error))
