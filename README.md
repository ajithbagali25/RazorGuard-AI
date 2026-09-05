# RazorGuard AI

## 1. Project Overview

RazorGuard AI is an AI-assisted payment risk investigation dashboard. It combines transparent rule-based suspicious-signal detection with an ML Risk Score and Razorpay Test Mode payment workflows.

This is a hackathon prototype for investigation and human decision support. It is not a production payment-risk service.

## 2. Problem

Payment teams need to investigate suspicious transactions using understandable signals while also benefiting from machine-learning assistance. Payment data alone may not contain the customer history and behavioral context needed for a complete fraud assessment.

## 3. Solution

RazorGuard AI keeps rule-based reasoning visible, adds a separate ML Risk Score, and connects Razorpay Test Mode order and payment workflows to the same investigation view. The dashboard keeps Razorpay-provided data separate from merchant or demo context.

## 4. Key Features

- 50-row mock transaction dashboard with filtering and risk distribution.
- Rule-based risk score, risk band, recommendation, reasons, and signals.
- Saved ML model inference with a 0-100 ML Risk Score.
- ML review decision at the prototype threshold of 70.
- Razorpay Test Mode order creation and lookup.
- Razorpay Test Mode payment lookup by payment ID.
- Razorpay Standard Checkout Test Mode workflow.
- Server-side Checkout signature verification.
- Payment-to-transaction mapping with feature-source information.
- Explicit `Feature Coverage: Limited` handling for unavailable fraud features.

## 5. How It Works

### Transaction/demo workflow

```text
Transaction
    -> Rule Engine
    -> Suspicious Signals
    -> ML Risk Score
    -> Human-readable risk decision
```

### Razorpay workflow

```text
Razorpay Test Mode
    -> Order/payment data
    -> Server-side validation and signature verification
    -> Payment-specific analysis
    -> Risk investigation
```

The rule engine and ML model remain independent decision-support signals. A human retains the final decision.

## 6. Rule-Based Risk Engine

The existing rule engine checks signals including:

- Multiple failed attempts
- New-customer status
- Unusual amount compared with average order value
- IP and billing-country mismatch
- Very new account
- Large first-time purchase
- Large transaction combined with failed attempts

The score is capped at 100 and mapped to:

- Below 45: `Low Risk`
- 45-74: `Medium Risk`
- 75 or higher: `High Risk`

Recommendations remain:

- Low Risk: `APPROVE`
- Medium Risk: `REVIEW`
- High Risk: `REJECT`

## 7. ML Risk Scoring

The dashboard uses the saved v2 scikit-learn pipeline at `models/razorguard_model_v2.joblib`. It uses numeric and categorical transaction features through preprocessing and logistic regression.

The displayed value is an **ML Risk Score from 0 to 100**, not a calibrated fraud probability. The prototype review threshold is 70:

- ML score below 70: `NO_REVIEW`
- ML score 70 or higher: `REVIEW`

ML does not automatically reject payments. The existing rule engine retains its independent `APPROVE` / `REVIEW` / `REJECT` recommendation.

## 8. Razorpay Integration

Razorpay integration is **Test Mode only**. The dashboard supports:

- Creating a Test Mode order.
- Fetching a Test Mode order.
- Fetching a specific Test Mode payment by payment ID.
- Verifying that a payment belongs to the expected order.
- Opening Razorpay Standard Checkout in Test Mode.
- Mapping verified payment data into the existing RazorGuard transaction shape.

The Razorpay client is isolated in `src/razorpay_client.py`. No Live Mode or production payment flow is implemented.

## 9. Payment Verification and Security

Checkout callback data is treated as untrusted browser input. Before analysis, the application:

1. Requires the callback to match the pending Test Mode order.
2. Verifies the Razorpay signature server-side using the environment secret.
3. Fetches the specific payment from Razorpay.
4. Verifies the payment/order relationship.
5. Requires an authorized or captured payment status.
6. Only then runs risk analysis.

The API secret is loaded server-side from the environment and is never sent to browser JavaScript or displayed in the dashboard. Failed, cancelled, unverified, or mismatched payments are not treated as successful.

## 10. Handling of Razorpay vs Merchant/Demo Data

Razorpay payment data can provide payment-specific values such as:

- Payment ID and related order ID
- Amount and currency
- Payment status
- Payment method when returned
- Payment timestamp when returned
- Payment failure information when returned

Razorpay does not directly provide the full merchant context used by the current risk features, such as account age, previous order count, average order value, customer history, device history, IP country, billing country, or historical failed attempts.

When those values are unavailable, the adapter keeps them marked as unavailable and supplies only clearly labelled compatibility values required by the existing model and rule interfaces. The dashboard identifies these as merchant/demo context and displays `Feature Coverage: Limited`. They must not be interpreted as verified Razorpay facts or as evidence of low risk.

## 11. Tech Stack

- Python
- Streamlit
- pandas
- scikit-learn pipeline persisted with joblib
- Razorpay Python SDK
- python-dotenv
- unittest with mocked Razorpay responses

## 12. Project Structure

```text
app.py                              Streamlit dashboard entrypoint
data/
  mock_transactions.csv             50-row demo dashboard data
  ml_training_transactions.csv      Earlier training dataset
  ml_training_transactions_v2.csv   Current v2 training dataset
models/
  razorguard_model.joblib           Earlier saved model
  razorguard_model_v2.joblib        Active dashboard model
src/
  risk_engine.py                    Rule scoring and risk bands
  signal_detection.py               Suspicious-signal rules
  ml_inference.py                   ML feature preparation and inference
  razorpay_client.py                Test Mode orders, payments, and verification
  train_model.py                    Model training utility
  evaluate_model_v2.py              Model evaluation utility
tests/
  test_razorpay_client.py           Mocked Razorpay integration tests
requirements.txt                    Python dependencies
.env.example                        Credential variable names only
```

## 13. How to Run Locally

From the project root on Windows:

```powershell
.\razorpay.venv\Scripts\python.exe -m pip install -r requirements.txt
.\razorpay.venv\Scripts\python.exe -m streamlit run app.py
```

Then open the local Streamlit URL shown in the terminal.

## 14. Test Mode Setup

Create a local `.env` file from `.env.example` and provide Razorpay Test Mode credentials there or in the environment:

```text
RAZORPAY_KEY_ID=
RAZORPAY_KEY_SECRET=
```

Use only a Razorpay Test Mode key ID beginning with `rzp_test_`. Never commit `.env`, expose the secret in the browser, or use production credentials.

## 15. Testing

Final validation included:

- 50-row demo dashboard verified.
- `TXN-1001`: rule `0` / `Low Risk` / `APPROVE`; ML `12` / `NO_REVIEW`.
- `TXN-1003`: rule `100` / `High Risk` / `REJECT`; ML `100` / `REVIEW`.
- Razorpay Test Order creation, fetch, and analysis verified.
- Razorpay Checkout opened successfully in Test Mode.
- Python syntax validation passed.
- 6/6 existing mocked integration tests passed.
- Mocked success and signature verification paths passed.
- Mocked invalid signature, wrong-order, missing-field, and API-failure paths passed.
- A browser-based successful Test Mode payment was not completed during final validation; no real money was used.

## 16. Known Limitations

- This prototype uses local mock data and a locally saved model.
- Razorpay integration is Test Mode only; no production payment processing is implemented.
- Some fraud-analysis features require merchant-side customer, device, location, and historical data that Razorpay payment data alone does not provide.
- Compatibility fallback values are required by the existing model interface and are explicitly marked as unavailable or merchant/demo context.
- The ML Risk Score is not calibrated as a probability.
- The Checkout bridge currently uses `st.components.v1.html`, which produces a non-blocking Streamlit deprecation warning but is required for the current Checkout JavaScript bridge.

## 17. Future Improvements

- Collect verified merchant-side customer, device, location, and transaction-history features before scoring.
- Replace compatibility fallbacks with persisted merchant context and feature provenance.
- Add automated end-to-end tests around the hosted Checkout callback.
- Migrate the Checkout browser bridge when Streamlit provides a supported JavaScript-capable replacement.
- Add production controls only after separate security, compliance, monitoring, and payment-review work.
