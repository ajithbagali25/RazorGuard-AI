from src.signal_detection import detect_suspicious_signals


def get_risk_band(score):
    """Convert a numeric risk score into a readable band."""
    if score >= 75:
        return "High Risk"
    if score >= 45:
        return "Medium Risk"
    return "Low Risk"


def score_transaction(row):
    """Calculate risk score and reasons for one transaction."""
    signals = detect_suspicious_signals(row)
    total_score = sum(signal["weight"] for signal in signals)
    total_score = min(total_score, 100)

    return {
        "risk_score": total_score,
        "risk_band": get_risk_band(total_score),
        "signals": signals,
        "reasons": [signal["reason"] for signal in signals],
    }


def score_transactions(df):
    """Create a scored copy of the transaction DataFrame."""
    scored = df.copy()
    scored["risk_score"] = 0
    scored["risk_band"] = "Low Risk"
    scored["risk_reasons"] = ""

    for index, row in scored.iterrows():
        result = score_transaction(row)
        scored.at[index, "risk_score"] = result["risk_score"]
        scored.at[index, "risk_band"] = result["risk_band"]
        scored.at[index, "risk_reasons"] = "; ".join(result["reasons"]) if result["reasons"] else "No major suspicious signals detected."

    return scored
