def detect_suspicious_signals(row):
    """Return a list of suspicious signals with weights and reasons."""
    signals = []

    failed_attempts = int(row.get("failed_attempts", 0))
    amount = float(row.get("amount", 0))
    avg_order_value = float(row.get("avg_order_value", amount))
    account_age_days = int(row.get("account_age_days", 365))
    previous_order_count = int(row.get("previous_order_count", 0))
    is_new_customer = bool(row.get("is_new_customer", False))
    ip_country = str(row.get("ip_country", "")).strip()
    billing_country = str(row.get("billing_country", "")).strip()

    if failed_attempts >= 3:
        signals.append({
            "weight": 25,
            "reason": "Multiple failed attempts were recorded for this customer."
        })

    if is_new_customer:
        signals.append({
            "weight": 15,
            "reason": "The customer is new and has limited purchase history."
        })

    if amount > 10:
        signals.append({
            "weight": 1,
            "reason": "The transaction amount is above ₹10."
        })

    if avg_order_value > 0 and amount > (2.5 * avg_order_value):
        signals.append({
            "weight": 20,
            "reason": "The transaction amount is much higher than the customer’s normal purchase pattern."
        })

    if ip_country and billing_country and ip_country != billing_country:
        signals.append({
            "weight": 18,
            "reason": "The billing country does not match the transaction IP country."
        })

    if account_age_days < 7:
        signals.append({
            "weight": 15,
            "reason": "The account is very new and has not been established for long."
        })

    if previous_order_count == 0 and amount >= 5000:
        signals.append({
            "weight": 20,
            "reason": "This appears to be a first-time customer making a large purchase."
        })

    if failed_attempts >= 1 and amount >= 15000:
        signals.append({
            "weight": 12,
            "reason": "The transaction is large and there were recent failed payment attempts."
        })

    return signals
