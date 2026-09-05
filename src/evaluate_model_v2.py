"""Evaluate v2 model thresholds, calibration, and time-series CV without changing it.

Run from the project root with:
    .\\razorpay.venv\\Scripts\\python.exe src/evaluate_model_v2.py
"""

from pathlib import Path
import sys

import joblib
import pandas as pd
from sklearn.base import clone
from sklearn.calibration import calibration_curve
from sklearn.metrics import (
    accuracy_score,
    brier_score_loss,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import TimeSeriesSplit

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.risk_engine import score_transactions
from src.train_model import DATA_PATH, FEATURES, MODEL_PATH, TARGET, prepare_features


THRESHOLDS = [0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80]
REVIEW_THRESHOLD = 0.70


def threshold_metrics(target: pd.Series, probabilities, threshold: float) -> dict:
    """Return payment-review operating metrics at one probability threshold."""
    predictions = (probabilities >= threshold).astype(int)
    false_positives = int(((predictions == 1) & (target.to_numpy() == 0)).sum())
    false_negatives = int(((predictions == 0) & (target.to_numpy() == 1)).sum())
    return {
        "threshold": threshold,
        "precision": precision_score(target, predictions, zero_division=0),
        "recall": recall_score(target, predictions, zero_division=0),
        "f1": f1_score(target, predictions, zero_division=0),
        "false_positives": false_positives,
        "false_negatives": false_negatives,
        "flagged_percent": 100 * predictions.mean(),
    }


def print_threshold_table(target: pd.Series, probabilities) -> None:
    """Print threshold trade-offs without selecting by a single metric."""
    print("\nThreshold analysis on the held-out chronological test set")
    print("threshold | precision | recall | f1    | false positives | false negatives | flagged")
    for threshold in THRESHOLDS:
        metrics = threshold_metrics(target, probabilities, threshold)
        print(
            f"{metrics['threshold']:.2f}      | {metrics['precision']:.3f}     | "
            f"{metrics['recall']:.3f}  | {metrics['f1']:.3f} | "
            f"{metrics['false_positives']:>15} | {metrics['false_negatives']:>15} | "
            f"{metrics['flagged_percent']:.1f}%"
        )


def print_calibration(target: pd.Series, probabilities) -> None:
    """Report a simple held-out calibration/reliability check."""
    observed, predicted = calibration_curve(target, probabilities, n_bins=5, strategy="uniform")
    probability_bins = pd.cut(probabilities, bins=[0, 0.2, 0.4, 0.6, 0.8, 1], include_lowest=True)
    bin_counts = pd.Series(probability_bins).value_counts(sort=False)
    print("\nHeld-out probability calibration (uniform bins)")
    print(f"Brier score: {brier_score_loss(target, probabilities):.3f}")
    print("predicted probability | observed fraud rate | transactions")
    for index, (mean_predicted, observed_rate) in enumerate(zip(predicted, observed)):
        print(f"{mean_predicted:.3f}                 | {observed_rate:.3f}               | {bin_counts.iloc[index]}")
    print(f"Mean predicted probability: {probabilities.mean():.3f}")
    print(f"Observed fraud rate:       {target.mean():.3f}")


def time_series_cv(train_features: pd.DataFrame, train_target: pd.Series, saved_model) -> None:
    """Estimate stability within the training period without using final test rows."""
    splitter = TimeSeriesSplit(n_splits=4)
    rows = []
    for fold, (fit_index, validation_index) in enumerate(splitter.split(train_features), start=1):
        model = clone(saved_model)
        model.fit(train_features.iloc[fit_index], train_target.iloc[fit_index])
        probabilities = model.predict_proba(train_features.iloc[validation_index])[:, 1]
        validation_target = train_target.iloc[validation_index]
        rows.append({
            "fold": fold,
            "fit_rows": len(fit_index),
            "validation_rows": len(validation_index),
            "fraud_rate": validation_target.mean(),
            "roc_auc": roc_auc_score(validation_target, probabilities),
            "brier": brier_score_loss(validation_target, probabilities),
        })
    report = pd.DataFrame(rows)
    print("\nExpanding-window time-series cross-validation (training period only)")
    print(report.to_string(index=False, formatters={
        "fraud_rate": "{:.3f}".format, "roc_auc": "{:.3f}".format, "brier": "{:.3f}".format,
    }))
    print(f"Mean CV ROC-AUC: {report['roc_auc'].mean():.3f} (range {report['roc_auc'].min():.3f}-{report['roc_auc'].max():.3f})")


def main() -> None:
    data = pd.read_csv(DATA_PATH)
    data["timestamp"] = pd.to_datetime(data["timestamp"], utc=True, errors="raise")
    data = data.sort_values("timestamp", kind="stable").reset_index(drop=True)
    split_index = int(len(data) * 0.80)
    train_rows, test_rows = data.iloc[:split_index].copy(), data.iloc[split_index:].copy()
    train_prepared, test_prepared = prepare_features(train_rows), prepare_features(test_rows)
    X_train, y_train = train_prepared[FEATURES], train_prepared[TARGET]
    X_test, y_test = test_prepared[FEATURES], test_prepared[TARGET]

    model = joblib.load(MODEL_PATH)
    probabilities = model.predict_proba(X_test)[:, 1]
    print("RazorGuard AI v2 threshold and calibration evaluation")
    print(f"Held-out test rows: {len(X_test)} | fraud rate: {y_test.mean():.2%}")
    print(f"Test ROC-AUC: {roc_auc_score(y_test, probabilities):.3f}")
    print_threshold_table(y_test, probabilities)
    print_calibration(y_test, probabilities)
    time_series_cv(X_train, y_train, model)

    selected = threshold_metrics(y_test, probabilities, REVIEW_THRESHOLD)
    rule_scored = score_transactions(test_rows)
    rule_predictions = (rule_scored["risk_score"] >= 45).astype(int)
    always_legitimate = pd.Series(0, index=y_test.index)
    print(f"\nComparison at the proposed {REVIEW_THRESHOLD:.2f} REVIEW threshold")
    print(
        f"Logistic Regression: precision={selected['precision']:.3f}, recall={selected['recall']:.3f}, "
        f"F1={selected['f1']:.3f}, flagged={selected['flagged_percent']:.1f}%"
    )
    print(
        f"Always legitimate: accuracy={accuracy_score(y_test, always_legitimate):.3f}, "
        f"recall=0.000, false negatives={int(y_test.sum())}"
    )
    print(
        f"Rule baseline (medium/high): precision={precision_score(y_test, rule_predictions, zero_division=0):.3f}, "
        f"recall={recall_score(y_test, rule_predictions, zero_division=0):.3f}, "
        f"F1={f1_score(y_test, rule_predictions, zero_division=0):.3f}, "
        f"flagged={100 * rule_predictions.mean():.1f}%"
    )


if __name__ == "__main__":
    main()
