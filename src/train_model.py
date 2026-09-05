"""Train and evaluate RazorGuard AI's v2 fraud model with a time-based split.

Run from the project root with:
    .\\razorpay.venv\\Scripts\\python.exe src/train_model.py
"""

from pathlib import Path
import sys

import joblib
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_score, recall_score, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

# Allow direct execution from the project root while preserving package imports.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.risk_engine import score_transactions


DATA_PATH = PROJECT_ROOT / "data" / "ml_training_transactions_v2.csv"
MODEL_PATH = PROJECT_ROOT / "models" / "razorguard_model_v2.joblib"
TEST_FRACTION = 0.20
RANDOM_STATE = 42

NUMERIC_FEATURES = [
    "amount", "failed_attempts", "account_age_days", "previous_order_count",
    "avg_order_value", "is_new_customer", "country_mismatch", "amount_to_avg_ratio",
]
CATEGORICAL_FEATURES = ["payment_method", "device_type", "ip_country", "billing_country"]
FEATURES = NUMERIC_FEATURES + CATEGORICAL_FEATURES
TARGET = "is_fraud"
RAW_REQUIRED_COLUMNS = set([
    "transaction_id", "customer_id", "timestamp", "amount", "payment_method", "device_type",
    "ip_country", "billing_country", "failed_attempts", "account_age_days",
    "previous_order_count", "avg_order_value", "is_new_customer", "country_mismatch",
    "amount_to_avg_ratio", TARGET,
])


def prepare_features(dataframe: pd.DataFrame) -> pd.DataFrame:
    """Create permitted inputs; IDs, timestamp, label, and rule outputs stay excluded."""
    prepared = dataframe.copy()
    prepared["is_new_customer"] = (
        prepared["is_new_customer"].astype(str).str.lower().eq("true").astype(int)
    )
    # Recompute the field to protect against a stale derived value in the CSV.
    prepared["country_mismatch"] = (
        prepared["ip_country"] != prepared["billing_country"]
    ).astype(int)
    return prepared


def calculate_metrics(name: str, target: pd.Series, predictions, probabilities) -> dict:
    """Evaluate a classifier with emphasis on the fraud (positive) class."""
    matrix = confusion_matrix(target, predictions, labels=[0, 1])
    return {
        "model": name,
        "accuracy": accuracy_score(target, predictions),
        "precision": precision_score(target, predictions, zero_division=0),
        "recall": recall_score(target, predictions, zero_division=0),
        "f1": f1_score(target, predictions, zero_division=0),
        "roc_auc": roc_auc_score(target, probabilities),
        "confusion_matrix": matrix.tolist(),
        "false_positives": int(matrix[0, 1]),
        "false_negatives": int(matrix[1, 0]),
    }


def evaluate_estimator(name: str, model, features: pd.DataFrame, target: pd.Series) -> dict:
    """Evaluate an sklearn estimator that exposes fraud probabilities."""
    return calculate_metrics(name, target, model.predict(features), model.predict_proba(features)[:, 1])


def evaluate_rule_baseline(test_rows: pd.DataFrame, target: pd.Series) -> dict:
    """Score existing rules for comparison only, never as ML input features."""
    scored = score_transactions(test_rows)
    scores = scored["risk_score"] / 100.0
    # Medium- and high-risk rule bands are treated as a fraud alert.
    predictions = (scored["risk_score"] >= 45).astype(int)
    return calculate_metrics("Rule baseline: medium/high risk alert", target, predictions, scores)


def print_metrics(metrics: dict) -> None:
    """Print a compact report that makes error types explicit."""
    print(f"\n{metrics['model']}")
    print(f"Accuracy:  {metrics['accuracy']:.3f}")
    print(f"Precision: {metrics['precision']:.3f}")
    print(f"Recall:    {metrics['recall']:.3f}")
    print(f"F1 score:  {metrics['f1']:.3f}")
    print(f"ROC-AUC:   {metrics['roc_auc']:.3f}")
    print("Confusion matrix [[true negatives, false positives], [false negatives, true positives]]:")
    print(metrics["confusion_matrix"])
    print(f"False positives: {metrics['false_positives']}")
    print(f"False negatives: {metrics['false_negatives']}")


def print_split_summary(name: str, rows: pd.DataFrame, target: pd.Series) -> None:
    """Display time range and fraud class balance for one chronological partition."""
    print(
        f"{name}: {len(rows)} rows | "
        f"{rows['timestamp'].min().isoformat()} to {rows['timestamp'].max().isoformat()} | "
        f"{int(target.sum())} fraud ({target.mean():.2%})"
    )


def main() -> None:
    if not DATA_PATH.exists():
        raise FileNotFoundError(f"Training data not found: {DATA_PATH}")

    data = pd.read_csv(DATA_PATH)
    missing_columns = RAW_REQUIRED_COLUMNS - set(data.columns)
    if missing_columns:
        raise ValueError(f"Training data is missing columns: {sorted(missing_columns)}")
    if not set(data[TARGET].unique()).issubset({0, 1}):
        raise ValueError("is_fraud must contain only 0 and 1.")

    data["timestamp"] = pd.to_datetime(data["timestamp"], utc=True, errors="raise")
    data = data.sort_values("timestamp", kind="stable").reset_index(drop=True)
    split_index = int(len(data) * (1 - TEST_FRACTION))
    train_rows, test_rows = data.iloc[:split_index].copy(), data.iloc[split_index:].copy()
    prepared_train, prepared_test = prepare_features(train_rows), prepare_features(test_rows)
    X_train, y_train = prepared_train[FEATURES], prepared_train[TARGET]
    X_test, y_test = prepared_test[FEATURES], prepared_test[TARGET]

    preprocessor = ColumnTransformer([
        ("numeric", StandardScaler(), NUMERIC_FEATURES),
        ("categorical", OneHotEncoder(handle_unknown="ignore"), CATEGORICAL_FEATURES),
    ])
    model_pipeline = Pipeline([
        ("preprocessor", preprocessor),
        ("classifier", LogisticRegression(max_iter=2000, class_weight="balanced", random_state=RANDOM_STATE)),
    ])
    baseline = DummyClassifier(strategy="most_frequent")
    baseline.fit(X_train, y_train)
    model_pipeline.fit(X_train, y_train)

    print("RazorGuard AI v2 supervised fraud-model evaluation")
    print(f"Features used: {', '.join(FEATURES)}")
    print("Time-based split: first 80% of chronologically ordered transactions train; final 20% test.")
    print_split_summary("Training", train_rows, y_train)
    print_split_summary("Test", test_rows, y_test)
    print_metrics(evaluate_estimator("Baseline: always legitimate", baseline, X_test, y_test))
    print_metrics(evaluate_rule_baseline(test_rows, y_test))
    print_metrics(evaluate_estimator("Logistic Regression", model_pipeline, X_test, y_test))

    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model_pipeline, MODEL_PATH)
    print(f"\nSaved trained pipeline to: {MODEL_PATH}")


if __name__ == "__main__":
    main()
