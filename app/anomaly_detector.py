"""Train and run an Isolation Forest anomaly detector."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.metrics import precision_recall_fscore_support

from app.config import (
    BUSINESS_THRESHOLDS,
    EVALUATION_METRICS_PATH,
    HISTORICAL_DATA_PATH,
    MODE_THRESHOLDS,
    MODEL_FEATURES,
    MODEL_PARAMS,
    MODEL_PATH,
    NUMERIC_FEATURES,
    PROCESSED_STREAM_PATH,
    ensure_directories,
)
from app.preprocessing import add_derived_features, clean_data, load_historical_data, prepare_features


ANOMALY_REASONS = {
    "overheat": "OVERHEAT",
    "high_vibration": "HIGH_VIBRATION",
    "high_pressure": "HIGH_PRESSURE",
    "power_surge": "POWER_SURGE",
    "motor_speed_drop": "MOTOR_SPEED_DROP",
    "multi_signal_degradation": "MULTI_SIGNAL_DEGRADATION",
    "ml_outlier": "ML_OUTLIER",
}


def _thresholds_for_mode(row: pd.Series) -> dict[str, float]:
    mode = str(row.get("operating_mode", "normal"))
    return MODE_THRESHOLDS.get(mode, BUSINESS_THRESHOLDS)


def _business_rule_points(row: pd.Series) -> int:
    thresholds = _thresholds_for_mode(row)
    points = 0
    if row["temperature"] >= thresholds["temperature_critical"]:
        points += 3
    elif row["temperature"] >= thresholds["temperature_warning"]:
        points += 1

    if row["pressure"] >= thresholds["pressure_critical"]:
        points += 3
    elif row["pressure"] >= thresholds["pressure_warning"]:
        points += 1

    if row["vibration"] >= thresholds["vibration_critical"]:
        points += 3
    elif row["vibration"] >= thresholds["vibration_warning"]:
        points += 1

    if row["power_consumption"] >= thresholds["power_critical"]:
        points += 3
    elif row["power_consumption"] >= thresholds["power_warning"]:
        points += 1

    if row["motor_speed"] <= thresholds["motor_speed_low_critical"]:
        points += 3
    elif row["motor_speed"] <= thresholds["motor_speed_low_warning"]:
        points += 1

    health_index = row.get("health_index")
    if health_index is not None and not pd.isna(health_index):
        if health_index <= BUSINESS_THRESHOLDS["health_critical"]:
            points += 2
        elif health_index <= BUSINESS_THRESHOLDS["health_warning"]:
            points += 1
    return points


def _severity_from_score(score: float, rule_points: int, is_anomaly: bool) -> str:
    if not is_anomaly and rule_points == 0:
        return "normal"
    if rule_points >= 8 or score <= -0.16:
        return "critical"
    if rule_points >= 5 or score <= -0.10:
        return "high"
    if rule_points >= 3 or score <= -0.05:
        return "medium"
    return "low"


def _anomaly_reasons(row: pd.Series, is_ml_anomaly: bool) -> list[str]:
    thresholds = _thresholds_for_mode(row)
    reasons: list[str] = []
    if row["temperature"] >= thresholds["temperature_warning"]:
        reasons.append(ANOMALY_REASONS["overheat"])
    if row["vibration"] >= thresholds["vibration_warning"]:
        reasons.append(ANOMALY_REASONS["high_vibration"])
    if row["pressure"] >= thresholds["pressure_warning"]:
        reasons.append(ANOMALY_REASONS["high_pressure"])
    if row["power_consumption"] >= thresholds["power_warning"]:
        reasons.append(ANOMALY_REASONS["power_surge"])
    if row["motor_speed"] <= thresholds["motor_speed_low_warning"]:
        reasons.append(ANOMALY_REASONS["motor_speed_drop"])
    if len(reasons) >= 2:
        reasons.append(ANOMALY_REASONS["multi_signal_degradation"])
    if is_ml_anomaly and not reasons:
        reasons.append(ANOMALY_REASONS["ml_outlier"])
    return reasons


def _explain_row(row: pd.Series) -> str:
    drivers: list[str] = []
    thresholds = _thresholds_for_mode(row)
    if row["temperature"] >= thresholds["temperature_warning"]:
        drivers.append("temperature above warning threshold")
    if row["pressure"] >= thresholds["pressure_warning"]:
        drivers.append("pressure above warning threshold")
    if row["vibration"] >= thresholds["vibration_warning"]:
        drivers.append("vibration above warning threshold")
    if row["power_consumption"] >= thresholds["power_warning"]:
        drivers.append("power consumption above warning threshold")
    if row["motor_speed"] <= thresholds["motor_speed_low_warning"]:
        drivers.append("motor speed below warning threshold")
    if not drivers:
        drivers.append("unusual multivariate pattern detected by Isolation Forest")
    return (
        f"Severity {row['anomaly_severity']}: {', '.join(drivers)} "
        f"(score={row['anomaly_score']:.5f})."
    )


def train_model(data_path: str | Path = HISTORICAL_DATA_PATH, model_path: str | Path = MODEL_PATH) -> dict[str, Any]:
    """Train and persist the anomaly detector plus scaler."""
    ensure_directories()
    df = add_derived_features(load_historical_data(data_path))
    scaled_features, scaler = prepare_features(df, fit=True)
    model = IsolationForest(**MODEL_PARAMS)
    model.fit(scaled_features)

    artifact = {
        "model": model,
        "scaler": scaler,
        "features": MODEL_FEATURES,
        "model_params": MODEL_PARAMS,
    }
    joblib.dump(artifact, model_path)
    if "status" in df.columns:
        metrics = evaluate_model(df, artifact=artifact)
        metrics_path = (
            EVALUATION_METRICS_PATH
            if Path(model_path).resolve() == MODEL_PATH.resolve()
            else Path(model_path).with_name("evaluation_metrics.json")
        )
        metrics_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    return artifact


def load_model(model_path: str | Path = MODEL_PATH) -> dict[str, Any]:
    """Load the persisted anomaly detector artifact."""
    if not Path(model_path).exists():
        raise FileNotFoundError(
            f"No model found at {model_path}. Run python -m app.anomaly_detector --train first."
        )
    return joblib.load(model_path)


def predict_anomalies(df: pd.DataFrame, artifact: dict[str, Any] | None = None) -> pd.DataFrame:
    """Predict anomaly labels and raw Isolation Forest scores."""
    artifact = artifact or load_model()
    cleaned = add_derived_features(df)
    scaled_features, _ = prepare_features(cleaned, scaler=artifact["scaler"], fit=False)
    predictions = artifact["model"].predict(scaled_features)
    scores = artifact["model"].decision_function(scaled_features)

    output = cleaned.copy()
    output["anomaly_prediction"] = predictions
    output["anomaly_score"] = np.round(scores, 5)
    output["is_anomaly"] = predictions == -1
    return output


def enrich_with_anomaly_columns(df: pd.DataFrame, artifact: dict[str, Any] | None = None) -> pd.DataFrame:
    """Add anomaly prediction, score, boolean flag, and severity."""
    enriched = predict_anomalies(df, artifact=artifact)
    rule_points = enriched.apply(_business_rule_points, axis=1)
    ml_flags = enriched["is_anomaly"].copy()
    enriched["is_anomaly"] = enriched["is_anomaly"] | (rule_points >= 3)
    enriched["anomaly_severity"] = [
        _severity_from_score(score, int(points), bool(is_anomaly))
        for score, points, is_anomaly in zip(
            enriched["anomaly_score"],
            rule_points,
            enriched["is_anomaly"],
        )
    ]
    enriched["anomaly_reason"] = [
        ",".join(_anomaly_reasons(row, bool(is_ml_anomaly))) if bool(is_anomaly) else "NORMAL"
        for (_, row), is_ml_anomaly, is_anomaly in zip(enriched.iterrows(), ml_flags, enriched["is_anomaly"])
    ]
    enriched["explanation"] = enriched.apply(
        lambda row: _explain_row(row) if row["is_anomaly"] else "",
        axis=1,
    )
    return enriched


def evaluate_model(df: pd.DataFrame, artifact: dict[str, Any] | None = None) -> dict[str, float]:
    """Evaluate against simulated labels when available."""
    if "status" not in df.columns:
        return {}
    predictions = enrich_with_anomaly_columns(df, artifact=artifact)
    y_true = (predictions["status"] != "normal").astype(int)
    y_pred = predictions["is_anomaly"].astype(int)
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true,
        y_pred,
        average="binary",
        zero_division=0,
    )
    return {
        "precision": round(float(precision), 4),
        "recall": round(float(recall), 4),
        "f1_score": round(float(f1), 4),
        "evaluated_rows": int(len(predictions)),
        "simulated_positive_labels": int(y_true.sum()),
    }


def _predict_processed_file() -> pd.DataFrame:
    if not PROCESSED_STREAM_PATH.exists():
        raise FileNotFoundError(
            f"No processed stream file found at {PROCESSED_STREAM_PATH}. "
            "Run python -m app.pathway_pipeline first, or train on historical data."
        )
    df = pd.read_csv(PROCESSED_STREAM_PATH)
    enriched = enrich_with_anomaly_columns(df)
    output_path = PROCESSED_STREAM_PATH.with_name("stream_anomalies.csv")
    enriched.to_csv(output_path, index=False)
    from app.database import insert_anomalies, insert_sensor_events

    inserted_events = insert_sensor_events(enriched)
    inserted_anomalies = insert_anomalies(enriched)
    print(f"Saved anomaly predictions to {output_path}")
    print(f"Inserted {inserted_events} events and {inserted_anomalies} anomalies into SQLite")
    return enriched


def main() -> None:
    parser = argparse.ArgumentParser(description="Train or run anomaly detection.")
    parser.add_argument("--train", action="store_true", help="Train the Isolation Forest model.")
    parser.add_argument("--predict", action="store_true", help="Predict anomalies on processed stream data.")
    args = parser.parse_args()

    if args.predict:
        _predict_processed_file()
    else:
        artifact = train_model()
        print(f"Model trained with features: {', '.join(artifact['features'])}")
        print(f"Saved model artifact to {MODEL_PATH}")
        if EVALUATION_METRICS_PATH.exists():
            print(f"Saved evaluation metrics to {EVALUATION_METRICS_PATH}")


if __name__ == "__main__":
    main()
