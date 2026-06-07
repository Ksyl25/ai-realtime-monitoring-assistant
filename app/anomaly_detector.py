"""Train and run an Isolation Forest anomaly detector."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest

from app.config import (
    BUSINESS_THRESHOLDS,
    HISTORICAL_DATA_PATH,
    MODEL_PARAMS,
    MODEL_PATH,
    NUMERIC_FEATURES,
    PROCESSED_STREAM_PATH,
    ensure_directories,
)
from app.preprocessing import clean_data, load_historical_data, prepare_features


def _business_rule_points(row: pd.Series) -> int:
    points = 0
    if row["temperature"] >= BUSINESS_THRESHOLDS["temperature_critical"]:
        points += 3
    elif row["temperature"] >= BUSINESS_THRESHOLDS["temperature_warning"]:
        points += 1

    if row["pressure"] >= BUSINESS_THRESHOLDS["pressure_critical"]:
        points += 3
    elif row["pressure"] >= BUSINESS_THRESHOLDS["pressure_warning"]:
        points += 1

    if row["vibration"] >= BUSINESS_THRESHOLDS["vibration_critical"]:
        points += 3
    elif row["vibration"] >= BUSINESS_THRESHOLDS["vibration_warning"]:
        points += 1

    if row["power_consumption"] >= BUSINESS_THRESHOLDS["power_critical"]:
        points += 3
    elif row["power_consumption"] >= BUSINESS_THRESHOLDS["power_warning"]:
        points += 1

    if row["motor_speed"] <= BUSINESS_THRESHOLDS["motor_speed_low_critical"]:
        points += 3
    elif row["motor_speed"] <= BUSINESS_THRESHOLDS["motor_speed_low_warning"]:
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


def _explain_row(row: pd.Series) -> str:
    drivers: list[str] = []
    if row["temperature"] >= BUSINESS_THRESHOLDS["temperature_warning"]:
        drivers.append("temperature above warning threshold")
    if row["pressure"] >= BUSINESS_THRESHOLDS["pressure_warning"]:
        drivers.append("pressure above warning threshold")
    if row["vibration"] >= BUSINESS_THRESHOLDS["vibration_warning"]:
        drivers.append("vibration above warning threshold")
    if row["power_consumption"] >= BUSINESS_THRESHOLDS["power_warning"]:
        drivers.append("power consumption above warning threshold")
    if row["motor_speed"] <= BUSINESS_THRESHOLDS["motor_speed_low_warning"]:
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
    df = clean_data(load_historical_data(data_path))
    scaled_features, scaler = prepare_features(df, fit=True)
    model = IsolationForest(**MODEL_PARAMS)
    model.fit(scaled_features)

    artifact = {
        "model": model,
        "scaler": scaler,
        "features": NUMERIC_FEATURES,
        "model_params": MODEL_PARAMS,
    }
    joblib.dump(artifact, model_path)
    return artifact


def load_model(model_path: str | Path = MODEL_PATH) -> dict[str, Any]:
    """Load the persisted anomaly detector artifact."""
    return joblib.load(model_path)


def predict_anomalies(df: pd.DataFrame, artifact: dict[str, Any] | None = None) -> pd.DataFrame:
    """Predict anomaly labels and raw Isolation Forest scores."""
    artifact = artifact or load_model()
    cleaned = clean_data(df)
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
    enriched["is_anomaly"] = enriched["is_anomaly"] | (rule_points >= 3)
    enriched["anomaly_severity"] = [
        _severity_from_score(score, int(points), bool(is_anomaly))
        for score, points, is_anomaly in zip(
            enriched["anomaly_score"],
            rule_points,
            enriched["is_anomaly"],
        )
    ]
    enriched["explanation"] = enriched.apply(
        lambda row: _explain_row(row) if row["is_anomaly"] else "",
        axis=1,
    )
    return enriched


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


if __name__ == "__main__":
    main()
