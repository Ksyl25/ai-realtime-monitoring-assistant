"""Data loading and preprocessing utilities."""

from __future__ import annotations

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from app.config import BUSINESS_THRESHOLDS, HISTORICAL_DATA_PATH, MODEL_FEATURES, NUMERIC_FEATURES


def load_historical_data(path=HISTORICAL_DATA_PATH) -> pd.DataFrame:
    """Load the historical simulated sensor dataset."""
    return pd.read_csv(path)


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """Clean sensor data and enforce expected types."""
    cleaned = df.copy()
    cleaned["timestamp"] = pd.to_datetime(cleaned["timestamp"], errors="coerce")
    cleaned = cleaned.dropna(subset=["timestamp", "machine_id", *NUMERIC_FEATURES])
    for feature in NUMERIC_FEATURES:
        cleaned[feature] = pd.to_numeric(cleaned[feature], errors="coerce")
    cleaned = cleaned.dropna(subset=NUMERIC_FEATURES)
    return cleaned.sort_values("timestamp").reset_index(drop=True)


def add_derived_features(df: pd.DataFrame, rolling_window: int = 12) -> pd.DataFrame:
    """Add V2 feature engineering columns used by the anomaly model."""
    engineered = clean_data(df)
    engineered = engineered.sort_values(["machine_id", "timestamp"]).reset_index(drop=True)
    grouped = engineered.groupby("machine_id", group_keys=False)

    engineered["temperature_delta"] = grouped["temperature"].diff().fillna(0.0)
    engineered["vibration_delta"] = grouped["vibration"].diff().fillna(0.0)
    engineered["power_ratio"] = (
        engineered["power_consumption"] / BUSINESS_THRESHOLDS["power_warning"]
    ).fillna(1.0)
    engineered["pressure_ratio"] = (
        engineered["pressure"] / BUSINESS_THRESHOLDS["pressure_warning"]
    ).fillna(1.0)
    engineered["rolling_mean_temperature"] = grouped["temperature"].transform(
        lambda values: values.rolling(rolling_window, min_periods=1).mean()
    )
    engineered["rolling_mean_vibration"] = grouped["vibration"].transform(
        lambda values: values.rolling(rolling_window, min_periods=1).mean()
    )
    engineered["rolling_std_temperature"] = grouped["temperature"].transform(
        lambda values: values.rolling(rolling_window, min_periods=2).std()
    ).fillna(0.0)
    engineered["rolling_std_vibration"] = grouped["vibration"].transform(
        lambda values: values.rolling(rolling_window, min_periods=2).std()
    ).fillna(0.0)
    return engineered


def prepare_features(df: pd.DataFrame, scaler: StandardScaler | None = None, fit: bool = True):
    """Return scaled numeric features and the scaler used."""
    engineered = add_derived_features(df)
    features = engineered[MODEL_FEATURES].astype(float)
    scaler = scaler or StandardScaler()
    if fit:
        values = scaler.fit_transform(features)
    else:
        values = scaler.transform(features)
    return values, scaler


def split_features_labels(df: pd.DataFrame, test_size: float = 0.2, random_state: int = 42):
    """Split features and optional labels for model validation."""
    labels = (df.get("status", "normal") != "normal").astype(int)
    return train_test_split(
        df[NUMERIC_FEATURES],
        labels,
        test_size=test_size,
        random_state=random_state,
        stratify=labels if labels.nunique() > 1 else None,
    )
