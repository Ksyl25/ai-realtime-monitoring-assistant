"""Data loading and preprocessing utilities."""

from __future__ import annotations

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from app.config import HISTORICAL_DATA_PATH, NUMERIC_FEATURES


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


def prepare_features(df: pd.DataFrame, scaler: StandardScaler | None = None, fit: bool = True):
    """Return scaled numeric features and the scaler used."""
    features = df[NUMERIC_FEATURES].astype(float)
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

