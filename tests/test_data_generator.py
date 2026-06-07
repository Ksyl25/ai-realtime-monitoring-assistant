import pandas as pd

from app.config import NUMERIC_FEATURES
from app.data_generator import generate_historical_data


def test_generate_historical_data_columns_and_size():
    df = generate_historical_data(n_rows=500, anomaly_rate=0.05)
    expected = {
        "timestamp",
        "machine_id",
        "temperature",
        "pressure",
        "vibration",
        "power_consumption",
        "motor_speed",
        "operating_mode",
        "status",
    }
    assert expected == set(df.columns)
    assert len(df) == 500
    assert df[NUMERIC_FEATURES].notna().all().all()


def test_generate_historical_data_contains_anomalies():
    df = generate_historical_data(n_rows=1000, anomaly_rate=0.08)
    assert (df["status"] != "normal").sum() > 0
    assert pd.to_datetime(df["timestamp"], errors="coerce").notna().all()

