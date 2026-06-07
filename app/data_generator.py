"""Generate historical sensor data for simulated industrial machines."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

import numpy as np
import pandas as pd

from app.config import (
    HISTORICAL_DATA_PATH,
    MACHINES,
    OPERATING_MODES,
    ensure_directories,
)


RNG = np.random.default_rng(42)


def utc_now_naive() -> datetime:
    return datetime.now(timezone.utc).replace(tzinfo=None)


MODE_PROFILES = {
    "idle": {
        "temperature": (58, 4),
        "pressure": (3.6, 0.3),
        "vibration": (0.12, 0.04),
        "power_consumption": (260, 35),
        "motor_speed": (950, 80),
    },
    "normal": {
        "temperature": (70, 5),
        "pressure": (5.0, 0.45),
        "vibration": (0.28, 0.08),
        "power_consumption": (450, 65),
        "motor_speed": (1400, 90),
    },
    "high_load": {
        "temperature": (78, 5),
        "pressure": (5.7, 0.5),
        "vibration": (0.42, 0.09),
        "power_consumption": (590, 70),
        "motor_speed": (1570, 85),
    },
    "maintenance": {
        "temperature": (62, 3),
        "pressure": (4.2, 0.35),
        "vibration": (0.18, 0.05),
        "power_consumption": (320, 45),
        "motor_speed": (1150, 100),
    },
}


ANOMALY_TYPES = [
    "overheating",
    "high_vibration",
    "high_pressure",
    "excessive_power",
    "motor_speed_drop",
    "multi_signal_degradation",
]


def _bounded_normal(mean: float, std: float, lower: float | None = None) -> float:
    value = float(RNG.normal(mean, std))
    if lower is not None:
        return max(value, lower)
    return value


def _normal_measurement(mode: str) -> dict[str, float]:
    profile = MODE_PROFILES[mode]
    return {
        "temperature": _bounded_normal(*profile["temperature"], lower=45),
        "pressure": _bounded_normal(*profile["pressure"], lower=2.5),
        "vibration": _bounded_normal(*profile["vibration"], lower=0.02),
        "power_consumption": _bounded_normal(*profile["power_consumption"], lower=180),
        "motor_speed": _bounded_normal(*profile["motor_speed"], lower=650),
    }


def inject_anomaly(row: dict[str, object]) -> dict[str, object]:
    """Mutate a simulated row with a realistic anomaly pattern."""
    anomaly_type = str(RNG.choice(ANOMALY_TYPES))

    if anomaly_type == "overheating":
        row["temperature"] = float(RNG.normal(96, 7))
    elif anomaly_type == "high_vibration":
        row["vibration"] = float(RNG.normal(1.05, 0.18))
    elif anomaly_type == "high_pressure":
        row["pressure"] = float(RNG.normal(7.8, 0.45))
    elif anomaly_type == "excessive_power":
        row["power_consumption"] = float(RNG.normal(850, 80))
    elif anomaly_type == "motor_speed_drop":
        row["motor_speed"] = float(RNG.normal(840, 90))
    else:
        row["temperature"] = float(RNG.normal(91, 6))
        row["vibration"] = float(RNG.normal(0.9, 0.16))
        row["power_consumption"] = float(RNG.normal(760, 75))
        if RNG.random() < 0.5:
            row["motor_speed"] = float(RNG.normal(930, 80))

    row["status"] = anomaly_type
    return row


def generate_historical_data(n_rows: int = 10_000, anomaly_rate: float = 0.05) -> pd.DataFrame:
    """Generate a labeled historical dataset with simulated sensor readings."""
    start = utc_now_naive() - timedelta(minutes=n_rows)
    records: list[dict[str, object]] = []

    for idx in range(n_rows):
        mode = str(RNG.choice(OPERATING_MODES, p=[0.12, 0.55, 0.25, 0.08]))
        record: dict[str, object] = {
            "timestamp": (start + timedelta(minutes=idx)).isoformat(timespec="seconds"),
            "machine_id": str(RNG.choice(MACHINES)),
            "operating_mode": mode,
            "status": "normal",
        }
        record.update(_normal_measurement(mode))

        if RNG.random() < anomaly_rate:
            record = inject_anomaly(record)

        records.append(record)

    df = pd.DataFrame(records)
    ordered_columns = [
        "timestamp",
        "machine_id",
        "temperature",
        "pressure",
        "vibration",
        "power_consumption",
        "motor_speed",
        "operating_mode",
        "status",
    ]
    return df[ordered_columns].round(
        {
            "temperature": 2,
            "pressure": 2,
            "vibration": 3,
            "power_consumption": 2,
            "motor_speed": 2,
        }
    )


def main() -> None:
    ensure_directories()
    df = generate_historical_data()
    df.to_csv(HISTORICAL_DATA_PATH, index=False)
    anomalies = int((df["status"] != "normal").sum())
    print(f"Generated {len(df)} rows at {HISTORICAL_DATA_PATH}")
    print(f"Injected anomalies: {anomalies} ({anomalies / len(df):.1%})")


if __name__ == "__main__":
    main()
