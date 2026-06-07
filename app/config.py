"""Central project configuration.

All paths are resolved from the repository root so commands work from Windows,
Linux, Docker, and test runners.
"""

from __future__ import annotations

from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]

DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
STREAM_DATA_DIR = DATA_DIR / "stream"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
MODELS_DIR = PROJECT_ROOT / "models"

HISTORICAL_DATA_PATH = RAW_DATA_DIR / "historical_sensor_data.csv"
PROCESSED_STREAM_PATH = PROCESSED_DATA_DIR / "stream_processed.csv"
DATABASE_PATH = PROJECT_ROOT / "monitoring.db"
MODEL_PATH = MODELS_DIR / "anomaly_model.pkl"
EVALUATION_METRICS_PATH = MODELS_DIR / "evaluation_metrics.json"

NUMERIC_FEATURES = [
    "temperature",
    "pressure",
    "vibration",
    "power_consumption",
    "motor_speed",
]

DERIVED_FEATURES = [
    "temperature_delta",
    "vibration_delta",
    "power_ratio",
    "pressure_ratio",
    "rolling_mean_temperature",
    "rolling_mean_vibration",
    "rolling_std_temperature",
    "rolling_std_vibration",
]

MODEL_FEATURES = [*NUMERIC_FEATURES, *DERIVED_FEATURES]

MACHINES = [f"MACHINE_{idx:02d}" for idx in range(1, 6)]
OPERATING_MODES = ["idle", "normal", "high_load", "maintenance"]

MODEL_PARAMS = {
    "n_estimators": 150,
    "contamination": 0.05,
    "random_state": 42,
}

BUSINESS_THRESHOLDS = {
    "temperature_warning": 82.0,
    "temperature_critical": 92.0,
    "pressure_warning": 6.4,
    "pressure_critical": 7.4,
    "vibration_warning": 0.65,
    "vibration_critical": 0.95,
    "power_warning": 680.0,
    "power_critical": 820.0,
    "motor_speed_low_warning": 1080.0,
    "motor_speed_low_critical": 950.0,
    "health_warning": 70.0,
    "health_critical": 45.0,
}

MODE_THRESHOLDS = {
    "idle": {
        "temperature_warning": 74.0,
        "temperature_critical": 84.0,
        "pressure_warning": 5.4,
        "pressure_critical": 6.5,
        "vibration_warning": 0.45,
        "vibration_critical": 0.75,
        "power_warning": 520.0,
        "power_critical": 700.0,
        "motor_speed_low_warning": 800.0,
        "motor_speed_low_critical": 650.0,
    },
    "normal": {
        "temperature_warning": 82.0,
        "temperature_critical": 92.0,
        "pressure_warning": 6.4,
        "pressure_critical": 7.4,
        "vibration_warning": 0.65,
        "vibration_critical": 0.95,
        "power_warning": 680.0,
        "power_critical": 820.0,
        "motor_speed_low_warning": 1080.0,
        "motor_speed_low_critical": 950.0,
    },
    "high_load": {
        "temperature_warning": 90.0,
        "temperature_critical": 100.0,
        "pressure_warning": 7.0,
        "pressure_critical": 8.1,
        "vibration_warning": 0.78,
        "vibration_critical": 1.1,
        "power_warning": 760.0,
        "power_critical": 920.0,
        "motor_speed_low_warning": 1150.0,
        "motor_speed_low_critical": 980.0,
    },
    "maintenance": {
        "temperature_warning": 76.0,
        "temperature_critical": 86.0,
        "pressure_warning": 5.8,
        "pressure_critical": 6.8,
        "vibration_warning": 0.55,
        "vibration_critical": 0.85,
        "power_warning": 560.0,
        "power_critical": 720.0,
        "motor_speed_low_warning": 900.0,
        "motor_speed_low_critical": 700.0,
    },
}


def ensure_directories() -> None:
    """Create required project folders if they do not exist."""
    for directory in [
        RAW_DATA_DIR,
        STREAM_DATA_DIR,
        PROCESSED_DATA_DIR,
        MODELS_DIR,
    ]:
        directory.mkdir(parents=True, exist_ok=True)


ensure_directories()
