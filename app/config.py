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

NUMERIC_FEATURES = [
    "temperature",
    "pressure",
    "vibration",
    "power_consumption",
    "motor_speed",
]

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

