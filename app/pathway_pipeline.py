"""Pathway streaming pipeline for simulated sensor events.

Pathway separates pipeline definition from execution: this module defines the
schema, transformations, and output sink, then starts the computation with
pw.run(). A small pandas fallback is included only to keep local demos usable
when Pathway is not installed; the primary implementation is the Pathway graph.
"""

from __future__ import annotations

import argparse
import platform
from pathlib import Path

import pandas as pd

from app.config import MODE_THRESHOLDS, PROCESSED_STREAM_PATH, STREAM_DATA_DIR, ensure_directories


def calculate_health_index(
    temperature: float,
    pressure: float,
    vibration: float,
    power_consumption: float,
    motor_speed: float,
    operating_mode: str = "normal",
) -> float:
    """Business health score from 0 to 100, where lower means riskier."""
    thresholds = MODE_THRESHOLDS.get(str(operating_mode), MODE_THRESHOLDS["normal"])
    penalty = 0.0
    penalty += max(0.0, temperature - thresholds["temperature_warning"]) * 1.8
    penalty += max(0.0, pressure - thresholds["pressure_warning"]) * 8.0
    penalty += max(0.0, vibration - thresholds["vibration_warning"]) * 35.0
    penalty += max(0.0, power_consumption - thresholds["power_warning"]) * 0.05
    penalty += max(0.0, thresholds["motor_speed_low_warning"] - motor_speed) * 0.04
    if operating_mode == "high_load":
        penalty *= 0.85
    elif operating_mode == "idle":
        penalty *= 1.1
    return round(max(0.0, min(100.0, 100.0 - penalty)), 2)


def run_pandas_fallback() -> pd.DataFrame:
    """One-shot processor used when Pathway is unavailable in the environment."""
    files = sorted(Path(STREAM_DATA_DIR).glob("*.csv"))
    if not files:
        raise FileNotFoundError(f"No stream CSV files found in {STREAM_DATA_DIR}")
    frames = [pd.read_csv(file) for file in files]
    df = pd.concat(frames, ignore_index=True)
    df["health_index"] = df.apply(
        lambda row: calculate_health_index(
            row["temperature"],
            row["pressure"],
            row["vibration"],
            row["power_consumption"],
            row["motor_speed"],
            row.get("operating_mode", "normal"),
        ),
        axis=1,
    )
    df.to_csv(PROCESSED_STREAM_PATH, index=False)
    from app.database import insert_sensor_events

    inserted = insert_sensor_events(df)
    print(f"Processed {len(df)} rows with pandas one-shot mode at {PROCESSED_STREAM_PATH}")
    print(f"Inserted {inserted} processed events into SQLite")
    return df


def run_pathway_pipeline() -> None:
    """Define and execute the Pathway streaming pipeline."""
    ensure_directories()
    if platform.system().lower() == "windows":
        print(
            "Pathway's native streaming runtime is not available on Windows in this environment. "
            "Falling back to one-shot pandas processing. Use Docker/Linux for the continuous Pathway demo."
        )
        run_pandas_fallback()
        return

    try:
        import pathway as pw
    except ImportError:
        run_pandas_fallback()
        return

    if not hasattr(pw, "Schema"):
        print(
            "The installed pathway package does not expose the real Pathway runtime on this platform. "
            "Falling back to one-shot pandas processing."
        )
        run_pandas_fallback()
        return

    class SensorEventSchema(pw.Schema):
        timestamp: str
        machine_id: str
        temperature: float
        pressure: float
        vibration: float
        power_consumption: float
        motor_speed: float
        operating_mode: str

    @pw.udf
    def health_index_udf(
        temperature: float,
        pressure: float,
        vibration: float,
        power_consumption: float,
        motor_speed: float,
        operating_mode: str,
    ) -> float:
        return calculate_health_index(
            temperature,
            pressure,
            vibration,
            power_consumption,
            motor_speed,
            operating_mode,
        )

    stream = pw.io.csv.read(
        str(STREAM_DATA_DIR),
        schema=SensorEventSchema,
        mode="streaming",
    )

    transformed = stream.select(
        timestamp=pw.this.timestamp,
        machine_id=pw.this.machine_id,
        temperature=pw.this.temperature,
        pressure=pw.this.pressure,
        vibration=pw.this.vibration,
        power_consumption=pw.this.power_consumption,
        motor_speed=pw.this.motor_speed,
        operating_mode=pw.this.operating_mode,
        health_index=health_index_udf(
            pw.this.temperature,
            pw.this.pressure,
            pw.this.vibration,
            pw.this.power_consumption,
            pw.this.motor_speed,
            pw.this.operating_mode,
        ),
    )

    pw.io.csv.write(transformed, str(PROCESSED_STREAM_PATH))
    print(f"Starting Pathway pipeline from {STREAM_DATA_DIR} to {PROCESSED_STREAM_PATH}")
    pw.run()


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the Pathway stream processor.")
    parser.add_argument(
        "--fallback",
        action="store_true",
        help="Run one-shot pandas processing instead of the continuous Pathway pipeline.",
    )
    args = parser.parse_args()
    if args.fallback:
        run_pandas_fallback()
    else:
        run_pathway_pipeline()


if __name__ == "__main__":
    main()
