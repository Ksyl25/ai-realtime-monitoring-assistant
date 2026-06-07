"""Prepare a complete local demo in one command.

This command is designed for portfolio demos and recruiter interviews. It
generates simulated data, trains the model, prepares processed events, stores
events/anomalies in SQLite, and creates a Markdown report.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from app.anomaly_detector import enrich_with_anomaly_columns, train_model
from app.config import (
    DATABASE_PATH,
    HISTORICAL_DATA_PATH,
    PROCESSED_STREAM_PATH,
    ensure_directories,
)
from app.data_generator import generate_historical_data
from app.database import init_db, insert_anomalies, insert_sensor_events, save_report
from app.pathway_pipeline import calculate_health_index
from app.report_generator import generate_monitoring_report


def _reset_demo_files() -> None:
    """Remove runtime demo artifacts while keeping tracked placeholders."""
    for path in [
        DATABASE_PATH,
        HISTORICAL_DATA_PATH,
        PROCESSED_STREAM_PATH,
        PROCESSED_STREAM_PATH.with_name("stream_anomalies.csv"),
    ]:
        if Path(path).exists():
            Path(path).unlink()


def _prepare_processed_demo_events(df: pd.DataFrame, n_rows: int = 750) -> pd.DataFrame:
    """Create a processed stream sample from the historical simulated data."""
    processed = df.tail(n_rows).copy()
    processed["health_index"] = processed.apply(
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
    processed.to_csv(PROCESSED_STREAM_PATH, index=False)
    return processed


def run_bootstrap(reset: bool = True) -> dict[str, object]:
    """Run the complete demo bootstrap and return summary metrics."""
    ensure_directories()
    if reset:
        _reset_demo_files()

    print("[1/6] Generating simulated historical sensor data...")
    historical = generate_historical_data(n_rows=10_000, anomaly_rate=0.05)
    historical.to_csv(HISTORICAL_DATA_PATH, index=False)

    print("[2/6] Training Isolation Forest anomaly model...")
    artifact = train_model()

    print("[3/6] Preparing processed demo stream sample...")
    processed = _prepare_processed_demo_events(historical)

    print("[4/6] Detecting anomalies and severity levels...")
    enriched = enrich_with_anomaly_columns(processed, artifact=artifact)
    enriched.to_csv(PROCESSED_STREAM_PATH.with_name("stream_anomalies.csv"), index=False)

    print("[5/6] Initializing SQLite and storing monitoring data...")
    init_db()
    inserted_events = insert_sensor_events(enriched)
    inserted_anomalies = insert_anomalies(enriched)

    print("[6/6] Generating Markdown monitoring report...")
    report_df = enriched.copy()
    report = generate_monitoring_report(report_df)
    report_id = save_report(report, report_type="bootstrap_demo")

    summary = {
        "historical_rows": len(historical),
        "demo_events": inserted_events,
        "detected_anomalies": inserted_anomalies,
        "anomaly_rate": round(inserted_anomalies / inserted_events, 4) if inserted_events else 0.0,
        "report_id": report_id,
        "database": str(DATABASE_PATH),
        "processed_stream": str(PROCESSED_STREAM_PATH),
    }
    return summary


def main() -> None:
    summary = run_bootstrap(reset=True)
    print("\nDemo environment ready.")
    print("- Historical rows:", summary["historical_rows"])
    print("- Stored demo events:", summary["demo_events"])
    print("- Stored anomalies:", summary["detected_anomalies"])
    print("- Anomaly rate:", f"{summary['anomaly_rate']:.2%}")
    print("- SQLite database:", summary["database"])
    print("- Processed stream:", summary["processed_stream"])
    print("\nNext commands:")
    print("  uvicorn app.api:app --reload")
    print("  streamlit run app/dashboard.py")


if __name__ == "__main__":
    main()
