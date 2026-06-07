"""SQLite persistence layer for sensor events, anomalies, and reports."""

from __future__ import annotations

import sqlite3
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from app.config import DATABASE_PATH, NUMERIC_FEATURES, PROCESSED_STREAM_PATH, ensure_directories


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).replace(tzinfo=None).isoformat(timespec="seconds")


def get_connection(db_path: str | Path = DATABASE_PATH) -> sqlite3.Connection:
    connection = sqlite3.connect(db_path)
    connection.row_factory = sqlite3.Row
    return connection


def init_db(db_path: str | Path = DATABASE_PATH) -> None:
    """Create database tables if they do not exist."""
    ensure_directories()
    with get_connection(db_path) as conn:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS sensor_events (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT NOT NULL,
                machine_id TEXT NOT NULL,
                temperature REAL NOT NULL,
                pressure REAL NOT NULL,
                vibration REAL NOT NULL,
                power_consumption REAL NOT NULL,
                motor_speed REAL NOT NULL,
                operating_mode TEXT NOT NULL,
                health_index REAL,
                created_at TEXT NOT NULL
            )
            """
        )
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS anomalies (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT NOT NULL,
                machine_id TEXT NOT NULL,
                anomaly_score REAL NOT NULL,
                anomaly_severity TEXT NOT NULL,
                temperature REAL NOT NULL,
                pressure REAL NOT NULL,
                vibration REAL NOT NULL,
                power_consumption REAL NOT NULL,
                motor_speed REAL NOT NULL,
                explanation TEXT,
                created_at TEXT NOT NULL
            )
            """
        )
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS reports (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                generated_at TEXT NOT NULL,
                report_type TEXT NOT NULL,
                content TEXT NOT NULL
            )
            """
        )


def _records_from_df(df: pd.DataFrame, columns: list[str]) -> list[dict[str, object]]:
    now = utc_now_iso()
    records = df.copy()
    for column in columns:
        if column not in records.columns:
            records[column] = None
    if "timestamp" in records.columns:
        records["timestamp"] = pd.to_datetime(records["timestamp"], errors="coerce").dt.strftime(
            "%Y-%m-%dT%H:%M:%S"
        )
    records["created_at"] = now
    return records[[*columns, "created_at"]].to_dict(orient="records")


def insert_sensor_events(df: pd.DataFrame, db_path: str | Path = DATABASE_PATH) -> int:
    """Insert sensor events into SQLite."""
    if df.empty:
        return 0
    init_db(db_path)
    columns = [
        "timestamp",
        "machine_id",
        *NUMERIC_FEATURES,
        "operating_mode",
        "health_index",
    ]
    records = _records_from_df(df, columns)
    with get_connection(db_path) as conn:
        conn.executemany(
            """
            INSERT INTO sensor_events (
                timestamp, machine_id, temperature, pressure, vibration,
                power_consumption, motor_speed, operating_mode, health_index, created_at
            )
            VALUES (
                :timestamp, :machine_id, :temperature, :pressure, :vibration,
                :power_consumption, :motor_speed, :operating_mode, :health_index, :created_at
            )
            """,
            records,
        )
    return len(records)


def insert_anomalies(df: pd.DataFrame, db_path: str | Path = DATABASE_PATH) -> int:
    """Insert detected anomalies into SQLite."""
    if df.empty:
        return 0
    init_db(db_path)
    if "is_anomaly" in df.columns:
        anomalies = df[df["is_anomaly"] == True].copy()  # noqa: E712
    else:
        anomalies = df.copy()
    if anomalies.empty:
        return 0
    if "explanation" not in anomalies.columns:
        anomalies["explanation"] = ""
    columns = [
        "timestamp",
        "machine_id",
        "anomaly_score",
        "anomaly_severity",
        *NUMERIC_FEATURES,
        "explanation",
    ]
    records = _records_from_df(anomalies, columns)
    with get_connection(db_path) as conn:
        conn.executemany(
            """
            INSERT INTO anomalies (
                timestamp, machine_id, anomaly_score, anomaly_severity,
                temperature, pressure, vibration, power_consumption, motor_speed,
                explanation, created_at
            )
            VALUES (
                :timestamp, :machine_id, :anomaly_score, :anomaly_severity,
                :temperature, :pressure, :vibration, :power_consumption, :motor_speed,
                :explanation, :created_at
            )
            """,
            records,
        )
    return len(records)


def get_latest_events(limit: int = 100, db_path: str | Path = DATABASE_PATH) -> pd.DataFrame:
    init_db(db_path)
    with get_connection(db_path) as conn:
        return pd.read_sql_query(
            "SELECT * FROM sensor_events ORDER BY timestamp DESC, id DESC LIMIT ?",
            conn,
            params=(limit,),
        )


def get_latest_anomalies(limit: int = 100, db_path: str | Path = DATABASE_PATH) -> pd.DataFrame:
    init_db(db_path)
    with get_connection(db_path) as conn:
        return pd.read_sql_query(
            "SELECT * FROM anomalies ORDER BY timestamp DESC, id DESC LIMIT ?",
            conn,
            params=(limit,),
        )


def get_machine_anomalies(machine_id: str, limit: int = 100, db_path: str | Path = DATABASE_PATH) -> pd.DataFrame:
    init_db(db_path)
    with get_connection(db_path) as conn:
        return pd.read_sql_query(
            "SELECT * FROM anomalies WHERE machine_id = ? ORDER BY timestamp DESC, id DESC LIMIT ?",
            conn,
            params=(machine_id, limit),
        )


def get_metrics(db_path: str | Path = DATABASE_PATH) -> dict[str, object]:
    init_db(db_path)
    with get_connection(db_path) as conn:
        total_events = conn.execute("SELECT COUNT(*) FROM sensor_events").fetchone()[0]
        total_anomalies = conn.execute("SELECT COUNT(*) FROM anomalies").fetchone()[0]
        critical_alerts = conn.execute(
            "SELECT COUNT(*) FROM anomalies WHERE anomaly_severity = 'critical'"
        ).fetchone()[0]
        most_affected = conn.execute(
            """
            SELECT machine_id, COUNT(*) AS count
            FROM anomalies
            GROUP BY machine_id
            ORDER BY count DESC
            LIMIT 1
            """
        ).fetchone()
    return {
        "total_events": int(total_events),
        "total_anomalies": int(total_anomalies),
        "anomaly_rate": round(total_anomalies / total_events, 4) if total_events else 0.0,
        "most_affected_machine": most_affected["machine_id"] if most_affected else None,
        "critical_alerts": int(critical_alerts),
    }


def save_report(content: str, report_type: str = "monitoring", db_path: str | Path = DATABASE_PATH) -> int:
    init_db(db_path)
    generated_at = utc_now_iso()
    with get_connection(db_path) as conn:
        cursor = conn.execute(
            "INSERT INTO reports (generated_at, report_type, content) VALUES (?, ?, ?)",
            (generated_at, report_type, content),
        )
        return int(cursor.lastrowid)


def main() -> None:
    init_db()
    anomaly_path = PROCESSED_STREAM_PATH.with_name("stream_anomalies.csv")
    if PROCESSED_STREAM_PATH.exists():
        df = pd.read_csv(PROCESSED_STREAM_PATH)
        inserted = insert_sensor_events(df)
        inserted_anomalies = 0
        if anomaly_path.exists():
            anomaly_df = pd.read_csv(anomaly_path)
            inserted_anomalies = insert_anomalies(anomaly_df)
        print(
            f"Database initialized. Inserted {inserted} processed sensor events "
            f"and {inserted_anomalies} anomalies."
        )
    else:
        print(f"Database initialized at {DATABASE_PATH}")


if __name__ == "__main__":
    main()
