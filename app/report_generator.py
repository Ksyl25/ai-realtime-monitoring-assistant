"""Markdown monitoring reports for simulated industrial sensor data."""

from __future__ import annotations

import pandas as pd

from app.config import BUSINESS_THRESHOLDS, NUMERIC_FEATURES
from app.database import get_latest_anomalies, get_latest_events


def _signal_summary(df: pd.DataFrame) -> list[str]:
    if df.empty:
        return ["- No signal deviation available."]

    signals = []
    checks = {
        "temperature": BUSINESS_THRESHOLDS["temperature_warning"],
        "pressure": BUSINESS_THRESHOLDS["pressure_warning"],
        "vibration": BUSINESS_THRESHOLDS["vibration_warning"],
        "power_consumption": BUSINESS_THRESHOLDS["power_warning"],
    }
    for feature, threshold in checks.items():
        share = float((df[feature] >= threshold).mean()) if feature in df else 0.0
        if share > 0:
            signals.append(f"- {feature}: {share:.1%} of analyzed rows exceed the warning threshold.")
    if "motor_speed" in df:
        share = float((df["motor_speed"] <= BUSINESS_THRESHOLDS["motor_speed_low_warning"]).mean())
        if share > 0:
            signals.append(f"- motor_speed: {share:.1%} of analyzed rows show a low-speed condition.")
    return signals or ["- No dominant threshold breach identified."]


def generate_monitoring_report(df: pd.DataFrame) -> str:
    """Generate a Markdown report from events or enriched anomaly rows."""
    total = len(df)
    if total == 0:
        return (
            "# Monitoring Report\n\n"
            "No observation is available yet. Start the simulator and processing pipeline to populate the database.\n"
        )

    anomalies = df[df["is_anomaly"] == True] if "is_anomaly" in df.columns else df  # noqa: E712
    anomaly_count = len(anomalies)
    anomaly_rate = anomaly_count / total if total else 0
    machine_counts = (
        anomalies["machine_id"].value_counts().head(3).to_dict()
        if "machine_id" in anomalies and not anomalies.empty
        else {}
    )

    severity_order = {"critical": 4, "high": 3, "medium": 2, "low": 1, "normal": 0}
    most_critical = "No anomaly detected."
    if "anomaly_severity" in anomalies and not anomalies.empty:
        ranked = anomalies.assign(
            severity_rank=anomalies["anomaly_severity"].map(severity_order).fillna(0)
        ).sort_values(["severity_rank", "anomaly_score"], ascending=[False, True])
        top = ranked.iloc[0]
        most_critical = (
            f"{top['machine_id']} at {top['timestamp']} "
            f"({top['anomaly_severity']}, score={top.get('anomaly_score', 'n/a')})."
        )

    affected = "\n".join([f"- {machine}: {count} anomalies" for machine, count in machine_counts.items()])
    if not affected:
        affected = "- No recurring affected machine identified."

    signals = "\n".join(_signal_summary(anomalies if not anomalies.empty else df))

    return f"""# Monitoring Report

## Global Summary
- Total observations analyzed: {total}
- Detected anomalies: {anomaly_count}
- Anomaly rate: {anomaly_rate:.2%}

## Most Affected Machines
{affected}

## Most Critical Anomaly
{most_critical}

## Dominant Signal Analysis
{signals}

## Technical Recommendations
- Inspect machines with repeated high-severity alerts before increasing production load.
- Compare temperature and vibration trends to identify mechanical friction or cooling issues.
- Validate sensor calibration when a single signal is abnormal without operational context.
- Use this prototype as a decision-support layer, not as an automatic shutdown system.

## Analysis Limits
- Data is fully simulated and does not represent real industrial equipment.
- The Isolation Forest model is unsupervised and should be validated with domain experts before production usage.
- Business thresholds are illustrative and must be calibrated for each industrial asset.
"""


def generate_machine_report(machine_id: str) -> str:
    """Generate a Markdown report for one machine from SQLite data."""
    events = get_latest_events(limit=1000)
    anomalies = get_latest_anomalies(limit=1000)
    machine_events = events[events["machine_id"] == machine_id] if not events.empty else events
    machine_anomalies = anomalies[anomalies["machine_id"] == machine_id] if not anomalies.empty else anomalies

    if machine_events.empty and machine_anomalies.empty:
        return f"# Machine Report - {machine_id}\n\nNo data found for this machine.\n"

    enriched = machine_anomalies.copy()
    if not enriched.empty:
        enriched["is_anomaly"] = True
    else:
        enriched = machine_events
        enriched["is_anomaly"] = False

    report = generate_monitoring_report(enriched)
    avg_signals = ""
    if not machine_events.empty:
        averages = machine_events[NUMERIC_FEATURES].mean(numeric_only=True)
        avg_signals = "\n".join([f"- {feature}: {value:.2f}" for feature, value in averages.items()])
    return f"# Machine Report - {machine_id}\n\n## Average Recent Signals\n{avg_signals or '- n/a'}\n\n{report}"
