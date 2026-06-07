"""Rule-based AI assistant with an optional LLM extension point."""

from __future__ import annotations

import os
from dataclasses import dataclass

import pandas as pd
from dotenv import load_dotenv

from app.config import BUSINESS_THRESHOLDS, MACHINES
from app.database import get_latest_anomalies, get_latest_events, get_metrics
from app.report_generator import generate_monitoring_report


load_dotenv()


@dataclass
class AgentResponse:
    answer: str
    sources: list[str]


def get_machine_history(machine_id: str, limit: int = 200) -> pd.DataFrame:
    events = get_latest_events(limit=1000)
    if events.empty:
        return events
    return events[events["machine_id"] == machine_id].head(limit)


def get_global_metrics() -> dict[str, object]:
    return get_metrics()


def _extract_machine_id(question: str) -> str | None:
    upper_question = question.upper()
    for machine in MACHINES:
        if machine in upper_question:
            return machine
    return None


def _describe_signal_drivers(row: pd.Series) -> list[str]:
    drivers: list[str] = []
    if row["temperature"] >= BUSINESS_THRESHOLDS["temperature_warning"]:
        drivers.append(f"temperature elevated at {row['temperature']:.1f} C")
    if row["pressure"] >= BUSINESS_THRESHOLDS["pressure_warning"]:
        drivers.append(f"pressure elevated at {row['pressure']:.2f} bar")
    if row["vibration"] >= BUSINESS_THRESHOLDS["vibration_warning"]:
        drivers.append(f"vibration high at {row['vibration']:.3f}")
    if row["power_consumption"] >= BUSINESS_THRESHOLDS["power_warning"]:
        drivers.append(f"power consumption high at {row['power_consumption']:.1f} W")
    if row["motor_speed"] <= BUSINESS_THRESHOLDS["motor_speed_low_warning"]:
        drivers.append(f"motor speed low at {row['motor_speed']:.0f} rpm")
    return drivers or ["the ML score is unusual compared with normal operating patterns"]


def _recommendation(drivers: list[str]) -> str:
    joined = " ".join(drivers).lower()
    if "temperature" in joined and "vibration" in joined:
        return "Prioritize mechanical inspection and cooling checks; the combined pattern can indicate friction or load stress."
    if "temperature" in joined:
        return "Check cooling, ventilation, lubrication, and recent high-load operation."
    if "vibration" in joined:
        return "Inspect bearings, alignment, mounting, and imbalance before sustained operation."
    if "pressure" in joined:
        return "Review pressure regulation, filters, valves, and process constraints."
    if "power" in joined:
        return "Compare load demand with motor behavior and inspect for inefficient operation."
    if "motor speed" in joined:
        return "Inspect drive train, motor controller, and possible torque overload."
    return "Review the recent machine history and validate sensor quality."


def _maybe_llm_rewrite(answer: str, question: str) -> str:
    """Extension point for OpenAI/LangGraph without making the V1 fragile.

    The prototype remains fully functional without an API key. In a future
    version, this function can call a LangGraph workflow or an OpenAI model to
    rewrite and enrich the deterministic answer.
    """
    if not os.getenv("OPENAI_API_KEY"):
        return answer
    return (
        f"{answer}\n\n"
        "Note: OPENAI_API_KEY is configured, but this V1 keeps the response deterministic. "
        "The agent module is structured so this step can be replaced by a LangGraph/LLM node."
    )


def answer_question(question: str) -> dict[str, object]:
    """Answer a business question about recent anomalies."""
    normalized = question.lower()
    machine_id = _extract_machine_id(question)
    anomalies = get_latest_anomalies(limit=200)
    metrics = get_metrics()
    sources = ["sqlite:anomalies", "sqlite:sensor_events"]

    if anomalies.empty:
        answer = (
            "No recent anomaly is stored yet. Start the simulator, run the processing pipeline, "
            "then run anomaly prediction to populate the monitoring database."
        )
        return AgentResponse(answer=answer, sources=sources).__dict__

    target = anomalies
    if machine_id:
        target = anomalies[anomalies["machine_id"] == machine_id]
        if target.empty:
            answer = f"{machine_id} has no stored recent anomaly. Current global metrics: {metrics}."
            return AgentResponse(answer=answer, sources=sources).__dict__

    if "plus risqu" in normalized or "most risky" in normalized or "la plus risqu" in normalized:
        counts = anomalies["machine_id"].value_counts()
        riskiest = counts.index[0]
        answer = (
            f"The riskiest machine is {riskiest}, with {counts.iloc[0]} recent anomalies. "
            f"Critical alerts stored globally: {metrics['critical_alerts']}."
        )
        return AgentResponse(answer=_maybe_llm_rewrite(answer, question), sources=sources).__dict__

    if "rapport" in normalized or "résume" in normalized or "resume" in normalized:
        report_df = anomalies.copy()
        report_df["is_anomaly"] = True
        answer = generate_monitoring_report(report_df)
        sources.append("report_generator")
        return AgentResponse(answer=_maybe_llm_rewrite(answer, question), sources=sources).__dict__

    latest = target.iloc[0]
    drivers = _describe_signal_drivers(latest)
    recommendation = _recommendation(drivers)
    machine_label = machine_id or latest["machine_id"]
    answer = (
        f"{machine_label} is in alert because {', '.join(drivers)}. "
        f"The latest anomaly severity is {latest['anomaly_severity']} "
        f"with an Isolation Forest score of {latest['anomaly_score']:.5f}. "
        f"Recommended action: {recommendation}"
    )

    if "surchauffe" in normalized or "vibration" in normalized:
        answer += (
            " Based on the latest drivers, this looks more related to "
            f"{'vibration' if latest['vibration'] >= BUSINESS_THRESHOLDS['vibration_warning'] else 'temperature' if latest['temperature'] >= BUSINESS_THRESHOLDS['temperature_warning'] else 'a combined weak signal'}."
        )

    return AgentResponse(answer=_maybe_llm_rewrite(answer, question), sources=sources).__dict__


def main() -> None:
    question = "Résume les anomalies récentes."
    response = answer_question(question)
    print(response["answer"])


if __name__ == "__main__":
    main()
