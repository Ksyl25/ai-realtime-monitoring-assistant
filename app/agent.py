"""V2 rule-based AI assistant with structured tools and optional LLM rewrite."""

from __future__ import annotations

import os
from collections import deque
from dataclasses import dataclass, field
from typing import Any

import pandas as pd
import requests
from dotenv import load_dotenv

from app.config import BUSINESS_THRESHOLDS, MACHINES
from app.database import (
    get_latest_anomalies as db_get_latest_anomalies,
    get_latest_events,
    get_machine_history as db_get_machine_history,
    get_machine_status as db_get_machine_status,
    get_metrics,
)
from app.report_generator import generate_monitoring_report


load_dotenv()

CONVERSATION_MEMORY: deque[dict[str, str]] = deque(maxlen=6)


@dataclass
class AgentResponse:
    answer: str
    sources: list[str]
    risk_level: str = "unknown"
    recommendation: str = ""
    data_used: dict[str, Any] = field(default_factory=dict)


def get_latest_anomalies(limit: int = 100) -> pd.DataFrame:
    """Tool: read recent anomalies from SQLite."""
    return db_get_latest_anomalies(limit=limit)


def get_machine_status(machine_id: str) -> dict[str, object]:
    """Tool: get the latest operational status for one machine."""
    return db_get_machine_status(machine_id.upper())


def get_machine_history(machine_id: str, limit: int = 200) -> pd.DataFrame:
    """Tool: get recent sensor events for one machine."""
    return db_get_machine_history(machine_id.upper(), limit=limit)


def get_global_metrics() -> dict[str, object]:
    """Tool: get global monitoring metrics."""
    return get_metrics()


def generate_report() -> str:
    """Tool: generate a Markdown report from recent anomalies."""
    anomalies = get_latest_anomalies(limit=500)
    if not anomalies.empty:
        anomalies["is_anomaly"] = True
    return generate_monitoring_report(anomalies)


def _extract_machine_id(question: str) -> str | None:
    upper_question = question.upper()
    for machine in MACHINES:
        if machine in upper_question:
            return machine
    return None


def _risk_from_severity(severity: str | None) -> str:
    mapping = {
        "critical": "critical",
        "high": "high",
        "medium": "medium",
        "low": "low",
        "normal": "normal",
    }
    return mapping.get(str(severity).lower(), "unknown")


def _describe_signal_drivers(row: pd.Series) -> list[str]:
    drivers: list[str] = []
    reason = str(row.get("anomaly_reason", ""))
    if "OVERHEAT" in reason or row["temperature"] >= BUSINESS_THRESHOLDS["temperature_warning"]:
        drivers.append(f"temperature elevated at {row['temperature']:.1f} C")
    if "HIGH_PRESSURE" in reason or row["pressure"] >= BUSINESS_THRESHOLDS["pressure_warning"]:
        drivers.append(f"pressure elevated at {row['pressure']:.2f} bar")
    if "HIGH_VIBRATION" in reason or row["vibration"] >= BUSINESS_THRESHOLDS["vibration_warning"]:
        drivers.append(f"vibration high at {row['vibration']:.3f}")
    if "POWER_SURGE" in reason or row["power_consumption"] >= BUSINESS_THRESHOLDS["power_warning"]:
        drivers.append(f"power consumption high at {row['power_consumption']:.1f} W")
    if "MOTOR_SPEED_DROP" in reason or row["motor_speed"] <= BUSINESS_THRESHOLDS["motor_speed_low_warning"]:
        drivers.append(f"motor speed low at {row['motor_speed']:.0f} rpm")
    return drivers or ["the multivariate ML score is unusual compared with normal patterns"]


def _recommendation(drivers: list[str], severity: str) -> str:
    joined = " ".join(drivers).lower()
    prefix = "Immediate action: " if severity in {"critical", "high"} else "Recommended action: "
    if "temperature" in joined and "vibration" in joined:
        return prefix + "inspect cooling, lubrication, bearings, alignment, and load conditions."
    if "temperature" in joined:
        return prefix + "check cooling, ventilation, lubrication, and recent high-load operation."
    if "vibration" in joined:
        return prefix + "inspect bearings, alignment, mounting, and imbalance."
    if "pressure" in joined:
        return prefix + "review pressure regulation, filters, valves, and process constraints."
    if "power" in joined:
        return prefix + "compare load demand with motor behavior and inspect for inefficient operation."
    if "motor speed" in joined:
        return prefix + "inspect drive train, motor controller, and possible torque overload."
    return prefix + "review recent history and validate sensor quality."


def _maybe_llm_rewrite(response: AgentResponse, question: str) -> AgentResponse:
    """Optional OpenAI-compatible rewrite, with deterministic fallback."""
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        return response
    try:
        payload = {
            "model": os.getenv("OPENAI_MODEL", "gpt-4o-mini"),
            "messages": [
                {
                    "role": "system",
                    "content": "Rewrite the monitoring assistant answer clearly and concisely. Do not invent facts.",
                },
                {
                    "role": "user",
                    "content": f"Question: {question}\nAnswer: {response.answer}",
                },
            ],
            "temperature": 0.2,
        }
        result = requests.post(
            "https://api.openai.com/v1/chat/completions",
            headers={"Authorization": f"Bearer {api_key}"},
            json=payload,
            timeout=15,
        )
        if result.ok:
            response.answer = result.json()["choices"][0]["message"]["content"]
    except Exception:
        response.answer += "\n\nLLM rewrite unavailable; deterministic fallback was used."
    return response


def _machine_answer(question: str, machine_id: str, anomalies: pd.DataFrame) -> AgentResponse:
    target = anomalies[anomalies["machine_id"] == machine_id] if not anomalies.empty else anomalies
    status = get_machine_status(machine_id)
    if target.empty:
        recommendation = str(status.get("recommendation", "Continue monitoring."))
        return AgentResponse(
            answer=f"{machine_id} has no stored recent anomaly. Current status is {status['status']}.",
            sources=["sqlite:anomalies", "sqlite:sensor_events"],
            risk_level=str(status["status"]),
            recommendation=recommendation,
            data_used={"machine_status": status},
        )
    latest = target.iloc[0]
    drivers = _describe_signal_drivers(latest)
    risk = _risk_from_severity(str(latest.get("anomaly_severity")))
    recommendation = _recommendation(drivers, risk)
    answer = (
        f"{machine_id} is in alert because {', '.join(drivers)}. "
        f"Severity is {latest['anomaly_severity']} and anomaly score is {latest['anomaly_score']:.5f}."
    )
    if "surchauffe" in question.lower() or "overheat" in question.lower():
        answer += " The current evidence indicates overheating risk." if "temperature" in " ".join(drivers).lower() else " The latest anomaly is not primarily a temperature issue."
    return AgentResponse(
        answer=answer,
        sources=["sqlite:anomalies", "sqlite:sensor_events"],
        risk_level=risk,
        recommendation=recommendation,
        data_used={
            "machine_id": machine_id,
            "latest_anomaly": latest.to_dict(),
            "machine_status": status,
        },
    )


def answer_question(question: str) -> dict[str, object]:
    """Answer a business question about recent anomalies."""
    normalized = question.lower()
    machine_id = _extract_machine_id(question)
    anomalies = get_latest_anomalies(limit=200)
    metrics = get_global_metrics()

    if anomalies.empty:
        response = AgentResponse(
            answer=(
                "No recent anomaly is stored yet. Run python -m app.bootstrap_demo "
                "to prepare a complete monitoring demo."
            ),
            sources=["sqlite:anomalies", "sqlite:sensor_events"],
            risk_level="unknown",
            recommendation="Bootstrap the demo data before analysis.",
            data_used={"metrics": metrics},
        )
        return response.__dict__

    if machine_id:
        response = _machine_answer(question, machine_id, anomalies)
    elif "plus risqu" in normalized or "most risky" in normalized or "risquee" in normalized:
        counts = anomalies["machine_id"].value_counts()
        riskiest = counts.index[0]
        status = get_machine_status(riskiest)
        response = AgentResponse(
            answer=(
                f"The riskiest machine is {riskiest}, with {counts.iloc[0]} recent anomalies. "
                f"Current machine status is {status['status']}."
            ),
            sources=["sqlite:anomalies", "sqlite:sensor_events"],
            risk_level=str(status["status"]),
            recommendation=str(status["recommendation"]),
            data_used={"metrics": metrics, "machine_status": status},
        )
    elif "rapport" in normalized or "resume" in normalized or "résume" in normalized:
        report = generate_report()
        response = AgentResponse(
            answer=report,
            sources=["sqlite:anomalies", "report_generator"],
            risk_level="summary",
            recommendation="Review the most affected machines and critical alerts first.",
            data_used={"metrics": metrics},
        )
    elif "surchauffe" in normalized or "overheat" in normalized or "temperature" in normalized:
        overheating = anomalies[anomalies["temperature"] >= BUSINESS_THRESHOLDS["temperature_warning"]]
        count = len(overheating)
        response = AgentResponse(
            answer=f"There are {count} recent anomalies with elevated temperature signals.",
            sources=["sqlite:anomalies"],
            risk_level="high" if count else "normal",
            recommendation="Inspect cooling and high-load operating periods when temperature anomalies repeat.",
            data_used={"temperature_anomaly_count": count},
        )
    else:
        latest = anomalies.iloc[0]
        response = _machine_answer(question, str(latest["machine_id"]), anomalies)

    CONVERSATION_MEMORY.append({"question": question, "answer": response.answer[:500]})
    response.data_used["conversation_memory_size"] = len(CONVERSATION_MEMORY)
    return _maybe_llm_rewrite(response, question).__dict__


def main() -> None:
    response = answer_question("Resume les anomalies recentes.")
    print(response["answer"])


if __name__ == "__main__":
    main()

