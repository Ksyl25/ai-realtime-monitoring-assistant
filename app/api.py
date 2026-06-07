"""FastAPI service for the monitoring assistant."""

from __future__ import annotations

from contextlib import asynccontextmanager
from typing import Any

from fastapi import FastAPI
from pydantic import BaseModel

from app.agent import answer_question
from app.database import (
    get_latest_anomalies,
    get_latest_events,
    get_machine_anomalies,
    get_metrics,
    init_db,
    save_report,
)
from app.report_generator import generate_monitoring_report


@asynccontextmanager
async def lifespan(_: FastAPI):
    init_db()
    yield


app = FastAPI(
    title="AI Realtime Monitoring Assistant",
    description="Prototype API for simulated industrial machine monitoring.",
    version="1.0.0",
    lifespan=lifespan,
)


class AgentQuery(BaseModel):
    question: str


def _records(df) -> list[dict[str, Any]]:
    return df.where(df.notna(), None).to_dict(orient="records")


@app.get("/")
def root() -> dict[str, str]:
    return {
        "service": "AI Realtime Monitoring Assistant",
        "message": "Simulated industrial monitoring prototype.",
    }


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.get("/metrics")
def metrics() -> dict[str, object]:
    return get_metrics()


@app.get("/events/latest")
def latest_events(limit: int = 100) -> dict[str, object]:
    df = get_latest_events(limit=limit)
    return {"count": len(df), "events": _records(df)}


@app.get("/anomalies/latest")
def latest_anomalies(limit: int = 100) -> dict[str, object]:
    df = get_latest_anomalies(limit=limit)
    return {"count": len(df), "anomalies": _records(df)}


@app.get("/anomalies/{machine_id}")
def anomalies_by_machine(machine_id: str, limit: int = 100) -> dict[str, object]:
    df = get_machine_anomalies(machine_id=machine_id.upper(), limit=limit)
    return {"count": len(df), "machine_id": machine_id.upper(), "anomalies": _records(df)}


@app.get("/report")
def report() -> dict[str, object]:
    anomalies = get_latest_anomalies(limit=500)
    if not anomalies.empty:
        anomalies["is_anomaly"] = True
    content = generate_monitoring_report(anomalies)
    report_id = save_report(content)
    return {"report_id": report_id, "content": content}


@app.post("/agent/query")
def query_agent(payload: AgentQuery) -> dict[str, object]:
    return answer_question(payload.question)
