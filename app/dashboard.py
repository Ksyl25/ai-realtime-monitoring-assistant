"""Streamlit V2 dashboard for the monitoring assistant."""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import plotly.express as px
import streamlit as st

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.agent import answer_question
from app.config import MACHINES, NUMERIC_FEATURES
from app.database import get_latest_anomalies, get_latest_events, get_metrics, init_db
from app.report_generator import generate_monitoring_report


st.set_page_config(
    page_title="AI Realtime Monitoring Assistant",
    page_icon=None,
    layout="wide",
)


STATUS_COLORS = {
    "Normal": "#1f9d55",
    "Warning": "#d97706",
    "Critical": "#dc2626",
}


@st.cache_data(ttl=5)
def load_data() -> tuple[pd.DataFrame, pd.DataFrame, dict[str, object]]:
    init_db()
    events = get_latest_events(limit=2_000)
    anomalies = get_latest_anomalies(limit=1_000)
    return events, anomalies, get_metrics()


def _as_datetime(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty or "timestamp" not in df.columns:
        return df
    output = df.copy()
    output["timestamp"] = pd.to_datetime(output["timestamp"], errors="coerce")
    return output.sort_values("timestamp")


def _latest_by_machine(events: pd.DataFrame) -> pd.DataFrame:
    if events.empty:
        return events
    ordered = _as_datetime(events)
    return ordered.dropna(subset=["timestamp"]).groupby("machine_id", as_index=False).tail(1)


def _machine_status(row: pd.Series, anomalies: pd.DataFrame) -> tuple[str, str]:
    recent = anomalies[anomalies["machine_id"] == row["machine_id"]] if not anomalies.empty else anomalies
    health = float(row.get("health_index") or 100)
    has_critical = not recent.empty and (recent["anomaly_severity"] == "critical").any()
    if has_critical or health < 45:
        return "Critical", "Immediate inspection recommended."
    if not recent.empty or health < 70:
        return "Warning", "Review recent signals and monitor trend."
    return "Normal", "No urgent action required."


def _average_health(events: pd.DataFrame) -> float:
    if events.empty or "health_index" not in events.columns:
        return 0.0
    return float(events["health_index"].dropna().mean()) if events["health_index"].notna().any() else 0.0


def _global_summary(metrics: dict[str, object], average_health: float) -> str:
    anomaly_rate = float(metrics.get("anomaly_rate", 0.0))
    critical = int(metrics.get("critical_alerts", 0))
    machine = metrics.get("most_affected_machine") or "n/a"
    if critical:
        risk = "critical"
    elif anomaly_rate >= 0.12 or average_health < 70:
        risk = "warning"
    else:
        risk = "stable"
    return (
        f"Global state: {risk}. Most affected machine: {machine}. "
        f"Average health score: {average_health:.1f}/100. "
        f"Current anomaly rate: {anomaly_rate:.2%}."
    )


def render_metric_cards(events: pd.DataFrame, metrics: dict[str, object]) -> None:
    cols = st.columns(6)
    cols[0].metric("Total events", metrics["total_events"])
    cols[1].metric("Anomalies", metrics["total_anomalies"])
    cols[2].metric("Anomaly rate", f"{metrics['anomaly_rate']:.2%}")
    cols[3].metric("Critical alerts", metrics["critical_alerts"])
    cols[4].metric("Most affected", metrics["most_affected_machine"] or "n/a")
    cols[5].metric("Avg health", f"{_average_health(events):.1f}")


def render_overview(events: pd.DataFrame, anomalies: pd.DataFrame, metrics: dict[str, object]) -> None:
    st.title("AI Realtime Monitoring Assistant")
    st.caption("Portfolio V2 - simulated industrial monitoring. No real industrial data is used.")
    render_metric_cards(events, metrics)
    st.info(_global_summary(metrics, _average_health(events)))

    if events.empty:
        st.warning("No monitoring data yet. Run: python -m app.bootstrap_demo")
        return

    latest = _latest_by_machine(events)
    anomaly_counts = anomalies.groupby("machine_id").size().reset_index(name="anomalies") if not anomalies.empty else pd.DataFrame(columns=["machine_id", "anomalies"])
    totals = events.groupby("machine_id").size().reset_index(name="events")
    rate = totals.merge(anomaly_counts, on="machine_id", how="left").fillna(0)
    rate["anomaly_rate"] = rate["anomalies"] / rate["events"]

    left, right = st.columns(2)
    with left:
        st.plotly_chart(
            px.bar(rate, x="machine_id", y="anomaly_rate", title="Anomaly rate by machine"),
            use_container_width=True,
        )
    with right:
        st.plotly_chart(
            px.bar(latest, x="machine_id", y="health_index", title="Latest health score by machine"),
            use_container_width=True,
        )


def render_live_monitoring(events: pd.DataFrame) -> None:
    st.header("Live Monitoring")
    if st.button("Refresh data"):
        st.cache_data.clear()
        st.rerun()
    if events.empty:
        st.warning("No live event stored yet. Run: python -m app.bootstrap_demo")
        return

    machine_options = ["All machines", *sorted(events["machine_id"].dropna().unique())]
    selected_machine = st.selectbox("Machine", machine_options)
    metric = st.selectbox("Metric", [*NUMERIC_FEATURES, "health_index"])
    filtered = _as_datetime(events)
    if selected_machine != "All machines":
        filtered = filtered[filtered["machine_id"] == selected_machine]

    st.plotly_chart(
        px.line(filtered, x="timestamp", y=metric, color="machine_id", title=f"{metric} over time"),
        use_container_width=True,
    )
    st.dataframe(filtered.tail(200).sort_values("timestamp", ascending=False), use_container_width=True)


def render_machine_details(events: pd.DataFrame, anomalies: pd.DataFrame) -> None:
    st.header("Machine Details")
    if events.empty:
        st.warning("No machine data available.")
        return

    latest = _latest_by_machine(events)
    for _, row in latest.sort_values("machine_id").iterrows():
        status, recommendation = _machine_status(row, anomalies)
        color = STATUS_COLORS[status]
        with st.container(border=True):
            st.markdown(
                f"### {row['machine_id']} "
                f"<span style='color:{color}; font-size:0.9rem'>[{status}]</span>",
                unsafe_allow_html=True,
            )
            cols = st.columns(6)
            cols[0].metric("Health", f"{row.get('health_index', 0):.1f}")
            cols[1].metric("Temp", f"{row['temperature']:.1f}")
            cols[2].metric("Vibration", f"{row['vibration']:.3f}")
            cols[3].metric("Pressure", f"{row['pressure']:.2f}")
            cols[4].metric("Power", f"{row['power_consumption']:.0f}")
            cols[5].metric("Speed", f"{row['motor_speed']:.0f}")
            recent = anomalies[anomalies["machine_id"] == row["machine_id"]].head(5) if not anomalies.empty else anomalies
            st.write(f"Recommendation: {recommendation}")
            if not recent.empty:
                st.dataframe(recent[["timestamp", "anomaly_severity", "anomaly_score", "explanation"]], use_container_width=True)


def _explain_anomaly(row: pd.Series) -> str:
    drivers = []
    if row["vibration"] >= 0.65:
        drivers.append("high vibration")
    if row["temperature"] >= 82:
        drivers.append("abnormal temperature")
    if row["pressure"] >= 6.4:
        drivers.append("high pressure")
    if row["power_consumption"] >= 680:
        drivers.append("power surge")
    if row["motor_speed"] <= 1080:
        drivers.append("low motor speed")
    joined = " and ".join(drivers) if drivers else "an unusual multivariate pattern"
    return f"{row['machine_id']} shows a {row['anomaly_severity']} anomaly linked to {joined}."


def render_anomalies(anomalies: pd.DataFrame) -> None:
    st.header("Anomalies")
    if anomalies.empty:
        st.info("No anomaly stored yet.")
        return

    machine = st.selectbox("Filter by machine", ["All machines", *sorted(anomalies["machine_id"].unique())])
    severity = st.selectbox("Filter by severity", ["All severities", *sorted(anomalies["anomaly_severity"].unique())])
    filtered = _as_datetime(anomalies)
    if machine != "All machines":
        filtered = filtered[filtered["machine_id"] == machine]
    if severity != "All severities":
        filtered = filtered[filtered["anomaly_severity"] == severity]

    if not filtered.empty:
        st.info(_explain_anomaly(filtered.iloc[0]))
        st.plotly_chart(
            px.scatter(
                filtered,
                x="timestamp",
                y="machine_id",
                color="anomaly_severity",
                size=filtered["anomaly_score"].abs(),
                title="Anomaly timeline",
            ),
            use_container_width=True,
        )
    st.dataframe(filtered.sort_values("timestamp", ascending=False), use_container_width=True)


def render_ai_assistant() -> None:
    st.header("AI Assistant")
    examples = [
        "Pourquoi MACHINE_03 est en alerte ?",
        "Quelle machine est la plus risquee ?",
        "Resume les anomalies recentes.",
        "Quelle action recommandes-tu ?",
        "Y a-t-il un risque de surchauffe ?",
    ]
    question = st.selectbox("Example questions", examples)
    custom = st.text_input("Or ask your own question")
    final_question = custom.strip() or question
    if st.button("Ask assistant"):
        response = answer_question(final_question)
        st.markdown(response["answer"])
        if "risk_level" in response:
            st.metric("Risk level", response["risk_level"])
        if "recommendation" in response:
            st.success(response["recommendation"])
        st.caption(f"Sources: {', '.join(response.get('sources', []))}")


def render_reports(events: pd.DataFrame, anomalies: pd.DataFrame) -> None:
    st.header("Reports")
    report_df = anomalies.copy()
    if not report_df.empty:
        report_df["is_anomaly"] = True
    elif not events.empty:
        report_df = events.copy()
        report_df["is_anomaly"] = False
    report = generate_monitoring_report(report_df)
    if st.button("Generate Monitoring Report"):
        st.cache_data.clear()
    st.markdown(report)
    st.download_button("Download report (.md)", report, file_name="monitoring_report.md")
    csv_source = anomalies if not anomalies.empty else events
    st.download_button(
        "Download data (.csv)",
        csv_source.to_csv(index=False),
        file_name="monitoring_export.csv",
        mime="text/csv",
    )


def render_project_info() -> None:
    st.header("Project Info")
    st.markdown(
        """
This project demonstrates an end-to-end AI/Data monitoring assistant on simulated industrial data.

**Architecture**

Simulated sensors write events, Pathway processes incoming streams when available, SQLite stores events and anomalies, Isolation Forest detects unusual behavior, FastAPI exposes monitoring endpoints, Streamlit presents the dashboard, and the assistant explains alerts.

**Pathway**

Pathway is the streaming processing layer. On Windows, the dashboard remains demonstrable through the pandas fallback because the native Pathway runtime is not available on every Windows setup.

**SQLite**

SQLite keeps the project easy to run locally while still showing persistence for events, anomalies, and reports.

**Isolation Forest**

The anomaly detector combines unsupervised ML with business thresholds so alerts are both data-driven and explainable.

**AI Assistant**

The assistant is deterministic by default and can be extended later with an LLM or LangGraph without breaking the fallback.

**Limits**

All data is simulated, thresholds are illustrative, and the project is a portfolio prototype, not production software.

**Future improvements**

LangGraph orchestration, model drift monitoring, direct streaming-to-SQLite sinks, CI quality gates, and richer industrial asset profiles.
"""
    )


def main() -> None:
    events, anomalies, metrics = load_data()
    st.sidebar.title("Monitoring V2")
    page = st.sidebar.radio(
        "Navigation",
        [
            "Overview",
            "Live Monitoring",
            "Machine Details",
            "Anomalies",
            "AI Assistant",
            "Reports",
            "Project Info",
        ],
    )
    st.sidebar.caption("Run `python -m app.bootstrap_demo` to prepare demo data.")

    if page == "Overview":
        render_overview(events, anomalies, metrics)
    elif page == "Live Monitoring":
        render_live_monitoring(events)
    elif page == "Machine Details":
        render_machine_details(events, anomalies)
    elif page == "Anomalies":
        render_anomalies(anomalies)
    elif page == "AI Assistant":
        render_ai_assistant()
    elif page == "Reports":
        render_reports(events, anomalies)
    else:
        render_project_info()


if __name__ == "__main__":
    main()
