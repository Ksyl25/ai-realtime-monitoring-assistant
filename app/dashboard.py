"""Streamlit dashboard for the monitoring assistant."""

from __future__ import annotations

import pandas as pd
import plotly.express as px
import streamlit as st

from app.agent import answer_question
from app.database import get_latest_anomalies, get_latest_events, get_metrics, init_db
from app.report_generator import generate_monitoring_report


st.set_page_config(
    page_title="AI Realtime Monitoring Assistant",
    page_icon=None,
    layout="wide",
)


@st.cache_data(ttl=5)
def load_data():
    init_db()
    events = get_latest_events(limit=1000)
    anomalies = get_latest_anomalies(limit=500)
    return events, anomalies, get_metrics()


def render_metric_cards(metrics: dict[str, object]) -> None:
    cols = st.columns(5)
    cols[0].metric("Total events", metrics["total_events"])
    cols[1].metric("Anomalies", metrics["total_anomalies"])
    cols[2].metric("Anomaly rate", f"{metrics['anomaly_rate']:.2%}")
    cols[3].metric("Most affected", metrics["most_affected_machine"] or "n/a")
    cols[4].metric("Critical alerts", metrics["critical_alerts"])


def render_overview(events: pd.DataFrame, anomalies: pd.DataFrame, metrics: dict[str, object]) -> None:
    st.title("AI Realtime Monitoring Assistant")
    st.caption(
        "Prototype de monitoring sur données capteurs industrielles simulées. "
        "Les données ne proviennent pas de machines réelles."
    )
    render_metric_cards(metrics)

    if events.empty:
        st.info("Aucune donnée en base pour le moment. Lancez le simulateur et le pipeline pour alimenter le dashboard.")
        return

    events = events.sort_values("timestamp")
    chart_cols = st.columns(2)
    with chart_cols[0]:
        st.plotly_chart(
            px.line(events, x="timestamp", y="temperature", color="machine_id", title="Temperature over time"),
            use_container_width=True,
        )
    with chart_cols[1]:
        st.plotly_chart(
            px.line(events, x="timestamp", y="vibration", color="machine_id", title="Vibration over time"),
            use_container_width=True,
        )

    if "health_index" in events.columns:
        st.plotly_chart(
            px.bar(
                events.groupby("machine_id", as_index=False)["health_index"].mean(),
                x="machine_id",
                y="health_index",
                title="Average health index by machine",
            ),
            use_container_width=True,
        )


def render_live_monitoring(events: pd.DataFrame) -> None:
    st.subheader("Live Monitoring")
    if events.empty:
        st.info("No live event stored yet.")
        return
    st.dataframe(events.head(200), use_container_width=True)


def render_anomalies(anomalies: pd.DataFrame) -> None:
    st.subheader("Anomalies")
    if anomalies.empty:
        st.info("No anomaly stored yet.")
        return
    st.dataframe(anomalies.head(200), use_container_width=True)
    severity_counts = anomalies["anomaly_severity"].value_counts().reset_index()
    severity_counts.columns = ["severity", "count"]
    st.plotly_chart(
        px.bar(severity_counts, x="severity", y="count", title="Anomalies by severity"),
        use_container_width=True,
    )


def render_ai_assistant() -> None:
    st.subheader("AI Assistant")
    question = st.text_input("Question", value="Pourquoi MACHINE_03 est en alerte ?")
    if st.button("Ask assistant"):
        response = answer_question(question)
        st.markdown(response["answer"])
        st.caption(f"Sources: {', '.join(response['sources'])}")


def render_reports(anomalies: pd.DataFrame) -> None:
    st.subheader("Reports")
    report_df = anomalies.copy()
    if not report_df.empty:
        report_df["is_anomaly"] = True
    st.markdown(generate_monitoring_report(report_df))


def main() -> None:
    events, anomalies, metrics = load_data()
    tabs = st.tabs(["Overview", "Live Monitoring", "Anomalies", "AI Assistant", "Reports"])
    with tabs[0]:
        render_overview(events, anomalies, metrics)
    with tabs[1]:
        render_live_monitoring(events)
    with tabs[2]:
        render_anomalies(anomalies)
    with tabs[3]:
        render_ai_assistant()
    with tabs[4]:
        render_reports(anomalies)


if __name__ == "__main__":
    main()
