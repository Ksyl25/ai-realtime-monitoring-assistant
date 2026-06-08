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
from app.config import MACHINES, MODE_THRESHOLDS, NUMERIC_FEATURES
from app.database import get_latest_anomalies, get_latest_events, get_metrics, init_db
from app.report_generator import generate_monitoring_report
from app.root_cause import analyze_root_cause, signal_flags


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
STATUS_RISK = {
    "Normal": "Low",
    "Warning": "Medium",
    "Critical": "High",
}

RECENT_ANOMALY_WINDOW_HOURS = 6
ANOMALY_PENALTIES = {
    "low": 5,
    "medium": 15,
    "high": 30,
    "critical": 45,
}
SENSOR_PENALTIES = {
    "temperature": {"warning": 10, "critical": 25},
    "vibration": {"warning": 15, "critical": 35},
    "pressure": {"warning": 10, "critical": 20},
    "power_consumption": {"warning": 10, "critical": 20},
    "motor_speed": {"warning": 15, "critical": 30},
}


st.markdown(
    """
    <style>
    .monitor-card {
        border: 1px solid #d8dee9;
        border-radius: 8px;
        padding: 14px 16px;
        background: #ffffff;
        min-height: 94px;
    }
    .monitor-label {
        color: #607086;
        font-size: 0.78rem;
        font-weight: 700;
        text-transform: uppercase;
        letter-spacing: 0.02em;
    }
    .monitor-value {
        color: #172033;
        font-size: 1.55rem;
        font-weight: 800;
        margin-top: 4px;
    }
    .monitor-subtle {
        color: #69778c;
        font-size: 0.88rem;
    }
    .status-row {
        display: flex;
        flex-wrap: wrap;
        gap: 10px;
        align-items: center;
        margin: 10px 0 18px 0;
    }
    .status-pill {
        border-radius: 999px;
        padding: 7px 12px;
        color: white;
        font-weight: 700;
        font-size: 0.85rem;
    }
    .section-title {
        margin-top: 0.8rem;
        margin-bottom: 0.4rem;
        color: #253044;
        font-size: 1rem;
        font-weight: 800;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


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


def _reference_time(events: pd.DataFrame, anomalies: pd.DataFrame) -> pd.Timestamp | None:
    timestamps = []
    for df in [events, anomalies]:
        if not df.empty and "timestamp" in df.columns:
            parsed = pd.to_datetime(df["timestamp"], errors="coerce").dropna()
            if not parsed.empty:
                timestamps.append(parsed.max())
    if not timestamps:
        return None
    return max(timestamps)


def _recent_anomalies_for_machine(
    machine_id: str,
    anomalies: pd.DataFrame,
    reference_time: pd.Timestamp | None,
    window_hours: int = RECENT_ANOMALY_WINDOW_HOURS,
) -> pd.DataFrame:
    if anomalies.empty or reference_time is None:
        return anomalies.iloc[0:0].copy()
    prepared = _as_datetime(anomalies)
    cutoff = reference_time - pd.Timedelta(hours=window_hours)
    return prepared[
        (prepared["machine_id"] == machine_id)
        & (prepared["timestamp"].notna())
        & (prepared["timestamp"] >= cutoff)
    ].copy()


def _recent_anomaly_penalty(recent_anomalies: pd.DataFrame) -> int:
    if recent_anomalies.empty or "anomaly_severity" not in recent_anomalies.columns:
        return 0
    severities = recent_anomalies["anomaly_severity"].fillna("").str.lower()
    return min(100, int(sum(ANOMALY_PENALTIES.get(severity, 0) for severity in severities)))


def _thresholds_for_row(row: pd.Series) -> dict[str, float]:
    mode = str(row.get("operating_mode", "normal"))
    return MODE_THRESHOLDS.get(mode, MODE_THRESHOLDS["normal"])


def _sensor_health_score(row: pd.Series) -> float:
    thresholds = _thresholds_for_row(row)
    penalty = 0

    if row["temperature"] >= thresholds["temperature_critical"]:
        penalty += SENSOR_PENALTIES["temperature"]["critical"]
    elif row["temperature"] >= thresholds["temperature_warning"]:
        penalty += SENSOR_PENALTIES["temperature"]["warning"]

    if row["vibration"] >= thresholds["vibration_critical"]:
        penalty += SENSOR_PENALTIES["vibration"]["critical"]
    elif row["vibration"] >= thresholds["vibration_warning"]:
        penalty += SENSOR_PENALTIES["vibration"]["warning"]

    if row["pressure"] >= thresholds["pressure_critical"]:
        penalty += SENSOR_PENALTIES["pressure"]["critical"]
    elif row["pressure"] >= thresholds["pressure_warning"]:
        penalty += SENSOR_PENALTIES["pressure"]["warning"]

    if row["power_consumption"] >= thresholds["power_critical"]:
        penalty += SENSOR_PENALTIES["power_consumption"]["critical"]
    elif row["power_consumption"] >= thresholds["power_warning"]:
        penalty += SENSOR_PENALTIES["power_consumption"]["warning"]

    if row["motor_speed"] <= thresholds["motor_speed_low_critical"]:
        penalty += SENSOR_PENALTIES["motor_speed"]["critical"]
    elif row["motor_speed"] <= thresholds["motor_speed_low_warning"]:
        penalty += SENSOR_PENALTIES["motor_speed"]["warning"]

    return _clamp_health(100 - min(100, penalty))


def _clamp_health(value: float) -> float:
    return round(max(0.0, min(100.0, value)), 1)


def _render_card(column, label: str, value: object, color: str | None = None) -> None:
    color_style = f" style='color:{color}'" if color else ""
    value_style = "font-size:1.0rem; line-height:1.25" if len(str(value)) > 32 else ""
    value_style = f" style='{value_style}; color:{color}'" if color and value_style else color_style or (f" style='{value_style}'" if value_style else "")
    column.markdown(
        f"""
        <div class="monitor-card">
            <div class="monitor-label">{label}</div>
            <div class="monitor-value"{value_style}>{value}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def _sensor_state(row: pd.Series, metric: str) -> tuple[str, str, float, float]:
    thresholds = _thresholds_for_row(row)
    value = float(row[metric])

    if metric == "temperature":
        warning = thresholds["temperature_warning"]
        critical = thresholds["temperature_critical"]
        state = "Critical" if value >= critical else "Warning" if value >= warning else "Normal"
    elif metric == "pressure":
        warning = thresholds["pressure_warning"]
        critical = thresholds["pressure_critical"]
        state = "Critical" if value >= critical else "Warning" if value >= warning else "Normal"
    elif metric == "vibration":
        warning = thresholds["vibration_warning"]
        critical = thresholds["vibration_critical"]
        state = "Critical" if value >= critical else "Warning" if value >= warning else "Normal"
    elif metric == "power_consumption":
        warning = thresholds["power_warning"]
        critical = thresholds["power_critical"]
        state = "Critical" if value >= critical else "Warning" if value >= warning else "Normal"
    else:
        warning = thresholds["motor_speed_low_warning"]
        critical = thresholds["motor_speed_low_critical"]
        state = "Critical" if value <= critical else "Warning" if value <= warning else "Normal"

    return state, STATUS_COLORS[state], warning, critical


def _render_sensor_card(column, label: str, row: pd.Series, metric: str, unit: str) -> None:
    state, color, warning, critical = _sensor_state(row, metric)
    value = float(row[metric])
    precision = 3 if metric == "vibration" else 1 if metric in {"temperature", "pressure"} else 0
    formatted_value = f"{value:.{precision}f} {unit}".strip()
    column.markdown(
        f"""
        <div class="monitor-card">
            <div class="monitor-label">{label}</div>
            <div class="monitor-value">{formatted_value}</div>
            <div class="monitor-subtle" style="color:{color}; font-weight:700">{state}</div>
            <div class="monitor-subtle">Warning: {warning:g} | Critical: {critical:g}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def _metric_thresholds(metric: str, mode: str = "normal") -> tuple[float | None, float | None]:
    thresholds = MODE_THRESHOLDS.get(mode, MODE_THRESHOLDS["normal"])
    if metric == "temperature":
        return thresholds["temperature_warning"], thresholds["temperature_critical"]
    if metric == "pressure":
        return thresholds["pressure_warning"], thresholds["pressure_critical"]
    if metric == "vibration":
        return thresholds["vibration_warning"], thresholds["vibration_critical"]
    if metric == "power_consumption":
        return thresholds["power_warning"], thresholds["power_critical"]
    if metric == "motor_speed":
        return thresholds["motor_speed_low_warning"], thresholds["motor_speed_low_critical"]
    return None, None


def _metric_unit(metric: str) -> str:
    return {
        "temperature": "C",
        "pressure": "bar",
        "vibration": "",
        "power_consumption": "W",
        "motor_speed": "rpm",
        "health_index": "",
    }.get(metric, "")


def _plot_metric_timeseries(
    events: pd.DataFrame,
    anomalies: pd.DataFrame,
    metric: str,
    selected_machine: str,
):
    fig = px.line(events, x="timestamp", y=metric, color="machine_id", title=f"{metric} over time")
    mode = "normal"
    if selected_machine != "All machines" and not events.empty and "operating_mode" in events.columns:
        mode_values = events["operating_mode"].dropna()
        if not mode_values.empty:
            mode = str(mode_values.iloc[-1])

    warning, critical = _metric_thresholds(metric, mode)
    if warning is not None:
        fig.add_hline(
            y=warning,
            line_dash="dash",
            line_color="#d97706",
            annotation_text="warning",
            annotation_position="top left",
        )
    if critical is not None:
        fig.add_hline(
            y=critical,
            line_dash="dash",
            line_color="#dc2626",
            annotation_text="critical",
            annotation_position="top right",
        )

    if not anomalies.empty and metric in anomalies.columns:
        marker_data = _as_datetime(anomalies)
        if selected_machine != "All machines":
            marker_data = marker_data[marker_data["machine_id"] == selected_machine]
        marker_data = marker_data.dropna(subset=["timestamp"])
        if not marker_data.empty:
            fig.add_scatter(
                x=marker_data["timestamp"],
                y=marker_data[metric],
                mode="markers",
                name="anomalies",
                marker={"color": "#dc2626", "size": 9, "symbol": "x"},
                text=marker_data["anomaly_severity"],
                hovertemplate="Anomaly<br>%{x}<br>value=%{y}<br>severity=%{text}<extra></extra>",
            )
    fig.update_layout(legend_title_text="Machine", margin=dict(l=20, r=20, t=50, b=20))
    return fig


def _status_from_health(final_health_score: float, recent_anomalies: pd.DataFrame) -> str:
    severities = (
        set(recent_anomalies["anomaly_severity"].fillna("").str.lower())
        if not recent_anomalies.empty and "anomaly_severity" in recent_anomalies.columns
        else set()
    )
    if final_health_score < 50 or "critical" in severities:
        return "Critical"
    if final_health_score < 75 or severities.intersection({"medium", "high"}):
        return "Warning"
    return "Normal"


def _recommendation_from_status(status: str) -> str:
    if status == "Critical":
        return "Immediate inspection recommended before sustained operation."
    if status == "Warning":
        return "Review recent anomalies and monitor the next operating cycle."
    return "No urgent action required."


def _machine_health_snapshot(events: pd.DataFrame, anomalies: pd.DataFrame) -> pd.DataFrame:
    latest = _latest_by_machine(events)
    if latest.empty:
        return latest

    reference_time = _reference_time(events, anomalies)
    snapshots: list[dict[str, object]] = []
    for _, row in latest.iterrows():
        machine_id = str(row["machine_id"])
        latest_sensor_health = _sensor_health_score(row)
        recent = _recent_anomalies_for_machine(machine_id, anomalies, reference_time)
        penalty = _recent_anomaly_penalty(recent)
        final_health_score = _clamp_health(latest_sensor_health - penalty)
        status = _status_from_health(final_health_score, recent)
        root_cause = analyze_root_cause(row)
        snapshots.append(
            {
                **row.to_dict(),
                "latest_sensor_health": latest_sensor_health,
                "recent_anomaly_penalty": penalty,
                "final_health_score": final_health_score,
                "status": status,
                "risk_level": STATUS_RISK[status],
                "recent_anomaly_count": int(len(recent)),
                "recommendation": _recommendation_from_status(status),
                "probable_cause": root_cause.probable_cause,
                "root_cause_confidence": root_cause.confidence,
                "root_cause_action": root_cause.recommended_action,
            }
        )
    return pd.DataFrame(snapshots)


def _average_final_health(events: pd.DataFrame, anomalies: pd.DataFrame) -> float:
    snapshots = _machine_health_snapshot(events, anomalies)
    if snapshots.empty or "final_health_score" not in snapshots.columns:
        return 0.0
    return float(snapshots["final_health_score"].dropna().mean())


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


def render_metric_cards(events: pd.DataFrame, anomalies: pd.DataFrame, metrics: dict[str, object]) -> None:
    health = _machine_health_snapshot(events, anomalies)
    healthy = int((health["status"] == "Normal").sum()) if not health.empty else 0
    warning = int((health["status"] == "Warning").sum()) if not health.empty else 0
    critical = int((health["status"] == "Critical").sum()) if not health.empty else 0

    cols = st.columns(6)
    cols[0].metric("Total events", metrics["total_events"])
    cols[1].metric("Anomalies", metrics["total_anomalies"])
    cols[2].metric("Anomaly rate", f"{metrics['anomaly_rate']:.2%}")
    cols[3].metric("Healthy", healthy)
    cols[4].metric("Warning", warning)
    cols[5].metric("Critical", critical)


def render_overview(events: pd.DataFrame, anomalies: pd.DataFrame, metrics: dict[str, object]) -> None:
    st.title("AI Realtime Monitoring Assistant")
    st.caption("Portfolio V2 - simulated industrial monitoring. No real industrial data is used.")
    render_metric_cards(events, anomalies, metrics)
    st.info(_global_summary(metrics, _average_final_health(events, anomalies)))

    if events.empty:
        st.warning("No monitoring data yet. Run: python -m app.bootstrap_demo")
        return

    machine_health = _machine_health_snapshot(events, anomalies)
    status_badges = []
    for _, row in machine_health.sort_values("machine_id").iterrows():
        color = STATUS_COLORS[str(row["status"])]
        status_badges.append(
            f"<span class='status-pill' style='background:{color}'>{row['machine_id']} - {row['status']}</span>"
        )
    st.markdown("<div class='status-row'>" + "".join(status_badges) + "</div>", unsafe_allow_html=True)

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
            px.bar(
                machine_health,
                x="machine_id",
                y="final_health_score",
                color="status",
                title="Final health score by machine",
            ),
            use_container_width=True,
        )


def render_live_monitoring(events: pd.DataFrame, anomalies: pd.DataFrame) -> None:
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

    st.plotly_chart(_plot_metric_timeseries(filtered, anomalies, metric, selected_machine), use_container_width=True)
    st.dataframe(filtered.tail(200).sort_values("timestamp", ascending=False), use_container_width=True)


def render_machine_details(events: pd.DataFrame, anomalies: pd.DataFrame) -> None:
    st.header("Machine Details")
    if events.empty:
        st.warning("No machine data available.")
        return

    machine_health = _machine_health_snapshot(events, anomalies)
    for _, row in machine_health.sort_values("machine_id").iterrows():
        status = str(row["status"])
        recommendation = str(row["recommendation"])
        color = STATUS_COLORS[status]
        with st.container(border=True):
            st.markdown(
                f"### {row['machine_id']} "
                f"<span style='color:{color}; font-size:0.9rem'>[{status}]</span>",
                unsafe_allow_html=True,
            )

            st.markdown("<div class='section-title'>Global State</div>", unsafe_allow_html=True)
            global_cols = st.columns(3)
            _render_card(global_cols[0], "Status", status, color=color)
            _render_card(global_cols[1], "Final Health Score", f"{row['final_health_score']:.1f}/100")
            _render_card(global_cols[2], "Risk Level", row["risk_level"])

            st.markdown("<div class='section-title'>Diagnostic</div>", unsafe_allow_html=True)
            diagnostic_cols = st.columns(3)
            _render_card(diagnostic_cols[0], "Sensor Health", f"{row['latest_sensor_health']:.1f}/100")
            _render_card(diagnostic_cols[1], "Anomaly Penalty", f"{int(row['recent_anomaly_penalty'])}")
            _render_card(diagnostic_cols[2], "Recent Anomalies", f"{int(row['recent_anomaly_count'])}")

            cause_cols = st.columns(3)
            _render_card(cause_cols[0], "Probable Cause", row["probable_cause"])
            _render_card(cause_cols[1], "Confidence", f"{int(row['root_cause_confidence'])}%")
            _render_card(cause_cols[2], "Recommended Action", row["root_cause_action"])

            st.markdown("<div class='section-title'>Sensors</div>", unsafe_allow_html=True)
            signal_cols = st.columns(5)
            _render_sensor_card(signal_cols[0], "Temperature", row, "temperature", "C")
            _render_sensor_card(signal_cols[1], "Pressure", row, "pressure", "bar")
            _render_sensor_card(signal_cols[2], "Vibration", row, "vibration", "")
            _render_sensor_card(signal_cols[3], "Power", row, "power_consumption", "W")
            _render_sensor_card(signal_cols[4], "Speed", row, "motor_speed", "rpm")

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


def _extract_machine_from_question(question: str) -> str | None:
    upper_question = question.upper()
    for machine in MACHINES:
        if machine in upper_question:
            return machine
    return None


def _assistant_probable_cause(question: str, anomalies: pd.DataFrame) -> str:
    if anomalies.empty:
        return "No anomaly context available"
    machine_id = _extract_machine_from_question(question)
    context = anomalies
    if machine_id:
        context = anomalies[anomalies["machine_id"] == machine_id]
    if context.empty:
        context = anomalies
    return analyze_root_cause(context.iloc[0]).probable_cause


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
        filtered = filtered.copy()
        root_causes = filtered.apply(analyze_root_cause, axis=1)
        filtered["probable_cause"] = [item.probable_cause for item in root_causes]
        filtered["root_cause_confidence"] = [item.confidence for item in root_causes]
        st.info(_explain_anomaly(filtered.iloc[0]))
        st.plotly_chart(
            px.scatter(
                filtered,
                x="timestamp",
                y="machine_id",
                color="anomaly_severity",
                size=filtered["anomaly_score"].abs(),
                title="Anomaly timeline",
                hover_data={
                    "machine_id": True,
                    "anomaly_severity": True,
                    "anomaly_score": ":.5f",
                    "probable_cause": True,
                    "root_cause_confidence": True,
                    "timestamp": True,
                },
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
        summary, cause, risk, recommendation = st.columns(4)
        _render_card(summary, "Summary", response.get("answer", "No answer available"))
        _render_card(cause, "Probable Cause", _assistant_probable_cause(final_question, get_latest_anomalies(limit=200)))
        _render_card(risk, "Risk Level", response.get("risk_level", "unknown"))
        _render_card(recommendation, "Recommendation", response.get("recommendation", "Review latest anomalies."))
        with st.expander("Data used"):
            st.json(response.get("data_used", {}))
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
        render_live_monitoring(events, anomalies)
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
