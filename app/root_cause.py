"""Root cause analysis helpers for monitoring alerts."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

import pandas as pd

from app.config import MODE_THRESHOLDS


@dataclass(frozen=True)
class RootCauseAnalysis:
    probable_cause: str
    confidence: int
    recommended_action: str


def _thresholds(row: Mapping[str, object]) -> dict[str, float]:
    mode = str(row.get("operating_mode", "normal"))
    return MODE_THRESHOLDS.get(mode, MODE_THRESHOLDS["normal"])


def signal_flags(row: Mapping[str, object]) -> dict[str, bool]:
    """Return warning and critical signal flags for one event/anomaly row."""
    thresholds = _thresholds(row)
    temperature = float(row.get("temperature", 0) or 0)
    pressure = float(row.get("pressure", 0) or 0)
    vibration = float(row.get("vibration", 0) or 0)
    power = float(row.get("power_consumption", 0) or 0)
    speed = float(row.get("motor_speed", 0) or 0)

    return {
        "temperature_warning": temperature >= thresholds["temperature_warning"],
        "temperature_critical": temperature >= thresholds["temperature_critical"],
        "pressure_warning": pressure >= thresholds["pressure_warning"],
        "pressure_critical": pressure >= thresholds["pressure_critical"],
        "vibration_warning": vibration >= thresholds["vibration_warning"],
        "vibration_critical": vibration >= thresholds["vibration_critical"],
        "power_warning": power >= thresholds["power_warning"],
        "power_critical": power >= thresholds["power_critical"],
        "speed_warning": speed <= thresholds["motor_speed_low_warning"],
        "speed_critical": speed <= thresholds["motor_speed_low_critical"],
    }


def analyze_root_cause(row: Mapping[str, object] | pd.Series) -> RootCauseAnalysis:
    """Infer a probable root cause from signal threshold patterns."""
    flags = signal_flags(row)

    if flags["temperature_warning"] and flags["vibration_warning"]:
        confidence = 87 if flags["temperature_critical"] or flags["vibration_critical"] else 78
        return RootCauseAnalysis(
            probable_cause="Mechanical Overheating",
            confidence=confidence,
            recommended_action="Inspect cooling system, lubrication, bearings, and mechanical alignment.",
        )

    if flags["pressure_warning"] and flags["temperature_warning"]:
        confidence = 84 if flags["pressure_critical"] or flags["temperature_critical"] else 74
        return RootCauseAnalysis(
            probable_cause="System Overload",
            confidence=confidence,
            recommended_action="Review process load, pressure regulation, filters, and thermal constraints.",
        )

    if flags["power_warning"] and flags["speed_warning"]:
        confidence = 82 if flags["power_critical"] or flags["speed_critical"] else 72
        return RootCauseAnalysis(
            probable_cause="Motor Efficiency Degradation",
            confidence=confidence,
            recommended_action="Inspect motor controller, drive train, torque demand, and power efficiency.",
        )

    if flags["vibration_critical"]:
        return RootCauseAnalysis(
            probable_cause="Bearing Wear Suspected",
            confidence=81,
            recommended_action="Inspect bearings, mounting, imbalance, and shaft alignment.",
        )

    if flags["temperature_critical"]:
        return RootCauseAnalysis(
            probable_cause="Thermal Stress",
            confidence=76,
            recommended_action="Check cooling, ventilation, lubrication, and recent high-load periods.",
        )

    if flags["pressure_critical"]:
        return RootCauseAnalysis(
            probable_cause="Pressure Regulation Issue",
            confidence=73,
            recommended_action="Inspect valves, filters, pressure control loop, and process constraints.",
        )

    if flags["power_critical"]:
        return RootCauseAnalysis(
            probable_cause="Power Surge",
            confidence=72,
            recommended_action="Compare load demand with expected operating mode and inspect electrical drive.",
        )

    if flags["speed_critical"]:
        return RootCauseAnalysis(
            probable_cause="Motor Speed Drop",
            confidence=70,
            recommended_action="Inspect drive train, motor controller, and possible torque overload.",
        )

    return RootCauseAnalysis(
        probable_cause="Multivariate Pattern Drift",
        confidence=62,
        recommended_action="Review recent sensor history and validate sensor calibration.",
    )

