"""Append-only simulator for real-time industrial sensor events."""

from __future__ import annotations

import argparse
import time
from datetime import datetime, timezone

import numpy as np
import pandas as pd

from app.config import MACHINES, OPERATING_MODES, STREAM_DATA_DIR, ensure_directories
from app.data_generator import _normal_measurement, inject_anomaly


RNG = np.random.default_rng()


def utc_now_naive() -> datetime:
    return datetime.now(timezone.utc).replace(tzinfo=None)


def generate_stream_event(anomaly_probability: float = 0.08) -> dict[str, object]:
    mode = str(RNG.choice(OPERATING_MODES, p=[0.12, 0.58, 0.22, 0.08]))
    event: dict[str, object] = {
        "timestamp": utc_now_naive().isoformat(timespec="seconds"),
        "machine_id": str(RNG.choice(MACHINES)),
        "operating_mode": mode,
    }
    event.update(_normal_measurement(mode))
    if RNG.random() < anomaly_probability:
        event["status"] = "normal"
        event = inject_anomaly(event)
    for key in ["temperature", "pressure", "vibration", "power_consumption", "motor_speed"]:
        event[key] = round(float(event[key]), 3)
    return event


def append_event(event: dict[str, object]) -> None:
    ensure_directories()
    batch_file = STREAM_DATA_DIR / f"sensor_stream_{utc_now_naive().strftime('%Y%m%d_%H')}.csv"
    df = pd.DataFrame([event])
    df = df[
        [
            "timestamp",
            "machine_id",
            "temperature",
            "pressure",
            "vibration",
            "power_consumption",
            "motor_speed",
            "operating_mode",
        ]
    ]
    df.to_csv(batch_file, mode="a", index=False, header=not batch_file.exists())


def run_simulator(max_events: int | None = None) -> None:
    count = 0
    print(f"Writing simulated stream events to {STREAM_DATA_DIR}")
    while max_events is None or count < max_events:
        event = generate_stream_event()
        append_event(event)
        count += 1
        print(f"[{count}] {event}")
        time.sleep(float(RNG.uniform(1, 3)))


def main() -> None:
    parser = argparse.ArgumentParser(description="Simulate real-time sensor events.")
    parser.add_argument("--max-events", type=int, default=None, help="Optional number of events for demos/tests.")
    args = parser.parse_args()
    run_simulator(max_events=args.max_events)


if __name__ == "__main__":
    main()
