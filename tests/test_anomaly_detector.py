import pandas as pd

from app.anomaly_detector import enrich_with_anomaly_columns, train_model
from app.data_generator import generate_historical_data


def test_train_model_and_predict(tmp_path):
    data_path = tmp_path / "historical.csv"
    model_path = tmp_path / "model.pkl"
    df = generate_historical_data(n_rows=800, anomaly_rate=0.05)
    df.to_csv(data_path, index=False)

    artifact = train_model(data_path=data_path, model_path=model_path)
    assert model_path.exists()
    assert "model" in artifact
    assert "scaler" in artifact

    sample = pd.concat(
        [
            df.head(20),
            pd.DataFrame(
                [
                    {
                        "timestamp": "2026-01-01T00:00:00",
                        "machine_id": "MACHINE_01",
                        "temperature": 105.0,
                        "pressure": 8.0,
                        "vibration": 1.2,
                        "power_consumption": 900.0,
                        "motor_speed": 850.0,
                        "operating_mode": "high_load",
                        "status": "multi_signal_degradation",
                    }
                ]
            ),
        ],
        ignore_index=True,
    )
    enriched = enrich_with_anomaly_columns(sample, artifact=artifact)
    for column in ["anomaly_prediction", "anomaly_score", "is_anomaly", "anomaly_severity"]:
        assert column in enriched.columns
    assert enriched["is_anomaly"].any()

