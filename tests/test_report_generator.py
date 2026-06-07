from app.anomaly_detector import enrich_with_anomaly_columns, train_model
from app.data_generator import generate_historical_data
from app.report_generator import generate_monitoring_report


def test_generate_monitoring_report_contains_expected_sections(tmp_path):
    data_path = tmp_path / "historical.csv"
    model_path = tmp_path / "model.pkl"
    df = generate_historical_data(n_rows=700, anomaly_rate=0.06)
    df.to_csv(data_path, index=False)
    artifact = train_model(data_path=data_path, model_path=model_path)
    enriched = enrich_with_anomaly_columns(df.head(100), artifact=artifact)

    report = generate_monitoring_report(enriched)

    assert "# Monitoring Report" in report
    assert "Global Summary" in report
    assert "Technical Recommendations" in report
    assert "Analysis Limits" in report

