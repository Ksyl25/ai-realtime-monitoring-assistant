# Architecture V2

This project is a prototype for continuous monitoring of simulated industrial machines.
It does not use real industrial data and is not production software.

## Text Diagram

```text
Simulated historical data
        |
        v
Isolation Forest training + StandardScaler
        |
        v
models/anomaly_model.pkl

Streaming simulator --> data/stream/*.csv --> Pathway pipeline --> data/processed/stream_processed.csv
                                                            |
                                                            v
                                                anomaly prediction + severity
                                                            |
                                                            v
                                                     SQLite database
                                                            |
                              +-----------------------------+-----------------------------+
                              v                                                           v
                         FastAPI service                                           Streamlit dashboard
                              |
                              v
                       Rule-based AI assistant
```

## V2 Portfolio Flow

```text
python -m app.bootstrap_demo
        |
        +--> data/raw/historical_sensor_data.csv
        +--> models/anomaly_model.pkl
        +--> data/processed/stream_processed.csv
        +--> data/processed/stream_anomalies.csv
        +--> monitoring.db
        +--> Markdown report stored in SQLite

FastAPI reads monitoring.db --> JSON endpoints and Swagger docs
Streamlit reads monitoring.db --> portfolio dashboard with 7 pages
Agent reads monitoring.db --> explanations, risk level, recommendations
```

## Data Flow

1. `app.data_generator` creates `data/raw/historical_sensor_data.csv` with simulated sensor readings.
2. `app.anomaly_detector --train` trains an Isolation Forest model on numeric sensor features.
3. `app.streaming_simulator` appends new simulated sensor events into `data/stream/`.
4. `app.pathway_pipeline` reads incoming stream CSV files, applies transformations, and writes processed rows.
5. `app.anomaly_detector --predict` enriches processed rows with anomaly columns.
6. `app.database` stores sensor events, anomalies, and reports in SQLite.
7. FastAPI and Streamlit expose metrics, recent events, anomaly tables, reports, and assistant answers.

## Role of Pathway

Pathway is used as the streaming data processing layer. The pipeline defines a typed input schema, reads new CSV events from the stream folder in streaming mode, computes a business `health_index`, and writes transformed data to a processed output.

The module also includes a one-shot pandas fallback for local environments where Pathway is unavailable, but the main implementation demonstrates Pathway pipeline definition and execution.

## Role of the ML Model

The anomaly detector uses `IsolationForest`, an unsupervised algorithm suitable for identifying rare behavior in numeric sensor patterns. A `StandardScaler` is fitted during training and saved in the same `joblib` artifact as the model.

Severity is not based on the model alone. The project combines ML scores with simple business thresholds to make alerts more interpretable.

## Role of FastAPI

FastAPI exposes operational endpoints:

- health checks;
- global metrics;
- latest events;
- latest anomalies;
- machine-specific anomalies;
- report generation;
- assistant query endpoint.

The API is documented automatically through Swagger at `/docs`.

## Role of Streamlit

Streamlit provides a practical dashboard for recruiters and technical reviewers:

- overview metrics;
- time-series charts;
- machine detail cards;
- anomaly timeline and filters;
- health index by machine;
- latest event and anomaly tables;
- AI assistant question box;
- Markdown report rendering.

## Role of the AI Agent

The V1 agent is deterministic and rule-based. It reads recent anomalies and metrics from SQLite, identifies likely signal drivers, and produces natural-language recommendations.

The agent module is intentionally structured with tool-like functions such as `get_machine_history`, `get_global_metrics`, and report generation so it can later be replaced or extended with LangGraph and an LLM.
