# AI Realtime Monitoring Assistant

## Overview

AI Realtime Monitoring Assistant is a clean V1 prototype for continuous monitoring of simulated industrial machines.

It demonstrates a realistic AI/Data architecture with simulated sensor streams, Pathway-based stream processing, anomaly detection, SQLite persistence, a FastAPI service, a Streamlit dashboard, and an explanatory AI assistant.

Important: all data is simulated. This project does not use real industrial data and is not intended for production deployment.

## Business Use Case

Industrial teams often need to monitor temperature, pressure, vibration, energy consumption, and motor speed to detect early signs of machine degradation.

This prototype simulates that use case for five machines and shows how an AI/Data engineer can structure a monitoring assistant from ingestion to explanation.

## Key Features

- Historical sensor data generation with realistic operating modes.
- Continuous stream simulator writing append-only CSV events.
- Pathway pipeline for streaming ingestion and transformation.
- Health index calculation from sensor signals.
- Isolation Forest anomaly detection with saved scaler/model artifact.
- SQLite storage for events, anomalies, and reports.
- FastAPI endpoints with Swagger documentation.
- Streamlit dashboard with metrics, charts, tables, reports, and assistant.
- Rule-based AI assistant that explains recent anomalies and recommends actions.
- Pytest coverage for data generation, model behavior, reports, and API endpoints.
- Docker and Docker Compose support.

## Architecture

```text
data_generator -> historical CSV -> Isolation Forest training -> model artifact

streaming_simulator -> data/stream CSV files -> Pathway pipeline -> processed CSV
                                                        |
                                                        v
                                             anomaly detection
                                                        |
                                                        v
                                                 SQLite database
                                                        |
                                  +---------------------+---------------------+
                                  v                                           v
                              FastAPI                                   Streamlit
                                  |
                                  v
                            AI assistant
```

More details are available in `docs/architecture.md`.

## Tech Stack

- Python 3.10+
- Pathway
- pandas
- numpy
- scikit-learn
- FastAPI
- Uvicorn
- Streamlit
- SQLite
- Plotly and matplotlib
- joblib
- pytest
- Docker
- python-dotenv

## Project Structure

```text
ai-realtime-monitoring-assistant/
├── app/
│   ├── config.py
│   ├── data_generator.py
│   ├── streaming_simulator.py
│   ├── pathway_pipeline.py
│   ├── preprocessing.py
│   ├── anomaly_detector.py
│   ├── database.py
│   ├── report_generator.py
│   ├── agent.py
│   ├── api.py
│   └── dashboard.py
├── data/
├── models/
├── notebooks/
├── tests/
├── docs/
├── README.md
├── requirements.txt
├── Dockerfile
├── docker-compose.yml
├── .env.example
└── .gitignore
```

## Data Simulation

The project simulates five machines:

- MACHINE_01
- MACHINE_02
- MACHINE_03
- MACHINE_04
- MACHINE_05

Operating modes:

- idle
- normal
- high_load
- maintenance

Sensor features:

- temperature
- pressure
- vibration
- power_consumption
- motor_speed

The historical generator creates at least 10,000 rows and injects around 5 percent anomalies such as overheating, high vibration, high pressure, excessive power consumption, motor speed drop, and multi-signal degradation.

## How Pathway Is Used

`app/pathway_pipeline.py` defines a Pathway schema for incoming sensor events, reads CSV events from `data/stream/` in streaming mode, selects useful columns, computes a `health_index`, and writes processed rows to `data/processed/stream_processed.csv`.

Pathway is used for actual pipeline definition and execution when the real Pathway runtime is available. On Windows, the installed `pathway` package may expose a platform stub instead of the real engine; in that case the module falls back to a one-shot pandas processor so the rest of the prototype remains runnable. Docker/Linux is the recommended environment for demonstrating the continuous Pathway runtime.

## Machine Learning Approach

The anomaly model uses `IsolationForest` from scikit-learn. Numeric features are standardized with `StandardScaler`, and the scaler is saved together with the model in `models/anomaly_model.pkl`.

The final severity combines:

- Isolation Forest anomaly score;
- business thresholds for temperature, pressure, vibration, power consumption, and motor speed;
- optional health index when available.

Severity levels:

- normal
- low
- medium
- high
- critical

## AI Agent

The V1 assistant is rule-based and robust without an API key. It reads SQLite anomalies, identifies the likely signal drivers, explains why a machine is in alert, and recommends a technical action.

The code is structured for future LangGraph/OpenAI integration through tool-like functions and a dedicated rewrite extension point. If `OPENAI_API_KEY` is absent, the local deterministic fallback is used.

## API Documentation

Run:

```powershell
uvicorn app.api:app --reload
```

Open:

```text
http://127.0.0.1:8000/docs
```

Endpoints:

- `GET /`
- `GET /health`
- `GET /metrics`
- `GET /events/latest`
- `GET /anomalies/latest`
- `GET /anomalies/{machine_id}`
- `GET /report`
- `POST /agent/query`

Example agent request:

```json
{
  "question": "Pourquoi MACHINE_03 est en alerte ?"
}
```

## Dashboard

Run:

```powershell
streamlit run app/dashboard.py
```

Sections:

- Overview
- Live Monitoring
- Anomalies
- AI Assistant
- Reports

The dashboard reads SQLite directly for simplicity and reliability.

## Installation

```powershell
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
```

## Run Locally

Initialize data and model:

```powershell
python -m app.data_generator
python -m app.anomaly_detector --train
python -m app.database
```

Launch the stream simulator:

```powershell
python -m app.streaming_simulator
```

Launch the Pathway pipeline in another terminal:

```powershell
python -m app.pathway_pipeline
```

Run anomaly prediction on processed stream data:

```powershell
python -m app.anomaly_detector --predict
```

Launch the API:

```powershell
uvicorn app.api:app --reload
```

Launch the dashboard:

```powershell
streamlit run app/dashboard.py
```

## Run with Docker

Build and run the dashboard:

```powershell
docker build -t ai-realtime-monitoring-assistant .
docker run -p 8501:8501 ai-realtime-monitoring-assistant
```

Run API and dashboard:

```powershell
docker compose up --build
```

## Tests

```powershell
pytest
```

## Limitations

- All data is simulated.
- The project is a prototype and is not production-ready.
- Thresholds are illustrative and not calibrated on real equipment.
- The model is unsupervised and should be validated against expert labels before real deployment.
- The V1 assistant is rule-based, not a full LangGraph agent.

## Future Improvements

- Add LangGraph orchestration for the assistant.
- Add OpenAI-based answer rewriting when an API key is configured.
- Store stream output directly in SQLite from the processing layer.
- Add model monitoring and drift detection.
- Add richer sensor profiles per machine type.
- Add authentication and production-grade observability for the API.

## CV Pitch

Assistant IA de Surveillance Continue — Projet personnel

Développement d'un prototype de monitoring temps réel sur données capteurs industrielles simulées, intégrant Pathway pour le traitement continu, Isolation Forest pour la détection d'anomalies, FastAPI, Streamlit et un agent IA d'aide à l'analyse.

## GitHub Setup

```powershell
git init
git branch -M main
git add .
git commit -m "Initial commit: AI realtime monitoring assistant"
git remote add origin git@github.com:TON_USERNAME/ai-realtime-monitoring-assistant.git
git push -u origin main
```
