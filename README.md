# Solar Energy Management System — ML & Cloud Pipeline

## Folder structure
```
├── api.py                     FastAPI — single entry point
├── mqtt_to_influx.py          MQTT bridge → calls FastAPI /ingest
├── scheduler.py               Hourly/daily/monthly forecast jobs
├── mock_esp32.py              Optional test simulator (independent)
├── core/
│   ├── physics_and_models.py  5-min per-home River models
│   └── forecast_models.py     Hourly/daily/monthly per-home models
├── db/
│   └── influx_client.py       InfluxDB read/write helpers
├── utils/
│   ├── constants.py           Voltage-SOC curves + time encoding
│   ├── weather.py             OpenWeatherMap integration
│   └── home_registry.py      Home config storage (homes/registry.json)
├── .env                       Secrets (never commit)
├── .env.example               Template for new deployments
└── requirements.txt
```

## Setup
```bash
pip install -r requirements.txt
```

## Running (3 terminals)
```bash
# Terminal 1 — FastAPI
python3 api.py

# Terminal 2 — MQTT bridge
python3 mqtt_to_influx.py

# Terminal 3 — Scheduler
python3 scheduler.py
```

## Testing with mock data (optional 4th terminal)
```bash
python3 mock_esp32.py
```

## Register a home before ingesting data
```bash
curl -X POST http://localhost:8000/homes/register \
  -H "X-API-Key: solar-pipeline-secret-key-2026" \
  -H "Content-Type: application/json" \
  -d '{
    "home_id": "home1",
    "lat": 4.8156,
    "lon": 7.0498,
    "battery_type": "LEAD_ACID",
    "nominal_voltage": "12V",
    "battery_capacity_wh": 100
  }'
```

## API docs
Visit http://localhost:8000/docs after starting the API.

## API Key
Set `X-API-Key` header on every request. Key is in `.env` as `API_KEY`.
