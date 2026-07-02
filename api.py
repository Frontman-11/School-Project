"""
FastAPI — single entry point for the solar pipeline.
All ML logic, InfluxDB writes, and forecast reads go through here.
The mobile app backend, MQTT bridge, and scheduler all call this API.
"""

import os
import logging
import threading
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from typing import Optional

import httpx
import paho.mqtt.client as mqtt
from apscheduler.schedulers.background import BackgroundScheduler
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, Security, Depends
from fastapi.security.api_key import APIKeyHeader
from pydantic import BaseModel

from utils.home_registry import register_home, get_home, list_homes, home_exists
from core.physics_and_models import train, predict
from core.forecast_models import (
    run_hourly_forecast,
    run_daily_forecast,
    run_monthly_forecast,
)
from db.influx_client import (
    write_sensor_reading,
    write_model_prediction,
    write_forecast,
    get_latest_prediction,
    get_latest_sensor,
    get_aggregate,
    get_temperature_mean,
    get_latest_forecast,
)

load_dotenv()
logger = logging.getLogger(name=__file__)

API_KEY = os.getenv("API_KEY", "solar-pipeline-secret-key-2026")
API_BASE = f"http://localhost:{os.getenv('API_PORT', 8000)}"
HEADERS = {"X-API-Key": API_KEY}
api_key_header = APIKeyHeader(name="X-API-Key", auto_error=False)

# ── MQTT config ───────────────────────────────────────────────────
MQTT_BROKER = os.getenv("MQTT_BROKER")
MQTT_PORT = int(os.getenv("MQTT_PORT", 8883))
MQTT_TOPIC = os.getenv("MQTT_TOPIC", "solar/#")
MQTT_USER = os.getenv("MQTT_USER")
MQTT_PASSWORD = os.getenv("MQTT_PASSWORD")
HOME_ID = os.getenv("HOME_ID", "home1")
BATTERY_TYPE = os.getenv("BATTERY_TYPE", "LEAD_ACID")
NOMINAL_VOLTAGE = os.getenv("NOMINAL_VOLTAGE", "12V")
BATTERY_CAPACITY_WH = int(os.getenv("BATTERY_CAPACITY_WH", 100))
HOME_LAT = float(os.getenv("HOME_LAT", 4.8156))
HOME_LON = float(os.getenv("HOME_LON", 7.0498))


# ── MQTT handlers ─────────────────────────────────────────────────
def on_connect(client, userdata, flags, rc, properties=None):
    if rc == 0:
        print("[MQTT] Connected to HiveMQ")
        client.subscribe(MQTT_TOPIC)
        print(f"[MQTT] Subscribed to {MQTT_TOPIC}")
    else:
        print(f"[MQTT] Connection failed with code {rc}")


def on_message(client, userdata, msg):
    try:
        import json

        payload = json.loads(msg.payload.decode())
        print(f"\n[IN] {payload}")
        home_id = payload.get("home_id", HOME_ID)

        reading = {
            "solar_voltage": float(payload["solar_voltage"]),
            "solar_current": float(payload["solar_current"]),
            "battery_voltage": float(payload["battery_voltage"]),
            "battery_current": float(payload["battery_current"]),
            "load_current": float(payload["load_current"]),
            "temperature": float(payload["temperature"]),
            "recorded_at": payload.get("recorded_at"),
        }

        resp = httpx.post(
            f"{API_BASE}/ingest/{home_id}",
            json=reading,
            headers=HEADERS,
            timeout=10,
        )

        if resp.status_code == 200:
            print(f"[API] {resp.json()}")
        elif resp.status_code == 404:
            print(f"[API] Home '{home_id}' not registered — auto-registering...")
            config = {
                "home_id": home_id,
                "lat": payload.get("lat", HOME_LAT),
                "lon": payload.get("lon", HOME_LON),
                "battery_type": payload.get("battery_type", BATTERY_TYPE),
                "nominal_voltage": payload.get("nominal_voltage", NOMINAL_VOLTAGE),
                "battery_capacity_wh": payload.get(
                    "battery_capacity_wh", BATTERY_CAPACITY_WH
                ),
            }
            httpx.post(
                f"{API_BASE}/homes/register", json=config, headers=HEADERS, timeout=10
            )
        else:
            print(f"[API] Error {resp.status_code}: {resp.text}")

    except Exception as e:
        print(f"[MQTT ERROR] {e}")


def start_mqtt():
    mqtt_client = mqtt.Client(mqtt.CallbackAPIVersion.VERSION2)
    mqtt_client.username_pw_set(MQTT_USER, MQTT_PASSWORD)
    mqtt_client.tls_set()
    mqtt_client.on_connect = on_connect
    mqtt_client.on_message = on_message
    print(f"[MQTT] Connecting to {MQTT_BROKER}...")
    mqtt_client.connect(MQTT_BROKER, MQTT_PORT, keepalive=60)
    mqtt_client.loop_forever()


# ── Scheduler jobs ────────────────────────────────────────────────
def hourly_job():
    homes = list_homes()
    print(f"\n[Scheduler] Hourly forecast for {homes}")
    for home_id in homes:
        try:
            agg = get_aggregate(home_id, "-1h")
            temp_c = get_temperature_mean(home_id, "-1h")
            if not agg:
                continue
            result = run_hourly_forecast(home_id, agg, temp_c)
            write_forecast("hourly_forecast", home_id, result, result["forecast_for"])
            print(f"[Scheduler] hourly done for {home_id}: {result}")
        except Exception as e:
            print(f"[Scheduler] hourly error for {home_id}: {e}")


def daily_job():
    homes = list_homes()
    print(f"\n[Scheduler] Daily forecast for {homes}")
    for home_id in homes:
        try:
            agg = get_aggregate(home_id, "-24h")
            temp_c = get_temperature_mean(home_id, "-24h")
            if not agg:
                continue
            result = run_daily_forecast(home_id, agg, temp_c)
            write_forecast("daily_forecast", home_id, result, result["forecast_for"])
            print(f"[Scheduler] daily done for {home_id}: {result}")
        except Exception as e:
            print(f"[Scheduler] daily error for {home_id}: {e}")


def monthly_job():
    homes = list_homes()
    print(f"\n[Scheduler] Monthly forecast for {homes}")
    for home_id in homes:
        try:
            agg = get_aggregate(home_id, "-30d")
            temp_c = get_temperature_mean(home_id, "-30d")
            if not agg:
                continue
            result = run_monthly_forecast(home_id, agg, temp_c)
            write_forecast("monthly_forecast", home_id, result, result["forecast_for"])
            print(f"[Scheduler] monthly done for {home_id}: {result}")
        except Exception as e:
            print(f"[Scheduler] monthly error for {home_id}: {e}")


# ── Lifespan — starts MQTT + scheduler when API boots ────────────
@asynccontextmanager
async def lifespan(app: FastAPI):
    # start MQTT in background thread
    mqtt_thread = threading.Thread(target=start_mqtt, daemon=True)
    mqtt_thread.start()
    print("[API] MQTT listener started")

    # start scheduler
    scheduler = BackgroundScheduler(timezone="UTC")
    scheduler.add_job(hourly_job, "interval", hours=1)
    scheduler.add_job(daily_job, "cron", hour=0, minute=5)
    scheduler.add_job(monthly_job, "cron", day=1, hour=1, minute=0)
    scheduler.start()
    print("[API] Scheduler started")

    yield  # API runs here

    scheduler.shutdown()
    print("[API] Scheduler stopped")


app = FastAPI(
    title="Solar Energy Management API",
    description="ML pipeline for real-time solar monitoring and prediction",
    version="1.0.0",
    lifespan=lifespan,
)


# ── Auth ──────────────────────────────────────────────────────────


async def require_api_key(key: str = Security(api_key_header)):
    logger.info(key)
    if key != API_KEY:
        raise HTTPException(status_code=403, detail="Invalid or missing API key")
    return key


# ── Pydantic models ───────────────────────────────────────────────


class HomeConfig(BaseModel):
    home_id: str
    lat: float
    lon: float
    battery_type: str = "LEAD_ACID"
    nominal_voltage: str = "12V"
    battery_capacity_wh: int = 100


class SensorReading(BaseModel):
    solar_voltage: float
    solar_current: float
    battery_voltage: float
    battery_current: float
    load_current: float
    temperature: float
    recorded_at: Optional[str] = None


# ── Health ────────────────────────────────────────────────────────


@app.get("/health")
def health():
    return {"status": "ok", "timestamp": datetime.now(timezone.utc).isoformat()}


# ── Home registration ─────────────────────────────────────────────


@app.post("/homes/register", dependencies=[Depends(require_api_key)])
def register(config: HomeConfig):
    """
    Register a new home or update an existing home's config.
    Must be called once before any data is ingested for a home.
    """
    logger.info(config)
    logger.info(require_api_key)
    saved = register_home(config.model_dump())
    return {"message": "Home registered", "home": saved}


@app.get("/homes", dependencies=[Depends(require_api_key)])
def get_homes():
    return {"homes": list_homes()}


@app.get("/homes/{home_id}", dependencies=[Depends(require_api_key)])
def get_home_config(home_id: str):
    config = get_home(home_id)
    if not config:
        raise HTTPException(status_code=404, detail=f"Home '{home_id}' not registered")
    return config


# ── Ingest ────────────────────────────────────────────────────────


@app.post("/ingest/{home_id}", dependencies=[Depends(require_api_key)])
def ingest(home_id: str, reading: SensorReading):
    """
    Receives a sensor reading from the ESP32 (via MQTT bridge or direct).
    Trains the 5-min model, generates a prediction, writes both to InfluxDB.
    Returns the prediction result.
    """
    config = get_home(home_id)
    if not config:
        raise HTTPException(
            status_code=404,
            detail=f"Home '{home_id}' not registered. Call POST /homes/register first.",
        )

    recorded_at = (
        datetime.fromisoformat(reading.recorded_at)
        if reading.recorded_at
        else datetime.now(timezone.utc)
    )

    # merge sensor reading with home config into one flat dict
    data = {
        **config,
        **reading.model_dump(exclude={"recorded_at"}),
        "recorded_at": recorded_at.isoformat(),
    }

    # train → predict → write
    train(data)
    result = predict(data)

    write_sensor_reading(data, recorded_at)
    write_model_prediction(result, home_id, recorded_at)

    return result


# ── Current readings ──────────────────────────────────────────────


@app.get("/current/{home_id}", dependencies=[Depends(require_api_key)])
def current(home_id: str):
    """
    Returns the latest sensor reading and 5-min prediction from InfluxDB.
    This is what the app displays as the live dashboard.
    """
    _check_home(home_id)
    prediction = get_latest_prediction(home_id)
    sensor = get_latest_sensor(home_id)

    if not prediction and not sensor:
        raise HTTPException(status_code=404, detail="No data yet for this home")

    return {
        "home_id": home_id,
        "sensor": sensor,
        "prediction": prediction,
    }


# ── Averages ──────────────────────────────────────────────────────


@app.get("/averages/{home_id}", dependencies=[Depends(require_api_key)])
def averages(home_id: str):
    """
    Returns today's average solar and load power.
    Used by app items 1 (avg power usage today) and 3 (avg solar today).
    """
    _check_home(home_id)
    agg = get_aggregate(home_id, "-24h")
    if not agg:
        raise HTTPException(status_code=404, detail="Not enough data yet")
    return {"home_id": home_id, "period": "last_24h", "averages": agg}


# ── Forecasts ─────────────────────────────────────────────────────


@app.get("/forecast/hourly/{home_id}", dependencies=[Depends(require_api_key)])
def forecast_hourly(home_id: str):
    """
    Returns the latest pre-computed 1-hour ahead forecast.
    Updated every hour by the scheduler.
    """
    _check_home(home_id)
    result = get_latest_forecast(home_id, "hourly_forecast")
    if not result:
        raise HTTPException(
            status_code=404, detail="No hourly forecast yet. Scheduler runs every hour."
        )
    return {"home_id": home_id, "forecast": result}


@app.get("/forecast/daily/{home_id}", dependencies=[Depends(require_api_key)])
def forecast_daily(home_id: str):
    """
    Returns the latest pre-computed daily (tomorrow) forecast.
    Updated at midnight by the scheduler.
    """
    _check_home(home_id)
    result = get_latest_forecast(home_id, "daily_forecast")
    if not result:
        raise HTTPException(
            status_code=404, detail="No daily forecast yet. Scheduler runs at midnight."
        )
    return {"home_id": home_id, "forecast": result}


@app.get("/forecast/monthly/{home_id}", dependencies=[Depends(require_api_key)])
def forecast_monthly(home_id: str):
    """
    Returns the latest pre-computed monthly forecast.
    Updated on the 1st of each month by the scheduler.
    """
    _check_home(home_id)
    result = get_latest_forecast(home_id, "monthly_forecast")
    if not result:
        raise HTTPException(
            status_code=404,
            detail="No monthly forecast yet. Scheduler runs on the 1st of each month.",
        )
    return {"home_id": home_id, "forecast": result}


@app.get("/forecast/custom/{home_id}", dependencies=[Depends(require_api_key)])
def forecast_custom(
    home_id: str, hours: Optional[float] = None, days: Optional[float] = None
):
    """
    On-demand forecast for an arbitrary horizon.
    Pass either ?hours=X or ?days=X.
    Routes to the appropriate pre-computed forecast based on the horizon.
    - Up to 1 hour   → hourly model
    - 1h to 30 days  → daily model
    - Beyond 30 days → monthly model
    """
    _check_home(home_id)

    if hours is None and days is None:
        raise HTTPException(status_code=400, detail="Provide ?hours=X or ?days=X")

    total_hours = hours if hours is not None else (days * 24)

    if total_hours <= 1:
        result = get_latest_forecast(home_id, "hourly_forecast")
        model_used = "hourly"
    elif total_hours <= 720:  # up to 30 days
        result = get_latest_forecast(home_id, "daily_forecast")
        model_used = "daily"
    else:
        result = get_latest_forecast(home_id, "monthly_forecast")
        model_used = "monthly"

    if not result:
        raise HTTPException(
            status_code=404, detail=f"No {model_used} forecast available yet"
        )

    return {
        "home_id": home_id,
        "requested": f"{total_hours}h ahead",
        "model_used": model_used,
        "note": _horizon_note(model_used),
        "forecast": result,
    }


# ── Scheduler trigger endpoints (called by scheduler.py) ──────────


@app.post("/internal/run-hourly/{home_id}", dependencies=[Depends(require_api_key)])
def trigger_hourly(home_id: str):
    _check_home(home_id)
    agg = get_aggregate(home_id, "-1h")
    temp_c = get_temperature_mean(home_id, "-1h")
    if not agg:
        return {"message": "Not enough data yet"}
    result = run_hourly_forecast(home_id, agg, temp_c)
    write_forecast("hourly_forecast", home_id, result, result["forecast_for"])
    return result


@app.post("/internal/run-daily/{home_id}", dependencies=[Depends(require_api_key)])
def trigger_daily(home_id: str):
    _check_home(home_id)
    agg = get_aggregate(home_id, "-24h")
    temp_c = get_temperature_mean(home_id, "-24h")
    if not agg:
        return {"message": "Not enough data yet"}
    result = run_daily_forecast(home_id, agg, temp_c)
    write_forecast("daily_forecast", home_id, result, result["forecast_for"])
    return result


@app.post("/internal/run-monthly/{home_id}", dependencies=[Depends(require_api_key)])
def trigger_monthly(home_id: str):
    _check_home(home_id)
    agg = get_aggregate(home_id, "-30d")
    temp_c = get_temperature_mean(home_id, "-30d")
    if not agg:
        return {"message": "Not enough data yet"}
    result = run_monthly_forecast(home_id, agg, temp_c)
    write_forecast("monthly_forecast", home_id, result, result["forecast_for"])
    return result


# ── Helpers ───────────────────────────────────────────────────────


def _check_home(home_id: str):
    if not home_exists(home_id):
        raise HTTPException(
            status_code=404,
            detail=f"Home '{home_id}' not registered. Call POST /homes/register first.",
        )


def _horizon_note(model: str) -> str:
    notes = {
        "hourly": "Trained on hourly data. Reliable up to ~1 hour ahead.",
        "daily": "Trained on daily summaries. Best for day-level estimates.",
        "monthly": "Trained on monthly summaries. Broad trend only.",
    }
    return notes.get(model, "")


# ── Run ───────────────────────────────────────────────────────────

if __name__ == "__main__":
    import uvicorn

    host = os.getenv("API_HOST", "0.0.0.0")
    port = int(os.getenv("API_PORT", 8000))
    uvicorn.run("api:app", host=host, port=port, reload=True)
