"""
FastAPI -- single entry point for the solar pipeline.

Flow, in order:
  1. The ESP32 publishes a sensor reading to HiveMQ. This happens
     automatically and continuously, on the device's own schedule --
     nothing here waits for or requires a frontend action.
  2. The MQTT listener (a background thread started in the lifespan
     below) receives that message the instant it is published and
     forwards it to POST /ingest, in this same process, over loopback.
  3. /ingest auto-provisions a home_config row for the device_id if one
     does not exist yet (with placeholder defaults), trains the 5-minute
     models on the previous reading, predicts the next 5 minutes,
     and writes the raw reading + prediction to InfluxDB.
  4. The frontend can register/update the real config for that same
     device_id at any time via POST /homes/register -- it does not need
     to happen before step 1-3 start working, only before the physics
     baseline reflects the real battery/location.
  5. The frontend polls GET /current, /history, /forecast/* to display
     data. It never needs to "trigger" training or prediction -- that
     already happened automatically in step 2-3, continuously, driven
     purely by the device's own publish schedule.

This is a true publish/subscribe pipeline: the MQTT thread is always
subscribed and always reacting to whatever the device sends, with no
polling or frontend-initiated step anywhere in the ingest path.
"""

import os
import json
import logging
import threading
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from typing import Optional, Literal

import httpx
import paho.mqtt.client as mqtt
from apscheduler.schedulers.background import BackgroundScheduler
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, Security, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security.api_key import APIKeyHeader
from pydantic import BaseModel

from utils.home_registry import register_home, get_home, list_homes, home_exists
from utils.esp_payload import ESPPayload, flatten_esp_payload
from utils.weather import clear_weather_cache
from core import model_store
from core.physics_and_models import train, predict
from core.forecast_models import run_hourly_forecast, run_daily_forecast
from core.ingest_pipeline import process_reading_for_ml
from db.influx_client import (
    write_sensor_reading,
    write_model_prediction,
    write_forecast,
    get_latest_prediction,
    get_latest_sensor,
    get_last_valid_temperature,
    get_aggregate,
    get_temperature_mean,
    get_latest_forecast,
    get_history_log,
    load_pipeline_state,
    delete_home_data,
)

load_dotenv()
logger = logging.getLogger(__name__)

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

# Placeholder config used only when a device sends data before the
# frontend has registered its real details. `configured: False` lets
# the frontend detect "this device exists but hasn't been set up yet".
DEFAULT_HOME_CONFIG = {
    "lat": 5.5167,
    "lon": 5.7500,
    "battery_type": "LEAD_ACID",
    "nominal_voltage": "12V",
    "battery_capacity_wh": 100,
}


def _ensure_home(home_id: str) -> dict:
    config = get_home(home_id)
    if config:
        return config
    config = {
        "home_id": home_id,
        **DEFAULT_HOME_CONFIG,
        "configured": False,
        "created_at": datetime.now(timezone.utc).isoformat(),
    }
    register_home(config)
    print(f"[API] Auto-provisioned new device '{home_id}' with placeholder config")
    return config


# ── MQTT handlers ─────────────────────────────────────────────────


def on_connect(client, userdata, flags, rc, properties=None):
    if rc == 0:
        print("[MQTT] Connected to HiveMQ")
        client.subscribe(MQTT_TOPIC)
        print(f"[MQTT] Subscribed to {MQTT_TOPIC}")
    else:
        print(f"[MQTT] Connection failed with code {rc}")


def on_message(client, userdata, msg):
    """
    Forwards the raw ESP JSON straight to /ingest. device_id is inside
    the payload itself now, so there is no topic-based routing to get
    wrong -- whatever device_id the message carries is the home_id it
    is stored and trained under.
    """
    try:
        payload_dict = json.loads(msg.payload.decode())
        print(f"\n[MQTT IN] {payload_dict}")

        resp = httpx.post(
            f"{API_BASE}/ingest",
            json=payload_dict,
            headers=HEADERS,
            timeout=15,
        )
        if resp.status_code == 200:
            print(f"[MQTT->API] ok: {resp.json()}")
        else:
            print(f"[MQTT->API] error {resp.status_code}: {resp.text}")

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


# ── Lifespan -- starts MQTT + scheduler when API boots ────────────


@asynccontextmanager
async def lifespan(app: FastAPI):
    mqtt_thread = threading.Thread(target=start_mqtt, daemon=True)
    mqtt_thread.start()
    print("[API] MQTT listener started")

    scheduler = BackgroundScheduler(timezone="UTC")
    scheduler.add_job(hourly_job, "interval", hours=1)
    scheduler.add_job(daily_job, "cron", hour=0, minute=5)
    scheduler.start()
    print("[API] Scheduler started (hourly + daily)")

    yield

    scheduler.shutdown()
    print("[API] Scheduler stopped")


app = FastAPI(
    title="Solar Energy Management API",
    description="ML pipeline for real-time solar monitoring and prediction",
    version="2.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── Auth ──────────────────────────────────────────────────────────


async def require_api_key(key: str = Security(api_key_header)):
    if key != API_KEY:
        raise HTTPException(status_code=403, detail="Invalid or missing API key")
    return key


# ── Pydantic models ───────────────────────────────────────────────


class HomeConfig(BaseModel):
    home_id: str
    lat: float
    lon: float
    # Literal so a bad value is rejected here (422 with a clear message)
    # instead of crashing the ingest pipeline later at
    # VOLTAGE_SOC_CURVE[battery_type][nominal_voltage].
    battery_type: Literal["LEAD_ACID", "LIFEPO4"] = "LEAD_ACID"
    nominal_voltage: Literal["12V", "24V", "48V"] = "12V"
    # Wh values can be fractional; int would 422 on e.g. 1200.5.
    battery_capacity_wh: float = 100


# ── Health ────────────────────────────────────────────────────────


@app.get("/health")
def health():
    return {"status": "ok", "timestamp": datetime.now(timezone.utc).isoformat()}


# ── Home registration ─────────────────────────────────────────────


@app.post("/homes/register", dependencies=[Depends(require_api_key)])
def register(config: HomeConfig):
    """
    Called by the frontend to attach real configuration (location,
    battery type/voltage/capacity) to a device_id. Can be called before
    or after the device has started sending data -- if the device
    already auto-provisioned a placeholder config, this overwrites it
    and flips configured to True.
    """
    saved = register_home({**config.model_dump(), "configured": True})
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


@app.delete("/homes/{home_id}", dependencies=[Depends(require_api_key)])
def delete_home(home_id: str):
    """
    Wipes every record tagged with this home_id from InfluxDB: sensor
    readings, predictions, forecasts, config, and all model/pipeline
    state. This is a development-stage reset tool, not something an
    end user should have easy access to -- if the same device_id sends
    new data afterwards, it simply starts over from a clean slate.
    """
    _check_home(home_id)
    delete_home_data(home_id)
    model_store.clear_home_cache(home_id)
    clear_weather_cache(home_id)
    return {"message": f"All data for '{home_id}' has been deleted."}


# ── Ingest ────────────────────────────────────────────────────────


@app.post("/ingest", dependencies=[Depends(require_api_key)])
def ingest(payload: ESPPayload):
    """
    Two independent steps, deliberately not sharing state:

    1. ESP -> InfluxDB. Parses the reading and writes it straight to
       sensor_reading. This always happens, and its success does not
       depend on anything ML-related below it.
    2. DB -> ML -> DB. Reads that same reading back out of InfluxDB
       (not off any in-memory object from step 1), trains, predicts,
       and writes the prediction back to InfluxDB.

    If step 2 throws for any reason -- a model error, a transient
    InfluxDB write failure, anything -- the reading from step 1 is
    already safely persisted. The response still returns 200 with
    ml_status: "unavailable" rather than losing the sensor data behind
    a 500, since the device's measurement was never actually at risk.
    """
    config = _ensure_home(payload.device_id)
    fallback_temp = get_last_valid_temperature(payload.device_id)
    data = flatten_esp_payload(payload, fallback_temp=fallback_temp)
    recorded_at = datetime.fromisoformat(data["recorded_at"])

    # Step 1 -- ESP -> DB, independent of ML.
    write_sensor_reading(
        {**data, "battery_type": config.get("battery_type", "LEAD_ACID")}, recorded_at
    )

    response = {
        "home_id": payload.device_id,
        "configured": config.get("configured", False),
        "temperature_valid": data["temperature_valid"],
        "recorded_at": data["recorded_at"],
        "ml_status": "ok",
    }

    # Step 2 -- DB -> ML -> DB.
    try:
        result = process_reading_for_ml(
            payload.device_id, expected_recorded_at=data["recorded_at"]
        )
        response.update(result)
    except Exception as e:
        logger.exception(f"[ML] processing failed for '{payload.device_id}'")
        response["ml_status"] = "unavailable"
        response["ml_error"] = str(e)

    return response


# ── Current readings ──────────────────────────────────────────────


@app.get("/current/{home_id}", dependencies=[Depends(require_api_key)])
def current(home_id: str):
    """Latest sensor reading + 5-min prediction. Powers the live dashboard."""
    _check_home(home_id)
    prediction = get_latest_prediction(home_id)
    sensor = get_latest_sensor(home_id)

    if not prediction and not sensor:
        raise HTTPException(status_code=404, detail="No data yet for this home")

    return {"home_id": home_id, "sensor": sensor, "prediction": prediction}


# ── History ───────────────────────────────────────────────────────


@app.get("/history/{home_id}", dependencies=[Depends(require_api_key)])
def history(home_id: str, hours: int = 24):
    """
    Per-reading log of actual vs predicted solar/load power. Default is
    the last 24 hours; pass ?hours=48 or ?hours=168 to look further back.
    Pass ?hours=0 for all available data.
    """
    _check_home(home_id)
    rows = get_history_log(home_id, hours=hours)
    if not rows:
        raise HTTPException(status_code=404, detail="No data yet for this home")
    return {"home_id": home_id, "period": f"last_{hours}h", "hours": rows}


# ── Averages ──────────────────────────────────────────────────────


@app.get("/averages/{home_id}", dependencies=[Depends(require_api_key)])
def averages(home_id: str):
    _check_home(home_id)
    agg = get_aggregate(home_id, "-24h")
    if not agg:
        raise HTTPException(status_code=404, detail="Not enough data yet")
    return {"home_id": home_id, "period": "last_24h", "averages": agg}


# ── Forecasts ─────────────────────────────────────────────────────


@app.get("/forecast/hourly/{home_id}", dependencies=[Depends(require_api_key)])
def forecast_hourly(home_id: str):
    _check_home(home_id)
    result = get_latest_forecast(home_id, "hourly_forecast")
    if not result:
        raise HTTPException(
            status_code=404, detail="No hourly forecast yet. Scheduler runs every hour."
        )
    return {"home_id": home_id, "forecast": result}


@app.get("/forecast/daily/{home_id}", dependencies=[Depends(require_api_key)])
def forecast_daily(home_id: str):
    _check_home(home_id)
    result = get_latest_forecast(home_id, "daily_forecast")
    if not result:
        raise HTTPException(
            status_code=404, detail="No daily forecast yet. Scheduler runs at midnight."
        )
    return {"home_id": home_id, "forecast": result}


@app.get("/forecast/custom/{home_id}", dependencies=[Depends(require_api_key)])
def forecast_custom(
    home_id: str, hours: Optional[float] = None, days: Optional[float] = None
):
    """
    On-demand forecast for an arbitrary horizon.
    Up to 1 hour -> hourly model. Beyond that -> daily model.
    """
    _check_home(home_id)
    if hours is None and days is None:
        raise HTTPException(status_code=400, detail="Provide ?hours=X or ?days=X")

    total_hours = hours if hours is not None else (days * 24)

    if total_hours <= 1:
        result = get_latest_forecast(home_id, "hourly_forecast")
        model_used = "hourly"
    else:
        result = get_latest_forecast(home_id, "daily_forecast")
        model_used = "daily"

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


# ── Scheduler trigger endpoints (also usable for backfilling) ─────


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


# ── Helpers ───────────────────────────────────────────────────────


def _check_home(home_id: str):
    if not home_exists(home_id):
        raise HTTPException(
            status_code=404,
            detail=f"Home '{home_id}' not registered. It will appear automatically once the device sends its first reading, or you can call POST /homes/register.",
        )


def _horizon_note(model: str) -> str:
    notes = {
        "hourly": "Trained on hourly data. Reliable up to ~1 hour ahead.",
        "daily": "Trained on daily summaries. Best for day-level estimates.",
    }
    return notes.get(model, "")


# ── Run ───────────────────────────────────────────────────────────

if __name__ == "__main__":
    import uvicorn

    host = os.getenv("API_HOST", "0.0.0.0")
    port = int(os.getenv("API_PORT", 8000))
    uvicorn.run("api:app", host=host, port=port, reload=True)
