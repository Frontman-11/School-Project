import json
import os
import paho.mqtt.client as mqtt
from influxdb_client import InfluxDBClient, Point, WritePrecision
from influxdb_client.client.write_api import SYNCHRONOUS
from datetime import datetime, timezone
from dotenv import load_dotenv
from physics_and_models import train, predict

load_dotenv()

# ── HiveMQ credentials ────────────────────────────────────────────
MQTT_BROKER   = os.getenv("MQTT_BROKER")
MQTT_PORT     = int(os.getenv("MQTT_PORT", 8883))
MQTT_TOPIC    = os.getenv("MQTT_TOPIC", "solar/#")
MQTT_USER     = os.getenv("MQTT_USER")
MQTT_PASSWORD = os.getenv("MQTT_PASSWORD")

# ── InfluxDB credentials ──────────────────────────────────────────
INFLUX_URL    = os.getenv("INFLUX_URL")
INFLUX_TOKEN  = os.getenv("INFLUX_TOKEN")
INFLUX_ORG    = os.getenv("INFLUX_ORG")
INFLUX_BUCKET = os.getenv("INFLUX_BUCKET")

# ── Home defaults (overridden per-message if ESP32 sends them) ────
HOME_ID             = os.getenv("HOME_ID",             "home1")
HOME_LAT            = float(os.getenv("HOME_LAT",      4.8156))
HOME_LON            = float(os.getenv("HOME_LON",      7.0498))
BATTERY_TYPE        = os.getenv("BATTERY_TYPE",        "LEAD_ACID")
NOMINAL_VOLTAGE     = os.getenv("NOMINAL_VOLTAGE",     "12V")
BATTERY_CAPACITY_WH = int(os.getenv("BATTERY_CAPACITY_WH", 100))

# ── InfluxDB client ───────────────────────────────────────────────
influx_client = InfluxDBClient(url=INFLUX_URL, token=INFLUX_TOKEN, org=INFLUX_ORG)
write_api     = influx_client.write_api(write_options=SYNCHRONOUS)


# ── Write raw sensor reading ──────────────────────────────────────
def write_sensor_reading(payload: dict, recorded_at: datetime):
    point = (
        Point("sensor_reading")
        .tag("home_id",      payload["home_id"])
        .tag("battery_type", payload["battery_type"])
        .field("solar_voltage",   float(payload["solar_voltage"]))
        .field("solar_current",   float(payload["solar_current"]))
        .field("battery_voltage", float(payload["battery_voltage"]))
        .field("battery_current", float(payload["battery_current"]))
        .field("load_current",    float(payload["load_current"]))
        .field("temperature",     float(payload["temperature"]))
        .time(recorded_at, WritePrecision.S)
    )
    write_api.write(bucket=INFLUX_BUCKET, org=INFLUX_ORG, record=point)


# ── Write model prediction ────────────────────────────────────────
def write_model_prediction(result: dict, home_id: str, recorded_at: datetime):
    point = (
        Point("model_prediction")
        .tag("home_id", home_id)
        .field("forecast_for",      result["forecast_for"])
        .field("solar_power_now_w", float(result["solar_power_now_w"]))
        .field("load_power_now_w",  float(result["load_power_now_w"]))
        .field("soc_now_percent",   float(result["soc_now_percent"]))
        .field("solar_next_w",      float(result["solar_next_w"]))
        .field("load_next_w",       float(result["load_next_w"]))
        .field("runtime_hours",     float(result["runtime_hours"]))
        .field("cloud_cover_pct",   float(result["cloud_cover_pct"]))
        .field("weather_condition", result["weather_condition"])
        .field("soc_physics_pct",   float(result["soc_physics_pct"]))
        .field("soc_coulomb_pct",   float(result["soc_coulomb_pct"]))
        .time(recorded_at, WritePrecision.S)
    )
    write_api.write(bucket=INFLUX_BUCKET, org=INFLUX_ORG, record=point)


# ── MQTT callbacks ────────────────────────────────────────────────
def on_connect(client, userdata, flags, rc, properties=None):
    if rc == 0:
        print("[MQTT] Connected to HiveMQ")
        client.subscribe(MQTT_TOPIC)
        print(f"[MQTT] Subscribed to {MQTT_TOPIC}")
    else:
        print(f"[MQTT] Connection failed with code {rc}")


def on_message(client, userdata, msg):
    try:
        payload = json.loads(msg.payload.decode())
        print(f"\n[IN] {payload}")

        # use ESP32 timestamp if sent, else server time
        if "recorded_at" in payload:
            recorded_at = datetime.fromisoformat(payload["recorded_at"])
        else:
            recorded_at = datetime.now(timezone.utc)
            payload["recorded_at"] = recorded_at.isoformat()

        # fill in config fields from env if ESP32 does not send them
        payload.setdefault("home_id",             HOME_ID)
        payload.setdefault("lat",                 HOME_LAT)
        payload.setdefault("lon",                 HOME_LON)
        payload.setdefault("battery_type",        BATTERY_TYPE)
        payload.setdefault("nominal_voltage",     NOMINAL_VOLTAGE)
        payload.setdefault("battery_capacity_wh", BATTERY_CAPACITY_WH)

        # 1. write raw sensor data
        write_sensor_reading(payload, recorded_at)
        print("[DB] sensor_reading written")

        # 2. train on new reading (uses previous saved state internally)
        train(payload)
        print("[ML] trained")

        # 3. predict with new reading
        result = predict(payload)
        print(f"[ML] {result}")

        # 4. write model prediction
        write_model_prediction(result, payload["home_id"], recorded_at)
        print("[DB] model_prediction written")

    except Exception as e:
        print(f"[ERROR] {e}")


# ── MQTT client setup ─────────────────────────────────────────────
client = mqtt.Client(mqtt.CallbackAPIVersion.VERSION2)
client.username_pw_set(MQTT_USER, MQTT_PASSWORD)
client.tls_set()
client.on_connect = on_connect
client.on_message = on_message

print(f"[MQTT] Connecting to {MQTT_BROKER}...")
client.connect(MQTT_BROKER, MQTT_PORT, keepalive=60)
client.loop_forever()
