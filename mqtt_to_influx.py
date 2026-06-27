import json
import paho.mqtt.client as mqtt
from influxdb_client import InfluxDBClient, Point, WritePrecision
from influxdb_client.client.write_api import SYNCHRONOUS
from datetime import datetime, timezone
from physics_and_models import train, predict

# ── HiveMQ credentials ────────────────────────────────────────────
MQTT_BROKER   = "68d36c1becfe4592a74352c9d79b150b.s1.eu.hivemq.cloud"
MQTT_PORT     = 8883
MQTT_TOPIC    = "solar/#"
MQTT_USER     = "cloud_subscriber"
MQTT_PASSWORD = "rw@LeHx!Bd8Xc3J"

# ── InfluxDB credentials ──────────────────────────────────────────
INFLUX_URL    = "https://us-east-1-1.aws.cloud2.influxdata.com"
INFLUX_TOKEN  = "o_0ud-jO7o-4pmgqigCQci1pCqpjAa5oAjW8k1gW11UnzQqmD6XezEbmnfoMKJaaWHU9Vk0QPiC6DRZAVNk87A=="
INFLUX_ORG    = "School-project"
INFLUX_BUCKET = "solar_data"

# ── InfluxDB client ───────────────────────────────────────────────
influx_client = InfluxDBClient(
    url=INFLUX_URL,
    token=INFLUX_TOKEN,
    org=INFLUX_ORG
)
write_api = influx_client.write_api(write_options=SYNCHRONOUS)


# ── Write raw sensor reading ──────────────────────────────────────
def write_sensor_reading(payload, recorded_at):
    point = (
        Point("sensor_reading")
        .tag("home_id",       payload.get("home_id", "unknown"))
        .tag("battery_type",  payload.get("battery_type", "LEAD_ACID"))
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
def write_model_prediction(result, home_id, recorded_at):
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
        print("Connected to HiveMQ")
        client.subscribe(MQTT_TOPIC)
        print(f"Subscribed to {MQTT_TOPIC}")
    else:
        print(f"Connection failed with code {rc}")


def on_message(client, userdata, msg):
    try:
        payload = json.loads(msg.payload.decode())
        print(f"\n[IN] {payload}")

        # use ESP32 timestamp if provided, else server time
        if "recorded_at" in payload:
            recorded_at = datetime.fromisoformat(payload["recorded_at"])
        else:
            recorded_at = datetime.now(timezone.utc)
            payload["recorded_at"] = recorded_at.isoformat()

        # set defaults for config fields if not sent by ESP32
        payload.setdefault("battery_type",        "LEAD_ACID")
        payload.setdefault("nominal_voltage",      "12V")
        payload.setdefault("battery_capacity_wh",  100)
        payload.setdefault("lat",                  4.8156)
        payload.setdefault("lon",                  7.0498)

        # 1. write raw sensor data
        write_sensor_reading(payload, recorded_at)
        print("[DB] sensor_reading written")

        # 2. train on new reading (uses previous state internally)
        train(payload)
        print("[ML] trained")

        # 3. predict with new reading
        result = predict(payload)
        print(f"[ML] prediction: {result}")

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

print("Connecting to HiveMQ...")
client.connect(MQTT_BROKER, MQTT_PORT, keepalive=60)
client.loop_forever()
