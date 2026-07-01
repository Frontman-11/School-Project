"""
MQTT bridge — subscribes to HiveMQ and forwards sensor readings
to the FastAPI /ingest endpoint. This keeps the MQTT listener
thin and ensures all business logic lives in one place (the API).
"""

import json
import os
import httpx
import paho.mqtt.client as mqtt
from dotenv import load_dotenv

load_dotenv()

MQTT_BROKER   = os.getenv("MQTT_BROKER")
MQTT_PORT     = int(os.getenv("MQTT_PORT", 8883))
MQTT_TOPIC    = os.getenv("MQTT_TOPIC", "solar/#")
MQTT_USER     = os.getenv("MQTT_USER")
MQTT_PASSWORD = os.getenv("MQTT_PASSWORD")

API_BASE      = f"http://localhost:{os.getenv('API_PORT', 8000)}"
API_KEY       = os.getenv("API_KEY")
HEADERS       = {"X-API-Key": API_KEY, "Content-Type": "application/json"}

# home defaults — only used if ESP32 does not send them
# these are overridden by the registered home config stored in the API
HOME_ID             = os.getenv("HOME_ID", "home1")
BATTERY_TYPE        = os.getenv("BATTERY_TYPE", "LEAD_ACID")
NOMINAL_VOLTAGE     = os.getenv("NOMINAL_VOLTAGE", "12V")
BATTERY_CAPACITY_WH = int(os.getenv("BATTERY_CAPACITY_WH", 100))
HOME_LAT            = float(os.getenv("HOME_LAT", 4.8156))
HOME_LON            = float(os.getenv("HOME_LON", 7.0498))


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

        home_id = payload.get("home_id", HOME_ID)

        # build sensor reading body for FastAPI
        reading = {
            "solar_voltage":   float(payload["solar_voltage"]),
            "solar_current":   float(payload["solar_current"]),
            "battery_voltage": float(payload["battery_voltage"]),
            "battery_current": float(payload["battery_current"]),
            "load_current":    float(payload["load_current"]),
            "temperature":     float(payload["temperature"]),
            "recorded_at":     payload.get("recorded_at"),
        }

        # POST to FastAPI /ingest/{home_id}
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
            _auto_register(home_id, payload)
        else:
            print(f"[API] Error {resp.status_code}: {resp.text}")

    except Exception as e:
        print(f"[ERROR] {e}")


def _auto_register(home_id: str, payload: dict):
    """
    If the home is not registered yet, register it automatically
    using env defaults. In production the backend does this explicitly.
    """
    config = {
        "home_id":             home_id,
        "lat":                 payload.get("lat",                 HOME_LAT),
        "lon":                 payload.get("lon",                 HOME_LON),
        "battery_type":        payload.get("battery_type",        BATTERY_TYPE),
        "nominal_voltage":     payload.get("nominal_voltage",     NOMINAL_VOLTAGE),
        "battery_capacity_wh": payload.get("battery_capacity_wh", BATTERY_CAPACITY_WH),
    }
    try:
        resp = httpx.post(
            f"{API_BASE}/homes/register",
            json=config,
            headers=HEADERS,
            timeout=10,
        )
        print(f"[API] Auto-registered: {resp.json()}")
    except Exception as e:
        print(f"[API] Auto-register failed: {e}")


client = mqtt.Client(mqtt.CallbackAPIVersion.VERSION2)
client.username_pw_set(MQTT_USER, MQTT_PASSWORD)
client.tls_set()
client.on_connect = on_connect
client.on_message = on_message

print(f"[MQTT] Connecting to {MQTT_BROKER}...")
client.connect(MQTT_BROKER, MQTT_PORT, keepalive=60)
client.loop_forever()
