"""
Minimal standalone MQTT publisher -- connects, publishes ONE message,
disconnects. No FastAPI, no InfluxDB. Use this together with
debug_subscribe.py (running in another terminal) to confirm a message
sent here actually shows up there -- proving the broker path works
end to end, independent of the ESP or the backend.

    python3 debug_publish.py
"""

import json
import threading
import paho.mqtt.client as mqtt

MQTT_BROKER = "68d36c1becfe4592a74352c9d79b150b.s1.eu.hivemq.cloud"
MQTT_PORT = 8883
MQTT_TOPIC = "solar/solar_ems_001"
MQTT_USER = "esp32_home1"
MQTT_PASSWORD = "hivemqpassword123"

PAYLOAD = {
    "device_id": "solar_ems_001",
    "timestamp": "2026-08-23T18:00:00Z",
    "solar": {"voltage_v": 18.4, "current_a": 2.1, "power_w": 38.6, "shunt_mv": 5.3},
    "battery": {
        "voltage_v": 12.6,
        "current_a": 1.8,
        "power_w": 22.7,
        "shunt_mv": 2.7,
        "charging": True,
    },
    "load": {"current_a": 3.2},
    "temperature_c": 31.5,
    "interval_s": 300,
}

connected = threading.Event()


connect_count = [0]


def on_connect(client, userdata, flags, rc, properties=None):
    connect_count[0] += 1
    print(
        f"[CONNECT #{connect_count[0]}] rc={rc} ({'OK' if rc == 0 else 'FAILED'}) flags={flags}"
    )
    if rc == 0:
        connected.set()


def on_disconnect(client, userdata, flags, rc, properties=None):
    print(f"[DISCONNECT] rc={rc}")
    connected.clear()


def on_publish(client, userdata, mid, reason_codes=None, properties=None):
    print(f"[PUBLISHED] mid={mid}")


client = mqtt.Client(mqtt.CallbackAPIVersion.VERSION2)
client.username_pw_set(MQTT_USER, MQTT_PASSWORD)
client.tls_set()
client.on_connect = on_connect
client.on_disconnect = on_disconnect
client.on_publish = on_publish

print(f"Connecting to {MQTT_BROKER}:{MQTT_PORT} as '{MQTT_USER}'...")
client.connect(MQTT_BROKER, MQTT_PORT, keepalive=60)
client.loop_start()

# Wait for on_connect to actually fire (CONNACK received) before
# publishing anything -- publishing too early is a real race condition
# that can silently drop the message even though the script appears
# to finish normally.
if not connected.wait(timeout=10):
    print("[ERROR] Never received CONNACK within 10s -- not publishing.")
    client.loop_stop()
    raise SystemExit(1)

body = json.dumps(PAYLOAD)
print(f"Publishing to {MQTT_TOPIC}:\n{body}")
info = client.publish(MQTT_TOPIC, body, qos=1)
info.wait_for_publish(timeout=10)
print(
    f"[RESULT] rc={info.rc} ({'success' if info.rc == mqtt.MQTT_ERR_SUCCESS else 'FAILED'}) is_published={info.is_published()}"
)

client.loop_stop()
client.disconnect()
print("Done.")
