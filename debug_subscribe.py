"""
Minimal standalone MQTT subscriber -- no FastAPI, no InfluxDB, no
processing. Connects, subscribes, prints whatever arrives. Run this
alone to see if anything is actually reaching the broker at all,
before worrying about anything downstream.

    python3 debug_subscribe.py
    (Ctrl+C to stop)
"""

import paho.mqtt.client as mqtt

MQTT_BROKER = "68d36c1becfe4592a74352c9d79b150b.s1.eu.hivemq.cloud"
MQTT_PORT = 8883
MQTT_TOPIC = "solar/#"
MQTT_USER = "cloud_subscriber"
MQTT_PASSWORD = "hivemqpassword123"


def on_connect(client, userdata, flags, rc, properties=None):
    print(f"[CONNECT] rc={rc} ({'OK' if rc == 0 else 'FAILED'})")
    if rc == 0:
        client.subscribe(MQTT_TOPIC)
        print(f"[SUBSCRIBE] {MQTT_TOPIC}")


def on_subscribe(client, userdata, mid, reason_codes, properties=None):
    print(f"[SUBACK] mid={mid} reason_codes={reason_codes}")


def on_message(client, userdata, msg):
    print(f"\n[MESSAGE] topic={msg.topic}")
    print(msg.payload.decode(errors="replace"))


def on_disconnect(client, userdata, flags, rc, properties=None):
    print(f"[DISCONNECT] rc={rc}")


client = mqtt.Client(mqtt.CallbackAPIVersion.VERSION2)
client.username_pw_set(MQTT_USER, MQTT_PASSWORD)
client.tls_set()
client.on_connect = on_connect
client.on_subscribe = on_subscribe
client.on_message = on_message
client.on_disconnect = on_disconnect

print(f"Connecting to {MQTT_BROKER}:{MQTT_PORT} as '{MQTT_USER}'...")
client.connect(MQTT_BROKER, MQTT_PORT, keepalive=60)

try:
    client.loop_forever()
except KeyboardInterrupt:
    print("\nStopped.")
