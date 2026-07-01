"""
Mock ESP32 publisher — optional test tool, independent of the main pipeline.
Run this to simulate a solar installation sending sensor data to HiveMQ.
The main pipeline (api.py + mqtt_to_influx.py) does not depend on this file.
"""

import os
import json
import time
import math
import random
import paho.mqtt.client as mqtt
from datetime import datetime, timedelta, timezone
from dotenv import load_dotenv

load_dotenv()

MQTT_BROKER = os.getenv("MQTT_BROKER")
MQTT_PORT   = int(os.getenv("MQTT_PORT", 8883))
MQTT_USER   = os.getenv("MQTT_PUBLISHER_USER")
MQTT_PASSWORD = os.getenv("MQTT_PUBLISHER_PASSWORD")
HOME_ID     = os.getenv("HOME_ID", "home1")
TOPIC       = f"solar/{HOME_ID}/data"

PUBLISH_INTERVAL_SECONDS = 10
SIM_MINUTES_PER_TICK     = 5

state = {
    "battery_voltage": 12.6,
    "sim_time":        datetime.now(timezone.utc),
}


def solar_factor(hour: float) -> float:
    if hour < 6 or hour > 18:
        return 0.0
    x = (hour - 12) / 6
    return max(0.0, math.cos(x * math.pi / 2))


def simulate_reading() -> dict:
    state["sim_time"] += timedelta(minutes=SIM_MINUTES_PER_TICK)
    now         = state["sim_time"]
    hour_float  = now.hour + now.minute / 60
    sun         = solar_factor(hour_float)
    cloud_noise = random.uniform(0.7, 1.0)

    solar_voltage = round(17.0 + sun * 2.5, 2) if sun > 0 else round(random.uniform(0, 0.3), 2)
    solar_current = round(sun * cloud_noise * 4.5, 2)

    base_load     = 2.0
    evening       = 2.5 if 18 <= hour_float <= 23 else 0.0
    load_current  = round(max(base_load + evening + random.uniform(-0.4, 0.4), 0.3), 2)

    solar_power   = solar_voltage * solar_current
    load_power    = state["battery_voltage"] * load_current
    net_power     = solar_power - load_power
    batt_current  = round(net_power / state["battery_voltage"], 2) if state["battery_voltage"] > 0 else 0.0

    voltage_delta = (batt_current * (SIM_MINUTES_PER_TICK / 60)) * 0.05
    state["battery_voltage"] = round(min(max(state["battery_voltage"] + voltage_delta, 11.0), 12.9), 2)

    temperature = round(26 + sun * 12 + random.uniform(-1, 1), 1)

    return {
        "home_id":         HOME_ID,
        "recorded_at":     now.isoformat(),
        "solar_voltage":   solar_voltage,
        "solar_current":   solar_current,
        "battery_voltage": state["battery_voltage"],
        "battery_current": batt_current,
        "load_current":    load_current,
        "temperature":     temperature,
    }


client = mqtt.Client(mqtt.CallbackAPIVersion.VERSION2)
client.username_pw_set(MQTT_USER, MQTT_PASSWORD)
client.tls_set()

print(f"[MockESP32] Connecting to {MQTT_BROKER}...")
client.connect(MQTT_BROKER, MQTT_PORT, keepalive=60)
client.loop_start()

print(f"[MockESP32] Publishing to {TOPIC} every {PUBLISH_INTERVAL_SECONDS}s")
print(f"[MockESP32] Each tick = {SIM_MINUTES_PER_TICK} real-world minutes")
print("[MockESP32] Press Ctrl+C to stop\n")

try:
    while True:
        reading = simulate_reading()
        client.publish(TOPIC, json.dumps(reading))
        print(f"[OUT] {reading}")
        time.sleep(PUBLISH_INTERVAL_SECONDS)
except KeyboardInterrupt:
    print("\n[MockESP32] Stopped.")
    client.loop_stop()
    client.disconnect()
