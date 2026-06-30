"""
Mock ESP32 publisher.

Simulates a solar installation publishing sensor readings to HiveMQ
every few seconds (instead of every 5 minutes) so you can test the
full pipeline quickly without waiting for real hardware.

Solar output follows a rough daylight curve based on the actual
clock hour, load has random household-like variation, and battery
behaves accordingly (charges when solar > load, discharges otherwise).
"""

import os
import json
import time
import math
import random
import paho.mqtt.client as mqtt
from datetime import datetime, timezone
from dotenv import load_dotenv

load_dotenv()

MQTT_BROKER = os.getenv("MQTT_BROKER")
MQTT_PORT = int(os.getenv("MQTT_PORT", 8883))
MQTT_USER = os.getenv("MQTT_PUBLISHER_USER")
MQTT_PASSWORD = os.getenv("MQTT_PUBLISHER_PASSWORD")
HOME_ID = os.getenv("HOME_ID", "home1")
TOPIC = f"solar/{HOME_ID}/data"

# simulation settings
PUBLISH_INTERVAL_SECONDS = 10  # how often to send (real ESP32 = 300s)
SIM_MINUTES_PER_TICK = 5  # each tick simulates 5 real-world minutes
START_BATTERY_VOLTAGE = 12.6  # starting near-full lead acid 12V battery

state = {
    "battery_voltage": START_BATTERY_VOLTAGE,
    "sim_time": datetime.now(timezone.utc),
}


def solar_irradiance_factor(hour: float) -> float:
    """
    Returns 0.0–1.0 representing how much sun is out at this hour.
    Peaks at noon, zero before 6am and after 6pm.
    """
    if hour < 6 or hour > 18:
        return 0.0
    # bell curve peaking at hour 12
    x = (hour - 12) / 6
    return max(0.0, math.cos(x * math.pi / 2))


def simulate_reading():
    state["sim_time"] += timedelta_minutes(SIM_MINUTES_PER_TICK)
    now = state["sim_time"]
    hour_float = now.hour + now.minute / 60

    # ── Solar side ──────────────────────────────────────────────
    sun_factor = solar_irradiance_factor(hour_float)
    cloud_noise = random.uniform(0.7, 1.0)  # simulate passing clouds
    solar_voltage = (
        round(17.0 + sun_factor * 2.5, 2)
        if sun_factor > 0
        else round(random.uniform(0, 0.5), 2)
    )
    solar_current = round(sun_factor * cloud_noise * 4.5, 2)  # up to ~4.5A peak

    # ── Load side ───────────────────────────────────────────────
    base_load = 2.0
    evening_boost = 2.5 if 18 <= hour_float <= 23 else 0.0
    load_current = round(base_load + evening_boost + random.uniform(-0.4, 0.4), 2)
    load_current = max(load_current, 0.3)

    # ── Battery side ────────────────────────────────────────────
    solar_power = solar_voltage * solar_current
    load_power = state["battery_voltage"] * load_current
    net_power = solar_power - load_power
    battery_current = (
        round(net_power / state["battery_voltage"], 2)
        if state["battery_voltage"] > 0
        else 0.0
    )

    # nudge battery voltage based on net current (simple simulation, not Coulomb-accurate)
    voltage_delta = (battery_current * (SIM_MINUTES_PER_TICK / 60)) * 0.05
    state["battery_voltage"] = round(
        min(max(state["battery_voltage"] + voltage_delta, 11.0), 12.9), 2
    )

    # ── Temperature ─────────────────────────────────────────────
    temperature = round(26 + sun_factor * 12 + random.uniform(-1, 1), 1)

    return {
        "home_id": HOME_ID,
        "recorded_at": now.isoformat(),
        "solar_voltage": solar_voltage,
        "solar_current": solar_current,
        "battery_voltage": state["battery_voltage"],
        "battery_current": battery_current,
        "load_current": load_current,
        "temperature": temperature,
    }


def timedelta_minutes(minutes):
    from datetime import timedelta

    return timedelta(minutes=minutes)


# ── MQTT setup ─────────────────────────────────────────────────────
client = mqtt.Client(mqtt.CallbackAPIVersion.VERSION2)
client.username_pw_set(MQTT_USER, MQTT_PASSWORD)
client.tls_set()

print(f"[MockESP32] Connecting to {MQTT_BROKER}...")
client.connect(MQTT_BROKER, MQTT_PORT, keepalive=60)
client.loop_start()

print(f"[MockESP32] Publishing to {TOPIC} every {PUBLISH_INTERVAL_SECONDS}s")
print(f"[MockESP32] Each tick simulates {SIM_MINUTES_PER_TICK} real-world minutes")
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
