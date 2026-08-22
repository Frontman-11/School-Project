"""
Live mock ESP publisher -- posts real-time readings straight to the
running API's /ingest endpoint, using the actual current time for every
reading (no backdating). Every reading's request and response is also
appended to a local JSON-lines log file for offline analysis.

Run the SAME script twice, in two terminals, with different --device-id
and --log-file values, to simulate two independent homes at once:

    python3 seed_sample_home.py --device-id sample_home   --log-file sample_home_log.jsonl
    python3 seed_sample_home.py --device-id sample_home_2 --log-file sample_home_2_log.jsonl

Each process registers its own device, streams its own physics-driven
readings, and logs independently -- they do not interact with each
other at all, exactly like two real, separate installations would.
"""

import argparse
import json
import math
import os
import random
import time
from datetime import datetime, timezone

import httpx
from dotenv import load_dotenv

load_dotenv()

API_BASE = f"http://localhost:{os.getenv('API_PORT', 8000)}"
API_KEY = os.getenv("API_KEY")
HEADERS = {"X-API-Key": API_KEY, "Content-Type": "application/json"}

LAT, LON = 5.5167, 5.7500  # Warri, Delta State area
PUBLISH_INTERVAL_SECONDS = 10

# For visibility during testing, trigger the hourly/daily forecast jobs
# every so many readings instead of waiting for the real scheduler to
# hit its actual hourly/midnight cadence.
TRIGGER_HOURLY_EVERY_N_READINGS = 12  # roughly every 2 minutes at a 10s interval
TRIGGER_DAILY_EVERY_N_READINGS = 60  # roughly every 10 minutes at a 10s interval

STATE = {"battery_voltage": 12.6}


def solar_factor(hour: float) -> float:
    if hour < 6 or hour > 18:
        return 0.0
    x = (hour - 12) / 6
    return max(0.0, math.cos(x * math.pi / 2))


def build_reading(device_id: str) -> dict:
    now = datetime.now(timezone.utc)
    hour_float = now.hour + now.minute / 60
    sun = solar_factor(hour_float)
    cloud_noise = random.uniform(0.7, 1.0)

    solar_voltage = (
        round(17.0 + sun * 2.5, 2) if sun > 0 else round(random.uniform(0, 0.3), 2)
    )
    solar_current = round(sun * cloud_noise * 4.5, 2)

    base_load = 2.0
    evening = 2.5 if 18 <= hour_float <= 23 else 0.0
    load_current = round(max(base_load + evening + random.uniform(-0.4, 0.4), 0.3), 2)

    solar_power = solar_voltage * solar_current
    load_power = STATE["battery_voltage"] * load_current
    net_power = solar_power - load_power
    battery_current_signed = (
        net_power / STATE["battery_voltage"] if STATE["battery_voltage"] > 0 else 0.0
    )
    charging = battery_current_signed >= 0

    voltage_delta = (battery_current_signed * (PUBLISH_INTERVAL_SECONDS / 3600)) * 0.05
    STATE["battery_voltage"] = round(
        min(max(STATE["battery_voltage"] + voltage_delta, 11.0), 12.9), 2
    )

    temperature_c = round(26 + sun * 12 + random.uniform(-1, 1), 1)

    return {
        "device_id": device_id,
        "timestamp": now.isoformat().replace("+00:00", "Z"),
        "solar": {
            "voltage_v": solar_voltage,
            "current_a": solar_current,
            "power_w": round(solar_power, 2),
            "shunt_mv": round(solar_current * 75 / 30, 2),
        },
        "battery": {
            "voltage_v": STATE["battery_voltage"],
            "current_a": round(abs(battery_current_signed), 2),
            "power_w": round(abs(battery_current_signed) * STATE["battery_voltage"], 2),
            "shunt_mv": round(abs(battery_current_signed) * 75 / 50, 2),
            "charging": charging,
        },
        "load": {"current_a": load_current},
        "temperature_c": temperature_c,
        "interval_s": PUBLISH_INTERVAL_SECONDS,
    }


def register_device(device_id: str):
    resp = httpx.post(
        f"{API_BASE}/homes/register",
        json={
            "home_id": device_id,
            "lat": LAT,
            "lon": LON,
            "battery_type": "LEAD_ACID",
            "nominal_voltage": "12V",
            "battery_capacity_wh": 100,
        },
        headers=HEADERS,
        timeout=15,
    )
    print(f"[Seed] Registered {device_id}: {resp.status_code} {resp.json()}")


def append_log(log_file: str, entry: dict):
    with open(log_file, "a") as f:
        f.write(json.dumps(entry) + "\n")


def main(device_id: str, log_file: str, count: int | None):
    register_device(device_id)
    print(
        f"[Seed] Streaming live readings for '{device_id}' every {PUBLISH_INTERVAL_SECONDS}s"
    )
    print(f"[Seed] Logging each request/response to {log_file}")
    print("[Seed] Press Ctrl+C to stop\n")

    i = 0
    try:
        while count is None or i < count:
            reading = build_reading(device_id)
            body = None
            status_code = None
            try:
                resp = httpx.post(
                    f"{API_BASE}/ingest", json=reading, headers=HEADERS, timeout=15
                )
                status_code = resp.status_code
                body = resp.json()
                drift = body.get("drift")
                drift_str = ""
                if drift:
                    drift_str = (
                        f" | drift: solar {drift['solar_error_w']:+.1f}W "
                        f"({drift['solar_abs_pct_error']}%), "
                        f"load {drift['load_error_w']:+.1f}VA ({drift['load_abs_pct_error']}%)"
                    )
                print(
                    f"[Seed] {reading['timestamp']} -> {status_code} ml_status={body.get('ml_status')}{drift_str}"
                )
            except Exception as e:
                print(f"[Seed] ingest failed: {e}")

            append_log(
                log_file,
                {
                    "logged_at": datetime.now(timezone.utc).isoformat(),
                    "device_id": device_id,
                    "http_status": status_code,
                    "request": reading,
                    "response": body,
                },
            )

            i += 1

            if i % TRIGGER_HOURLY_EVERY_N_READINGS == 0:
                try:
                    r = httpx.post(
                        f"{API_BASE}/internal/run-hourly/{device_id}",
                        headers=HEADERS,
                        timeout=15,
                    )
                    print("[Seed] triggered hourly forecast")
                    append_log(
                        log_file,
                        {
                            "logged_at": datetime.now(timezone.utc).isoformat(),
                            "device_id": device_id,
                            "event": "hourly_forecast_triggered",
                            "response": r.json(),
                        },
                    )
                except Exception as e:
                    print(f"[Seed] hourly trigger failed: {e}")

            if i % TRIGGER_DAILY_EVERY_N_READINGS == 0:
                try:
                    r = httpx.post(
                        f"{API_BASE}/internal/run-daily/{device_id}",
                        headers=HEADERS,
                        timeout=15,
                    )
                    print("[Seed] triggered daily forecast")
                    append_log(
                        log_file,
                        {
                            "logged_at": datetime.now(timezone.utc).isoformat(),
                            "device_id": device_id,
                            "event": "daily_forecast_triggered",
                            "response": r.json(),
                        },
                    )
                except Exception as e:
                    print(f"[Seed] daily trigger failed: {e}")

            time.sleep(PUBLISH_INTERVAL_SECONDS)
    except KeyboardInterrupt:
        print("\n[Seed] Stopped.")

    print(f"[Seed] Sent {i} readings for '{device_id}'.")
    print(f"[Seed] Log file: {log_file}")
    print(f"[Seed] Try: GET {API_BASE}/current/{device_id}")
    print(f"[Seed] Try: GET {API_BASE}/forecast/hourly/{device_id}")
    print(f"[Seed] Try: GET {API_BASE}/forecast/daily/{device_id}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--device-id",
        type=str,
        default="sample_home",
        help="device_id / home_id to simulate",
    )
    parser.add_argument(
        "--log-file",
        type=str,
        default=None,
        help="JSONL file to log every request/response to (default: <device-id>_log.jsonl)",
    )
    parser.add_argument(
        "--count",
        type=int,
        default=None,
        help="number of readings to send (default: run until Ctrl+C)",
    )
    args = parser.parse_args()

    log_file = args.log_file or f"{args.device_id}_log.jsonl"
    main(args.device_id, log_file, args.count)
