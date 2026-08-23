"""
Pings the deployed Render app's /health endpoint on a loop, purely to
keep Render's free-tier web service from spinning down after 15
minutes of no inbound HTTP traffic.

Why this matters: MQTT listening and the hourly/daily scheduler run
as background threads INSIDE the same process as the API. If Render
spins that process down, those threads die too -- any ESP32 readings
published to HiveMQ during that sleep window are lost, since nothing
is running to receive them. An inbound HTTP request (like this ping)
is the only thing that resets Render's inactivity clock; outbound
activity (MQTT, InfluxDB, weather calls) does not count.

The ping interval must be shorter than Render's 15-minute spin-down
window, with enough safety margin to survive a slow network hiccup --
10 minutes is used here.

Usage:
    python3 keep_alive.py
    (Ctrl+C to stop)

Requires RENDER_HEALTH_URL in .env, e.g.:
    RENDER_HEALTH_URL=https://your-app-name.onrender.com/health
"""

import os
import time
from datetime import datetime, timezone

import httpx
from dotenv import load_dotenv

load_dotenv()

HEALTH_URL = os.getenv("RENDER_HEALTH_URL")
PING_INTERVAL_SECONDS = int(os.getenv("KEEP_ALIVE_INTERVAL_SECONDS", 600))  # 10 min


def ping_once() -> bool:
    try:
        resp = httpx.get(HEALTH_URL, timeout=30)
        ok = resp.status_code == 200
        status = "ok" if ok else f"unexpected status {resp.status_code}"
        print(f"[KeepAlive] {datetime.now(timezone.utc).isoformat()} -> {status}")
        return ok
    except Exception as e:
        print(f"[KeepAlive] {datetime.now(timezone.utc).isoformat()} -> ping failed: {e}")
        return False


def main():
    if not HEALTH_URL:
        print("[KeepAlive] RENDER_HEALTH_URL is not set in .env. Add it and try again.")
        print('[KeepAlive] Example: RENDER_HEALTH_URL=https://your-app-name.onrender.com/health')
        return

    print(f"[KeepAlive] Pinging {HEALTH_URL} every {PING_INTERVAL_SECONDS}s")
    print("[KeepAlive] Press Ctrl+C to stop\n")

    try:
        while True:
            ping_once()
            time.sleep(PING_INTERVAL_SECONDS)
    except KeyboardInterrupt:
        print("\n[KeepAlive] Stopped.")


if __name__ == "__main__":
    main()
