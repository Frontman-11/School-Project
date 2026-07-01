"""
Scheduler — runs hourly, daily, and monthly forecast jobs
by calling the FastAPI internal trigger endpoints.
Reads all registered homes from the registry and runs
forecasts for each one.
"""

import os
import httpx
from dotenv import load_dotenv
from apscheduler.schedulers.blocking import BlockingScheduler
from utils.home_registry import list_homes

load_dotenv()

API_BASE = f"http://localhost:{os.getenv('API_PORT', 8000)}"
API_KEY  = os.getenv("API_KEY")
HEADERS  = {"X-API-Key": API_KEY}

scheduler = BlockingScheduler(timezone="UTC")


def _call(endpoint: str, home_id: str):
    try:
        resp = httpx.post(f"{API_BASE}{endpoint}/{home_id}", headers=HEADERS, timeout=30)
        print(f"[Scheduler] {endpoint}/{home_id} → {resp.status_code}: {resp.json()}")
    except Exception as e:
        print(f"[Scheduler] {endpoint}/{home_id} failed: {e}")


@scheduler.scheduled_job("interval", hours=1, id="hourly_all")
def hourly_job():
    homes = list_homes()
    print(f"\n[Scheduler] Hourly forecast for {len(homes)} home(s): {homes}")
    for home_id in homes:
        _call("/internal/run-hourly", home_id)


@scheduler.scheduled_job("cron", hour=0, minute=5, id="daily_all")
def daily_job():
    homes = list_homes()
    print(f"\n[Scheduler] Daily forecast for {len(homes)} home(s): {homes}")
    for home_id in homes:
        _call("/internal/run-daily", home_id)


@scheduler.scheduled_job("cron", day=1, hour=1, minute=0, id="monthly_all")
def monthly_job():
    homes = list_homes()
    print(f"\n[Scheduler] Monthly forecast for {len(homes)} home(s): {homes}")
    for home_id in homes:
        _call("/internal/run-monthly", home_id)


if __name__ == "__main__":
    print("[Scheduler] Starting")
    print("  - Hourly  : every hour")
    print("  - Daily   : midnight 00:05 UTC")
    print("  - Monthly : 1st of month 01:00 UTC")
    scheduler.start()
