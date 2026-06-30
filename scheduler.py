import os
from dotenv import load_dotenv
from apscheduler.schedulers.blocking import BlockingScheduler
from influx_reader import get_aggregate, get_temperature_mean, write_forecast
from forecast_models import run_hourly_forecast, run_daily_forecast, run_monthly_forecast

load_dotenv()

HOME_ID = os.getenv("HOME_ID", "home1")

scheduler = BlockingScheduler(timezone="UTC")


@scheduler.scheduled_job("interval", hours=1, id="hourly_forecast")
def hourly_job():
    print(f"\n[Scheduler] Running hourly forecast for {HOME_ID}")
    try:
        agg    = get_aggregate(HOME_ID, "-1h")
        temp_c = get_temperature_mean(HOME_ID, "-1h")
        if not agg:
            print("[Scheduler] Not enough data for hourly forecast. Skipping.")
            return
        result = run_hourly_forecast(HOME_ID, agg, temp_c)
        write_forecast("hourly_forecast", HOME_ID, result, result["forecast_for"])
        print(f"[Scheduler] Hourly forecast written: {result}")
    except Exception as e:
        print(f"[Scheduler] Hourly job error: {e}")


@scheduler.scheduled_job("cron", hour=0, minute=5, id="daily_forecast")
def daily_job():
    print(f"\n[Scheduler] Running daily forecast for {HOME_ID}")
    try:
        agg    = get_aggregate(HOME_ID, "-24h")
        temp_c = get_temperature_mean(HOME_ID, "-24h")
        if not agg:
            print("[Scheduler] Not enough data for daily forecast. Skipping.")
            return
        result = run_daily_forecast(HOME_ID, agg, temp_c)
        write_forecast("daily_forecast", HOME_ID, result, result["forecast_for"])
        print(f"[Scheduler] Daily forecast written: {result}")
    except Exception as e:
        print(f"[Scheduler] Daily job error: {e}")


@scheduler.scheduled_job("cron", day=1, hour=1, minute=0, id="monthly_forecast")
def monthly_job():
    print(f"\n[Scheduler] Running monthly forecast for {HOME_ID}")
    try:
        agg    = get_aggregate(HOME_ID, "-30d")
        temp_c = get_temperature_mean(HOME_ID, "-30d")
        if not agg:
            print("[Scheduler] Not enough data for monthly forecast. Skipping.")
            return
        result = run_monthly_forecast(HOME_ID, agg, temp_c)
        write_forecast("monthly_forecast", HOME_ID, result, result["forecast_for"])
        print(f"[Scheduler] Monthly forecast written: {result}")
    except Exception as e:
        print(f"[Scheduler] Monthly job error: {e}")


if __name__ == "__main__":
    print(f"[Scheduler] Starting for home: {HOME_ID}")
    print("[Scheduler] Jobs scheduled:")
    print("  - Hourly forecast : every hour")
    print("  - Daily forecast  : every midnight at 00:05 UTC")
    print("  - Monthly forecast: 1st of each month at 01:00 UTC")
    scheduler.start()
