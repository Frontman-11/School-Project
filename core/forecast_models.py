"""
Hourly and daily forecast models, per home, InfluxDB-backed like the
5-minute models. The monthly horizon was dropped -- it added a third
pair of models and a third pipeline_state to keep in sync for a
forecast the app was not using.
"""

from datetime import datetime, timezone, timedelta
from utils.constants import (
    encode_hour,
    encode_day,
    encode_month,
    MAX_PLAUSIBLE_SOLAR_W,
    MAX_PLAUSIBLE_LOAD_VA,
)
from core import model_store
from db.influx_client import load_pipeline_state, save_pipeline_state


# ── Feature builders ──────────────────────────────────────────────

def _hourly_features(agg: dict, temp_c: float, now: datetime) -> dict:
    f = {
        "mean_solar_w":   agg.get("solar_power_now_w", 0.0),
        "mean_load_w":    agg.get("load_power_now_w",  0.0),
        "mean_soc_pct":   agg.get("soc_now_percent",  50.0),
        "mean_cloud_pct": agg.get("cloud_cover_pct",  50.0),
        "mean_temp_c":    temp_c,
    }
    f.update(encode_hour(now.hour))
    f.update(encode_day(now.weekday()))
    f.update(encode_month(now.month))
    return f


def _daily_features(agg: dict, temp_c: float, now: datetime) -> dict:
    f = {
        "total_solar_wh": agg.get("solar_power_now_w", 0.0) * 24,
        "mean_load_w":    agg.get("load_power_now_w",  0.0),
        "mean_cloud_pct": agg.get("cloud_cover_pct",  50.0),
        "mean_temp_c":    temp_c,
    }
    f.update(encode_day(now.weekday()))
    f.update(encode_month(now.month))
    return f


# ── Hourly forecast ───────────────────────────────────────────────

def run_hourly_forecast(home_id: str, agg: dict, temp_c: float) -> dict:
    now = datetime.now(timezone.utc)
    features = _hourly_features(agg, temp_c, now)
    last = load_pipeline_state(home_id, "hourly")
    models = model_store.get_hourly_models(home_id)

    if last:
        models["solar_hourly"].learn_one(last["features"], agg.get("solar_power_now_w", 0.0))
        models["load_hourly"].learn_one(last["features"], agg.get("load_power_now_w", 0.0))
        model_store.save_models(home_id, models)

    try:
        solar_h = models["solar_hourly"].predict_one(features)
        load_h = models["load_hourly"].predict_one(features)
    except Exception:
        solar_h = agg.get("solar_power_now_w", 0.0)
        load_h = agg.get("load_power_now_w", 0.0)

    if solar_h is None or (solar_h <= 0 and agg.get("solar_power_now_w", 0) > 0):
        solar_h = agg.get("solar_power_now_w", 0.0)
    if load_h is None or (load_h <= 0 and agg.get("load_power_now_w", 0) > 0):
        load_h = agg.get("load_power_now_w", 0.0)

    # Upper bounds. Without these an upward divergence is stored and
    # served: this model wrote 6.87e13 W on 26 August 2026 and the value
    # reached the app unchallenged.
    if solar_h > MAX_PLAUSIBLE_SOLAR_W:
        solar_h = agg.get("solar_power_now_w", 0.0)
    if load_h > MAX_PLAUSIBLE_LOAD_VA:
        load_h = agg.get("load_power_now_w", 0.0)

    save_pipeline_state(home_id, "hourly", {"features": features, "recorded_at": now.isoformat()})

    return {
        "forecast_for":   (now + timedelta(hours=1)).isoformat(),
        "solar_next_h_w": round(max(solar_h, 0), 2),
        "load_next_h_w":  round(max(load_h, 0), 2),
    }


# ── Daily forecast ────────────────────────────────────────────────

def run_daily_forecast(home_id: str, agg: dict, temp_c: float) -> dict:
    now = datetime.now(timezone.utc)
    features = _daily_features(agg, temp_c, now)
    last = load_pipeline_state(home_id, "daily")
    models = model_store.get_daily_models(home_id)

    if last:
        models["solar_daily"].learn_one(last["features"], agg.get("solar_power_now_w", 0.0) * 24)
        models["load_daily"].learn_one(last["features"], agg.get("load_power_now_w", 0.0))
        model_store.save_models(home_id, models)

    try:
        solar_d = models["solar_daily"].predict_one(features)
        load_d = models["load_daily"].predict_one(features)
    except Exception:
        solar_d = features["total_solar_wh"]
        load_d = features["mean_load_w"]

    if solar_d is None:
        solar_d = features["total_solar_wh"]
    if load_d is None:
        load_d = features["mean_load_w"]

    # The daily solar figure is an energy over 24 h rather than a power,
    # so its bound is scaled accordingly. The load figure remains a mean
    # power and uses the bound unscaled.
    if solar_d > MAX_PLAUSIBLE_SOLAR_W * 24:
        solar_d = features["total_solar_wh"]
    if load_d > MAX_PLAUSIBLE_LOAD_VA:
        load_d = features["mean_load_w"]

    tomorrow = (now + timedelta(days=1)).replace(hour=0, minute=0, second=0, microsecond=0)
    save_pipeline_state(home_id, "daily", {"features": features, "recorded_at": now.isoformat()})

    return {
        "forecast_for":      tomorrow.isoformat(),
        "solar_tomorrow_wh": round(max(solar_d, 0), 2),
        "load_tomorrow_w":   round(max(load_d, 0), 2),
    }
