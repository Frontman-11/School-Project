import json
import os
import pickle
from datetime import datetime, timezone, timedelta
from river import ensemble, preprocessing, compose
from utils.constants import encode_hour, encode_day, encode_month

MODEL_DIR = "model_state"
os.makedirs(MODEL_DIR, exist_ok=True)

# ── Model paths ───────────────────────────────────────────────────
HOURLY_SOLAR_PATH  = f"{MODEL_DIR}/hourly_solar_model.pkl"
HOURLY_LOAD_PATH   = f"{MODEL_DIR}/hourly_load_model.pkl"
DAILY_SOLAR_PATH   = f"{MODEL_DIR}/daily_solar_model.pkl"
DAILY_LOAD_PATH    = f"{MODEL_DIR}/daily_load_model.pkl"
MONTHLY_SOLAR_PATH = f"{MODEL_DIR}/monthly_solar_model.pkl"
MONTHLY_LOAD_PATH  = f"{MODEL_DIR}/monthly_load_model.pkl"

# ── State paths ───────────────────────────────────────────────────
HOURLY_STATE_PATH  = f"{MODEL_DIR}/last_hourly_state.json"
DAILY_STATE_PATH   = f"{MODEL_DIR}/last_daily_state.json"
MONTHLY_STATE_PATH = f"{MODEL_DIR}/last_monthly_state.json"


# ── Model helpers ─────────────────────────────────────────────────
def make_model():
    return compose.Pipeline(
        preprocessing.StandardScaler(),
        ensemble.SRPRegressor(seed=42)
    )


def load_model(path):
    if os.path.exists(path):
        with open(path, "rb") as f:
            return pickle.load(f)
    return make_model()


def save_model(model, path):
    with open(path, "wb") as f:
        pickle.dump(model, f)


def load_state(path):
    if os.path.exists(path):
        with open(path, "r") as f:
            return json.load(f)
    return None


def save_state(path, state):
    with open(path, "w") as f:
        json.dump(state, f)


# ── Load all models at startup ────────────────────────────────────
hourly_solar_model  = load_model(HOURLY_SOLAR_PATH)
hourly_load_model   = load_model(HOURLY_LOAD_PATH)
daily_solar_model   = load_model(DAILY_SOLAR_PATH)
daily_load_model    = load_model(DAILY_LOAD_PATH)
monthly_solar_model = load_model(MONTHLY_SOLAR_PATH)
monthly_load_model  = load_model(MONTHLY_LOAD_PATH)


# ── Feature builders ──────────────────────────────────────────────
def _build_hourly_features(agg: dict, temp_c: float, now: datetime) -> dict:
    """
    Features for the 1-hour ahead model.
    Uses mean values from the current hour as context.
    """
    features = {
        "mean_solar_w":   agg.get("solar_power_now_w", 0.0),
        "mean_load_w":    agg.get("load_power_now_w",  0.0),
        "mean_soc_pct":   agg.get("soc_now_percent",  50.0),
        "mean_cloud_pct": agg.get("cloud_cover_pct",  50.0),
        "mean_temp_c":    temp_c,
    }
    features.update(encode_hour(now.hour))
    features.update(encode_day(now.weekday()))
    features.update(encode_month(now.month))
    return features


def _build_daily_features(agg: dict, temp_c: float, now: datetime) -> dict:
    """
    Features for the daily (tomorrow) model.
    Uses mean W × 24h to estimate today's total solar energy (Wh).
    """
    mean_solar_w = agg.get("solar_power_now_w", 0.0)
    features = {
        "total_solar_wh": mean_solar_w * 24,
        "mean_load_w":    agg.get("load_power_now_w", 0.0),
        "mean_cloud_pct": agg.get("cloud_cover_pct",  50.0),
        "mean_temp_c":    temp_c,
    }
    features.update(encode_day(now.weekday()))
    features.update(encode_month(now.month))
    return features


def _build_monthly_features(agg: dict, temp_c: float, now: datetime) -> dict:
    """
    Features for the monthly model.
    Uses mean W × 24h × 30d to estimate this month's total solar energy (Wh).
    """
    mean_solar_w = agg.get("solar_power_now_w", 0.0)
    features = {
        "total_solar_wh": mean_solar_w * 24 * 30,
        "mean_load_w":    agg.get("load_power_now_w", 0.0),
        "mean_cloud_pct": agg.get("cloud_cover_pct",  50.0),
        "mean_temp_c":    temp_c,
    }
    features.update(encode_month(now.month))
    return features


# ── Hourly forecast ───────────────────────────────────────────────
def run_hourly_forecast(home_id: str, agg: dict, temp_c: float) -> dict:
    """
    Called every hour by the scheduler.
    Trains on last hour's features vs current actuals,
    then predicts next hour's solar and load.
    """
    global hourly_solar_model, hourly_load_model

    now      = datetime.now(timezone.utc)
    features = _build_hourly_features(agg, temp_c, now)
    last     = load_state(HOURLY_STATE_PATH)

    # train on last hour's features → current actuals
    if last:
        actual_solar = agg.get("solar_power_now_w", 0.0)
        actual_load  = agg.get("load_power_now_w",  0.0)
        hourly_solar_model.learn_one(last["features"], actual_solar)
        hourly_load_model.learn_one(last["features"],  actual_load)
        save_model(hourly_solar_model, HOURLY_SOLAR_PATH)
        save_model(hourly_load_model,  HOURLY_LOAD_PATH)

    # predict next hour
    try:
        solar_next_h = hourly_solar_model.predict_one(features)
        load_next_h  = hourly_load_model.predict_one(features)
    except Exception:
        solar_next_h = agg.get("solar_power_now_w", 0.0)
        load_next_h  = agg.get("load_power_now_w",  0.0)

    # sanity check
    if solar_next_h <= 0 and agg.get("solar_power_now_w", 0) > 0:
        solar_next_h = agg.get("solar_power_now_w", 0.0)
    if load_next_h <= 0 and agg.get("load_power_now_w", 0) > 0:
        load_next_h = agg.get("load_power_now_w", 0.0)

    solar_next_h = max(solar_next_h, 0)
    load_next_h  = max(load_next_h,  0)
    forecast_for = (now + timedelta(hours=1)).isoformat()

    save_state(HOURLY_STATE_PATH, {
        "features":    features,
        "recorded_at": now.isoformat(),
    })

    return {
        "forecast_for":   forecast_for,
        "solar_next_h_w": round(solar_next_h, 2),
        "load_next_h_w":  round(load_next_h,  2),
    }


# ── Daily forecast ────────────────────────────────────────────────
def run_daily_forecast(home_id: str, agg: dict, temp_c: float) -> dict:
    """
    Called at midnight by the scheduler.
    Trains on yesterday's features → today's actuals,
    then predicts tomorrow's solar energy (Wh) and mean load (W).
    """
    global daily_solar_model, daily_load_model

    now      = datetime.now(timezone.utc)
    features = _build_daily_features(agg, temp_c, now)
    last     = load_state(DAILY_STATE_PATH)

    if last:
        actual_solar_wh = agg.get("solar_power_now_w", 0.0) * 24
        actual_load_w   = agg.get("load_power_now_w",  0.0)
        daily_solar_model.learn_one(last["features"], actual_solar_wh)
        daily_load_model.learn_one(last["features"],  actual_load_w)
        save_model(daily_solar_model, DAILY_SOLAR_PATH)
        save_model(daily_load_model,  DAILY_LOAD_PATH)

    try:
        solar_tomorrow_wh = daily_solar_model.predict_one(features)
        load_tomorrow_w   = daily_load_model.predict_one(features)
    except Exception:
        solar_tomorrow_wh = features["total_solar_wh"]
        load_tomorrow_w   = features["mean_load_w"]

    solar_tomorrow_wh = max(solar_tomorrow_wh, 0)
    load_tomorrow_w   = max(load_tomorrow_w,   0)

    tomorrow     = now + timedelta(days=1)
    forecast_for = tomorrow.replace(
        hour=0, minute=0, second=0, microsecond=0
    ).isoformat()

    save_state(DAILY_STATE_PATH, {
        "features":    features,
        "recorded_at": now.isoformat(),
    })

    return {
        "forecast_for":      forecast_for,
        "solar_tomorrow_wh": round(solar_tomorrow_wh, 2),
        "load_tomorrow_w":   round(load_tomorrow_w,   2),
    }


# ── Monthly forecast ──────────────────────────────────────────────
def run_monthly_forecast(home_id: str, agg: dict, temp_c: float) -> dict:
    """
    Called on the 1st of each month by the scheduler.
    Trains on last month's features → this month's actuals,
    then predicts next month's solar energy (Wh) and mean load (W).
    """
    global monthly_solar_model, monthly_load_model

    now      = datetime.now(timezone.utc)
    features = _build_monthly_features(agg, temp_c, now)
    last     = load_state(MONTHLY_STATE_PATH)

    if last:
        actual_solar_wh = agg.get("solar_power_now_w", 0.0) * 24 * 30
        actual_load_w   = agg.get("load_power_now_w",  0.0)
        monthly_solar_model.learn_one(last["features"], actual_solar_wh)
        monthly_load_model.learn_one(last["features"],  actual_load_w)
        save_model(monthly_solar_model, MONTHLY_SOLAR_PATH)
        save_model(monthly_load_model,  MONTHLY_LOAD_PATH)

    try:
        solar_next_month_wh = monthly_solar_model.predict_one(features)
        load_next_month_w   = monthly_load_model.predict_one(features)
    except Exception:
        solar_next_month_wh = features["total_solar_wh"]
        load_next_month_w   = features["mean_load_w"]

    solar_next_month_wh = max(solar_next_month_wh, 0)
    load_next_month_w   = max(load_next_month_w,   0)

    # first day of next month
    if now.month == 12:
        next_month = now.replace(
            year=now.year + 1, month=1, day=1,
            hour=0, minute=0, second=0, microsecond=0
        )
    else:
        next_month = now.replace(
            month=now.month + 1, day=1,
            hour=0, minute=0, second=0, microsecond=0
        )

    save_state(MONTHLY_STATE_PATH, {
        "features":    features,
        "recorded_at": now.isoformat(),
    })

    return {
        "forecast_for":        next_month.isoformat(),
        "solar_next_month_wh": round(solar_next_month_wh, 2),
        "load_next_month_w":   round(load_next_month_w,   2),
    }
