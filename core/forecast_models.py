import json
import os
import pickle
from datetime import datetime, timezone, timedelta
from river import ensemble, preprocessing, compose
from utils.constants import encode_hour, encode_day, encode_month

MODEL_DIR = "model_state"


def _home_dir(home_id: str) -> str:
    path = f"{MODEL_DIR}/{home_id}"
    os.makedirs(path, exist_ok=True)
    return path


def _make_model():
    return compose.Pipeline(
        preprocessing.StandardScaler(),
        ensemble.SRPRegressor(seed=42)
    )


def _load_model(path: str):
    if os.path.exists(path):
        with open(path, "rb") as f:
            return pickle.load(f)
    return _make_model()


def _save_model(model, path: str):
    with open(path, "wb") as f:
        pickle.dump(model, f)


def _load_state(path: str):
    if os.path.exists(path):
        with open(path, "r") as f:
            return json.load(f)
    return None


def _save_state(path: str, state: dict):
    with open(path, "w") as f:
        json.dump(state, f)


# In-memory forecast model cache keyed by home_id
_cache: dict = {}


def _get(home_id: str) -> dict:
    if home_id not in _cache:
        d = _home_dir(home_id)
        _cache[home_id] = {
            "hourly_solar":  _load_model(f"{d}/hourly_solar_model.pkl"),
            "hourly_load":   _load_model(f"{d}/hourly_load_model.pkl"),
            "daily_solar":   _load_model(f"{d}/daily_solar_model.pkl"),
            "daily_load":    _load_model(f"{d}/daily_load_model.pkl"),
            "monthly_solar": _load_model(f"{d}/monthly_solar_model.pkl"),
            "monthly_load":  _load_model(f"{d}/monthly_load_model.pkl"),
        }
    return _cache[home_id]


def _save_pair(home_id: str, key_solar: str, key_load: str):
    d = _home_dir(home_id)
    m = _get(home_id)
    _save_model(m[key_solar], f"{d}/{key_solar}_model.pkl")
    _save_model(m[key_load],  f"{d}/{key_load}_model.pkl")


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


def _monthly_features(agg: dict, temp_c: float, now: datetime) -> dict:
    f = {
        "total_solar_wh": agg.get("solar_power_now_w", 0.0) * 24 * 30,
        "mean_load_w":    agg.get("load_power_now_w",  0.0),
        "mean_cloud_pct": agg.get("cloud_cover_pct",  50.0),
        "mean_temp_c":    temp_c,
    }
    f.update(encode_month(now.month))
    return f


# ── Hourly forecast ───────────────────────────────────────────────

def run_hourly_forecast(home_id: str, agg: dict, temp_c: float) -> dict:
    now      = datetime.now(timezone.utc)
    features = _hourly_features(agg, temp_c, now)
    state_p  = f"{_home_dir(home_id)}/last_hourly_state.json"
    last     = _load_state(state_p)
    m        = _get(home_id)

    if last:
        m["hourly_solar"].learn_one(last["features"], agg.get("solar_power_now_w", 0.0))
        m["hourly_load"].learn_one(last["features"],  agg.get("load_power_now_w",  0.0))
        _save_pair(home_id, "hourly_solar", "hourly_load")

    try:
        solar_h = m["hourly_solar"].predict_one(features)
        load_h  = m["hourly_load"].predict_one(features)
    except Exception:
        solar_h = agg.get("solar_power_now_w", 0.0)
        load_h  = agg.get("load_power_now_w",  0.0)

    if solar_h <= 0 and agg.get("solar_power_now_w", 0) > 0:
        solar_h = agg.get("solar_power_now_w", 0.0)
    if load_h <= 0 and agg.get("load_power_now_w", 0) > 0:
        load_h = agg.get("load_power_now_w", 0.0)

    _save_state(state_p, {"features": features, "recorded_at": now.isoformat()})

    return {
        "forecast_for":   (now + timedelta(hours=1)).isoformat(),
        "solar_next_h_w": round(max(solar_h, 0), 2),
        "load_next_h_w":  round(max(load_h,  0), 2),
    }


# ── Daily forecast ────────────────────────────────────────────────

def run_daily_forecast(home_id: str, agg: dict, temp_c: float) -> dict:
    now      = datetime.now(timezone.utc)
    features = _daily_features(agg, temp_c, now)
    state_p  = f"{_home_dir(home_id)}/last_daily_state.json"
    last     = _load_state(state_p)
    m        = _get(home_id)

    if last:
        m["daily_solar"].learn_one(last["features"], agg.get("solar_power_now_w", 0.0) * 24)
        m["daily_load"].learn_one(last["features"],  agg.get("load_power_now_w",  0.0))
        _save_pair(home_id, "daily_solar", "daily_load")

    try:
        solar_d = m["daily_solar"].predict_one(features)
        load_d  = m["daily_load"].predict_one(features)
    except Exception:
        solar_d = features["total_solar_wh"]
        load_d  = features["mean_load_w"]

    tomorrow = (now + timedelta(days=1)).replace(
        hour=0, minute=0, second=0, microsecond=0
    )
    _save_state(state_p, {"features": features, "recorded_at": now.isoformat()})

    return {
        "forecast_for":      tomorrow.isoformat(),
        "solar_tomorrow_wh": round(max(solar_d, 0), 2),
        "load_tomorrow_w":   round(max(load_d,  0), 2),
    }


# ── Monthly forecast ──────────────────────────────────────────────

def run_monthly_forecast(home_id: str, agg: dict, temp_c: float) -> dict:
    now      = datetime.now(timezone.utc)
    features = _monthly_features(agg, temp_c, now)
    state_p  = f"{_home_dir(home_id)}/last_monthly_state.json"
    last     = _load_state(state_p)
    m        = _get(home_id)

    if last:
        m["monthly_solar"].learn_one(last["features"], agg.get("solar_power_now_w", 0.0) * 24 * 30)
        m["monthly_load"].learn_one(last["features"],  agg.get("load_power_now_w",  0.0))
        _save_pair(home_id, "monthly_solar", "monthly_load")

    try:
        solar_m = m["monthly_solar"].predict_one(features)
        load_m  = m["monthly_load"].predict_one(features)
    except Exception:
        solar_m = features["total_solar_wh"]
        load_m  = features["mean_load_w"]

    if now.month == 12:
        next_month = now.replace(year=now.year + 1, month=1, day=1,
                                  hour=0, minute=0, second=0, microsecond=0)
    else:
        next_month = now.replace(month=now.month + 1, day=1,
                                  hour=0, minute=0, second=0, microsecond=0)

    _save_state(state_p, {"features": features, "recorded_at": now.isoformat()})

    return {
        "forecast_for":        next_month.isoformat(),
        "solar_next_month_wh": round(max(solar_m, 0), 2),
        "load_next_month_w":   round(max(load_m,  0), 2),
    }
