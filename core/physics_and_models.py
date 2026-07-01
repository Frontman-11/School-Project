import json
import os
import pickle
from datetime import datetime, timedelta, timezone
from river import ensemble, preprocessing, compose
from utils.constants import VOLTAGE_SOC_CURVE, encode_hour, encode_day, encode_month
from utils.weather import get_weather


# ── Per-home model paths ──────────────────────────────────────────

def _home_dir(home_id: str) -> str:
    path = f"model_state/{home_id}"
    os.makedirs(path, exist_ok=True)
    return path


def _paths(home_id: str) -> dict:
    d = _home_dir(home_id)
    return {
        "solar": f"{d}/solar_forecast_model.pkl",
        "load":  f"{d}/load_forecast_model.pkl",
        "soc":   f"{d}/soc_correction_model.pkl",
        "state": f"{d}/last_state.json",
    }


# ── Model helpers ─────────────────────────────────────────────────

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


# In-memory model cache keyed by home_id
_model_cache: dict = {}


def _get_models(home_id: str) -> dict:
    if home_id not in _model_cache:
        p = _paths(home_id)
        _model_cache[home_id] = {
            "solar": _load_model(p["solar"]),
            "load":  _load_model(p["load"]),
            "soc":   _load_model(p["soc"]),
        }
    return _model_cache[home_id]


def _save_models(home_id: str):
    p      = _paths(home_id)
    models = _get_models(home_id)
    _save_model(models["solar"], p["solar"])
    _save_model(models["load"],  p["load"])
    _save_model(models["soc"],   p["soc"])


# ── Physics ───────────────────────────────────────────────────────

def voltage_to_soc(voltage: float, curve: list) -> float:
    for i in range(len(curve) - 1):
        v_high, soc_high = curve[i]
        v_low,  soc_low  = curve[i + 1]
        if voltage >= v_low:
            ratio = (voltage - v_low) / (v_high - v_low)
            return soc_low + ratio * (soc_high - soc_low)
    return 0.0


def coulomb_counting_soc(soc_prev: float, battery_current: float,
                          time_delta_h: float, battery_capacity_wh: float) -> float:
    delta = (battery_current * time_delta_h) / battery_capacity_wh
    return min(max(soc_prev + delta, 0.0), 1.0)


def compute_physics(data: dict) -> dict:
    solar_power = data["solar_voltage"] * data["solar_current"]
    load_power  = data["battery_voltage"] * data["load_current"]
    curve       = VOLTAGE_SOC_CURVE[data["battery_type"]][data["nominal_voltage"]]
    soc_physics = voltage_to_soc(data["battery_voltage"], curve)
    cap         = data["battery_capacity_wh"]
    runtime     = (soc_physics * cap) / load_power if load_power > 0 else 0
    return {
        "solar_power_physics": round(solar_power, 3),
        "load_power_physics":  round(load_power, 3),
        "soc_physics":         round(soc_physics, 4),
        "runtime_physics":     round(runtime, 3),
    }


# ── Feature builders ──────────────────────────────────────────────

def build_forecast_features(data: dict, physics: dict, weather: dict) -> dict:
    recorded_at = datetime.fromisoformat(data["recorded_at"])
    features = {
        "solar_power_now":    physics["solar_power_physics"],
        "load_power_now":     physics["load_power_physics"],
        "sensor_temp_c":      data["temperature"],
        "cloud_cover_pct":    weather["cloud_cover_pct"],
        "ambient_temp_c":     weather["ambient_temp_c"],
        "precipitation_prob": weather["precipitation_prob"],
        "minute":             recorded_at.minute,
    }
    features.update(encode_hour(recorded_at.hour))
    features.update(encode_day(recorded_at.weekday()))
    features.update(encode_month(recorded_at.month))
    return features


def build_soc_features(data: dict, physics: dict,
                        soc_coulomb: float, weather: dict) -> dict:
    recorded_at = datetime.fromisoformat(data["recorded_at"])
    features = {
        "soc_physics":     physics["soc_physics"],
        "soc_coulomb":     soc_coulomb,
        "battery_voltage": data["battery_voltage"],
        "battery_current": data["battery_current"],
        "sensor_temp_c":   data["temperature"],
        "ambient_temp_c":  weather["ambient_temp_c"],
    }
    features.update(encode_hour(recorded_at.hour))
    return features


# ── Last state ────────────────────────────────────────────────────

def _load_last_state(home_id: str) -> dict | None:
    path = _paths(home_id)["state"]
    if os.path.exists(path):
        with open(path, "r") as f:
            return json.load(f)
    return None


def _save_last_state(home_id: str, data: dict, physics: dict,
                     forecast_features: dict, soc_features: dict,
                     soc_estimate: float):
    path = _paths(home_id)["state"]
    with open(path, "w") as f:
        json.dump({
            "data":              data,
            "physics":           physics,
            "forecast_features": forecast_features,
            "soc_features":      soc_features,
            "soc_estimate":      soc_estimate,
            "recorded_at":       data["recorded_at"],
        }, f)


# ── Train ─────────────────────────────────────────────────────────

def train(data: dict):
    home_id = data["home_id"]
    last    = _load_last_state(home_id)
    if not last:
        return

    prev_time    = datetime.fromisoformat(last["recorded_at"])
    curr_time    = datetime.fromisoformat(data["recorded_at"])
    time_delta_h = (curr_time - prev_time).total_seconds() / 3600

    physics = compute_physics(data)
    models  = _get_models(home_id)

    models["solar"].learn_one(last["forecast_features"], physics["solar_power_physics"])
    models["load"].learn_one(last["forecast_features"],  physics["load_power_physics"])

    soc_coulomb = coulomb_counting_soc(
        last["soc_estimate"],
        data["battery_current"],
        time_delta_h,
        data["battery_capacity_wh"]
    )
    models["soc"].learn_one(last["soc_features"], soc_coulomb)
    _save_models(home_id)


# ── Predict ───────────────────────────────────────────────────────

def predict(data: dict) -> dict:
    home_id    = data["home_id"]
    physics    = compute_physics(data)
    weather    = get_weather(home_id, data["lat"], data["lon"])
    last       = _load_last_state(home_id)
    models     = _get_models(home_id)
    recorded_at = datetime.fromisoformat(data["recorded_at"])
    forecast_for = recorded_at + timedelta(minutes=5)

    if last:
        prev_time    = datetime.fromisoformat(last["recorded_at"])
        time_delta_h = (recorded_at - prev_time).total_seconds() / 3600
        soc_coulomb  = coulomb_counting_soc(
            last["soc_estimate"],
            data["battery_current"],
            time_delta_h,
            data["battery_capacity_wh"]
        )
        if abs(soc_coulomb - physics["soc_physics"]) > 0.3:
            soc_coulomb = physics["soc_physics"]
    else:
        soc_coulomb = physics["soc_physics"]

    forecast_features = build_forecast_features(data, physics, weather)
    soc_features      = build_soc_features(data, physics, soc_coulomb, weather)

    try:
        solar_next    = models["solar"].predict_one(forecast_features)
        load_next     = models["load"].predict_one(forecast_features)
        soc_corrected = models["soc"].predict_one(soc_features)
    except Exception:
        solar_next    = physics["solar_power_physics"]
        load_next     = physics["load_power_physics"]
        soc_corrected = soc_coulomb

    # sanity checks
    if solar_next <= 0 and physics["solar_power_physics"] > 0:
        solar_next = physics["solar_power_physics"]
    if load_next <= 0 and physics["load_power_physics"] > 0:
        load_next = physics["load_power_physics"]
    if abs(soc_corrected - soc_coulomb) > 0.3:
        soc_corrected = soc_coulomb

    solar_next    = max(solar_next, 0)
    load_next     = max(load_next, 0)
    soc_corrected = min(max(soc_corrected, 0.0), 1.0)

    cap     = data["battery_capacity_wh"]
    runtime = (soc_corrected * cap) / load_next if load_next > 0 else 0

    _save_last_state(home_id, data, physics,
                     forecast_features, soc_features, soc_corrected)

    return {
        "recorded_at":       data["recorded_at"],
        "forecast_for":      forecast_for.isoformat(),
        "solar_power_now_w": round(physics["solar_power_physics"], 2),
        "load_power_now_w":  round(physics["load_power_physics"], 2),
        "soc_now_percent":   round(soc_corrected * 100, 1),
        "solar_next_w":      round(solar_next, 2),
        "load_next_w":       round(load_next, 2),
        "runtime_hours":     round(runtime, 2),
        "cloud_cover_pct":   weather["cloud_cover_pct"],
        "weather_condition": weather["weather_condition"],
        "soc_physics_pct":   round(physics["soc_physics"] * 100, 1),
        "soc_coulomb_pct":   round(soc_coulomb * 100, 1),
    }
