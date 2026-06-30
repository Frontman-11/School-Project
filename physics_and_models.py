import json
import os
import pickle
from datetime import datetime, timedelta, timezone
from river import ensemble, preprocessing, compose
from utils.constants import VOLTAGE_SOC_CURVE, encode_hour, encode_day, encode_month
from utils.weather import get_weather

# ── Model persistence paths ───────────────────────────────────────
MODEL_DIR = "model_state"
os.makedirs(MODEL_DIR, exist_ok=True)

SOLAR_MODEL_PATH = f"{MODEL_DIR}/solar_forecast_model.pkl"
LOAD_MODEL_PATH  = f"{MODEL_DIR}/load_forecast_model.pkl"
SOC_MODEL_PATH   = f"{MODEL_DIR}/soc_correction_model.pkl"
LAST_STATE_PATH  = f"{MODEL_DIR}/last_state.json"


# ── Voltage to SOC lookup ─────────────────────────────────────────
def voltage_to_soc(voltage, curve):
    for i in range(len(curve) - 1):
        v_high, soc_high = curve[i]
        v_low,  soc_low  = curve[i + 1]
        if voltage >= v_low:
            ratio = (voltage - v_low) / (v_high - v_low)
            return soc_low + ratio * (soc_high - soc_low)
    return 0.0


# ── Coulomb counting ──────────────────────────────────────────────
def coulomb_counting_soc(soc_prev, battery_current, time_delta_hours, battery_capacity_wh):
    """
    battery_current: positive = charging, negative = discharging.
    Returns clamped SOC between 0.0 and 1.0.
    """
    delta = (battery_current * time_delta_hours) / battery_capacity_wh
    return min(max(soc_prev + delta, 0.0), 1.0)


# ── Physics baseline ──────────────────────────────────────────────
def compute_physics(data):
    solar_power = data["solar_voltage"] * data["solar_current"]
    load_power  = data["battery_voltage"] * data["load_current"]
    curve       = VOLTAGE_SOC_CURVE[data["battery_type"]][data["nominal_voltage"]]
    soc_physics = voltage_to_soc(data["battery_voltage"], curve)

    battery_capacity_wh = data["battery_capacity_wh"]
    runtime_hours = (soc_physics * battery_capacity_wh) / load_power if load_power > 0 else 0

    return {
        "solar_power_physics": round(solar_power, 3),
        "load_power_physics":  round(load_power, 3),
        "soc_physics":         round(soc_physics, 4),
        "runtime_physics":     round(runtime_hours, 3),
    }


# ── Model loading / saving ────────────────────────────────────────
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


solar_forecast_model = load_model(SOLAR_MODEL_PATH)
load_forecast_model  = load_model(LOAD_MODEL_PATH)
soc_correction_model = load_model(SOC_MODEL_PATH)


# ── Feature builders ──────────────────────────────────────────────
def build_forecast_features(data, physics, weather):
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


def build_soc_features(data, physics, soc_coulomb, weather):
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


# ── Last state helpers ────────────────────────────────────────────
def load_last_state():
    if os.path.exists(LAST_STATE_PATH):
        with open(LAST_STATE_PATH, "r") as f:
            return json.load(f)
    return None


def save_last_state(data, physics, forecast_features, soc_features, soc_estimate):
    with open(LAST_STATE_PATH, "w") as f:
        json.dump({
            "data":              data,
            "physics":           physics,
            "forecast_features": forecast_features,
            "soc_features":      soc_features,
            "soc_estimate":      soc_estimate,
            "recorded_at":       data["recorded_at"],
        }, f)


# ── Train ─────────────────────────────────────────────────────────
def train(data):
    """
    Called first when a new reading arrives.
    Uses previous state's features + current actual V×I as ground truth.
    """
    last = load_last_state()
    if not last:
        return

    prev_time    = datetime.fromisoformat(last["recorded_at"])
    curr_time    = datetime.fromisoformat(data["recorded_at"])
    time_delta_h = (curr_time - prev_time).total_seconds() / 3600

    physics = compute_physics(data)

    # ground truth for forecasting = actual V×I arriving now
    solar_forecast_model.learn_one(last["forecast_features"], physics["solar_power_physics"])
    load_forecast_model.learn_one(last["forecast_features"],  physics["load_power_physics"])

    # ground truth for SOC = Coulomb counting from last known SOC
    soc_coulomb = coulomb_counting_soc(
        last["soc_estimate"],
        data["battery_current"],
        time_delta_h,
        data["battery_capacity_wh"]
    )
    soc_correction_model.learn_one(last["soc_features"], soc_coulomb)

    save_model(solar_forecast_model, SOLAR_MODEL_PATH)
    save_model(load_forecast_model,  LOAD_MODEL_PATH)
    save_model(soc_correction_model, SOC_MODEL_PATH)


# ── Predict ───────────────────────────────────────────────────────
def predict(data):
    physics     = compute_physics(data)
    weather     = get_weather(data["home_id"], data["lat"], data["lon"])
    last        = load_last_state()

    recorded_at  = datetime.fromisoformat(data["recorded_at"])
    forecast_for = recorded_at + timedelta(minutes=5)

    # Coulomb counting from last known SOC
    if last:
        prev_time    = datetime.fromisoformat(last["recorded_at"])
        time_delta_h = (recorded_at - prev_time).total_seconds() / 3600
        soc_coulomb  = coulomb_counting_soc(
            last["soc_estimate"],
            data["battery_current"],
            time_delta_h,
            data["battery_capacity_wh"]
        )
        # re-anchor if Coulomb has drifted too far from voltage lookup
        if abs(soc_coulomb - physics["soc_physics"]) > 0.3:
            soc_coulomb = physics["soc_physics"]
    else:
        soc_coulomb = physics["soc_physics"]

    forecast_features = build_forecast_features(data, physics, weather)
    soc_features      = build_soc_features(data, physics, soc_coulomb, weather)

    # model predictions
    try:
        solar_next    = solar_forecast_model.predict_one(forecast_features)
        load_next     = load_forecast_model.predict_one(forecast_features)
        soc_corrected = soc_correction_model.predict_one(soc_features)
    except Exception:
        solar_next    = physics["solar_power_physics"]
        load_next     = physics["load_power_physics"]
        soc_corrected = soc_coulomb

    # sanity checks — always runs regardless of exception
    if solar_next <= 0 and physics["solar_power_physics"] > 0:
        solar_next = physics["solar_power_physics"]
    if load_next <= 0 and physics["load_power_physics"] > 0:
        load_next = physics["load_power_physics"]
    if abs(soc_corrected - soc_coulomb) > 0.3:
        soc_corrected = soc_coulomb

    solar_next    = max(solar_next, 0)
    load_next     = max(load_next, 0)
    soc_corrected = min(max(soc_corrected, 0.0), 1.0)

    battery_capacity_wh = data["battery_capacity_wh"]
    runtime = (soc_corrected * battery_capacity_wh) / load_next if load_next > 0 else 0

    save_last_state(data, physics, forecast_features, soc_features, soc_corrected)

    return {
        "recorded_at":       data["recorded_at"],
        "forecast_for":      forecast_for.isoformat(),
        # current actuals — exact physics
        "solar_power_now_w": round(physics["solar_power_physics"], 2),
        "load_power_now_w":  round(physics["load_power_physics"], 2),
        "soc_now_percent":   round(soc_corrected * 100, 1),
        # forecast for next 5 minutes
        "solar_next_w":      round(solar_next, 2),
        "load_next_w":       round(load_next, 2),
        "runtime_hours":     round(runtime, 2),
        # weather context
        "cloud_cover_pct":   weather["cloud_cover_pct"],
        "weather_condition": weather["weather_condition"],
        # debug
        "soc_physics_pct":   round(physics["soc_physics"] * 100, 1),
        "soc_coulomb_pct":   round(soc_coulomb * 100, 1),
    }


# ── Quick test ────────────────────────────────────────────────────
if __name__ == "__main__":
    import os
    from dotenv import load_dotenv
    load_dotenv()

    sample = {
        "home_id":             os.getenv("HOME_ID", "home1"),
        "lat":                 float(os.getenv("HOME_LAT", 4.8156)),
        "lon":                 float(os.getenv("HOME_LON", 7.0498)),
        "recorded_at":         datetime.now(timezone.utc).isoformat(),
        "solar_voltage":       18.4,
        "solar_current":       2.1,
        "battery_voltage":     12.6,
        "battery_current":     1.8,
        "load_current":        3.2,
        "temperature":         31.5,
        "battery_capacity_wh": int(os.getenv("BATTERY_CAPACITY_WH", 100)),
        "battery_type":        os.getenv("BATTERY_TYPE", "LEAD_ACID"),
        "nominal_voltage":     os.getenv("NOMINAL_VOLTAGE", "12V"),
    }
    train(sample)
    result = predict(sample)
    print(json.dumps(result, indent=2))