"""
Physics baseline + 5-minute-ahead River models, per home. All model
state and the train/predict pairing state ("pipeline_state") live in
InfluxDB via model_store and db.influx_client -- nothing is written to
local disk, so this is safe to run on Render's ephemeral filesystem and
survives restarts.
"""

from datetime import datetime, timedelta
from utils.constants import (
    VOLTAGE_SOC_CURVE,
    encode_hour,
    encode_day,
    encode_month,
    NOMINAL_AC_VOLTAGE_V,
    SOC_REANCHOR_THRESHOLD,
    SOC_REST_CURRENT_A,
    MIN_DISCHARGE_FOR_RUNTIME_W,
    MAX_RUNTIME_HOURS,
    MAX_PLAUSIBLE_SOLAR_W,
    MAX_PLAUSIBLE_LOAD_VA,
)
from utils.weather import get_weather
from core import model_store
from db.influx_client import load_pipeline_state, save_pipeline_state


# ── Physics ───────────────────────────────────────────────────────


def voltage_to_soc(voltage: float, curve: list) -> float:
    """
    Interpolates state of charge from terminal voltage against a
    chemistry/voltage-specific lookup curve.

    The result is clamped to 0-100%. Without that clamp, a voltage above
    the curve's top point extrapolates past 100% -- e.g. a battery
    sitting at 13.4V read against the 12V LEAD_ACID curve (which tops
    out at 12.70V) returned 135%, which then fed a nonsense runtime
    estimate and constantly re-anchored the Coulomb counter to a value
    that could never be right. Voltage outside the curve's range now
    pins to the nearest endpoint instead of running off the end of it.
    """
    v_top, soc_top = curve[0]
    if voltage >= v_top:
        return soc_top

    for i in range(len(curve) - 1):
        v_high, soc_high = curve[i]
        v_low, soc_low = curve[i + 1]
        if voltage >= v_low:
            ratio = (voltage - v_low) / (v_high - v_low)
            soc = soc_low + ratio * (soc_high - soc_low)
            return min(max(soc, 0.0), 1.0)
    return 0.0


def coulomb_counting_soc(
    soc_prev: float,
    battery_current: float,
    battery_voltage: float,
    time_delta_h: float,
    battery_capacity_wh: float,
) -> float:
    """
    Integrates net energy transfer (in watt-hours) and divides by the
    battery's rated capacity (also in watt-hours) to get a SOC fraction.

    Note: an earlier version of this divided amp-hours (current x time)
    directly by a watt-hour capacity, which is dimensionally wrong --
    it only happened to be correct at exactly 1 volt. Multiplying by
    battery_voltage first converts the integration into watt-hours so
    the division against battery_capacity_wh is unit-correct.
    """
    delta_wh = battery_current * battery_voltage * time_delta_h
    return min(max(soc_prev + delta_wh / battery_capacity_wh, 0.0), 1.0)


def compute_physics(data: dict) -> dict:
    solar_power = data["solar_voltage"] * data["solar_current"]

    # Load side: SCT-013-050 measures AC RMS current with no voltage or
    # phase reference, so apparent power (V_nominal x I_rms) is the only
    # quantity computed -- no power factor assumption is made anywhere.
    load_apparent_power = NOMINAL_AC_VOLTAGE_V * data["load_current"]

    # Battery side: directly measured DC power flow at the battery
    # terminal (INA226 + shunt). Positive = charging, negative = net
    # discharge. This already reflects whatever the inverter itself
    # draws to supply the AC load, including its own conversion losses
    # -- unlike the AC-side load figures above, it needs no assumptions.
    battery_power_flow = data["battery_voltage"] * data["battery_current"]
    battery_discharge_power = max(-battery_power_flow, 0.0)

    curve = VOLTAGE_SOC_CURVE[data["battery_type"]][data["nominal_voltage"]]
    soc_physics = voltage_to_soc(data["battery_voltage"], curve)
    cap = data["battery_capacity_wh"]

    # Runtime uses the directly-measured battery discharge rate, not the
    # AC-side load estimate, so it is not affected by the load power
    # factor assumption or by unmeasured inverter conversion losses.
    # None (not 0) when the battery is charging or idle: there is no
    # discharge to divide by, so runtime is genuinely undefined rather
    # than zero. Sending 0 made the app show "0 min remaining / LOW"
    # next to a full battery, because 0 is a real number that passes
    # every "is there a value?" check downstream.
    #
    # The discharge floor matters as much as the zero check. Testing only
    # for "> 0" let the quotient run away whenever the battery was
    # trickling at the noise floor of the current measurement: 142 of 209
    # defined runtimes exceeded 100 h and the largest reached 56,219 h on
    # a 2.5 kWh battery. Below the floor the answer is undefined, not
    # enormous, and above it the result is capped.
    if battery_discharge_power >= MIN_DISCHARGE_FOR_RUNTIME_W:
        runtime = min(
            (soc_physics * cap) / battery_discharge_power,
            MAX_RUNTIME_HOURS,
        )
    else:
        runtime = None

    return {
        "solar_power_physics": round(solar_power, 3),
        "load_power_physics": round(
            load_apparent_power, 3
        ),  # apparent power (VA), exact
        "battery_power_flow": round(battery_power_flow, 3),
        "battery_discharge_power": round(battery_discharge_power, 3),
        "soc_physics": round(soc_physics, 4),
        "runtime_physics": round(runtime, 3) if runtime is not None else None,
    }


# ── Feature builders ──────────────────────────────────────────────


def build_forecast_features(data: dict, physics: dict, weather: dict) -> dict:
    recorded_at = datetime.fromisoformat(data["recorded_at"])
    features = {
        "solar_power_now": physics["solar_power_physics"],
        "load_power_now": physics["load_power_physics"],
        "sensor_temp_c": data["temperature"],
        "cloud_cover_pct": weather["cloud_cover_pct"],
        "ambient_temp_c": weather["ambient_temp_c"],
        "precipitation_prob": weather["precipitation_prob"],
        "minute": recorded_at.minute,
    }
    features.update(encode_hour(recorded_at.hour))
    features.update(encode_day(recorded_at.weekday()))
    features.update(encode_month(recorded_at.month))
    return features


def build_soc_features(
    data: dict, physics: dict, soc_coulomb: float, weather: dict
) -> dict:
    recorded_at = datetime.fromisoformat(data["recorded_at"])
    features = {
        "soc_physics": physics["soc_physics"],
        "soc_coulomb": soc_coulomb,
        "battery_voltage": data["battery_voltage"],
        "battery_current": data["battery_current"],
        "sensor_temp_c": data["temperature"],
        "ambient_temp_c": weather["ambient_temp_c"],
    }
    features.update(encode_hour(recorded_at.hour))
    return features


# ── Train ─────────────────────────────────────────────────────────


def train(data: dict) -> dict | None:
    """
    Returns a drift dict comparing the PREVIOUS cycle's prediction
    against what actually arrived just now (the closest thing to a
    ground-truth check this system has), or None on the very first
    reading for a home, when there's nothing yet to compare against.
    """
    home_id = data["home_id"]
    last = load_pipeline_state(home_id, "five_min")
    if not last:
        return None

    prev_time = datetime.fromisoformat(last["recorded_at"])
    curr_time = datetime.fromisoformat(data["recorded_at"])
    time_delta_h = (curr_time - prev_time).total_seconds() / 3600
    if time_delta_h <= 0:
        # out-of-order or duplicate reading -- do not train on it
        return None

    physics = compute_physics(data)
    models = model_store.get_five_min_models(home_id)

    models["solar_5min"].learn_one(
        last["forecast_features"], physics["solar_power_physics"]
    )
    models["load_5min"].learn_one(
        last["forecast_features"], physics["load_power_physics"]
    )

    soc_coulomb = coulomb_counting_soc(
        last["soc_estimate"],
        data["battery_current"],
        data["battery_voltage"],
        time_delta_h,
        data["battery_capacity_wh"],
    )
    models["soc_5min"].learn_one(last["soc_features"], soc_coulomb)

    model_store.save_models(home_id, models)

    drift = None
    if "predicted_solar_next_w" in last and "predicted_load_next_w" in last:
        solar_actual = physics["solar_power_physics"]
        load_actual = physics["load_power_physics"]
        solar_pred = last["predicted_solar_next_w"]
        load_pred = last["predicted_load_next_w"]
        drift = {
            "compared_to_forecast_for": last.get("forecast_for"),
            "solar_predicted_w": round(solar_pred, 2),
            "solar_actual_w": round(solar_actual, 2),
            "solar_error_w": round(solar_actual - solar_pred, 2),
            "solar_abs_pct_error": round(
                abs(solar_actual - solar_pred) / solar_actual * 100, 1
            )
            if solar_actual > 0
            else None,
            "load_predicted_w": round(load_pred, 2),
            "load_actual_w": round(load_actual, 2),
            "load_error_w": round(load_actual - load_pred, 2),
            "load_abs_pct_error": round(
                abs(load_actual - load_pred) / load_actual * 100, 1
            )
            if load_actual > 0
            else None,
        }
    return drift


# ── Predict ───────────────────────────────────────────────────────


def predict(data: dict) -> dict:
    home_id = data["home_id"]
    physics = compute_physics(data)
    weather = get_weather(home_id, data["lat"], data["lon"])
    last = load_pipeline_state(home_id, "five_min")
    models = model_store.get_five_min_models(home_id)

    recorded_at = datetime.fromisoformat(data["recorded_at"])
    forecast_for = recorded_at + timedelta(minutes=5)

    if last:
        prev_time = datetime.fromisoformat(last["recorded_at"])
        time_delta_h = (recorded_at - prev_time).total_seconds() / 3600
        if time_delta_h > 0:
            soc_coulomb = coulomb_counting_soc(
                last["soc_estimate"],
                data["battery_current"],
                data["battery_voltage"],
                time_delta_h,
                data["battery_capacity_wh"],
            )
        else:
            soc_coulomb = last["soc_estimate"]
        # Re-anchor to the voltage curve either when the battery is at
        # rest, where terminal voltage is the most trustworthy indicator
        # available, or when the two estimates have drifted apart. The
        # rest condition exists because the drift test alone is not
        # enough: a counter saturated at 1.00 against a voltage estimate
        # of 0.79 differs by only 0.21 and would sit there indefinitely
        # behind a looser threshold, which is what happened in
        # deployment.
        at_rest = abs(data["battery_current"]) < SOC_REST_CURRENT_A
        drifted = (
            abs(soc_coulomb - physics["soc_physics"]) > SOC_REANCHOR_THRESHOLD
        )
        if at_rest or drifted:
            soc_coulomb = physics["soc_physics"]
    else:
        soc_coulomb = physics["soc_physics"]

    forecast_features = build_forecast_features(data, physics, weather)
    soc_features = build_soc_features(data, physics, soc_coulomb, weather)

    try:
        solar_next = models["solar_5min"].predict_one(forecast_features)
        load_next = models["load_5min"].predict_one(forecast_features)
        soc_corrected = models["soc_5min"].predict_one(soc_features)
    except Exception:
        solar_next = physics["solar_power_physics"]
        load_next = physics["load_power_physics"]
        soc_corrected = soc_coulomb

    # sanity checks -- always applied, never skipped
    if solar_next is None or (solar_next <= 0 and physics["solar_power_physics"] > 0):
        solar_next = physics["solar_power_physics"]
    if load_next is None or (load_next <= 0 and physics["load_power_physics"] > 0):
        load_next = physics["load_power_physics"]
    if soc_corrected is None or abs(soc_corrected - soc_coulomb) > SOC_REANCHOR_THRESHOLD:
        soc_corrected = soc_coulomb

    # Upper bounds. The checks above only catch a model that collapses
    # toward zero; nothing caught one that diverged upward, and the
    # hourly model once produced 6.87e13 W, which was stored and served.
    # A forecast beyond what the installation can physically do is a
    # diverged model, so fall back to the physics estimate.
    if solar_next > MAX_PLAUSIBLE_SOLAR_W:
        solar_next = physics["solar_power_physics"]
    if load_next > MAX_PLAUSIBLE_LOAD_VA:
        load_next = physics["load_power_physics"]

    solar_next = max(solar_next, 0)
    load_next = max(load_next, 0)
    soc_corrected = min(max(soc_corrected, 0.0), 1.0)

    cap = data["battery_capacity_wh"]
    # Runtime uses the current, exactly-measured battery discharge rate
    # rather than the ML-forecast AC load figure -- this keeps it free
    # of both the load power factor assumption and the model's own
    # short-horizon forecast noise, at the cost of not anticipating a
    # load change in the next 5 minutes. See compute_physics for why,
    # including why the discharge floor and the cap are both needed.
    discharge = physics["battery_discharge_power"]
    if discharge >= MIN_DISCHARGE_FOR_RUNTIME_W:
        runtime = min((soc_corrected * cap) / discharge, MAX_RUNTIME_HOURS)
    else:
        runtime = None

    save_pipeline_state(
        home_id,
        "five_min",
        {
            "data": data,
            "physics": physics,
            "forecast_features": forecast_features,
            "soc_features": soc_features,
            "soc_estimate": soc_corrected,
            "recorded_at": data["recorded_at"],
            "predicted_solar_next_w": solar_next,
            "predicted_load_next_w": load_next,
            "forecast_for": forecast_for.isoformat(),
        },
    )

    return {
        "recorded_at": data["recorded_at"],
        "forecast_for": forecast_for.isoformat(),
        # NOTE: load_power_now_w / load_next_w are APPARENT power (VA),
        # not active power (W) -- the SCT-013-050 clamp has no voltage
        # or phase reference, so V_nominal x I_rms is the only quantity
        # computed. No power factor is assumed anywhere in this system.
        # Field names kept as *_w for compatibility with the existing
        # API contract.
        "solar_power_now_w": round(physics["solar_power_physics"], 2),
        "load_power_now_w": round(physics["load_power_physics"], 2),
        "battery_discharge_power_w": round(physics["battery_discharge_power"], 2),
        "soc_now_percent": round(soc_corrected * 100, 1),
        "solar_next_w": round(solar_next, 2),
        "load_next_w": round(load_next, 2),
        "runtime_hours": round(runtime, 2) if runtime is not None else None,
        "soc_physics_pct": round(physics["soc_physics"] * 100, 1),
        "soc_coulomb_pct": round(soc_coulomb * 100, 1),
        "weather": {
            "cloud_cover_pct": weather["cloud_cover_pct"],
            "ambient_temp_c": weather["ambient_temp_c"],
            "precipitation_prob": weather["precipitation_prob"],
            "weather_condition": weather["weather_condition"],
        },
    }
