"""
Run with: pytest tests/

Covers the core ML/physics pipeline: ESP payload parsing, physics
calculations, the 5-minute train/predict cycle, drift tracking, model
persistence (including the 64KB chunking workaround), and the
generation-based delete mechanism. Uses the fake_backend fixture from
conftest.py so no real InfluxDB or OpenWeatherMap connection is needed.
"""

import math
from datetime import datetime, timedelta, timezone

import pytest


# ── ESP payload parsing ─────────────────────────────────────────────

def test_faulty_temperature_falls_back_to_previous_value():
    from utils.esp_payload import ESPPayload, flatten_esp_payload

    raw = {
        "device_id": "test_home",
        "timestamp": "2026-08-21T09:35:00Z",
        "solar": {"voltage_v": 0.0, "current_a": 0.89},
        "battery": {"voltage_v": 12.4, "current_a": 0.10, "charging": True},
        "load": {"current_a": 27.85},
        "temperature_c": -127.0,  # DS18B20 disconnected sentinel
        "interval_s": 300,
    }
    flat = flatten_esp_payload(ESPPayload(**raw), fallback_temp=31.0)

    assert flat["temperature_valid"] is False
    assert flat["temperature"] == 31.0
    assert flat["temperature_raw"] == -127.0


def test_battery_current_sign_follows_charging_flag():
    from utils.esp_payload import ESPPayload, flatten_esp_payload

    base = {
        "device_id": "test_home",
        "timestamp": "2026-08-21T09:35:00Z",
        "solar": {"voltage_v": 18.0, "current_a": 2.0},
        "battery": {"voltage_v": 12.4, "current_a": 0.10, "charging": True},
        "load": {"current_a": 2.0},
        "temperature_c": 30.0,
    }
    charging = flatten_esp_payload(ESPPayload(**base))
    assert charging["battery_current"] == 0.10

    base["battery"]["charging"] = False
    discharging = flatten_esp_payload(ESPPayload(**base))
    assert discharging["battery_current"] == -0.10


# ── Physics ──────────────────────────────────────────────────────────

def test_apparent_power_and_runtime_use_correct_quantities():
    from core.physics_and_models import compute_physics

    data = {
        "solar_voltage": 18.0, "solar_current": 2.0,
        "battery_voltage": 12.6, "battery_current": -2.0,  # discharging
        "load_current": 1.5,  # AC RMS amps
        "battery_type": "LEAD_ACID", "nominal_voltage": "12V", "battery_capacity_wh": 100,
    }
    p = compute_physics(data)

    assert p["solar_power_physics"] == 36.0                     # 18 x 2, DC, exact
    assert p["load_power_physics"] == 240.0 * 1.5                # apparent power, nominal 240V AC
    assert p["battery_power_flow"] == 12.6 * -2.0                # negative = discharging
    assert p["battery_discharge_power"] == pytest.approx(25.2)
    # runtime is driven by the real DC discharge rate, not the AC load figure
    expected_runtime = (p["soc_physics"] * 100) / p["battery_discharge_power"]
    assert p["runtime_physics"] == pytest.approx(expected_runtime, abs=0.01)


def test_coulomb_counting_is_dimensionally_correct():
    from core.physics_and_models import coulomb_counting_soc

    # 100Wh battery, discharging at 12V x 2A = 24W for 1 hour -> SOC drops by 24/100 = 0.24
    result = coulomb_counting_soc(
        soc_prev=0.80, battery_current=-2.0, battery_voltage=12.0,
        time_delta_h=1.0, battery_capacity_wh=100.0,
    )
    assert result == pytest.approx(0.80 - 0.24, abs=1e-9)


# ── Train / predict cycle ───────────────────────────────────────────

def _make_reading(home_id, recorded_at, sun=0.5):
    return {
        "home_id": home_id, "lat": 5.5, "lon": 5.7,
        "battery_type": "LEAD_ACID", "nominal_voltage": "12V", "battery_capacity_wh": 100,
        "recorded_at": recorded_at.isoformat(),
        "solar_voltage": 18.0 if sun > 0 else 0.0,
        "solar_current": round(sun * 3.0, 2),
        "battery_voltage": 12.5,
        "battery_current": 0.5,
        "load_current": 2.0,
        "temperature": 28.0 + sun * 10,
    }


def test_full_train_predict_cycle_runs_without_error(fake_backend):
    from core.physics_and_models import train, predict

    home_id = "cycle_test_home"
    t0 = datetime(2026, 8, 20, 6, 0, 0, tzinfo=timezone.utc)

    for i in range(10):
        recorded_at = t0 + timedelta(minutes=5 * i)
        sun = max(0.0, math.cos((recorded_at.hour - 12) / 6 * math.pi / 2))
        data = _make_reading(home_id, recorded_at, sun=sun)

        train(data)
        result = predict(data)

        assert result["solar_power_now_w"] >= 0
        assert result["load_power_now_w"] >= 0
        assert 0 <= result["soc_now_percent"] <= 100
        assert "weather" in result


def test_out_of_order_reading_does_not_corrupt_state(fake_backend):
    from core.physics_and_models import train
    import db.influx_client as ic

    home_id = "out_of_order_test"
    t0 = datetime(2026, 8, 20, 6, 0, 0, tzinfo=timezone.utc)

    train(_make_reading(home_id, t0))  # no prior state -> no-op
    from core.physics_and_models import predict
    predict(_make_reading(home_id, t0))

    state_before = ic.load_pipeline_state(home_id, "five_min")

    # a reading claiming to be EARLIER than what's already recorded
    train(_make_reading(home_id, t0 - timedelta(minutes=5)))

    state_after = ic.load_pipeline_state(home_id, "five_min")
    assert state_after == state_before


def test_drift_compares_previous_prediction_to_new_actual(fake_backend):
    from core.physics_and_models import train, predict

    home_id = "drift_test_home"
    t0 = datetime(2026, 8, 22, 12, 0, 0, tzinfo=timezone.utc)

    data1 = _make_reading(home_id, t0, sun=0.5)
    drift1 = train(data1)
    result1 = predict(data1)
    assert drift1 is None  # nothing to compare on the very first reading

    t1 = t0 + timedelta(minutes=5)
    data2 = _make_reading(home_id, t1, sun=0.7)  # different actual solar output
    drift2 = train(data2)

    assert drift2 is not None
    assert drift2["solar_predicted_w"] == result1["solar_next_w"]
    assert drift2["solar_actual_w"] == pytest.approx(18.0 * round(0.7 * 3.0, 2))
    assert drift2["solar_error_w"] == pytest.approx(drift2["solar_actual_w"] - drift2["solar_predicted_w"])


# ── Model persistence / chunking ────────────────────────────────────

def test_model_survives_a_simulated_restart(fake_backend):
    from core.physics_and_models import train, predict
    import core.model_store as ms

    home_id = "restart_test_home"
    t0 = datetime(2026, 8, 20, 6, 0, 0, tzinfo=timezone.utc)

    # train() is a no-op on the very first reading (nothing to compare
    # against yet) -- a real save only happens from the second reading
    # onward, same as in production.
    train(_make_reading(home_id, t0))
    predict(_make_reading(home_id, t0))
    train(_make_reading(home_id, t0 + timedelta(minutes=5)))
    predict(_make_reading(home_id, t0 + timedelta(minutes=5)))

    assert any(k[0] == home_id and k[2] == "solar_5min" for k in fake_backend["model_blobs"])

    ms.clear_home_cache(home_id)  # simulates a fresh process picking this home up again
    reloaded = ms.get_five_min_models(home_id)
    assert reloaded["solar_5min"] is not None


def test_large_model_blob_survives_the_64kb_field_limit(monkeypatch):
    """
    InfluxDB rejects any single string field over 64KB. This confirms
    the chunking in db.influx_client.save_model_blob/load_model_blob
    correctly splits and reassembles a blob well past that limit.
    """
    import db.influx_client as ic

    fake_points = {}

    def fake_write(bucket, org, record):
        fields = dict(record._fields)
        tags = dict(record._tags)
        for k, v in fields.items():
            if isinstance(v, str):
                assert len(v.encode("utf-8")) <= 65536, f"field {k} exceeds InfluxDB's 64KB limit"
        fake_points[(tags["home_id"], tags["model_name"])] = fields

    class FakePoint:
        def __init__(self, measurement):
            self.measurement = measurement
            self._tags, self._fields, self._time = {}, {}, None
        def tag(self, k, v): self._tags[k] = v; return self
        def field(self, k, v): self._fields[k] = v; return self
        def time(self, t, precision=None): self._time = t; return self

    class FakeWriteApi:
        def write(self, bucket, org, record): fake_write(bucket, org, record)

    monkeypatch.setattr(ic, "Point", FakePoint)
    monkeypatch.setattr(ic, "_write_api", FakeWriteApi())

    def fake_load(home_id, model_name):
        fields = fake_points.get((home_id, model_name))
        if not fields or "chunk_count" not in fields:
            return None
        parts = [fields[f"chunk_{i}"] for i in range(int(fields["chunk_count"]))]
        return "".join(parts)

    import random, string
    random.seed(1)
    big_blob = "".join(random.choices(string.ascii_letters + string.digits, k=250_000))

    ic.save_model_blob("home1", "solar_5min", big_blob)
    monkeypatch.setattr(ic, "load_model_blob", fake_load)
    reloaded = ic.load_model_blob("home1", "solar_5min")

    assert reloaded == big_blob


# ── Generation-based delete ──────────────────────────────────────────

def test_deleting_a_home_hides_prior_data_even_if_backdated(fake_backend):
    """
    A device_id's data must become invisible immediately after delete,
    and any NEW data written afterwards must be visible regardless of
    what timestamp it claims to represent (a backfill/seed script may
    legitimately write readings timestamped days in the past).
    """
    from db.influx_client import write_sensor_reading, get_latest_sensor, delete_home_data

    home_id = "delete_test_home"
    now = datetime.now(timezone.utc)

    write_sensor_reading({"home_id": home_id, "battery_type": "LEAD_ACID",
                           "solar_voltage": 1.0, "solar_current": 1.0,
                           "battery_voltage": 12.0, "battery_current": 0.1,
                           "load_current": 1.0, "temperature": 25.0,
                           "recorded_at": (now - timedelta(days=10)).isoformat()},
                          now - timedelta(days=10))
    assert get_latest_sensor(home_id) is not None

    delete_home_data(home_id)
    assert get_latest_sensor(home_id) is None

    # backdated data written AFTER the delete must still be visible
    write_sensor_reading({"home_id": home_id, "battery_type": "LEAD_ACID",
                           "solar_voltage": 5.0, "solar_current": 1.0,
                           "battery_voltage": 12.0, "battery_current": 0.1,
                           "load_current": 1.0, "temperature": 25.0,
                           "recorded_at": (now - timedelta(days=3)).isoformat()},
                          now - timedelta(days=3))
    result = get_latest_sensor(home_id)
    assert result is not None
    assert result["solar_voltage"] == 5.0


# ── Regression: SOC must never exceed 100% ──────────────────────────

def test_soc_never_exceeds_100_percent_above_curve_top():
    """
    Regression test for a bug seen in production: a battery reading
    13.4V against the 12V LEAD_ACID curve (top point 12.70V) linearly
    extrapolated to 135% SOC, which fed a nonsense runtime and kept
    re-anchoring the Coulomb counter to an impossible value.
    """
    from utils.constants import VOLTAGE_SOC_CURVE
    from core.physics_and_models import voltage_to_soc

    lead = VOLTAGE_SOC_CURVE["LEAD_ACID"]["12V"]
    for v in [12.70, 13.0, 13.40625, 13.4975, 20.0]:
        soc = voltage_to_soc(v, lead)
        assert 0.0 <= soc <= 1.0, f"{v}V produced out-of-range SOC {soc}"

    for v in [10.5, 9.0, 0.0, -5.0]:
        soc = voltage_to_soc(v, lead)
        assert 0.0 <= soc <= 1.0, f"{v}V produced out-of-range SOC {soc}"

    # a real 13.4V reading on the correct LiFePO4 curve is ~90%, not pinned
    lifepo4 = VOLTAGE_SOC_CURVE["LIFEPO4"]["12V"]
    assert 0.85 < voltage_to_soc(13.40625, lifepo4) < 0.95


def test_history_load_matches_live_load_formula():
    """
    The history screen and the live dashboard must agree on what
    "actual load" means. Both derive it from NOMINAL_AC_VOLTAGE_V x
    load_current, never from battery_voltage x load_current.
    """
    from utils.constants import NOMINAL_AC_VOLTAGE_V
    from core.physics_and_models import compute_physics

    data = {
        "solar_voltage": 22.2825, "solar_current": 3.504,
        "battery_voltage": 13.40625, "battery_current": -6.316,
        "load_current": 0.65662, "temperature": 30.0,
        "battery_type": "LIFEPO4", "nominal_voltage": "12V",
        "battery_capacity_wh": 1200,
    }
    physics = compute_physics(data)
    expected_load = round(NOMINAL_AC_VOLTAGE_V * data["load_current"], 3)
    assert physics["load_power_physics"] == expected_load
    # and the wrong (battery-rail) formula must NOT match it
    assert abs(physics["load_power_physics"] - abs(data["battery_voltage"] * data["load_current"])) > 1.0


# ── Model store read failures must not destroy training ────────────
#
# Regression tests for the failure observed in deployment on 27 August
# 2026, where InfluxDB returned a 500 on load_model_blob and every
# affected model was silently replaced by an untrained one and saved
# back over days of accumulated learning.


def test_read_failure_raises_instead_of_returning_a_blank_model(fake_backend):
    """A read error must propagate, not masquerade as 'no model yet'."""
    import core.model_store as ms
    from db.influx_client import ModelStoreUnavailable

    def exploding_load(home_id, model_name):
        raise ModelStoreUnavailable("simulated InfluxDB 500")

    ms.load_model_blob = exploding_load
    ms._cache.clear()

    with pytest.raises(ModelStoreUnavailable):
        ms.get_five_min_models("home1")


def test_read_failure_leaves_the_stored_model_untouched(fake_backend):
    """
    The damaging part was never the failed read, it was the save that
    followed it. Nothing may be written when the read failed.
    """
    import core.model_store as ms
    from db.influx_client import ModelStoreUnavailable

    ms._cache.clear()
    trained = ms.get_five_min_models("home1")
    ms.save_models("home1", trained)
    before = dict(fake_backend["model_blobs"])
    assert before, "a baseline model should have been stored"

    ms._cache.clear()
    ms.load_model_blob = lambda h, n: (_ for _ in ()).throw(
        ModelStoreUnavailable("simulated InfluxDB 500")
    )

    with pytest.raises(ModelStoreUnavailable):
        ms.get_five_min_models("home1")

    assert fake_backend["model_blobs"] == before


def test_absent_model_still_starts_fresh(fake_backend):
    """
    The fix must not break the genuine cold-start case: no stored model
    means a new one, with no exception raised.
    """
    import core.model_store as ms

    ms._cache.clear()
    models = ms.get_five_min_models("a_home_never_seen_before")
    assert set(models) == {"solar_5min", "load_5min", "soc_5min"}
    for model in models.values():
        assert model is not None


def test_cache_is_not_populated_when_a_load_fails(fake_backend):
    """
    A half-filled cache would let a later call quietly succeed with a
    blank model for whichever one failed.
    """
    import core.model_store as ms
    from db.influx_client import ModelStoreUnavailable

    ms._cache.clear()

    def load_only_the_first(home_id, model_name):
        if model_name == "solar_5min":
            return None
        raise ModelStoreUnavailable("simulated InfluxDB 500")

    ms.load_model_blob = load_only_the_first

    with pytest.raises(ModelStoreUnavailable):
        ms.get_five_min_models("home1")

    assert not ms._cache.get("home1")
