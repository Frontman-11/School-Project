"""
Shared test fixtures. Replaces InfluxDB and the weather API with
in-memory fakes so the whole pipeline can be exercised without network
access -- these tests check the actual logic (physics, training,
persistence, generation-based delete), not whether your InfluxDB
credentials happen to work at the moment you run them.
"""

import json
import pytest


@pytest.fixture
def fake_backend(monkeypatch):
    """
    Installs an in-memory fake in place of every InfluxDB read/write
    function used across db.influx_client, core.model_store, and
    core.physics_and_models, and a fake weather lookup. Returns a
    small object with helpers for inspecting what was written, for
    tests that need to assert on persisted state directly.
    """
    import db.influx_client as ic
    import core.model_store as ms
    import core.physics_and_models as pm
    import utils.weather as w

    store = {
        "model_blobs": {},          # (home_id, model_name) -> blob string
        "pipeline_states": {},      # (home_id, kind) -> state dict
        "sensor_points": [],        # list of (payload dict, recorded_at)
        "prediction_points": [],    # list of (result dict, home_id, recorded_at)
        "generations": {},          # home_id -> int
    }

    def get_gen(home_id):
        return store["generations"].get(home_id, 0)

    def delete_home_data(home_id):
        store["generations"][home_id] = get_gen(home_id) + 1

    def save_model_blob(home_id, model_name, blob):
        store["model_blobs"][(home_id, get_gen(home_id), model_name)] = blob

    def load_model_blob(home_id, model_name):
        return store["model_blobs"].get((home_id, get_gen(home_id), model_name))

    def save_pipeline_state(home_id, kind, state):
        store["pipeline_states"][(home_id, get_gen(home_id), kind)] = json.loads(json.dumps(state))

    def load_pipeline_state(home_id, kind):
        return store["pipeline_states"].get((home_id, get_gen(home_id), kind))

    def write_sensor_reading(payload, recorded_at):
        store["sensor_points"].append((dict(payload), get_gen(payload["home_id"]), recorded_at))

    def get_latest_sensor(home_id, range_str=None):
        gen = get_gen(home_id)
        matches = [p for p, g, t in store["sensor_points"] if p.get("home_id") == home_id and g == gen]
        if not matches:
            return None
        latest = matches[-1]
        out = {k: v for k, v in latest.items() if k != "home_id"}
        out["recorded_at"] = latest["recorded_at"]
        return out

    def write_model_prediction(result, home_id, recorded_at):
        store["prediction_points"].append((dict(result), home_id, recorded_at))

    def fake_get_weather(home_id, lat, lon):
        return {
            "cloud_cover_pct": 40.0,
            "ambient_temp_c": 29.0,
            "precipitation_prob": 0.0,
            "weather_condition": "Clouds",
        }

    monkeypatch.setattr(ic, "save_model_blob", save_model_blob)
    monkeypatch.setattr(ic, "load_model_blob", load_model_blob)
    monkeypatch.setattr(ic, "save_pipeline_state", save_pipeline_state)
    monkeypatch.setattr(ic, "load_pipeline_state", load_pipeline_state)
    monkeypatch.setattr(ic, "write_sensor_reading", write_sensor_reading)
    monkeypatch.setattr(ic, "get_latest_sensor", get_latest_sensor)
    monkeypatch.setattr(ic, "write_model_prediction", write_model_prediction)
    monkeypatch.setattr(ic, "delete_home_data", delete_home_data)

    monkeypatch.setattr(ms, "save_model_blob", save_model_blob)
    monkeypatch.setattr(ms, "load_model_blob", load_model_blob)

    monkeypatch.setattr(pm, "load_pipeline_state", load_pipeline_state)
    monkeypatch.setattr(pm, "save_pipeline_state", save_pipeline_state)
    monkeypatch.setattr(pm, "get_weather", fake_get_weather)

    ms._cache.clear()

    return store
