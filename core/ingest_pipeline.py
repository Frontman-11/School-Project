"""
The glue between "a reading landed in InfluxDB" and "the ML pipeline
processed it" -- deliberately its own module, separate from both the
pure I/O layer (db.influx_client) and the pure ML logic
(core.physics_and_models), so the two paths only ever talk to each
other through the database, never through a shared in-memory object.

This means: the ESP -> InfluxDB write always succeeds or fails on its
own, and never depends on the ML step. If the ML step is slow, broken,
or throws, the sensor reading that triggered it is already durable in
InfluxDB by the time this function is even called.
"""

import time
from datetime import datetime, timedelta, timezone
from utils.home_registry import get_home
from db.influx_client import get_latest_sensor, write_model_prediction
from core.physics_and_models import train, predict

# InfluxDB Cloud writes are usually visible to a subsequent read within
# a few hundred milliseconds, but that's not a hard real-time guarantee.
# A retry loop absorbs that gap instead of failing outright on the
# (rare) occasion the read happens before the write has propagated.
_READ_RETRY_ATTEMPTS = 15
_READ_RETRY_DELAY_SECONDS = 0.5


def _as_utc_datetime(iso_str: str) -> datetime:
    dt = datetime.fromisoformat(iso_str.replace("Z", "+00:00"))
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt


def process_reading_for_ml(home_id: str, expected_recorded_at: str) -> dict:
    """
    Reads the just-persisted sensor reading back out of InfluxDB (never
    off any in-memory object from the request that wrote it), combines
    it with the device's registered config, trains the 5-minute models,
    predicts the next 5 minutes, and writes the prediction back to
    InfluxDB. Returns the prediction dict.

    Raises if no config exists, or if the expected reading never shows
    up in InfluxDB within the retry window -- in both cases the caller
    (api.ingest) already wrote the raw reading beforehand, so nothing
    is lost even if this raises.
    """
    config = get_home(home_id)
    if not config:
        raise RuntimeError(f"no config found for '{home_id}'")

    expected_dt = _as_utc_datetime(expected_recorded_at).replace(microsecond=0)

    # get_latest_sensor's default range only looks back 24 hours, which
    # is right for a live dashboard but wrong here: the reading we're
    # confirming might be a backdated one from a historical backfill
    # (days in the past), not a live one from just now. Search a window
    # that starts slightly before the reading's own timestamp instead of
    # before "now", so this works correctly for both live and backdated
    # data.
    search_start = (expected_dt - timedelta(minutes=10)).strftime("%Y-%m-%dT%H:%M:%SZ")

    sensor = None
    attempt_log = []
    for attempt in range(_READ_RETRY_ATTEMPTS):
        sensor = get_latest_sensor(home_id, range_str=search_start)
        if sensor and "recorded_at" in sensor:
            sensor_dt = _as_utc_datetime(sensor["recorded_at"]).replace(microsecond=0)
            attempt_log.append(
                f"attempt {attempt}: found recorded_at={sensor_dt.isoformat()} (need >= {expected_dt.isoformat()})"
            )
            if sensor_dt >= expected_dt:
                break
        else:
            attempt_log.append(
                f"attempt {attempt}: get_latest_sensor returned nothing at all"
            )
        time.sleep(_READ_RETRY_DELAY_SECONDS)
    else:
        # loop completed without a break -- log everything we saw
        print(
            f"[ingest_pipeline] DIAGNOSTIC for '{home_id}', expected_recorded_at={expected_recorded_at}:"
        )
        for line in attempt_log:
            print(f"  {line}")

    if not sensor or "recorded_at" not in sensor:
        raise RuntimeError(
            f"sensor reading for '{home_id}' not yet visible in InfluxDB "
            f"after {_READ_RETRY_ATTEMPTS} attempts over "
            f"{_READ_RETRY_ATTEMPTS * _READ_RETRY_DELAY_SECONDS:.1f}s. "
            f"get_latest_sensor returned: {sensor!r}. "
            f"expected_recorded_at={expected_recorded_at}"
        )

    data = {
        "home_id": home_id,
        "recorded_at": sensor["recorded_at"],
        "solar_voltage": sensor["solar_voltage"],
        "solar_current": sensor["solar_current"],
        "battery_voltage": sensor["battery_voltage"],
        "battery_current": sensor["battery_current"],
        "load_current": sensor["load_current"],
        "temperature": sensor["temperature"],
        "lat": config["lat"],
        "lon": config["lon"],
        "battery_type": config.get("battery_type", "LEAD_ACID"),
        "nominal_voltage": config.get("nominal_voltage", "12V"),
        "battery_capacity_wh": config.get("battery_capacity_wh", 100),
    }

    drift = train(data)
    result = predict(data)
    if drift:
        result["drift"] = drift

    write_model_prediction(result, home_id, datetime.fromisoformat(data["recorded_at"]))
    return result
