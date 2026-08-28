import json
import os
import time
from datetime import datetime, timezone, timedelta
from dotenv import load_dotenv
from influxdb_client import InfluxDBClient, Point, WritePrecision
from influxdb_client.client.write_api import SYNCHRONOUS
from utils.constants import NOMINAL_AC_VOLTAGE_V

load_dotenv()

INFLUX_URL = os.getenv("INFLUX_URL")
INFLUX_TOKEN = os.getenv("INFLUX_TOKEN")
INFLUX_ORG = os.getenv("INFLUX_ORG")
INFLUX_BUCKET = os.getenv("INFLUX_BUCKET")

_client = InfluxDBClient(url=INFLUX_URL, token=INFLUX_TOKEN, org=INFLUX_ORG)
_query_api = _client.query_api()
_write_api = _client.write_api(write_options=SYNCHRONOUS)

# Truthy environment values only. Testing the raw os.getenv result meant
# any non-empty string switched debug logging ON, including "0" and
# "false", so the flag could be set but never cleared.
_DEBUG_TRUE = {"1", "true", "yes", "on"}

# Reading a model blob can fail transiently. Retry before giving up,
# because giving up costs a whole training cycle.
_MODEL_READ_ATTEMPTS = 3
_MODEL_READ_RETRY_SECONDS = 0.75

# Search windows for the newest model_state point, narrowest first. A
# save happens on every reading, so the first window matches in normal
# operation and the query never has to scan the weeks of superseded
# blobs sitting behind it. The wider windows exist only for a process
# resuming after a long outage.
_MODEL_READ_WINDOWS = ["-2h", "-24h", "-7d", "-35d"]


def _debug_enabled() -> bool:
    return os.getenv("DEBUG_INFLUX_QUERIES", "").strip().lower() in _DEBUG_TRUE


class ModelStoreUnavailable(Exception):
    """
    Raised when the stored model state could not be READ -- a query
    error, a timeout, a 5xx from InfluxDB.

    This is deliberately distinct from load_model_blob() returning None,
    which means "there is genuinely no model stored yet". Collapsing the
    two is how training gets destroyed: a read failure that looks like
    "no model yet" causes the caller to build a fresh untrained model,
    train one sample into it, and save it back over a model that had
    been learning for days. The stored copy is fine; only our ability to
    read it failed, so the correct response is to abandon this cycle and
    leave the stored copy alone.
    """


# InfluxDB's hard ceiling is 64KB (65536 bytes) per string field. Base64
# text is 1 byte per character, so this leaves comfortable headroom.
_MODEL_CHUNK_SIZE = 60000


# ── Logical delete via a generation counter ────────────────────────
#
# InfluxDB Cloud Serverless (v3) buckets do not support the delete API
# at all -- confirmed directly from the API: "Deletes ranges are not
# supported for serverless v3 buckets". Since bytes cannot actually be
# removed, "deleting" a home is implemented as a logical cutoff.
#
# An earlier version of this compared timestamps (a "deleted before
# time X" marker against each point's own recorded_at). That broke the
# moment any *backdated* data was written after a delete -- exactly
# what the seed/backfill script does on purpose, writing readings
# timestamped days in the past. Those points looked, to a
# timestamp-based filter, like they predated the delete, even though
# they were written moments after it.
#
# A generation counter has no such problem: every write is tagged with
# whatever the device's *current* generation number is at write time
# (irrespective of what recorded_at value the point claims), and every
# read only considers points tagged with the current generation.
# Deleting a home just increments the counter -- nothing about what
# timestamp any past or future data represents matters at all.

# home_id -> current generation (int). Avoids an extra InfluxDB query
# on every single read/write after the first lookup for that home.
_generation_cache: dict[str, int] = {}


def _get_current_generation(home_id: str) -> int:
    if home_id in _generation_cache:
        if _debug_enabled():
            print(
                f"[InfluxDB DEBUG] _get_current_generation({home_id}) cache hit -> {_generation_cache[home_id]}"
            )
        return _generation_cache[home_id]

    query = f'''
    from(bucket: "{INFLUX_BUCKET}")
      |> range(start: -365d)
      |> filter(fn: (r) => r._measurement == "home_generation")
      |> filter(fn: (r) => r.home_id == "{home_id}")
      |> filter(fn: (r) => r._field == "generation")
      |> last()
    '''
    gen = 0
    try:
        tables = _query_api.query(query)
        for table in tables:
            for record in table.records:
                v = record.get_value()
                if v is not None:
                    gen = int(v)
    except Exception as e:
        print(f"[InfluxDB] _get_current_generation error: {e}")
    _generation_cache[home_id] = gen
    if _debug_enabled():
        print(
            f"[InfluxDB DEBUG] _get_current_generation({home_id}) queried InfluxDB -> {gen}"
        )
    return gen


def delete_home_data(home_id: str):
    new_gen = _get_current_generation(home_id) + 1
    point = (
        Point("home_generation")
        .tag("home_id", home_id)
        .field("generation", new_gen)
        .time(datetime.now(timezone.utc), WritePrecision.S)
    )
    _write_api.write(bucket=INFLUX_BUCKET, org=INFLUX_ORG, record=point)
    _generation_cache[home_id] = new_gen


# ── Sensor + prediction writes ─────────────────────────────────────


def write_sensor_reading(payload: dict, recorded_at: datetime):
    gen = _get_current_generation(payload["home_id"])
    if _debug_enabled():
        print(
            f"[InfluxDB DEBUG] write_sensor_reading(home_id={payload['home_id']}, generation={gen}, recorded_at={recorded_at.isoformat()})"
        )
    point = (
        Point("sensor_reading")
        .tag("home_id", payload["home_id"])
        .tag("generation", str(gen))
        .tag("battery_type", payload.get("battery_type", "LEAD_ACID"))
        .field("solar_voltage", float(payload["solar_voltage"]))
        .field("solar_current", float(payload["solar_current"]))
        .field("battery_voltage", float(payload["battery_voltage"]))
        .field("battery_current", float(payload["battery_current"]))
        .field("load_current", float(payload["load_current"]))
        .field("temperature", float(payload["temperature"]))
        .field("temperature_valid", bool(payload.get("temperature_valid", True)))
        .field(
            "temperature_raw",
            float(payload.get("temperature_raw", payload["temperature"])),
        )
        .field("interval_s", int(payload.get("interval_s", 300)))
        .time(recorded_at, WritePrecision.S)
    )
    _write_api.write(bucket=INFLUX_BUCKET, org=INFLUX_ORG, record=point)


def write_model_prediction(result: dict, home_id: str, recorded_at: datetime):
    gen = _get_current_generation(home_id)
    weather = result.get("weather", {})
    point = (
        Point("model_prediction")
        .tag("home_id", home_id)
        .tag("generation", str(gen))
        .field("forecast_for", result["forecast_for"])
        .field("solar_power_now_w", float(result["solar_power_now_w"]))
        .field("load_power_now_w", float(result["load_power_now_w"]))
        .field("soc_now_percent", float(result["soc_now_percent"]))
        .field("solar_next_w", float(result["solar_next_w"]))
        .field("load_next_w", float(result["load_next_w"]))
        .field(
            "runtime_hours",
            float(result["runtime_hours"])
            if result.get("runtime_hours") is not None
            else -1.0,  # sentinel: not discharging, runtime undefined
        )
        .field("soc_physics_pct", float(result["soc_physics_pct"]))
        .field("soc_coulomb_pct", float(result["soc_coulomb_pct"]))
        .field("cloud_cover_pct", float(weather.get("cloud_cover_pct", 50.0)))
        .field("ambient_temp_c", float(weather.get("ambient_temp_c", 30.0)))
        .field("precipitation_prob", float(weather.get("precipitation_prob", 0.0)))
        .field("weather_condition", str(weather.get("weather_condition", "Unknown")))
    )
    drift = result.get("drift")
    if drift:
        point = point.field("solar_error_w", float(drift["solar_error_w"]))
        point = point.field("load_error_w", float(drift["load_error_w"]))
        if drift.get("solar_abs_pct_error") is not None:
            point = point.field(
                "solar_abs_pct_error", float(drift["solar_abs_pct_error"])
            )
        if drift.get("load_abs_pct_error") is not None:
            point = point.field(
                "load_abs_pct_error", float(drift["load_abs_pct_error"])
            )
    point = point.time(recorded_at, WritePrecision.S)
    _write_api.write(bucket=INFLUX_BUCKET, org=INFLUX_ORG, record=point)


def write_forecast(measurement: str, home_id: str, fields: dict, forecast_for: str):
    gen = _get_current_generation(home_id)
    point = Point(measurement).tag("home_id", home_id).tag("generation", str(gen))
    point = point.field("forecast_for", forecast_for)
    for key, value in fields.items():
        if key == "forecast_for":
            continue
        point = point.field(key, value if isinstance(value, str) else float(value))
    point = point.time(datetime.now(timezone.utc), WritePrecision.S)
    _write_api.write(bucket=INFLUX_BUCKET, org=INFLUX_ORG, record=point)


# ── Read helpers ──────────────────────────────────────────────────

LATEST_RANGE = os.getenv("LATEST_RANGE", "-24h")


def _last_fields(
    measurement: str, home_id: str, range_str: str = LATEST_RANGE
) -> dict | None:
    gen = _get_current_generation(home_id)
    query = f'''
    from(bucket: "{INFLUX_BUCKET}")
      |> range(start: {range_str})
      |> filter(fn: (r) => r._measurement == "{measurement}")
      |> filter(fn: (r) => r.home_id == "{home_id}")
      |> filter(fn: (r) => r.generation == "{gen}")
      |> last()
    '''
    try:
        tables = _query_api.query(query)
        result: dict = {}
        recorded_time = None
        for table in tables:
            for record in table.records:
                if record.get_value() is not None:
                    result[record.get_field()] = record.get_value()
                    if recorded_time is None:
                        recorded_time = record.get_time()
        if recorded_time is not None:
            result["recorded_at"] = recorded_time.isoformat()
        if _debug_enabled():
            print(
                f"[InfluxDB DEBUG] _last_fields({measurement}, home_id={home_id}, generation={gen}): "
                f"{len(tables)} table(s), result={result if result else None}"
            )
        return result if result else None
    except Exception as e:
        print(f"[InfluxDB] _last_fields({measurement}) error: {e}")
        return None


def get_last_valid_temperature(home_id: str) -> float | None:
    """
    Used as the fallback when a fresh ESP reading has a faulty
    temperature (e.g. the -127 DS18B20 disconnect code). Deliberately
    reads from sensor_reading -- the raw data of record -- rather than
    from any ML-internal state, since "what was the last real
    temperature this device reported" is a fact about the sensor, not
    about the model.
    """
    gen = _get_current_generation(home_id)
    query = f'''
    from(bucket: "{INFLUX_BUCKET}")
      |> range(start: -35d)
      |> filter(fn: (r) => r._measurement == "sensor_reading")
      |> filter(fn: (r) => r.home_id == "{home_id}")
      |> filter(fn: (r) => r.generation == "{gen}")
      |> filter(fn: (r) => r._field == "temperature" or r._field == "temperature_valid")
      |> pivot(rowKey: ["_time"], columnKey: ["_field"], valueColumn: "_value")
      |> filter(fn: (r) => r.temperature_valid == true)
      |> keep(columns: ["_time", "temperature"])
      |> rename(columns: {{temperature: "_value"}})
      |> last()
    '''
    try:
        tables = _query_api.query(query)
        for table in tables:
            for record in table.records:
                v = record.get_value()
                if v is not None:
                    return float(v)
    except Exception as e:
        print(f"[InfluxDB] get_last_valid_temperature error: {e}")
    return None


def get_latest_prediction(home_id: str) -> dict | None:
    return _last_fields("model_prediction", home_id)


def get_latest_sensor(home_id: str, range_str: str = LATEST_RANGE) -> dict | None:
    return _last_fields("sensor_reading", home_id, range_str=range_str)


def get_aggregate(home_id: str, range_str: str) -> dict | None:
    gen = _get_current_generation(home_id)
    fields = [
        "solar_power_now_w",
        "load_power_now_w",
        "soc_now_percent",
        "cloud_cover_pct",
    ]
    field_filter = " or ".join([f'r._field == "{f}"' for f in fields])
    query = f'''
    from(bucket: "{INFLUX_BUCKET}")
      |> range(start: {range_str})
      |> filter(fn: (r) => r._measurement == "model_prediction")
      |> filter(fn: (r) => r.home_id == "{home_id}")
      |> filter(fn: (r) => r.generation == "{gen}")
      |> filter(fn: (r) => {field_filter})
      |> mean()
    '''
    try:
        tables = _query_api.query(query)
        result: dict = {}
        for table in tables:
            for record in table.records:
                if record.get_value() is not None:
                    result[record.get_field()] = record.get_value()
        return result if result else None
    except Exception as e:
        print(f"[InfluxDB] aggregate error: {e}")
        return None


def get_temperature_mean(home_id: str, range_str: str) -> float:
    """
    Excludes readings where temperature_valid is False (e.g. a
    disconnected DS18B20 reporting -127) from the mean. temperature
    and temperature_valid are written as separate field rows, so they
    have to be pivoted into the same row before the valid/invalid
    filter can apply -- filtering on r.temperature_valid before the
    pivot would silently match nothing, since that column does not
    exist yet at that point in the pipeline.
    """
    gen = _get_current_generation(home_id)
    query = f'''
    from(bucket: "{INFLUX_BUCKET}")
      |> range(start: {range_str})
      |> filter(fn: (r) => r._measurement == "sensor_reading")
      |> filter(fn: (r) => r.home_id == "{home_id}")
      |> filter(fn: (r) => r.generation == "{gen}")
      |> filter(fn: (r) => r._field == "temperature" or r._field == "temperature_valid")
      |> pivot(rowKey: ["_time"], columnKey: ["_field"], valueColumn: "_value")
      |> filter(fn: (r) => r.temperature_valid == true)
      |> keep(columns: ["_time", "temperature"])
      |> rename(columns: {{temperature: "_value"}})
      |> mean()
    '''
    try:
        tables = _query_api.query(query)
        for table in tables:
            for record in table.records:
                if record.get_value() is not None:
                    return float(record.get_value())
    except Exception as e:
        print(f"[InfluxDB] temperature_mean error: {e}")
    return 30.0


def get_latest_forecast(home_id: str, measurement: str) -> dict | None:
    return _last_fields(measurement, home_id, range_str="-48h")


def get_hourly_history(home_id: str, hours: int = 24) -> list[dict]:
    """
    Hourly averages of actual vs predicted solar/load power over the last N hours.
    Powers the app's History screen (chart + table). Also usable for "as at
    yesterday" / "hours ago" lookback by increasing `hours`.
    """
    gen = _get_current_generation(home_id)
    range_str = "0" if hours <= 0 else f"-{hours}h"

    def _run(query: str) -> dict[str, float]:
        try:
            tables = _query_api.query(query)
            out: dict[str, float] = {}
            for table in tables:
                for record in table.records:
                    v = record.get_value()
                    if v is not None:
                        key = record.get_time().strftime("%Y-%m-%d %H:00")
                        out[key] = round(float(v), 1)
            return out
        except Exception as e:
            print(f"[InfluxDB] history error: {e}")
            return {}

    solar_actual = _run(f'''
        from(bucket: "{INFLUX_BUCKET}")
          |> range(start: {range_str})
          |> filter(fn: (r) => r._measurement == "sensor_reading" and r.home_id == "{home_id}" and r.generation == "{gen}")
          |> filter(fn: (r) => r._field == "solar_voltage" or r._field == "solar_current")
          |> pivot(rowKey: ["_time"], columnKey: ["_field"], valueColumn: "_value")
          |> map(fn: (r) => ({{ r with _value: r.solar_voltage * r.solar_current }}))
          |> aggregateWindow(every: 1h, fn: mean, createEmpty: false)
    ''')

    load_actual = _run(f'''
        import "math"
        from(bucket: "{INFLUX_BUCKET}")
          |> range(start: {range_str})
          |> filter(fn: (r) => r._measurement == "sensor_reading" and r.home_id == "{home_id}" and r.generation == "{gen}")
          |> filter(fn: (r) => r._field == "battery_voltage" or r._field == "load_current")
          |> pivot(rowKey: ["_time"], columnKey: ["_field"], valueColumn: "_value")
          |> map(fn: (r) => ({{ r with _value: math.abs(x: r.battery_voltage * r.load_current) }}))
          |> aggregateWindow(every: 1h, fn: mean, createEmpty: false)
    ''')

    solar_pred = _run(f'''
        from(bucket: "{INFLUX_BUCKET}")
          |> range(start: {range_str})
          |> filter(fn: (r) => r._measurement == "model_prediction" and r.home_id == "{home_id}" and r.generation == "{gen}")
          |> filter(fn: (r) => r._field == "solar_power_now_w")
          |> aggregateWindow(every: 1h, fn: mean, createEmpty: false)
    ''')

    load_pred = _run(f'''
        from(bucket: "{INFLUX_BUCKET}")
          |> range(start: {range_str})
          |> filter(fn: (r) => r._measurement == "model_prediction" and r.home_id == "{home_id}" and r.generation == "{gen}")
          |> filter(fn: (r) => r._field == "load_power_now_w")
          |> aggregateWindow(every: 1h, fn: mean, createEmpty: false)
    ''')

    all_hours = sorted(
        set(solar_actual) | set(solar_pred) | set(load_actual) | set(load_pred)
    )
    return [
        {
            "time": h,
            "solar_actual": solar_actual.get(h),
            "solar_pred": solar_pred.get(h),
            "load_actual": load_actual.get(h),
            "load_pred": load_pred.get(h),
        }
        for h in all_hours
    ]


# ── Home config persistence ────────────────────────────────────────


def write_home_config(config: dict):
    gen = _get_current_generation(config["home_id"])
    point = (
        Point("home_config")
        .tag("home_id", config["home_id"])
        .tag("generation", str(gen))
        .field("lat", float(config["lat"]))
        .field("lon", float(config["lon"]))
        .field("battery_type", config.get("battery_type", "LEAD_ACID"))
        .field("nominal_voltage", config.get("nominal_voltage", "12V"))
        .field("battery_capacity_wh", float(config.get("battery_capacity_wh", 100)))
        .field("configured", bool(config.get("configured", False)))
        .field(
            "created_at",
            config.get("created_at", datetime.now(timezone.utc).isoformat()),
        )
        .time(datetime.now(timezone.utc), WritePrecision.S)
    )
    _write_api.write(bucket=INFLUX_BUCKET, org=INFLUX_ORG, record=point)


def get_home_config(home_id: str) -> dict | None:
    gen = _get_current_generation(home_id)
    query = f'''
    from(bucket: "{INFLUX_BUCKET}")
      |> range(start: -365d)
      |> filter(fn: (r) => r._measurement == "home_config")
      |> filter(fn: (r) => r.home_id == "{home_id}")
      |> filter(fn: (r) => r.generation == "{gen}")
      |> last()
    '''
    try:
        tables = _query_api.query(query)
        result: dict = {"home_id": home_id}
        for table in tables:
            for record in table.records:
                if record.get_value() is not None:
                    result[record.get_field()] = record.get_value()
        return result if len(result) > 1 else None
    except Exception as e:
        print(f"[InfluxDB] get_home_config error: {e}")
        return None


def list_home_ids() -> list[str]:
    """
    schema.tagValues sees the home_id tag on any point ever written,
    including ones from a previous (now-deleted) generation -- so each
    candidate is re-checked through get_home_config (which is itself
    generation-aware) and only kept if it still resolves to a live
    config under the current generation.
    """
    query = f'''
    import "influxdata/influxdb/schema"
    schema.tagValues(
      bucket: "{INFLUX_BUCKET}",
      tag: "home_id",
      predicate: (r) => r._measurement == "home_config",
      start: -365d,
    )
    '''
    try:
        tables = _query_api.query(query)
        candidates = []
        for table in tables:
            for record in table.records:
                val = record.get_value()
                if val and val != "_value":
                    candidates.append(val)
        return [hid for hid in candidates if get_home_config(hid) is not None]
    except Exception as e:
        print(f"[InfluxDB] list_home_ids error: {e}")
        return []


# ── Model state persistence (River models, base64-pickled) ────────


def save_model_blob(home_id: str, model_name: str, blob: str):
    """
    InfluxDB rejects any single string field over 64KB, and a trained
    River SRPRegressor's pickled, base64-encoded size grows past that
    within a few hundred training cycles -- this was hit for real
    during testing. The blob is therefore split into fixed-size chunks,
    each stored as its own field (chunk_0, chunk_1, ...), well under
    the 64KB ceiling, with chunk_count recorded so load_model_blob
    knows how many fields to read back and reassemble. This scales to
    arbitrarily large models instead of failing again once a model
    grows a bit more.
    """
    gen = _get_current_generation(home_id)
    chunks = [
        blob[i : i + _MODEL_CHUNK_SIZE] for i in range(0, len(blob), _MODEL_CHUNK_SIZE)
    ] or [""]

    point = (
        Point("model_state")
        .tag("home_id", home_id)
        .tag("generation", str(gen))
        .tag("model_name", model_name)
        .field("chunk_count", len(chunks))
        .field("updated_at", datetime.now(timezone.utc).isoformat())
    )
    for i, chunk in enumerate(chunks):
        point = point.field(f"chunk_{i}", chunk)
    point = point.time(datetime.now(timezone.utc), WritePrecision.S)
    _write_api.write(bucket=INFLUX_BUCKET, org=INFLUX_ORG, record=point)


def _fetch_model_field(home_id: str, model_name: str, gen: int,
                       window: str, field: str):
    """
    Newest value of ONE field, with its timestamp.

    Restricting the query to a single field is the entire point. Asking
    for every field at once returns the complete serialised model in one
    response, and model_state is the only measurement in this system
    holding large string fields -- every other last() call in the app
    reads floats and has never failed. Pulling one field at a time keeps
    each response to a single chunk.
    """
    query = f'''
    from(bucket: "{INFLUX_BUCKET}")
      |> range(start: {window})
      |> filter(fn: (r) => r._measurement == "model_state")
      |> filter(fn: (r) => r.home_id == "{home_id}" and r.model_name == "{model_name}")
      |> filter(fn: (r) => r.generation == "{gen}")
      |> filter(fn: (r) => r._field == "{field}")
      |> last()
    '''
    for table in _query_api.query(query):
        for record in table.records:
            v = record.get_value()
            if v is not None:
                return v, record.get_time()
    return None, None


def _fetch_model_field_at(home_id: str, model_name: str, gen: int,
                          field: str, at):
    """
    One field, pinned to the exact save it belongs to.

    Each chunk is fetched in its own query, so without pinning the time
    two chunks could come from different saves and reassemble into a
    blob that never existed. A one-second window either side of the
    chunk_count timestamp keeps every chunk from the same write.
    """
    start = (at - timedelta(seconds=1)).isoformat().replace("+00:00", "Z")
    stop = (at + timedelta(seconds=1)).isoformat().replace("+00:00", "Z")
    query = f'''
    from(bucket: "{INFLUX_BUCKET}")
      |> range(start: {start}, stop: {stop})
      |> filter(fn: (r) => r._measurement == "model_state")
      |> filter(fn: (r) => r.home_id == "{home_id}" and r.model_name == "{model_name}")
      |> filter(fn: (r) => r.generation == "{gen}")
      |> filter(fn: (r) => r._field == "{field}")
      |> last()
    '''
    for table in _query_api.query(query):
        for record in table.records:
            v = record.get_value()
            if v is not None:
                return v
    return None


def load_model_blob(home_id: str, model_name: str) -> str | None:
    """
    Returns the stored blob, or None if no model has been saved yet.

    Raises ModelStoreUnavailable if the read itself failed. The caller
    must treat that as "unknown", never as "absent" -- see the exception
    docstring.

    A blob that is present but unusable (an interrupted save that left a
    chunk missing) returns None rather than raising. That state cannot
    be recovered by waiting, since nothing will repair it until a new
    save succeeds, so starting fresh is the only way out. A read error
    is the opposite: the stored copy is very likely intact and will be
    readable on the next attempt.

    The read is done one field at a time. Requesting every field in a
    single query returns the whole serialised model in one response and
    made InfluxDB return 500s across every time window once the models
    had grown. Fetching chunk_count first and then each chunk
    individually keeps each response small, at the cost of one query per
    chunk. At a five-minute reading interval that cost is irrelevant.

    Windows are searched narrowest-first because save_model_blob writes
    a NEW point on every save, so model_state accumulates thousands of
    superseded blobs behind the current one.
    """
    gen = _get_current_generation(home_id)
    last_error = None
    read_failed = False

    for window in _MODEL_READ_WINDOWS:
        try:
            count_value, saved_at = _fetch_model_field(
                home_id, model_name, gen, window, "chunk_count"
            )
        except Exception as e:
            last_error = e
            read_failed = True
            print(
                f"[InfluxDB] load_model_blob({model_name}) chunk_count read "
                f"failed over {window}: {e}"
            )
            continue

        if count_value is None:
            continue  # nothing stored in this window; widen the search

        chunk_count = int(count_value)
        parts = []
        for i in range(chunk_count):
            key = f"chunk_{i}"
            chunk = None
            for attempt in range(_MODEL_READ_ATTEMPTS):
                try:
                    chunk = _fetch_model_field_at(
                        home_id, model_name, gen, key, saved_at
                    )
                    break
                except Exception as e:
                    last_error = e
                    read_failed = True
                    print(
                        f"[InfluxDB] load_model_blob({model_name}) {key} read "
                        f"failed (attempt {attempt + 1}/{_MODEL_READ_ATTEMPTS}): {e}"
                    )
                    if attempt + 1 < _MODEL_READ_ATTEMPTS:
                        time.sleep(_MODEL_READ_RETRY_SECONDS * (attempt + 1))
            else:
                # every attempt on this chunk errored -- the blob is
                # intact in the database, we simply could not read it
                raise ModelStoreUnavailable(
                    f"could not read {key} of model '{model_name}' for "
                    f"'{home_id}': {last_error}"
                )

            if chunk is None:
                print(
                    f"[InfluxDB] load_model_blob({model_name}) missing {key} -- partial save, discarding"
                )
                return None
            parts.append(chunk)

        return "".join(parts)

    # Every window came back clean but empty: no model has been saved.
    if not read_failed:
        return None

    raise ModelStoreUnavailable(
        f"could not read model '{model_name}' for '{home_id}' over "
        f"{len(_MODEL_READ_WINDOWS)} time windows: {last_error}"
    )


# ── Pipeline (train/predict pairing) state persistence ─────────────


def save_pipeline_state(home_id: str, kind: str, state: dict):
    gen = _get_current_generation(home_id)
    point = (
        Point("pipeline_state")
        .tag("home_id", home_id)
        .tag("generation", str(gen))
        .tag("kind", kind)
        .field("state_json", json.dumps(state))
        .time(datetime.now(timezone.utc), WritePrecision.S)
    )
    _write_api.write(bucket=INFLUX_BUCKET, org=INFLUX_ORG, record=point)


def load_pipeline_state(home_id: str, kind: str) -> dict | None:
    gen = _get_current_generation(home_id)
    query = f'''
    from(bucket: "{INFLUX_BUCKET}")
      |> range(start: -35d)
      |> filter(fn: (r) => r._measurement == "pipeline_state")
      |> filter(fn: (r) => r.home_id == "{home_id}" and r.kind == "{kind}")
      |> filter(fn: (r) => r.generation == "{gen}")
      |> filter(fn: (r) => r._field == "state_json")
      |> last()
    '''
    try:
        tables = _query_api.query(query)
        for table in tables:
            for record in table.records:
                v = record.get_value()
                if v is not None:
                    return json.loads(v)
        return None
    except Exception as e:
        print(f"[InfluxDB] load_pipeline_state({kind}) error: {e}")
        return None


def get_history_log(home_id: str, hours: int = 24) -> list[dict]:
    """
    Per-reading history, joining sensor_reading and model_prediction at
    matching timestamps. No extra measurement needed -- both are already
    written on every ingest.

    solar_actual and load_actual are recomputed here with exactly the
    same formulas compute_physics() uses, so the history screen and the
    live dashboard can never disagree about what "actual" means:
      solar  = panel_voltage x panel_current            (W, DC, exact)
      load   = NOMINAL_AC_VOLTAGE_V x load_current      (VA, apparent)
    An earlier version of this used battery_voltage x load_current for
    the load figure, which mixes the DC battery rail with an AC-side
    current reading -- it produced numbers roughly 18x too small and
    disagreed with load_power_now_w everywhere else in the system.
    """
    gen = _get_current_generation(home_id)
    range_str = "0" if hours <= 0 else f"-{hours}h"

    def _query(fields: str, measurement: str) -> dict[str, dict]:
        query = f'''
        from(bucket: "{INFLUX_BUCKET}")
          |> range(start: {range_str})
          |> filter(fn: (r) => r._measurement == "{measurement}")
          |> filter(fn: (r) => r.home_id == "{home_id}")
          |> filter(fn: (r) => r.generation == "{gen}")
          |> filter(fn: (r) => {fields})
          |> pivot(rowKey: ["_time"], columnKey: ["_field"], valueColumn: "_value")
        '''
        try:
            tables = _query_api.query(query)
            out: dict[str, dict] = {}
            for table in tables:
                for record in table.records:
                    ts = record.get_time().strftime("%Y-%m-%d %H:%M")
                    out[ts] = record.values
            return out
        except Exception as e:
            print(f"[InfluxDB] get_history_log query error: {e}")
            return {}

    sensor_rows = _query(
        'r._field == "solar_voltage" or r._field == "solar_current" or r._field == "load_current"',
        "sensor_reading",
    )
    pred_rows = _query(
        'r._field == "solar_next_w" or r._field == "load_next_w"',
        "model_prediction",
    )

    all_times = sorted(set(sensor_rows) | set(pred_rows))
    entries: list[dict] = []
    for ts in all_times:
        s_row = sensor_rows.get(ts, {})
        p_row = pred_rows.get(ts, {})
        solar_v = s_row.get("solar_voltage")
        solar_c = s_row.get("solar_current")
        load_c = s_row.get("load_current")
        solar_actual = (
            round(solar_v * solar_c, 2)
            if solar_v is not None and solar_c is not None
            else None
        )
        load_actual = (
            round(NOMINAL_AC_VOLTAGE_V * load_c, 2) if load_c is not None else None
        )
        entries.append({
            "time": ts,
            "solar_actual": solar_actual,
            "solar_pred": p_row.get("solar_next_w"),
            "load_actual": load_actual,
            "load_pred": p_row.get("load_next_w"),
        })
    return entries
