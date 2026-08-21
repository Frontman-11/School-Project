import os
from datetime import datetime, timezone
from dotenv import load_dotenv
from influxdb_client import InfluxDBClient, Point, WritePrecision
from influxdb_client.client.write_api import SYNCHRONOUS

load_dotenv()

INFLUX_URL    = os.getenv("INFLUX_URL")
INFLUX_TOKEN  = os.getenv("INFLUX_TOKEN")
INFLUX_ORG    = os.getenv("INFLUX_ORG")
INFLUX_BUCKET = os.getenv("INFLUX_BUCKET")

_client    = InfluxDBClient(url=INFLUX_URL, token=INFLUX_TOKEN, org=INFLUX_ORG)
_query_api = _client.query_api()
_write_api = _client.write_api(write_options=SYNCHRONOUS)


# ── Write helpers ─────────────────────────────────────────────────

def write_sensor_reading(payload: dict, recorded_at: datetime):
    point = (
        Point("sensor_reading")
        .tag("home_id",      payload["home_id"])
        .tag("battery_type", payload["battery_type"])
        .field("solar_voltage",   float(payload["solar_voltage"]))
        .field("solar_current",   float(payload["solar_current"]))
        .field("battery_voltage", float(payload["battery_voltage"]))
        .field("battery_current", float(payload["battery_current"]))
        .field("load_current",    float(payload["load_current"]))
        .field("temperature",     float(payload["temperature"]))
        .time(recorded_at, WritePrecision.S)
    )
    _write_api.write(bucket=INFLUX_BUCKET, org=INFLUX_ORG, record=point)


def write_model_prediction(result: dict, home_id: str, recorded_at: datetime):
    point = (
        Point("model_prediction")
        .tag("home_id", home_id)
        .field("forecast_for",      result["forecast_for"])
        .field("solar_power_now_w", float(result["solar_power_now_w"]))
        .field("load_power_now_w",  float(result["load_power_now_w"]))
        .field("soc_now_percent",   float(result["soc_now_percent"]))
        .field("solar_next_w",      float(result["solar_next_w"]))
        .field("load_next_w",       float(result["load_next_w"]))
        .field("runtime_hours",     float(result["runtime_hours"]))
        .field("cloud_cover_pct",   float(result["cloud_cover_pct"]))
        .field("weather_condition", result["weather_condition"])
        .field("soc_physics_pct",   float(result["soc_physics_pct"]))
        .field("soc_coulomb_pct",   float(result["soc_coulomb_pct"]))
        .time(recorded_at, WritePrecision.S)
    )
    _write_api.write(bucket=INFLUX_BUCKET, org=INFLUX_ORG, record=point)


def write_forecast(measurement: str, home_id: str, fields: dict, forecast_for: str):
    point = Point(measurement).tag("home_id", home_id)
    point = point.field("forecast_for", forecast_for)
    for key, value in fields.items():
        if key == "forecast_for":
            continue
        point = point.field(key, value if isinstance(value, str) else float(value))
    point = point.time(datetime.now(timezone.utc), WritePrecision.S)
    _write_api.write(bucket=INFLUX_BUCKET, org=INFLUX_ORG, record=point)


# ── Read helpers ──────────────────────────────────────────────────

LATEST_RANGE = os.getenv("LATEST_RANGE", "-24h")


def get_latest_prediction(home_id: str) -> dict | None:
    query = f'''
    from(bucket: "{INFLUX_BUCKET}")
      |> range(start: {LATEST_RANGE})
      |> filter(fn: (r) => r._measurement == "model_prediction")
      |> filter(fn: (r) => r.home_id == "{home_id}")
      |> last()
    '''
    try:
        tables = _query_api.query(query)
        result = {}
        for table in tables:
            for record in table.records:
                if record.get_value() is not None:
                    result[record.get_field()] = record.get_value()
        return result if result else None
    except Exception as e:
        print(f"[InfluxDB] latest_prediction error: {e}")
        return None


def get_latest_sensor(home_id: str) -> dict | None:
    query = f'''
    from(bucket: "{INFLUX_BUCKET}")
      |> range(start: {LATEST_RANGE})
      |> filter(fn: (r) => r._measurement == "sensor_reading")
      |> filter(fn: (r) => r.home_id == "{home_id}")
      |> last()
    '''
    try:
        tables = _query_api.query(query)
        result = {}
        for table in tables:
            for record in table.records:
                if record.get_value() is not None:
                    result[record.get_field()] = record.get_value()
        return result if result else None
    except Exception as e:
        print(f"[InfluxDB] latest_sensor error: {e}")
        return None


def get_aggregate(home_id: str, range_str: str) -> dict | None:
    fields       = ["solar_power_now_w", "load_power_now_w", "soc_now_percent", "cloud_cover_pct"]
    field_filter = " or ".join([f'r._field == "{f}"' for f in fields])
    query = f'''
    from(bucket: "{INFLUX_BUCKET}")
      |> range(start: {range_str})
      |> filter(fn: (r) => r._measurement == "model_prediction")
      |> filter(fn: (r) => r.home_id == "{home_id}")
      |> filter(fn: (r) => {field_filter})
      |> mean()
    '''
    try:
        tables = _query_api.query(query)
        result = {}
        for table in tables:
            for record in table.records:
                if record.get_value() is not None:
                    result[record.get_field()] = record.get_value()
        return result if result else None
    except Exception as e:
        print(f"[InfluxDB] aggregate error: {e}")
        return None


def get_temperature_mean(home_id: str, range_str: str) -> float:
    query = f'''
    from(bucket: "{INFLUX_BUCKET}")
      |> range(start: {range_str})
      |> filter(fn: (r) => r._measurement == "sensor_reading")
      |> filter(fn: (r) => r.home_id == "{home_id}")
      |> filter(fn: (r) => r._field == "temperature")
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
    query = f'''
    from(bucket: "{INFLUX_BUCKET}")
      |> range(start: -48h)
      |> filter(fn: (r) => r._measurement == "{measurement}")
      |> filter(fn: (r) => r.home_id == "{home_id}")
      |> last()
    '''
    try:
        tables = _query_api.query(query)
        result = {}
        for table in tables:
            for record in table.records:
                if record.get_value() is not None:
                    result[record.get_field()] = record.get_value()
        return result if result else None
    except Exception as e:
        print(f"[InfluxDB] latest_forecast error: {e}")
        return None


def get_hourly_history(home_id: str, hours: int = 24) -> list[dict]:
    """
    Hourly averages of actual vs predicted solar/load power over the last N hours.
    Actual power is derived from raw sensor readings (V x I); predictions come
    from the model_prediction measurement written on each ingest.
    Returns rows like: {"time": "14:00", "solar_actual": .., "solar_pred": ..,
                        "load_actual": .., "load_pred": ..} (null when missing).
    """
    range_str = f"-{hours}h"

    def _run(query: str) -> dict[str, float]:
        try:
            tables = _query_api.query(query)
            out: dict[str, float] = {}
            for table in tables:
                for record in table.records:
                    v = record.get_value()
                    if v is not None:
                        key = record.get_time().strftime("%H:00")
                        out[key] = round(float(v), 1)
            return out
        except Exception as e:
            print(f"[InfluxDB] history error: {e}")
            return {}

    solar_actual = _run(f'''
        from(bucket: "{INFLUX_BUCKET}")
          |> range(start: {range_str})
          |> filter(fn: (r) => r._measurement == "sensor_reading" and r.home_id == "{home_id}")
          |> filter(fn: (r) => r._field == "solar_voltage" or r._field == "solar_current")
          |> pivot(rowKey: ["_time"], columnKey: ["_field"], valueColumn: "_value")
          |> map(fn: (r) => ({{ r with value: r.solar_voltage * r.solar_current }}))
          |> aggregateWindow(every: 1h, fn: mean, column: "value", createEmpty: false)
    ''')

    load_actual = _run(f'''
        import "math"
        from(bucket: "{INFLUX_BUCKET}")
          |> range(start: {range_str})
          |> filter(fn: (r) => r._measurement == "sensor_reading" and r.home_id == "{home_id}")
          |> filter(fn: (r) => r._field == "battery_voltage" or r._field == "load_current")
          |> pivot(rowKey: ["_time"], columnKey: ["_field"], valueColumn: "_value")
          |> map(fn: (r) => ({{ r with value: math.abs(x: r.battery_voltage * r.load_current) }}))
          |> aggregateWindow(every: 1h, fn: mean, column: "value", createEmpty: false)
    ''')

    solar_pred = _run(f'''
        from(bucket: "{INFLUX_BUCKET}")
          |> range(start: {range_str})
          |> filter(fn: (r) => r._measurement == "model_prediction" and r.home_id == "{home_id}")
          |> filter(fn: (r) => r._field == "solar_power_now_w")
          |> aggregateWindow(every: 1h, fn: mean, createEmpty: false)
    ''')

    load_pred = _run(f'''
        from(bucket: "{INFLUX_BUCKET}")
          |> range(start: {range_str})
          |> filter(fn: (r) => r._measurement == "model_prediction" and r.home_id == "{home_id}")
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


# ── Home config persistence ──────────────────────────────────────

def write_home_config(config: dict):
    point = (
        Point("home_config")
        .tag("home_id", config["home_id"])
        .field("lat", float(config["lat"]))
        .field("lon", float(config["lon"]))
        .field("battery_type", config.get("battery_type", "LEAD_ACID"))
        .field("nominal_voltage", config.get("nominal_voltage", "12V"))
        .field("battery_capacity_wh", float(config.get("battery_capacity_wh", 100)))
        .time(datetime.now(timezone.utc), WritePrecision.S)
    )
    _write_api.write(bucket=INFLUX_BUCKET, org=INFLUX_ORG, record=point)


def get_home_config(home_id: str) -> dict | None:
    query = f'''
    from(bucket: "{INFLUX_BUCKET}")
      |> range(start: -365d)
      |> filter(fn: (r) => r._measurement == "home_config")
      |> filter(fn: (r) => r.home_id == "{home_id}")
      |> last()
    '''
    try:
        tables = _query_api.query(query)
        result = {"home_id": home_id}
        for table in tables:
            for record in table.records:
                if record.get_value() is not None:
                    result[record.get_field()] = record.get_value()
        return result if len(result) > 1 else None
    except Exception as e:
        print(f"[InfluxDB] get_home_config error: {e}")
        return None


def list_home_ids() -> list[str]:
    query = f'''
    import "influxdata/influxdb/schema"
    from(bucket: "{INFLUX_BUCKET}")
      |> range(start: -365d)
      |> filter(fn: (r) => r._measurement == "home_config")
      |> schema.tagValues(tag: "home_id")
    '''
    try:
        tables = _query_api.query(query)
        ids = []
        for table in tables:
            for record in table.records:
                val = record.get_value()
                if val and val != "_value":
                    ids.append(val)
        return ids
    except Exception as e:
        print(f"[InfluxDB] list_home_ids error: {e}")
        return []
