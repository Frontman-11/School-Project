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

def get_latest_prediction(home_id: str) -> dict | None:
    query = f'''
    from(bucket: "{INFLUX_BUCKET}")
      |> range(start: -1h)
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
      |> range(start: -1h)
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
