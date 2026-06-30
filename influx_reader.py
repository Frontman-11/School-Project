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


def get_aggregate(home_id: str, range_str: str) -> dict | None:
    """
    Queries model_prediction for mean values over a time window.

    range_str examples: "-1h", "-24h", "-30d"
    Returns a dict of field means or None if no data found.
    """
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
        print(f"[InfluxDB] Aggregate query error: {e}")
        return None


def get_temperature_mean(home_id: str, range_str: str) -> float:
    """
    Gets mean panel temperature from sensor_reading over the range.
    Falls back to 30.0 if unavailable.
    """
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
        print(f"[InfluxDB] Temperature query error: {e}")
    return 30.0


def write_forecast(measurement: str, home_id: str, fields: dict, forecast_for: str):
    """
    Writes a forecast point to InfluxDB.

    measurement: "hourly_forecast", "daily_forecast", or "monthly_forecast"
    fields: dict of field_name → value to write
    forecast_for: ISO timestamp string of when the forecast is valid
    """
    point = Point(measurement).tag("home_id", home_id)
    point = point.field("forecast_for", forecast_for)
    for key, value in fields.items():
        if key == "forecast_for":
            continue
        if isinstance(value, str):
            point = point.field(key, value)
        else:
            point = point.field(key, float(value))
    point = point.time(datetime.now(timezone.utc), WritePrecision.S)
    _write_api.write(bucket=INFLUX_BUCKET, org=INFLUX_ORG, record=point)
