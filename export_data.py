"""
Exports everything in the InfluxDB bucket to CSV for offline evaluation.

Drop this file in School-Project/ next to api.py and run it inside the
venv:

    source venv/bin/activate
    python3 export_data.py

Writes into ./export/ :
    sensor_reading.csv      raw measurements, one row per reading
    model_prediction.csv    derived + forecast values, one row per reading
    hourly_forecast.csv     scheduled hourly forecasts
    daily_forecast.csv      scheduled daily forecasts
    home_config.csv         registered installation parameters
    evaluation.csv          forecast at t paired with the actual at t+1

Model blobs (model_state) and pipeline_state are skipped on purpose:
they are large serialised objects, not measurements.
"""

import os
import csv
from pathlib import Path
from datetime import datetime

from dotenv import load_dotenv
from influxdb_client import InfluxDBClient

load_dotenv()

URL = os.getenv("INFLUX_URL")
TOKEN = os.getenv("INFLUX_TOKEN")
ORG = os.getenv("INFLUX_ORG")
BUCKET = os.getenv("INFLUX_BUCKET")

OUT = Path("export")
OUT.mkdir(exist_ok=True)

# Everything, as far back as retention allows.
MEASUREMENTS = [
    "sensor_reading",
    "model_prediction",
    "hourly_forecast",
    "daily_forecast",
    "home_config",
]

# Nominal AC voltage used to turn load current into apparent power.
# Must match utils/constants.py or the actuals will not line up.
NOMINAL_AC_VOLTAGE_V = 240.0


def fetch(client, measurement):
    """Returns a list of dicts, one per timestamp, fields pivoted wide."""
    query = f'''
    from(bucket: "{BUCKET}")
      |> range(start: 0)
      |> filter(fn: (r) => r._measurement == "{measurement}")
      |> pivot(rowKey: ["_time"], columnKey: ["_field"], valueColumn: "_value")
      |> sort(columns: ["_time"])
    '''
    rows = []
    for table in client.query_api().query(query, org=ORG):
        for record in table.records:
            values = dict(record.values)
            for junk in ("result", "table", "_start", "_stop", "_measurement"):
                values.pop(junk, None)
            time_value = values.pop("_time", None)
            row = {"time": time_value.isoformat() if time_value else ""}
            row.update(values)
            rows.append(row)
    rows.sort(key=lambda r: r["time"])
    return rows


def write_csv(name, rows):
    path = OUT / f"{name}.csv"
    if not rows:
        print(f"  {name}: no data")
        return []
    columns = []
    for row in rows:
        for key in row:
            if key not in columns:
                columns.append(key)
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)
    print(f"  {name}: {len(rows)} rows -> {path}")
    return rows


def build_evaluation(sensor_rows, prediction_rows):
    """
    Pairs each forecast with the actual value that arrived next.

    solar_next_w / load_next_w written at time t are forecasts FOR the
    reading at t+1, so scoring them against the actual at t (which is
    how the History screen displays them) would be wrong. This walks
    the prediction series in order and attaches the actual from the
    following sensor reading.
    """
    by_home = {}
    for row in sensor_rows:
        home = row.get("home_id", "")
        by_home.setdefault(home, []).append(row)

    def actual_solar(row):
        v, i = row.get("solar_voltage"), row.get("solar_current")
        return round(v * i, 3) if v is not None and i is not None else None

    def actual_load(row):
        i = row.get("load_current")
        return round(NOMINAL_AC_VOLTAGE_V * i, 3) if i is not None else None

    out = []
    preds_by_home = {}
    for row in prediction_rows:
        preds_by_home.setdefault(row.get("home_id", ""), []).append(row)

    for home, preds in preds_by_home.items():
        sensors = by_home.get(home, [])
        times = [s["time"] for s in sensors]
        for pred in preds:
            t = pred["time"]
            nxt = None
            for idx, stamp in enumerate(times):
                if stamp > t:
                    nxt = sensors[idx]
                    break
            if nxt is None:
                continue
            out.append({
                "home_id": home,
                "generation": pred.get("generation", ""),
                "predicted_at": t,
                "actual_at": nxt["time"],
                "solar_predicted_w": pred.get("solar_next_w"),
                "solar_actual_w": actual_solar(nxt),
                "load_predicted_va": pred.get("load_next_w"),
                "load_actual_va": actual_load(nxt),
                "solar_physics_at_prediction_w": pred.get("solar_power_now_w"),
                "load_physics_at_prediction_va": pred.get("load_power_now_w"),
                "soc_now_percent": pred.get("soc_now_percent"),
                "soc_physics_pct": pred.get("soc_physics_pct"),
                "soc_coulomb_pct": pred.get("soc_coulomb_pct"),
                "cloud_cover_pct": pred.get("cloud_cover_pct"),
                "ambient_temp_c": pred.get("ambient_temp_c"),
                "weather_condition": pred.get("weather_condition"),
            })
    out.sort(key=lambda r: (r["home_id"], r["predicted_at"]))
    return out


def main():
    missing = [n for n, v in
               [("INFLUX_URL", URL), ("INFLUX_TOKEN", TOKEN),
                ("INFLUX_ORG", ORG), ("INFLUX_BUCKET", BUCKET)] if not v]
    if missing:
        raise SystemExit(f"Missing in .env: {', '.join(missing)}")

    print(f"Connecting to {URL}, org={ORG}, bucket={BUCKET}")
    with InfluxDBClient(url=URL, token=TOKEN, org=ORG, timeout=120_000) as client:
        collected = {}
        for measurement in MEASUREMENTS:
            print(f"Fetching {measurement} ...")
            try:
                collected[measurement] = write_csv(measurement, fetch(client, measurement))
            except Exception as exc:
                print(f"  {measurement}: FAILED ({exc})")
                collected[measurement] = []

    evaluation = build_evaluation(
        collected.get("sensor_reading", []),
        collected.get("model_prediction", []),
    )
    write_csv("evaluation", evaluation)

    sensors = collected.get("sensor_reading", [])
    if sensors:
        homes = sorted({r.get("home_id", "") for r in sensors})
        print()
        print(f"Devices present : {', '.join(homes)}")
        print(f"First reading   : {sensors[0]['time']}")
        print(f"Last reading    : {sensors[-1]['time']}")
        print(f"Total readings  : {len(sensors)}")
        bad = sum(1 for r in sensors if r.get("temperature_valid") is False)
        print(f"Faulty temps    : {bad} of {len(sensors)}")
    print()
    print(f"Done. Files are in {OUT.resolve()}")


if __name__ == "__main__":
    main()
