# Solar Energy Management System — ML & Cloud Pipeline

## What changed in this version

1. **Model state now lives in InfluxDB, not local disk.** Render's
   filesystem is ephemeral, so the previous pickle-files-on-disk
   approach lost all training progress on every restart. Every model
   is loaded from InfluxDB on first use and written back after every
   `learn_one` call.
2. **Everything is scoped by `home_id`** (== the ESP's `device_id`):
   sensor data, predictions, forecasts, model state, and device config
   all carry a `home_id` tag, and a single `DELETE /homes/{home_id}`
   wipes all of it in one call.
3. **The ESP payload parser matches the real firmware format** — nested
   `solar` / `battery` / `load` blocks, `device_id`, signed battery
   current derived from `current_a` + `charging`, and a fault-code
   check for `temperature_c == -127` (disconnected DS18B20).
4. **The Coulomb-counting SOC formula had a real unit bug** (dividing
   amp-hours by a watt-hour capacity) — fixed to integrate in
   watt-hours throughout.
5. **The monthly forecast model was removed** to reduce complexity —
   only 5-minute, hourly, and daily horizons remain.
6. **Weather is returned in every `/ingest` response** and snapshotted
   onto the paired `model_prediction` point, but is not persisted as
   its own long-term series — see `INFLUXDB_SCHEMA.md`.

See `INFLUXDB_SCHEMA.md` for the full measurement/field layout.

## Folder structure
```
├── api.py                     FastAPI — single entry point, MQTT + scheduler run inside it
├── seed_sample_home.py        Backfills mock ESP-format data for "sample_home"
├── core/
│   ├── model_store.py         InfluxDB-backed River model load/save (with in-memory cache)
│   ├── physics_and_models.py  5-min per-home physics + models
│   └── forecast_models.py     Hourly/daily per-home models
├── db/
│   └── influx_client.py       All InfluxDB reads/writes (sensor, prediction, forecast,
│                               model_state, pipeline_state, home_config, delete)
├── utils/
│   ├── constants.py           Voltage-SOC curves + time encoding
│   ├── weather.py             OpenWeatherMap, in-memory cache
│   ├── esp_payload.py         Parses the real nested ESP JSON shape
│   └── home_registry.py       Thin wrapper over db.influx_client's home_config functions
├── .env / .env.example
└── requirements.txt
```

## Data flow

1. ESP32 publishes to HiveMQ, in its own schedule — nothing waits on the frontend.
2. The MQTT listener (background thread inside `api.py`) receives it instantly
   and forwards the raw JSON to `POST /ingest` over loopback.
3. `/ingest` auto-provisions a placeholder `home_config` if the `device_id`
   has never been seen before, so ingestion never blocks on registration.
4. The frontend can call `POST /homes/register` at any time (before or after
   step 1) to attach the real location, battery type, and capacity.
5. `/ingest` trains the 5-minute models on the previous reading, predicts
   the next 5 minutes, and writes both the raw reading and the prediction
   to InfluxDB.
6. A background scheduler (also inside `api.py`) runs the hourly forecast
   every hour and the daily forecast at midnight, for every registered home.
7. The frontend polls `GET /current`, `/history`, `/forecast/*` — it never
   triggers training or prediction itself; that already happened automatically
   in step 5.

## Running

```bash
pip install -r requirements.txt
python3 api.py
```

Visit `http://localhost:8000/docs` for interactive API docs.

## Populating test data

With the API running in one terminal:
```bash
python3 seed_sample_home.py --days 3
```

This registers a device called `sample_home`, posts several days of
realistic mock readings through the real `/ingest` endpoint (in the
actual nested ESP JSON shape), and fires the hourly/daily forecast
jobs at the appropriate simulated boundaries — so `/current/sample_home`,
`/forecast/hourly/sample_home`, and `/forecast/daily/sample_home` all
have real, trained output to show on the app immediately.

When the real hardware is ready, just point the frontend at the real
device's `device_id` instead of `sample_home`.

## Deleting a device's data (development only)

```bash
curl -X DELETE http://localhost:8000/homes/{home_id} -H "X-API-Key: <your key>"
```

Wipes every InfluxDB record tagged with that `home_id`. If the same
device sends new data afterwards, it starts completely fresh.

## API Key

Every request needs the `X-API-Key` header, value from `.env` as `API_KEY`.

## Units — read this before wiring up the frontend

Field names were kept exactly as they already are in the DB and API responses (all still end in `_w`), but **not every one of them is actually watts**. This matters because mixing them up silently produces numbers that are wrong by a factor of 240, not just slightly off.

| Field | Appears in | Actual unit | Why |
|---|---|---|---|
| `solar_power_now_w` | `/ingest`, `/current` | **Watts (W)**, exact | DC side: `panel_voltage × panel_current`, both directly measured. No ambiguity. |
| `solar_next_w` | `/ingest`, `/current` | **Watts (W)** | ML forecast of the above, same unit. |
| `battery_discharge_power_w` | `/ingest`, `/current` | **Watts (W)**, exact | DC side: `battery_voltage × |battery_current|` from the INA226, directly measured. |
| `load_power_now_w` | `/ingest`, `/current`, `/averages` | **Volt-Amps (VA)**, not Watts | AC side: `240V (assumed nominal mains) × load_current (measured RMS)`. The SCT-013-030 clamp has no voltage or phase reference, so this is apparent power, not active power. No power factor is applied anywhere — there was deliberately no attempt to estimate active power. |
| `load_next_w` | `/ingest`, `/current` | **Volt-Amps (VA)**, not Watts | ML forecast of the above, same caveat. |
| `load_next_h_w` | `/forecast/hourly` | **Volt-Amps (VA)**, not Watts | Hourly forecast of the same AC-side quantity. |
| `load_tomorrow_w` | `/forecast/daily` | **Volt-Amps (VA)**, not Watts | Daily forecast of the same AC-side quantity. |
| `load_error_w`, `load_predicted_w`, `load_actual_w` | `/ingest` (`drift`) | **Volt-Amps (VA)**, not Watts | Error/prediction/actual for the load figure above — same unit as what it's measuring. |
| `solar_error_w`, `solar_predicted_w`, `solar_actual_w` | `/ingest` (`drift`) | **Watts (W)**, exact | Same unit as `solar_power_now_w`. |
| `runtime_hours` | `/ingest`, `/current` | Hours | Derived from `battery_discharge_power_w`, unaffected by the VA/W distinction above. |
| `soc_now_percent`, `soc_physics_pct`, `soc_coulomb_pct` | everywhere | Percent (0–100) | |
| `solar_abs_pct_error`, `load_abs_pct_error` | `/ingest` (`drift`) | Percent | |

**Rule of thumb for the frontend: anything with `solar_` or `battery_` in the name is Watts and exact. Anything with `load_` in the name (except `load_current` itself, which is amps) is VA, not Watts — because there is currently no way to measure true active power without adding a voltage-and-phase reference circuit that isn't in the hardware.**