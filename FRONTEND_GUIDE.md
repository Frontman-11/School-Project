# Frontend Integration Guide — Solar Energy Management API

This document explains how the app should talk to the backend: the
overall data flow, every endpoint you'll actually use, exact
request/response shapes, what the numbers mean, and what's changed
since earlier integration work.

---

## 1. The big picture

```
ESP32 device ──MQTT──► HiveMQ broker ──► backend (auto) ──► InfluxDB
                                              │
                                              ▼
                                     trains + predicts
                                              │
                                              ▼
                                          InfluxDB
                                              │
                                              ▼
                                       Your app ◄── polls the REST API below
```

Two things worth understanding before anything else:

**You never trigger training or prediction.** The moment a device
publishes a reading, the backend automatically writes it, trains the
model, and generates a fresh prediction — all before your app ever
asks for anything. Your app's only job is to **poll and display**
what's already there. There is no "run prediction" button to wire up.

**A device's raw measurement is never lost, even if the ML step has a
problem.** The backend writes the raw sensor reading to the database
first, independently of training/prediction. If something goes wrong
in the ML step, you'll see `"ml_status": "unavailable"` in the
response instead of missing data — see §5.

---

## 2. Authentication

Every request needs this header:

```
X-API-Key: <the key given to you separately>
```

Missing or wrong key → `403 Forbidden`.

---

## 3. Home / device lifecycle

A "home" and a "device" are the same thing — `home_id` in every URL
below is exactly the `device_id` the physical ESP32 sends.

### A device doesn't need to be pre-registered before it shows data

If a device starts sending readings before your app has configured it,
the backend auto-creates a placeholder entry so nothing is lost. You
can tell this happened because the response includes:

```json
"configured": false
```

Once your app calls `POST /homes/register` with the real location and
battery details, this flips to `true`. Show this to the user — e.g. a
banner like "Finish setting up your device" while `configured` is
`false`.

### Register / update a home's configuration

```
POST /homes/register
```
```json
{
  "home_id": "sample_home",
  "lat": 5.5167,
  "lon": 5.75,
  "battery_type": "LEAD_ACID",
  "nominal_voltage": "12V",
  "battery_capacity_wh": 100
}
```
Response:
```json
{
  "message": "Home registered",
  "home": {
    "home_id": "sample_home",
    "lat": 5.5167,
    "lon": 5.75,
    "battery_type": "LEAD_ACID",
    "nominal_voltage": "12V",
    "battery_capacity_wh": 100,
    "configured": true
  }
}
```
This is safe to call again later if the user edits their settings —
it always overwrites the previous config.

### List all homes / get one home's config

```
GET /homes
GET /homes/{home_id}
```

### Delete a home (dev/reset tool — be careful with this one)

```
DELETE /homes/{home_id}
```
Wipes everything the backend knows about that device — sensor
history, predictions, forecasts, model training — as if it had never
existed. **Don't expose this in the production app UI** unless you
genuinely want a "factory reset this device" feature; it's mainly a
development tool. If the same physical device sends new data
afterwards, it starts completely fresh with no memory of before.

---

## 4. Displaying live data

### Current reading + prediction (the main dashboard call)

```
GET /current/{home_id}
```
```json
{
  "home_id": "sample_home",
  "sensor": {
    "solar_voltage": 18.98,
    "solar_current": 3.38,
    "battery_voltage": 12.6,
    "battery_current": 3.11,
    "load_current": 1.98,
    "temperature": 36.4,
    "temperature_valid": true,
    "interval_s": 10,
    "recorded_at": "2026-08-22T14:31:23+00:00"
  },
  "prediction": {
    "recorded_at": "2026-08-22T14:31:23+00:00",
    "forecast_for": "2026-08-22T14:36:23+00:00",
    "solar_power_now_w": 64.15,
    "load_power_now_w": 24.95,
    "battery_discharge_power_w": 39.19,
    "soc_now_percent": 95.8,
    "solar_next_w": 61.19,
    "load_next_w": 25.7,
    "runtime_hours": 3.73,
    "soc_physics_pct": 95.0,
    "soc_coulomb_pct": 96.0,
    "cloud_cover_pct": 90.0,
    "ambient_temp_c": 28.24,
    "weather_condition": "Clouds"
  }
}
```

`sensor` is the raw ESP measurement, exactly as reported.
`prediction` is what the ML pipeline computed from it. Poll this every
5–15 seconds for a live dashboard feel.

If a brand-new device has never sent a single reading yet, this
returns `404`.

### Today's averages

```
GET /averages/{home_id}
```
```json
{
  "home_id": "sample_home",
  "period": "last_24h",
  "averages": {
    "solar_power_now_w": 41.2,
    "load_power_now_w": 210.5,
    "soc_now_percent": 88.3,
    "cloud_cover_pct": 62.0
  }
}
```

### History (for charts)

```
GET /history/{home_id}?hours=24
```
`hours` defaults to 24; pass 48, 168, etc. for a longer look-back.
```json
{
  "home_id": "sample_home",
  "period": "last_24h",
  "hours": [
    {
      "time": "2026-08-22 06:00",
      "solar_actual": 12.4,
      "solar_pred": 11.9,
      "load_actual": 205.1,
      "load_pred": 198.0
    }
  ]
}
```

---

## 5. Forecasts

Three horizons, each backed by a different model:

```
GET /forecast/hourly/{home_id}
GET /forecast/daily/{home_id}
GET /forecast/custom/{home_id}?hours=6      (or ?days=2)
```

Hourly example:
```json
{
  "home_id": "sample_home",
  "forecast": {
    "forecast_for": "2026-08-22T15:30:00+00:00",
    "solar_next_h_w": 55.2,
    "load_next_h_w": 220.0
  }
}
```

Daily example:
```json
{
  "home_id": "sample_home",
  "forecast": {
    "forecast_for": "2026-08-23T00:00:00+00:00",
    "solar_tomorrow_wh": 480.0,
    "load_tomorrow_w": 200.0
  }
}
```

`/forecast/custom` auto-picks hourly or daily depending on the horizon
and adds a `note` field with a plain-language reliability caveat —
show that note to the user rather than hiding it, especially for
longer horizons.

These update on a schedule (hourly on the hour, daily at midnight),
not on every poll — no need to hit these more than once every few
minutes.

---

## 6. Reading `ml_status` and `drift` (new)

Two fields you may not have seen before, both only present in
responses that come from a live device ingest, not from `/current`:

- **`ml_status`**: `"ok"` normally. If it's `"unavailable"`, the raw
  reading was still saved successfully — only the prediction step had
  a transient problem. Don't treat this as an error state for the raw
  data; just show the last known prediction if this shows up.
- **`drift`**: present once a device has sent at least two readings.
  It's the model quietly grading its own last prediction against what
  actually happened:
  ```json
  "drift": {
    "solar_predicted_w": 50.0, "solar_actual_w": 15.0,
    "solar_error_w": -35.0, "solar_abs_pct_error": 233.3,
    "load_predicted_w": 25.0, "load_actual_w": 25.0,
    "load_error_w": 0.0, "load_abs_pct_error": 0.0
  }
  ```
  Optional to show in the UI — could be a nice "model accuracy" widget
  later, but not required for launch.

---

## 7. Units — read this carefully, it's the biggest gotcha

Every field below still ends in `_w` in the API/DB for historical
reasons, but **not all of them are actually watts**:

| Field | Unit | Notes |
|---|---|---|
| `solar_power_now_w`, `solar_next_w` | **Watts (W)**, exact | DC side, directly measured |
| `battery_discharge_power_w` | **Watts (W)**, exact | DC side, directly measured |
| `load_power_now_w`, `load_next_w`, `load_next_h_w`, `load_tomorrow_w`, `load_error_w` | **Volt-Amps (VA)**, not Watts | AC side — measured current × assumed 240V mains, no power factor applied |
| `runtime_hours` | Hours | |
| `soc_now_percent`, `soc_physics_pct`, `soc_coulomb_pct` | Percent | |

**Rule of thumb: anything with `load_` in the name is VA, everything
else with `_w` is genuinely Watts.** Label the units accordingly in
the UI (e.g. "162 VA" for load, "162 W" for solar) — don't just print
"W" everywhere.

---

## 8. Error handling

| Status | Meaning | What to do |
|---|---|---|
| `403` | Bad/missing API key | Check the header is being sent |
| `404` | Home not registered, or no data yet | Show an empty/setup state, not an error toast |
| `400` | Bad query params (e.g. `/forecast/custom` with neither `hours` nor `days`) | Fix the request |
| `500` | Something broke server-side | Retry once; if persistent, that's a backend bug to report |

A `404` from `/current` for a brand-new device is normal and expected
— it just means the ESP hasn't sent its first reading yet. Show a
"waiting for your device..." state rather than an error.

---

## 9. Quick reference — endpoints you'll actually call

| Endpoint | When to call |
|---|---|
| `POST /homes/register` | Once, when the user finishes device setup (or edits it) |
| `GET /current/{home_id}` | Poll every 5–15s for the live dashboard |
| `GET /averages/{home_id}` | Poll every minute or so |
| `GET /history/{home_id}` | On demand, when the user opens a chart/history screen |
| `GET /forecast/hourly/{home_id}` | Poll every few minutes |
| `GET /forecast/daily/{home_id}` | Poll every few minutes |
| `GET /forecast/custom/{home_id}` | On demand, when the user picks a custom time range |

`/docs` on the running API (interactive Swagger UI) is always the
source of truth if anything here looks out of date — this document is
a guide, not a substitute for it.
