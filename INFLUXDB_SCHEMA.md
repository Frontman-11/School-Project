# InfluxDB structure

Single bucket. Every measurement is tagged by `home_id` (== the ESP's
`device_id`), so everything about one device can be found, aggregated,
or deleted by filtering on that one tag.

## Measurements

### `home_config`
Frontend-supplied (or auto-provisioned placeholder) device configuration.
| field | type | notes |
|---|---|---|
| lat, lon | float | |
| battery_type | string | `LEAD_ACID` or `LIFEPO4` |
| nominal_voltage | string | `12V`, `24V`, `48V` |
| battery_capacity_wh | float | |
| configured | bool | `false` until the frontend explicitly registers real values |
| created_at | string (ISO) | |

Written by: `POST /homes/register`, or auto-provisioned on first `/ingest` for an unseen device.

### `sensor_reading`
Raw ESP measurements, one point per ingest.
| field | type |
|---|---|
| solar_voltage, solar_current | float |
| battery_voltage, battery_current | float (battery_current signed: + charging, − discharging) |
| load_current | float |
| temperature | float (corrected value used by the models) |
| temperature_valid | bool |
| temperature_raw | float (what the sensor actually reported, fault codes included) |
| interval_s | int |

Written by: `POST /ingest`, every time the device publishes.

### `model_prediction`
5-minute-ahead prediction, one point per ingest, paired 1:1 with `sensor_reading`.
| field | type |
|---|---|
| solar_power_now_w, load_power_now_w | float (exact physics, V×I) |
| soc_now_percent | float |
| solar_next_w, load_next_w | float (5-min-ahead ML forecast) |
| runtime_hours | float |
| soc_physics_pct, soc_coulomb_pct | float (debug/diagnostic) |
| cloud_cover_pct, ambient_temp_c, precipitation_prob, weather_condition | weather snapshot at prediction time |

### `hourly_forecast` / `daily_forecast`
One point per scheduler run (hourly / midnight).
| field | type |
|---|---|
| forecast_for | string (ISO timestamp the forecast is valid for) |
| solar_next_h_w, load_next_h_w | float — hourly only |
| solar_tomorrow_wh, load_tomorrow_w | float — daily only |

### `model_state`
The actual trained River models, base64-pickled. Tagged by `home_id` **and** `model_name`.
| model_name values |
|---|
| `solar_5min`, `load_5min`, `soc_5min` |
| `solar_hourly`, `load_hourly` |
| `solar_daily`, `load_daily` |

| field | type |
|---|---|
| payload | string (base64-encoded pickle) |
| updated_at | string (ISO) |

Written every time `train()` runs, at a **fixed sentinel timestamp**
(2000-01-01), so each save genuinely replaces the previous one for
that `(home_id, model_name)` pair rather than accumulating a new point
per training step.

### `pipeline_state`
The bridge state needed to pair "features computed at T" with "actual
value that arrived at T+5min" for training. Tagged by `home_id` **and**
`kind` (`five_min`, `hourly`, `daily`).
| field | type |
|---|---|
| state_json | string (JSON blob) |

Same fixed-timestamp overwrite behaviour as `model_state`.

## Deleting a device

`DELETE /homes/{home_id}` removes every point across every measurement
above that carries that `home_id` tag, in one call (the delete
predicate filters only on the tag, not the measurement). If the same
device later sends new data, it starts completely fresh.
