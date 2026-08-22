"""
Parses the actual nested JSON format the ESP firmware sends, and
flattens it into the internal flat dict shape the physics/ML pipeline
has always used. This is the single place that understands the wire
format, so if the firmware format changes again, only this file needs
to change.

Example payload the ESP sends:
{
  "device_id": "solar_ems_001",
  "timestamp": "2026-08-21T09:35:00Z",
  "solar": {"voltage_v": 0.00, "current_a": 0.89, "power_w": 0.00, "shunt_mv": 0.00},
  "battery": {"voltage_v": 0.00, "current_a": 0.10, "power_w": 0.00, "shunt_mv": 0.00, "charging": true},
  "load": {"current_a": 27.85},
  "temperature_c": -127.00,
  "interval_s": 300
}

device_id becomes home_id everywhere downstream. battery.current_a is
treated as an unsigned magnitude, with battery.charging supplying the
sign (positive = charging, negative = discharging) -- this is safe even
if the firmware later starts sending a signed value directly, since
abs(x) * sign(x) == x.

temperature_c of -127 is the standard DS18B20 disconnected-sensor error
code, not a real reading. Any temperature outside a physically plausible
range is treated as invalid: it is replaced with the last known good
temperature for that home (falls back to 30C if none exists yet), and
the reading is tagged temperature_valid=False so it can be surfaced to
the app and to the anomaly-monitoring logic instead of silently
poisoning model training.
"""

from datetime import datetime, timezone
from typing import Optional
from pydantic import BaseModel


class SolarBlock(BaseModel):
    voltage_v: float
    current_a: float
    power_w: Optional[float] = None
    shunt_mv: Optional[float] = None


class BatteryBlock(BaseModel):
    voltage_v: float
    current_a: float
    power_w: Optional[float] = None
    shunt_mv: Optional[float] = None
    charging: bool = True


class LoadBlock(BaseModel):
    current_a: float


class ESPPayload(BaseModel):
    device_id: str
    timestamp: Optional[str] = None
    solar: SolarBlock
    battery: BatteryBlock
    load: LoadBlock
    temperature_c: float
    interval_s: Optional[int] = 300


TEMP_MIN_VALID = -40.0   # coldest plausible reading (ambient or panel)
TEMP_MAX_VALID = 95.0    # hottest plausible panel surface temperature


def _normalize_timestamp(ts: Optional[str]) -> str:
    if not ts:
        return datetime.now(timezone.utc).isoformat()
    # datetime.fromisoformat on Python < 3.11 chokes on a trailing "Z"
    if ts.endswith("Z"):
        ts = ts[:-1] + "+00:00"
    return ts


def flatten_esp_payload(payload: ESPPayload, fallback_temp: Optional[float] = None) -> dict:
    battery_current = abs(payload.battery.current_a) * (1 if payload.battery.charging else -1)

    raw_temp = payload.temperature_c
    temperature_valid = TEMP_MIN_VALID <= raw_temp <= TEMP_MAX_VALID
    temperature = raw_temp if temperature_valid else (
        fallback_temp if fallback_temp is not None else 30.0
    )

    return {
        "home_id": payload.device_id,
        "recorded_at": _normalize_timestamp(payload.timestamp),
        "solar_voltage": payload.solar.voltage_v,
        "solar_current": payload.solar.current_a,
        "battery_voltage": payload.battery.voltage_v,
        "battery_current": battery_current,
        "load_current": payload.load.current_a,
        "temperature": temperature,
        "temperature_valid": temperature_valid,
        "temperature_raw": raw_temp,
        "interval_s": payload.interval_s or 300,
    }
