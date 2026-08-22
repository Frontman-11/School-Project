"""
Weather lookup with an in-memory cache (no disk I/O).

Render's filesystem is ephemeral, so a disk-based cache was pointless
anyway -- it reset on every restart. An in-memory cache with a 60-minute
TTL achieves the same goal (avoid hammering the OpenWeatherMap API) and
is simpler. Weather itself is never persisted long-term in InfluxDB;
a snapshot of it is stored alongside each 5-minute prediction instead,
which is enough to answer "what was the weather when this prediction
was made" without needing a full standalone weather history.
"""

import os
import time
import requests
from dotenv import load_dotenv

load_dotenv()

OPENWEATHER_API_KEY = os.getenv("OPENWEATHER_API_KEY")
CACHE_TTL_SECONDS = 60 * 60

# home_id -> (fetched_at_epoch_seconds, data)
_cache: dict[str, tuple[float, dict]] = {}


def clear_weather_cache(home_id: str) -> None:
    _cache.pop(home_id, None)


def get_weather(home_id: str, lat: float, lon: float) -> dict:
    cached = _cache.get(home_id)
    if cached and (time.time() - cached[0]) < CACHE_TTL_SECONDS:
        return cached[1]

    try:
        url = (
            f"https://api.openweathermap.org/data/2.5/weather"
            f"?lat={lat}&lon={lon}&appid={OPENWEATHER_API_KEY}&units=metric"
        )
        resp = requests.get(url, timeout=5)
        resp.raise_for_status()
        raw = resp.json()
        data = {
            "cloud_cover_pct": raw["clouds"]["all"],
            "ambient_temp_c": raw["main"]["temp"],
            "precipitation_prob": 1.0 if raw.get("rain") else 0.0,
            "weather_condition": raw["weather"][0]["main"],
        }
        _cache[home_id] = (time.time(), data)
        return data
    except Exception as e:
        print(f"[Weather] API failed for {home_id}: {e}. Using neutral defaults.")
        return {
            "cloud_cover_pct": 50.0,
            "ambient_temp_c": 30.0,
            "precipitation_prob": 0.0,
            "weather_condition": "Unknown",
        }
