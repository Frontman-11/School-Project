import json
import os
import requests
from datetime import datetime, timezone
from dotenv import load_dotenv

load_dotenv()

WEATHER_CACHE_DIR   = "model_state/weather_cache"
OPENWEATHER_API_KEY = os.getenv("OPENWEATHER_API_KEY")
CACHE_TTL_MINUTES   = 60

os.makedirs(WEATHER_CACHE_DIR, exist_ok=True)


def _cache_path(home_id):
    return f"{WEATHER_CACHE_DIR}/{home_id}_weather.json"


def _load_cache(home_id):
    path = _cache_path(home_id)
    if not os.path.exists(path):
        return None
    with open(path, "r") as f:
        cache = json.load(f)
    fetched_at  = datetime.fromisoformat(cache["fetched_at"])
    age_minutes = (datetime.now(timezone.utc) - fetched_at).total_seconds() / 60
    if age_minutes > CACHE_TTL_MINUTES:
        return None
    return cache["data"]


def _save_cache(home_id, data):
    with open(_cache_path(home_id), "w") as f:
        json.dump({
            "fetched_at": datetime.now(timezone.utc).isoformat(),
            "data":       data,
        }, f)


def get_weather(home_id, lat, lon):
    """
    Returns current weather for the home location.
    Cached for 60 minutes to avoid burning API quota.
    Falls back to neutral defaults if API is unreachable.
    """
    cached = _load_cache(home_id)
    if cached:
        return cached

    try:
        url = (
            f"https://api.openweathermap.org/data/2.5/weather"
            f"?lat={lat}&lon={lon}&appid={OPENWEATHER_API_KEY}&units=metric"
        )
        resp = requests.get(url, timeout=5)
        resp.raise_for_status()
        raw  = resp.json()

        data = {
            "cloud_cover_pct":    raw["clouds"]["all"],
            "ambient_temp_c":     raw["main"]["temp"],
            "precipitation_prob": 1.0 if raw.get("rain") else 0.0,
            "weather_condition":  raw["weather"][0]["main"],
        }
        _save_cache(home_id, data)
        return data

    except Exception as e:
        print(f"[Weather] API failed for {home_id}: {e}. Using neutral defaults.")
        return {
            "cloud_cover_pct":    50.0,
            "ambient_temp_c":     30.0,
            "precipitation_prob": 0.0,
            "weather_condition":  "Unknown",
        }
