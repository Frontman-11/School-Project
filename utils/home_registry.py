"""
Stores and retrieves home configuration.
Each home registers once with its hardware config.
Models, state files, and weather cache are all keyed by home_id.

Backend: InfluxDB Cloud (persistent across Render restarts).
"""

from db.influx_client import write_home_config, get_home_config, list_home_ids


def register_home(config: dict) -> dict:
    """
    Registers or updates a home's configuration.
    config must contain: home_id, lat, lon, battery_type,
                         nominal_voltage, battery_capacity_wh
    """
    write_home_config(config)
    return config


def get_home(home_id: str) -> dict | None:
    return get_home_config(home_id)


def list_homes() -> list[str]:
    return list_home_ids()


def home_exists(home_id: str) -> bool:
    return get_home(home_id) is not None
