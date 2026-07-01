"""
Stores and retrieves home configuration.
Each home registers once with its hardware config.
Models, state files, and weather cache are all keyed by home_id.
"""

import json
import os

REGISTRY_PATH = "homes/registry.json"
os.makedirs("homes", exist_ok=True)


def _load_registry() -> dict:
    if not os.path.exists(REGISTRY_PATH):
        return {}
    with open(REGISTRY_PATH, "r") as f:
        return json.load(f)


def _save_registry(registry: dict):
    with open(REGISTRY_PATH, "w") as f:
        json.dump(registry, f, indent=2)


def register_home(config: dict) -> dict:
    """
    Registers or updates a home's configuration.
    config must contain: home_id, lat, lon, battery_type,
                         nominal_voltage, battery_capacity_wh
    """
    registry = _load_registry()
    home_id  = config["home_id"]
    registry[home_id] = config
    _save_registry(registry)
    return config


def get_home(home_id: str) -> dict | None:
    registry = _load_registry()
    return registry.get(home_id)


def list_homes() -> list[str]:
    return list(_load_registry().keys())


def home_exists(home_id: str) -> bool:
    return home_id in _load_registry()
