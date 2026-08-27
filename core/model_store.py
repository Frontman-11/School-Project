"""
Loads and saves the River model objects for each home, backed by
InfluxDB instead of local disk. This is what makes model state survive
a server restart or redeploy -- the previous local-pickle-file approach
lost everything every time the process restarted, since the filesystem
on Render is not persistent.

An in-memory cache sits in front of InfluxDB so a running process does
not deserialize a model on every single request; every call to
save_models() still writes through to InfluxDB immediately, so the
cache is purely a speed optimisation and never the only copy of the
truth. A cold process (fresh restart, or a home never seen by this
process before) transparently loads from InfluxDB on first use.
"""

import base64
import pickle
from river import ensemble, preprocessing, compose
from db.influx_client import save_model_blob, load_model_blob, ModelStoreUnavailable

FIVE_MIN_MODEL_NAMES = ["solar_5min", "load_5min", "soc_5min"]
HOURLY_MODEL_NAMES   = ["solar_hourly", "load_hourly"]
DAILY_MODEL_NAMES    = ["solar_daily", "load_daily"]

# home_id -> {model_name: model_object}
_cache: dict[str, dict[str, object]] = {}


def _make_model():
    return compose.Pipeline(
        preprocessing.StandardScaler(),
        ensemble.SRPRegressor(seed=42)
    )


def _encode(model) -> str:
    return base64.b64encode(pickle.dumps(model)).decode("ascii")


def _decode(blob: str):
    return pickle.loads(base64.b64decode(blob.encode("ascii")))


def _load_one(home_id: str, name: str):
    """
    Returns the stored model, or a fresh one if none exists yet.

    Propagates ModelStoreUnavailable rather than swallowing it. That is
    the whole point of this function: if the read failed we do not know
    whether a trained model exists, and returning a fresh one here would
    cause the caller to train a single sample into it and save it back
    over days of accumulated learning. Letting the exception through
    abandons the cycle instead, which costs one reading of training and
    keeps everything that came before it.

    A blob that is present but will not deserialise is a different case.
    The stored bytes are unusable and no amount of waiting will change
    that, so a fresh model is the only way forward.
    """
    blob = load_model_blob(home_id, name)
    if blob:
        try:
            return _decode(blob)
        except Exception as e:
            print(f"[ModelStore] Failed to decode {name} for {home_id}, starting fresh: {e}")
    return _make_model()


def _get_bundle(home_id: str, names: list[str]) -> dict:
    """
    Loads the named models, using the in-memory cache where possible.

    Nothing is written into the cache unless every requested model
    loaded cleanly. A partially populated cache would let a later call
    silently succeed with a blank model for the one that failed, which
    is the same data loss this module now exists to prevent.
    """
    bundle = _cache.setdefault(home_id, {})
    missing = [name for name in names if name not in bundle]
    if missing:
        loaded = {name: _load_one(home_id, name) for name in missing}
        bundle.update(loaded)
    return {name: bundle[name] for name in names}


def get_five_min_models(home_id: str) -> dict:
    return _get_bundle(home_id, FIVE_MIN_MODEL_NAMES)


def get_hourly_models(home_id: str) -> dict:
    return _get_bundle(home_id, HOURLY_MODEL_NAMES)


def get_daily_models(home_id: str) -> dict:
    return _get_bundle(home_id, DAILY_MODEL_NAMES)


def save_models(home_id: str, models: dict):
    """models: dict of {model_name: model_object}. Writes each one
    through to InfluxDB and updates the in-memory cache."""
    bundle = _cache.setdefault(home_id, {})
    for name, model in models.items():
        save_model_blob(home_id, name, _encode(model))
        bundle[name] = model


def clear_home_cache(home_id: str):
    """Called after a home is deleted, so a running process does not
    keep serving stale in-memory models for an id whose InfluxDB
    record has just been wiped."""
    _cache.pop(home_id, None)
