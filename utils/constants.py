import math

VOLTAGE_SOC_CURVE = {
    "LEAD_ACID": {
        "12V": [
            (12.70, 1.00),
            (12.50, 0.90),
            (12.42, 0.80),
            (12.32, 0.70),
            (12.20, 0.60),
            (12.06, 0.50),
            (11.90, 0.40),
            (11.75, 0.30),
            (11.58, 0.20),
            (11.31, 0.10),
            (10.50, 0.00),
        ],
        "24V": [
            (25.40, 1.00),
            (25.00, 0.90),
            (24.84, 0.80),
            (24.64, 0.70),
            (24.40, 0.60),
            (24.12, 0.50),
            (23.80, 0.40),
            (23.50, 0.30),
            (23.16, 0.20),
            (22.62, 0.10),
            (21.00, 0.00),
        ],
        "48V": [
            (50.80, 1.00),
            (50.00, 0.90),
            (49.68, 0.80),
            (49.28, 0.70),
            (48.80, 0.60),
            (48.24, 0.50),
            (47.60, 0.40),
            (47.00, 0.30),
            (46.32, 0.20),
            (45.24, 0.10),
            (42.00, 0.00),
        ],
    },
    "LIFEPO4": {
        "12V": [
            (13.60, 1.00),
            (13.40, 0.90),
            (13.30, 0.80),
            (13.20, 0.70),
            (13.10, 0.60),
            (13.00, 0.50),
            (12.90, 0.40),
            (12.80, 0.30),
            (12.50, 0.20),
            (12.00, 0.10),
            (10.00, 0.00),
        ],
        "24V": [
            (27.20, 1.00),
            (26.80, 0.90),
            (26.60, 0.80),
            (26.40, 0.70),
            (26.20, 0.60),
            (26.00, 0.50),
            (25.80, 0.40),
            (25.60, 0.30),
            (25.00, 0.20),
            (24.00, 0.10),
            (20.00, 0.00),
        ],
        "48V": [
            (54.40, 1.00),
            (53.60, 0.90),
            (53.20, 0.80),
            (52.80, 0.70),
            (52.40, 0.60),
            (52.00, 0.50),
            (51.60, 0.40),
            (51.20, 0.30),
            (50.00, 0.20),
            (48.00, 0.10),
            (40.00, 0.00),
        ],
    },
}


def encode_hour(hour: int) -> dict:
    return {
        "hour_sin": math.sin(2 * math.pi * hour / 24),
        "hour_cos": math.cos(2 * math.pi * hour / 24),
    }


def encode_day(day: int) -> dict:
    return {
        "day_sin": math.sin(2 * math.pi * day / 7),
        "day_cos": math.cos(2 * math.pi * day / 7),
    }


def encode_month(month: int) -> dict:
    return {
        "month_sin": math.sin(2 * math.pi * month / 12),
        "month_cos": math.cos(2 * math.pi * month / 12),
    }


# ── AC load-side assumptions ───────────────────────────────────────
#
# The load-side SCT-013-050 clamp measures AC RMS current only -- it
# has no voltage or phase reference. Without a dedicated AC voltage
# sensing circuit (e.g. a ZMPT101B synced to the current sampling),
# active/real power (P = V x I x cos(phi)) cannot be measured, since
# the household's power factor is unknown. Only apparent power
# (S = V_rms x I_rms) is computed, using this fixed nominal voltage in
# place of a real measurement. No power factor is assumed anywhere.
NOMINAL_AC_VOLTAGE_V = 240.0  # Nigeria mains, RMS, 50Hz


# ── State of charge estimator bounds ───────────────────────────────
#
# Maximum tolerated divergence between the Coulomb-counted estimate and
# the voltage-based estimate before the counter is re-anchored to the
# voltage curve.
#
# This was 0.3, and that value produced a trapped estimator in
# deployment. The Coulomb integration is clamped to a maximum of 1.0, so
# once it saturates at full charge it can only come down again if the
# re-anchoring test fires. With the voltage estimate near 0.79 and the
# counter at 1.00 the divergence is 0.21 -- large enough to be plainly
# wrong, but never large enough to trigger a 0.3 threshold. The reported
# state of charge sat at 100.0% for 286 consecutive readings as a result.
SOC_REANCHOR_THRESHOLD = 0.12

# A battery with little current flowing either way has a terminal
# voltage that reflects its true state of charge, so at those moments
# the counter is re-anchored unconditionally regardless of divergence.
# This gives the estimator a way back to truth that does not depend on
# the drift threshold firing at all.
SOC_REST_CURRENT_A = 0.25


# ── Runtime estimate bounds ────────────────────────────────────────
#
# Runtime is stored energy divided by discharge power. Below this
# discharge rate the quotient is dominated by the noise floor of the
# current measurement rather than by any real drain, and the result is
# not meaningful -- values above 56,000 hours were produced on a 2.5 kWh
# battery before this bound existed.
MIN_DISCHARGE_FOR_RUNTIME_W = 5.0

# No residential runtime estimate beyond three days carries information
# a household can act on, so the figure is capped rather than reported.
MAX_RUNTIME_HOURS = 72.0


# ── Forecast plausibility bounds ───────────────────────────────────
#
# Every other sanity check in the system tests only for values at or
# below zero, so a regressor that diverges upward passes through
# unchecked. The hourly solar model once wrote 6.87e13 W, which was
# stored and served to the app. These bounds are set well above anything
# the installation can physically produce or draw, so they reject
# divergence without ever clipping a genuine forecast.
MAX_PLAUSIBLE_SOLAR_W = 1000.0
MAX_PLAUSIBLE_LOAD_VA = 2000.0
