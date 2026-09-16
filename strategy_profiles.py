"""Production strategy profiles selected from deterministic OOS validation.

USDJPY keeps the baseline. XAUUSD defaults to the pair-specific ADX25 plus
faster-target profile after seven rolling 180-day OOS windows. The profile can
still be overridden explicitly with FOREX_XAU_PROFILE=baseline.
"""
from __future__ import annotations

import os
from dataclasses import replace

from technical_engine import StrategyConfig

BASELINE = StrategyConfig()

# Pair-specific candidate validated against seven rolling 180-day OOS windows.
# It uses the baseline risk model with TP1=1.25R, TP2=2.0R and a hard 4H
# ADX>=25 entry filter. Numeric decisions remain deterministic.
XAUUSD_ADX25_FAST_TARGET = replace(
    BASELINE,
    target1_r=1.25,
    target2_r=2.0,
)

XAUUSD_PROFILES = {
    "baseline": BASELINE,
    "adx25_faster_target": XAUUSD_ADX25_FAST_TARGET,
}


def production_profile(symbol: str) -> str:
    """Return the selected production profile for a symbol."""
    symbol = symbol.upper()
    if symbol == "XAUUSD":
        requested = os.environ.get(
            "FOREX_XAU_PROFILE", "adx25_faster_target"
        ).strip().lower()
        if requested in XAUUSD_PROFILES:
            return requested
    return "baseline"


def production_config(symbol: str) -> StrategyConfig:
    """Return the deterministic numeric config for the selected profile."""
    if symbol.upper() == "XAUUSD":
        return XAUUSD_PROFILES[production_profile(symbol)]
    return BASELINE


def production_adx_gate(symbol: str) -> float | None:
    """Return the optional production ADX gate for the selected profile."""
    if (
        symbol.upper() == "XAUUSD"
        and production_profile(symbol) == "adx25_faster_target"
    ):
        return 25.0
    return None
