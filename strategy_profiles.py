"""Production strategy profiles and explicit experimental overrides.

The deterministic baseline remains the default. Pair-specific alternatives are
opt-in so validation results cannot silently change live signals.
"""
from __future__ import annotations

import os
from dataclasses import replace

from technical_engine import StrategyConfig

BASELINE = StrategyConfig()

# Evidence-backed research candidate from rolling OOS screening. It is not the
# default until pair-specific validation is reviewed.
XAUUSD_ADX25_FAST_TARGET = replace(BASELINE, target1_r=1.25, target2_r=2.0)

XAUUSD_PROFILES = {
    "baseline": BASELINE,
    "adx25_faster_target": XAUUSD_ADX25_FAST_TARGET,
}


def production_profile(symbol: str) -> str:
    """Return the explicitly selected production profile for a symbol."""
    symbol = symbol.upper()
    if symbol == "XAUUSD":
        requested = os.environ.get("FOREX_XAU_PROFILE", "baseline").strip().lower()
        if requested in XAUUSD_PROFILES:
            return requested
    return "baseline"


def production_config(symbol: str) -> StrategyConfig:
    """Return the deterministic numeric config for the selected profile."""
    if symbol.upper() == "XAUUSD":
        return XAUUSD_PROFILES[production_profile(symbol)]
    return BASELINE


def production_adx_gate(symbol: str) -> float | None:
    """Return an optional production ADX gate for the selected profile."""
    if symbol.upper() == "XAUUSD" and production_profile(symbol) == "adx25_faster_target":
        return 25.0
    return None
