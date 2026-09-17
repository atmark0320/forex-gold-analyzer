"""Production strategy profiles selected from deterministic OOS validation.

The default profile is baseline. When an automatic profile was previously saved
in active_profiles.json, it takes precedence over the default, unless the
caller explicitly overrides it through the environment.
"""
from __future__ import annotations

import json
import os
from dataclasses import replace
from pathlib import Path

from technical_engine import StrategyConfig

BASELINE = StrategyConfig()
PROFILES_FILE = Path("active_profiles.json")

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


def _load_active_profiles() -> dict[str, str]:
    if not PROFILES_FILE.exists():
        return {}
    try:
        obj = json.loads(PROFILES_FILE.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    profiles = obj.get("profiles", {})
    if not isinstance(profiles, dict):
        return {}
    return {str(k): str(v) for k, v in profiles.items()}


def production_profile(symbol: str) -> str:
    """Return the selected production profile for a symbol."""
    symbol = symbol.upper()
    explicit = os.environ.get("FOREX_PROFILE", "").strip().lower()
    if explicit:
        return explicit
    if symbol == "XAUUSD":
        env_override = os.environ.get("FOREX_XAU_PROFILE", "").strip().lower()
        if env_override:
            return env_override if env_override in XAUUSD_PROFILES else "baseline"
        saved = _load_active_profiles().get(symbol, "")
        if saved and saved in XAUUSD_PROFILES:
            return saved
        return "adx25_faster_target" if "adx25_faster_target" in XAUUSD_PROFILES else "baseline"
    saved = _load_active_profiles().get(symbol, "")
    if saved:
        return saved
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


def save_active_profile(symbol: str, profile: str, reason: str | None = None) -> None:
    """Persist an auto-selected profile for the next scheduled run."""
    symbol = symbol.upper()
    if symbol == "XAUUSD":
        allowed = set(XAUUSD_PROFILES)
    else:
        allowed = {"baseline"}
    if profile not in allowed:
        raise ValueError(f"Unsupported production profile for {symbol}: {profile}")
    payload = {"profiles": _load_active_profiles()}
    payload["profiles"][symbol] = profile
    if reason:
        history = payload.setdefault("history", [])
        if not isinstance(history, list):
            history = []
        history.append({
            "symbol": symbol,
            "from": payload["profiles"].get(symbol),
            "to": profile,
            "reason": reason,
            "generated_at_utc": __import__("datetime").datetime.now(__import__("datetime").timezone.utc).isoformat(),
        })
        payload["history"] = history
    PROFILES_FILE.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
