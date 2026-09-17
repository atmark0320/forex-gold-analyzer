"""Production strategy profiles selected from deterministic OOS validation."""
from __future__ import annotations

import json
import os
from dataclasses import replace
from pathlib import Path

from strategy_candidates import ADX_GATES, CANDIDATES, FULL_ALIGNMENT, RISING_ADX
from technical_engine import StrategyConfig

BASELINE = StrategyConfig()
PROFILES_FILE = Path("active_profiles.json")
PROFILE_CONFIGS = dict(CANDIDATES)


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
    return {str(k).upper(): str(v) for k, v in profiles.items()}


def production_profile(symbol: str) -> str:
    """Return the selected production profile for a symbol."""
    symbol = symbol.upper()
    explicit = os.environ.get("FOREX_PROFILE", "").strip().lower()
    if explicit and explicit in PROFILE_CONFIGS:
        return explicit
    saved = _load_active_profiles().get(symbol, "")
    if saved in PROFILE_CONFIGS:
        return saved
    return "baseline"


def production_config(symbol: str) -> StrategyConfig:
    return PROFILE_CONFIGS[production_profile(symbol)]


def production_adx_gate(symbol: str) -> float | None:
    return ADX_GATES.get(production_profile(symbol))


def production_full_alignment(symbol: str) -> bool:
    return production_profile(symbol) in FULL_ALIGNMENT


def production_rising_adx(symbol: str) -> bool:
    return production_profile(symbol) in RISING_ADX


def save_active_profile(symbol: str, profile: str, reason: str | None = None) -> None:
    """Persist an auto-selected profile for the next scheduled run."""
    symbol = symbol.upper()
    if profile not in PROFILE_CONFIGS:
        raise ValueError(f"Unsupported production profile for {symbol}: {profile}")
    profiles = _load_active_profiles()
    previous = profiles.get(symbol, "baseline")
    profiles[symbol] = profile
    payload = {"profiles": profiles}
    if reason:
        history = payload.setdefault("history", [])
        history.append({
            "symbol": symbol,
            "from": previous,
            "to": profile,
            "reason": reason,
        })
    PROFILES_FILE.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
