from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from backtest_strategy import load_data
from strategy_candidates import ADX_GATES, CANDIDATES, evaluate_candidate
from strategy_profiles import production_profile, save_active_profile

WINDOW_DAYS = 180
STEP_DAYS = 90
WINDOWS = 7
HEALTH_FILE = Path("production_health.json")
OUTPUT = Path("auto_profile_recommendation.json")
AUTO_SWITCH_ENV = "AUTO_PROFILE_SWITCH"


def _candidate_names(symbol: str) -> list[str]:
    """Return only conservative production candidates for a symbol."""
    if symbol == "XAUUSD":
        return [name for name in ("baseline", "adx25_faster_target") if name in CANDIDATES]
    return ["baseline"] if "baseline" in CANDIDATES else []


def _rolling_summary(symbol: str, variant: str):
    if variant not in CANDIDATES:
        raise ValueError(f"Unsupported candidate {variant} for {symbol}")
    h1 = load_data(symbol)
    end = h1.index.max()
    rows = []
    for window in range(1, WINDOWS + 1):
        window_end = end - pd.Timedelta(days=STEP_DAYS * (window - 1))
        window_start = window_end - pd.Timedelta(days=WINDOW_DAYS)
        cfg = CANDIDATES[variant]
        result = evaluate_candidate(
            symbol,
            h1,
            cfg=cfg,
            adx_gate=ADX_GATES.get(variant),
            trade_start=window_start,
            trade_end=window_end + pd.Timedelta(hours=1),
        )
        rows.append(
            {
                "symbol": symbol,
                "variant": variant,
                "window": window,
                "window_start": window_start.isoformat(),
                "window_end": window_end.isoformat(),
                "trades": int(result.trades),
                "wins": int(result.wins),
                "losses": int(result.losses),
                "total_r": float(result.total_r),
                "expectancy_r": float(result.expectancy_r),
                "profit_factor": float(result.profit_factor),
                "max_drawdown_r": float(result.max_drawdown_r),
                "win_rate_pct": float(result.win_rate_pct),
            }
        )
    frame = pd.DataFrame(rows)
    if frame.empty:
        return {
            "symbol": symbol,
            "variant": variant,
            "windows": 0,
            "positive_windows": 0,
            "negative_windows": 0,
            "total_r": 0.0,
            "median_window_r": 0.0,
            "worst_window_r": 0.0,
            "best_window_r": 0.0,
            "total_trades": 0,
        }, rows
    summary = {
        "symbol": symbol,
        "variant": variant,
        "windows": int(len(frame)),
        "positive_windows": int((frame["total_r"] > 0).sum()),
        "negative_windows": int((frame["total_r"] < 0).sum()),
        "total_r": round(float(frame["total_r"].sum()), 3),
        "median_window_r": round(float(frame["total_r"].median()), 3),
        "worst_window_r": round(float(frame["total_r"].min()), 3),
        "best_window_r": round(float(frame["total_r"].max()), 3),
        "total_trades": int(frame["trades"].sum()),
    }
    return summary, rows


def _degraded_symbols() -> list[str]:
    if not HEALTH_FILE.exists():
        return []
    try:
        payload = json.loads(HEALTH_FILE.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return []
    symbols: list[str] = []
    for row in payload.get("symbols", []):
        status = str(row.get("status", "")).lower()
        if status in {"degraded", "weak"}:
            symbol = str(row.get("symbol", "")).upper()
            symbols.append(symbol)
    return symbols


def _select_replacement(symbol: str):
    current = production_profile(symbol)
    current_summary, _ = _rolling_summary(symbol, current)
    best = None
    for variant in _candidate_names(symbol):
        if variant == current:
            continue
        summary, rows = _rolling_summary(symbol, variant)
        if summary["total_trades"] < 20:
            continue
        if summary["total_r"] <= current_summary["total_r"] + 2.0:
            continue
        if summary["median_window_r"] <= current_summary["median_window_r"] + 0.5:
            continue
        if summary["positive_windows"] < max(4, WINDOWS // 2):
            continue
        if summary["worst_window_r"] < current_summary["worst_window_r"] - 2.0:
            continue
        score = (
            summary["total_r"]
            + summary["median_window_r"] * 1.5
            + summary["best_window_r"] * 0.5
            - max(summary["worst_window_r"], 0.0) * 0.25
        )
        candidate = {
            "symbol": symbol,
            "current_profile": current,
            "recommended_profile": variant,
            "score": round(float(score), 3),
            "reason": (
                f"総合成績 {summary['total_r']}R / median {summary['median_window_r']}R / "
                f"positive_windows {summary['positive_windows']}/{summary['windows']}"
            ),
            "current_summary": current_summary,
            "candidate_summary": summary,
            "rows": rows,
        }
        if best is None or candidate["score"] > best["score"]:
            best = candidate
    return best


def main() -> None:
    recommendations = []
    for symbol in _degraded_symbols():
        candidate = _select_replacement(symbol)
        if candidate is None:
            recommendations.append(
                {
                    "symbol": symbol,
                    "status": "no_safe_replacement",
                    "current_profile": production_profile(symbol),
                    "reason": "有効な候補が見つからず現行プロファイルを維持",
                }
            )
            continue
        recommendations.append(
            {
                "symbol": symbol,
                "status": "recommended",
                "current_profile": candidate["current_profile"],
                "recommended_profile": candidate["recommended_profile"],
                "reason": candidate["reason"],
                "score": candidate["score"],
                "current_summary": candidate["current_summary"],
                "candidate_summary": candidate["candidate_summary"],
            }
        )

    payload = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "auto_profile_switch": os.environ.get(AUTO_SWITCH_ENV, "0").strip() == "1",
        "recommendations": recommendations,
    }
    OUTPUT.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    if payload["auto_profile_switch"]:
        for item in recommendations:
            if item.get("status") != "recommended":
                continue
            save_active_profile(
                item["symbol"],
                item["recommended_profile"],
                reason=f"degraded -> {item['recommended_profile']} ({item['reason']})",
            )

    for item in recommendations:
        print(
            f"{item['symbol']}: {item.get('status', 'unknown')} / "
            f"{item.get('current_profile', '-') } -> {item.get('recommended_profile', '-') }"
        )


if __name__ == "__main__":
    main()
