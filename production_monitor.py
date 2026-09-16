"""Monitor live production profiles without changing them automatically.

The monitor evaluates the currently selected production profile on two recent
rolling windows. It only raises a warning when degradation is supported by
both negative expectancy and weak profit factor with enough trades. It never
changes the production profile automatically.
"""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from backtest_strategy import load_data
from strategy_candidates import ADX_GATES, CANDIDATES, evaluate_candidate
from strategy_profiles import production_adx_gate, production_profile

WINDOW_DAYS = 180
MIN_TRADES = 20
PF_WARNING = 0.90
OUTPUT = Path("production_health.json")

CASES = ("USDJPY", "XAUUSD")


def _evaluate(symbol: str, h1: pd.DataFrame, variant: str, start, end):
    cfg = CANDIDATES[variant]
    result = evaluate_candidate(
        symbol,
        h1,
        cfg=cfg,
        adx_gate=ADX_GATES.get(variant),
        trade_start=start,
        trade_end=end + pd.Timedelta(hours=1),
    )
    return result


def _health_label(recent, previous) -> tuple[str, list[str]]:
    reasons: list[str] = []
    if recent.trades < MIN_TRADES:
        return "insufficient_data", [f"直近{WINDOW_DAYS}日 trades={recent.trades} < {MIN_TRADES}"]

    recent_bad = recent.expectancy_r < 0 and recent.profit_factor < PF_WARNING
    previous_good = previous.expectancy_r >= 0 or previous.profit_factor >= 1.0
    if recent_bad:
        reasons.append(
            f"直近: Expectancy={recent.expectancy_r:.3f}R, PF={recent.profit_factor:.3f}"
        )
        if previous_good:
            reasons.append(
                f"前期間: Expectancy={previous.expectancy_r:.3f}R, PF={previous.profit_factor:.3f}"
            )
            return "degraded", reasons
        return "weak", reasons

    if recent.expectancy_r < 0 or recent.profit_factor < 1.0:
        return "watch", [
            f"直近: Expectancy={recent.expectancy_r:.3f}R, PF={recent.profit_factor:.3f}"
        ]
    return "healthy", [
        f"直近: Expectancy={recent.expectancy_r:.3f}R, PF={recent.profit_factor:.3f}"
    ]


def main() -> None:
    now = datetime.now(timezone.utc)
    all_rows = []
    alerts = []

    for symbol in CASES:
        profile = production_profile(symbol)
        variant = profile
        if variant not in CANDIDATES:
            raise RuntimeError(f"production profile {symbol}/{variant} is not a backtest candidate")

        h1 = load_data(symbol)
        end = h1.index.max()
        recent_start = end - pd.Timedelta(days=WINDOW_DAYS)
        previous_end = recent_start
        previous_start = previous_end - pd.Timedelta(days=WINDOW_DAYS)

        recent = _evaluate(symbol, h1, variant, recent_start, end)
        previous = _evaluate(symbol, h1, variant, previous_start, previous_end)
        label, reasons = _health_label(recent, previous)
        gate = production_adx_gate(symbol)

        row = {
            "symbol": symbol,
            "profile": profile,
            "adx_gate": gate,
            "status": label,
            "recent": {
                "start": recent_start.isoformat(),
                "end": end.isoformat(),
                "trades": recent.trades,
                "wins": recent.wins,
                "losses": recent.losses,
                "win_rate_pct": recent.win_rate_pct,
                "total_r": recent.total_r,
                "expectancy_r": recent.expectancy_r,
                "profit_factor": recent.profit_factor,
                "max_drawdown_r": recent.max_drawdown_r,
                "max_consecutive_losses": recent.max_consecutive_losses,
            },
            "previous": {
                "start": previous_start.isoformat(),
                "end": previous_end.isoformat(),
                "trades": previous.trades,
                "wins": previous.wins,
                "losses": previous.losses,
                "win_rate_pct": previous.win_rate_pct,
                "total_r": previous.total_r,
                "expectancy_r": previous.expectancy_r,
                "profit_factor": previous.profit_factor,
                "max_drawdown_r": previous.max_drawdown_r,
                "max_consecutive_losses": previous.max_consecutive_losses,
            },
            "reasons": reasons,
        }
        all_rows.append(row)
        if label == "degraded":
            alerts.append(f"{symbol}: 本番プロファイル劣化を検知。再検証が必要")

    payload = {
        "generated_at_utc": now.isoformat(),
        "policy": {
            "automatic_profile_switch": False,
            "window_days": WINDOW_DAYS,
            "minimum_trades": MIN_TRADES,
            "degraded_rule": "recent expectancy < 0 AND profit factor < 0.90; previous window is used for context",
        },
        "overall_status": "degraded" if alerts else "ok",
        "alerts": alerts,
        "symbols": all_rows,
    }
    OUTPUT.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    for row in all_rows:
        r = row["recent"]
        print(
            f"{row['symbol']} / {row['profile']} / {row['status']}: "
            f"trades={r['trades']} totalR={r['total_r']} "
            f"Expectancy={r['expectancy_r']} PF={r['profit_factor']} DD={r['max_drawdown_r']}",
            flush=True,
        )
    print(f"Production health: {payload['overall_status']}", flush=True)


if __name__ == "__main__":
    main()
