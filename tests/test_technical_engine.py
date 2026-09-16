import unittest

import numpy as np
import pandas as pd

from market_data_provider import build_timeframes
from technical_engine import StrategyConfig, dow_structure, generate_trade_plan, add_indicators
from backtest_strategy import _dow_snapshots, _snapshot, _prepared_plan


def make_ohlc(closes: list[float]) -> pd.DataFrame:
    index = pd.date_range("2025-01-01", periods=len(closes), freq="h", tz="UTC")
    close = np.asarray(closes, dtype=float)
    return pd.DataFrame(
        {
            "Open": close - 0.1,
            "High": close + 0.3,
            "Low": close - 0.3,
            "Close": close,
        },
        index=index,
    )


class TechnicalEngineTests(unittest.TestCase):
    def test_dow_uptrend(self):
        closes = [
            100, 95, 92, 105, 99, 96, 112, 106, 102,
            120, 114, 110, 128, 121, 118,
        ]
        result = dow_structure(make_ohlc(closes), StrategyConfig())
        self.assertEqual(result["trend"], "up")
        self.assertGreater(result["last_swing_high"], result["last_swing_low"])
        self.assertGreaterEqual(result["confirmed_highs"], 2)
        self.assertGreaterEqual(result["confirmed_lows"], 2)

    def test_dow_downtrend(self):
        closes = [
            120, 125, 128, 115, 121, 124, 110, 116, 119,
            105, 112, 115, 100, 108, 111,
        ]
        result = dow_structure(make_ohlc(closes), StrategyConfig())
        self.assertEqual(result["trend"], "down")
        self.assertLess(result["last_swing_low"], result["last_swing_high"])
        self.assertGreaterEqual(result["confirmed_highs"], 2)
        self.assertGreaterEqual(result["confirmed_lows"], 2)

    def test_trade_plan_never_inverts_risk(self):
        base = 100 + np.linspace(0, 30, 320) + 2 * np.sin(np.arange(320) / 3.0)
        h1 = make_ohlc(base.tolist())
        daily = h1.resample("1D").agg({"Open": "first", "High": "max", "Low": "min", "Close": "last"}).dropna()
        h4 = h1.resample("4h").agg({"Open": "first", "High": "max", "Low": "min", "Close": "last"}).dropna()
        plan = generate_trade_plan(daily, h4, h1, StrategyConfig())
        if plan.direction == "buy":
            self.assertLess(plan.stop_price, plan.entry_price)
            self.assertGreater(plan.target1, plan.entry_price)
            self.assertGreater(plan.target2, plan.target1)
        elif plan.direction == "sell":
            self.assertGreater(plan.stop_price, plan.entry_price)
            self.assertLess(plan.target1, plan.entry_price)
            self.assertLess(plan.target2, plan.target1)

    def test_prepared_backtest_plan_matches_canonical_engine(self):
        """The optimized backtest signal must remain semantically identical to the live engine."""
        n = 24 * 45
        t = np.arange(n, dtype=float)
        close = 100 + 0.018 * t + 3.0 * np.sin(t / 9.0) + 1.2 * np.sin(t / 3.7)
        h1 = make_ohlc(close.tolist())
        daily_raw, h4_raw = build_timeframes(h1)
        cfg = StrategyConfig()
        h1i = add_indicators(h1, cfg)
        h4i = add_indicators(h4_raw, cfg)
        dailyi = add_indicators(daily_raw, cfg)
        h1d = _dow_snapshots(h1i, cfg)
        h4d = _dow_snapshots(h4i, cfg)
        dd = _dow_snapshots(dailyi, cfg)

        h4pos = {ts: i for i, ts in enumerate(h4i.index)}
        checked = 0
        for i in range(100, len(h1i) - 1):
            ts = h1i.index[i]
            hp = h4pos.get(ts - pd.Timedelta(hours=3))
            if hp is None or hp < 29:
                continue
            dp = int(dailyi.index.searchsorted(ts, side="right") - 1)
            if dp < 1:
                continue
            prepared = _prepared_plan(
                _snapshot(dailyi, dd[dp], dp),
                _snapshot(h4i, h4d[hp], hp),
                _snapshot(h1i, h1d[i], i),
                h1i.iloc[i],
                cfg,
            )
            canonical = generate_trade_plan(
                daily_raw.iloc[: dp + 1],
                h4_raw.iloc[: hp + 1],
                h1.iloc[: i + 1],
                cfg,
            )
            self.assertEqual(prepared.direction, canonical.direction)
            self.assertEqual(prepared.score, canonical.score)
            self.assertEqual(prepared.confidence, canonical.confidence)
            self.assertEqual(prepared.entry_type, canonical.entry_type)
            self.assertEqual(prepared.entry_price, canonical.entry_price)
            self.assertEqual(prepared.stop_price, canonical.stop_price)
            self.assertEqual(prepared.target1, canonical.target1)
            self.assertEqual(prepared.target2, canonical.target2)
            self.assertEqual(prepared.risk_per_unit, canonical.risk_per_unit)
            self.assertEqual(prepared.reward_r1, canonical.reward_r1)
            self.assertEqual(prepared.reward_r2, canonical.reward_r2)
            self.assertEqual(prepared.reasons, canonical.reasons)
            self.assertEqual(prepared.invalidation, canonical.invalidation)
            checked += 1
            if checked >= 12:
                break
        self.assertEqual(checked, 12)


if __name__ == "__main__":
    unittest.main()
