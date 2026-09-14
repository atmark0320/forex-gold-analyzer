import unittest

import numpy as np
import pandas as pd

from technical_engine import StrategyConfig, dow_structure, generate_trade_plan


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


if __name__ == "__main__":
    unittest.main()
