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
        # Repeating stair-steps create confirmed higher highs and higher lows.
        closes = [100, 99, 101, 100, 103, 101, 105, 103, 107, 105, 109, 107, 111, 109, 113, 111, 115]
        result = dow_structure(make_ohlc(closes), StrategyConfig())
        self.assertEqual(result["trend"], "up")
        self.assertGreater(result["last_swing_high"], result["last_swing_low"])

    def test_dow_downtrend(self):
        closes = [115, 116, 113, 115, 111, 113, 109, 111, 107, 109, 105, 107, 103, 105, 101, 103, 99]
        result = dow_structure(make_ohlc(closes), StrategyConfig())
        self.assertEqual(result["trend"], "down")
        self.assertLess(result["last_swing_low"], result["last_swing_high"])

    def test_trade_plan_never_inverts_risk(self):
        base = 100 + np.linspace(0, 30, 320) + 2 * np.sin(np.arange(320) / 3.0)
        h1 = make_ohlc(base.tolist())
        daily = h1.resample("1D").agg({"Open": "first", "High": "max", "Low": "min", "Close": "last"}).dropna()
        h4 = h1.resample("4h").agg({"Open": "first", "High": "max", "Low": "min", "Close": "last"}).dropna()
        plan = generate_trade_plan(daily, h4, h1, StrategyConfig())
        if plan.direction == "buy":
            self.assertLess(plan.stop_price, plan.entry_price)
            self.assertGreater(plan.target1, plan.entry_price)
        elif plan.direction == "sell":
            self.assertGreater(plan.stop_price, plan.entry_price)
            self.assertLess(plan.target1, plan.entry_price)


if __name__ == "__main__":
    unittest.main()
