import unittest
from datetime import datetime, timezone

import numpy as np
import pandas as pd

from market_data_provider import build_timeframes
from strategy_runner import expected_4h_start


class MarketScheduleTests(unittest.TestCase):
    def test_expected_4h_start_is_previous_utc_block(self):
        now = datetime(2026, 9, 14, 12, 5, tzinfo=timezone.utc)
        self.assertEqual(expected_4h_start(now).isoformat(), "2026-09-14T08:00:00+00:00")

        now = datetime(2026, 9, 14, 0, 5, tzinfo=timezone.utc)
        self.assertEqual(expected_4h_start(now).isoformat(), "2026-09-13T20:00:00+00:00")

    def test_build_timeframes_keeps_only_full_4h_bars(self):
        index = pd.date_range("2026-01-05 00:00", periods=10, freq="h", tz="UTC")
        close = np.arange(100.0, 110.0)
        h1 = pd.DataFrame(
            {
                "Open": close,
                "High": close + 0.5,
                "Low": close - 0.5,
                "Close": close + 0.1,
            },
            index=index,
        )
        daily, h4 = build_timeframes(h1)
        self.assertEqual(len(h4), 2)
        self.assertEqual(h4.index[0].isoformat(), "2026-01-05T00:00:00+00:00")
        self.assertEqual(h4.index[1].isoformat(), "2026-01-05T04:00:00+00:00")
        self.assertEqual(len(daily), 0)


if __name__ == "__main__":
    unittest.main()
