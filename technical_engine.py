from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Literal

import numpy as np
import pandas as pd

Direction = Literal["buy", "sell", "wait"]
DowTrend = Literal["up", "down", "range", "transition"]


@dataclass(frozen=True)
class StrategyConfig:
    """Dow Theory first; indicators are confirmation/risk filters."""

    ema_fast: int = 20
    ema_mid: int = 50
    ema_slow: int = 200
    rsi_period: int = 14
    atr_period: int = 14
    adx_period: int = 14
    macd_fast: int = 12
    macd_slow: int = 26
    macd_signal: int = 9
    bb_period: int = 20
    bb_std: float = 2.0
    pivot_left: int = 2
    pivot_right: int = 2
    min_adx: float = 18.0
    strong_adx: float = 25.0
    entry_buffer_atr: float = 0.08
    stop_atr: float = 1.35
    target1_r: float = 1.5
    target2_r: float = 2.4
    min_score: int = 7
    strong_score: int = 9


@dataclass(frozen=True)
class TradePlan:
    direction: Direction
    score: int
    confidence: str
    regime: str
    entry_type: str
    entry_price: float | None
    stop_price: float | None
    target1: float | None
    target2: float | None
    risk_per_unit: float | None
    reward_r1: float | None
    reward_r2: float | None
    reasons: tuple[str, ...]
    invalidation: str

    def to_dict(self) -> dict:
        return asdict(self)


def _validate_ohlc(df: pd.DataFrame) -> pd.DataFrame:
    required = {"Open", "High", "Low", "Close"}
    missing = required.difference(df.columns)
    if missing:
        raise ValueError(f"OHLC columns are missing: {sorted(missing)}")
    out = df.copy().sort_index()
    out = out.loc[~out.index.duplicated(keep="last")]
    for col in ["Open", "High", "Low", "Close"]:
        out[col] = pd.to_numeric(out[col], errors="coerce")
    return out.dropna(subset=["Open", "High", "Low", "Close"])


def add_indicators(df: pd.DataFrame, cfg: StrategyConfig = StrategyConfig()) -> pd.DataFrame:
    x = _validate_ohlc(df)
    close, high, low = x["Close"], x["High"], x["Low"]

    x["ema20"] = close.ewm(span=cfg.ema_fast, adjust=False).mean()
    x["ema50"] = close.ewm(span=cfg.ema_mid, adjust=False).mean()
    x["ema200"] = close.ewm(span=cfg.ema_slow, adjust=False).mean()

    delta = close.diff()
    gain = delta.clip(lower=0).ewm(alpha=1 / cfg.rsi_period, adjust=False).mean()
    loss = (-delta.clip(upper=0)).ewm(alpha=1 / cfg.rsi_period, adjust=False).mean()
    rs = gain / loss.replace(0, np.nan)
    x["rsi"] = (100 - (100 / (1 + rs))).replace([np.inf, -np.inf], np.nan).fillna(50.0)

    tr = pd.concat(
        [high - low, (high - close.shift()).abs(), (low - close.shift()).abs()], axis=1
    ).max(axis=1)
    x["atr"] = tr.ewm(alpha=1 / cfg.atr_period, adjust=False).mean()

    up_move = high.diff()
    down_move = -low.diff()
    plus_dm = up_move.where((up_move > down_move) & (up_move > 0), 0.0)
    minus_dm = down_move.where((down_move > up_move) & (down_move > 0), 0.0)
    atr = x["atr"].replace(0, np.nan)
    plus_di = 100 * plus_dm.ewm(alpha=1 / cfg.adx_period, adjust=False).mean() / atr
    minus_di = 100 * minus_dm.ewm(alpha=1 / cfg.adx_period, adjust=False).mean() / atr
    dx = 100 * (plus_di - minus_di).abs() / (plus_di + minus_di).replace(0, np.nan)
    x["plus_di"] = plus_di
    x["minus_di"] = minus_di
    x["adx"] = dx.ewm(alpha=1 / cfg.adx_period, adjust=False).mean()

    ema_fast = close.ewm(span=cfg.macd_fast, adjust=False).mean()
    ema_slow = close.ewm(span=cfg.macd_slow, adjust=False).mean()
    x["macd"] = ema_fast - ema_slow
    x["macd_signal"] = x["macd"].ewm(span=cfg.macd_signal, adjust=False).mean()
    x["macd_hist"] = x["macd"] - x["macd_signal"]

    bb_mid = close.rolling(cfg.bb_period).mean()
    bb_std = close.rolling(cfg.bb_period).std(ddof=0)
    x["bb_mid"] = bb_mid
    x["bb_upper"] = bb_mid + cfg.bb_std * bb_std
    x["bb_lower"] = bb_mid - cfg.bb_std * bb_std
    x["bb_width"] = (x["bb_upper"] - x["bb_lower"]) / bb_mid.replace(0, np.nan)

    x["prior_20_high"] = high.shift(1).rolling(20).max()
    x["prior_20_low"] = low.shift(1).rolling(20).min()
    return x


def resample_ohlc(df: pd.DataFrame, rule: str) -> pd.DataFrame:
    x = _validate_ohlc(df)
    agg = {"Open": "first", "High": "max", "Low": "min", "Close": "last"}
    if "Volume" in x.columns:
        agg["Volume"] = "sum"
    return x.resample(rule).agg(agg).dropna(subset=["Open", "High", "Low", "Close"])


def _confirmed_pivots(df: pd.DataFrame, cfg: StrategyConfig) -> tuple[list[tuple[pd.Timestamp, float]], list[tuple[pd.Timestamp, float]]]:
    """Return pivots only after right-side confirmation, avoiding look-ahead in live use."""
    x = _validate_ohlc(df)
    left, right = cfg.pivot_left, cfg.pivot_right
    if len(x) < left + right + 5:
        return [], []

    highs: list[tuple[pd.Timestamp, float]] = []
    lows: list[tuple[pd.Timestamp, float]] = []
    hi = x["High"].to_numpy(dtype=float)
    lo = x["Low"].to_numpy(dtype=float)
    idx = x.index
    for i in range(left, len(x) - right):
        h_window = hi[i - left : i + right + 1]
        l_window = lo[i - left : i + right + 1]
        if hi[i] == np.max(h_window) and hi[i] > np.max(np.delete(h_window, left)):
            highs.append((idx[i + right], float(hi[i])))
        if lo[i] == np.min(l_window) and lo[i] < np.min(np.delete(l_window, left)):
            lows.append((idx[i + right], float(lo[i])))
    return highs, lows


def dow_structure(df: pd.DataFrame, cfg: StrategyConfig = StrategyConfig()) -> dict:
    """Objective Dow-structure snapshot using confirmed swing highs/lows.

    Uptrend = successive higher highs + higher lows.
    Downtrend = successive lower highs + lower lows.
    Otherwise the market is range/transition until structure confirms a new trend.
    """
    x = _validate_ohlc(df)
    highs, lows = _confirmed_pivots(x, cfg)
    high_values = [v for _, v in highs]
    low_values = [v for _, v in lows]
    trend: DowTrend = "transition"
    if len(high_values) >= 2 and len(low_values) >= 2:
        higher_high = high_values[-1] > high_values[-2]
        higher_low = low_values[-1] > low_values[-2]
        lower_high = high_values[-1] < high_values[-2]
        lower_low = low_values[-1] < low_values[-2]
        if higher_high and higher_low:
            trend = "up"
        elif lower_high and lower_low:
            trend = "down"
        else:
            trend = "range"
    elif len(high_values) >= 2 and high_values[-1] > high_values[-2]:
        trend = "transition"
    elif len(low_values) >= 2 and low_values[-1] < low_values[-2]:
        trend = "transition"

    close = float(x["Close"].iloc[-1])
    last_high = high_values[-1] if high_values else None
    last_low = low_values[-1] if low_values else None
    breakout = None
    if last_high is not None and close > last_high:
        breakout = "up"
    elif last_low is not None and close < last_low:
        breakout = "down"

    return {
        "trend": trend,
        "last_swing_high": last_high,
        "last_swing_low": last_low,
        "breakout": breakout,
        "swing_high_time": highs[-1][0].isoformat() if highs else None,
        "swing_low_time": lows[-1][0].isoformat() if lows else None,
        "confirmed_highs": len(highs),
        "confirmed_lows": len(lows),
    }


def _trend_label(row: pd.Series) -> str:
    """Compatibility label; Dow structure is the primary regime label elsewhere."""
    e20, e50, e200 = float(row["ema20"]), float(row["ema50"]), float(row["ema200"])
    close = float(row["Close"])
    adx = float(row["adx"]) if pd.notna(row["adx"]) else 0.0
    if close > e20 > e50 > e200 and adx >= 18:
        return "up"
    if close < e20 < e50 < e200 and adx >= 18:
        return "down"
    return "range"


def timeframe_snapshot(df: pd.DataFrame, cfg: StrategyConfig = StrategyConfig()) -> dict:
    ind = add_indicators(df, cfg)
    row = ind.iloc[-1]
    dow = dow_structure(ind, cfg)
    return {
        "trend": dow["trend"],
        "dow_trend": dow["trend"],
        "dow_breakout": dow["breakout"],
        "last_swing_high": dow["last_swing_high"],
        "last_swing_low": dow["last_swing_low"],
        "close": float(row["Close"]),
        "ema20": float(row["ema20"]),
        "ema50": float(row["ema50"]),
        "ema200": float(row["ema200"]),
        "rsi": float(row["rsi"]),
        "adx": float(row["adx"]) if pd.notna(row["adx"]) else 0.0,
        "atr": float(row["atr"]) if pd.notna(row["atr"]) else 0.0,
        "macd_hist": float(row["macd_hist"]) if pd.notna(row["macd_hist"]) else 0.0,
    }


def _merge_timeframes(daily: dict, h4: dict, h1: dict) -> tuple[int, list[str], str]:
    score = 0
    reasons: list[str] = []
    regimes = [daily["dow_trend"], h4["dow_trend"], h1["dow_trend"]]
    regime = "range"

    if regimes.count("up") >= 2:
        score += 4
        reasons.append("ダウ構造が複数時間足で高値・安値切り上げ")
        regime = "up"
    elif regimes.count("down") >= 2:
        score += 4
        reasons.append("ダウ構造が複数時間足で高値・安値切り下げ")
        regime = "down"
    elif regimes.count("transition") >= 2:
        reasons.append("ダウ構造がトレンド転換過程")
    else:
        reasons.append("ダウ構造が時間足間で不一致")

    if daily["dow_trend"] == h4["dow_trend"] == h1["dow_trend"] and daily["dow_trend"] in {"up", "down"}:
        score += 3
        reasons.append("日足・4H・1Hのダウ構造が完全一致")

    if regime == "up" and h4.get("last_swing_high") is not None and h4["close"] > h4["last_swing_high"]:
        score += 1
        reasons.append("4Hで直近スイング高値を上抜け")
    if regime == "down" and h4.get("last_swing_low") is not None and h4["close"] < h4["last_swing_low"]:
        score += 1
        reasons.append("4Hで直近スイング安値を下抜け")

    if h4["adx"] >= 25:
        score += 2
        reasons.append("4H ADXが25以上")
    elif h4["adx"] >= 18:
        score += 1
        reasons.append("4H ADXが18以上")
    return score, reasons, regime


def generate_trade_plan(
    daily_df: pd.DataFrame,
    h4_df: pd.DataFrame,
    h1_df: pd.DataFrame,
    cfg: StrategyConfig = StrategyConfig(),
) -> TradePlan:
    daily = timeframe_snapshot(daily_df, cfg)
    h4_ind, h1_ind = add_indicators(h4_df, cfg), add_indicators(h1_df, cfg)
    h4, h1 = timeframe_snapshot(h4_df, cfg), timeframe_snapshot(h1_df, cfg)
    row = h1_ind.iloc[-1]
    score, reasons, regime = _merge_timeframes(daily, h4, h1)
    close = float(row["Close"])
    atr = max(float(row["atr"]), close * 0.0005)
    rsi, macd_hist = float(row["rsi"]), float(row["macd_hist"])

    direction: Direction = "wait"
    if regime == "up":
        if 50 <= rsi <= 72:
            score += 1
            reasons.append("1H RSIが上昇トレンドの押し目圏")
        if macd_hist > 0:
            score += 1
            reasons.append("1H MACDが上昇方向")
        if h1["close"] > h1["ema20"]:
            score += 1
            reasons.append("1H価格が20EMAより上")
        direction = "buy"
    elif regime == "down":
        if 28 <= rsi <= 50:
            score += 1
            reasons.append("1H RSIが下降トレンドの戻り圏")
        if macd_hist < 0:
            score += 1
            reasons.append("1H MACDが下降方向")
        if h1["close"] < h1["ema20"]:
            score += 1
            reasons.append("1H価格が20EMAより下")
        direction = "sell"
    else:
        # Dow Theory prioritizes waiting for structure confirmation over guessing a reversal.
        return TradePlan(
            "wait",
            score,
            "low",
            regime,
            "wait",
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            tuple(reasons + ["ダウ理論のトレンド構造が未確定。転換を先回りしない"]),
            "新しい高値・安値の構造が確認されるまで見送り",
        )

    # Only allow a trade once the Dow structure plus confirmations reach the threshold.
    if score < cfg.min_score:
        return TradePlan(
            "wait", score, "low", regime, "conditional", None, None, None, None, None, None, None,
            tuple(reasons + [f"スコア{score}で最低基準{cfg.min_score}未満"]),
            "ダウ構造を維持しつつ確認条件を満たすまで見送り",
        )

    buffer = atr * cfg.entry_buffer_atr
    if direction == "buy":
        entry = close + buffer
        stop_candidates = [close - atr * cfg.stop_atr]
        if h1_ind["ema50"].iloc[-1] < entry:
            stop_candidates.append(float(h1_ind["ema50"].iloc[-1]))
        if h1["last_swing_low"] is not None and h1["last_swing_low"] < entry:
            stop_candidates.append(float(h1["last_swing_low"]))
        stop = min(stop_candidates)
        risk = entry - stop
        target1, target2 = entry + risk * cfg.target1_r, entry + risk * cfg.target2_r
        invalidation = (
            f"1H終値が直近ダウ安値 {h1['last_swing_low']:.5f} を明確に割る、"
            "または4Hダウ構造が下降へ転換"
            if h1["last_swing_low"] is not None
            else "1Hダウ構造が下降へ転換、または直近安値を明確に下抜け"
        )
    else:
        entry = close - buffer
        stop_candidates = [close + atr * cfg.stop_atr]
        if h1_ind["ema50"].iloc[-1] > entry:
            stop_candidates.append(float(h1_ind["ema50"].iloc[-1]))
        if h1["last_swing_high"] is not None and h1["last_swing_high"] > entry:
            stop_candidates.append(float(h1["last_swing_high"]))
        stop = max(stop_candidates)
        risk = stop - entry
        target1, target2 = entry - risk * cfg.target1_r, entry - risk * cfg.target2_r
        invalidation = (
            f"1H終値が直近ダウ高値 {h1['last_swing_high']:.5f} を明確に上回る、"
            "または4Hダウ構造が上昇へ転換"
            if h1["last_swing_high"] is not None
            else "1Hダウ構造が上昇へ転換、または直近高値を明確に上抜け"
        )

    if risk <= 0 or not np.isfinite(risk):
        return TradePlan(
            "wait", score, "low", regime, "wait", None, None, None, None, None, None, None,
            tuple(reasons + ["リスク幅を正常に計算できない"]), "有効なダウ構造と損切り位置が確定するまで見送り",
        )

    confidence = "high" if score >= cfg.strong_score else "medium"
    entry_type = "conditional" if score < cfg.strong_score else "immediate"
    return TradePlan(
        direction,
        score,
        confidence,
        regime,
        entry_type,
        round(entry, 5),
        round(stop, 5),
        round(target1, 5),
        round(target2, 5),
        round(risk, 5),
        cfg.target1_r,
        cfg.target2_r,
        tuple(reasons),
        invalidation,
    )


def analyze_multi_timeframe(raw_df: pd.DataFrame, cfg: StrategyConfig = StrategyConfig()) -> dict:
    h1 = _validate_ohlc(raw_df)
    if len(h1) < 260:
        raise ValueError("テクニカル分析には最低260本の1時間足データが必要です")
    h4, daily = resample_ohlc(h1, "4h"), resample_ohlc(h1, "1D")
    if len(h4) < 80 or len(daily) < 60:
        raise ValueError("十分な4時間足・日足履歴がありません")
    plan = generate_trade_plan(daily, h4, h1, cfg)
    return {
        "daily": timeframe_snapshot(daily, cfg),
        "h4": timeframe_snapshot(h4, cfg),
        "h1": timeframe_snapshot(h1, cfg),
        "plan": plan.to_dict(),
    }


__all__ = [
    "StrategyConfig",
    "TradePlan",
    "add_indicators",
    "resample_ohlc",
    "dow_structure",
    "timeframe_snapshot",
    "generate_trade_plan",
    "analyze_multi_timeframe",
]
