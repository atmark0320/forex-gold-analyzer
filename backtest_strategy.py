"""Fast deterministic backtest for the Dow-first strategy.

The backtest uses only completed 4H candles. A signal is calculated from the
last H1 candle inside that completed 4H block, and execution can begin only on
the following H1 candle. This prevents accidental use of a still-forming 4H
candle or future H1 data.
"""
from __future__ import annotations
import json
import os
import time
from dataclasses import asdict, dataclass
import numpy as np
import pandas as pd
from market_data_provider import build_timeframes, fetch_ohlc
from technical_engine import StrategyConfig, TradePlan, add_indicators

DEFAULT_SYMBOLS = ("USDJPY", "XAUUSD", "EURUSD", "GBPUSD", "AUDUSD", "USDCAD", "USDCHF")

@dataclass
class BacktestResult:
    symbol: str; data_source: str; trades: int; wins: int; losses: int
    win_rate_pct: float; total_r: float; expectancy_r: float; profit_factor: float
    avg_win_r: float; avg_loss_r: float; max_drawdown_r: float; max_consecutive_losses: int

def _env_int(name: str, default: int) -> int:
    try: return max(1, int(os.environ.get(name, str(default))))
    except ValueError: return default

def selected_symbols() -> tuple[str, ...]:
    raw = os.environ.get("BACKTEST_SYMBOLS", "").strip()
    return tuple(x.strip().upper() for x in raw.split(",") if x.strip()) or DEFAULT_SYMBOLS

def load_data(symbol: str) -> pd.DataFrame:
    return fetch_ohlc(symbol, days=_env_int("BACKTEST_DAYS", 730))

def _trade_result(direction: str, future: pd.DataFrame, entry: float, stop: float, target: float) -> float | None:
    if future.empty: return None
    if direction == "buy":
        if float(future.iloc[0]["High"]) < entry: return None
        for _, bar in future.iterrows():
            if float(bar["Low"]) <= stop: return -1.0
            if float(bar["High"]) >= target: return abs(target-entry)/abs(entry-stop)
    else:
        if float(future.iloc[0]["Low"]) > entry: return None
        for _, bar in future.iterrows():
            if float(bar["High"]) >= stop: return -1.0
            if float(bar["Low"]) <= target: return abs(entry-target)/abs(stop-entry)
    return None

def _dow_snapshots(df: pd.DataFrame, cfg: StrategyConfig) -> list[dict]:
    left, right = cfg.pivot_left, cfg.pivot_right
    highs: list[tuple[pd.Timestamp,float]] = []; lows: list[tuple[pd.Timestamp,float]] = []
    hi, lo, idx = df["High"].to_numpy(float), df["Low"].to_numpy(float), df.index
    out = []
    for i in range(len(df)):
        c = i - right
        if c >= left:
            hw = hi[c-left:c+right+1]; lw = lo[c-left:c+right+1]
            if hi[c] == np.max(hw) and hi[c] > np.max(np.delete(hw,left)): highs.append((idx[c+right],float(hi[c])))
            if lo[c] == np.min(lw) and lo[c] < np.min(np.delete(lw,left)): lows.append((idx[c+right],float(lo[c])))
        hv=[v for _,v in highs]; lv=[v for _,v in lows]; trend="transition"
        if len(hv)>=2 and len(lv)>=2:
            if hv[-1]>hv[-2] and lv[-1]>lv[-2]: trend="up"
            elif hv[-1]<hv[-2] and lv[-1]<lv[-2]: trend="down"
            else: trend="range"
        elif len(hv)>=2 and hv[-1]>hv[-2]: trend="transition"
        elif len(lv)>=2 and lv[-1]<lv[-2]: trend="transition"
        close=float(df["Close"].iloc[i]); lh=hv[-1] if hv else None; ll=lv[-1] if lv else None
        br="up" if lh is not None and close>lh else ("down" if ll is not None and close<ll else None)
        out.append({"trend":trend,"dow_trend":trend,"dow_breakout":br,"last_swing_high":lh,"last_swing_low":ll})
    return out

def _snapshot(ind: pd.DataFrame, dow: dict, pos: int) -> dict:
    r=ind.iloc[pos]
    return {"trend":dow["trend"],"dow_trend":dow["dow_trend"],"dow_breakout":dow["dow_breakout"],"last_swing_high":dow["last_swing_high"],"last_swing_low":dow["last_swing_low"],"close":float(r["Close"]),"ema20":float(r["ema20"]),"ema50":float(r["ema50"]),"ema200":float(r["ema200"]),"rsi":float(r["rsi"]),"adx":float(r["adx"]) if pd.notna(r["adx"]) else 0.0,"atr":float(r["atr"]) if pd.notna(r["atr"]) else 0.0,"macd_hist":float(r["macd_hist"]) if pd.notna(r["macd_hist"]) else 0.0}

def _merge(daily:dict,h4:dict,h1:dict)->tuple[int,list[str],str]:
    score=0; reasons=[]; regs=[daily["dow_trend"],h4["dow_trend"],h1["dow_trend"]]; regime="range"
    if regs.count("up")>=2: score+=4; reasons.append("ダウ構造が複数時間足で高値・安値切り上げ"); regime="up"
    elif regs.count("down")>=2: score+=4; reasons.append("ダウ構造が複数時間足で高値・安値切り下げ"); regime="down"
    elif regs.count("transition")>=2: reasons.append("ダウ構造がトレンド転換過程")
    else: reasons.append("ダウ構造が時間足間で不一致")
    if daily["dow_trend"]==h4["dow_trend"]==h1["dow_trend"] and daily["dow_trend"] in {"up","down"}: score+=3; reasons.append("日足・4H・1Hのダウ構造が完全一致")
    if regime=="up" and h4["last_swing_high"] is not None and h4["close"]>h4["last_swing_high"]: score+=1; reasons.append("4Hで直近スイング高値を上抜け")
    if regime=="down" and h4["last_swing_low"] is not None and h4["close"]<h4["last_swing_low"]: score+=1; reasons.append("4Hで直近スイング安値を下抜け")
    if h4["adx"]>=25: score+=2; reasons.append("4H ADXが25以上")
    elif h4["adx"]>=18: score+=1; reasons.append("4H ADXが18以上")
    return score,reasons,regime

def _prepared_plan(daily,h4,h1,row,cfg):
    score,reasons,regime=_merge(daily,h4,h1); close=float(row["Close"]); atr=max(float(row["atr"]),close*0.0005); rsi=float(row["rsi"]); macd=float(row["macd_hist"])
    if regime=="up":
        if 50<=rsi<=72: score+=1; reasons.append("1H RSIが上昇トレンドの押し目圏")
        if macd>0: score+=1; reasons.append("1H MACDが上昇方向")
        if h1["close"]>h1["ema20"]: score+=1; reasons.append("1H価格が20EMAより上")
        direction="buy"
    elif regime=="down":
        if 28<=rsi<=50: score+=1; reasons.append("1H RSIが下降トレンドの戻り圏")
        if macd<0: score+=1; reasons.append("1H MACDが下降方向")
        if h1["close"]<h1["ema20"]: score+=1; reasons.append("1H価格が20EMAより下")
        direction="sell"
    else:
        return TradePlan("wait",score,"low",regime,"wait",None,None,None,None,None,None,None,tuple(reasons+["ダウ理論のトレンド構造が未確定。転換を先回りしない"]),"新しい高値・安値の構造が確認されるまで見送り")
    if score<cfg.min_score:
        return TradePlan("wait",score,"low",regime,"conditional",None,None,None,None,None,None,None,tuple(reasons+[f"スコア{score}で最低基準{cfg.min_score}未満"]),"ダウ構造を維持しつつ確認条件を満たすまで見送り")
    buf=atr*cfg.entry_buffer_atr
    if direction=="buy":
        entry=close+buf; stops=[close-atr*cfg.stop_atr]
        if float(row["ema50"])<entry: stops.append(float(row["ema50"]))
        if h1["last_swing_low"] is not None and h1["last_swing_low"]<entry: stops.append(float(h1["last_swing_low"]))
        stop=min(stops); risk=entry-stop; t1=entry+risk*cfg.target1_r; t2=entry+risk*cfg.target2_r
        inv=f"1H終値が直近ダウ安値 {h1['last_swing_low']:.5f} を明確に割る、または4Hダウ構造が下降へ転換" if h1["last_swing_low"] is not None else "1Hダウ構造が下降へ転換、または直近安値を明確に下抜け"
    else:
        entry=close-buf; stops=[close+atr*cfg.stop_atr]
        if float(row["ema50"])>entry: stops.append(float(row["ema50"]))
        if h1["last_swing_high"] is not None and h1["last_swing_high"]>entry: stops.append(float(h1["last_swing_high"])
        stop=max(stops); risk=stop-entry; t1=entry-risk*cfg.target1_r; t2=entry-risk*cfg.target2_r
        inv=f"1H終値が直近ダウ高値 {h1['last_swing_high']:.5f} を明確に上回る、または4Hダウ構造が上昇へ転換" if h1["last_swing_high"] is not None else "1Hダウ構造が上昇へ転換、または直近高値を明確に上抜け"
    if risk<=0: return TradePlan("wait",score,"low",regime,"wait",None,None,None,None,None,None,None,tuple(reasons+["リスク幅を正常に計算できない"]),"有効なダウ構造と損切り位置が確定するまで見送り")
    return TradePlan(direction,score,"high" if score>=cfg.strong_score else "medium","immediate" if score>=cfg.strong_score else "conditional",round(entry,5),round(stop,5),round(t1,5),round(t2,5),round(risk,5),cfg.target1_r,cfg.target2_r,tuple(reasons),inv)

def evaluate(symbol,h1,cfg=StrategyConfig(),max_hold_bars=48,entry_valid_bars=3):
    h1=h1.sort_index().copy(); daily_raw,h4_raw=build_timeframes(h1)
    h1i,h4i,dailyi=add_indicators(h1,cfg),add_indicators(h4_raw,cfg),add_indicators(daily_raw,cfg)
    h1d,h4d,dd=_dow_snapshots(h1i,cfg),_dow_snapshots(h4i,cfg),_dow_snapshots(dailyi,cfg)
    h4pos={ts:i for i,ts in enumerate(h4i.index)}; decisions=[]
    for i,ts in enumerate(h1i.index):
        block_start=ts-pd.Timedelta(hours=3)
        hp=h4pos.get(block_start)
        if hp is not None and i>=80: decisions.append((i,hp))
    outcomes=[]; equity=peak=maxdd=0.0; blocked=-1; total=len(decisions)
    for n,(i,hp) in enumerate(decisions,1):
        if i<blocked or hp<29: continue
        ts=h1i.index[i]
        dp=int(dailyi.index.searchsorted(ts,side="right")-1)
        if dp<59: continue
        plan=_prepared_plan(_snapshot(dailyi,dd[dp],dp),_snapshot(h4i,h4d[hp],hp),_snapshot(h1i,h1d[i],i),h1i.iloc[i],cfg)
        if plan.direction=="wait" or plan.entry_price is None or plan.stop_price is None or plan.target1 is None: continue
        entry,stop,target=float(plan.entry_price),float(plan.stop_price),float(plan.target1)
        trigger=None
        for j in range(i+1,min(i+1+entry_valid_bars,len(h1i))):
            b=h1i.iloc[j]
            if plan.direction=="buy" and float(b["High"])>=entry: trigger=j; break
            if plan.direction=="sell" and float(b["Low"])<=entry: trigger=j; break
        if trigger is None: continue
        r=_trade_result(plan.direction,h1i.iloc[trigger:min(trigger+max_hold_bars,len(h1i))],entry,stop,target)
        if r is None: continue
        outcomes.append(float(r)); equity+=float(r); peak=max(peak,equity); maxdd=max(maxdd,peak-equity); blocked=trigger+max_hold_bars
        if n%250==0: print(f"    {symbol}: decisions {n:,}/{total:,}, trades {len(outcomes):,}",flush=True)
    wins=[x for x in outcomes if x>0]; losses=[x for x in outcomes if x<0]; trades=len(outcomes); gp=sum(wins); gl=abs(sum(losses)); streak=maxstreak=0
    for x in outcomes:
        if x<0: streak+=1; maxstreak=max(maxstreak,streak)
        else: streak=0
    return BacktestResult(symbol,"Dukascopy spot BID 1H",trades,len(wins),len(losses),round(100*len(wins)/trades,2) if trades else 0.0,round(sum(outcomes),3),round(sum(outcomes)/trades,4) if trades else 0.0,round(gp/gl,3) if gl else (999.0 if gp else 0.0),round(gp/len(wins),4) if wins else 0.0,round(sum(losses)/len(losses),4) if losses else 0.0,round(maxdd,3),maxstreak)

def main():
    symbols=selected_symbols(); days=_env_int("BACKTEST_DAYS",730); started=time.monotonic(); results=[]
    print(f"BACKTEST start: days={days}, symbols={','.join(symbols)}",flush=True)
    for n,symbol in enumerate(symbols,1):
        t=time.monotonic(); print(f"[{n}/{len(symbols)}] loading {symbol}...",flush=True); h1=load_data(symbol)
        print(f"[{n}/{len(symbols)}] {symbol}: {len(h1):,} H1 bars; evaluating completed 4H decisions...",flush=True)
        r=evaluate(symbol,h1); results.append(asdict(r)); print(f"[{n}/{len(symbols)}] {symbol}: trades={r.trades}, win={r.win_rate_pct}%, totalR={r.total_r}, PF={r.profit_factor}, elapsed={time.monotonic()-t:.1f}s",flush=True)
    with open("backtest_results.json","w",encoding="utf-8") as f: json.dump(results,f,ensure_ascii=False,indent=2)
    print(f"BACKTEST complete: {len(results)} symbols in {time.monotonic()-started:.1f}s",flush=True)

if __name__=="__main__": main()
