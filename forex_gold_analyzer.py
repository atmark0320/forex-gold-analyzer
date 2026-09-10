import asyncio
import os
import re
import sys
import json
import time
from datetime import datetime, timezone, timedelta

import httpx
import requests
import pandas as pd
import yfinance as yf
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm

from google import genai
from openai import AsyncOpenAI

JST = timezone(timedelta(hours=9))
RUN_TS = datetime.now(JST).strftime("%Y%m%d_%H%M")
CHART_DIR = "charts"
NOTIFY_PAYLOAD_FILE = "notify_payloads.json"

# ---------------------------------------------------------
# 0. 起動時チェック
# ---------------------------------------------------------
def check_env_keys():
    required = ["GEMINI_API_KEY", "GROQ_API_KEY", "OPENROUTER_API_KEY"]
    missing = [k for k in required if not os.environ.get(k)]
    if missing:
        print(f"⚠️ 以下の環境変数が設定されていません: {', '.join(missing)}")
        print("   該当プロバイダの呼び出しは全て失敗します。\n")

def check_groq_connectivity():
    key = os.environ.get("GROQ_API_KEY")
    if not key:
        return
    try:
        resp = requests.get(
            "https://api.groq.com/openai/v1/models",
            headers={"Authorization": f"Bearer {key}"},
            timeout=10,
        )
        if resp.status_code == 200:
            print("✅ Groqへの疎通OK")
        else:
            print(f"⚠️ Groq疎通チェックがHTTP {resp.status_code} を返しました: {resp.text[:200]}")
    except Exception as e:
        print(f"⚠️ Groq疎通チェックで例外: {type(e).__name__}: {e}")

# ---------------------------------------------------------
# 1. チャートデータ自動取得・分析用データ加工
# ---------------------------------------------------------
def fetch_and_process_chart_data(symbol_ticker: str, symbol_name: str):
    """テキストサマリーに加え、チャート描画用の1時間足DataFrameも返す"""
    ticker = yf.Ticker(symbol_ticker)
    df_daily = ticker.history(period="60d", interval="1d")
    df_1h = ticker.history(period="14d", interval="1h")

    if df_daily.empty or df_1h.empty:
        return f"{symbol_name} のデータ取得に失敗しました。", None

    df_4h = df_1h.resample('4h').agg({
        'Open': 'first', 'High': 'max', 'Low': 'min', 'Close': 'last', 'Volume': 'sum'
    }).dropna()

    def get_summary(df, name):
        latest = df.iloc[-1]
        prev_10 = df.tail(10)
        high_10 = prev_10['High'].max()
        low_10 = prev_10['Low'].min()
        ma20 = df['Close'].rolling(20).mean().iloc[-1]
        return f"""
  [{name}]
  - 現在値(終値): {latest['Close']:.3f} (高値: {latest['High']:.3f}, 安値: {latest['Low']:.3f})
  - 直近10本の最高値: {high_10:.3f} / 最安値: {low_10:.3f}
  - 20移動平均線(20MA): {ma20:.3f}
"""

    summary_daily = get_summary(df_daily, "日足")
    summary_4h = get_summary(df_4h, "4時間足")
    summary_1h = get_summary(df_1h, "1時間足")
    text = f"【{symbol_name} チャート自動取得データ】\n{summary_daily}{summary_4h}{summary_1h}"
    return text, df_1h

# ---------------------------------------------------------
# 2. AIクライアント設定
# ---------------------------------------------------------
gemini_client = genai.Client(api_key=os.environ.get("GEMINI_API_KEY"))

groq_http_client = httpx.AsyncClient(trust_env=False)
groq_client = AsyncOpenAI(
    base_url="https://api.groq.com/openai/v1",
    api_key=os.environ.get("GROQ_API_KEY"),
    http_client=groq_http_client,
)

openrouter_client = AsyncOpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=os.environ.get("OPENROUTER_API_KEY")
)

GEMINI_MODEL_CANDIDATES = ["gemini-3.6-flash", "gemini-3.5-flash-lite", "gemini-flash-latest"]
GEMINI_RETRY_ON_503 = 2
GEMINI_RETRY_WAIT_SEC = 4
GEMINI_PER_CANDIDATE_TIMEOUT_SEC = 45
GEMINI_TOTAL_TIMEOUT_SEC = 160
GEMINI_FACILITATOR_TIMEOUT_SEC = 150

GROQ_MODEL = "openai/gpt-oss-120b"
OPENROUTER_EXCLUDE_SUBSTRINGS = ["nex-n2.5-pro"]
AGENT_HARD_TIMEOUT_SEC = 90

# ---------------------------------------------------------
# 1b. 経済指標カレンダー(ForexFactoryの無料公開JSONフィード、APIキー不要)
# ---------------------------------------------------------
ECONOMIC_CALENDAR_URL = "https://nfs.faireconomy.media/ff_calendar_thisweek.json"
RELEVANT_CURRENCIES = {"USD", "JPY"}  # USD/JPYと金(ドル建て)の値動きに影響しやすい通貨
IMPACT_LABEL = {"High": "危険度:高", "Medium": "危険度:中", "Low": "危険度:低"}

def fetch_todays_key_events() -> list[dict]:
    try:
        resp = requests.get(ECONOMIC_CALENDAR_URL, timeout=15)
        resp.raise_for_status()
        events = resp.json()
    except Exception as e:
        print(f"⚠️ 経済指標カレンダーの取得に失敗しました: {type(e).__name__}: {e}")
        return []

    today_jst = datetime.now(JST).date()
    result = []
    for e in events:
        raw_date = e.get("date")
        if not raw_date:
            continue
        try:
            dt = datetime.fromisoformat(raw_date.replace("Z", "+00:00"))
            if dt.tzinfo is None:
                dt = dt.replace(tzinfo=timezone.utc)
            dt_jst = dt.astimezone(JST)
        except ValueError:
            continue
        if dt_jst.date() != today_jst:
            continue
        if e.get("country") not in RELEVANT_CURRENCIES:
            continue
        if e.get("impact") not in ("High", "Medium"):
            continue
        result.append({
            "time": dt_jst.strftime("%H:%M"),
            "currency": e.get("country"),
            "title": e.get("title", "(名称不明)"),
            "impact": e.get("impact"),
        })
    result.sort(key=lambda x: x["time"])
    return result

def format_calendar_report(events: list[dict]) -> str:
    header = f"【本日の要注意経済指標】 {datetime.now(JST).strftime('%Y-%m-%d')} (JST)\n対象: USD・JPY関連の中〜高重要度指標\n"
    if not events:
        return header + "\n本日、該当する指標発表はありません。"
    lines = [header]
    for e in events:
        danger = IMPACT_LABEL.get(e["impact"], e["impact"])
        lines.append(f"{e['time']} [{e['currency']}] {e['title']} ({danger})")
    return "\n".join(lines)

# ---------------------------------------------------------
# 2b. OpenRouterの「今使える無料モデル」を動的に取得
# ---------------------------------------------------------
def get_live_openrouter_free_models(limit: int = 3) -> list[str]:
    try:
        resp = requests.get("https://openrouter.ai/api/v1/models", timeout=15)
        resp.raise_for_status()
        data = resp.json().get("data", [])
        free_ids = []
        for m in data:
            model_id = m.get("id", "")
            pricing = m.get("pricing", {})
            try:
                prompt_price = float(pricing.get("prompt", "1"))
                completion_price = float(pricing.get("completion", "1"))
            except (TypeError, ValueError):
                continue
            if model_id.endswith(":free") and prompt_price == 0.0 and completion_price == 0.0:
                if any(bad in model_id for bad in OPENROUTER_EXCLUDE_SUBSTRINGS):
                    continue
                free_ids.append(model_id)
        return free_ids[:limit]
    except Exception as e:
        print(f"⚠️ OpenRouter無料モデル一覧の取得に失敗しました: {type(e).__name__}: {e}")
        return []

# ---------------------------------------------------------
# 3. エージェント呼び出し関数
# ---------------------------------------------------------
async def _call_gemini(prompt: str, status_info: dict, json_mode: bool = False):
    last_err = None
    for candidate in GEMINI_MODEL_CANDIDATES:
        attempts = GEMINI_RETRY_ON_503 if candidate == "gemini-flash-latest" else 1
        for attempt in range(attempts):
            try:
                kwargs = {"model": candidate, "contents": prompt}
                if json_mode:
                    kwargs["config"] = {"response_mime_type": "application/json"}
                res = await asyncio.wait_for(
                    gemini_client.aio.models.generate_content(**kwargs),
                    timeout=GEMINI_PER_CANDIDATE_TIMEOUT_SEC,
                )
                status_info["success"] = True
                status_info["response"] = res.text
                status_info["rate_limit_info"] = f"正常応答 (Google AI Studio, model={candidate})"
                return
            except asyncio.TimeoutError:
                last_err = TimeoutError(f"{candidate}が{GEMINI_PER_CANDIDATE_TIMEOUT_SEC}秒応答しませんでした")
                break
            except TypeError as e:
                # config引数が古いSDKで使えない場合はjson_modeなしで再試行
                if json_mode:
                    try:
                        res = await asyncio.wait_for(
                            gemini_client.aio.models.generate_content(model=candidate, contents=prompt),
                            timeout=GEMINI_PER_CANDIDATE_TIMEOUT_SEC,
                        )
                        status_info["success"] = True
                        status_info["response"] = res.text
                        status_info["rate_limit_info"] = f"正常応答 (Google AI Studio, model={candidate}, json_mode無効)"
                        return
                    except Exception as e2:
                        last_err = e2
                        break
                last_err = e
                break
            except Exception as e:
                last_err = e
                if "503" in str(e) and attempt < attempts - 1:
                    await asyncio.sleep(GEMINI_RETRY_WAIT_SEC)
                    continue
                break
    raise last_err if last_err else RuntimeError("Gemini: 候補モデルが空です")

async def _call_openai_compatible(client: AsyncOpenAI, model_type: str, prompt: str, status_info: dict, label: str):
    res = await client.chat.completions.create(
        model=model_type,
        messages=[{"role": "user", "content": prompt}],
    )
    content = res.choices[0].message.content if res.choices else None
    if content is None:
        raise ValueError(f"{label}からの応答が空でした")
    status_info["success"] = True
    status_info["response"] = content
    status_info["rate_limit_info"] = f"正常応答 ({label})"

async def call_agent(agent_name: str, model_type: str, provider: str, prompt: str):
    status_info = {"agent": agent_name, "success": False, "response": "", "rate_limit_info": "不明/未提供"}
    try:
        if provider == "gemini":
            await asyncio.wait_for(_call_gemini(prompt, status_info), timeout=GEMINI_TOTAL_TIMEOUT_SEC)
        elif provider == "groq":
            await asyncio.wait_for(
                _call_openai_compatible(groq_client, model_type, prompt, status_info, "Groq"),
                timeout=AGENT_HARD_TIMEOUT_SEC,
            )
        elif provider == "openrouter":
            await asyncio.wait_for(
                _call_openai_compatible(openrouter_client, model_type, prompt, status_info, "OpenRouter"),
                timeout=AGENT_HARD_TIMEOUT_SEC,
            )
    except asyncio.TimeoutError:
        status_info["response"] = f"エラー: タイムアウトで打ち切りました"
        status_info["rate_limit_info"] = "失敗詳細: タイムアウト"
    except Exception as e:
        cause = getattr(e, "__cause__", None)
        err_msg = f"{type(e).__name__}: {e}"
        if cause is not None:
            err_msg += f" | 根本原因: {type(cause).__name__}: {cause}"
        status_info["response"] = f"エラー: {err_msg}"
        status_info["rate_limit_info"] = f"失敗詳細: {err_msg[:200]}"
    return status_info

async def gather_with_progress(labeled_coros, heartbeat_interval=15):
    tasks = {asyncio.ensure_future(coro): label for coro, label in labeled_coros}
    pending = set(tasks.keys())
    start = time.monotonic()
    results = []
    while pending:
        done, pending = await asyncio.wait(pending, timeout=heartbeat_interval)
        if not done:
            elapsed = int(time.monotonic() - start)
            waiting = [tasks[t] for t in pending]
            print(f"  ...まだ処理中です({elapsed}秒経過) 応答待ち: {', '.join(waiting)}")
            continue
        for t in done:
            elapsed = int(time.monotonic() - start)
            res = t.result()
            icon = "✅" if res["success"] else "❌"
            print(f"  {icon} {tasks[t]} 完了 ({elapsed}秒)")
            results.append(res)
    return results

# ---------------------------------------------------------
# 3b. JSON解析 & 固定フォーマット組み立て
# ---------------------------------------------------------
REQUIRED_SCENARIO_KEYS = ["action", "probability_pct", "entry_price", "target1", "target2", "ko_price", "ko_reason"]
REQUIRED_TOP_KEYS = ["common_view", "disagreement", "daily_trend", "h4_trend", "h1_trend",
                     "support", "resistance", "main_scenario", "sub_scenario", "risk_note"]

def parse_json_response(raw_text: str):
    text = raw_text.strip()
    text = re.sub(r"^```(?:json)?", "", text).strip()
    text = re.sub(r"```$", "", text).strip()
    candidates = [text]
    m = re.search(r"\{.*\}", text, re.DOTALL)
    if m:
        candidates.append(m.group(0))
    for c in candidates:
        try:
            data = json.loads(c)
            if all(k in data for k in REQUIRED_TOP_KEYS) and \
               all(k in data["main_scenario"] for k in REQUIRED_SCENARIO_KEYS) and \
               all(k in data["sub_scenario"] for k in REQUIRED_SCENARIO_KEYS):
                return data
        except (json.JSONDecodeError, TypeError, KeyError):
            continue
    return None

def format_fixed_report(symbol_name: str, current_price: float, data: dict) -> str:
    m, s = data["main_scenario"], data["sub_scenario"]
    return f"""【{symbol_name}】統合レポート ({RUN_TS} JST)
現在値: {current_price}

■環境認識
日足:{data['daily_trend']} / 4H:{data['h4_trend']} / 1H:{data['h1_trend']}
サポート:{data['support']} / レジスタンス:{data['resistance']}

■共通認識
{data['common_view']}

■意見の相違点
{data['disagreement']}

■メインシナリオ(確率{m['probability_pct']}%): {m['action']}
エントリー:{m['entry_price']} / 目標1:{m['target1']} / 目標2:{m['target2']}
ノックアウト:{m['ko_price']}
根拠:{m['ko_reason']}

■サブシナリオ(確率{s['probability_pct']}%): {s['action']}
エントリー:{s['entry_price']} / 目標1:{s['target1']} / 目標2:{s['target2']}
ノックアウト:{s['ko_price']}
根拠:{s['ko_reason']}

■注意点
{data['risk_note']}
"""

# ---------------------------------------------------------
# 3c. チャート画像生成
# ---------------------------------------------------------
def _set_japanese_font():
    available = {f.name for f in fm.fontManager.ttflist}
    for name in ["Noto Sans CJK JP", "IPAexGothic", "TakaoGothic", "Noto Sans JP"]:
        if name in available:
            plt.rcParams["font.family"] = name
            return
    print("⚠️ 日本語フォントが見つかりません。チャート内の日本語が文字化けする可能性があります。")

def generate_chart(symbol_name: str, df_1h: pd.DataFrame, data: dict, out_path: str):
    _set_japanese_font()
    m = data["main_scenario"]
    levels = [
        ("サポート", data["support"], "#2ca02c"),
        ("レジスタンス", data["resistance"], "#d62728"),
        ("エントリー", m["entry_price"], "#9467bd"),
        ("目標1", m["target1"], "#1a9850"),
        ("目標2", m["target2"], "#1a9850"),
        ("KO", m["ko_price"], "#d62728"),
    ]
    fig, ax = plt.subplots(figsize=(9, 5), dpi=140)
    ax.plot(df_1h.index, df_1h["Close"], color="#1f77b4", linewidth=1.3, label="1時間足 終値")
    for label, price, color in levels:
        try:
            price = float(price)
        except (TypeError, ValueError):
            continue
        ax.axhline(price, color=color, linestyle="--", linewidth=1)
        ax.annotate(f"{label} {price:g}", xy=(df_1h.index[-1], price),
                    xytext=(6, 0), textcoords="offset points",
                    va="center", fontsize=8, color=color)
    ax.set_title(f"{symbol_name}  1時間足 + 主要レベル ({RUN_TS} JST)")
    ax.legend(loc="upper left", fontsize=8)
    fig.autofmt_xdate()
    fig.tight_layout()
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    fig.savefig(out_path)
    plt.close(fig)

# ---------------------------------------------------------
# 4. メイン処理 & ファシリテーターによる統合
# ---------------------------------------------------------
async def analyze_market(symbol_name: str, ticker_symbol: str, symbol_slug: str, free_models: list[str]) -> list[dict]:
    """成功時はLINE送信用のpayload(list of dict)を返す。失敗時は空リスト。"""
    print(f"\n==================================================")
    print(f"  {symbol_name} のリアルタイムデータ取得 & 分析開始")
    print(f"==================================================")

    chart_data, df_1h = fetch_and_process_chart_data(ticker_symbol, symbol_name)
    print(chart_data)
    if df_1h is None:
        print("データ取得に失敗したためスキップします。")
        return []

    current_price = round(float(df_1h["Close"].iloc[-1]), 3)

    base_prompt = f"""
あなたはFX・貴金属のテクニカルアナリストです。
提供されたチャート数値データからダウ理論を軸に分析を行ってください。

{chart_data}

【指示】
1. 日足・4時間足・1時間足のダウ理論トレンド判定（目線：上/下/レンジ）と押し安値・戻り高値の特定
2. 今後の想定シナリオ（メインシナリオとサブシナリオ、それぞれ確率(%)を明記）
3. IG証券ノックアウトオプションの戦略
   - ノックアウト価格（KO幅）の設定位置と根拠（ひげ・ノイズ回避距離）
   - エントリー価格・利確目標を具体的な数値で
"""

    viewpoints = [
        "サポート/レジスタンスラインの重なり判定",
        "トレンドの勢いとブレイクアウトの信頼性",
        "騙し発生時のリスクと逆張り/調整波視点",
    ]

    labeled_coros = [
        (call_agent("Gemini (ダウ理論重視)", "", "gemini",
                    base_prompt + "\n[視点: ダウ理論の波形・構造判定]"), "Gemini(ダウ理論重視)"),
        (call_agent("Groq (資金管理重視)", GROQ_MODEL, "groq",
                    base_prompt + "\n[視点: IGノックアウトでの厳格なリスク/リワード・資金管理]"), "Groq(資金管理重視)"),
    ]
    for i, model_id in enumerate(free_models):
        vp = viewpoints[i % len(viewpoints)]
        labeled_coros.append(
            (call_agent(f"OpenRouter {model_id} ({vp})", model_id, "openrouter",
                        base_prompt + f"\n[視点: {vp}]"), f"OpenRouter:{model_id}")
        )

    print(f"\n--- {len(labeled_coros)}個のAIエージェントへ並列問い合せ中... ---")
    results = await gather_with_progress(labeled_coros)

    successful_analyses = []
    status_summary_text = "【各AIエージェントの応答状況】\n"
    for res in results:
        icon = "✅ 成功" if res["success"] else "❌ 失敗"
        status_summary_text += f"・{res['agent']}: {icon} ({res['rate_limit_info']})\n"
        if res["success"]:
            successful_analyses.append(f"### {res['agent']} の分析回答:\n{res['response']}")
    print("\n" + status_summary_text)

    if not successful_analyses:
        print("すべてのAIからの回答取得に失敗しました。")
        return []

    combined_inputs = "\n\n".join(successful_analyses)
    facilitator_prompt = f"""
あなたは客観性を重視するファシリテーター（トレーディングデスク責任者）です。
複数のAIアナリストの回答を精査し、中立的な統合戦略を作成してください。

{status_summary_text}

【アナリスト群の回答内容】
{combined_inputs}

【厳守事項】
1. 多数決で安易に決めず、ダウ理論の根拠が最も客観的なものを採用する
2. 意見が対立している点があれば「メインシナリオ」「サブシナリオ」として併記し、両方に確率(%)を必ずつける(2つの確率は合計100に近い値にする)
3. ノックアウト価格はノイズ回避を考慮した安全な水準にする
4. 出力は必ず以下のJSON形式のみ。説明文やコードブロック記号(```)は一切付けない。数値は全て数字のみ(単位や記号を含めない)。

{{
  "common_view": "共通認識を1〜2文で",
  "disagreement": "意見が分かれた点を1〜2文で",
  "daily_trend": "上昇/下降/レンジのいずれか",
  "h4_trend": "上昇/下降/レンジのいずれか",
  "h1_trend": "上昇/下降/レンジのいずれか",
  "support": 数値,
  "resistance": 数値,
  "main_scenario": {{
    "action": "戻り売り/押し目買い/ブレイク追随など短い一言",
    "probability_pct": 数値,
    "entry_price": 数値,
    "target1": 数値,
    "target2": 数値,
    "ko_price": 数値,
    "ko_reason": "1文で"
  }},
  "sub_scenario": {{
    "action": "短い一言",
    "probability_pct": 数値,
    "entry_price": 数値,
    "target1": 数値,
    "target2": 数値,
    "ko_price": 数値,
    "ko_reason": "1文で"
  }},
  "risk_note": "注意すべき失敗シナリオを1〜2文で"
}}
"""

    print("--- ファシリテーター（Gemini）による中立統合を実行中... ---")
    facilitator_start = time.monotonic()
    parsed = None
    for candidate in GEMINI_MODEL_CANDIDATES:
        try:
            print(f"  ...model={candidate} で試行中 ({int(time.monotonic() - facilitator_start)}秒経過)")
            kwargs = {"model": candidate, "contents": facilitator_prompt,
                      "config": {"response_mime_type": "application/json"}}
            try:
                final_response = await asyncio.wait_for(
                    gemini_client.aio.models.generate_content(**kwargs),
                    timeout=GEMINI_FACILITATOR_TIMEOUT_SEC,
                )
            except TypeError:
                final_response = await asyncio.wait_for(
                    gemini_client.aio.models.generate_content(model=candidate, contents=facilitator_prompt),
                    timeout=GEMINI_FACILITATOR_TIMEOUT_SEC,
                )
            parsed = parse_json_response(final_response.text)
            if parsed is not None:
                print(f"  ✅ model={candidate} でJSON形式の統合レポートを取得しました")
                break
            else:
                print(f"  ⚠️ model={candidate} の応答をJSONとして解析できませんでした。次の候補を試します。")
                print(f"     生の応答(先頭300字): {final_response.text[:300]}")
        except asyncio.TimeoutError:
            print(f"  (model={candidate} がタイムアウトしました)")
        except Exception as e:
            print(f"  (model={candidate} で失敗: {type(e).__name__}: {e})")

    if parsed is None:
        print(f"ファシリテーターによる統合に失敗しました({symbol_name})。")
        return []

    fixed_text = format_fixed_report(symbol_name, current_price, parsed)
    print("\n" + fixed_text)

    chart_path = os.path.join(CHART_DIR, f"{symbol_slug}_{RUN_TS}.png")
    try:
        generate_chart(symbol_name, df_1h, parsed, chart_path)
        print(f"✅ チャート画像を生成しました: {chart_path}")
    except Exception as e:
        print(f"⚠️ チャート生成に失敗しました: {type(e).__name__}: {e}")
        chart_path = None

    payload = []
    if chart_path:
        payload.append({"type": "image", "path": chart_path})
    payload.append({"type": "text", "text": fixed_text})
    return payload

# 実行
if __name__ == "__main__":
    check_env_keys()
    check_groq_connectivity()
    live_free_models = get_live_openrouter_free_models(limit=3)
    print(f"現在OpenRouterで利用可能な無料モデル: {live_free_models or '(取得できませんでした)'}")

    all_payloads = []

    calendar_events = fetch_todays_key_events()
    calendar_text = format_calendar_report(calendar_events)
    print("\n" + calendar_text)
    all_payloads.append({"type": "text", "text": calendar_text})

    usdjpy_payload = asyncio.run(analyze_market("ドル円 (USD/JPY)", "JPY=X", "usdjpy", live_free_models))
    gold_payload = asyncio.run(analyze_market("金 (XAU/USD)", "GC=F", "gold", live_free_models))
    all_payloads += usdjpy_payload
    all_payloads += gold_payload

    with open(NOTIFY_PAYLOAD_FILE, "w", encoding="utf-8") as f:
        json.dump(all_payloads, f, ensure_ascii=False, indent=2)
    print(f"\n✅ Telegram送信用ペイロードを {NOTIFY_PAYLOAD_FILE} に書き出しました({len(all_payloads)}件)")

    if not usdjpy_payload and not gold_payload:
        print("❌ ドル円・金の両方で分析が完全に失敗しました。ワークフローを失敗扱いにします。")
        sys.exit(1)
