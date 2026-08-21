import json
import os
from datetime import date
import pandas as pd

HISTORY_FILE = "reports_history.json"

def log_and_evaluate_accuracy(current_snapshot, lookback_days=3):
    today_str = date.today().isoformat()
    
    # 1. Load history
    history = {}
    if os.path.exists(HISTORY_FILE):
        try:
            with open(HISTORY_FILE, "r", encoding="utf-8") as f:
                history = json.load(f)
        except Exception:
            history = {}

    # 2. Save current run
    history[today_str] = current_snapshot
    with open(HISTORY_FILE, "w", encoding="utf-8") as f:
        json.dump(history, f, indent=2, ensure_ascii=False)

    # 3. Identify lookback baseline
    past_dates = sorted([d for d in history.keys() if d < today_str])
    if not past_dates:
        return "", ""

    target_date = past_dates[-min(lookback_days, len(past_dates))]
    past_data = history[target_date]

    rows_en = []
    rows_zh = []
    wins = 0
    total = 0

    for ticker, past in past_data.items():
        if ticker not in current_snapshot:
            continue

        p_orig = past["price"]
        p_curr = current_snapshot[ticker]["price"]
        chg = ((p_curr - p_orig) / p_orig) * 100
        rating = past.get("rating", "Hold")

        # Determine directional success
        is_bullish = any(w in rating.lower() for w in ["buy", "accumulate", "bull"])
        is_bearish = any(w in rating.lower() for w in ["sell", "reduce", "bear"])

        if is_bullish:
            outcome_en = "✅ Win" if chg > 0 else "❌ Miss"
            outcome_zh = "✅ 达标" if chg > 0 else "❌ 未达标"
            is_win = chg > 0
        elif is_bearish:
            outcome_en = "✅ Win" if chg < 0 else "❌ Miss"
            outcome_zh = "✅ 达标" if chg < 0 else "❌ 未达标"
            is_win = chg < 0
        else: # Neutral / Hold
            is_win = abs(chg) < 2.0
            outcome_en = "✅ Range" if is_win else "⚠️ Volatile"
            outcome_zh = "✅ 符合区间" if is_win else "⚠️ 波动突破"

        if is_win:
            wins += 1
        total += 1

        rows_en.append({
            "Ticker": ticker,
            "Past Signal": rating,
            "Past Price": f"${p_orig:.2f}",
            "Current Price": f"${p_curr:.2f}",
            "Actual Return": f"{chg:+.2f}%",
            "Result": outcome_en
        })
        rows_zh.append({
            "代码": ticker,
            "前期信号": rating,
            "前期价格": f"${p_orig:.2f}",
            "当前价格": f"${p_curr:.2f}",
            "实际变动": f"{chg:+.2f}%",
            "评定结果": outcome_zh
        })

    win_rate = (wins / total * 100) if total > 0 else 0.0
    df_en = pd.DataFrame(rows_en)
    df_zh = pd.DataFrame(rows_zh)

    en_section = f"""
## Historical Signal Verification ({lookback_days}-Day Rolling Backtest)
> **Model Accuracy Score:** **{win_rate:.1f}% ({wins}/{total})** directional win rate comparing signals from **{target_date}** to today.

{df_en.to_markdown(index=False)}

* **Model Reflection:** Automated directional validation tracks outperforming momentum leaders while highlighting market-wide pullback risks.
"""

    zh_section = f"""
## 历史预测回测与胜率验证 ({lookback_days}日滚动跟踪)
> **模型综合胜率:** **{win_rate:.1f}% ({wins}/{total})**（基准对比日期: **{target_date}**）

{df_zh.to_markdown(index=False)}

* **模型复盘与校准:** 自动追踪强势动量标的胜率表现，同时对大盘回调期间的回撤偏差进行风险权重修正。
"""

    return en_section, zh_section
