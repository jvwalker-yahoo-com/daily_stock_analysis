import os
import smtplib
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from email.mime.base import MIMEBase
from email import encoders
import yfinance as yf
import pandas as pd
import markdown2
from weasyprint import HTML
from openai import OpenAI
from signal_verifier import log_and_evaluate_accuracy

# ==========================================
# 1. Configuration & Watchlist Setup
# ==========================================
WATCHLIST = [
    "MARA", "IREN", "SOXL", "TQQQ", "MSFT", 
    "META", "APLD", "SPY", "QQQ", "BULL", 
    "URA", "HOOD", "SOFI"
]

OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY") or os.environ.get("DEEPSEEK_API_KEY")
OPENAI_BASE_URL = os.environ.get("OPENAI_BASE_URL", "https://api.openai.com/v1")
MODEL_NAME = os.environ.get("OPENAI_MODEL", "gpt-4o")

EMAIL_SENDER = os.environ.get("EMAIL_SENDER")
EMAIL_PASSWORD = os.environ.get("EMAIL_PASSWORD")
EMAIL_RECIPIENT = os.environ.get("EMAIL_RECIPIENT")

# ==========================================
# 2. PDF Styling & Exporter (Bilingual CJK)
# ==========================================
PDF_CSS = """
@page {
    size: A4;
    margin: 15mm 12mm;
    background-color: #ffffff;
}
body {
    font-family: "PingFang SC", "Microsoft YaHei", "Noto Sans CJK SC", "Helvetica Neue", Arial, sans-serif;
    color: #1e293b;
    font-size: 9pt;
    line-height: 1.5;
}
h1 { color: #0f172a; font-size: 15pt; margin-bottom: 6px; border-bottom: 2px solid #0284c7; padding-bottom: 4px; }
h2 { color: #0284c7; font-size: 11.5pt; margin-top: 12px; margin-bottom: 6px; border-bottom: 1px solid #e2e8f0; padding-bottom: 3px; }
h3 { color: #334155; font-size: 10pt; margin-top: 10px; margin-bottom: 4px; }
table { width: 100%; border-collapse: collapse; margin: 10px 0; font-size: 8.5pt; }
th { background-color: #0f172a; color: #ffffff; padding: 6px 8px; text-align: left; font-weight: 600; }
td { padding: 5px 8px; border-bottom: 1px solid #e2e8f0; }
tr:nth-child(even) { background-color: #f8fafc; }
blockquote { border-left: 3px solid #0284c7; padding-left: 10px; margin: 8px 0; color: #475569; background-color: #f0f9ff; padding-top: 4px; padding-bottom: 4px; }
ul { padding-left: 16px; margin: 6px 0; }
li { margin-bottom: 3px; }
"""

def render_pdf(markdown_content, output_filename, header_title):
    html_body = markdown2.markdown(
        markdown_content, 
        extras=["tables", "fenced-code-blocks", "cjk-words", "break-on-newline"]
    )
    full_html = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <meta charset="utf-8">
        <style>{PDF_CSS}</style>
    </head>
    <body>
        <h1>{header_title}</h1>
        {html_body}
    </body>
    </html>
    """
    HTML(string=full_html).write_pdf(output_filename)

# ==========================================
# 3. Market Data & Comparison Matrix
# ==========================================
def fetch_market_data(tickers):
    print("-> Fetching live market metrics...")
    matrix_rows = []
    current_snapshot = {}

    for sym in tickers:
        try:
            ticker = yf.Ticker(sym)
            hist = ticker.history(period="3mo")
            info = ticker.info
            
            if hist.empty or len(hist) < 15:
                continue
            
            price = hist['Close'].iloc[-1]
            prev_price = hist['Close'].iloc[-2]
            five_day_price = hist['Close'].iloc[-6] if len(hist) >= 6 else hist['Close'].iloc[0]
            
            ret_1d = ((price - prev_price) / prev_price) * 100
            ret_5d = ((price - five_day_price) / five_day_price) * 100
            
            # RSI 14
            delta = hist['Close'].diff()
            gain = (delta.where(delta > 0, 0)).rolling(14).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
            rs = gain / loss
            rsi = 100 - (100 / (1 + rs)).iloc[-1]
            
            pe = info.get("forwardPE") or info.get("trailingPE") or "N/A"
            pe_str = f"{pe:.1f}" if isinstance(pe, (int, float)) else "N/A"
            
            if rsi > 60 and ret_5d > 0:
                rating = "Strong Buy" if rsi < 75 else "Tactical Buy"
                score = 8.5
            elif rsi < 40:
                rating = "Accumulate"
                score = 7.0
            else:
                rating = "Hold"
                score = 6.5

            current_snapshot[sym] = {
                "price": round(float(price), 2),
                "rating": rating,
                "score": score
            }

            matrix_rows.append({
                "Ticker": sym,
                "Price": f"${price:.2f}",
                "1D %": f"{ret_1d:+.2f}%",
                "5D %": f"{ret_5d:+.2f}%",
                "RSI (14)": f"{rsi:.1f}",
                "P/E": pe_str,
                "Rating": rating,
                "Score": f"{score:.1f}"
            })
        except Exception as e:
            print(f"Error fetching {sym}: {e}")

    return pd.DataFrame(matrix_rows), current_snapshot

# ==========================================
# 4. LLM Cross-Comparison Generator
# ==========================================
def generate_bilingual_llm_reports(matrix_df):
    print("-> Requesting LLM comparative analysis...")
    if not OPENAI_API_KEY:
        print("Warning: No API key found. Using default markdown block.")
        return "### Watchlist Analysis\nActive market overview.", "### 自选股横向对比\n市场全景速览。"

    client = OpenAI(api_key=OPENAI_API_KEY, base_url=OPENAI_BASE_URL)
    table_md = matrix_df.to_markdown(index=False)
    
    prompt = f"""
Below is today's stock watchlist matrix:
{table_md}

Produce two sections separated exactly by `===DIVIDER===`:

SECTION 1 (ENGLISH):
### Multi-Stock Comparison & Relative Strength
- **Top Momentum Leaders:** Identify top relative strength outperformers.
- **Valuation / Momentum Divergence:** Note high-P/E or oversold setups.
- **Risk & Alert Levels:** Note overbought (RSI > 70) or weak trend tickers.
- **Tactical Allocation Verdict:** 1-sentence positioning summary.

===DIVIDER===

SECTION 2 (CHINESE 简体中文):
### 自选股横向对比与强弱评级
- **强势领涨标的:** 评估动量与相对强弱领先标的。
- **估值与动量背离:** 提示高估值超买或超跌反弹潜力的个股。
- **风险预警与超买超卖:** 标记 RSI > 70 或弱势跌破均线的标的。
- **多资产配置策略建议:** 一句话投资组合配置与风控建议。
"""
    try:
        res = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.2
        )
        content = res.choices[0].message.content
        parts = content.split("===DIVIDER===")
        en_sec = parts[0].strip()
        zh_sec = parts[1].strip() if len(parts) > 1 else ""
        return en_sec, zh_sec
    except Exception as e:
        print(f"LLM API Error: {e}")
        return "### Analysis\nFailed to fetch LLM response.", "### 分析\n未能获取大模型响应。"

# ==========================================
# 5. Email Dispatcher
# ==========================================
def send_report_email(pdf_files, recipient_email):
    if not EMAIL_SENDER or not EMAIL_PASSWORD:
        print("Skipping email dispatch: EMAIL_SENDER or EMAIL_PASSWORD not configured.")
        return
    if not recipient_email:
        print("Skipping email dispatch: No recipient email specified.")
        return

    print(f"-> Sending email report to {recipient_email}...")
    try:
        msg = MIMEMultipart()
        msg['From'] = EMAIL_SENDER
        msg['To'] = recipient_email
        msg['Subject'] = "Daily Stock Watchlist Analysis & Rolling Verification Scorecard"

        body = "Attached are today's English and Chinese stock analysis reports, featuring the comparison matrix and rolling prediction scorecard."
        msg.attach(MIMEText(body, 'plain'))

        for file_path in pdf_files:
            if os.path.exists(file_path):
                with open(file_path, "rb") as f:
                    part = MIMEBase("application", "octet-stream")
                    part.set_payload(f.read())
                encoders.encode_base64(part)
                part.add_header("Content-Disposition", f"attachment; filename={os.path.basename(file_path)}")
                msg.attach(part)

        with smtplib.SMTP_SSL("smtp.gmail.com", 465) as server:
            server.login(EMAIL_SENDER, EMAIL_PASSWORD)
            server.send_message(msg)
        print("-> Email successfully dispatched!")
    except Exception as e:
        print(f"Failed to send email: {e}")

# ==========================================
# 6. Main Pipeline Execution
# ==========================================
def main():
    print("=== Starting Daily Stock Analysis Pipeline ===")
    
    # 1. Fetch live data & snapshot
    matrix_df, current_snapshot = fetch_market_data(WATCHLIST)
    matrix_md = matrix_df.to_markdown(index=False)
    
    # 2. Run signal verification & rolling backtest (3-day lookback)
    print("-> Running signal verification against historical predictions...")
    en_eval_sec, zh_eval_sec = log_and_evaluate_accuracy(current_snapshot, lookback_days=3)
    
    # 3. Generate LLM cross-comparison analysis
    en_llm_sec, zh_llm_sec = generate_bilingual_llm_reports(matrix_df)
    
    # 4. Assemble English Markdown Document
    en_doc = f"""
{en_eval_sec}

## Active Watchlist Comparison Matrix
{matrix_md}

{en_llm_sec}
"""

    # 5. Assemble Chinese Markdown Document
    zh_doc = f"""
{zh_eval_sec}

## 自选股横向对比与全景矩阵
{matrix_md}

{zh_llm_sec}
"""

    # 6. Render English & Chinese PDFs
    pdf_en = "Active_Watchlist_Analysis_EN.pdf"
    pdf_zh = "Active_Watchlist_Analysis_ZH.pdf"
    
    print("-> Compiling PDF documents...")
    render_pdf(en_doc, pdf_en, "Daily Stock Analysis Decision Dashboard")
    render_pdf(zh_doc, pdf_zh, "每日股票自选横向对比与决策看板")
    
    # 7. Optional Email Dispatch
    send_report_email([pdf_en, pdf_zh], EMAIL_RECIPIENT)
    
    print("=== Pipeline Complete! ===")
    print(f"Generated: {pdf_en} and {pdf_zh}")

if __name__ == "__main__":
    main()
