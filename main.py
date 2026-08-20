# -*- coding: utf-8 -*-
"""
A股自选股智能分析系统 - 主调度程序
"""

from __future__ import annotations

import os
import uuid
import logging
import requests
from pathlib import Path
from typing import List, Dict, Any

from dotenv import dotenv_values
from src.config import setup_env, get_config
from src.core.pipeline import StockAnalysisPipeline
from src.logging_config import setup_logging

# ============================================================
# Load tickers from file
# ============================================================
with open("stock_list.txt") as f:
    tickers: List[str] = [line.strip() for line in f if line.strip()]

print("Loaded tickers:", tickers)

# ============================================================
# Inject tickers into config
# ============================================================
config = get_config()
config.stock_list = tickers

# Prevent .env reload from wiping your tickers
config.refresh_stock_list = lambda: None

# ============================================================
# Environment setup
# ============================================================
_INITIAL_PROCESS_ENV = dict(os.environ)
setup_env()

# ============================================================
# Danelfin API Integration
# ============================================================
def fetch_danelfin_rankings(tickers: List[str]) -> List[Dict[str, Any]]:
    api_key = os.getenv("DANELFIN_API_KEY")
    if not api_key:
        print("Warning: DANELFIN_API_KEY not found in environment variables.")
        return []

    url = "https://apirest.danelfin.com/ranking"
    headers = {"x-api-key": api_key}
    
    rankings = []
    for ticker in tickers:
        clean_ticker = ticker.strip().upper()
        try:
            response = requests.get(url, headers=headers, params={"ticker": clean_ticker}, timeout=10)
            if response.status_code == 200:
                data = response.json()
                scores = {}
                if data:
                    first_key = list(data.keys())[0]
                    if isinstance(data[first_key], dict) and "aiscore" in data[first_key]:
                        scores = data[first_key]
                    elif "aiscore" in data:
                        scores = data
                
                rankings.append({
                    "ticker": clean_ticker,
                    "ai_score": scores.get("aiscore", "N/A"),
                    "fundamental": scores.get("fundamental", "N/A"),
                    "technical": scores.get("technical", "N/A"),
                    "sentiment": scores.get("sentiment", "N/A"),
                    "low_risk": scores.get("low_risk", "N/A")
                })
            else:
                print(f"Failed to fetch Danelfin data for {clean_ticker}: Status {response.status_code}")
        except Exception as e:
            print(f"Error connecting to Danelfin API for {clean_ticker}: {e}")

    # Sort watchlist head-to-head descending by AI Score
    rankings.sort(key=lambda x: x["ai_score"] if isinstance(x["ai_score"], (int, float)) else -1, reverse=True)
    return rankings

# ============================================================
# Proxy configuration
# ============================================================
if os.getenv("GITHUB_ACTIONS") != "true" and os.getenv("USE_PROXY", "false").lower() == "true":
    proxy_host = os.getenv("PROXY_HOST", "127.0.0.1")
    proxy_port = os.getenv("PROXY_PORT", "10809")
    proxy_url = f"http://{proxy_host}:{proxy_port}"
    os.environ["http_proxy"] = proxy_url
    os.environ["https_proxy"] = proxy_url

# ============================================================
# CLI argument parsing
# ============================================================
import argparse

def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="A股自选股智能分析系统")
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--workers", type=int, default=None)
    parser.add_argument("--no-context-snapshot", action="store_true")
    return parser.parse_args()

# ============================================================
# Main entrypoint
# ============================================================
def main():
    args = parse_arguments()

    # Logging
    setup_logging(log_prefix="stock_analysis", debug=args.debug, log_dir="logs")

    logger = logging.getLogger(__name__)
    logger.info("启动分析系统…")

    # Fetch Danelfin comparative metrics (called once cleanly)
    logger.info("正在获取 Danelfin AI 多因子评分...")
    danelfin_results = fetch_danelfin_rankings(config.stock_list)
    logger.info("Danelfin 评分获取完成: %s", danelfin_results)
    
    # Store results in config context so downstream report builders can embed them into PDFs
    config.danelfin_rankings = danelfin_results

    # Create pipeline
    pipeline = StockAnalysisPipeline(
        config=config,
        max_workers=args.workers,
        query_id=uuid.uuid4().hex,
        query_source="cli",
        save_context_snapshot=not args.no_context_snapshot,
        daily_market_context_enabled=True,
        daily_market_context_allow_generate=True,
    )

    try:
        if args.dry_run:
            success = pipeline.run(config.stock_list, dry_run=True)
        else:
            success = pipeline.run(config.stock_list)
    except Exception as exc:
        logger.error("分析失败: %s", exc)
        exit(1)

    if not success:
        logger.error("分析失败")
        exit(1)

    logger.info("分析完成")
    exit(0)

# ============================================================
# CLI entrypoint
# ============================================================
if __name__ == "__main__":
    main()
