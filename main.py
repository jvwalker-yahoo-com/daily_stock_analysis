# -*- coding: utf-8 -*-
"""
A股自选股智能分析系统 - 主调度程序
"""

from __future__ import annotations

import os
import uuid
import logging
from pathlib import Path
from typing import List

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
    parser.add_argument("--no-market-review", action="store_true")
    parser.add_argument("--force-run", action="store_true")
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

    # ⭐ Correct call — this avoids the Namespace error
    success = pipeline.run_full_analysis(
        config=config,
        args=args,
        stock_codes=config.stock_list,
        raise_errors=False,
    )

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
