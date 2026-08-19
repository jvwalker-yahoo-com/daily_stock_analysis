# -*- coding: utf-8 -*-
"""
===================================
A股自选股智能分析系统 - 主调度程序
===================================

职责：
1. 协调各模块完成股票分析流程
2. 实现低并发的线程池调度
3. 全局异常处理，确保单股失败不影响整体
4. 提供命令行入口

使用方式：
    python main.py              # 正常运行
    python main.py --debug      # 调试模式
    python main.py --dry-run    # 仅获取数据不分析

交易理念（已融入分析）：
- 严进策略：不追高，乖离率 > 5% 不买入
- 趋势交易：只做 MA5>MA10>MA20 多头排列
- 效率优先：关注筹码集中度好的股票
- 买点偏好：缩量回踩 MA5/MA10 支撑
"""

from __future__ import annotations

import json
import multiprocessing
import os
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

from dotenv import dotenv_values
from src.config import setup_env, get_config

# ============================================================
# ⭐ Load tickers from file and inject into config
# ============================================================
with open("stock_list.txt") as f:
    tickers = [line.strip() for line in f if line.strip()]

print("Loaded tickers:", tickers)

config = get_config()
config.stock_list = tickers

# Prevent .env reload from wiping your tickers
config.refresh_stock_list = lambda: None

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

_packaged_import_probe = os.getenv("DSA_PACKAGED_IMPORT_PROBE")
if _packaged_import_probe:
    import importlib
    import sys

    try:
        importlib.import_module(_packaged_import_probe)
    except Exception as exc:
        print(
            f"ERROR: packaged import failed for {_packaged_import_probe}: {exc}",
            file=sys.stderr,
        )
        sys.exit(1)

    print(f"OK: packaged import succeeded for {_packaged_import_probe}")
    sys.exit(0)

import argparse
import logging
import sys
import time
import uuid
from datetime import date, datetime, timezone, timedelta

from src.webui_frontend import prepare_webui_frontend_assets
from src.config import get_config, Config
from src.logging_config import setup_logging
from src.brokers.futu.portfolio import FutuPortfolioError
from data_provider.base import canonical_stock_code
from src.services.stock_list_parser import split_stock_list
from src.services.stock_code_utils import resolve_index_stock_code_for_analysis

logger = logging.getLogger(__name__)
_RUNTIME_ENV_FILE_KEYS = set()
_PUBLIC_BIND_HOSTS = frozenset({"0.0.0.0", "::", "[::]", "*"})
_LAST_ANALYSIS_FAILURE_REASON: Optional[str] = None

# ============================================================
# (Everything below here is unchanged from your original file)
# ============================================================

def _get_active_env_path() -> Path:
    env_file = os.getenv("ENV_FILE")
    if env_file:
        return Path(env_file)
    return Path(__file__).resolve().parent / ".env"

def _is_public_bind_host(host: str) -> bool:
    return (host or "").strip().lower() in _PUBLIC_BIND_HOSTS

def _warn_if_public_webui_without_auth(host: str) -> None:
    if not _is_public_bind_host(host):
        return

    from src.auth import is_auth_enabled

    if is_auth_enabled():
        return
    logger.warning(
        "WEBUI_HOST=%s binds the Web UI to a public interface while "
        "ADMIN_AUTH_ENABLED=false. Keep this service behind a trusted network "
        "boundary or enable admin authentication before exposing it.",
        host,
    )

def _resolve_web_service_bind(args: argparse.Namespace, config: Config) -> Tuple[str, int]:
    host = args.host if args.host is not None else (config.webui_host or "127.0.0.1")
    port = args.port if args.port is not None else config.webui_port
    return host, port

def _read_active_env_values() -> Optional[Dict[str, str]]:
    env_path = _get_active_env_path()
    if not env_path.exists():
        return {}

    try:
        values = dotenv_values(env_path)
    except Exception as exc:
        logger.warning("读取配置文件 %s 失败，继续沿用当前环境变量: %s", env_path, exc)
        return None

    return {
        str(key): "" if value is None else str(value)
        for key, value in values.items()
        if key is not None
    }

_ACTIVE_ENV_FILE_VALUES = _read_active_env_values() or {}
_RUNTIME_ENV_FILE_KEYS = {
    key for key in _ACTIVE_ENV_FILE_VALUES
    if key not in _INITIAL_PROCESS_ENV
}

_env_bootstrapped = True

# (… the rest of your file continues unchanged …)
