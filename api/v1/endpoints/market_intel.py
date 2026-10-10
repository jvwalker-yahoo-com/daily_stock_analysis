# -*- coding: utf-8 -*-
"""
Market Intelligence Endpoints (Frankfurter, CoinGecko, FRED, SEC EDGAR, Polygon/Finnhub)
"""

import asyncio
from typing import Optional
from fastapi import APIRouter
from data_provider.market_intel import MarketIntelSuite

router = APIRouter()
intel_suite = MarketIntelSuite()

@router.get("/forex", summary="Real-time Forex Exchange Rates (Frankfurter)")
async def get_forex_rates(base: str = "USD"):
    return await asyncio.to_thread(intel_suite.get_forex_rates, base)

@router.get("/crypto", summary="Real-time Crypto Quotes (CoinGecko)")
async def get_crypto_pulse():
    return await asyncio.to_thread(intel_suite.get_crypto_prices)

@router.get("/macro", summary="Federal Reserve Economic Data (FRED)")
async def get_fred_macro():
    return await asyncio.to_thread(intel_suite.get_fred_macro_indicators)

@router.get("/sec/{symbol}", summary="Direct SEC EDGAR Filings")
async def get_sec_filings(symbol: str, limit: int = 10):
    return await asyncio.to_thread(intel_suite.get_sec_edgar_filings, symbol, limit)

@router.get("/ticks/{symbol}", summary="Polygon / Finnhub Intraday Ticks")
async def get_stock_ticks(symbol: str):
    return await asyncio.to_thread(intel_suite.get_stock_ticks_and_quote, symbol)
