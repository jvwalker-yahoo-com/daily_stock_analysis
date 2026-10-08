# -*- coding: utf-8 -*-
"""
===================================
Market Flow & Transparency Endpoint
===================================
Combines:
1. Finnhub real-time quotes
2. Congressional trading (QuiverQuant + Senate/House disclosures)
3. Institutional 13F holdings & ownership
4. OpenInsider Form 4 insider transactions
"""

from __future__ import annotations

import os
import re
import logging
from typing import Dict, Any, List, Optional
from fastapi import APIRouter, Query, Request
from fastapi.responses import JSONResponse

logger = logging.getLogger(__name__)
router = APIRouter()

def fetch_openinsider_trades(symbol: str, limit: int = 20) -> List[Dict[str, Any]]:
    trades = []
    try:
        import urllib.request
        from html.parser import HTMLParser

        url = f"http://openinsider.com/screener?s={symbol.upper().strip()}"
        req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"})
        with urllib.request.urlopen(req, timeout=5) as resp:
            html = resp.read().decode("utf-8", errors="ignore")

        rows = re.findall(r'<tr[^>]*>(.*?)</tr>', html, re.DOTALL)
        for row in rows:
            cols = [re.sub(r'<[^>]+>', '', c).strip() for c in re.findall(r'<td[^>]*>(.*?)</td>', row, re.DOTALL)]
            if len(cols) >= 12:
                filing_date = cols[1]
                trade_date = cols[2]
                insider_name = cols[4]
                title = cols[5]
                trade_type = cols[6]
                price_str = cols[7].replace("$", "").replace(",", "")
                qty_str = cols[8].replace(",", "").replace("+", "")
                val_str = cols[11].replace("$", "").replace(",", "")

                try:
                    price = float(price_str) if price_str else 0.0
                except ValueError:
                    price = 0.0
                try:
                    qty = int(qty_str) if qty_str else 0
                except ValueError:
                    qty = 0
                try:
                    val = float(val_str) if val_str else 0.0
                except ValueError:
                    val = 0.0

                is_buy = "P - Purchase" in trade_type or "Purchase" in trade_type
                is_sale = "S - Sale" in trade_type or "Sale" in trade_type

                trades.append({
                    "filing_date": filing_date,
                    "trade_date": trade_date,
                    "insider_name": insider_name,
                    "title": title,
                    "trade_type": trade_type,
                    "is_purchase": is_buy,
                    "is_sale": is_sale,
                    "shares": abs(qty),
                    "price_usd": price,
                    "value_usd": val,
                    "source": "OpenInsider"
                })
                if len(trades) >= limit:
                    break
    except Exception as exc:
        logger.debug(f"[OpenInsider] Screener scrape note: {exc}")
    return trades

def fetch_quiver_congress_trades(symbol: str, limit: int = 20) -> List[Dict[str, Any]]:
    trades = []
    quiver_key = os.getenv("QUIVER_API_KEY", "")
    try:
        import urllib.request
        import json
        
        # Try direct QuiverQuant beta endpoint if key available or public
        headers = {"User-Agent": "Mozilla/5.0"}
        if quiver_key:
            headers["Authorization"] = f"Bearer {quiver_key}"
            url = f"https://api.quiverquant.com/beta/historical/congresstrading/{symbol.upper().strip()}"
        else:
            url = f"https://api.quiverquant.com/beta/live/congresstrading"

        req = urllib.request.Request(url, headers=headers)
        with urllib.request.urlopen(req, timeout=5) as resp:
            data = json.loads(resp.read().decode("utf-8"))
            if isinstance(data, list):
                for item in data:
                    item_ticker = str(item.get("Ticker", "")).upper().strip()
                    if not quiver_key or item_ticker == symbol.upper().strip():
                        trades.append({
                            "representative": item.get("Representative", "Unknown"),
                            "transaction_type": item.get("Transaction", "Unknown"),
                            "amount_range": item.get("Amount", "Unknown"),
                            "transaction_date": item.get("TransactionDate", ""),
                            "disclosure_date": item.get("ReportDate", ""),
                            "house": item.get("House", "Senate/House"),
                            "party": item.get("Party", ""),
                            "source": "QuiverQuant"
                        })
                    if len(trades) >= limit:
                        break
    except Exception as exc:
        logger.debug(f"[QuiverQuant] Congress trade fetch note: {exc}")
    return trades

def fetch_finnhub_13f_holdings(symbol: str, finnhub_key: str) -> List[Dict[str, Any]]:
    holdings = []
    if not finnhub_key:
        return holdings
    try:
        import urllib.request
        import json

        url = f"https://finnhub.io/api/v1/stock/ownership?symbol={symbol.upper().strip()}&token={finnhub_key}"
        req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
        with urllib.request.urlopen(req, timeout=5) as resp:
            data = json.loads(resp.read().decode("utf-8"))
            items = data.get("ownership", [])
            for item in items[:15]:
                holdings.append({
                    "investor_name": item.get("name", "Unknown"),
                    "shares": item.get("share", 0),
                    "change": item.get("change", 0),
                    "filing_date": item.get("filingDate", ""),
                    "source": "Finnhub 13F"
                })
    except Exception as exc:
        logger.debug(f"[Finnhub] 13F fetch note: {exc}")
    return holdings

def get_realtime_quote(symbol: str, finnhub_key: str) -> float:
    if not finnhub_key:
        return 0.0
    try:
        import urllib.request
        import json
        url = f"https://finnhub.io/api/v1/quote?symbol={symbol.upper().strip()}&token={finnhub_key}"
        req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
        with urllib.request.urlopen(req, timeout=4) as resp:
            data = json.loads(resp.read().decode("utf-8"))
            return float(data.get("c", 0.0) or 0.0)
    except Exception:
        return 0.0

@router.get("/transparency", summary="Unified Market Flow Transparency Feed")
@router.get("/transparency/{symbol}", summary="Unified Market Flow Transparency Feed by Symbol")
def get_flow_transparency(symbol: str = "AAPL") -> Dict[str, Any]:
    sym = symbol.upper().strip()
    finnhub_key = os.getenv("FINNHUB_API_KEY", "da02ec1r01qgk75qq5v0da02ec1r01qgk75qq5vg")
    
    # 1. Real-time quote
    price = get_realtime_quote(sym, finnhub_key)
    
    # 2. OpenInsider trades
    insider_trades = fetch_openinsider_trades(sym, limit=20)
    
    # 3. Congressional trades
    congress_trades = fetch_quiver_congress_trades(sym, limit=20)
    
    # 4. Institutional 13F holdings
    institutional_13f = fetch_finnhub_13f_holdings(sym, finnhub_key)
    
    # 5. Conviction synthesis
    insider_buys = sum(1 for t in insider_trades if t.get("is_purchase"))
    insider_sells = sum(1 for t in insider_trades if t.get("is_sale"))
    c_buys = sum(1 for t in congress_trades if "BUY" in str(t.get("transaction_type", "")).upper() or "PURCHASE" in str(t.get("transaction_type", "")).upper())
    c_sells = sum(1 for t in congress_trades if "SELL" in str(t.get("transaction_type", "")).upper() or "SALE" in str(t.get("transaction_type", "")).upper())

    sentiment = "NEUTRAL"
    if insider_buys > insider_sells or c_buys > c_sells:
        sentiment = "BULLISH_FLOW"
    elif insider_sells > insider_buys and c_sells > c_buys:
        sentiment = "BEARISH_FLOW"

    return {
        "symbol": sym,
        "price": price,
        "sentiment": sentiment,
        "metrics": {
            "insider_buys_count": insider_buys,
            "insider_sells_count": insider_sells,
            "congress_buys_count": c_buys,
            "congress_sells_count": c_sells,
            "institutional_holders_sampled": len(institutional_13f)
        },
        "congress": congress_trades,
        "institutional": institutional_13f,
        "insiders": insider_trades,
        "sources": [
            "Finnhub (Real-time Price & 13F Ownership)",
            "QuiverQuant (Congressional Stock Trades)",
            "OpenInsider (SEC Form 4 Officer/Director Trades)"
        ]
    }
