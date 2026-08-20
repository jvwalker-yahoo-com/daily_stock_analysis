import os
import requests
from typing import List, Dict, Any

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
        except Exception as e:
            print(f"Error connecting to Danelfin API for {clean_ticker}: {e}")

    # Sort watchlist head-to-head descending by AI Score
    rankings.sort(key=lambda x: x["ai_score"] if isinstance(x["ai_score"], (int, float)) else -1, reverse=True)
    return rankings
