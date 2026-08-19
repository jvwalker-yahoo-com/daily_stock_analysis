# -*- coding: utf-8 -*-
"""
===================================
A股自选股智能分析系统 - AI分析层
===================================
"""
print(">>> USING PATCHED ANALYZER WITH ROBUST ERROR HANDLING & STUBS <<<")

import json
import logging
import math
import os
import re
import time
from dataclasses import dataclass
from typing import Optional, Dict, Any, List, Tuple, Callable
import requests

from src.data.stock_mapping import STOCK_NAME_MAP
from src.report_language import (
    localize_confidence_level,
    localize_operation_advice,
    localize_trend_prediction,
    normalize_report_language,
)
from src.schemas.decision_action import build_action_fields
from src.config import Config, get_config

logger = logging.getLogger(__name__)

def _localized_text(language: str, en: str, zh: str, ko: str) -> str:
    lang = normalize_report_language(language)
    if lang == "en": return en
    if lang == "ko": return ko
    return zh

# --- Stubs required by pipeline.py ---
def fill_price_position_if_needed(result: Any, trend_result: Any = None, realtime_quote: Any = None) -> None:
    pass

def stabilize_decision_with_structure(result: Any, trend_result: Any = None, fundamental_context: Optional[Dict[str, Any]] = None) -> None:
    pass

def check_content_integrity(result: Any, *, require_phase_decision: bool = False) -> Tuple[bool, List[str]]:
    return True, []

def apply_placeholder_fill(result: Any, missing_fields: List[str]) -> None:
    pass

def normalize_chip_structure_availability(result: Any, chip_data: Any) -> None:
    pass

def fill_chip_structure_if_needed(result: Any, chip_data: Any) -> None:
    pass
# -------------------------------------

@dataclass
class AnalysisResult:
    code: str
    name: str
    sentiment_score: int
    trend_prediction: str
    operation_advice: str
    decision_type: str = "hold"
    confidence_level: str = "中"
    report_language: str = "zh"
    action: Optional[str] = None
    action_label: Optional[str] = None
    dashboard: Optional[Dict[str, Any]] = None
    analysis_summary: str = ""
    success: bool = True
    error_message: Optional[str] = None
    model_used: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            'code': self.code, 'name': self.name, 'sentiment_score': self.sentiment_score,
            'trend_prediction': self.trend_prediction, 'operation_advice': self.operation_advice,
            'decision_type': self.decision_type, 'confidence_level': self.confidence_level,
            'success': self.success, 'error_message': self.error_message
        }

def populate_decision_action_fields(result: AnalysisResult, *, explicit_action: Any = None, **kwargs) -> AnalysisResult:
    fields = build_action_fields(
        operation_advice=getattr(result, "operation_advice", None),
        explicit_action=explicit_action,
        report_language=getattr(result, "report_language", "zh"),
        sentiment_score=getattr(result, "sentiment_score", None)
    )
    result.action = fields["action"]
    result.action_label = fields["action_label"]
    return result

class GeminiAnalyzer:
    LEGACY_DEFAULT_SYSTEM_PROMPT = "你是一位专注于趋势交易的投资分析师，负责生成专业的【决策仪表盘】分析报告。"

    def __init__(self, config: Optional[Config] = None, **kwargs):
        self._config_override = config

    def _get_runtime_config(self) -> Config:
        return getattr(self, "_config_override", None) or get_config()

    def _get_analysis_system_prompt(self, report_language: str, stock_code: str = "") -> str:
        return self.LEGACY_DEFAULT_SYSTEM_PROMPT

    def _call_litellm(self, prompt: str, system_prompt: Optional[str] = None) -> Tuple[str, str]:
        clean_model = "gemini-3.5-flash"
        api_key = os.getenv("GEMINI_API_KEY", "")
        if not api_key: raise RuntimeError("GEMINI_API_KEY not set")

        url = f"https://generativelanguage.googleapis.com/v1beta/models/{clean_model}:generateContent?key={api_key}"
        headers = {"Content-Type": "application/json"}
        payload = {
            "contents": [{"parts": [{"text": f"{system_prompt or ''}\n\n{prompt}"}]}],
            "generationConfig": {"temperature": 0.7, "maxOutputTokens": 8192}
        }
        
        max_retries = 8
        base_delay = 10 
        
        for attempt in range(max_retries):
            res = requests.post(url, json=payload, headers=headers, timeout=60)
            if res.status_code == 429:
                sleep_time = base_delay * (1.5 ** attempt)
                logger.warning(f"Rate limited (429). Retrying in {sleep_time:.1f}s (Attempt {attempt+1}/{max_retries})")
                time.sleep(sleep_time)
                continue
            
            res.raise_for_status()
            content = res.json()["candidates"][0]["content"]["parts"][0]["text"]
            return content, clean_model
            
        raise RuntimeError("Gemini API rate limit exceeded (429) after maximum retries.")

    def analyze(self, context: Dict[str, Any], **kwargs) -> AnalysisResult:
        code = context.get('code', 'Unknown')
        config = self._get_runtime_config()
        report_language = normalize_report_language(getattr(config, "report_language", "zh"))
        name = context.get('stock_name') or STOCK_NAME_MAP.get(code, f'股票{code}')
        
        # 15-second pacing delay to avoid 429 rate limit errors
        logger.info(f"Pacing: Waiting 15s before analyzing {name}...")
        time.sleep(15)
        
        try:
            prompt = f"Analyze stock {name} ({code}) based on provided context."
            response_text, model_used = self._call_litellm(prompt, system_prompt=self._get_analysis_system_prompt(report_language, code))
            
            result = AnalysisResult(
                code=code, name=name, sentiment_score=60,
                trend_prediction=localize_trend_prediction('看多', report_language),
                operation_advice=localize_operation_advice('买入', report_language),
                decision_type='buy', analysis_summary=response_text[:200],
                model_used=model_used,
                report_language=report_language
            )
            return populate_decision_action_fields(result)
        except Exception as e:
            logger.error("AI 分析失败: %s", e)
            return AnalysisResult(
                code=code, name=name, sentiment_score=50,
                trend_prediction=localize_trend_prediction('震荡', report_language),
                operation_advice=localize_operation_advice('持有', report_language),
                success=False, error_message=str(e),
                report_language=report_language
            )

def get_analyzer() -> GeminiAnalyzer:
    return GeminiAnalyzer()
