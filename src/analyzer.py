# -*- coding: utf-8 -*-
"""
===================================
A股自选股智能分析系统 - AI分析层
===================================

职责：
1. 封装 LLM 调用逻辑（通过 LiteLLM 统一调用 Gemini/Anthropic/OpenAI 等）
2. 结合技术面和消息面生成分析报告
3. 解析 LLM 响应为结构化 AnalysisResult
"""
print(">>> USING PATCHED ANALYZER <<<")

import json
import logging
import math
import re
import time
from dataclasses import dataclass
from typing import Optional, Dict, Any, List, Tuple, Callable

# ⭐ LiteLLM + Gemini patch
import litellm
litellm._turn_on_debug()

try:
    from litellm.types import Message
    Message.model_rebuild()
except Exception:
    pass

from json_repair import repair_json
from litellm import Router

from src.agent.llm_adapter import (
    get_thinking_extra_body,
    resolve_fallback_litellm_wire_models,
    register_fallback_model_pricing,
)
from src.agent.provider_trace import resolved_model_provider_identity
from src.agent.skills.defaults import CORE_TRADING_SKILL_POLICY_ZH
from src.config import (
    Config,
    extra_litellm_params,
    get_api_keys_for_model,
    get_config,
    get_configured_llm_models,
    resolve_news_window_days,
)
from src.llm.hermes import (
    HERMES_CHANNEL_NAME,
    build_hermes_redaction_values,
    canonicalize_hermes_model_ref,
    filter_non_hermes_deployments,
    hermes_blocked_route_candidates,
    is_masked_secret_placeholder,
    open_hermes_no_proxy_client,
    route_deployment_origins,
    route_has_hermes,
    sanitize_hermes_error_text,
)
from src.llm.generation_params import apply_litellm_generation_params
from src.llm.errors import call_litellm_with_param_recovery
from src.llm.backend_registry import (
    LOCAL_CLI_GENERATION_BACKEND_IDS,
    LITELLM_BACKEND_ID,
    resolve_generation_backend_id,
    resolve_generation_fallback_backend_id,
)
from src.llm.backend_factory import create_generation_backend
from src.llm.generation_backend import (
    GenerationBackend,
    GenerationError,
    GenerationErrorCode,
)
from src.llm.usage import (
    attach_legacy_message_stability_audit,
    attach_message_hmacs,
    extract_usage_payload,
    normalize_litellm_usage,
    should_persist_usage_telemetry,
)
from src.llm.local_cli_backend import redact_diagnostic_text
from src.llm.provider_cache import (
    apply_prompt_cache_hints,
    build_provider_cache_route_context,
    filter_prompt_cache_telemetry,
)
from src.llm.response_content import strip_leading_think_wrapper
from src.storage import persist_llm_usage
from src.data.stock_mapping import STOCK_NAME_MAP
from src.report_language import (
    get_signal_level,
    get_no_data_text,
    get_placeholder_text,
    get_unknown_text,
    get_chip_unavailable_text,
    infer_decision_type_from_advice,
    is_chip_placeholder_value,
    localize_chip_health,
    localize_confidence_level,
    localize_operation_advice,
    localize_trend_prediction,
    normalize_report_language,
)
from src.schemas.decision_action import build_action_fields
from src.schemas.decision_scale import (
    CANONICAL_DECISION_SCALE_PROMPT_ZH,
    score_band_metadata,
)
from src.schemas.report_schema import AnalysisReportSchema
from src.market_context import detect_market, get_market_role, get_market_guidelines
from src.services.daily_market_context import format_daily_market_context_prompt_section
from src.market_phase_prompt import format_market_phase_prompt_section
from src.market_structure_prompt import format_market_structure_prompt_section

logger = logging.getLogger(__name__)


def _localized_text(language: str, en: str, zh: str, ko: str) -> str:
    """Helper to return localized text matching report_language."""
    lang = normalize_report_language(language)
    if lang == "en":
        return en
    if lang == "ko":
        return ko
    return zh


def _normalize_risk_warning_values(value: Any) -> List[str]:
    if value is None:
        return []
    if isinstance(value, str):
        text = value.strip()
        return [text] if text else []
    if isinstance(value, (list, tuple, set)):
        normalized: List[str] = []
        for item in value:
            normalized.extend(_normalize_risk_warning_values(item))
        return normalized
    if isinstance(value, dict):
        if not value:
            return []
        try:
            dumped = json.dumps(value, ensure_ascii=False)
            text = dumped.strip()
        except (TypeError, ValueError):
            text = str(value).strip()
        return [text] if text else []
    text = str(value).strip()
    return [text] if text else []


def _today_has_realtime_overlay(today: Any) -> bool:
    if not isinstance(today, dict):
        return False
    data_source = today.get("data_source") or today.get("dataSource")
    if isinstance(data_source, str) and data_source.startswith("realtime:"):
        return True
    if today.get("is_partial_bar") is True or today.get("isPartialBar") is True:
        return True
    if today.get("is_estimated") is True or today.get("isEstimated") is True:
        return True
    return bool(today.get("estimated_fields") or today.get("estimatedFields"))


def _today_looks_complete_daily_bar(
    context: Dict[str, Any],
    phase_context: Dict[str, Any],
) -> bool:
    today = context.get("today")
    if (
        not isinstance(today, dict)
        or today.get("close") in (None, "")
        or _today_has_realtime_overlay(today)
    ):
        return False

    effective_date = phase_context.get("effective_daily_bar_date")
    today_date = today.get("date") or today.get("trade_date") or context.get("date")
    if effective_date and today_date and str(today_date) != str(effective_date):
        return False
    return True


def _phase_aware_quote_labels(context: Dict[str, Any]) -> Tuple[str, str]:
    phase_context = context.get("market_phase_context")
    if not isinstance(phase_context, dict):
        return "今日行情", "收盘价"

    phase = str(phase_context.get("phase") or "").strip()
    if phase in {"premarket", "non_trading"}:
        today = context.get("today")
        if _today_looks_complete_daily_bar(context, phase_context):
            return "上一完整交易日行情", "上一完整交易日收盘价"
        if _today_has_realtime_overlay(today):
            return "最新行情", "实时估算价"
        if isinstance(today, dict) and today.get("close") not in (None, ""):
            return "最新行情", "最新价"
        return "今日行情", "收盘价"

    if (
        phase in {"intraday", "lunch_break", "closing_auction"}
        and phase_context.get("is_partial_bar") is True
    ):
        return "最新行情", "盘中估算价"

    return "今日行情", "收盘价"


def _should_hide_regular_session_ohlc(context: Dict[str, Any]) -> bool:
    phase_context = context.get("market_phase_context")
    if not isinstance(phase_context, dict):
        return False

    phase = str(phase_context.get("phase") or "").strip()
    return phase in {"premarket", "non_trading"} and not _today_looks_complete_daily_bar(
        context,
        phase_context,
    )


def _legacy_market_group(stock_code: Any) -> str:
    code = str(stock_code or "").strip()
    if not code or code.lower() == "unknown":
        return "unknown"
    market = detect_market(code)
    return market if market in {"cn", "hk", "us"} else "unknown"


def _legacy_audit_marker_specs(
    context: Dict[str, Any],
    *,
    code: str,
    stock_name: str,
    report_language: str,
    news_context: Optional[str],
    analysis_context_pack_summary: Optional[str],
) -> List[Dict[str, Any]]:
    markers: List[Dict[str, Any]] = []

    def add(marker_name: str, value: Any) -> None:
        if value is None:
            return
        text = str(value).strip()
        if not text:
            return
        markers.append(
            {
                "marker_name": marker_name,
                "message_role": "user",
                "text": text,
            }
        )

    add("stock_code", code)
    add("stock_name", stock_name)
    add("analysis_date", context.get("date"))
    add("market_phase", "## Market Phase Context" if report_language in ("en", "ko") else "## 市场阶段上下文")
    add("daily_market_context", "## Daily Market Context" if report_language in ("en", "ko") else "## 大盘环境摘要")
    add("market_structure_context", "## Market Structure Context" if report_language in ("en", "ko") else "## 市场结构上下文")
    add("analysis_context_pack", analysis_context_pack_summary)
    add("quote", "## 📈 技术面数据")
    add("news_context", "## 📰 舆情情报" if news_context else None)
    return markers


class _LiteLLMStreamError(RuntimeError):
    def __init__(self, message: str, *, partial_received: bool = False):
        super().__init__(message)
        self.partial_received = partial_received


class _AllModelsFailedError(Exception):
    def __init__(
        self,
        message: str,
        *,
        last_response_text: Optional[str] = None,
        last_model: Optional[str] = None,
        last_usage: Optional[Dict[str, Any]] = None,
    ):
        super().__init__(message)
        self.last_response_text = last_response_text
        self.last_model = last_model
        self.last_usage = last_usage or {}


from src.utils.data_processing import normalize_report_signal_attribution


def check_content_integrity(
    result: "AnalysisResult",
    *,
    require_phase_decision: bool = False,
) -> Tuple[bool, List[str]]:
    missing: List[str] = []

    def _is_blank_text(value: Any) -> bool:
        if value is None:
            return True
        if isinstance(value, str):
            return not value.strip()
        return True

    def _is_invalid_risk_alerts(value: Any) -> bool:
        return not isinstance(value, list)

    def _is_invalid_stop_loss(value: Any) -> bool:
        if value is None:
            return True
        if isinstance(value, (list, tuple, dict)):
            return True
        if isinstance(value, str):
            return not value.strip()
        return False

    if result.sentiment_score is None:
        missing.append("sentiment_score")
    advice = result.operation_advice
    if not advice or not isinstance(advice, str) or _is_blank_text(advice):
        missing.append("operation_advice")
    summary = result.analysis_summary
    if not summary or not isinstance(summary, str) or _is_blank_text(summary):
        missing.append("analysis_summary")
    dash = result.dashboard if isinstance(result.dashboard, dict) else {}
    core = dash.get("core_conclusion")
    core = core if isinstance(core, dict) else {}
    if _is_blank_text(core.get("one_sentence")):
        missing.append("dashboard.core_conclusion.one_sentence")
    intel = dash.get("intelligence")
    intel = intel if isinstance(intel, dict) else None
    if intel is None or _is_invalid_risk_alerts(intel.get("risk_alerts")):
        missing.append("dashboard.intelligence.risk_alerts")
    if result.decision_type in ("buy", "hold"):
        battle = dash.get("battle_plan")
        battle = battle if isinstance(battle, dict) else {}
        sp = battle.get("sniper_points")
        sp = sp if isinstance(sp, dict) else {}
        stop_loss = sp.get("stop_loss")
        if _is_invalid_stop_loss(stop_loss):
            missing.append("dashboard.battle_plan.sniper_points.stop_loss")
    if require_phase_decision:
        phase_decision = dash.get("phase_decision")
        phase_decision = phase_decision if isinstance(phase_decision, dict) else {}
        if not isinstance(phase_decision.get("phase_context"), dict):
            missing.append("dashboard.phase_decision.phase_context")
        if _is_blank_text(phase_decision.get("action_window")):
            missing.append("dashboard.phase_decision.action_window")
        if _is_blank_text(phase_decision.get("immediate_action")):
            missing.append("dashboard.phase_decision.immediate_action")
        if not isinstance(phase_decision.get("watch_conditions"), list):
            missing.append("dashboard.phase_decision.watch_conditions")
        if _is_blank_text(phase_decision.get("next_check_time")):
            missing.append("dashboard.phase_decision.next_check_time")
        if _is_blank_text(phase_decision.get("confidence_reason")):
            missing.append("dashboard.phase_decision.confidence_reason")
        if not isinstance(phase_decision.get("data_limitations"), list):
            missing.append("dashboard.phase_decision.data_limitations")
    return len(missing) == 0, missing


def apply_placeholder_fill(result: "AnalysisResult", missing_fields: List[str]) -> None:
    def _is_blank_text(value: Any) -> bool:
        if value is None:
            return True
        if isinstance(value, str):
            return not value.strip()
        return True

    def _is_invalid_risk_alerts(value: Any) -> bool:
        return not isinstance(value, list)

    def _is_invalid_stop_loss(value: Any) -> bool:
        if value is None:
            return True
        if isinstance(value, (list, tuple, dict)):
            return True
        if isinstance(value, str):
            return not value.strip()
        return False

    report_language = normalize_report_language(getattr(result, "report_language", "zh"))
    placeholder = get_placeholder_text(report_language)
    phase_decision_placeholders = {
        "dashboard.phase_decision.action_window": _localized_text(
            report_language,
            en="Model did not provide a phase action window",
            zh="模型未提供阶段化行动窗口",
            ko="모델이 단계별 행동 구간을 제공하지 않았습니다",
        ),
        "dashboard.phase_decision.immediate_action": _localized_text(
            report_language,
            en="Model did not provide a phase-aware immediate action",
            zh="模型未提供阶段化即时动作",
            ko="모델이 단계 인식 즉시 동작을 제공하지 않았습니다",
        ),
        "dashboard.phase_decision.next_check_time": _localized_text(
            report_language,
            en="Model did not provide a next check point",
            zh="模型未提供下一次检查点",
            ko="모델이 다음 점검 시점을 제공하지 않았습니다",
        ),
        "dashboard.phase_decision.confidence_reason": _localized_text(
            report_language,
            en="Model did not provide a phase confidence rationale",
            zh="模型未提供阶段化置信度理由",
            ko="모델이 단계별 신뢰도 근거를 제공하지 않았습니다",
        ),
    }
    for field in missing_fields:
        if field == "sentiment_score":
            result.sentiment_score = 50
        elif field == "operation_advice":
            if _is_blank_text(result.operation_advice):
                result.operation_advice = placeholder
        elif field == "analysis_summary":
            if _is_blank_text(result.analysis_summary):
                result.analysis_summary = placeholder
        elif field == "dashboard.core_conclusion.one_sentence":
            if not result.dashboard:
                result.dashboard = {}
            core = result.dashboard.get("core_conclusion")
            if not isinstance(core, dict):
                core = {}
                result.dashboard["core_conclusion"] = core
            fallback_sentence = (
                result.analysis_summary
                or result.operation_advice
                or placeholder
            )
            if _is_blank_text(core.get("one_sentence")):
                result.dashboard["core_conclusion"]["one_sentence"] = fallback_sentence
        elif field == "dashboard.intelligence.risk_alerts":
            if not result.dashboard:
                result.dashboard = {}
            intelligence = result.dashboard.get("intelligence")
            if not isinstance(intelligence, dict):
                intelligence = {}
                result.dashboard["intelligence"] = intelligence
            if _is_invalid_risk_alerts(intelligence.get("risk_alerts")):
                risk_warning_values = _normalize_risk_warning_values(result.risk_warning)
                intelligence["risk_alerts"] = risk_warning_values
        elif field == "dashboard.battle_plan.sniper_points.stop_loss":
            if not result.dashboard:
                result.dashboard = {}
            battle_plan = result.dashboard.get("battle_plan")
            if not isinstance(battle_plan, dict):
                battle_plan = {}
                result.dashboard["battle_plan"] = battle_plan
            sniper_points = battle_plan.get("sniper_points")
            if not isinstance(sniper_points, dict):
                sniper_points = {}
                battle_plan["sniper_points"] = sniper_points
            if _is_invalid_stop_loss(sniper_points.get("stop_loss")):
                sniper_points["stop_loss"] = placeholder
        elif field.startswith("dashboard.phase_decision."):
            if not result.dashboard:
                result.dashboard = {}
            phase_decision = result.dashboard.get("phase_decision")
            if not isinstance(phase_decision, dict):
                phase_decision = {}
                result.dashboard["phase_decision"] = phase_decision
            if field == "dashboard.phase_decision.phase_context":
                if not isinstance(phase_decision.get("phase_context"), dict):
                    phase_decision["phase_context"] = {}
            elif field == "dashboard.phase_decision.watch_conditions":
                if not isinstance(phase_decision.get("watch_conditions"), list):
                    phase_decision["watch_conditions"] = []
            elif field == "dashboard.phase_decision.data_limitations":
                if not isinstance(phase_decision.get("data_limitations"), list):
                    phase_decision["data_limitations"] = []
            elif field in phase_decision_placeholders:
                if _is_blank_text(phase_decision.get(field.rsplit(".", 1)[-1])):
                    phase_decision[field.rsplit(".", 1)[-1]] = phase_decision_placeholders[field]


def _safe_float(v: Any, default: float = 0.0) -> float:
    if v is None:
        return default
    if isinstance(v, (int, float)):
        try:
            return default if math.isnan(float(v)) else float(v)
        except (ValueError, TypeError):
            return default
    try:
        return float(str(v).strip())
    except (TypeError, ValueError):
        return default


def _coerce_chip_metric(v: Any) -> Optional[float]:
    if v is None:
        return None
    try:
        numeric = float(v)
    except (TypeError, ValueError):
        try:
            numeric = float(str(v).strip())
        except (TypeError, ValueError):
            return None
    return None if math.isnan(numeric) else numeric


def _has_meaningful_chip_data(chip_data: Any) -> bool:
    if not chip_data:
        return False
    if hasattr(chip_data, "avg_cost"):
        avg_cost = _coerce_chip_metric(getattr(chip_data, "avg_cost", None))
        concentration_90 = _coerce_chip_metric(getattr(chip_data, "concentration_90", None))
        concentration_70 = _coerce_chip_metric(getattr(chip_data, "concentration_70", None))
    else:
        d = chip_data if isinstance(chip_data, dict) else {}
        avg_cost = _coerce_chip_metric(d.get("avg_cost"))
        concentration_90_value = d.get("concentration_90")
        if concentration_90_value is None:
            concentration_90_value = d.get("concentration")
        concentration_90 = _coerce_chip_metric(concentration_90_value)
        concentration_70 = _coerce_chip_metric(d.get("concentration_70"))
    return (
        avg_cost is not None
        and avg_cost > 0
        and (
            (concentration_90 is not None and concentration_90 >= 0)
            or (concentration_70 is not None and concentration_70 >= 0)
        )
    )


def _mark_chip_structure_unavailable(result: "AnalysisResult", language: str) -> None:
    if not result or not isinstance(result.dashboard, dict):
        return
    data_perspective = result.dashboard.get("data_perspective")
    if not isinstance(data_perspective, dict):
        return
    data_perspective["chip_structure"] = {}
    data_perspective["chip_unavailable_reason"] = get_chip_unavailable_text(language)


def normalize_chip_structure_availability(result: "AnalysisResult", chip_data: Any) -> None:
    if not result:
        return
    language = getattr(result, "report_language", "zh")
    if _has_meaningful_chip_data(chip_data):
        return
    _mark_chip_structure_unavailable(result, language)


def _capital_flow_bias_with_status(
    fundamental_context: Optional[Dict[str, Any]],
) -> tuple[str, str]:
    if not isinstance(fundamental_context, dict):
        return "unavailable", "invalid_context"
    block = fundamental_context.get("capital_flow")
    if not isinstance(block, dict):
        return "unavailable", "capital_flow_block_missing"
    status = str(block.get("status") or "").strip().lower()
    return "unavailable", status or "not_supported"


def get_stock_name_multi_source(
    stock_code: str,
    context: Optional[Dict] = None,
    data_manager = None
) -> str:
    if context:
        if context.get('stock_name'):
            name = context['stock_name']
            if name and not name.startswith('股票'):
                return name
        if 'realtime' in context and context['realtime'].get('name'):
            return context['realtime']['name']

    if stock_code in STOCK_NAME_MAP:
        return STOCK_NAME_MAP[stock_code]

    return f'股票{stock_code}'


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
    trend_analysis: str = ""
    short_term_outlook: str = ""
    medium_term_outlook: str = ""
    technical_analysis: str = ""
    ma_analysis: str = ""
    volume_analysis: str = ""
    pattern_analysis: str = ""
    fundamental_analysis: str = ""
    sector_position: str = ""
    company_highlights: str = ""
    news_summary: str = ""
    market_sentiment: str = ""
    hot_topics: str = ""
    analysis_summary: str = ""
    key_points: str = ""
    risk_warning: str = ""
    buy_reason: str = ""
    market_snapshot: Optional[Dict[str, Any]] = None
    raw_response: Optional[str] = None
    search_performed: bool = False
    data_sources: str = ""
    success: bool = True
    error_message: Optional[str] = None
    current_price: Optional[float] = None
    change_pct: Optional[float] = None
    model_used: Optional[str] = None
    query_id: Optional[str] = None
    fundamental_context: Optional[Dict[str, Any]] = None
    market_structure_context: Optional[Dict[str, Any]] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            'code': self.code,
            'name': self.name,
            'sentiment_score': self.sentiment_score,
            'trend_prediction': self.trend_prediction,
            'operation_advice': self.operation_advice,
            'decision_type': self.decision_type,
            'confidence_level': self.confidence_level,
            'report_language': self.report_language,
            'action': self.action,
            'action_label': self.action_label,
            'dashboard': self.dashboard,
            'success': self.success,
            'error_message': self.error_message,
        }


def populate_decision_action_fields(
    result: AnalysisResult,
    *,
    explicit_action: Any = None,
    report_type: Any = None,
    use_existing_action: bool = True,
    align_with_score: bool = True,
) -> AnalysisResult:
    action_source = explicit_action
    if action_source is None and use_existing_action:
        action_source = getattr(result, "action", None)

    fields = build_action_fields(
        operation_advice=getattr(result, "operation_advice", None),
        explicit_action=action_source,
        report_type=report_type,
        report_language=getattr(result, "report_language", "zh"),
        sentiment_score=getattr(result, "sentiment_score", None),
        guardrail_reason=getattr(result, "guardrail_reason", None),
        align_with_score=align_with_score,
    )
    result.action = fields["action"]
    result.action_label = fields["action_label"]
    return result


class GeminiAnalyzer:
    LEGACY_DEFAULT_SYSTEM_PROMPT = """你是一位专注于趋势交易的{market_placeholder}投资分析师，负责生成专业的【决策仪表盘】分析报告。

{guidelines_placeholder}

""" + CORE_TRADING_SKILL_POLICY_ZH + """

""" + CANONICAL_DECISION_SCALE_PROMPT_ZH + """

## 输出格式：决策仪表盘 JSON

请严格按照以下 JSON 格式输出，这是一个完整的【决策仪表盘】：

```json
{
    "stock_name": "股票中文名称",
    "sentiment_score": 0-100整数,
    "trend_prediction": "强烈看多/看多/震荡/看空/强烈看空",
    "operation_advice": "买入/加仓/持有/减仓/卖出/观望",
    "decision_type": "buy/hold/sell",
    "action": "buy/add/hold/reduce/sell/watch/avoid/alert",
    "guardrail_reason": "",
    "confidence_level": "高/中/低",
    "dashboard": {},
    "analysis_summary": "100字综合分析摘要"
}
