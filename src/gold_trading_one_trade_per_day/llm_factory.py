from __future__ import annotations

import os
import socket
from dataclasses import dataclass
from urllib.parse import urlparse

from crewai import LLM


@dataclass(frozen=True)
class LLMRuntimeConfig:
    analyst_model: str
    strategy_model: str
    timeout_sec: int
    max_retries: int
    temperature: float
    crew_max_rpm: int
    agent_max_rpm: int
    source: str


@dataclass(frozen=True)
class LLMRoute:
    analyst_model: str
    strategy_model: str
    analyst_route: str
    strategy_route: str
    analyst_fallback_used: bool
    analyst_local_available: bool
    strategy_auto_overridden: bool


def _source_for(var_name: str, default_used: bool) -> str:
    if var_name in os.environ:
        return f"env:{var_name}"
    return "default" if default_used else "derived"


def load_llm_runtime_config() -> LLMRuntimeConfig:
    analyst_default = "ollama/llama4:8b"
    strategy_default = "gemini/gemini-2.0-flash"
    timeout_default = 120
    max_retries_default = 5
    temp_default = 0.2
    crew_rpm_default = 8
    agent_rpm_default = 6

    analyst_model = os.getenv("AGENT_MODEL_ANALYST", analyst_default)
    strategy_model = os.getenv("AGENT_MODEL_STRATEGY", strategy_default)
    timeout_sec = int(os.getenv("AGENT_LLM_TIMEOUT_SEC", str(timeout_default)))
    max_retries = int(os.getenv("AGENT_LLM_MAX_RETRIES", str(max_retries_default)))
    temperature = float(os.getenv("AGENT_LLM_TEMPERATURE", str(temp_default)))
    crew_max_rpm = int(os.getenv("CREW_MAX_RPM", str(crew_rpm_default)))
    agent_max_rpm = int(os.getenv("AGENT_MAX_RPM", str(agent_rpm_default)))

    source = _source_for("AGENT_MODEL_ANALYST", analyst_model == analyst_default)
    if "AGENT_MODEL_STRATEGY" in os.environ:
        source = "env:AGENT_MODEL_STRATEGY"
    elif "AGENT_LLM_TIMEOUT_SEC" in os.environ:
        source = "env:AGENT_LLM_TIMEOUT_SEC"
    elif "AGENT_LLM_MAX_RETRIES" in os.environ:
        source = "env:AGENT_LLM_MAX_RETRIES"
    elif "CREW_MAX_RPM" in os.environ:
        source = "env:CREW_MAX_RPM"

    return LLMRuntimeConfig(
        analyst_model=analyst_model,
        strategy_model=strategy_model,
        timeout_sec=max(timeout_sec, 5),
        max_retries=max(max_retries, 0),
        temperature=temperature,
        crew_max_rpm=max(crew_max_rpm, 1),
        agent_max_rpm=max(agent_max_rpm, 1),
        source=source,
    )


def is_local_model_available(model_name: str) -> bool:
    if not model_name.lower().startswith("ollama/"):
        return True

    base_url = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
    parsed = urlparse(base_url)
    host = parsed.hostname or "localhost"
    port = parsed.port or 11434

    try:
        with socket.create_connection((host, port), timeout=0.8):
            return True
    except OSError:
        return False


def resolve_llm_route(mode: str, cfg: LLMRuntimeConfig) -> LLMRoute:
    local_ok = is_local_model_available(cfg.analyst_model)
    fallback_allowed_shadow = (
        os.getenv("ALLOW_ANALYST_GEMINI_FALLBACK_SHADOW", "true").lower() == "true"
    )
    fallback_allowed_live = (
        os.getenv("ALLOW_ANALYST_GEMINI_FALLBACK_PAPER_LIVE", "false").lower() == "true"
    )

    analyst_model = cfg.analyst_model
    strategy_model = cfg.strategy_model
    strategy_auto_overridden = False

    use_local_paper_strategy = (
        os.getenv("PAPER_USE_LOCAL_STRATEGY", "true").lower() == "true"
    )
    if mode == "paper" and use_local_paper_strategy:
        strategy_model = os.getenv("AGENT_MODEL_STRATEGY_PAPER", cfg.analyst_model)
        strategy_auto_overridden = strategy_model != cfg.strategy_model

    fallback_used = False
    if cfg.analyst_model.lower().startswith("ollama/") and not local_ok:
        if mode == "shadow" and fallback_allowed_shadow:
            analyst_model = strategy_model
            fallback_used = True
        elif mode in {"paper", "live"} and fallback_allowed_live:
            analyst_model = strategy_model
            fallback_used = True

    def route_label(model_name: str) -> str:
        if model_name.lower().startswith("ollama/"):
            return "local"
        if "gemini" in model_name.lower():
            return "gemini"
        return "other"

    return LLMRoute(
        analyst_model=analyst_model,
        strategy_model=strategy_model,
        analyst_route=route_label(analyst_model),
        strategy_route=route_label(strategy_model),
        analyst_fallback_used=fallback_used,
        analyst_local_available=local_ok,
        strategy_auto_overridden=strategy_auto_overridden,
    )


def build_llm(
    model_name: str,
    cfg: LLMRuntimeConfig,
) -> LLM:
    base_url = None
    if model_name.lower().startswith("ollama/"):
        base_url = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")

    return LLM(
        model=model_name,
        temperature=cfg.temperature,
        timeout=cfg.timeout_sec,
        max_retries=cfg.max_retries,
        base_url=base_url,
    )
