"""
utils/llm_factory.py
Centralized LLM factory for the AC-RAG pipeline.

All pipeline nodes call get_llm() instead of instantiating a provider directly.
Switching the entire pipeline between OpenAI / Google / Anthropic is done by
changing LLM_PROVIDER in config/settings.py — nothing else needs to change.

Supported providers:
  "openai"    → ChatOpenAI       (gpt-4o by default)
  "google"    → ChatGoogleGenerativeAI (gemini-2.0-flash by default)
  "anthropic" → ChatAnthropic    (claude-sonnet-4-6 by default)

Environment variables required per provider:
  OpenAI    : OPENAI_API_KEY
  Google    : GOOGLE_API_KEY
  Anthropic : ANTHROPIC_API_KEY
"""

import logging
from functools import lru_cache
from typing import Optional

from langchain_core.language_models import BaseChatModel

from config.settings import (
    LLM_PROVIDER,
    LLM_MODEL,
    LLM_TEMPERATURE,
    LLM_MAX_TOKENS,
)

logger = logging.getLogger(__name__)


@lru_cache(maxsize=4)
def get_llm(
    provider: Optional[str] = None,
    model: Optional[str] = None,
    temperature: Optional[float] = None,
    max_tokens: Optional[int] = None,
) -> BaseChatModel:
    """
    Returns a cached LangChain chat model for the configured provider.

    Args:
        provider    : override LLM_PROVIDER from settings (useful in tests)
        model       : override LLM_MODEL from settings
        temperature : override LLM_TEMPERATURE
        max_tokens  : override LLM_MAX_TOKENS

    Returns:
        A LangChain BaseChatModel instance (OpenAI / Google / Anthropic).

    lru_cache: same (provider, model, temperature, max_tokens) combo returns
    the same instance — avoids re-initialising the client on every node call.
    """
    _provider    = (provider    or LLM_PROVIDER).lower()
    _model       = model        or LLM_MODEL
    _temperature = temperature  if temperature is not None else LLM_TEMPERATURE
    _max_tokens  = max_tokens   if max_tokens  is not None else LLM_MAX_TOKENS

    logger.info(
        "[LLMFactory] provider=%s | model=%s | temperature=%s",
        _provider, _model, _temperature,
    )

    if _provider == "openai":
        from langchain_openai import ChatOpenAI
        return ChatOpenAI(
            model=_model,
            temperature=_temperature,
            max_tokens=_max_tokens,
        )

    if _provider == "google":
        from langchain_google_genai import ChatGoogleGenerativeAI
        return ChatGoogleGenerativeAI(
            model=_model,
            temperature=_temperature,
            max_output_tokens=_max_tokens,
        )

    if _provider == "anthropic":
        from langchain_anthropic import ChatAnthropic
        return ChatAnthropic(
            model=_model,
            temperature=_temperature,
            max_tokens=_max_tokens,
        )

    if _provider == "groq":
        from langchain_groq import ChatGroq
        return ChatGroq(
            model=_model,
            temperature=_temperature,
            max_tokens=_max_tokens,
        )

    raise ValueError(
        f"Unknown LLM_PROVIDER: '{_provider}'. "
        "Choose 'openai', 'google', 'anthropic', or 'groq' in config/settings.py."
    )
