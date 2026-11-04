from __future__ import annotations

from ..config import TranslatorConfig
from .base import NoTranslator, Translator


def create_translator(cfg: TranslatorConfig) -> Translator:
    if cfg.engine == "deepl":
        from .deepl_tr import DeepLTranslator

        return DeepLTranslator(cfg.deepl_api_key)
    if cfg.engine == "claude":
        from .claude_tr import ClaudeTranslator

        return ClaudeTranslator(cfg.claude_model, cfg.anthropic_api_key)
    if cfg.engine == "gemini":
        from .gemini_tr import GeminiTranslator

        return GeminiTranslator(cfg.gemini_model, cfg.gemini_api_key)
    if cfg.engine == "none":
        return NoTranslator()
    raise ValueError(f"알 수 없는 번역 엔진: {cfg.engine} (deepl / claude / gemini / none)")


__all__ = ["Translator", "NoTranslator", "create_translator"]
