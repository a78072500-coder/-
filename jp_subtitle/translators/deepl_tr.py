from __future__ import annotations

from .base import Translator


class DeepLTranslator(Translator):
    def __init__(self, api_key: str):
        import deepl

        if not api_key:
            raise ValueError("config.toml 의 [translator] deepl_api_key 를 입력하세요.")
        self._client = deepl.Translator(api_key)

    def translate(self, ja: str, history: list[tuple[str, str]]) -> str:
        context = "\n".join(j for j, _ in history) or None
        result = self._client.translate_text(
            ja, source_lang="JA", target_lang="KO", context=context
        )
        return result.text
