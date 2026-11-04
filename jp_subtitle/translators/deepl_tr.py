from __future__ import annotations

from .base import Translator


class DeepLTranslator(Translator):
    def __init__(self, api_key: str):
        import deepl

        if not api_key:
            raise ValueError("DeepL 키가 없어요. 설정 화면에서 넣어 주세요 (deepl_api_key).")
        self._client = deepl.Translator(api_key)

    def translate(self, ja: str, history: list[tuple[str, str]]) -> str:
        context = "\n".join(j for j, _ in history) or None
        result = self._client.translate_text(
            ja, source_lang="JA", target_lang="KO", context=context
        )
        return result.text
