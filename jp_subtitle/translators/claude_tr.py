from __future__ import annotations

from .base import TRANSLATE_SYSTEM_PROMPT, Translator, build_translate_prompt


class ClaudeTranslator(Translator):
    def __init__(self, model: str, api_key: str = ""):
        import anthropic

        # api_key 가 비어 있으면 SDK 가 ANTHROPIC_API_KEY 환경변수에서 읽는다
        self._client = anthropic.Anthropic(api_key=api_key or None, timeout=15.0)
        self._model = model

    def translate(self, ja: str, history: list[tuple[str, str]]) -> str:
        response = self._client.beta.messages.create(
            model=self._model,
            max_tokens=1024,
            system=TRANSLATE_SYSTEM_PROMPT,
            output_config={"effort": "low"},
            betas=["server-side-fallback-2026-07-01"],
            fallbacks="default",
            messages=[{"role": "user", "content": build_translate_prompt(ja, history)}],
        )
        if response.stop_reason == "refusal":
            return ""
        return "".join(b.text for b in response.content if b.type == "text").strip()
