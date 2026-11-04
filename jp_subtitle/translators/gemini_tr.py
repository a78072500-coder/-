"""Gemini 호출과 자막 번역.

Interactions API 는 기본으로 대화를 Google 서버에 저장하므로(무료 1일, 유료 55일)
통화 내용이 남지 않도록 항상 store=False 로 보낸다.
"""

from __future__ import annotations

import os

from .base import TRANSLATE_SYSTEM_PROMPT, Translator, build_translate_prompt

TRANSLATE_TIMEOUT_SEC = 15.0

# 생각 단계를 minimal 까지 낮출 수 있는 모델. 나머지는 low 가 가장 낮다.
_MINIMAL_THINKING = ("gemini-3.6-flash", "gemini-3.5-flash-lite", "gemini-3-flash-preview")


def fastest_thinking(model: str) -> str:
    return "minimal" if model.startswith(_MINIMAL_THINKING) else "low"


def has_gemini_key(api_key: str) -> bool:
    return bool(api_key or os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY"))


def make_client(api_key: str):
    if not has_gemini_key(api_key):
        raise ValueError("Gemini 키가 없어요. 설정 화면에서 넣어 주세요 (gemini_api_key).")
    from google import genai

    return genai.Client(api_key=api_key or None)


def generate(
    client,
    model: str,
    system: str,
    prompt: str,
    thinking: str,
    timeout: float,
    schema: dict | None = None,
):
    options: dict = {}
    if schema is not None:
        options["response_format"] = {
            "type": "text",
            "mime_type": "application/json",
            "schema": schema,
        }
    return client.interactions.create(
        model=model,
        system_instruction=system,
        input=prompt,
        store=False,
        generation_config={"thinking_level": thinking},
        timeout=timeout,
        **options,
    )


def gemini_error_message(error: Exception) -> str:
    """Gemini 오류를 사용자가 바로 조치할 수 있는 문장으로 바꾼다."""
    status = getattr(error, "status_code", None)
    text = str(error)
    if status in (401, 403) or (status == 400 and "API key" in text):
        return "Gemini API 키를 확인해 주세요. 설정 화면에서 바꿀 수 있어요."
    if status == 404:
        return "Gemini 모델을 찾지 못했어요. 설정 화면에서 모델 이름을 확인해 주세요."
    if status == 429:
        return "Gemini 요청 한도를 넘었어요. 잠시 뒤 다시 시도해 주세요. 무료 사용량이 끝났을 수도 있어요."
    if status is not None and status >= 500:
        return f"Gemini 서버 오류가 났어요 ({status}). 다시 시도해 주세요."
    if status is not None:
        return f"Gemini가 요청을 거절했어요 ({status}): {text[:200]}"
    if any(cls.__name__ == "APIConnectionError" for cls in type(error).__mro__):
        return "Gemini에 연결하지 못했어요. 인터넷 연결을 확인해 주세요."
    return f"Gemini 오류가 났어요: {text[:200]}"


class GeminiTranslator(Translator):
    def __init__(self, model: str, api_key: str = ""):
        self._client = make_client(api_key)
        self._model = model

    def translate(self, ja: str, history: list[tuple[str, str]]) -> str:
        try:
            result = generate(
                self._client,
                self._model,
                TRANSLATE_SYSTEM_PROMPT,
                build_translate_prompt(ja, history),
                fastest_thinking(self._model),
                TRANSLATE_TIMEOUT_SEC,
            )
        except Exception as e:
            raise RuntimeError(gemini_error_message(e)) from e
        return (result.output_text or "").strip()
