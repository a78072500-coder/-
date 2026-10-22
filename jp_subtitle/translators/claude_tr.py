from __future__ import annotations

from .base import Translator

SYSTEM_PROMPT = """\
당신은 한국인 사용자가 일본인과 음성통화하는 동안 상대방의 말을 실시간으로 번역합니다.
입력은 음성인식 결과라 오타나 끊긴 문장이 있을 수 있습니다. 문맥상 가장 자연스러운 뜻으로 옮기세요.
- 상대방의 말투(반말/존댓말, 감정)를 한국어에서도 살립니다.
- 번역문만 출력합니다. 설명, 따옴표, 원문은 붙이지 않습니다.
- 지연이 중요하니 바로 번역문으로 답합니다."""


class ClaudeTranslator(Translator):
    def __init__(self, model: str):
        import anthropic

        # API 키는 ANTHROPIC_API_KEY 환경변수에서 읽는다
        self._client = anthropic.Anthropic(timeout=15.0)
        self._model = model

    def translate(self, ja: str, history: list[tuple[str, str]]) -> str:
        prompt = ""
        if history:
            lines = "\n".join(f"- {j} → {k}" for j, k in history)
            prompt += f"<recent_conversation>\n{lines}\n</recent_conversation>\n\n"
        prompt += f"번역할 문장:\n{ja}"

        response = self._client.beta.messages.create(
            model=self._model,
            max_tokens=1024,
            system=SYSTEM_PROMPT,
            output_config={"effort": "low"},
            betas=["server-side-fallback-2026-07-01"],
            fallbacks="default",
            messages=[{"role": "user", "content": prompt}],
        )
        if response.stop_reason == "refusal":
            return ""
        return "".join(b.text for b in response.content if b.type == "text").strip()
