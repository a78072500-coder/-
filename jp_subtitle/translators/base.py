from __future__ import annotations

from abc import ABC, abstractmethod

# AI 번역기(Claude, Gemini)가 함께 쓰는 지시문
TRANSLATE_SYSTEM_PROMPT = """\
당신은 한국인 사용자가 일본인과 음성통화하는 동안 상대방의 말을 실시간으로 번역합니다.
입력은 음성인식 결과라 오타나 끊긴 문장이 있을 수 있습니다. 문맥상 가장 자연스러운 뜻으로 옮기세요.
최근 대화에서 [나] 로 시작하는 줄은 사용자가 한 말이고, 나머지는 상대방이 한 말입니다.
- 상대방의 말투(반말/존댓말, 감정)를 한국어에서도 살립니다.
- 번역문만 출력합니다. 설명, 따옴표, 원문은 붙이지 않습니다.
- 지연이 중요하니 바로 번역문으로 답합니다."""


def _context_line(ja: str, ko: str) -> str:
    return f"- {ja} → {ko}" if ko else f"- {ja}"


def build_translate_prompt(ja: str, history: list[tuple[str, str]]) -> str:
    prompt = ""
    if history:
        lines = "\n".join(_context_line(j, k) for j, k in history)
        prompt += f"<recent_conversation>\n{lines}\n</recent_conversation>\n\n"
    return prompt + f"번역할 문장:\n{ja}"


class Translator(ABC):
    @abstractmethod
    def translate(self, ja: str, history: list[tuple[str, str]]) -> str:
        """일본어 한 문장을 한국어로 번역한다.

        history: 직전 (일본어, 한국어) 쌍들. 문맥을 쓰는 번역기만 사용한다.
        """


class NoTranslator(Translator):
    def translate(self, ja: str, history: list[tuple[str, str]]) -> str:
        return ""
