from __future__ import annotations

from abc import ABC, abstractmethod


class Translator(ABC):
    @abstractmethod
    def translate(self, ja: str, history: list[tuple[str, str]]) -> str:
        """일본어 한 문장을 한국어로 번역한다.

        history: 직전 (일본어, 한국어) 쌍들. 문맥을 쓰는 번역기만 사용한다.
        """


class NoTranslator(Translator):
    def translate(self, ja: str, history: list[tuple[str, str]]) -> str:
        return ""
