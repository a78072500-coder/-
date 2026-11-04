"""파이프라인, 화면, 기록이 함께 쓰는 자료형."""

from __future__ import annotations

from dataclasses import dataclass

THEM = "them"  # 상대방 (스피커로 나오는 소리)
ME = "me"      # 나 (마이크)


@dataclass
class Transcript:
    """음성인식 결과 하나."""

    text: str = ""
    low_confidence: bool = False
    avg_logprob: float | None = None
    language: str = "ja"
    sentence_end_sec: float | None = None  # 임시 자막에서 첫 문장이 끝난 위치 (이어지는 말이 있을 때만)


@dataclass
class Line:
    """대화 한 줄. 인식이 끝나면 만들어지고 번역이 끝나면 ko 가 채워진다."""

    uid: int
    ja: str  # 인식된 원문. 내 말이면 한국어일 수도 있다 (language 참고)
    speaker: str = THEM
    ko: str = ""
    reading: str = ""
    low_confidence: bool = False
    avg_logprob: float | None = None
    backchannel: bool = False  # うん, へえ 같은 맞장구
    language: str = "ja"


@dataclass
class Timing:
    """한 문장이 화면에 나오기까지 걸린 시간 (초)."""

    audio_sec: float = 0.0
    silence_wait: float = 0.0  # 말이 끝났다고 판단할 때까지 기다린 시간
    asr_wait: float = 0.0      # 앞 문장 인식이 끝나길 기다린 시간
    asr: float = 0.0           # 음성인식
    translate_wait: float = 0.0
    translate: float = 0.0     # 번역 API

    @property
    def to_subtitle(self) -> float:
        return self.silence_wait + self.asr_wait + self.asr

    @property
    def to_translation(self) -> float:
        return self.to_subtitle + self.translate_wait + self.translate
