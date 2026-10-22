"""VAD 로 오디오 블록 흐름을 발화(문장) 단위로 나눈다."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import Callable, Iterator

import numpy as np

from .audio_capture import BLOCK_SAMPLES, SAMPLE_RATE
from .config import SegmenterConfig

BLOCK_SEC = BLOCK_SAMPLES / SAMPLE_RATE
PRE_ROLL_BLOCKS = 6  # 말 시작 직전 ~0.2초도 함께 넣어 첫 음절이 잘리지 않게


@dataclass
class Utterance:
    audio: np.ndarray
    final: bool        # False 면 말하는 중의 임시 조각
    utterance_id: int  # 같은 발화의 임시/확정본은 같은 id


class Segmenter:
    def __init__(self, cfg: SegmenterConfig, speech_prob: Callable[[np.ndarray], float]):
        self.cfg = cfg
        self.speech_prob = speech_prob
        self._pre_roll: deque[np.ndarray] = deque(maxlen=PRE_ROLL_BLOCKS)
        self._blocks: list[np.ndarray] = []
        self._in_speech = False
        self._silent_blocks = 0
        self._since_partial = 0
        self._next_id = 0

    def _emit(self, final: bool) -> Utterance:
        return Utterance(np.concatenate(self._blocks), final, self._next_id)

    def _finish(self) -> Iterator[Utterance]:
        # 끝의 무음은 잘라내고 길이 확인
        speech = self._blocks[: len(self._blocks) - self._silent_blocks] or self._blocks
        if len(speech) * BLOCK_SEC >= self.cfg.min_utterance_sec:
            self._blocks = speech
            yield self._emit(final=True)
            self._next_id += 1
        self._blocks = []
        self._in_speech = False
        self._silent_blocks = 0
        self._since_partial = 0

    def feed(self, block: np.ndarray) -> Iterator[Utterance]:
        is_speech = self.speech_prob(block) >= self.cfg.vad_threshold

        if not self._in_speech:
            if is_speech:
                self._in_speech = True
                self._blocks = [*self._pre_roll, block]
                self._pre_roll.clear()
            else:
                self._pre_roll.append(block)
            return

        self._blocks.append(block)
        self._silent_blocks = 0 if is_speech else self._silent_blocks + 1
        self._since_partial += 1

        if (
            self._silent_blocks * BLOCK_SEC >= self.cfg.silence_end_sec
            or len(self._blocks) * BLOCK_SEC >= self.cfg.max_utterance_sec
        ):
            yield from self._finish()
        elif (
            self.cfg.partial_interval_sec > 0
            and self._since_partial * BLOCK_SEC >= self.cfg.partial_interval_sec
        ):
            self._since_partial = 0
            yield self._emit(final=False)

    def flush(self) -> Iterator[Utterance]:
        if self._in_speech:
            yield from self._finish()


def energy_speech_prob(block: np.ndarray) -> float:
    """간단한 음량 기반 판정 (테스트용 / silero 를 못 쓸 때)."""
    rms = float(np.sqrt(np.mean(block**2)))
    return min(1.0, rms / 0.02)


class SileroSpeechProb:
    """faster-whisper 에 들어있는 silero VAD 로 말소리 확률을 계산한다.

    faster-whisper 의 모델 래퍼는 호출마다 내부 상태를 초기화하므로
    최근 0.5초 창을 함께 넣고 마지막 프레임의 확률을 쓴다.
    """

    WINDOW_BLOCKS = 16

    def __init__(self) -> None:
        from faster_whisper.vad import get_vad_model

        self._model = get_vad_model()
        self._window: deque[np.ndarray] = deque(maxlen=self.WINDOW_BLOCKS)

    def __call__(self, block: np.ndarray) -> float:
        self._window.append(block)
        audio = np.concatenate(self._window)
        probs = self._model(audio, num_samples=BLOCK_SAMPLES)
        return float(np.asarray(probs).reshape(-1)[-1])
