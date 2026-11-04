"""VAD 로 오디오 블록 흐름을 발화(문장) 단위로 나눈다."""

from __future__ import annotations

import time
from collections import deque
from dataclasses import dataclass, field
from typing import Callable, Iterator

import numpy as np

from .audio_capture import BLOCK_SAMPLES, SAMPLE_RATE
from .config import SegmenterConfig
from .models import THEM

BLOCK_SEC = BLOCK_SAMPLES / SAMPLE_RATE
PRE_ROLL_BLOCKS = 6  # 말 시작 직전 ~0.2초도 함께 넣어 첫 음절이 잘리지 않게
PAUSE_LOOKBACK_SEC = 4.0  # 길이 제한에 걸리면 최근 이만큼 안에서 쉬는 곳을 찾는다
PAUSE_MIN_BLOCKS = 5      # 이 정도(~0.16초) 쉬면 숨 고른 곳으로 본다
QUIET_LOOKBACK_BLOCKS = int(1.0 / BLOCK_SEC)  # 쉬는 곳이 없으면 최근 1초 중 가장 조용한 곳에서 자른다
SENTENCE_SNAP_BLOCKS = int(0.15 / BLOCK_SEC)  # 문장 끝 시각 근처에서 가장 조용한 곳을 찾는 범위


@dataclass
class Utterance:
    audio: np.ndarray
    final: bool        # False 면 말하는 중의 임시 조각
    utterance_id: int  # 같은 발화의 임시/확정본은 같은 id
    speaker: str = THEM
    silence_wait: float = 0.0  # 말이 끝난 뒤 판단하기까지 기다린 시간 (초)
    emitted_at: float = field(default_factory=time.monotonic)


class Segmenter:
    def __init__(
        self,
        cfg: SegmenterConfig,
        speech_prob: Callable[[np.ndarray], float],
        speaker: str = THEM,
        first_id: int = 0,
    ):
        self.cfg = cfg  # 통화 중에 바꿔도 다음 블록부터 적용된다
        self.speech_prob = speech_prob
        self.speaker = speaker
        self._pre_roll: deque[np.ndarray] = deque(maxlen=PRE_ROLL_BLOCKS)
        self._blocks: list[np.ndarray] = []
        self._voiced: list[bool] = []  # 블록별 말소리 여부 (쉬는 곳을 찾을 때 쓴다)
        self._in_speech = False
        self._silent_blocks = 0
        self._since_partial = 0
        self._next_id = first_id
        self._pending_cut: tuple[int, float] | None = None

    def request_cut(self, utterance_id: int, at_sec: float) -> None:
        """임시 자막에서 찾은 문장 끝을 알려 준다. 다음 블록을 처리할 때 그 자리에서 자른다.

        인식 스레드에서 부르므로 값 하나만 바꾸고, 자르는 일은 캡처 스레드에서 한다.
        """
        self._pending_cut = (utterance_id, at_sec)

    def _emit(self, final: bool, silence_wait: float = 0.0) -> Utterance:
        return Utterance(
            np.concatenate(self._blocks), final, self._next_id, self.speaker, silence_wait
        )

    def _finish(self) -> Iterator[Utterance]:
        # 끝의 무음은 잘라내고 길이 확인
        waited = self._silent_blocks * BLOCK_SEC
        speech = self._blocks[: len(self._blocks) - self._silent_blocks] or self._blocks
        if len(speech) * BLOCK_SEC >= self.cfg.min_utterance_sec:
            self._blocks = speech
            yield self._emit(final=True, silence_wait=waited)
            self._next_id += 1
        self._reset()

    def _reset(self) -> None:
        self._blocks = []
        self._voiced = []
        self._in_speech = False
        self._silent_blocks = 0
        self._since_partial = 0

    def _long_enough(self, cut: int | None) -> bool:
        return cut is not None and cut * BLOCK_SEC >= self.cfg.min_utterance_sec

    def _find_pause(self) -> int | None:
        """최근 구간에서 가장 늦게 쉰 곳의 블록 위치. 없으면 None."""
        count = len(self._voiced)
        lookback = min(PAUSE_LOOKBACK_SEC, self.cfg.max_utterance_sec / 3) / BLOCK_SEC
        run = 0
        for i in range(count - 1, max(-1, count - int(lookback) - 1), -1):
            run = 0 if self._voiced[i] else run + 1
            if run >= PAUSE_MIN_BLOCKS:
                return i + run // 2  # 쉬는 구간의 가운데
        return None

    def _quietest_block(self, center: int, radius: int) -> int | None:
        """주어진 위치 근처에서 가장 조용한 블록. 단어 사이일 가능성이 높다."""
        low = max(1, center - radius)
        high = min(len(self._blocks) - 1, center + radius)
        if low > high:
            return None
        # 소리 크기가 같으면 뒤쪽을 고른다
        return min(range(high, low - 1, -1), key=lambda i: float(np.mean(self._blocks[i] ** 2)))

    def _split_at(self, cut: int) -> Iterator[Utterance]:
        """cut 앞을 확정하고, 뒤는 다음 발화로 이어간다."""
        tail_blocks, tail_voiced = self._blocks[cut:], self._voiced[cut:]
        while cut > 1 and not self._voiced[cut - 1]:
            cut -= 1  # 앞 조각 끝의 무음은 인식에 필요 없다
        self._blocks = self._blocks[:cut]
        yield self._emit(final=True)
        self._next_id += 1
        if not any(tail_voiced):
            self._reset()
            return
        self._blocks, self._voiced = tail_blocks, tail_voiced
        self._since_partial = 0
        self._silent_blocks = 0
        for voiced in reversed(tail_voiced):
            if voiced:
                break
            self._silent_blocks += 1

    def _cut_long(self) -> Iterator[Utterance]:
        """길이 제한에 걸렸을 때 숨 고른 곳에서 자른다. 없으면 가장 조용한 곳에서 자른다."""
        cut = self._find_pause()
        if not self._long_enough(cut):
            cut = self._quietest_block(len(self._blocks) - 1, QUIET_LOOKBACK_BLOCKS)
        if self._long_enough(cut):
            yield from self._split_at(cut)
        else:
            yield from self._finish()

    def _take_pending_cut(self) -> Iterator[Utterance]:
        pending, self._pending_cut = self._pending_cut, None
        if pending is None or not self._in_speech or pending[0] != self._next_id:
            return  # 그사이 이미 끝난 문장이면 무시한다
        cut = self._quietest_block(round(pending[1] / BLOCK_SEC), SENTENCE_SNAP_BLOCKS)
        if self._long_enough(cut):
            yield from self._split_at(cut)

    def feed(self, block: np.ndarray) -> Iterator[Utterance]:
        yield from self._take_pending_cut()
        is_speech = self.speech_prob(block) >= self.cfg.vad_threshold

        if not self._in_speech:
            if is_speech:
                self._in_speech = True
                self._blocks = [*self._pre_roll, block]
                self._voiced = [False] * len(self._pre_roll) + [True]
                self._pre_roll.clear()
            else:
                self._pre_roll.append(block)
            return

        self._blocks.append(block)
        self._voiced.append(is_speech)
        self._silent_blocks = 0 if is_speech else self._silent_blocks + 1
        self._since_partial += 1

        if self._silent_blocks * BLOCK_SEC >= self.cfg.silence_end_sec:
            yield from self._finish()
        elif len(self._blocks) * BLOCK_SEC >= self.cfg.max_utterance_sec:
            yield from self._cut_long()
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
