"""캡처 → 발화 분리 → 음성인식 → 번역을 스레드로 연결한다."""

from __future__ import annotations

import queue
import threading
import time
import traceback
from collections import deque
from datetime import datetime
from pathlib import Path
from typing import Callable, Iterator, Protocol

import numpy as np

from .segmenter import Segmenter, Utterance
from .translators import Translator


class Sink(Protocol):
    """파이프라인 결과를 받는 쪽 (자막 창 또는 콘솔)."""

    def partial(self, uid: int, ja: str) -> None: ...
    def final(self, uid: int, ja: str) -> None: ...
    def translated(self, uid: int, ja: str, ko: str) -> None: ...
    def status(self, text: str) -> None: ...


class AudioSource(Protocol):
    def blocks(self) -> Iterator[np.ndarray]: ...
    def stop(self) -> None: ...


class ConversationLog:
    """일본어 공부용으로 대화 내용을 txt 로 남긴다."""

    def __init__(self, directory: str | Path):
        path = Path(directory)
        path.mkdir(parents=True, exist_ok=True)
        self.path = path / datetime.now().strftime("%Y-%m-%d_%H%M%S.txt")
        self._lock = threading.Lock()

    def write(self, ja: str, ko: str) -> None:
        stamp = datetime.now().strftime("%H:%M:%S")
        with self._lock, self.path.open("a", encoding="utf-8") as f:
            f.write(f"[{stamp}] {ja}\n")
            if ko:
                f.write(f"           {ko}\n")


class _AsrQueue:
    """확정 발화는 모두 처리하고, 임시 조각은 가장 최근 것 하나만 남긴다."""

    def __init__(self) -> None:
        self._finals: deque[Utterance] = deque()
        self._partial: Utterance | None = None
        self._cond = threading.Condition()
        self._closed = False

    def put(self, utt: Utterance) -> None:
        with self._cond:
            if utt.final:
                self._finals.append(utt)
                if self._partial and self._partial.utterance_id == utt.utterance_id:
                    self._partial = None
            else:
                self._partial = utt
            self._cond.notify()

    def close(self) -> None:
        with self._cond:
            self._closed = True
            self._cond.notify_all()

    def get(self) -> Utterance | None:
        """다음 처리할 발화. 닫히고 비었으면 None."""
        with self._cond:
            while not self._finals and self._partial is None:
                if self._closed:
                    return None
                self._cond.wait()
            if self._finals:
                return self._finals.popleft()
            utt, self._partial = self._partial, None
            return utt


class Pipeline:
    def __init__(
        self,
        source: AudioSource,
        segmenter: Segmenter,
        transcribe: Callable[[np.ndarray, bool], str],
        translator: Translator,
        sink: Sink,
        context_lines: int = 4,
        log: ConversationLog | None = None,
    ):
        """transcribe(audio, final) -> 일본어 텍스트."""
        self.source = source
        self.segmenter = segmenter
        self.transcribe = transcribe
        self.translator = translator
        self.sink = sink
        self.log = log
        self._history: deque[tuple[str, str]] = deque(maxlen=context_lines)
        self._asr_q = _AsrQueue()
        self._tr_q: queue.Queue[tuple[int, str] | None] = queue.Queue()
        self._threads: list[threading.Thread] = []
        self._done = threading.Event()

    # --- 스레드 본문 ---------------------------------------------------------

    def _capture_loop(self) -> None:
        try:
            for block in self.source.blocks():
                for utt in self.segmenter.feed(block):
                    self._asr_q.put(utt)
            for utt in self.segmenter.flush():
                self._asr_q.put(utt)
        except Exception as e:
            traceback.print_exc()
            self.sink.status(f"오디오 캡처 오류: {e}")
        finally:
            self._asr_q.close()

    def _asr_loop(self) -> None:
        try:
            while (utt := self._asr_q.get()) is not None:
                try:
                    started = time.monotonic()
                    ja = self.transcribe(utt.audio, utt.final)
                    if not utt.final:
                        if ja:
                            self.sink.partial(utt.utterance_id, ja)
                        continue
                    if ja:
                        audio_sec = len(utt.audio) / 16000
                        print(f"[인식] {audio_sec:.1f}초 → {time.monotonic() - started:.2f}초: {ja}")
                        self.sink.final(utt.utterance_id, ja)
                        self._tr_q.put((utt.utterance_id, ja))
                    else:
                        self.sink.partial(utt.utterance_id, "")
                except Exception as e:
                    traceback.print_exc()
                    self.sink.status(f"음성인식 오류: {e}")
        finally:
            self._tr_q.put(None)

    def _translate_loop(self) -> None:
        while (item := self._tr_q.get()) is not None:
            uid, ja = item
            try:
                ko = self.translator.translate(ja, list(self._history))
            except Exception as e:
                traceback.print_exc()
                self.sink.status(f"번역 오류: {e}")
                ko = ""
            if ko:
                self._history.append((ja, ko))
            self.sink.translated(uid, ja, ko)
            if self.log:
                self.log.write(ja, ko)
        self._done.set()

    # --- 제어 -----------------------------------------------------------------

    def start(self) -> None:
        for target in (self._capture_loop, self._asr_loop, self._translate_loop):
            t = threading.Thread(target=target, name=target.__name__, daemon=True)
            t.start()
            self._threads.append(t)

    def stop(self) -> None:
        self.source.stop()

    def wait(self, timeout: float | None = None) -> bool:
        """입력이 끝나고 마지막 번역까지 마치면 True."""
        return self._done.wait(timeout)
