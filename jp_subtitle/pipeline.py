"""캡처 → 발화 분리 → 음성인식 → 번역을 스레드로 연결한다."""

from __future__ import annotations

import queue
import threading
import time
import traceback
from collections import deque
from dataclasses import dataclass, replace
from datetime import datetime
from pathlib import Path
from typing import Callable, Iterator, Protocol

import numpy as np

from .backchannel import backchannel_gloss
from .models import ME, THEM, Line, Timing, Transcript
from .segmenter import Segmenter, Utterance
from .sentences import split_sentences
from .translators import Translator

ME_PREFIX = "[나] "  # 번역 문맥에 내 말을 넣을 때 붙이는 표시
LEVEL_INTERVAL_SEC = 0.2
SPLIT_FIRST_ID = 2_000_000  # 한 조각을 여러 문장으로 나눌 때 뒤 문장에 주는 번호


class Sink(Protocol):
    """파이프라인 결과를 받는 쪽 (화면 또는 콘솔)."""

    def partial(self, uid: int, ja: str) -> None: ...
    def final(self, line: Line) -> None: ...
    def translated(self, line: Line) -> None: ...
    def status(self, text: str) -> None: ...
    def level(self, speaker: str, value: float) -> None: ...


class AudioSource(Protocol):
    def blocks(self) -> Iterator[np.ndarray]: ...
    def stop(self) -> None: ...


class Recorder(Protocol):
    def record(self, line: Line, timing: Timing) -> None: ...


@dataclass
class MicChannel:
    """내 목소리 입력. 인식 언어를 자동으로 고르므로 인식 함수를 따로 받는다."""

    source: AudioSource
    segmenter: Segmenter
    transcribe: Callable[[np.ndarray, bool], Transcript]


class ConversationLog:
    """일본어 공부용으로 대화 내용을 txt 로 남긴다."""

    def __init__(self, directory: str | Path):
        path = Path(directory)
        path.mkdir(parents=True, exist_ok=True)
        self.path = path / datetime.now().strftime("%Y-%m-%d_%H%M%S.txt")
        self._lock = threading.Lock()

    def record(self, line: Line, timing: Timing) -> None:
        stamp = datetime.now().strftime("%H:%M:%S")
        mark = "(나) " if line.speaker == ME else "(?) " if line.low_confidence else ""
        with self._lock, self.path.open("a", encoding="utf-8") as f:
            f.write(f"[{stamp}] {mark}{line.ja}\n")
            if line.ko:
                f.write(f"           {line.ko}\n")


class _AsrQueue:
    """확정 발화는 모두 처리하고, 임시 조각은 가장 최근 것 하나만 남긴다.

    상대방 말을 내 말보다 먼저 처리해서 내 목소리 때문에 자막이 늦어지지 않게 한다.
    """

    def __init__(self) -> None:
        self._them: deque[Utterance] = deque()
        self._me: deque[Utterance] = deque()
        self._partial: Utterance | None = None
        self._cond = threading.Condition()
        self._closed = False

    def put(self, utt: Utterance) -> None:
        with self._cond:
            if utt.final:
                (self._me if utt.speaker == ME else self._them).append(utt)
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
            while not self._them and not self._me and self._partial is None:
                if self._closed:
                    return None
                self._cond.wait()
            if self._them:
                return self._them.popleft()
            if self._partial is not None:
                utt, self._partial = self._partial, None
                return utt
            return self._me.popleft()


class Pipeline:
    def __init__(
        self,
        source: AudioSource,
        segmenter: Segmenter,
        transcribe: Callable[[np.ndarray, bool], Transcript],
        translator: Translator,
        sink: Sink,
        context_lines: int = 4,
        log: Recorder | None = None,
        reading: Callable[[str], str] | None = None,
        mic: MicChannel | None = None,
    ):
        """transcribe(audio, final) -> 인식 결과."""
        self.source = source
        self.segmenter = segmenter
        self.transcribe = transcribe
        self.translator = translator
        self.sink = sink
        self.log = log
        self.reading = reading
        self.mic = mic
        self._history: deque[tuple[str, str]] = deque(maxlen=context_lines)
        self._asr_q = _AsrQueue()
        self._tr_q: queue.Queue[tuple[Line, Timing, float] | None] = queue.Queue()
        self._threads: list[threading.Thread] = []
        self._done = threading.Event()
        self._captures_left = 2 if mic else 1
        self._captures_lock = threading.Lock()
        self._record_failed = False
        self._next_split_id = SPLIT_FIRST_ID
        self.capture_error = ""  # 상대방 소리를 받지 못해 끝났을 때의 이유

    # --- 스레드 본문 ---------------------------------------------------------

    def _capture_loop(self, source: AudioSource, segmenter: Segmenter, speaker: str) -> None:
        last_level = 0.0
        peak = 0.0
        try:
            for block in source.blocks():
                peak = max(peak, float(np.max(np.abs(block))))
                now = time.monotonic()
                if now - last_level >= LEVEL_INTERVAL_SEC:
                    self.sink.level(speaker, peak)
                    last_level, peak = now, 0.0
                for utt in segmenter.feed(block):
                    self._asr_q.put(utt)
            for utt in segmenter.flush():
                self._asr_q.put(utt)
        except Exception as e:
            traceback.print_exc()
            label = "마이크" if speaker == ME else "오디오"
            self.sink.status(f"{label} 캡처 오류: {e}")
            if speaker == THEM:
                self.capture_error = str(e)
        finally:
            self.sink.level(speaker, 0.0)
            with self._captures_lock:
                self._captures_left -= 1
                if self._captures_left == 0:
                    self._asr_q.close()

    def _make_line(self, utt: Utterance, result: Transcript, uid: int, text: str) -> Line:
        line = Line(
            uid=uid,
            ja=text,
            speaker=utt.speaker,
            low_confidence=result.low_confidence,
            avg_logprob=result.avg_logprob,
            language=result.language,
        )
        if utt.speaker == THEM:
            gloss = backchannel_gloss(text)
            if gloss:
                # 짧은 말은 확신도가 원래 낮게 나오고, 틀려도 대화에 지장이 없다
                line.backchannel, line.ko, line.low_confidence = True, gloss, False
        if self.reading and line.language == "ja" and not line.backchannel:
            try:
                line.reading = self.reading(text)
            except Exception:
                traceback.print_exc()  # 읽는 법 때문에 자막이 멈추면 안 된다
        return line

    def _sentences(self, utt: Utterance, text: str) -> list[str]:
        """확정된 조각에 여러 문장이 들어 있으면 나눠서 따로 번역한다."""
        if (
            utt.speaker != THEM
            or not self.segmenter.cfg.sentence_cut
            or backchannel_gloss(text) is not None
        ):
            return [text]
        return split_sentences(text) or [text]

    def _emit_final(self, utt: Utterance, result: Transcript, started: float) -> None:
        for index, text in enumerate(self._sentences(utt, result.text)):
            uid = utt.utterance_id
            if index > 0:
                uid, self._next_split_id = self._next_split_id, self._next_split_id + 1
            line = self._make_line(utt, result, uid, text)
            finished = time.monotonic()
            timing = Timing(
                audio_sec=len(utt.audio) / 16000,
                silence_wait=utt.silence_wait,
                asr_wait=started - utt.emitted_at,
                asr=finished - started,
            )
            self.sink.final(line)
            self._tr_q.put((line, replace(timing), finished))
        # 대화 내용은 실행 기록 파일에 남기지 않는다 (기록 끄기 설정을 지키려고)
        print(f"[인식] {timing.audio_sec:.1f}초 → {timing.asr:.2f}초")

    def _asr_loop(self) -> None:
        try:
            while (utt := self._asr_q.get()) is not None:
                try:
                    started = time.monotonic()
                    transcribe = self.mic.transcribe if utt.speaker == ME else self.transcribe
                    result = transcribe(utt.audio, utt.final)
                    if not utt.final:
                        if result.text:
                            self.sink.partial(utt.utterance_id, result.text)
                            if result.sentence_end_sec is not None:
                                # 문장이 끝난 게 보이면 쉬기를 기다리지 않고 바로 자른다
                                self.segmenter.request_cut(
                                    utt.utterance_id, result.sentence_end_sec
                                )
                        continue
                    if not result.text:
                        self.sink.partial(utt.utterance_id, "")
                        continue
                    self._emit_final(utt, result, started)
                except Exception as e:
                    traceback.print_exc()
                    self.sink.status(f"음성인식 오류: {e}")
        finally:
            self._tr_q.put(None)

    def _translate(self, line: Line) -> None:
        if line.speaker == ME:
            self._history.append((ME_PREFIX + line.ja, ""))
            return
        if line.backchannel:
            return
        try:
            line.ko = self.translator.translate(line.ja, list(self._history))
        except Exception as e:
            traceback.print_exc()
            self.sink.status(f"번역 오류: {e}")
            line.ko = ""
        if line.ko:
            self._history.append((line.ja, line.ko))

    def _record(self, line: Line, timing: Timing) -> None:
        if not self.log:
            return
        try:
            self.log.record(line, timing)
        except Exception as e:
            traceback.print_exc()
            if not self._record_failed:  # 파일 문제로 통화 중 안내가 반복되지 않게 한 번만
                self._record_failed = True
                self.sink.status(f"대화 기록 저장 오류: {e}")

    def _translate_loop(self) -> None:
        while (item := self._tr_q.get()) is not None:
            line, timing, asr_finished = item
            started = time.monotonic()
            self._translate(line)
            timing.translate_wait = started - asr_finished
            timing.translate = time.monotonic() - started
            self.sink.translated(line)
            print(
                f"[지연] 문장끝 {timing.silence_wait:.2f} + 대기 {timing.asr_wait:.2f} + "
                f"인식 {timing.asr:.2f} = 자막 {timing.to_subtitle:.2f}초 / "
                f"번역 {timing.translate:.2f} = 한국어 {timing.to_translation:.2f}초"
            )
            self._record(line, timing)
        self._done.set()

    # --- 제어 -----------------------------------------------------------------

    def _spawn(self, target, *args) -> None:
        t = threading.Thread(target=target, args=args, name=target.__name__, daemon=True)
        t.start()
        self._threads.append(t)

    def start(self) -> None:
        self._spawn(self._capture_loop, self.source, self.segmenter, THEM)
        if self.mic:
            self._spawn(self._capture_loop, self.mic.source, self.mic.segmenter, ME)
        self._spawn(self._asr_loop)
        self._spawn(self._translate_loop)

    def stop(self) -> None:
        self.source.stop()
        if self.mic:
            self.mic.source.stop()

    def wait(self, timeout: float | None = None) -> bool:
        """입력이 끝나고 마지막 번역까지 마치면 True."""
        return self._done.wait(timeout)
