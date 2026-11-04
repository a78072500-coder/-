"""화면과 파이프라인을 잇는다: 모델 준비, 통화 시작과 종료, 설정 적용."""

from __future__ import annotations

import sys
import threading
import traceback
from dataclasses import replace
from pathlib import Path

from PySide6.QtCore import QObject, Signal

from .config import Config, WhisperConfig
from .models import ME
from .pipeline import ConversationLog, MicChannel, Pipeline
from .reading import Reader, create_reader
from .segmenter import Segmenter, SileroSpeechProb
from .store import HistoryStore, SessionRecorder
from .translators import NoTranslator, create_translator

LOADING = "loading"      # 음성인식 모델 준비 중
READY = "ready"          # 통화를 시작할 수 있음
LISTENING = "listening"  # 듣는 중
STOPPING = "stopping"    # 마지막 문장을 마무리하는 중
FAILED = "failed"        # 모델을 불러오지 못함

MIC_FIRST_ID = 1_000_000  # 상대방 문장과 번호가 겹치지 않게


def _model_key(cfg: WhisperConfig) -> tuple[str, str, str]:
    """이 값이 바뀌면 모델을 다시 불러와야 한다."""
    return cfg.model, cfg.device, cfg.compute_type


class CallEngine(QObject):
    state_changed = Signal(str, str)  # 상태, 안내 문구
    reader_ready = Signal()

    def __init__(
        self,
        cfg: Config,
        config_path: Path,
        log_dir: Path,
        sink,
        store: HistoryStore,
        input_path: Path | None = None,
    ):
        super().__init__()
        self.cfg = cfg
        self.config_path = config_path
        self.log_dir = log_dir
        self.sink = sink
        self.store = store
        self.input_path = input_path  # 스피커 대신 wav 파일을 듣는 시험용
        self.reader: Reader | None = None
        self.state = LOADING
        self.notice = ""
        self._transcriber = None
        self._loaded_whisper: WhisperConfig | None = None
        self._start_when_ready = False
        self._pipeline: Pipeline | None = None
        self._segmenter: Segmenter | None = None
        self._mic_segmenter: Segmenter | None = None
        self._session_id: int | None = None
        self._lock = threading.Lock()

    def _set_state(self, state: str, message: str = "") -> None:
        self.state = state
        self.state_changed.emit(state, message)

    # --- 모델 준비 --------------------------------------------------------------

    def prepare(self) -> None:
        """음성인식 모델을 백그라운드에서 불러온다."""
        self._set_state(LOADING, "음성인식을 준비하고 있어요. 처음 한 번은 모델(약 1.6GB)을 내려받아 몇 분 걸립니다.")
        threading.Thread(target=self._load, daemon=True, name="load_models").start()

    def _load(self) -> None:
        from .transcriber import create_transcriber

        if self.reader is None:
            self.reader = create_reader()
            self.reader_ready.emit()
        wanted = replace(self.cfg.whisper)
        try:
            self._transcriber = create_transcriber(wanted)
            self._loaded_whisper = wanted
            self.notice = self._transcriber.notice
        except Exception as e:
            traceback.print_exc()
            self._start_when_ready = False
            self._set_state(FAILED, f"음성인식을 불러오지 못했어요: {e}")
            return
        self._set_state(READY, self.notice or "준비됐어요. 통화를 시작하세요.")
        self._refresh_whisper()  # 불러오는 동안 설정이 바뀌었을 수 있다
        if self.state == READY and self._start_when_ready:
            self._start_when_ready = False
            self.start()

    def apply_config(self, cfg: Config) -> None:
        """새 설정을 적용한다. 통화 중이면 음성인식 설정은 통화가 끝난 뒤에 적용한다."""
        self.cfg = cfg
        self.apply_segmenter()
        if self.state == FAILED:
            self.prepare()  # 설정을 고쳤으니 다시 시도한다
        else:
            self._refresh_whisper()

    def apply_segmenter(self) -> None:
        """문장 나누기 설정은 통화 중에도 다음 블록부터 바로 적용한다."""
        if self._segmenter is not None:
            self._segmenter.cfg = self.cfg.segmenter
        if self._mic_segmenter is not None:
            self._mic_segmenter.cfg = replace(self.cfg.segmenter, partial_interval_sec=0.0)

    def _refresh_whisper(self) -> None:
        wanted = self.cfg.whisper
        if self.state != READY or wanted == self._loaded_whisper:
            return
        if _model_key(wanted) != _model_key(self._loaded_whisper):
            self.prepare()
            return
        # 단어 목록이나 정확도만 바뀌었으면 모델을 다시 불러올 필요가 없다
        self._transcriber.update_settings(replace(wanted))
        self._loaded_whisper = replace(wanted)

    # --- 통화 ------------------------------------------------------------------

    def _make_source(self):
        from .audio_capture import FileSource, LoopbackSource, MacAudioSource

        if self.input_path:
            return FileSource(self.input_path)
        if sys.platform == "darwin":
            return MacAudioSource(self.cfg.audio.app)
        return LoopbackSource(self.cfg.audio.device)

    def _make_mic(self) -> MicChannel | None:
        if not self.cfg.audio.mic_enabled or self.input_path:
            return None
        from .audio_capture import create_mic_source

        source = create_mic_source(self.cfg.audio.mic_device)
        if source is None:
            return None
        transcriber = self._transcriber
        mic_cfg = replace(self.cfg.segmenter, partial_interval_sec=0.0)
        self._mic_segmenter = Segmenter(mic_cfg, SileroSpeechProb(), ME, MIC_FIRST_ID)
        return MicChannel(
            source,
            self._mic_segmenter,
            lambda audio, final: transcriber.transcribe(audio, language=None),
        )

    def _make_recorder(self):
        if not self.cfg.log.enabled:
            return None
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self._session_id = self.store.create_session()
        return SessionRecorder(ConversationLog(self.log_dir), self.store, self._session_id)

    def start(self) -> None:
        with self._lock:
            if self.state == LOADING:
                self._start_when_ready = True
                self._set_state(LOADING, "준비가 끝나면 바로 듣기 시작합니다.")
                return
            if self.state not in (READY,):
                return
            try:
                self._begin()
            except Exception as e:
                traceback.print_exc()
                self._finish_session()
                self._set_state(READY, f"시작하지 못했어요: {e}")

    def _begin(self) -> None:
        notices = [self.notice] if self.notice else []
        try:
            translator = create_translator(self.cfg.translator)
        except ValueError as e:
            notices.append(f"번역 꺼짐: {e}")
            translator = NoTranslator()
        transcriber = self._transcriber
        beam = self.cfg.whisper.beam_size

        def transcribe(audio, final: bool):
            # 임시 자막은 빠르게(beam 1), 확정본은 정확하게
            if final:
                return transcriber.transcribe(audio, beam_size=beam)
            # 슬라이더로 바꾼 값이 통화 중에도 적용되도록 매번 읽는다
            words = self.cfg.segmenter.sentence_cut
            return transcriber.transcribe(audio, beam_size=1, words=words)

        reader = self.reader
        self._mic_segmenter = None
        self._segmenter = Segmenter(self.cfg.segmenter, SileroSpeechProb())
        pipeline = Pipeline(
            self._make_source(),
            self._segmenter,
            transcribe,
            translator,
            self.sink,
            self.cfg.translator.context_lines,
            self._make_recorder(),
            reader.subtitle_reading if reader else None,
            self._make_mic(),
        )
        self._pipeline = pipeline
        pipeline.start()
        threading.Thread(
            target=self._watch, args=(pipeline,), daemon=True, name="watch_pipeline"
        ).start()
        message = " / ".join(notices) or "듣는 중이에요. 상대방이 말하면 자막이 나타납니다."
        self._set_state(LISTENING, message)

    def _watch(self, pipeline: Pipeline) -> None:
        """파이프라인이 끝나면(종료를 눌렀거나 캡처 오류) 통화를 마무리한다."""
        pipeline.wait()
        with self._lock:
            self._pipeline = None
            self._finish_session()
            if pipeline.capture_error:
                message = f"소리를 받지 못해 통화를 멈췄어요: {pipeline.capture_error}"
            else:
                message = "통화가 끝났어요. 기록 화면에서 다시 볼 수 있습니다."
            self._set_state(READY, message)
            self._refresh_whisper()

    def _finish_session(self) -> None:
        if self._session_id is not None:
            try:
                self.store.end_session(self._session_id)
            except Exception:
                traceback.print_exc()
            self._session_id = None

    def stop(self) -> None:
        pipeline = self._pipeline
        if pipeline is not None and self.state == LISTENING:
            self._set_state(STOPPING, "마지막 문장을 마무리하고 있어요.")
            pipeline.stop()

    def shutdown(self) -> None:
        """프로그램을 끝낼 때 호출한다. 번역이 늦어져도 오래 기다리지 않는다."""
        pipeline = self._pipeline
        if pipeline is not None:
            pipeline.stop()
            pipeline.wait(3.0)
        self._finish_session()
