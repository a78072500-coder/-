"""통화 화면: 실시간 자막과 답변 도우미."""

from __future__ import annotations

from collections import deque

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QCheckBox,
    QHBoxLayout,
    QPlainTextEdit,
    QProgressBar,
    QSlider,
    QSplitter,
    QVBoxLayout,
    QWidget,
)

from ..assistant import Option
from ..config import SENTENCE_PRESETS, find_preset, nearest_preset_index, save_config
from ..engine import FAILED, LISTENING, LOADING, READY, STOPPING
from ..hotkeys import hotkey_name
from ..models import ME, THEM, Line
from ..phrases import PHRASES
from .common import (
    AppContext,
    TranscriptView,
    clear_layout,
    make_button,
    make_label,
    make_scroll,
    option_card,
    run_async,
    set_role,
)

RECENT_LINES = 12


class _ReplyInput(QPlainTextEdit):
    """Enter 로 보내고 Shift+Enter 로 줄을 바꾼다."""

    def __init__(self, on_submit):
        super().__init__()
        self._on_submit = on_submit

    def keyPressEvent(self, event) -> None:
        is_enter = event.key() in (Qt.Key.Key_Return, Qt.Key.Key_Enter)
        if is_enter and not event.modifiers() & Qt.KeyboardModifier.ShiftModifier:
            self._on_submit()
            return
        super().keyPressEvent(event)


class LivePage(QWidget):
    def __init__(self, ctx: AppContext, open_settings, on_cards_changed, on_segmenter_changed):
        super().__init__()
        self.setObjectName("page")
        self.ctx = ctx
        self.open_settings = open_settings
        self.on_cards_changed = on_cards_changed
        self.on_segmenter_changed = on_segmenter_changed
        self._recent: deque[Line] = deque(maxlen=RECENT_LINES)
        self._phrase_index = -1
        self._last_state = ctx.engine.state

        root = QVBoxLayout(self)
        root.setContentsMargins(24, 20, 24, 20)
        root.setSpacing(14)
        root.addLayout(self._build_header())

        split = QSplitter(Qt.Orientation.Horizontal)
        split.setChildrenCollapsible(False)
        split.setHandleWidth(1)
        self.view = TranscriptView(
            "통화 시작을 누르면 상대방의 말이 여기에 나타납니다.", self._save_line
        )
        self.view.show_reading = ctx.cfg.overlay.show_reading
        split.addWidget(self._wrap_left(self.view))
        split.addWidget(self._build_assist())
        split.setStretchFactor(0, 3)
        split.setStretchFactor(1, 2)
        split.setSizes([640, 400])
        root.addWidget(split, 1)

        ctx.bridge.partial_sig.connect(self.view.on_partial)
        ctx.bridge.final_sig.connect(self._on_final)
        ctx.bridge.translated_sig.connect(self._on_translated)
        ctx.bridge.status_sig.connect(self._on_status)
        ctx.bridge.level_sig.connect(self._on_level)
        ctx.engine.state_changed.connect(self._on_state)
        if ctx.overlay is not None:
            ctx.overlay.hidden_by_user.connect(lambda: self.overlay_check.setChecked(False))
        self._on_state(ctx.engine.state, "")
        self.refresh_assist_state()

    # --- 윗줄 ------------------------------------------------------------------

    def _build_header(self) -> QHBoxLayout:
        header = QHBoxLayout()
        header.setSpacing(14)
        self.start_button = make_button("통화 시작", "primary", self.toggle_call)
        self.start_button.setMinimumWidth(120)
        header.addWidget(self.start_button)

        status = QVBoxLayout()
        status.setSpacing(4)
        self.status_label = make_label("", "muted")
        status.addWidget(self.status_label)
        meters = QHBoxLayout()
        meters.setSpacing(8)
        self.them_meter = self._meter()
        self.me_meter = self._meter()
        meters.addWidget(make_label("상대", "faint", wrap=False))
        meters.addWidget(self.them_meter)
        self.me_meter_label = make_label("나", "faint", wrap=False)
        meters.addWidget(self.me_meter_label)
        meters.addWidget(self.me_meter)
        meters.addStretch()
        status.addLayout(meters)
        header.addLayout(status, 1)

        header.addLayout(self._build_length_control())
        self.reading_check = QCheckBox("읽는 법")
        self.reading_check.setChecked(self.ctx.cfg.overlay.show_reading)
        self.reading_check.toggled.connect(self._set_show_reading)
        header.addWidget(self.reading_check)
        self.overlay_check = QCheckBox("자막 창")
        self.overlay_check.setChecked(self.ctx.cfg.overlay.enabled)
        self.overlay_check.toggled.connect(self._sync_overlay)
        header.addWidget(self.overlay_check)
        return header

    def _build_length_control(self) -> QVBoxLayout:
        """문장 길이 슬라이더. 통화 중에 움직여도 바로 적용된다."""
        control = QVBoxLayout()
        control.setSpacing(2)
        caption = QHBoxLayout()
        caption.addWidget(make_label("문장 길이", "faint", wrap=False))
        caption.addStretch(1)
        self.length_label = make_label("", "faint", wrap=False)
        caption.addWidget(self.length_label)
        control.addLayout(caption)
        self.length_slider = QSlider(Qt.Orientation.Horizontal)
        self.length_slider.setRange(0, len(SENTENCE_PRESETS) - 1)
        self.length_slider.setPageStep(1)
        self.length_slider.setFixedWidth(150)
        self.length_slider.setToolTip(
            "왼쪽으로 갈수록 문장을 빨리 끊어 번역이 빨리 나옵니다.\n"
            "문장 중간에서 자주 끊기면 오른쪽으로 옮기세요. 통화 중에도 바로 적용됩니다."
        )
        self.sync_length_control()
        self.length_slider.valueChanged.connect(self._on_length_changed)
        control.addWidget(self.length_slider)
        return control

    def sync_length_control(self) -> None:
        """설정 화면에서 바꾼 값을 슬라이더에 반영한다."""
        cfg = self.ctx.cfg.segmenter
        self.length_slider.blockSignals(True)
        self.length_slider.setValue(nearest_preset_index(cfg))
        self.length_slider.blockSignals(False)
        preset = find_preset(cfg)
        self.length_label.setText(preset.label if preset else "직접 설정")

    def _on_length_changed(self, index: int) -> None:
        preset = SENTENCE_PRESETS[index]
        cfg = self.ctx.cfg.segmenter
        for name, value in preset.values().items():
            setattr(cfg, name, value)
        self.ctx.engine.apply_segmenter()
        self.length_label.setText(preset.label)
        try:
            save_config(self.ctx.config_path, {"segmenter": preset.values()})
        except Exception as e:
            self._on_status(f"문장 길이를 저장하지 못했어요 (이번 실행에는 적용됩니다) 오류: {e}")
        self.on_segmenter_changed()

    @staticmethod
    def _meter() -> QProgressBar:
        meter = QProgressBar()
        meter.setRange(0, 100)
        meter.setTextVisible(False)
        meter.setFixedSize(110, 6)
        return meter

    def _wrap_left(self, view: QWidget) -> QWidget:
        wrapper = QWidget()
        layout = QVBoxLayout(wrapper)
        layout.setContentsMargins(0, 0, 14, 0)
        layout.addWidget(view)
        return wrapper

    # --- 오른쪽 답변 도우미 ------------------------------------------------------

    def _build_assist(self) -> QWidget:
        area, layout = make_scroll()
        layout.setContentsMargins(16, 0, 4, 0)
        layout.setSpacing(10)

        layout.addWidget(make_label("하고 싶은 말", "section"))
        self.reply_input = _ReplyInput(self.submit_reply)
        self.reply_input.setPlaceholderText(f"한국어로 적고 Enter ({hotkey_name('R')} 로 바로 입력)")
        self.reply_input.setFixedHeight(68)
        layout.addWidget(self.reply_input)
        buttons = QHBoxLayout()
        self.reply_button = make_button("일본어로 바꾸기", "primary", self.submit_reply)
        self.suggest_button = make_button("추천 답변", "", self.request_suggestions)
        self.suggest_button.setToolTip(hotkey_name("S"))
        buttons.addWidget(self.reply_button, 1)
        buttons.addWidget(self.suggest_button)
        layout.addLayout(buttons)

        self.assist_notice = QWidget()
        notice = QHBoxLayout(self.assist_notice)
        notice.setContentsMargins(0, 0, 0, 0)
        notice.addWidget(make_label("API 키를 넣으면 답변 만들기를 쓸 수 있어요.", "faint"), 1)
        notice.addWidget(make_button("설정 열기", "link", self.open_settings))
        layout.addWidget(self.assist_notice)

        self.results = QVBoxLayout()
        self.results.setSpacing(8)
        layout.addLayout(self.results)

        layout.addSpacing(8)
        layout.addWidget(make_label(f"바로 쓰는 말  ({hotkey_name('Q')} 로 자막 창에 하나씩)", "section"))
        for ja, ko in PHRASES:
            button = make_button(f"{ja}\n{ko}", "phrase")
            button.clicked.connect(lambda _=False, j=ja, k=ko: self._show_phrase(j, k))
            layout.addWidget(button)
        layout.addStretch(1)
        return area

    def refresh_assist_state(self) -> None:
        assistant = self.ctx.assistant
        self.assist_notice.setVisible(not assistant.can_reply)
        self.reply_button.setEnabled(assistant.can_reply)
        self.suggest_button.setEnabled(assistant.has_ai)
        self.suggest_button.setToolTip(
            hotkey_name("S") if assistant.has_ai else "추천 답변은 Claude 또는 Gemini 키가 필요해요."
        )

    def _show_results(self, options: list[Option]) -> None:
        clear_layout(self.results)
        for option in options:
            self.results.addWidget(option_card(option, self._save_option, self._to_overlay))

    def _show_message(self, text: str, role: str = "muted") -> None:
        clear_layout(self.results)
        self.results.addWidget(make_label(text, role))

    def _annotate(self, ja: str, ko: str) -> Option:
        option = Option(ja, ko=ko)
        reader = self.ctx.engine.reader
        if reader:
            note = reader.annotate(ja)
            option.reading, option.hangul = note.hiragana, note.hangul
        return option

    def _show_phrase(self, ja: str, ko: str) -> None:
        self._show_results([self._annotate(ja, ko)])

    def next_phrase(self) -> None:
        """단축키: 되묻기 문구를 하나씩 돌려 가며 자막 창에 띄운다."""
        self._phrase_index = (self._phrase_index + 1) % len(PHRASES)
        ja, ko = PHRASES[self._phrase_index]
        option = self._annotate(ja, ko)
        self._show_results([option])
        self._to_overlay(option)

    def focus_reply(self) -> None:
        self.reply_input.setFocus()
        self.reply_input.selectAll()

    def submit_reply(self) -> None:
        text = self.reply_input.toPlainText().strip()
        if not text or not self.ctx.assistant.can_reply:
            return
        self.reply_button.setEnabled(False)
        self._show_message("일본어로 바꾸는 중…")
        recent = list(self._recent)

        def done(options) -> None:
            self.refresh_assist_state()
            self._show_results(options)

        def failed(message: str) -> None:
            self.refresh_assist_state()
            self._show_message(message, "error")

        run_async(lambda: self.ctx.assistant.reply(text, recent), done, failed)

    def request_suggestions(self, to_overlay: bool = False) -> None:
        if not self.ctx.assistant.has_ai:
            self._show_message("추천 답변은 Claude 또는 Gemini 키가 필요해요.", "error")
            return
        self.suggest_button.setEnabled(False)
        self._show_message("답변을 고르는 중…")
        recent = list(self._recent)

        def done(options) -> None:
            self.refresh_assist_state()
            self._show_results(options)
            if to_overlay and options:
                self._to_overlay(options[0])

        def failed(message: str) -> None:
            self.refresh_assist_state()
            self._show_message(message, "error")

        run_async(lambda: self.ctx.assistant.suggest(recent), done, failed)

    def _to_overlay(self, option: Option) -> None:
        overlay = self.ctx.overlay
        if overlay is not None:
            overlay.show_hint(option.ja, option.reading or option.hangul, option.ko)
            if self.overlay_check.isChecked() and not overlay.isVisible():
                overlay.show()

    # --- 저장 ------------------------------------------------------------------

    def _save_line(self, line: Line) -> bool:
        added = self.ctx.store.add_card(line.ja, line.ko)
        self.on_cards_changed()
        return added

    def _save_option(self, option: Option) -> bool:
        added = self.ctx.store.add_card(option.ja, option.ko)
        self.on_cards_changed()
        return added

    # --- 파이프라인과 엔진 -------------------------------------------------------

    def toggle_call(self) -> None:
        if self.ctx.engine.state == LISTENING:
            self.ctx.engine.stop()
        else:
            self.ctx.engine.start()

    def _on_final(self, line: Line) -> None:
        self.view.on_final(line)

    def _on_translated(self, line: Line) -> None:
        self.view.on_translated(line)
        if not line.backchannel:
            self._recent.append(line)

    def _on_status(self, text: str) -> None:
        self.status_label.setText(text)
        set_role(self.status_label, "error" if "오류" in text else "muted")

    def _on_level(self, speaker: str, value: float) -> None:
        meter = self.me_meter if speaker == ME else self.them_meter
        meter.setValue(min(100, int(value * 140)))

    def _on_state(self, _signalled: str, message: str) -> None:
        # 알림은 다른 스레드에서 늦게 도착할 수 있으니 지금 상태를 다시 읽는다
        state = self.ctx.engine.state
        if message:
            self.status_label.setText(message)
            failed = state == FAILED or message.startswith(("소리를 받지", "시작하지 못했어요"))
            set_role(self.status_label, "error" if failed else "muted")
        button = self.start_button
        if state == LISTENING:
            button.setText("통화 끝내기")
            button.setEnabled(True)
            set_role(button, "stop")
            if self._last_state != LISTENING:
                self.view.clear()
                self._recent.clear()
        elif state == STOPPING:
            button.setText("마무리 중…")
            button.setEnabled(False)
        elif state == LOADING:
            button.setText("준비 중… 시작 예약")
            button.setEnabled(True)
            set_role(button, "primary")
        elif state == FAILED:
            button.setText("통화 시작")
            button.setEnabled(False)
            set_role(button, "primary")
        else:
            button.setText("통화 시작")
            button.setEnabled(state == READY)
            set_role(button, "primary")
        mic_on = self.ctx.cfg.audio.mic_enabled
        self.me_meter.setVisible(mic_on)
        self.me_meter_label.setVisible(mic_on)
        self._last_state = state
        self._sync_overlay()

    def _set_show_reading(self, value: bool) -> None:
        self.view.set_show_reading(value)
        if self.ctx.overlay is not None:
            self.ctx.overlay._set_show_reading(value)

    def _sync_overlay(self, *_args) -> None:
        """자막 창은 통화 중에만 띄운다."""
        overlay = self.ctx.overlay
        if overlay is None:
            return
        listening = self.ctx.engine.state in (LISTENING, STOPPING)
        overlay.setVisible(self.overlay_check.isChecked() and listening)

    def apply_config(self) -> None:
        self.sync_length_control()
        self.reading_check.setChecked(self.ctx.cfg.overlay.show_reading)
        self.overlay_check.setChecked(self.ctx.cfg.overlay.enabled)
        self.refresh_assist_state()
        self._on_state(self.ctx.engine.state, "")
