"""설정 화면. 메모장 없이 config.toml 을 고친다."""

from __future__ import annotations

import sys
from typing import Callable

from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDoubleSpinBox,
    QFormLayout,
    QHBoxLayout,
    QLineEdit,
    QPlainTextEdit,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)

from ..assistant import Assistant
from ..config import SENTENCE_PRESETS, find_preset, load_config, save_config
from ..hotkeys import hotkey_name
from ..overlay import open_path
from .common import AppContext, make_button, make_card, make_label, make_scroll, run_async

WHISPER_MODELS = ["large-v3-turbo", "large-v3", "medium", "small", "base", "tiny"]
CLAUDE_MODELS = ["claude-opus-5-5", "claude-sonnet-5-5", "claude-haiku-5-5"]
GEMINI_MODELS = ["gemini-3.5-flash-lite", "gemini-3.8-flash", "gemini-3.6-flash", "gemini-4-argon"]
ENGINES = [
    ("DeepL (무료 키)", "deepl"),
    ("Claude (말투가 자연스러움, 유료)", "claude"),
    ("Gemini (말투가 자연스러움)", "gemini"),
    ("번역 안 함", "none"),
]
AI_ENGINES = [("자동 (키가 있는 쪽)", "auto"), ("Claude", "claude"), ("Gemini", "gemini")]
ACCURACY = [("빠르게", 1), ("보통", 3), ("정확하게", 5)]
MAC_ALL_SOUND = "모든 소리"


def _select(combo: QComboBox, value) -> None:
    index = combo.findData(value)
    if index >= 0:
        combo.setCurrentIndex(index)
    elif combo.isEditable():
        combo.setEditText(str(value))


def _device_names(kind: str) -> list[str]:
    """고를 수 있는 오디오 장치 이름들. 못 읽으면 빈 목록. 맥은 마이크만 장치로 고른다."""
    if sys.platform != "win32" and not (sys.platform == "darwin" and kind == "input"):
        return []
    try:
        from ..audio_capture import list_input_devices, list_loopback_devices

        devices = list_loopback_devices() if kind == "output" else list_input_devices()
        return sorted({device["name"] for device in devices})
    except Exception as e:
        print(f"[설정] 오디오 장치 목록을 읽지 못했습니다: {e}")
        return []


class SettingsPage(QWidget):
    def __init__(self, ctx: AppContext, on_applied: Callable[[], None]):
        super().__init__()
        self.setObjectName("page")
        self.ctx = ctx
        self.on_applied = on_applied

        area, layout = make_scroll()
        layout.setContentsMargins(24, 20, 24, 20)
        layout.setSpacing(14)
        root = QVBoxLayout(self)
        root.setContentsMargins(0, 0, 0, 0)
        root.addWidget(area, 1)

        layout.addWidget(make_label("설정", "title"))
        self._build_translate(layout)
        self._build_speech(layout)
        self._build_sound(layout)
        self._build_sentence(layout)
        self._build_overlay(layout)
        self._build_record(layout)

        layout.addWidget(make_label("단축키", "section"))
        keys, keys_layout = make_card()
        for text in (
            f"{hotkey_name('R')}   메인 창을 띄우고 하고 싶은 말 입력",
            f"{hotkey_name('Q')}   되묻기 문구를 자막 창에 하나씩 표시",
            f"{hotkey_name('S')}   추천 답변 만들기",
        ):
            keys_layout.addWidget(make_label(text, ""))
        layout.addWidget(keys)
        layout.addStretch(1)

        footer = QHBoxLayout()
        footer.setContentsMargins(24, 10, 24, 14)
        self.message = make_label("", "muted")
        footer.addWidget(self.message, 1)
        footer.addWidget(make_button("설정 파일 열기", "", lambda: open_path(ctx.config_path)))
        footer.addWidget(make_button("저장", "primary", self.save))
        root.addLayout(footer)
        self.load_values()

    # --- 만들기 ------------------------------------------------------------------

    @staticmethod
    def _group(layout: QVBoxLayout, title: str) -> QFormLayout:
        layout.addWidget(make_label(title, "section"))
        card, inner = make_card()
        form = QFormLayout()
        form.setSpacing(10)
        form.setFieldGrowthPolicy(QFormLayout.FieldGrowthPolicy.AllNonFixedFieldsGrow)
        inner.addLayout(form)
        layout.addWidget(card)
        return form

    @staticmethod
    def _secret() -> QLineEdit:
        edit = QLineEdit()
        edit.setEchoMode(QLineEdit.EchoMode.Password)
        return edit

    def _build_translate(self, layout: QVBoxLayout) -> None:
        form = self._group(layout, "번역과 답변 도우미")
        self.engine = QComboBox()
        for label, value in ENGINES:
            self.engine.addItem(label, value)
        form.addRow("자막 번역", self.engine)
        self.ai_engine = QComboBox()
        for label, value in AI_ENGINES:
            self.ai_engine.addItem(label, value)
        form.addRow("답변·복습 AI", self.ai_engine)
        self.deepl_key = self._secret()
        self.deepl_key.setPlaceholderText("끝이 :fx 인 DeepL 무료 키")
        form.addRow("DeepL 키", self.deepl_key)
        self.claude_key = self._secret()
        self.claude_key.setPlaceholderText("Claude 나 Gemini 중 하나면 추천 답변과 AI 복습을 쓸 수 있습니다")
        form.addRow("Claude 키", self.claude_key)
        self.claude_model = QComboBox()
        self.claude_model.setEditable(True)
        self.claude_model.addItems(CLAUDE_MODELS)
        form.addRow("Claude 모델", self.claude_model)
        self.gemini_key = self._secret()
        self.gemini_key.setPlaceholderText("Google AI Studio 에서 받은 키")
        form.addRow("Gemini 키", self.gemini_key)
        self.gemini_model = QComboBox()
        self.gemini_model.setEditable(True)
        self.gemini_model.addItems(GEMINI_MODELS)
        form.addRow("Gemini 모델", self.gemini_model)
        form.addRow(
            "",
            make_label(
                "API를 쓰는 기능은 상대방 말이나 입력한 문장을 해당 서비스로 보냅니다. "
                "키는 이 PC의 config.toml 에 저장됩니다.",
                "faint",
            ),
        )
        form.addRow(
            "",
            make_label(
                "Gemini 무료 키는 보낸 내용을 Google이 제품 개선에 쓰고 사람이 검토할 수 있어요. "
                "통화 상대의 개인정보가 오갈 수 있다면 결제를 연결한 유료 키를 쓰세요.",
                "warn",
            ),
        )

    def _build_speech(self, layout: QVBoxLayout) -> None:
        form = self._group(layout, "음성인식")
        self.whisper_model = QComboBox()
        self.whisper_model.setEditable(True)
        self.whisper_model.addItems(WHISPER_MODELS)
        form.addRow("모델", self.whisper_model)
        self.device = QComboBox()
        self.device.addItem("Apple GPU" if sys.platform == "darwin" else "그래픽카드", "cuda")
        self.device.addItem("CPU (느림)", "cpu")
        form.addRow("실행 장치", self.device)
        self.accuracy = QComboBox()
        for label, value in ACCURACY:
            self.accuracy.addItem(label, value)
        form.addRow("정확도", self.accuracy)
        self.vocabulary = QPlainTextEdit()
        self.vocabulary.setFixedHeight(76)
        self.vocabulary.setPlaceholderText("상대방 이름이나 취미 단어를 한 줄에 하나씩\n田中\nヨルシカ")
        form.addRow("자주 나오는 단어", self.vocabulary)

    def _build_sound(self, layout: QVBoxLayout) -> None:
        form = self._group(layout, "소리")
        is_mac = sys.platform == "darwin"
        self._capture_key = "app" if is_mac else "device"
        self.output_device = QComboBox()
        if is_mac:
            # 맥은 장치가 아니라 앱 단위로 소리를 고른다
            self.output_device.setEditable(True)
            self.output_device.addItem(MAC_ALL_SOUND, "")
            form.addRow("상대방 소리", self.output_device)
            self.load_apps_button = make_button("실행 중인 앱 불러오기", "link", self._load_mac_apps)
            hint = QWidget()
            hint_layout = QHBoxLayout(hint)
            hint_layout.setContentsMargins(0, 0, 0, 0)
            hint_layout.addWidget(
                make_label("통화 앱 이름 일부를 적으면 그 앱 소리만 듣습니다. 예: maum", "faint"), 1
            )
            hint_layout.addWidget(self.load_apps_button)
            form.addRow("", hint)
        else:
            self.output_device.addItem("Windows 기본 출력 장치", "")
            for name in _device_names("output"):
                self.output_device.addItem(name, name)
            form.addRow("상대방 소리", self.output_device)
        self.mic_enabled = QCheckBox("내 목소리도 인식해서 기록과 번역 문맥에 넣기")
        form.addRow("", self.mic_enabled)
        self.mic_device = QComboBox()
        self.mic_device.addItem("기본 마이크", "")
        for name in _device_names("input"):
            self.mic_device.addItem(name, name)
        form.addRow("마이크", self.mic_device)
        form.addRow(
            "",
            make_label("이어폰을 꼭 쓰세요. 스피커로 들으면 상대방 말이 내 말로 다시 잡힙니다.", "faint"),
        )
        if sys.platform not in ("win32", "darwin"):
            self.mic_enabled.setEnabled(False)
            self.mic_device.setEnabled(False)

    def _load_mac_apps(self) -> None:
        """권한을 묻는 창이 뜰 수 있어 프로그램을 켤 때가 아니라 눌렀을 때만 읽는다."""
        from ..audio_capture import list_mac_apps

        self.load_apps_button.setEnabled(False)

        def done(apps) -> None:
            self.load_apps_button.setEnabled(True)
            current = self.output_device.currentText()
            self.output_device.clear()
            self.output_device.addItem(MAC_ALL_SOUND, "")
            for name in sorted(set(apps)):
                self.output_device.addItem(name, name)
            self.output_device.setEditText(current)

        def failed(message: str) -> None:
            self.load_apps_button.setEnabled(True)
            self.message.setText(message)

        run_async(list_mac_apps, done, failed)

    def _capture_value(self) -> str:
        if self._capture_key == "app":
            text = self.output_device.currentText().strip()
            return "" if text == MAC_ALL_SOUND else text
        return self.output_device.currentData()

    def _build_sentence(self, layout: QVBoxLayout) -> None:
        form = self._group(layout, "문장 나누기")
        self.sentence_preset = QComboBox()
        for preset in SENTENCE_PRESETS:
            self.sentence_preset.addItem(preset.label, preset.key)
        self.sentence_preset.addItem("직접 설정", "custom")
        form.addRow("문장 길이", self.sentence_preset)
        form.addRow(
            "",
            make_label("[통화] 화면의 슬라이더와 같습니다. 통화 중에 바꿔도 바로 적용됩니다.", "faint"),
        )
        self.silence_end = QDoubleSpinBox()
        self.silence_end.setRange(0.2, 3.0)
        self.silence_end.setSingleStep(0.05)
        self.silence_end.setSuffix(" 초")
        form.addRow("문장 끝 대기", self.silence_end)
        form.addRow("", make_label("줄이면 자막이 빨라지고, 문장이 자주 끊기면 늘리세요.", "faint"))
        self.max_length = QDoubleSpinBox()
        self.max_length.setRange(2.0, 30.0)
        self.max_length.setSingleStep(1.0)
        self.max_length.setSuffix(" 초")
        form.addRow("최대 길이", self.max_length)
        form.addRow("", make_label("쉬지 않고 말해도 이보다 길어지면 나눕니다.", "faint"))
        self.sentence_cut = QCheckBox("문장 끝이 보이면 쉬기를 기다리지 않고 바로 나누기")
        form.addRow("", self.sentence_cut)
        self.sentence_preset.activated.connect(self._on_preset_chosen)
        self.silence_end.valueChanged.connect(lambda _: self._sync_preset_choice())
        self.max_length.valueChanged.connect(lambda _: self._sync_preset_choice())
        self.sentence_cut.toggled.connect(lambda _: self._sync_preset_choice())
        self.vad = QDoubleSpinBox()
        self.vad.setRange(0.1, 0.95)
        self.vad.setSingleStep(0.05)
        form.addRow("말소리 판단 기준", self.vad)
        form.addRow("", make_label("잡음이 자막으로 뜨면 올리고, 말이 안 잡히면 내리세요.", "faint"))

    def _build_overlay(self, layout: QVBoxLayout) -> None:
        form = self._group(layout, "자막 창")
        self.overlay_enabled = QCheckBox("통화 중에 화면 위 자막 창 띄우기")
        form.addRow("", self.overlay_enabled)
        self.show_japanese = QCheckBox("일본어 원문 표시")
        form.addRow("", self.show_japanese)
        self.show_reading = QCheckBox("읽는 법 표시")
        form.addRow("", self.show_reading)
        self.font_size = QSpinBox()
        self.font_size.setRange(10, 60)
        form.addRow("글자 크기", self.font_size)
        self.opacity = QDoubleSpinBox()
        self.opacity.setRange(0.2, 1.0)
        self.opacity.setSingleStep(0.05)
        form.addRow("배경 불투명도", self.opacity)

    def _build_record(self, layout: QVBoxLayout) -> None:
        form = self._group(layout, "기록")
        self.log_enabled = QCheckBox("통화 내용을 이 PC에 저장 (기록·진단 화면에 필요)")
        form.addRow("", self.log_enabled)
        row = QWidget()
        row_layout = QHBoxLayout(row)
        row_layout.setContentsMargins(0, 0, 0, 0)
        row_layout.addWidget(make_label(str(self.ctx.log_dir), "faint"), 1)
        row_layout.addWidget(make_button("폴더 열기", "link", lambda: open_path(self.ctx.log_dir)))
        form.addRow("저장 위치", row)

    # --- 값 읽고 쓰기 -------------------------------------------------------------

    def load_values(self) -> None:
        cfg = self.ctx.cfg
        _select(self.engine, cfg.translator.engine)
        _select(self.ai_engine, cfg.translator.ai_engine)
        self.deepl_key.setText(cfg.translator.deepl_api_key)
        self.claude_key.setText(cfg.translator.anthropic_api_key)
        self.claude_model.setEditText(cfg.translator.claude_model)
        self.gemini_key.setText(cfg.translator.gemini_api_key)
        self.gemini_model.setEditText(cfg.translator.gemini_model)
        self.whisper_model.setEditText(cfg.whisper.model)
        _select(self.device, cfg.whisper.device)
        beams = [value for _, value in ACCURACY]
        nearest = min(beams, key=lambda beam: abs(beam - cfg.whisper.beam_size))
        _select(self.accuracy, nearest)
        self.vocabulary.setPlainText("\n".join(cfg.whisper.vocabulary))
        if self._capture_key == "app":
            self.output_device.setEditText(cfg.audio.app or MAC_ALL_SOUND)
        else:
            self._select_device(self.output_device, cfg.audio.device)
        self.mic_enabled.setChecked(cfg.audio.mic_enabled)
        self._select_device(self.mic_device, cfg.audio.mic_device)
        self.load_sentence_values()
        self.vad.setValue(cfg.segmenter.vad_threshold)
        self.overlay_enabled.setChecked(cfg.overlay.enabled)
        self.show_japanese.setChecked(cfg.overlay.show_japanese)
        self.show_reading.setChecked(cfg.overlay.show_reading)
        self.font_size.setValue(cfg.overlay.font_size)
        self.opacity.setValue(cfg.overlay.opacity)
        self.log_enabled.setChecked(cfg.log.enabled)

    def load_sentence_values(self) -> None:
        """문장 나누기 값을 화면에 불러온다. [통화] 화면의 슬라이더를 움직였을 때도 부른다."""
        cfg = self.ctx.cfg.segmenter
        self.silence_end.setValue(cfg.silence_end_sec)
        self.max_length.setValue(cfg.max_utterance_sec)
        self.sentence_cut.setChecked(cfg.sentence_cut)
        self._sync_preset_choice()

    def _chosen_preset(self):
        key = self.sentence_preset.currentData()
        return next((preset for preset in SENTENCE_PRESETS if preset.key == key), None)

    def _on_preset_chosen(self, _index: int) -> None:
        preset = self._chosen_preset()
        if preset is None:
            return
        self.silence_end.setValue(preset.silence_end_sec)
        self.max_length.setValue(preset.max_utterance_sec)
        self.sentence_cut.setChecked(preset.sentence_cut)

    def _sync_preset_choice(self) -> None:
        """입력한 값이 어느 단계와 같은지 보여 준다. 임시 자막 간격은 단계 값을 그대로 쓴다."""
        match = next(
            (
                preset
                for preset in SENTENCE_PRESETS
                if preset.silence_end_sec == round(self.silence_end.value(), 2)
                and preset.max_utterance_sec == round(self.max_length.value(), 2)
                and preset.sentence_cut == self.sentence_cut.isChecked()
            ),
            None,
        )
        _select(self.sentence_preset, match.key if match else "custom")

    def _segmenter_values(self) -> dict:
        values = {
            "silence_end_sec": round(self.silence_end.value(), 2),
            "max_utterance_sec": round(self.max_length.value(), 2),
            "sentence_cut": self.sentence_cut.isChecked(),
            "vad_threshold": round(self.vad.value(), 2),
        }
        preset = self._chosen_preset()
        if preset is not None:
            values["partial_interval_sec"] = preset.partial_interval_sec
        return values

    @staticmethod
    def _select_device(combo: QComboBox, value: str) -> None:
        """설정값은 이름의 일부일 수 있으니 포함 관계로 찾는다."""
        if not value:
            combo.setCurrentIndex(0)
            return
        for i in range(combo.count()):
            if value.lower() in combo.itemData(i).lower():
                combo.setCurrentIndex(i)
                return
        combo.addItem(value, value)
        combo.setCurrentIndex(combo.count() - 1)

    def collect(self) -> dict:
        device = self.device.currentData()
        vocabulary = [
            word.strip() for word in self.vocabulary.toPlainText().splitlines() if word.strip()
        ]
        return {
            "audio": {
                self._capture_key: self._capture_value(),
                "mic_enabled": self.mic_enabled.isChecked(),
                "mic_device": self.mic_device.currentData(),
            },
            "whisper": {
                "model": self.whisper_model.currentText().strip(),
                "device": device,
                "compute_type": "float16" if device == "cuda" else "int8",
                "beam_size": self.accuracy.currentData(),
                "vocabulary": vocabulary,
            },
            "translator": {
                "engine": self.engine.currentData(),
                "ai_engine": self.ai_engine.currentData(),
                "deepl_api_key": self.deepl_key.text().strip(),
                "anthropic_api_key": self.claude_key.text().strip(),
                "gemini_api_key": self.gemini_key.text().strip(),
                "claude_model": self.claude_model.currentText().strip(),
                "gemini_model": self.gemini_model.currentText().strip(),
            },
            "segmenter": self._segmenter_values(),
            "overlay": {
                "enabled": self.overlay_enabled.isChecked(),
                "show_japanese": self.show_japanese.isChecked(),
                "show_reading": self.show_reading.isChecked(),
                "font_size": self.font_size.value(),
                "opacity": round(self.opacity.value(), 2),
            },
            "log": {"enabled": self.log_enabled.isChecked()},
        }

    def save(self) -> None:
        ctx = self.ctx
        try:
            save_config(ctx.config_path, self.collect())
            cfg = load_config(ctx.config_path)
        except Exception as e:
            self.message.setText(f"저장하지 못했어요: {e}")
            return
        ctx.cfg = cfg
        ctx.assistant = Assistant(cfg.translator, ctx.engine.reader)
        ctx.engine.apply_config(cfg)
        if ctx.overlay is not None:
            ctx.overlay.apply_config(cfg.overlay)
        self.on_applied()
        listening = ctx.engine.state != "ready"
        self.message.setText("저장했어요." + (" 음성 설정은 다음 통화부터 적용됩니다." if listening else ""))
