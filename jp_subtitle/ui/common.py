"""여러 화면에서 함께 쓰는 위젯과 도구."""

from __future__ import annotations

import threading
import traceback
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

from PySide6.QtCore import QObject, Qt, Signal, Slot
from PySide6.QtWidgets import (
    QApplication,
    QFrame,
    QHBoxLayout,
    QLabel,
    QLayout,
    QPushButton,
    QScrollArea,
    QSizePolicy,
    QVBoxLayout,
    QWidget,
)

from ..assistant import Assistant, AssistantError, Option
from ..config import Config
from ..engine import CallEngine
from ..models import Line
from ..overlay import SinkBridge, SubtitleOverlay
from ..store import HistoryStore


@dataclass
class AppContext:
    """화면들이 함께 쓰는 프로그램 상태."""

    cfg: Config
    config_path: Path
    log_dir: Path
    bridge: SinkBridge
    store: HistoryStore
    engine: CallEngine
    assistant: Assistant
    overlay: SubtitleOverlay | None = None


# --- 백그라운드 작업 ----------------------------------------------------------------


class _Relay(QObject):
    """작업 스레드의 결과를 GUI 스레드에서 실행한다."""

    fire = Signal(object, object)

    def __init__(self) -> None:
        super().__init__()
        self.fire.connect(self._call, Qt.ConnectionType.QueuedConnection)

    @Slot(object, object)
    def _call(self, callback, value) -> None:
        callback(value)


_relay: _Relay | None = None


def run_async(
    work: Callable[[], object],
    on_done: Callable[[object], None],
    on_error: Callable[[str], None],
) -> None:
    """화면이 멈추지 않도록 다른 스레드에서 실행하고 결과만 화면 쪽으로 돌려준다."""
    global _relay
    if _relay is None:
        _relay = _Relay()
    relay = _relay

    def target() -> None:
        try:
            result = work()
        except AssistantError as e:
            relay.fire.emit(on_error, str(e))
        except Exception as e:
            traceback.print_exc()
            relay.fire.emit(on_error, f"오류가 났어요: {e}")
        else:
            relay.fire.emit(on_done, result)

    threading.Thread(target=target, daemon=True, name="ui_task").start()


# --- 위젯 만들기 --------------------------------------------------------------------


def make_label(
    text: str = "", role: str = "", wrap: bool = True, selectable: bool = False
) -> QLabel:
    label = QLabel(text)
    label.setTextFormat(Qt.TextFormat.PlainText)
    if role:
        label.setProperty("role", role)
    label.setWordWrap(wrap)
    if wrap:
        label.setSizePolicy(QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Minimum)
    if selectable:
        label.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
    return label


def make_button(text: str, role: str = "", on_click: Callable | None = None) -> QPushButton:
    button = QPushButton(text)
    if role:
        button.setProperty("role", role)
    if on_click is not None:
        button.clicked.connect(lambda: on_click())
    return button


def set_role(widget: QWidget, role: str) -> None:
    """실행 중에 역할을 바꾸면 스타일을 다시 계산해야 한다."""
    widget.setProperty("role", role)
    widget.style().unpolish(widget)
    widget.style().polish(widget)


def make_card(role: str = "card") -> tuple[QFrame, QVBoxLayout]:
    frame = QFrame()
    frame.setProperty("role", role)
    layout = QVBoxLayout(frame)
    layout.setContentsMargins(14, 11, 14, 11)
    layout.setSpacing(4)
    return frame, layout


def make_scroll() -> tuple[QScrollArea, QVBoxLayout]:
    """세로로만 스크롤되는 영역과 그 안의 레이아웃."""
    area = QScrollArea()
    area.setWidgetResizable(True)
    area.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
    body = QWidget()
    body.setObjectName("scrollBody")
    layout = QVBoxLayout(body)
    area.setWidget(body)
    return area, layout


def make_page() -> tuple[QWidget, QVBoxLayout]:
    page = QWidget()
    page.setObjectName("page")
    layout = QVBoxLayout(page)
    layout.setContentsMargins(24, 20, 24, 20)
    layout.setSpacing(12)
    return page, layout


def clear_layout(layout: QLayout) -> None:
    while layout.count():
        item = layout.takeAt(0)
        if widget := item.widget():
            widget.setParent(None)
            widget.deleteLater()
        elif child := item.layout():
            clear_layout(child)


# --- 문장 카드 ----------------------------------------------------------------------


def option_card(
    option: Option,
    on_save: Callable[[Option], bool] | None = None,
    on_overlay: Callable[[Option], None] | None = None,
) -> QFrame:
    """답변 문장 하나: 일본어, 읽는 법, 한글 발음, 뜻, 그리고 버튼들."""
    card, layout = make_card()
    if option.label:
        row = QHBoxLayout()
        row.addWidget(make_label(option.label, "tag", wrap=False))
        row.addStretch()
        layout.addLayout(row)
    layout.addWidget(make_label(option.ja, "jaBig", selectable=True))
    if option.reading:
        layout.addWidget(make_label(option.reading, "reading", selectable=True))
    if option.hangul:
        layout.addWidget(make_label(option.hangul, "hangul", selectable=True))
    if option.ko:
        layout.addWidget(make_label(option.ko, "muted", selectable=True))

    buttons = QHBoxLayout()
    buttons.setSpacing(2)
    buttons.addStretch()
    if on_overlay:
        buttons.addWidget(make_button("자막 창에 띄우기", "link", lambda: on_overlay(option)))

    def copy() -> None:
        QApplication.clipboard().setText(option.ja)

    buttons.addWidget(make_button("복사", "link", copy))
    if on_save:
        save = make_button("복습에 추가", "link")

        def saved() -> None:
            on_save(option)
            save.setText("추가됨")
            save.setEnabled(False)

        save.clicked.connect(lambda: saved())
        buttons.addWidget(save)
    layout.addLayout(buttons)
    return card


class TranscriptRow(QWidget):
    """대화 기록의 한 줄. 말하는 중에는 임시 문장을, 끝나면 번역까지 보여준다."""

    def __init__(self, on_save: Callable[[Line], bool] | None = None):
        super().__init__()
        self.on_save = on_save
        self.line: Line | None = None
        self._outer = QHBoxLayout(self)
        self._outer.setContentsMargins(0, 0, 0, 0)
        self._body: QWidget | None = None
        self._ko: QLabel | None = None
        self._ja: QLabel | None = None

    def _replace(self, widget: QWidget) -> None:
        clear_layout(self._outer)
        self._outer.addWidget(widget)
        self._body = widget

    def show_partial(self, text: str) -> None:
        if self._ja is None:
            card, layout = make_card()
            self._ja = make_label(text, "jaPartial")
            layout.addWidget(self._ja)
            self._replace(card)
        else:
            self._ja.setText(text)

    def show_final(self, line: Line, show_reading: bool = True) -> None:
        self.line = line
        self._ja = None
        if line.backchannel:
            self._replace(self._backchannel(line))
        elif line.speaker == "me":
            self._replace(self._mine(line))
        else:
            self._replace(self._theirs(line, show_reading))

    def _backchannel(self, line: Line) -> QWidget:
        return make_label(f"{line.ja}   {line.ko}", "aside", selectable=True)

    def _mine(self, line: Line) -> QWidget:
        wrapper = QWidget()
        row = QHBoxLayout(wrapper)
        row.setContentsMargins(0, 0, 0, 0)
        card, layout = make_card("mine")
        layout.addWidget(make_label("나", "faint", wrap=False))
        layout.addWidget(make_label(line.ja, "ja", selectable=True))
        row.addStretch(1)
        row.addWidget(card, 3)
        return wrapper

    def _theirs(self, line: Line, show_reading: bool) -> QWidget:
        card, layout = make_card()
        top = QHBoxLayout()
        top.setSpacing(8)
        top.addWidget(make_label(line.ja, "ja", selectable=True), 1)
        if self.on_save:
            save = make_button("복습에 추가", "link")

            def saved() -> None:
                self.on_save(self.line)
                save.setText("추가됨")
                save.setEnabled(False)

            save.clicked.connect(lambda: saved())
            top.addWidget(save, 0, Qt.AlignmentFlag.AlignTop)
        layout.addLayout(top)
        if line.reading and show_reading:
            layout.addWidget(make_label(line.reading, "reading", selectable=True))
        if line.low_confidence:
            layout.addWidget(make_label("잘못 들었을 수 있어요. 맞는지 되물어 보세요.", "warn"))
        self._ko = make_label(line.ko, "ko", selectable=True)
        self._ko.setVisible(bool(line.ko))
        layout.addWidget(self._ko)
        return card

    def show_translation(self, line: Line) -> None:
        self.line = line
        if self._ko is not None:
            self._ko.setText(line.ko)
            self._ko.setVisible(bool(line.ko))


class TranscriptView(QScrollArea):
    """대화 기록을 위에서 아래로 쌓는다. 맨 아래를 보고 있으면 새 문장을 따라간다."""

    MAX_ROWS = 400

    def __init__(self, empty_text: str, on_save: Callable[[Line], bool] | None = None):
        super().__init__()
        self.on_save = on_save
        self.show_reading = True
        self._rows: dict[int, TranscriptRow] = {}
        self._follow = True
        self.setWidgetResizable(True)
        self.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        body = QWidget()
        body.setObjectName("scrollBody")
        self._layout = QVBoxLayout(body)
        self._layout.setContentsMargins(0, 0, 6, 0)
        self._layout.setSpacing(8)
        self._empty = make_label(empty_text, "empty")
        self._empty.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._layout.addStretch(1)
        self._layout.addWidget(self._empty)
        self._layout.addStretch(1)
        self.setWidget(body)
        bar = self.verticalScrollBar()
        bar.rangeChanged.connect(self._on_range)
        bar.valueChanged.connect(self._on_value)

    def _on_range(self, _low: int, high: int) -> None:
        if self._follow:
            self.verticalScrollBar().setValue(high)

    def _on_value(self, value: int) -> None:
        self._follow = value >= self.verticalScrollBar().maximum() - 40

    def _row(self, uid: int) -> TranscriptRow:
        row = self._rows.get(uid)
        if row is None:
            if not self._rows:
                self._empty.setVisible(False)
            row = TranscriptRow(self.on_save)
            self._rows[uid] = row
            self._layout.insertWidget(self._layout.count() - 1, row)
            if len(self._rows) > self.MAX_ROWS:
                oldest = next(iter(self._rows))
                old = self._rows.pop(oldest)
                self._layout.removeWidget(old)
                old.deleteLater()
        return row

    def on_partial(self, uid: int, text: str) -> None:
        if not text:
            row = self._rows.get(uid)
            if row is not None and row.line is None:
                self._rows.pop(uid)
                self._layout.removeWidget(row)
                row.deleteLater()
            return
        row = self._row(uid)
        if row.line is None:
            row.show_partial(text)

    def on_final(self, line: Line) -> None:
        self._row(line.uid).show_final(line, self.show_reading)

    def on_translated(self, line: Line) -> None:
        row = self._rows.get(line.uid)
        if row is not None:
            row.show_translation(line)

    def set_show_reading(self, value: bool) -> None:
        """이미 나온 문장에도 바로 적용한다."""
        self.show_reading = value
        for row in self._rows.values():
            if row.line is not None:
                row.show_final(row.line, value)

    def clear(self) -> None:
        for row in self._rows.values():
            self._layout.removeWidget(row)
            row.hide()  # deleteLater 가 처리되기 전에 겹쳐 보이지 않게
            row.deleteLater()
        self._rows.clear()
        self._empty.setVisible(True)
        self._follow = True

    def set_lines(self, lines: list[Line]) -> None:
        self.clear()
        self._follow = False
        for line in lines:
            self.on_final(line)
        self.verticalScrollBar().setValue(0)
