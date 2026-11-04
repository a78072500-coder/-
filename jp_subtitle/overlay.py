"""항상 위에 떠 있는 반투명 자막 창 (PySide6)."""

from __future__ import annotations

import html
import subprocess
import sys
from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

from PySide6.QtCore import QObject, QPoint, QSettings, Qt, QTimer, QUrl, Signal
from PySide6.QtGui import (
    QAction,
    QColor,
    QDesktopServices,
    QFont,
    QGuiApplication,
    QIcon,
    QPainter,
    QPixmap,
)
from PySide6.QtWidgets import (
    QApplication,
    QLabel,
    QMenu,
    QSizeGrip,
    QSystemTrayIcon,
    QVBoxLayout,
    QWidget,
)

from .config import OverlayConfig
from .models import ME, Line
from .ui.theme import JA_FONT

HINT_SECONDS = 12
_JA_CSS = "font-family:" + JA_FONT.replace('"', "'") + ";"


def open_path(path: Path) -> None:
    if not path.exists() and path.suffix == "":
        path.mkdir(parents=True, exist_ok=True)
    if path.is_file():
        # .toml 은 연결된 앱이 없는 경우가 많아서 텍스트 편집기로 직접 연다
        if sys.platform == "darwin":
            subprocess.Popen(["open", "-t", str(path)])
            return
        if sys.platform == "win32":
            subprocess.Popen(["notepad.exe", str(path)])
            return
    QDesktopServices.openUrl(QUrl.fromLocalFile(str(path)))


def app_icon() -> QIcon:
    """볼륨 아이콘과 헷갈리지 않게 노란 바탕에 '字' 를 그린 아이콘."""
    pixmap = QPixmap(64, 64)
    pixmap.fill(Qt.GlobalColor.transparent)
    painter = QPainter(pixmap)
    painter.setRenderHint(QPainter.RenderHint.Antialiasing)
    painter.setPen(Qt.PenStyle.NoPen)
    painter.setBrush(QColor("#ffcc33"))
    painter.drawRoundedRect(2, 2, 60, 60, 14, 14)
    painter.setPen(QColor("#222222"))
    font = QFont()
    font.setPixelSize(44)
    font.setBold(True)
    painter.setFont(font)
    painter.drawText(pixmap.rect(), Qt.AlignmentFlag.AlignCenter, "字")
    painter.end()
    return QIcon(pixmap)


@dataclass
class _Line:
    ja: str = ""
    ko: str = ""
    reading: str = ""
    final: bool = False
    translated: bool = False
    low_confidence: bool = False


class SinkBridge(QObject):
    """작업 스레드에서 GUI 스레드로 결과를 넘기는 신호 묶음 (pipeline.Sink 구현)."""

    partial_sig = Signal(int, str)
    final_sig = Signal(object)
    translated_sig = Signal(object)
    status_sig = Signal(str)
    level_sig = Signal(str, float)

    def partial(self, uid: int, ja: str) -> None:
        self.partial_sig.emit(uid, ja)

    def final(self, line: Line) -> None:
        self.final_sig.emit(line)

    def translated(self, line: Line) -> None:
        self.translated_sig.emit(line)

    def status(self, text: str) -> None:
        self.status_sig.emit(text)

    def level(self, speaker: str, value: float) -> None:
        self.level_sig.emit(speaker, value)


class SubtitleOverlay(QWidget):
    hidden_by_user = Signal()  # 메뉴로 숨겼을 때 메인 창의 체크도 함께 끄려고

    def __init__(
        self,
        cfg: OverlayConfig,
        bridge: SinkBridge,
        open_main: Callable[[], None] | None = None,
    ):
        super().__init__(
            None,
            Qt.WindowType.FramelessWindowHint
            | Qt.WindowType.WindowStaysOnTopHint
            | Qt.WindowType.Tool,
        )
        self.cfg = cfg
        self.open_main = open_main
        # 맑은 고딕은 맥에 없으므로 기본값일 때는 맥 기본 한글 글꼴로
        self.font_family = (
            "Apple SD Gothic Neo"
            if sys.platform == "darwin" and cfg.font_family == OverlayConfig.font_family
            else cfg.font_family
        )
        self.font_size = cfg.font_size
        self.show_japanese = cfg.show_japanese
        self.show_reading = cfg.show_reading
        self._lines: OrderedDict[int, _Line] = OrderedDict()
        self._backchannel: Line | None = None
        self._hint: tuple[str, str, str] | None = None
        self._status = "상대방이 말하면 자막이 나타납니다"
        self._drag_offset: QPoint | None = None
        self._settings = QSettings("jp_subtitle", "overlay")
        self._hint_timer = QTimer(self, singleShot=True, interval=HINT_SECONDS * 1000)
        self._hint_timer.timeout.connect(self.clear_hint)

        self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground)
        # 맥에서 Tool 창은 다른 앱을 누르면 숨겨지므로 항상 보이게
        self.setAttribute(Qt.WidgetAttribute.WA_MacAlwaysShowToolWindow)
        self.setWindowTitle("일본어 실시간 자막")

        self.label = QLabel(self)
        self.label.setObjectName("overlayLabel")
        self.label.setWordWrap(True)
        self.label.setTextFormat(Qt.TextFormat.RichText)
        self.label.setAlignment(Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignBottom)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(18, 12, 18, 12)
        layout.addWidget(self.label)
        layout.addWidget(
            QSizeGrip(self), 0, Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignBottom
        )

        bridge.partial_sig.connect(self.on_partial)
        bridge.final_sig.connect(self.on_final)
        bridge.translated_sig.connect(self.on_translated)
        bridge.status_sig.connect(self.on_status)

        self._restore_geometry()
        self._render()

    # --- 창 위치 ---------------------------------------------------------------

    def _restore_geometry(self) -> None:
        saved = self._settings.value("geometry")
        if saved is not None and self.cfg.x < 0 and self.cfg.y < 0:
            self.restoreGeometry(saved)
            return
        self.resize(self.cfg.width, 160)
        screen = QGuiApplication.primaryScreen().availableGeometry()
        x = self.cfg.x if self.cfg.x >= 0 else screen.x() + (screen.width() - self.width()) // 2
        y = self.cfg.y if self.cfg.y >= 0 else screen.y() + screen.height() - self.height() - 60
        self.move(x, y)

    def closeEvent(self, event) -> None:
        self._settings.setValue("geometry", self.saveGeometry())
        super().closeEvent(event)

    # --- 메뉴 ------------------------------------------------------------------

    def contextMenuEvent(self, event) -> None:
        menu = QMenu(self)
        if self.open_main:
            menu.addAction("메인 창 열기", self.open_main)
            menu.addSeparator()
        toggle_ja = QAction("일본어 원문 표시", menu, checkable=True, checked=self.show_japanese)
        toggle_ja.toggled.connect(self._set_show_japanese)
        menu.addAction(toggle_ja)
        toggle_reading = QAction("읽는 법 표시", menu, checkable=True, checked=self.show_reading)
        toggle_reading.toggled.connect(self._set_show_reading)
        menu.addAction(toggle_reading)
        menu.addAction("글자 크게 (Ctrl+휠↑)", lambda: self._change_font(+2))
        menu.addAction("글자 작게 (Ctrl+휠↓)", lambda: self._change_font(-2))
        menu.addAction("자막 지우기", self._clear)
        menu.addSeparator()
        menu.addAction("자막 창 숨기기", self._hide_by_user)
        menu.exec(event.globalPos())

    def _hide_by_user(self) -> None:
        self.hide()
        self.hidden_by_user.emit()

    def _set_show_japanese(self, value: bool) -> None:
        self.show_japanese = value
        self._render()

    def _set_show_reading(self, value: bool) -> None:
        self.show_reading = value
        self._render()

    def _change_font(self, delta: int) -> None:
        self.font_size = max(10, min(60, self.font_size + delta))
        self._render()

    def _clear(self) -> None:
        self._lines.clear()
        self._backchannel = None
        self._render()

    def apply_config(self, cfg: OverlayConfig) -> None:
        self.cfg = cfg
        self.font_size = cfg.font_size
        self.show_japanese = cfg.show_japanese
        self.show_reading = cfg.show_reading
        self._render()
        self.update()

    # --- 마우스: 드래그 이동, Ctrl+휠 글자 크기 ------------------------------------

    def mousePressEvent(self, event) -> None:
        if event.button() == Qt.MouseButton.LeftButton:
            self._drag_offset = event.globalPosition().toPoint() - self.frameGeometry().topLeft()

    def mouseMoveEvent(self, event) -> None:
        if self._drag_offset is not None and event.buttons() & Qt.MouseButton.LeftButton:
            self.move(event.globalPosition().toPoint() - self._drag_offset)

    def mouseReleaseEvent(self, event) -> None:
        self._drag_offset = None
        self._settings.setValue("geometry", self.saveGeometry())

    def wheelEvent(self, event) -> None:
        if event.modifiers() & Qt.KeyboardModifier.ControlModifier:
            self._change_font(+1 if event.angleDelta().y() > 0 else -1)

    def paintEvent(self, event) -> None:
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        painter.setPen(Qt.PenStyle.NoPen)
        painter.setBrush(QColor(0, 0, 0, int(255 * self.cfg.opacity)))
        painter.drawRoundedRect(self.rect(), 14, 14)

    # --- 파이프라인 결과 -----------------------------------------------------------

    def _line(self, uid: int) -> _Line:
        if uid not in self._lines:
            self._lines[uid] = _Line()
            while len(self._lines) > self.cfg.history_lines + 1:
                self._lines.popitem(last=False)
        return self._lines[uid]

    def on_partial(self, uid: int, ja: str) -> None:
        if not ja:
            # 확정 결과가 비었으면(잡음 등) 임시 줄을 지운다
            line = self._lines.get(uid)
            if line and not line.final:
                del self._lines[uid]
        else:
            line = self._line(uid)
            if not line.final:
                line.ja = ja
        self._status = ""
        self._render()

    def on_final(self, line: Line) -> None:
        if line.speaker == ME:
            return
        self._status = ""
        if line.backchannel:
            # 맞장구는 앞 문장을 밀어내지 않고 맨 아래 한 줄에만 잠깐 보여 준다
            self._lines.pop(line.uid, None)
            self._backchannel = line
        else:
            item = self._line(line.uid)
            item.ja, item.reading, item.final = line.ja, line.reading, True
            item.low_confidence = line.low_confidence
            self._backchannel = None
        self._render()

    def on_translated(self, line: Line) -> None:
        if line.speaker == ME or line.backchannel:
            return
        item = self._lines.get(line.uid)
        if item is not None:
            item.ko, item.translated = line.ko, True
            self._render()

    def on_status(self, text: str) -> None:
        self._status = text
        self._render()

    def show_hint(self, ja: str, reading: str, korean: str) -> None:
        """단축키로 꺼낸 문구를 자막 위쪽에 잠깐 크게 보여 준다."""
        self._hint = (ja, reading, korean)
        self._hint_timer.start()
        self._render()

    def clear_hint(self) -> None:
        self._hint = None
        self._render()

    def _render(self) -> None:
        fs = self.font_size
        font = QFont(self.font_family)
        self.label.setFont(font)
        family = html.escape(self.font_family)
        parts: list[str] = []
        if self._hint:
            ja, reading, korean = self._hint
            parts.append(
                f'<div style="color:#8fe3b0;font-size:{fs}px;font-weight:600;{_JA_CSS}">{html.escape(ja)}</div>'
            )
            if reading:
                parts.append(
                    f'<div style="color:#b8e8cb;font-size:{int(fs * 0.7)}px;{_JA_CSS}">{html.escape(reading)}</div>'
                )
            if korean:
                parts.append(
                    f'<div style="color:#9fb5a8;font-size:{int(fs * 0.7)}px;">{html.escape(korean)}</div>'
                )
            parts.append('<div style="font-size:8px">&nbsp;</div>')
        items = list(self._lines.values())
        for i, line in enumerate(items):
            # Qt 리치텍스트는 opacity 를 지원하지 않아 이전 줄은 어두운 색으로 표시한다
            latest = i == len(items) - 1
            if self.show_japanese and line.ja:
                if not line.final:
                    color, style = "#9a9a9a", "font-style:italic;"
                else:
                    color, style = ("#ffffff" if latest else "#8c8c8c"), ""
                mark = (
                    '<span style="color:#ffb454;font-weight:600;"> ? 잘못 들었을 수 있어요</span>'
                    if line.low_confidence else ""
                )
                parts.append(
                    f'<div style="color:{color};font-size:{int(fs * 0.8)}px;{style}">'
                    f'<span style="{_JA_CSS}">{html.escape(line.ja)}</span>{mark}</div>'
                )
                if self.show_reading and line.reading:
                    parts.append(
                        f'<div style="color:{"#b9b9b9" if latest else "#777"};{_JA_CSS}'
                        f'font-size:{int(fs * 0.62)}px;">{html.escape(line.reading)}</div>'
                    )
            if line.ko:
                color = "#ffe066" if latest else "#a8964a"
                parts.append(
                    f'<div style="color:{color};font-size:{fs}px;font-weight:600;">'
                    f"{html.escape(line.ko)}</div>"
                )
            elif line.final and not line.translated:
                parts.append(
                    f'<div style="color:#777;font-size:{int(fs * 0.7)}px;">번역 중…</div>'
                )
            if not latest:
                parts.append('<div style="font-size:6px">&nbsp;</div>')
        if self._backchannel:
            parts.append(
                f'<div style="color:#8a8a8a;font-size:{int(fs * 0.7)}px;">'
                f"{html.escape(self._backchannel.ja)} · {html.escape(self._backchannel.ko)}</div>"
            )
        if self._status:
            parts.append(
                f'<div style="color:#8fd3ff;font-size:{int(fs * 0.7)}px">'
                f"{html.escape(self._status)}</div>"
            )
        self.label.setText(f'<div style="font-family:\'{family}\'">{"".join(parts)}</div>')
        # 내용에 맞춰 높이만 조절 (너비는 사용자가 정한 대로)
        height = self.label.heightForWidth(self.width() - 36) + 40
        bottom = self.geometry().bottom()
        self.resize(self.width(), max(70, height))
        if self.isVisible():
            self.move(self.x(), bottom - self.height() + 1)
