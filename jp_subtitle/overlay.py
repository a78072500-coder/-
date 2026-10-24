"""항상 위에 떠 있는 반투명 자막 창 (PySide6)."""

from __future__ import annotations

import html
from collections import OrderedDict
from dataclasses import dataclass

from PySide6.QtCore import QObject, QPoint, QSettings, Qt, Signal
from PySide6.QtGui import QAction, QColor, QFont, QGuiApplication, QIcon, QPainter, QPixmap
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


def _tray_icon() -> QIcon:
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
    final: bool = False


class SinkBridge(QObject):
    """작업 스레드에서 GUI 스레드로 결과를 넘기는 신호 묶음 (pipeline.Sink 구현)."""

    partial_sig = Signal(int, str)
    final_sig = Signal(int, str)
    translated_sig = Signal(int, str, str)
    status_sig = Signal(str)

    def partial(self, uid: int, ja: str) -> None:
        self.partial_sig.emit(uid, ja)

    def final(self, uid: int, ja: str) -> None:
        self.final_sig.emit(uid, ja)

    def translated(self, uid: int, ja: str, ko: str) -> None:
        self.translated_sig.emit(uid, ja, ko)

    def status(self, text: str) -> None:
        self.status_sig.emit(text)


class SubtitleOverlay(QWidget):
    def __init__(self, cfg: OverlayConfig, bridge: SinkBridge):
        super().__init__(
            None,
            Qt.WindowType.FramelessWindowHint
            | Qt.WindowType.WindowStaysOnTopHint
            | Qt.WindowType.Tool,
        )
        self.cfg = cfg
        self.font_size = cfg.font_size
        self.show_japanese = cfg.show_japanese
        self._lines: OrderedDict[int, _Line] = OrderedDict()
        self._status = "준비 중… (상대방이 말하면 자막이 나타납니다)"
        self._drag_offset: QPoint | None = None
        self._settings = QSettings("jp_subtitle", "overlay")

        self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground)
        self.setWindowTitle("일본어 실시간 자막")

        self.label = QLabel(self)
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

        self._init_tray()
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

    # --- 트레이 / 메뉴 -----------------------------------------------------------

    def _build_menu(self, menu: QMenu) -> None:
        toggle_ja = QAction("일본어 원문 표시", menu, checkable=True, checked=self.show_japanese)
        toggle_ja.toggled.connect(self._set_show_japanese)
        menu.addAction(toggle_ja)
        menu.addAction("글자 크게 (Ctrl+휠↑)", lambda: self._change_font(+2))
        menu.addAction("글자 작게 (Ctrl+휠↓)", lambda: self._change_font(-2))
        menu.addAction("기록 지우기", self._clear)
        menu.addSeparator()
        menu.addAction("창 숨기기/보이기", self._toggle_visible)
        menu.addAction("종료", QApplication.quit)

    def _init_tray(self) -> None:
        self.tray = None
        if not QSystemTrayIcon.isSystemTrayAvailable():
            return
        self.tray = QSystemTrayIcon(_tray_icon(), self)
        self.tray.setToolTip("일본어 실시간 자막")
        menu = QMenu()
        self._build_menu(menu)
        self.tray.setContextMenu(menu)
        self.tray.activated.connect(
            lambda reason: reason == QSystemTrayIcon.ActivationReason.Trigger
            and self._toggle_visible()
        )
        self.tray.show()

    def contextMenuEvent(self, event) -> None:
        menu = QMenu(self)
        self._build_menu(menu)
        menu.exec(event.globalPos())

    def _toggle_visible(self) -> None:
        self.setVisible(not self.isVisible())

    def _set_show_japanese(self, value: bool) -> None:
        self.show_japanese = value
        self._render()

    def _change_font(self, delta: int) -> None:
        self.font_size = max(10, min(60, self.font_size + delta))
        self._render()

    def _clear(self) -> None:
        self._lines.clear()
        self._render()

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

    def on_final(self, uid: int, ja: str) -> None:
        line = self._line(uid)
        line.ja, line.final = ja, True
        self._status = ""
        self._render()

    def on_translated(self, uid: int, ja: str, ko: str) -> None:
        if uid in self._lines:
            self._lines[uid].ko = ko or "(번역 실패)"
            self._render()

    def on_status(self, text: str) -> None:
        self._status = text
        self._render()
        if self.tray:
            self.tray.showMessage("일본어 실시간 자막", text)

    def _render(self) -> None:
        fs = self.font_size
        font = QFont(self.cfg.font_family)
        self.label.setFont(font)
        family = html.escape(self.cfg.font_family)
        parts: list[str] = []
        items = list(self._lines.values())
        for i, line in enumerate(items):
            # Qt 리치텍스트는 opacity 를 지원하지 않아 이전 줄은 어두운 색으로 표시한다
            latest = i == len(items) - 1
            if self.show_japanese and line.ja:
                if not line.final:
                    color, style = "#9a9a9a", "font-style:italic;"
                else:
                    color, style = ("#ffffff" if latest else "#8c8c8c"), ""
                parts.append(
                    f'<div style="color:{color};font-size:{int(fs * 0.8)}px;{style}">'
                    f"{html.escape(line.ja)}</div>"
                )
            if line.ko:
                color = "#ffe066" if latest else "#a8964a"
                parts.append(
                    f'<div style="color:{color};font-size:{fs}px;font-weight:600;">'
                    f"{html.escape(line.ko)}</div>"
                )
            elif line.final:
                parts.append(
                    f'<div style="color:#777;font-size:{int(fs * 0.7)}px;">번역 중…</div>'
                )
            if not latest:
                parts.append('<div style="font-size:6px">&nbsp;</div>')
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
