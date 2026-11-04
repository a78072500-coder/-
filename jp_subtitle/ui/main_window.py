"""메인 창: 왼쪽 메뉴와 다섯 화면."""

from __future__ import annotations

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QApplication,
    QButtonGroup,
    QHBoxLayout,
    QMainWindow,
    QPushButton,
    QStackedWidget,
    QSystemTrayIcon,
    QMenu,
    QVBoxLayout,
    QWidget,
)

from ..engine import READY
from ..hotkeys import GlobalHotkeys
from ..overlay import app_icon
from .common import AppContext, make_label
from .diagnostics_page import DiagnosticsPage
from .history_page import HistoryPage
from .live_page import LivePage
from .review_page import ReviewPage
from .settings_page import SettingsPage

LIVE, HISTORY, REVIEW, DIAGNOSTICS, SETTINGS = range(5)
NAV_LABELS = ["통화", "기록", "복습", "진단", "설정"]


class MainWindow(QMainWindow):
    def __init__(self, ctx: AppContext):
        super().__init__()
        self.ctx = ctx
        self.setWindowTitle("일본어 통화 도우미")
        self.setWindowIcon(app_icon())
        self.resize(1200, 780)
        self.setMinimumSize(900, 600)

        self.hotkeys = GlobalHotkeys()
        self.live = LivePage(
            ctx,
            lambda: self.go_to(SETTINGS),
            self._refresh_badge,
            lambda: self.settings.load_sentence_values(),
        )
        self.history = HistoryPage(ctx, self._refresh_badge)
        self.review = ReviewPage(ctx, self._refresh_badge)
        self.hotkeys.register("R", self._hotkey_reply)
        self.hotkeys.register("Q", self.live.next_phrase)
        self.hotkeys.register("S", lambda: self.live.request_suggestions(to_overlay=True))
        self.diagnostics = DiagnosticsPage(ctx, self.hotkeys.failed)
        self.settings = SettingsPage(ctx, self._on_settings_applied)

        self.stack = QStackedWidget()
        for page in (self.live, self.history, self.review, self.diagnostics, self.settings):
            self.stack.addWidget(page)

        central = QWidget()
        layout = QHBoxLayout(central)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)
        layout.addWidget(self._build_sidebar())
        layout.addWidget(self.stack, 1)
        self.setCentralWidget(central)

        ctx.engine.state_changed.connect(self._on_engine_state)
        ctx.engine.reader_ready.connect(self._on_reader_ready)
        self._init_tray()
        self._refresh_badge()

    # --- 왼쪽 메뉴 ---------------------------------------------------------------

    def _build_sidebar(self) -> QWidget:
        sidebar = QWidget()
        sidebar.setObjectName("sidebar")
        sidebar.setAttribute(Qt.WidgetAttribute.WA_StyledBackground, True)
        sidebar.setFixedWidth(176)
        layout = QVBoxLayout(sidebar)
        layout.setContentsMargins(12, 20, 12, 16)
        layout.setSpacing(4)
        brand = make_label("일본어 통화 도우미", "brand")
        brand.setContentsMargins(8, 0, 0, 14)
        layout.addWidget(brand)
        self.nav = QButtonGroup(self)
        self.nav.setExclusive(True)
        self.nav_buttons: list[QPushButton] = []
        for index, label in enumerate(NAV_LABELS):
            button = QPushButton(label)
            button.setProperty("role", "nav")
            button.setCheckable(True)
            button.setCursor(Qt.CursorShape.PointingHandCursor)
            self.nav.addButton(button, index)
            self.nav_buttons.append(button)
            layout.addWidget(button)
        layout.addStretch(1)
        self.nav.idClicked.connect(self.go_to)
        self.nav_buttons[LIVE].setChecked(True)
        return sidebar

    def go_to(self, index: int) -> None:
        self.nav_buttons[index].setChecked(True)
        self.stack.setCurrentIndex(index)
        if index == HISTORY:
            self.history.refresh()
        elif index == REVIEW:
            self.review.refresh()
        elif index == DIAGNOSTICS:
            self.diagnostics.refresh()
        elif index == SETTINGS:
            self.settings.message.setText("")

    def _refresh_badge(self) -> None:
        due = len(self.ctx.store.due_cards())
        self.nav_buttons[REVIEW].setText(f"복습  {due}" if due else "복습")

    # --- 신호 -------------------------------------------------------------------

    def _on_engine_state(self, state: str, _message: str) -> None:
        if state == READY:
            self._refresh_badge()
            if self.stack.currentIndex() == HISTORY:
                self.history.refresh()

    def _on_reader_ready(self) -> None:
        self.ctx.assistant.reader = self.ctx.engine.reader

    def _on_settings_applied(self) -> None:
        self.live.apply_config()
        self.history.refresh()
        self.diagnostics.refresh()

    def bring_to_front(self) -> None:
        self.showNormal()
        self.raise_()
        self.activateWindow()

    def _hotkey_reply(self) -> None:
        self.go_to(LIVE)
        self.bring_to_front()
        self.live.focus_reply()

    # --- 트레이와 종료 -----------------------------------------------------------

    def _init_tray(self) -> None:
        self.tray = None
        if not QSystemTrayIcon.isSystemTrayAvailable():
            return
        self.tray = QSystemTrayIcon(app_icon(), self)
        self.tray.setToolTip("일본어 통화 도우미")
        menu = QMenu(self)
        menu.addAction("메인 창 열기", self.bring_to_front)
        menu.addSeparator()
        menu.addAction("종료", QApplication.quit)
        self.tray.setContextMenu(menu)
        self.tray.activated.connect(self._on_tray)
        self.tray.show()

    def _on_tray(self, reason) -> None:
        if reason == QSystemTrayIcon.ActivationReason.Trigger:
            self.bring_to_front()

    def closeEvent(self, event) -> None:
        self.hotkeys.unregister_all()
        event.accept()
        QApplication.quit()
