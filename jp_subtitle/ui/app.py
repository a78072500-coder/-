"""프로그램을 조립해서 실행한다."""

from __future__ import annotations

import signal
import sys
from pathlib import Path

from PySide6.QtWidgets import QApplication

from ..assistant import Assistant
from ..config import Config
from ..engine import CallEngine
from ..overlay import SinkBridge, SubtitleOverlay, app_icon
from ..store import HistoryStore
from .common import AppContext
from .main_window import MainWindow
from .theme import STYLESHEET


def build_app(
    cfg: Config, config_path: Path, log_dir: Path, input_path: Path | None = None
) -> tuple[AppContext, MainWindow]:
    """창과 엔진을 만든다. QApplication 은 먼저 만들어 두어야 한다."""
    bridge = SinkBridge()
    store = HistoryStore(log_dir / "history.db")
    engine = CallEngine(cfg, config_path, log_dir, bridge, store, input_path)
    assistant = Assistant(cfg.translator)
    ctx = AppContext(cfg, config_path, log_dir, bridge, store, engine, assistant)
    window_holder: list[MainWindow] = []
    ctx.overlay = SubtitleOverlay(
        cfg.overlay, bridge, open_main=lambda: window_holder[0].bring_to_front()
    )
    window = MainWindow(ctx)
    window_holder.append(window)
    return ctx, window


def run_app(cfg: Config, config_path: Path, log_dir: Path, input_path: Path | None = None) -> int:
    app = QApplication(sys.argv[:1])
    app.setApplicationName("일본어 통화 도우미")
    app.setWindowIcon(app_icon())
    app.setStyleSheet(STYLESHEET)
    ctx, window = build_app(cfg, config_path, log_dir, input_path)
    signal.signal(signal.SIGINT, lambda *_: app.quit())
    app.aboutToQuit.connect(ctx.engine.shutdown)
    app.aboutToQuit.connect(ctx.overlay.close)
    window.show()
    if input_path:
        ctx.engine.start()  # wav 파일로 시험할 때는 준비되면 바로 재생한다
    ctx.engine.prepare()
    return app.exec()
