"""진단 화면: 자막이 왜 늦는지 구간별 시간과 실행 기록으로 보여준다."""

from __future__ import annotations

from PySide6.QtWidgets import (
    QAbstractItemView,
    QComboBox,
    QHBoxLayout,
    QHeaderView,
    QPlainTextEdit,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from ..store import LatencyStat
from .common import AppContext, make_button, make_label, make_page
from .history_page import session_title

LOG_TAIL_CHARS = 20000


def latency_advice(stats: list[LatencyStat]) -> str:
    """가장 크게 줄일 수 있는 구간을 알려준다."""
    if not stats:
        return "아직 측정한 통화가 없어요. 통화를 한 번 하면 여기에 구간별 시간이 나옵니다."
    typical = {stat.name: stat.typical for stat in stats}
    total = typical.get("한국어까지", 0.0)
    head = f"보통 말이 끝나고 {total:.1f}초 뒤에 한국어가 나옵니다. "
    if typical.get("인식 대기", 0.0) > 0.3:
        return head + "음성인식이 밀리고 있어요. 설정에서 정확도를 '빠르게' 쪽으로 낮춰 보세요."
    if typical.get("음성인식", 0.0) > 1.0:
        return head + "음성인식이 오래 걸려요. 그래픽카드를 쓰는지 확인하고, 정확도를 낮춰 보세요."
    if typical.get("번역", 0.0) > 1.0:
        return head + "번역 응답이 가장 느려요. 인터넷 상태를 보거나 번역 엔진을 바꿔 보세요."
    if typical.get("문장 끝 판단", 0.0) >= 0.5:
        return head + "가장 큰 부분은 문장이 끝났는지 기다리는 시간이에요. [통화] 화면의 문장 길이를 왼쪽으로 옮기면 줄어듭니다."
    return head + "지금은 어느 구간도 크게 느리지 않아요."


class DiagnosticsPage(QWidget):
    def __init__(self, ctx: AppContext, hotkey_failures: list[str]):
        super().__init__()
        self.setObjectName("page")
        self.ctx = ctx
        self.hotkey_failures = hotkey_failures

        page, layout = make_page()
        root = QVBoxLayout(self)
        root.setContentsMargins(0, 0, 0, 0)
        root.addWidget(page)

        layout.addWidget(make_label("진단", "title"))
        self.engine_info = make_label("", "muted")
        layout.addWidget(self.engine_info)

        row = QHBoxLayout()
        row.addWidget(make_label("지연 시간", "section", wrap=False))
        self.scope = QComboBox()
        self.scope.currentIndexChanged.connect(lambda _: self._fill_table())
        self.scope.setMinimumWidth(240)
        row.addWidget(self.scope)
        row.addStretch(1)
        layout.addLayout(row)

        self.table = QTableWidget(0, 3)
        self.table.setHorizontalHeaderLabels(["구간", "보통", "느릴 때 (상위 10%)"])
        header = self.table.horizontalHeader()
        header.setSectionResizeMode(0, QHeaderView.ResizeMode.Stretch)
        header.setSectionResizeMode(1, QHeaderView.ResizeMode.Stretch)
        header.setSectionResizeMode(2, QHeaderView.ResizeMode.Stretch)
        self.table.verticalHeader().setVisible(False)
        self.table.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        self.table.setSelectionMode(QAbstractItemView.SelectionMode.NoSelection)
        self.table.setFixedHeight(232)  # 구간 6줄이 스크롤 없이 보이는 높이
        layout.addWidget(self.table)
        self.advice = make_label("", "")
        layout.addWidget(self.advice)

        logs = QHBoxLayout()
        logs.addWidget(make_label("실행 기록", "section", wrap=False))
        logs.addStretch(1)
        logs.addWidget(make_button("새로 읽기", "link", self._load_log))
        layout.addLayout(logs)
        self.log_view = QPlainTextEdit()
        self.log_view.setReadOnly(True)
        layout.addWidget(self.log_view, 1)
        self.refresh()

    def refresh(self) -> None:
        current = self.scope.currentData()
        self.scope.blockSignals(True)
        self.scope.clear()
        self.scope.addItem("모든 통화", None)
        for info in self.ctx.store.list_sessions():
            self.scope.addItem(session_title(info), info.id)
        index = self.scope.findData(current)
        self.scope.setCurrentIndex(max(index, 0))
        self.scope.blockSignals(False)
        self._fill_table()
        self._load_log()
        self._update_info()

    def _update_info(self) -> None:
        whisper = self.ctx.cfg.whisper
        device = "그래픽카드" if whisper.device == "cuda" else "CPU"
        parts = [f"음성인식 {whisper.model} · {device}", f"번역 {self.ctx.cfg.translator.engine}"]
        if self.ctx.engine.notice:
            parts.append(self.ctx.engine.notice)
        if self.hotkey_failures:
            parts.append(
                "다른 프로그램이 쓰고 있어 등록하지 못한 단축키: " + ", ".join(self.hotkey_failures)
            )
        self.engine_info.setText(" · ".join(parts))

    def _fill_table(self) -> None:
        stats = self.ctx.store.latency_stats(self.scope.currentData())
        self.table.setRowCount(len(stats))
        for row, stat in enumerate(stats):
            self.table.setItem(row, 0, QTableWidgetItem(stat.name))
            self.table.setItem(row, 1, QTableWidgetItem(f"{stat.typical:.2f}초"))
            self.table.setItem(row, 2, QTableWidgetItem(f"{stat.slow:.2f}초"))
        self.advice.setText(latency_advice(stats))

    def _load_log(self) -> None:
        path = self.ctx.config_path.parent / "jp_subtitle.log"
        if not path.exists():
            self.log_view.setPlainText("실행 기록은 설치한 프로그램으로 실행할 때 저장됩니다.")
            return
        text = path.read_text(encoding="utf-8", errors="replace")[-LOG_TAIL_CHARS:]
        self.log_view.setPlainText(text)
        bar = self.log_view.verticalScrollBar()
        bar.setValue(bar.maximum())
