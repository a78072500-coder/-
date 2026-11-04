"""기록 화면: 지난 통화 보기, AI 복습 만들기, 기록 삭제."""

from __future__ import annotations

from datetime import datetime

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QHBoxLayout,
    QListWidget,
    QListWidgetItem,
    QMessageBox,
    QSplitter,
    QTabWidget,
    QVBoxLayout,
    QWidget,
)

from ..models import Line
from ..store import SessionInfo
from .common import (
    AppContext,
    TranscriptView,
    clear_layout,
    make_button,
    make_card,
    make_label,
    make_page,
    make_scroll,
    run_async,
)

_WEEKDAYS = "월화수목금토일"


def session_title(info: SessionInfo) -> str:
    started = datetime.fromisoformat(info.started_at)
    period = "오전" if started.hour < 12 else "오후"
    hour = started.hour % 12 or 12
    day = f"{started.month}월 {started.day}일 ({_WEEKDAYS[started.weekday()]})"
    return f"{day} {period} {hour}:{started.minute:02d}"


class HistoryPage(QWidget):
    def __init__(self, ctx: AppContext, on_cards_changed):
        super().__init__()
        self.setObjectName("page")
        self.ctx = ctx
        self.on_cards_changed = on_cards_changed
        self._session_id: int | None = None
        self._reviewing: set[int] = set()

        page, layout = make_page()
        root = QVBoxLayout(self)
        root.setContentsMargins(0, 0, 0, 0)
        root.addWidget(page)

        layout.addWidget(make_label("지난 통화", "title"))
        split = QSplitter(Qt.Orientation.Horizontal)
        split.setChildrenCollapsible(False)
        split.setHandleWidth(1)

        self.list = QListWidget()
        self.list.setWordWrap(True)
        self.list.currentItemChanged.connect(self._on_select)
        self.list.setMinimumWidth(240)
        split.addWidget(self.list)

        right = QWidget()
        right_layout = QVBoxLayout(right)
        right_layout.setContentsMargins(18, 0, 0, 0)
        right_layout.setSpacing(10)

        header = QHBoxLayout()
        self.heading = make_label("", "section")
        header.addWidget(self.heading, 1)
        self.delete_button = make_button("이 기록 삭제", "danger", self._delete)
        header.addWidget(self.delete_button)
        right_layout.addLayout(header)

        self.tabs = QTabWidget()
        self.view = TranscriptView("왼쪽에서 통화를 고르세요.", self._save_line)
        self.tabs.addTab(self.view, "대화")
        self.tabs.addTab(self._build_review_tab(), "AI 복습")
        right_layout.addWidget(self.tabs, 1)
        split.addWidget(right)
        split.setStretchFactor(0, 1)
        split.setStretchFactor(1, 3)
        split.setSizes([280, 760])
        layout.addWidget(split, 1)
        self.refresh()

    # --- 목록 ------------------------------------------------------------------

    def refresh(self) -> None:
        current = self._session_id
        sessions = self.ctx.store.list_sessions()
        self.list.blockSignals(True)
        self.list.clear()
        selected = None
        for info in sessions:
            preview = info.preview[:36] + ("…" if len(info.preview) > 36 else "")
            mark = " · 복습 있음" if info.has_review else ""
            item = QListWidgetItem(
                f"{session_title(info)}\n{info.line_count}문장{mark}\n{preview}".rstrip()
            )
            item.setData(Qt.ItemDataRole.UserRole, info.id)
            self.list.addItem(item)
            if info.id == current:
                selected = item
        self.list.blockSignals(False)
        if selected is None and self.list.count():
            selected = self.list.item(0)
        if selected is not None:
            self.list.blockSignals(True)
            self.list.setCurrentItem(selected)
            self.list.blockSignals(False)
            # 통화 중이던 기록이면 그사이 문장이 늘었을 수 있으니 항상 다시 읽는다
            self._show_session(selected.data(Qt.ItemDataRole.UserRole), reload=True)
        else:
            self._session_id = None
            self.view.clear()
            self.heading.setText("아직 기록이 없어요. 통화를 하면 여기에 쌓입니다.")
            self.delete_button.setVisible(False)
            self._render_review(None)

    def _on_select(self, item: QListWidgetItem | None, _previous=None) -> None:
        if item is not None:
            self._show_session(item.data(Qt.ItemDataRole.UserRole))

    def _show_session(self, session_id: int, reload: bool = False) -> None:
        if session_id == self._session_id and not reload:
            return
        self._session_id = session_id
        lines = self.ctx.store.lines(session_id)
        self.view.set_lines(lines)
        said = sum(1 for line in lines if not line.backchannel)
        self.heading.setText(f"{said}문장 · 맞장구 {len(lines) - said}번")
        self.delete_button.setVisible(True)
        self._render_review(self.ctx.store.review(session_id))

    def _delete(self) -> None:
        if self._session_id is None:
            return
        answer = QMessageBox.question(
            self,
            "기록 삭제",
            "이 통화 기록을 지울까요? 되돌릴 수 없습니다.\n복습 카드로 저장한 문장은 그대로 남습니다.",
        )
        if answer == QMessageBox.StandardButton.Yes:
            self.ctx.store.delete_session(self._session_id)
            self._session_id = None
            self.refresh()

    def _save_line(self, line: Line) -> bool:
        added = self.ctx.store.add_card(line.ja, line.ko, session_id=self._session_id)
        self.on_cards_changed()
        return added

    # --- AI 복습 ---------------------------------------------------------------

    def _build_review_tab(self) -> QWidget:
        area, layout = make_scroll()
        layout.setContentsMargins(0, 12, 6, 0)
        layout.setSpacing(10)
        self.review_action = QHBoxLayout()
        self.review_button = make_button("AI 복습 만들기", "primary", self._make_review)
        self.review_action.addWidget(self.review_button)
        self.review_info = make_label("", "faint")
        self.review_action.addWidget(self.review_info, 1)
        layout.addLayout(self.review_action)
        self.review_body = QVBoxLayout()
        self.review_body.setSpacing(10)
        layout.addLayout(self.review_body)
        layout.addStretch(1)
        return area

    def _make_review(self) -> None:
        session_id = self._session_id
        if session_id is None or session_id in self._reviewing:
            return
        self._reviewing.add(session_id)
        self.review_button.setEnabled(False)
        label = self.ctx.assistant.ai_label
        self.review_info.setText(f"대화를 {label}(으)로 보내 정리하는 중이에요. 1분 안팎 걸립니다.")
        lines = self.ctx.store.lines(session_id)

        def done(review: dict) -> None:
            self._reviewing.discard(session_id)
            self.ctx.store.save_review(session_id, review)
            self.refresh()
            if session_id == self._session_id:
                self._render_review(review)

        def failed(message: str) -> None:
            self._reviewing.discard(session_id)
            self.review_button.setEnabled(True)
            self.review_info.setText(message)

        run_async(lambda: self.ctx.assistant.review(lines), done, failed)

    def _render_review(self, review: dict | None) -> None:
        clear_layout(self.review_body)
        assistant = self.ctx.assistant
        busy = self._session_id in self._reviewing
        self.review_button.setEnabled(
            self._session_id is not None and assistant.has_ai and not busy
        )
        if busy:
            return
        if not assistant.has_ai:
            self.review_info.setText("AI 복습은 Claude 또는 Gemini 키가 필요해요. 설정 화면에서 넣어 주세요.")
        elif review:
            self.review_info.setText("다시 만들면 지금 내용을 덮어씁니다.")
        else:
            self.review_info.setText(
                f"대화 내용이 {assistant.ai_label}(으)로 전송됩니다. 유료 키는 사용량만큼 요금이 나와요."
            )
        self.review_button.setText("다시 만들기" if review else "AI 복습 만들기")
        if not review:
            return
        self._review_section("요약", [self._text_card(review.get("summary", ""))])
        expressions = review.get("expressions") or []
        if expressions:
            cards = [self._expression_card(e) for e in expressions]
            add_all = make_button("표현 전부 복습에 추가", "", lambda: self._add_all(expressions))
            self._review_section("외워 둘 표현", cards, add_all)
        corrections = review.get("corrections") or []
        if corrections:
            self._review_section("내 문장 고치기", [self._correction_card(c) for c in corrections])
        topics = review.get("next_topics") or []
        if topics:
            text = "\n".join(f"· {topic}" for topic in topics)
            self._review_section("다음 통화 주제", [self._text_card(text)])

    def _review_section(self, title: str, widgets: list, action=None) -> None:
        header = QHBoxLayout()
        header.addWidget(make_label(title, "section"), 1)
        if action is not None:
            header.addWidget(action)
        self.review_body.addLayout(header)
        for widget in widgets:
            self.review_body.addWidget(widget)

    def _text_card(self, text: str) -> QWidget:
        card, layout = make_card()
        layout.addWidget(make_label(text, "", selectable=True))
        return card

    def _expression_card(self, item: dict) -> QWidget:
        card, layout = make_card()
        top = QHBoxLayout()
        top.addWidget(make_label(item.get("ja", ""), "jaBig", selectable=True), 1)
        save = make_button("복습에 추가", "link")

        def saved() -> None:
            self.ctx.store.add_card(
                item.get("ja", ""), item.get("ko", ""), item.get("note", ""), self._session_id
            )
            self.on_cards_changed()
            save.setText("추가됨")
            save.setEnabled(False)

        save.clicked.connect(lambda: saved())
        top.addWidget(save, 0, Qt.AlignmentFlag.AlignTop)
        layout.addLayout(top)
        reader = self.ctx.engine.reader
        if reader and item.get("ja"):
            layout.addWidget(make_label(reader.annotate(item["ja"]).hiragana, "reading"))
        layout.addWidget(make_label(item.get("ko", ""), "ko", selectable=True))
        if item.get("note"):
            layout.addWidget(make_label(item["note"], "muted", selectable=True))
        return card

    def _correction_card(self, item: dict) -> QWidget:
        card, layout = make_card()
        layout.addWidget(make_label(item.get("original", ""), "reading", selectable=True))
        layout.addWidget(make_label(item.get("corrected", ""), "jaBig", selectable=True))
        if item.get("note"):
            layout.addWidget(make_label(item["note"], "muted", selectable=True))
        return card

    def _add_all(self, expressions: list[dict]) -> None:
        added = sum(
            self.ctx.store.add_card(
                e.get("ja", ""), e.get("ko", ""), e.get("note", ""), self._session_id
            )
            for e in expressions
            if e.get("ja")
        )
        self.on_cards_changed()
        self.review_info.setText(f"새 카드 {added}장을 복습에 추가했어요.")
