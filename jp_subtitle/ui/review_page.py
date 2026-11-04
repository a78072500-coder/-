"""복습 화면: 저장한 문장을 간격 반복 카드로 익힌다."""

from __future__ import annotations

from datetime import date

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QAbstractItemView,
    QFileDialog,
    QHBoxLayout,
    QHeaderView,
    QStackedWidget,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from ..store import Card, export_anki_csv
from .common import AppContext, make_button, make_card, make_label, make_page


def _due_text(due: str) -> str:
    days = (date.fromisoformat(due) - date.today()).days
    if days <= 0:
        return "오늘"
    return "내일" if days == 1 else f"{days}일 뒤"


class ReviewPage(QWidget):
    def __init__(self, ctx: AppContext, on_cards_changed):
        super().__init__()
        self.setObjectName("page")
        self.ctx = ctx
        self.on_cards_changed = on_cards_changed
        self._deck: list[Card] = []
        self._done_today = 0

        page, layout = make_page()
        root = QVBoxLayout(self)
        root.setContentsMargins(0, 0, 0, 0)
        root.addWidget(page)

        header = QHBoxLayout()
        header.addWidget(make_label("복습", "title", wrap=False))
        self.summary = make_label("", "muted")
        header.addWidget(self.summary, 1)
        self.toggle_button = make_button("카드 목록", "", self._toggle_list)
        header.addWidget(self.toggle_button)
        header.addWidget(make_button("Anki로 내보내기", "", self._export))
        layout.addLayout(header)

        self.stack = QStackedWidget()
        self.stack.addWidget(self._build_study())
        self.stack.addWidget(self._build_list())
        layout.addWidget(self.stack, 1)
        self.refresh()

    # --- 공부 화면 --------------------------------------------------------------

    def _build_study(self) -> QWidget:
        holder = QWidget()
        outer = QVBoxLayout(holder)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.addStretch(1)
        self.card_frame, card = make_card()
        self.card_frame.setMaximumWidth(720)
        self.card_frame.setMinimumHeight(300)
        card.setContentsMargins(32, 28, 32, 24)
        card.setSpacing(10)

        self.prompt = make_label("", "faint")
        card.addWidget(self.prompt)
        self.front = make_label("", "koHuge")
        card.addWidget(self.front)
        self.answer_ja = make_label("", "jaHuge", selectable=True)
        self.answer_reading = make_label("", "readingBig", selectable=True)
        self.answer_hangul = make_label("", "hangulBig", selectable=True)
        self.answer_note = make_label("", "muted", selectable=True)
        for widget in (self.answer_ja, self.answer_reading, self.answer_hangul, self.answer_note):
            card.addWidget(widget)
        card.addStretch(1)

        buttons = QHBoxLayout()
        buttons.addStretch(1)
        self.reveal_button = make_button("정답 보기", "primary", self._reveal)
        self.unknown_button = make_button("몰랐어요", "", lambda: self._grade(False))
        self.known_button = make_button("알았어요", "primary", lambda: self._grade(True))
        for button in (self.reveal_button, self.unknown_button, self.known_button):
            button.setMinimumWidth(110)
            buttons.addWidget(button)
        card.addLayout(buttons)

        self.empty = make_label("", "empty")
        self.empty.setAlignment(Qt.AlignmentFlag.AlignCenter)
        row = QHBoxLayout()
        row.addStretch(1)
        row.addWidget(self.card_frame, 8)
        row.addStretch(1)
        outer.addLayout(row)
        outer.addWidget(self.empty)
        outer.addStretch(2)
        return holder

    def refresh(self) -> None:
        self._deck = self.ctx.store.due_cards()
        self._done_today = 0
        self._update_summary()
        self._show_card()
        self._fill_table()

    def _update_summary(self) -> None:
        total = len(self.ctx.store.all_cards())
        left = len({card.id for card in self._deck})
        self.summary.setText(f"오늘 남은 카드 {left}장 · 전체 {total}장")

    def _show_card(self) -> None:
        has_card = bool(self._deck)
        self.card_frame.setVisible(has_card)
        self.empty.setVisible(not has_card)
        if not has_card:
            if self.ctx.store.all_cards():
                self.empty.setText("오늘 복습을 모두 끝냈어요. 다음 카드는 때가 되면 돌아옵니다.")
            else:
                self.empty.setText(
                    "아직 카드가 없어요.\n통화나 기록 화면에서 문장의 '복습에 추가'를 눌러 보세요."
                )
            return
        card = self._deck[0]
        if card.ko:
            self.prompt.setText("이 말을 일본어로 하면?")
            self.front.setText(card.ko)
            self.front.setProperty("role", "koHuge")
        else:
            self.prompt.setText("이 문장의 뜻은?")
            self.front.setText(card.ja)
            self.front.setProperty("role", "jaHuge")
        self.front.style().unpolish(self.front)
        self.front.style().polish(self.front)
        self._set_revealed(False)

    def _set_revealed(self, revealed: bool) -> None:
        card = self._deck[0] if self._deck else None
        self.reveal_button.setVisible(not revealed)
        self.unknown_button.setVisible(revealed)
        self.known_button.setVisible(revealed)
        for widget in (self.answer_ja, self.answer_reading, self.answer_hangul, self.answer_note):
            widget.setVisible(False)
        if not revealed or card is None:
            return
        reader = self.ctx.engine.reader
        note = reader.annotate(card.ja) if reader else None
        if card.ko:
            self.answer_ja.setText(card.ja)
            self.answer_ja.setVisible(True)
        self.answer_reading.setText(note.hiragana if note else "")
        self.answer_reading.setVisible(bool(note))
        self.answer_hangul.setText(note.hangul if note else "")
        self.answer_hangul.setVisible(bool(note))
        self.answer_note.setText(card.note)
        self.answer_note.setVisible(bool(card.note))

    def _reveal(self) -> None:
        self._set_revealed(True)

    def _grade(self, known: bool) -> None:
        if not self._deck:
            return
        card = self._deck.pop(0)
        self.ctx.store.grade_card(card.id, known)
        if not known:
            self._deck.append(card)  # 틀린 카드는 오늘 안에 한 번 더 본다
        self._update_summary()
        self._show_card()
        self._fill_table()
        self.on_cards_changed()

    # --- 카드 목록 --------------------------------------------------------------

    def _build_list(self) -> QWidget:
        holder = QWidget()
        layout = QVBoxLayout(holder)
        layout.setContentsMargins(0, 0, 0, 0)
        self.table = QTableWidget(0, 3)
        self.table.setHorizontalHeaderLabels(["일본어", "뜻", "다음 복습"])
        header = self.table.horizontalHeader()
        header.setSectionResizeMode(0, QHeaderView.ResizeMode.Stretch)
        header.setSectionResizeMode(1, QHeaderView.ResizeMode.Stretch)
        header.setSectionResizeMode(2, QHeaderView.ResizeMode.ResizeToContents)
        self.table.verticalHeader().setVisible(False)
        self.table.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
        self.table.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        self.table.setAlternatingRowColors(False)
        layout.addWidget(self.table, 1)
        buttons = QHBoxLayout()
        buttons.addStretch(1)
        buttons.addWidget(make_button("선택한 카드 삭제", "danger", self._delete_selected))
        layout.addLayout(buttons)
        return holder

    def _fill_table(self) -> None:
        cards = self.ctx.store.all_cards()
        self.table.setRowCount(len(cards))
        for row, card in enumerate(cards):
            first = QTableWidgetItem(card.ja)
            first.setData(Qt.ItemDataRole.UserRole, card.id)
            self.table.setItem(row, 0, first)
            self.table.setItem(row, 1, QTableWidgetItem(card.ko))
            self.table.setItem(row, 2, QTableWidgetItem(_due_text(card.due)))

    def _toggle_list(self) -> None:
        showing_list = self.stack.currentIndex() == 1
        self.stack.setCurrentIndex(0 if showing_list else 1)
        self.toggle_button.setText("카드 목록" if showing_list else "복습하기")

    def _delete_selected(self) -> None:
        rows = {index.row() for index in self.table.selectedIndexes()}
        for row in rows:
            self.ctx.store.delete_card(self.table.item(row, 0).data(Qt.ItemDataRole.UserRole))
        if rows:
            self.refresh()
            self.on_cards_changed()

    # --- 내보내기 ---------------------------------------------------------------

    def _export(self) -> None:
        cards = self.ctx.store.all_cards()
        if not cards:
            self.summary.setText("내보낼 카드가 아직 없어요.")
            return
        path, _ = QFileDialog.getSaveFileName(
            self, "Anki용 파일 저장", "일본어_복습카드.csv", "CSV (*.csv)"
        )
        if not path:
            return
        reader = self.ctx.engine.reader
        rows = (
            (card.ja, reader.annotate(card.ja).hiragana if reader else "", card.ko, card.note)
            for card in cards
        )
        count = export_anki_csv(path, rows)
        self.summary.setText(f"카드 {count}장을 내보냈어요.")
