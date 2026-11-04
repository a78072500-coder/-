"""색과 글꼴. 자막 창의 검정+노랑을 메인 창에도 이어 쓴다."""

from __future__ import annotations

import sys

BG = "#16171b"
SIDEBAR = "#0f1013"
SURFACE = "#1e2026"
SURFACE_HIGH = "#272a31"
BORDER = "#31343c"
TEXT = "#ecebe6"
MUTED = "#a2a5ad"
FAINT = "#71747d"
ACCENT = "#ffd24d"
ACCENT_HOVER = "#ffdd75"
ACCENT_INK = "#1f1a08"
KOREAN = "#ffe38a"
WARN = "#ffb454"
DANGER = "#ff8a7a"
GOOD = "#7fd8a6"
MINE = "#22303a"

if sys.platform == "darwin":
    KO_FONT = '"Apple SD Gothic Neo"'
    JA_FONT = '"Hiragino Sans", "Apple SD Gothic Neo"'
else:
    KO_FONT = '"Malgun Gothic"'
    # 맑은 고딕으로 일본어를 쓰면 한자가 한국식 모양으로 나와서 일본어 글꼴을 먼저 쓴다
    JA_FONT = '"Yu Gothic UI", "Meiryo", "Malgun Gothic"'

STYLESHEET = f"""
QWidget {{ color: {TEXT}; font-family: {KO_FONT}; font-size: 14px; }}
QMainWindow, QWidget#page, QStackedWidget, QScrollArea, QWidget#scrollBody {{ background: {BG}; }}
QScrollArea {{ border: none; }}
QWidget#sidebar {{ background: {SIDEBAR}; }}
QSplitter::handle {{ background: {BORDER}; }}
QSplitter::handle:horizontal {{ width: 1px; }}
QLabel {{ background: transparent; }}

QLabel[role="brand"] {{ font-size: 16px; font-weight: 700; }}
QLabel[role="title"] {{ font-size: 22px; font-weight: 700; }}
QLabel[role="section"] {{ font-size: 13px; font-weight: 700; color: {MUTED}; }}
QLabel[role="muted"] {{ color: {MUTED}; }}
QLabel[role="faint"] {{ color: {FAINT}; font-size: 12px; }}
QLabel[role="warn"] {{ color: {WARN}; font-size: 12px; font-weight: 600; }}
QLabel[role="error"] {{ color: {DANGER}; }}
QLabel[role="good"] {{ color: {GOOD}; }}
QLabel[role="empty"] {{ color: {FAINT}; font-size: 15px; }}
QLabel[role="ja"] {{ font-family: {JA_FONT}; font-size: 15px; }}
QLabel[role="jaPartial"] {{ font-family: {JA_FONT}; font-size: 15px; color: {FAINT}; font-style: italic; }}
QLabel[role="jaBig"] {{ font-family: {JA_FONT}; font-size: 19px; font-weight: 600; }}
QLabel[role="jaHuge"] {{ font-family: {JA_FONT}; font-size: 28px; font-weight: 600; }}
QLabel[role="reading"] {{ font-family: {JA_FONT}; font-size: 12px; color: {MUTED}; }}
QLabel[role="readingBig"] {{ font-family: {JA_FONT}; font-size: 16px; color: {MUTED}; }}
QLabel[role="hangul"] {{ font-size: 13px; color: {GOOD}; }}
QLabel[role="hangulBig"] {{ font-size: 16px; color: {GOOD}; }}
QLabel[role="ko"] {{ font-size: 16px; font-weight: 600; color: {KOREAN}; }}
QLabel[role="koHuge"] {{ font-size: 26px; font-weight: 700; color: {KOREAN}; }}
QLabel[role="tag"] {{
    font-size: 11px; font-weight: 700; color: {ACCENT};
    background: #3a3218; border-radius: 4px; padding: 1px 6px;
}}
QLabel[role="aside"] {{ font-family: {JA_FONT}; font-size: 13px; color: {FAINT}; }}

QFrame[role="card"] {{ background: {SURFACE}; border: 1px solid {BORDER}; border-radius: 10px; }}
QFrame[role="mine"] {{ background: {MINE}; border: 1px solid #33454f; border-radius: 10px; }}

QPushButton {{
    background: {SURFACE_HIGH}; border: 1px solid {BORDER}; border-radius: 7px;
    padding: 7px 14px; min-height: 18px;
}}
QPushButton:hover {{ background: #32363f; }}
QPushButton:pressed {{ background: #24262c; }}
QPushButton:disabled {{ color: {FAINT}; background: {SURFACE}; }}
QPushButton:focus {{ border-color: {ACCENT}; }}
QPushButton[role="primary"] {{
    background: {ACCENT}; color: {ACCENT_INK}; border: none; font-weight: 700;
}}
QPushButton[role="primary"]:hover {{ background: {ACCENT_HOVER}; }}
QPushButton[role="primary"]:disabled {{ background: #5b5222; color: #a89c63; }}
QPushButton[role="stop"] {{
    background: #4a2622; color: #ffc9c0; border: 1px solid #6e3a33; font-weight: 700;
}}
QPushButton[role="stop"]:hover {{ background: #5a2e29; }}
QPushButton[role="danger"] {{ color: {DANGER}; }}
QPushButton[role="link"] {{
    background: transparent; border: none; color: {MUTED}; padding: 2px 6px; font-size: 12px;
}}
QPushButton[role="link"]:hover {{ color: {ACCENT}; }}
QPushButton[role="link"]:disabled {{ color: {GOOD}; background: transparent; }}
QPushButton[role="nav"] {{
    background: transparent; border: none; border-radius: 8px; text-align: left;
    padding: 10px 14px; font-size: 15px; color: {MUTED};
}}
QPushButton[role="nav"]:hover {{ background: #1b1d22; color: {TEXT}; }}
QPushButton[role="nav"]:checked {{ background: {SURFACE_HIGH}; color: {ACCENT}; font-weight: 700; }}
QPushButton[role="phrase"] {{ text-align: left; padding: 8px 12px; font-family: {JA_FONT}; }}

QLineEdit, QPlainTextEdit, QComboBox, QSpinBox, QDoubleSpinBox {{
    background: {SURFACE}; border: 1px solid {BORDER}; border-radius: 7px; padding: 6px 8px;
    selection-background-color: {ACCENT}; selection-color: {ACCENT_INK};
}}
QLineEdit:focus, QPlainTextEdit:focus, QComboBox:focus, QSpinBox:focus, QDoubleSpinBox:focus {{
    border-color: {ACCENT};
}}
QComboBox QAbstractItemView {{
    background: {SURFACE_HIGH}; border: 1px solid {BORDER};
    selection-background-color: {ACCENT}; selection-color: {ACCENT_INK};
}}
QCheckBox {{ spacing: 8px; }}
QCheckBox::indicator {{
    width: 16px; height: 16px; border: 1px solid {FAINT}; border-radius: 4px; background: {SURFACE};
}}
QCheckBox::indicator:checked {{ background: {ACCENT}; border-color: {ACCENT}; }}

QListWidget {{
    background: {SURFACE}; border: 1px solid {BORDER}; border-radius: 10px; padding: 4px;
    outline: none;
}}
QListWidget::item {{ padding: 9px 10px; border-radius: 6px; }}
QListWidget::item:hover {{ background: {SURFACE_HIGH}; }}
QListWidget::item:selected {{ background: #3a3218; color: {ACCENT}; }}

QTableWidget {{
    background: {SURFACE}; border: 1px solid {BORDER}; border-radius: 10px;
    gridline-color: {BORDER}; outline: none;
}}
QTableWidget::item {{ padding: 6px; }}
QTableWidget::item:selected {{ background: #3a3218; color: {ACCENT}; }}
QHeaderView::section {{
    background: {SURFACE_HIGH}; color: {MUTED}; border: none; border-bottom: 1px solid {BORDER};
    padding: 7px; font-weight: 600;
}}
QTableCornerButton::section {{ background: {SURFACE_HIGH}; border: none; }}

QTabWidget::pane {{ border: none; }}
QTabBar::tab {{
    background: transparent; color: {MUTED}; padding: 8px 16px;
    border-bottom: 2px solid transparent;
}}
QTabBar::tab:selected {{ color: {ACCENT}; border-bottom-color: {ACCENT}; font-weight: 700; }}
QTabBar::tab:hover {{ color: {TEXT}; }}

QSlider::groove:horizontal {{ height: 4px; background: {BORDER}; border-radius: 2px; }}
QSlider::sub-page:horizontal {{ background: #8a7430; border-radius: 2px; }}
QSlider::handle:horizontal {{
    width: 14px; height: 14px; margin: -5px 0; border-radius: 7px; background: {ACCENT};
}}
QSlider::handle:horizontal:hover {{ background: {ACCENT_HOVER}; }}

QProgressBar {{ background: {SURFACE_HIGH}; border: none; border-radius: 3px; }}
QProgressBar::chunk {{ background: {GOOD}; border-radius: 3px; }}

QScrollBar:vertical {{ background: transparent; width: 10px; margin: 0; }}
QScrollBar::handle:vertical {{ background: {BORDER}; border-radius: 5px; min-height: 28px; }}
QScrollBar::handle:vertical:hover {{ background: {FAINT}; }}
QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {{ height: 0; }}
QScrollBar::add-page:vertical, QScrollBar::sub-page:vertical {{ background: transparent; }}
QScrollBar:horizontal {{ height: 0; }}

QMenu {{ background: {SURFACE_HIGH}; border: 1px solid {BORDER}; padding: 4px; }}
QMenu::item {{ padding: 6px 22px; border-radius: 4px; }}
QMenu::item:selected {{ background: {ACCENT}; color: {ACCENT_INK}; }}
QMenu::separator {{ height: 1px; background: {BORDER}; margin: 4px 8px; }}
QToolTip {{ background: {SURFACE_HIGH}; color: {TEXT}; border: 1px solid {BORDER}; }}
QLabel#overlayLabel {{ background: transparent; }}
QMessageBox, QDialog {{ background: {SURFACE}; }}
"""
