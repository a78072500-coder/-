"""통화 기록과 복습 카드를 SQLite 파일 하나에 저장한다."""

from __future__ import annotations

import csv
import json
import sqlite3
import threading
from dataclasses import dataclass
from datetime import date, datetime, timedelta
from pathlib import Path
from statistics import median, quantiles
from typing import Iterable

from .models import THEM, Line, Timing

SCHEMA = """
CREATE TABLE IF NOT EXISTS sessions (
    id INTEGER PRIMARY KEY,
    started_at TEXT NOT NULL,
    ended_at TEXT,
    review_json TEXT
);
CREATE TABLE IF NOT EXISTS lines (
    id INTEGER PRIMARY KEY,
    session_id INTEGER NOT NULL REFERENCES sessions(id) ON DELETE CASCADE,
    at TEXT NOT NULL,
    speaker TEXT NOT NULL,
    ja TEXT NOT NULL,
    ko TEXT NOT NULL DEFAULT '',
    reading TEXT NOT NULL DEFAULT '',
    low_confidence INTEGER NOT NULL DEFAULT 0,
    avg_logprob REAL,
    backchannel INTEGER NOT NULL DEFAULT 0,
    language TEXT NOT NULL DEFAULT 'ja',
    audio_sec REAL,
    silence_wait REAL,
    asr_wait REAL,
    asr REAL,
    translate_wait REAL,
    translate REAL
);
CREATE INDEX IF NOT EXISTS lines_session ON lines(session_id);
CREATE TABLE IF NOT EXISTS cards (
    id INTEGER PRIMARY KEY,
    ja TEXT NOT NULL UNIQUE,
    ko TEXT NOT NULL DEFAULT '',
    note TEXT NOT NULL DEFAULT '',
    session_id INTEGER REFERENCES sessions(id) ON DELETE SET NULL,
    created_at TEXT NOT NULL,
    box INTEGER NOT NULL DEFAULT 0,
    due TEXT NOT NULL
);
"""

# 복습 간격 (일). 알았다고 할 때마다 한 칸씩 늘어나고, 몰랐다고 하면 처음으로 돌아간다.
INTERVALS = [1, 3, 7, 14, 30, 60]

# 지연 시간을 보여줄 구간: (표시 이름, SQL 식)
LATENCY_STAGES = [
    ("문장 끝 판단", "silence_wait"),
    ("인식 대기", "asr_wait"),
    ("음성인식", "asr"),
    ("번역", "translate"),
    ("자막까지", "silence_wait + asr_wait + asr"),
    ("한국어까지", "silence_wait + asr_wait + asr + translate_wait + translate"),
]


@dataclass
class SessionInfo:
    id: int
    started_at: str
    ended_at: str | None
    line_count: int
    preview: str
    has_review: bool


@dataclass
class Card:
    id: int
    ja: str
    ko: str
    note: str
    box: int
    due: str


@dataclass
class LatencyStat:
    name: str
    typical: float
    slow: float
    count: int


class HistoryStore:
    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.RLock()
        self._db = sqlite3.connect(self.path, check_same_thread=False, timeout=10)
        self._db.row_factory = sqlite3.Row
        with self._lock, self._db:
            self._db.execute("PRAGMA foreign_keys = ON")
            self._db.executescript(SCHEMA)

    def close(self) -> None:
        with self._lock:
            self._db.close()

    # --- 통화 기록 -------------------------------------------------------------

    def create_session(self) -> int:
        now = datetime.now().isoformat(timespec="seconds")
        with self._lock, self._db:
            return self._db.execute("INSERT INTO sessions(started_at) VALUES (?)", (now,)).lastrowid

    def end_session(self, session_id: int) -> None:
        """통화를 마친다. 아무 말도 인식되지 않은 통화는 남기지 않는다."""
        now = datetime.now().isoformat(timespec="seconds")
        with self._lock, self._db:
            self._db.execute("UPDATE sessions SET ended_at = ? WHERE id = ?", (now, session_id))
            self._db.execute(
                "DELETE FROM sessions WHERE id = ? AND NOT EXISTS "
                "(SELECT 1 FROM lines WHERE session_id = sessions.id)",
                (session_id,),
            )

    def add_line(self, session_id: int, line: Line, timing: Timing) -> None:
        now = datetime.now().isoformat(timespec="seconds")
        with self._lock, self._db:
            self._db.execute(
                "INSERT INTO lines(session_id, at, speaker, ja, ko, reading, low_confidence,"
                " avg_logprob, backchannel, language, audio_sec, silence_wait, asr_wait, asr,"
                " translate_wait, translate) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
                (
                    session_id, now, line.speaker, line.ja, line.ko, line.reading,
                    int(line.low_confidence), line.avg_logprob, int(line.backchannel),
                    line.language, timing.audio_sec, timing.silence_wait, timing.asr_wait,
                    timing.asr, timing.translate_wait, timing.translate,
                ),
            )

    def list_sessions(self) -> list[SessionInfo]:
        with self._lock:
            rows = self._db.execute(
                "SELECT s.id, s.started_at, s.ended_at, s.review_json IS NOT NULL AS has_review,"
                " COUNT(l.id) AS line_count,"
                " (SELECT COALESCE(NULLIF(x.ko, ''), x.ja) FROM lines x WHERE x.session_id = s.id"
                "  AND x.speaker = 'them' AND x.backchannel = 0 ORDER BY x.id LIMIT 1) AS preview"
                " FROM sessions s LEFT JOIN lines l ON l.session_id = s.id"
                " GROUP BY s.id ORDER BY s.id DESC"
            ).fetchall()
        return [
            SessionInfo(
                r["id"], r["started_at"], r["ended_at"], r["line_count"],
                r["preview"] or "", bool(r["has_review"]),
            )
            for r in rows
        ]

    def lines(self, session_id: int) -> list[Line]:
        with self._lock:
            rows = self._db.execute(
                "SELECT * FROM lines WHERE session_id = ? ORDER BY id", (session_id,)
            ).fetchall()
        return [
            Line(
                uid=r["id"], ja=r["ja"], speaker=r["speaker"], ko=r["ko"], reading=r["reading"],
                low_confidence=bool(r["low_confidence"]), avg_logprob=r["avg_logprob"],
                backchannel=bool(r["backchannel"]), language=r["language"],
            )
            for r in rows
        ]

    def save_review(self, session_id: int, review: dict) -> None:
        with self._lock, self._db:
            self._db.execute(
                "UPDATE sessions SET review_json = ? WHERE id = ?",
                (json.dumps(review, ensure_ascii=False), session_id),
            )

    def review(self, session_id: int) -> dict | None:
        with self._lock:
            row = self._db.execute(
                "SELECT review_json FROM sessions WHERE id = ?", (session_id,)
            ).fetchone()
        return json.loads(row["review_json"]) if row and row["review_json"] else None

    def delete_session(self, session_id: int) -> None:
        with self._lock, self._db:
            self._db.execute("DELETE FROM sessions WHERE id = ?", (session_id,))

    # --- 복습 카드 -------------------------------------------------------------

    def add_card(self, ja: str, ko: str, note: str = "", session_id: int | None = None) -> bool:
        """카드를 만든다. 같은 문장이 이미 있으면 그대로 두고 False."""
        today = date.today().isoformat()
        now = datetime.now().isoformat(timespec="seconds")
        with self._lock, self._db:
            cursor = self._db.execute(
                "INSERT OR IGNORE INTO cards(ja, ko, note, session_id, created_at, due)"
                " VALUES (?,?,?,?,?,?)",
                (ja.strip(), ko.strip(), note.strip(), session_id, now, today),
            )
            return cursor.rowcount > 0

    @staticmethod
    def _card(row: sqlite3.Row) -> Card:
        return Card(row["id"], row["ja"], row["ko"], row["note"], row["box"], row["due"])

    def due_cards(self, today: date | None = None) -> list[Card]:
        day = (today or date.today()).isoformat()
        with self._lock:
            rows = self._db.execute(
                "SELECT * FROM cards WHERE due <= ? ORDER BY due, box, id", (day,)
            ).fetchall()
        return [self._card(r) for r in rows]

    def all_cards(self) -> list[Card]:
        with self._lock:
            rows = self._db.execute("SELECT * FROM cards ORDER BY id DESC").fetchall()
        return [self._card(r) for r in rows]

    def grade_card(self, card_id: int, known: bool, today: date | None = None) -> None:
        day = today or date.today()
        with self._lock, self._db:
            row = self._db.execute("SELECT box FROM cards WHERE id = ?", (card_id,)).fetchone()
            if row is None:
                return
            box = min(row["box"] + 1, len(INTERVALS) - 1) if known else 0
            due = day + timedelta(days=INTERVALS[row["box"]]) if known else day
            self._db.execute(
                "UPDATE cards SET box = ?, due = ? WHERE id = ?", (box, due.isoformat(), card_id)
            )

    def delete_card(self, card_id: int) -> None:
        with self._lock, self._db:
            self._db.execute("DELETE FROM cards WHERE id = ?", (card_id,))

    # --- 지연 시간 -------------------------------------------------------------

    def latency_stats(self, session_id: int | None = None) -> list[LatencyStat]:
        """구간별 보통(중앙값)과 느릴 때(90%) 시간. 맞장구와 내 말은 뺀다."""
        where = "speaker = ? AND backchannel = 0 AND asr IS NOT NULL"
        params: list = [THEM]
        if session_id is not None:
            where += " AND session_id = ?"
            params.append(session_id)
        stats = []
        with self._lock:
            for name, expression in LATENCY_STAGES:
                rows = self._db.execute(
                    f"SELECT {expression} AS v FROM lines WHERE {where}", params
                ).fetchall()
                values = [r["v"] for r in rows if r["v"] is not None]
                if values:
                    stats.append(LatencyStat(name, median(values), _p90(values), len(values)))
        return stats


def _p90(values: list[float]) -> float:
    return quantiles(values, n=10, method="inclusive")[-1] if len(values) > 1 else values[0]


def export_anki_csv(path: str | Path, cards: Iterable[tuple[str, str, str, str]]) -> int:
    """(일본어, 읽는 법, 뜻, 메모) 를 Anki 에서 가져올 수 있는 CSV 로 쓴다."""
    count = 0
    with Path(path).open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.writer(f)
        for row in cards:
            writer.writerow(row)
            count += 1
    return count


class SessionRecorder:
    """파이프라인의 한 통화 결과를 txt 와 데이터베이스에 함께 남긴다."""

    def __init__(self, text_log, store: HistoryStore, session_id: int):
        self.text_log = text_log
        self.store = store
        self.session_id = session_id

    def record(self, line: Line, timing: Timing) -> None:
        self.store.add_line(self.session_id, line, timing)
        self.text_log.record(line, timing)
