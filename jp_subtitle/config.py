"""TOML 설정 파일 읽기와 저장. 파일에 없는 항목은 기본값을 쓴다."""

from __future__ import annotations

import json
import os
import re
import tomllib
from dataclasses import dataclass, field, fields, is_dataclass
from pathlib import Path
from typing import Any


@dataclass
class AudioConfig:
    # 루프백으로 캡처할 출력 장치 이름의 일부 (빈 문자열이면 기본 출력 장치)
    # 예: 블루스택 소리를 헤드셋으로 보내고 "Headset" 을 적으면 그 소리만 캡처된다.
    device: str = ""
    # macOS: 이 이름이 들어간 앱 소리만 듣는다 (예: "maum", "BlueStacks"). 비우면 모든 소리.
    app: str = ""
    # 내 목소리도 인식해서 대화 기록과 번역 문맥에 넣는다 (Windows, 이어폰 필수)
    mic_enabled: bool = False
    mic_device: str = ""              # 마이크 이름의 일부 (빈 문자열이면 기본 마이크)


@dataclass
class SegmenterConfig:
    vad_threshold: float = 0.5        # 이 확률 이상이면 말하는 중으로 본다
    silence_end_sec: float = 0.35     # 이만큼 조용하면 문장 끝으로 본다
    max_utterance_sec: float = 6.0    # 너무 긴 발화는 숨 고른 곳에서 자른다
    min_utterance_sec: float = 0.4    # 이보다 짧은 소리는 무시한다
    partial_interval_sec: float = 1.0  # 말하는 중 임시 자막을 갱신하는 간격 (0 이면 끔)
    sentence_cut: bool = True         # 임시 자막에서 문장 끝이 보이면 쉬기를 기다리지 않고 자른다


@dataclass(frozen=True)
class SentencePreset:
    """문장 길이 단계. 화면의 슬라이더와 설정에서 고른다."""

    key: str
    label: str
    silence_end_sec: float
    max_utterance_sec: float
    partial_interval_sec: float
    sentence_cut: bool

    def values(self) -> dict[str, Any]:
        return {
            "silence_end_sec": self.silence_end_sec,
            "max_utterance_sec": self.max_utterance_sec,
            "partial_interval_sec": self.partial_interval_sec,
            "sentence_cut": self.sentence_cut,
        }


SENTENCE_PRESETS = (
    SentencePreset("very_short", "아주 짧게", 0.25, 4.0, 0.7, True),
    SentencePreset("short", "짧게", 0.35, 6.0, 1.0, True),
    SentencePreset("normal", "보통", 0.5, 9.0, 1.2, True),
    SentencePreset("long", "길게", 0.8, 14.0, 1.5, False),
)


def find_preset(cfg: SegmenterConfig) -> SentencePreset | None:
    """지금 값과 정확히 같은 단계. 사용자가 값을 직접 바꿨으면 None."""
    for preset in SENTENCE_PRESETS:
        if all(getattr(cfg, name) == value for name, value in preset.values().items()):
            return preset
    return None


def nearest_preset_index(cfg: SegmenterConfig) -> int:
    return min(
        range(len(SENTENCE_PRESETS)),
        key=lambda i: abs(SENTENCE_PRESETS[i].max_utterance_sec - cfg.max_utterance_sec),
    )


@dataclass
class WhisperConfig:
    model: str = "large-v3-turbo"
    device: str = "cuda"              # "cuda" 또는 "cpu"
    compute_type: str = "float16"     # CPU 라면 "int8"
    beam_size: int = 5
    vocabulary: list[str] = field(default_factory=list)  # 자주 나오는 이름·단어
    low_confidence_logprob: float = -0.6  # 이보다 낮으면 "잘못 들었을 수 있음" 표시


@dataclass
class TranslatorConfig:
    engine: str = "deepl"             # 자막 번역: "deepl" | "claude" | "gemini" | "none"
    ai_engine: str = "auto"           # 답변·복습 AI: "auto"(키가 있는 쪽) | "claude" | "gemini"
    deepl_api_key: str = ""
    anthropic_api_key: str = ""       # 비우면 ANTHROPIC_API_KEY 환경변수를 쓴다
    gemini_api_key: str = ""          # 비우면 GEMINI_API_KEY 환경변수를 쓴다
    claude_model: str = "claude-opus-5-5"
    gemini_model: str = "gemini-3.5-flash-lite"
    context_lines: int = 4            # 번역할 때 참고할 최근 대화 줄 수


@dataclass
class OverlayConfig:
    enabled: bool = True              # 통화 중 화면 위에 자막 창을 띄운다
    font_family: str = "Malgun Gothic"
    font_size: int = 20
    history_lines: int = 3            # 자막 창에 남길 이전 문장 수
    opacity: float = 0.85
    width: int = 1000
    x: int = -1                       # -1 이면 화면 아래 가운데
    y: int = -1
    show_japanese: bool = True
    show_reading: bool = True         # 한자가 있는 문장에 읽는 법을 함께 표시


@dataclass
class LogConfig:
    enabled: bool = True
    directory: str = "logs"           # 대화 기록 저장 폴더


@dataclass
class Config:
    audio: AudioConfig = field(default_factory=AudioConfig)
    segmenter: SegmenterConfig = field(default_factory=SegmenterConfig)
    whisper: WhisperConfig = field(default_factory=WhisperConfig)
    translator: TranslatorConfig = field(default_factory=TranslatorConfig)
    overlay: OverlayConfig = field(default_factory=OverlayConfig)
    log: LogConfig = field(default_factory=LogConfig)


def _apply(obj, data: dict, path: str = "") -> None:
    known = {f.name: f for f in fields(obj)}
    for key, value in data.items():
        if key not in known:
            raise ValueError(f"알 수 없는 설정 항목: {path}{key}")
        current = getattr(obj, key)
        if is_dataclass(current):
            if not isinstance(value, dict):
                raise ValueError(f"[{path}{key}] 는 섹션이어야 합니다")
            _apply(current, value, f"{path}{key}.")
        else:
            if isinstance(current, float) and isinstance(value, int):
                value = float(value)
            if type(value) is not type(current):
                raise ValueError(
                    f"{path}{key} 의 형식이 잘못됐습니다: "
                    f"{type(current).__name__} 이어야 하는데 {value!r}"
                )
            setattr(obj, key, value)


SMART_QUOTES = str.maketrans({"“": '"', "”": '"', "‘": "'", "’": "'"})

# 문장 길이 조절이 생기기 전에 기본 설정 파일에 적혀 있던 값들
_OLD_SENTENCE_DEFAULTS = {
    "silence_end_sec": 0.6,
    "max_utterance_sec": 12.0,
    "partial_interval_sec": 1.5,
}


def _migrate(data: dict) -> dict:
    """이전 기본값이 그대로면 사용자가 고른 값이 아니므로 새 기본값을 쓰게 한다."""
    segmenter = data.get("segmenter")
    if (
        isinstance(segmenter, dict)
        and "sentence_cut" not in segmenter
        and all(segmenter.get(key) == value for key, value in _OLD_SENTENCE_DEFAULTS.items())
    ):
        for key in _OLD_SENTENCE_DEFAULTS:
            del segmenter[key]
    return data


def load_config(path: str | Path | None) -> Config:
    config = Config()
    if path is None:
        return config
    path = Path(path)
    if not path.exists():
        return config
    text = path.read_text(encoding="utf-8-sig")
    # 맥 텍스트 편집기·워드 등이 바꿔 넣는 둥근 따옴표를 일반 따옴표로 되돌린다
    text = text.translate(SMART_QUOTES)
    _apply(config, _migrate(tomllib.loads(text)))
    return config


def _toml_value(value: Any) -> str:
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, str):
        return json.dumps(value, ensure_ascii=False)
    if isinstance(value, (int, float)):
        return repr(value)
    if isinstance(value, list):
        return "[" + ", ".join(_toml_value(v) for v in value) + "]"
    raise TypeError(f"설정에 저장할 수 없는 값: {value!r}")


_SECTION = re.compile(r"^\s*\[([^\[\]]+)\]\s*(#.*)?$")
_VALUE = r"""("(?:[^"\\]|\\.)*"|'[^']*'|\[.*?\]|[^#\s]+)"""


def _replace_value(lines: list[str], start: int, end: int, key: str, value: Any) -> bool:
    """섹션 안에서 항목을 찾아 값만 바꾼다. 줄 끝 주석은 그대로 둔다."""
    pattern = re.compile(rf"^(\s*{re.escape(key)}\s*=\s*){_VALUE}(\s*#.*)?\s*$")
    for i in range(start, end):
        if re.match(rf"^\s*{re.escape(key)}\s*=", lines[i]) is None:
            continue
        current = lines[i].split("=", 1)[1].strip()
        if current.startswith("[") and "]" not in current:
            # 여러 줄에 걸친 배열이면 닫는 괄호까지 지우고 한 줄로 쓴다
            last = i
            while last < end - 1 and "]" not in lines[last]:
                last += 1
            lines[i : last + 1] = [f"{key} = {_toml_value(value)}"]
            return True
        match = pattern.match(lines[i])
        if match is None:
            lines[i] = f"{key} = {_toml_value(value)}"
        else:
            lines[i] = f"{match.group(1)}{_toml_value(value)}{match.group(3) or ''}"
        return True
    return False


def save_config(path: str | Path, values: dict[str, dict[str, Any]]) -> None:
    """바뀐 값만 파일에 반영한다. 사용자의 주석과 순서는 그대로 둔다."""
    path = Path(path)
    text = path.read_text(encoding="utf-8-sig") if path.exists() else ""
    lines = text.translate(SMART_QUOTES).splitlines()
    for section, items in values.items():
        start = next(
            (i for i, line in enumerate(lines)
             if (m := _SECTION.match(line)) and m.group(1).strip() == section),
            None,
        )
        if start is None:
            lines += ["", f"[{section}]"]
            start = len(lines) - 1
        for key, value in items.items():
            end = next(
                (i for i in range(start + 1, len(lines)) if _SECTION.match(lines[i])),
                len(lines),
            )
            if not _replace_value(lines, start + 1, end, key, value):
                lines.insert(start + 1, f"{key} = {_toml_value(value)}")
    new_text = "\n".join(lines) + "\n"
    _apply(Config(), tomllib.loads(new_text))  # 잘못된 파일을 쓰기 전에 걸러낸다
    temp = path.with_name(path.name + ".tmp")
    temp.write_text(new_text, encoding="utf-8")
    os.replace(temp, path)
