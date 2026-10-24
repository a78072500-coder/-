"""TOML 설정 파일 로드. 파일에 없는 항목은 기본값을 쓴다."""

from __future__ import annotations

import tomllib
from dataclasses import dataclass, field, fields, is_dataclass
from pathlib import Path


@dataclass
class AudioConfig:
    # 루프백으로 캡처할 출력 장치 이름의 일부 (빈 문자열이면 기본 출력 장치)
    # 예: 블루스택 소리를 헤드셋으로 보내고 "Headset" 을 적으면 그 소리만 캡처된다.
    device: str = ""
    # macOS: 이 이름이 들어간 앱 소리만 듣는다 (예: "maum", "BlueStacks"). 비우면 모든 소리.
    app: str = ""


@dataclass
class SegmenterConfig:
    vad_threshold: float = 0.5        # 이 확률 이상이면 말하는 중으로 본다
    silence_end_sec: float = 0.6      # 이만큼 조용하면 문장 끝으로 본다
    max_utterance_sec: float = 12.0   # 너무 긴 발화는 강제로 자른다
    min_utterance_sec: float = 0.4    # 이보다 짧은 소리는 무시한다
    partial_interval_sec: float = 1.5  # 말하는 중 임시 자막을 갱신하는 간격 (0 이면 끔)


@dataclass
class WhisperConfig:
    model: str = "large-v3-turbo"
    device: str = "cuda"              # "cuda" 또는 "cpu"
    compute_type: str = "float16"     # CPU 라면 "int8"
    beam_size: int = 5


@dataclass
class TranslatorConfig:
    engine: str = "deepl"             # "deepl" | "claude" | "none"
    deepl_api_key: str = ""
    claude_model: str = "claude-opus-5-5"
    context_lines: int = 4            # Claude 에 넘길 최근 대화 줄 수


@dataclass
class OverlayConfig:
    font_family: str = "Malgun Gothic"
    font_size: int = 20
    history_lines: int = 3            # 자막 창에 남길 이전 문장 수
    opacity: float = 0.85
    width: int = 1000
    x: int = -1                       # -1 이면 화면 아래 가운데
    y: int = -1
    show_japanese: bool = True


@dataclass
class LogConfig:
    enabled: bool = True
    directory: str = "logs"           # 대화 기록 txt 저장 폴더


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


def load_config(path: str | Path | None) -> Config:
    config = Config()
    if path is None:
        return config
    path = Path(path)
    if not path.exists():
        return config
    with path.open("rb") as f:
        _apply(config, tomllib.load(f))
    return config
