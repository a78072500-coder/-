"""faster-whisper 로 일본어 음성을 텍스트로 바꾼다."""

from __future__ import annotations

import os
import platform
import re
import sys
from dataclasses import replace
from pathlib import Path

import numpy as np

from .config import WhisperConfig

# 무음/잡음에서 Whisper 가 자주 지어내는 문구 (유튜브 자막 학습의 흔적)
HALLUCINATIONS = {
    "ご視聴ありがとうございました",
    "ご視聴ありがとうございます",
    "チャンネル登録よろしくお願いします",
    "チャンネル登録お願いします",
    "最後までご視聴いただきありがとうございました",
    "字幕作成者",
    "字幕",
}
_PUNCT = re.compile(r"[\s。、！？!?．.…・「」『』()（）]+")


def _normalize(text: str) -> str:
    return _PUNCT.sub("", text)


def is_hallucination(text: str) -> bool:
    norm = _normalize(text)
    if not norm:
        return True
    if norm in HALLUCINATIONS:
        return True
    if any(norm.startswith(h) for h in ("字幕", "ご視聴")):
        return True
    # 같은 글자/짧은 패턴이 계속 반복되는 경우 (예: "あああああああ")
    for n in (1, 2, 3):
        unit = norm[:n]
        if len(norm) >= n * 6 and unit * (len(norm) // n) == norm[: n * (len(norm) // n)]:
            return True
    return False


def _add_nvidia_dll_dirs() -> None:
    """pip 로 설치한 nvidia-cublas/cudnn 의 DLL 을 Windows 가 찾을 수 있게 한다."""
    if sys.platform != "win32":
        return
    import site

    roots = [Path(p) for p in site.getsitepackages() + [site.getusersitepackages()]]
    if getattr(sys, "frozen", False):
        roots.append(Path(sys._MEIPASS))  # PyInstaller exe 안에 같이 묶인 DLL
    for root in roots:
        for bin_dir in root.glob("nvidia/*/bin"):
            os.add_dll_directory(str(bin_dir))
            os.environ["PATH"] = f"{bin_dir}{os.pathsep}{os.environ['PATH']}"


CPU_FALLBACK_MODEL = "small"  # GPU 를 못 쓸 때 실시간으로 돌릴 수 있는 크기


class Transcriber:
    def __init__(self, cfg: WhisperConfig):
        self.cfg = cfg
        self.notice = ""  # GPU 대신 CPU 로 바꿨을 때 사용자에게 보여줄 안내
        try:
            self._load(cfg.model, cfg.device, cfg.compute_type)
        except Exception as e:
            if cfg.device == "cpu":
                raise
            print(f"[인식] GPU 로 Whisper 를 실행하지 못했습니다: {e}")
            self.notice = (
                "그래픽카드(CUDA)를 쓰지 못해 CPU 로 실행 중입니다. "
                "자막이 느리거나 덜 정확할 수 있습니다 (README 문제 해결 참고)."
            )
            self._load(CPU_FALLBACK_MODEL, "cpu", "int8")

    def _load(self, model: str, device: str, compute_type: str) -> None:
        if device == "cuda":
            _add_nvidia_dll_dirs()
        from faster_whisper import WhisperModel

        print(f"[인식] Whisper 모델 로딩: {model} ({device}, {compute_type})")
        self.model = WhisperModel(model, device=device, compute_type=compute_type)
        # 첫 호출이 느리지 않게, 그리고 CUDA 라이브러리 문제를 여기서 드러내도록 한 번 돌려둔다
        self.transcribe(np.zeros(16000, dtype=np.float32), beam_size=1)

    def transcribe(self, audio: np.ndarray, beam_size: int | None = None) -> str:
        segments, _ = self.model.transcribe(
            audio,
            language="ja",
            beam_size=beam_size or self.cfg.beam_size,
            condition_on_previous_text=False,
            vad_filter=False,  # 이미 Segmenter 에서 잘랐다
            no_speech_threshold=0.6,
            log_prob_threshold=-1.0,
        )
        texts = [s.text.strip() for s in segments if s.no_speech_prob < 0.8]
        text = "".join(texts).strip()
        return "" if is_hallucination(text) else text


# Apple Silicon 맥: faster-whisper 는 맥 GPU 를 못 쓰므로 MLX 로 변환된 모델을 쓴다
MLX_MODELS = {
    "large-v3-turbo": "mlx-community/whisper-large-v3-turbo",
    "large-v3": "mlx-community/whisper-large-v3-mlx",
    "medium": "mlx-community/whisper-medium-mlx",
    "small": "mlx-community/whisper-small-mlx",
    "base": "mlx-community/whisper-base-mlx",
    "tiny": "mlx-community/whisper-tiny-mlx",
}


def mlx_repo(model: str) -> str:
    return model if "/" in model else MLX_MODELS.get(model, f"mlx-community/whisper-{model}-mlx")


class MLXTranscriber:
    """Apple GPU(Metal) 로 Whisper 를 돌린다. 빔 서치가 없어 항상 greedy 디코딩."""

    def __init__(self, cfg: WhisperConfig):
        import mlx_whisper

        self.cfg = cfg
        self.notice = ""
        self._mlx_whisper = mlx_whisper
        self.repo = mlx_repo(cfg.model)
        print(f"[인식] Whisper 모델 로딩: {self.repo} (mlx)")
        self.transcribe(np.zeros(16000, dtype=np.float32))

    def transcribe(self, audio: np.ndarray, beam_size: int | None = None) -> str:
        result = self._mlx_whisper.transcribe(
            audio,
            path_or_hf_repo=self.repo,
            language="ja",
            temperature=0.0,
            condition_on_previous_text=False,
            no_speech_threshold=0.6,
            logprob_threshold=-1.0,
            verbose=None,
        )
        texts = [s["text"].strip() for s in result["segments"] if s.get("no_speech_prob", 0) < 0.8]
        text = "".join(texts).strip()
        return "" if is_hallucination(text) else text


def create_transcriber(cfg: WhisperConfig):
    """플랫폼에 맞는 음성인식기를 고른다. Windows/Linux 는 faster-whisper(CUDA→CPU)."""
    if sys.platform != "darwin" or cfg.device == "cpu":
        return Transcriber(cfg)
    if platform.machine() == "arm64":
        try:
            return MLXTranscriber(cfg)
        except Exception as e:
            print(f"[인식] Apple GPU(MLX) 로 Whisper 를 실행하지 못했습니다: {e}")
            notice = "Apple GPU(MLX)를 쓰지 못해 CPU 로 실행 중입니다. 자막이 느릴 수 있습니다."
    else:
        notice = "Intel 맥은 그래픽 가속이 안 돼 CPU 로 실행 중입니다. 자막이 느릴 수 있습니다."
    t = Transcriber(replace(cfg, model=CPU_FALLBACK_MODEL, device="cpu", compute_type="int8"))
    t.notice = notice
    return t
