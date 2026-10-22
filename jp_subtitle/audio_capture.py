"""오디오 입력원. 모두 16kHz 모노 float32 블록을 yield 한다."""

from __future__ import annotations

import sys
import threading
import time
from math import gcd
from pathlib import Path
from typing import Iterator

import numpy as np
from scipy.signal import resample_poly

SAMPLE_RATE = 16000
BLOCK_SAMPLES = 512  # 32ms, silero VAD 프레임 크기


def to_mono_16k(data: np.ndarray, rate: int, channels: int) -> np.ndarray:
    """인터리브된 float32 PCM 을 16kHz 모노로 변환한다."""
    if channels > 1:
        data = data.reshape(-1, channels).mean(axis=1)
    if rate != SAMPLE_RATE:
        g = gcd(rate, SAMPLE_RATE)
        data = resample_poly(data, SAMPLE_RATE // g, rate // g)
    return data.astype(np.float32, copy=False)


class _Rechunker:
    """임의 길이 입력을 BLOCK_SAMPLES 단위로 잘라준다."""

    def __init__(self) -> None:
        self._buf = np.zeros(0, dtype=np.float32)

    def feed(self, data: np.ndarray) -> Iterator[np.ndarray]:
        self._buf = np.concatenate([self._buf, data])
        while len(self._buf) >= BLOCK_SAMPLES:
            yield self._buf[:BLOCK_SAMPLES]
            self._buf = self._buf[BLOCK_SAMPLES:]


class FileSource:
    """wav 파일을 실시간 속도로 흘려보낸다 (테스트/시연용)."""

    def __init__(self, path: str | Path, realtime: bool = True) -> None:
        self.path = Path(path)
        self.realtime = realtime
        self._stop = threading.Event()

    def stop(self) -> None:
        self._stop.set()

    def blocks(self) -> Iterator[np.ndarray]:
        from scipy.io import wavfile

        rate, data = wavfile.read(self.path)
        if data.dtype.kind == "i":
            data = data.astype(np.float32) / np.iinfo(data.dtype).max
        elif data.dtype.kind == "u":
            data = (data.astype(np.float32) - 128) / 128
        channels = 1 if data.ndim == 1 else data.shape[1]
        mono = to_mono_16k(data.reshape(-1).astype(np.float32), rate, channels)

        block_sec = BLOCK_SAMPLES / SAMPLE_RATE
        start = time.monotonic()
        for i, block in enumerate(_Rechunker().feed(mono)):
            if self._stop.is_set():
                return
            if self.realtime:
                delay = start + i * block_sec - time.monotonic()
                if delay > 0:
                    time.sleep(delay)
            yield block
        # 마지막 문장이 끝나도록 무음을 조금 덧붙인다
        for _ in range(int(1.5 / block_sec)):
            yield np.zeros(BLOCK_SAMPLES, dtype=np.float32)


def _require_windows() -> None:
    if sys.platform != "win32":
        raise RuntimeError(
            "스피커 루프백 캡처는 Windows 에서만 됩니다. "
            "다른 OS 에서는 --input 으로 wav 파일을 넣어 시험하세요."
        )


def list_loopback_devices() -> list[dict]:
    _require_windows()
    import pyaudiowpatch as pyaudio

    p = pyaudio.PyAudio()
    try:
        return list(p.get_loopback_device_info_generator())
    finally:
        p.terminate()


class LoopbackSource:
    """WASAPI 루프백으로 스피커(출력 장치)에서 나는 소리를 캡처한다."""

    def __init__(self, device_name: str = "") -> None:
        _require_windows()
        self.device_name = device_name
        self._stop = threading.Event()

    def stop(self) -> None:
        self._stop.set()

    def _find_device(self, p) -> dict:
        if not self.device_name:
            return p.get_default_wasapi_loopback()
        wanted = self.device_name.lower()
        for dev in p.get_loopback_device_info_generator():
            if wanted in dev["name"].lower():
                return dev
        names = [d["name"] for d in p.get_loopback_device_info_generator()]
        raise RuntimeError(
            f"'{self.device_name}' 장치를 찾지 못했습니다. 사용 가능한 장치: {names}"
        )

    def blocks(self) -> Iterator[np.ndarray]:
        import pyaudiowpatch as pyaudio

        p = pyaudio.PyAudio()
        dev = self._find_device(p)
        rate = int(dev["defaultSampleRate"])
        channels = int(dev["maxInputChannels"])
        frames = rate // 10  # 100ms 씩 읽는다
        stream = p.open(
            format=pyaudio.paFloat32,
            channels=channels,
            rate=rate,
            input=True,
            input_device_index=dev["index"],
            frames_per_buffer=frames,
        )
        print(f"[캡처] {dev['name']} ({rate}Hz, {channels}ch)")
        rechunker = _Rechunker()
        silence = np.zeros(SAMPLE_RATE // 10, dtype=np.float32)
        last_data = time.monotonic()
        try:
            while not self._stop.is_set():
                if stream.get_read_available() >= frames:
                    raw = stream.read(frames, exception_on_overflow=False)
                    data = np.frombuffer(raw, dtype=np.float32)
                    yield from rechunker.feed(to_mono_16k(data, rate, channels))
                    last_data = time.monotonic()
                    continue
                time.sleep(0.01)
                # 아무 소리도 안 날 때 WASAPI 루프백은 데이터를 보내지 않으므로
                # 무음을 직접 넣어줘야 마지막 문장이 끝난 것으로 처리된다
                if time.monotonic() - last_data >= 0.2:
                    yield from rechunker.feed(silence)
                    last_data += 0.1
        finally:
            stream.stop_stream()
            stream.close()
            p.terminate()
