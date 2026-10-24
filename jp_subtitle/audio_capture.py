"""오디오 입력원. 모두 16kHz 모노 float32 블록을 yield 한다."""

from __future__ import annotations

import queue
import shutil
import subprocess
import sys
import threading
import time
from math import gcd
from pathlib import Path
from typing import Iterator

import numpy as np
from scipy.signal import firwin, lfilter, lfilter_zi, resample_poly

SAMPLE_RATE = 16000
BLOCK_SAMPLES = 512  # 32ms, silero VAD 프레임 크기


def to_mono_16k(data: np.ndarray, rate: int, channels: int) -> np.ndarray:
    """인터리브된 float32 PCM 을 16kHz 모노로 변환한다 (한 번에 전체를 변환할 때)."""
    if channels > 1:
        data = data.reshape(-1, channels).mean(axis=1)
    if rate != SAMPLE_RATE:
        g = gcd(rate, SAMPLE_RATE)
        data = resample_poly(data, SAMPLE_RATE // g, rate // g)
    return data.astype(np.float32, copy=False)


class StreamResampler:
    """작은 조각으로 계속 들어오는 소리를 끊김 없이 16kHz 모노로 바꾼다.

    조각마다 따로 리샘플하면 경계마다 잡음이 생기므로, 저역통과 필터 상태와
    샘플 위치를 다음 조각으로 이어간다.
    """

    def __init__(self, rate: int, channels: int):
        self.channels = channels
        self.step = rate / SAMPLE_RATE  # 출력 한 샘플마다 입력에서 이동할 거리
        self._next = 0.0    # 다음 출력 샘플의 입력 위치 (처음부터 센 절대 위치)
        self._offset = 0    # 이번 조각 첫 샘플의 절대 위치
        self._last = 0.0    # 앞 조각의 마지막 샘플 (경계에서 보간용)
        if rate > SAMPLE_RATE:
            self._taps = firwin(63, 7600, fs=rate)  # 8kHz 위 성분을 걸러 앨리어싱 방지
            self._zi = lfilter_zi(self._taps, 1.0) * 0.0
        else:
            self._taps = None

    def process(self, data: np.ndarray) -> np.ndarray:
        x = data.astype(np.float64)
        if self.channels > 1:
            x = x.reshape(-1, self.channels).mean(axis=1)
        if not len(x):
            return np.zeros(0, dtype=np.float32)
        if self._taps is not None:
            x, self._zi = lfilter(self._taps, 1.0, x, zi=self._zi)
        # xx[i] 의 절대 위치는 offset - 1 + i
        xx = np.concatenate([[self._last], x])
        end = self._offset + len(x) - 1  # 이번 조각의 마지막 절대 위치
        positions = np.arange(self._next, end, self.step)
        out = np.interp(positions - (self._offset - 1), np.arange(len(xx)), xx)
        if len(positions):
            self._next = positions[-1] + self.step
        self._offset += len(x)
        self._last = x[-1]
        return out.astype(np.float32)


class _Rechunker:
    """임의 길이 입력을 BLOCK_SAMPLES 단위로 잘라준다."""

    def __init__(self) -> None:
        self._buf = np.zeros(0, dtype=np.float32)

    def feed(self, data: np.ndarray) -> Iterator[np.ndarray]:
        self._buf = np.concatenate([self._buf, data])
        while len(self._buf) >= BLOCK_SAMPLES:
            yield self._buf[:BLOCK_SAMPLES]
            self._buf = self._buf[BLOCK_SAMPLES:]


class _LevelMeter:
    """10초마다 들어온 소리 크기를 콘솔에 보여준다 (소리가 안 잡힐 때 원인 확인용)."""

    INTERVAL = 10.0

    def __init__(self) -> None:
        self._peak = 0.0
        self._samples = 0

    def add(self, audio: np.ndarray) -> None:
        if len(audio):
            self._peak = max(self._peak, float(np.max(np.abs(audio))))
        self._samples += len(audio)
        if self._samples >= self.INTERVAL * SAMPLE_RATE:
            if self._peak < 0.003:
                print("[캡처] 최근 10초 동안 소리가 들어오지 않았습니다 (재생 장치 확인)")
            else:
                print(f"[캡처] 최근 10초 최대 음량 {min(100, self._peak * 100):.0f}%")
            self._peak = 0.0
            self._samples = 0


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
            "macOS 는 MacAudioSource 를, 그 밖의 OS 는 --input 으로 wav 파일을 쓰세요."
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
        stream = p.open(
            format=pyaudio.paFloat32,
            channels=channels,
            rate=rate,
            input=True,
            input_device_index=dev["index"],
            frames_per_buffer=rate // 50,  # 20ms
        )
        print(f"[캡처] {dev['name']} ({rate}Hz, {channels}ch)")
        resampler = StreamResampler(rate, channels)
        rechunker = _Rechunker()
        meter = _LevelMeter()
        last_data = time.monotonic()
        try:
            while not self._stop.is_set():
                available = stream.get_read_available()
                now = time.monotonic()
                if available > 0:
                    # 쌓인 만큼 전부 읽는다 (덜 읽으면 버퍼가 넘쳐 소리가 잘려나간다)
                    raw = stream.read(available, exception_on_overflow=False)
                    audio = resampler.process(np.frombuffer(raw, dtype=np.float32))
                    meter.add(audio)
                    yield from rechunker.feed(audio)
                    last_data = now
                    continue
                time.sleep(0.005)
                # 아무 소리도 안 날 때 WASAPI 루프백은 데이터를 보내지 않으므로
                # 그동안의 무음을 직접 넣어줘야 마지막 문장이 끝난 것으로 처리된다
                gap = now - last_data
                if gap >= 0.3:
                    silence = np.zeros(int(gap * SAMPLE_RATE), dtype=np.float32)
                    meter.add(silence)
                    yield from rechunker.feed(silence)
                    last_data = now
        finally:
            stream.stop_stream()
            stream.close()
            p.terminate()


MAC_PERMISSION_HELP = (
    "시스템 오디오를 들으려면 권한이 필요합니다: 시스템 설정 → 개인정보 보호 및 보안 → "
    "'화면 및 시스템 오디오 녹음' 에서 jp_subtitle 을 켜고 프로그램을 다시 실행하세요."
)
_REPO_ROOT = Path(__file__).resolve().parent.parent


def find_mac_helper() -> Path:
    """ScreenCaptureKit 도우미(sck_audio) 위치. 개발 중이면 없을 때 swiftc 로 빌드한다."""
    if getattr(sys, "frozen", False):
        return Path(sys._MEIPASS) / "sck_audio"
    built = _REPO_ROOT / "build" / "sck_audio"
    source = _REPO_ROOT / "mac" / "sck_audio.swift"
    if not built.exists() or built.stat().st_mtime < source.stat().st_mtime:
        if shutil.which("swiftc") is None:
            raise RuntimeError("Xcode 명령행 도구가 필요합니다: 터미널에서 xcode-select --install")
        built.parent.mkdir(exist_ok=True)
        print("[캡처] 오디오 도우미 빌드 중 (처음 한 번)…")
        subprocess.run(["swiftc", "-O", str(source), "-o", str(built)], check=True)
    return built


def list_mac_apps() -> list[str]:
    out = subprocess.run([str(find_mac_helper()), "--list"], capture_output=True, text=True, timeout=30)
    if out.returncode != 0:
        msg = out.stderr.strip()
        raise RuntimeError(MAC_PERMISSION_HELP if msg == "PERMISSION" else msg)
    return [line.split("\t")[0] for line in out.stdout.splitlines() if line.strip()]


class MacAudioSource:
    """macOS ScreenCaptureKit 도우미를 실행해 시스템(또는 특정 앱) 소리를 받는다."""

    def __init__(self, app_name: str = "", helper: Path | list[str] | None = None) -> None:
        self.app_name = app_name
        self._helper = helper  # 테스트에서 가짜 도우미를 넣을 수 있게
        self._stop = threading.Event()
        self._proc: subprocess.Popen | None = None

    def stop(self) -> None:
        self._stop.set()
        if self._proc and self._proc.poll() is None:
            self._proc.terminate()

    def blocks(self) -> Iterator[np.ndarray]:
        helper = self._helper or find_mac_helper()
        cmd = list(helper) if isinstance(helper, list) else [str(helper)]
        if self.app_name:
            cmd += ["--app", self.app_name]
        proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        self._proc = proc
        chunks: queue.Queue[bytes | None] = queue.Queue()
        state: dict = {"rate": None, "error": None, "format": threading.Event()}

        def read_stdout() -> None:
            while data := proc.stdout.read1(65536):
                chunks.put(data)
            chunks.put(None)

        def read_stderr() -> None:
            for raw_line in proc.stderr:
                line = raw_line.decode("utf-8", "replace").strip()
                if line.startswith("FORMAT "):
                    state["rate"] = int(float(line.split()[1]))
                    state["format"].set()
                elif line == "READY":
                    target = f"앱: {self.app_name}" if self.app_name else "모든 소리"
                    print(f"[캡처] macOS 시스템 오디오 ({target})")
                elif line:
                    state["error"] = MAC_PERMISSION_HELP if line == "PERMISSION" else line
                    print(f"[캡처] {state['error']}")

        for target in (read_stdout, read_stderr):
            threading.Thread(target=target, daemon=True).start()

        rechunker = _Rechunker()
        meter = _LevelMeter()
        resampler: StreamResampler | None = None
        leftover = b""
        last_data = time.monotonic()
        try:
            while not self._stop.is_set():
                try:
                    data = chunks.get(timeout=0.05)
                except queue.Empty:
                    # 소리가 안 날 때는 데이터가 오지 않으므로 그동안의 무음을 넣는다
                    gap = time.monotonic() - last_data
                    if gap >= 0.3:
                        silence = np.zeros(int(gap * SAMPLE_RATE), dtype=np.float32)
                        meter.add(silence)
                        yield from rechunker.feed(silence)
                        last_data = time.monotonic()
                    continue
                if data is None:
                    if self._stop.is_set():
                        return
                    proc.wait(timeout=5)
                    raise RuntimeError(state["error"] or f"오디오 도우미가 종료되었습니다 (코드 {proc.returncode})")
                if resampler is None:
                    # 도우미는 FORMAT 을 stderr 에 먼저 쓰지만 두 파이프를 읽는 순서는 보장되지 않는다
                    state["format"].wait(timeout=2)
                    resampler = StreamResampler(state["rate"] or 48000, 1)
                data = leftover + data
                usable = len(data) // 4 * 4
                leftover = data[usable:]
                audio = resampler.process(np.frombuffer(data[:usable], dtype=np.float32))
                meter.add(audio)
                yield from rechunker.feed(audio)
                last_data = time.monotonic()
        finally:
            if proc.poll() is None:
                proc.terminate()
