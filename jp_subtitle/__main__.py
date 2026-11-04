"""실행: python -m jp_subtitle [--config config.toml] [--input test.wav] [--no-gui]"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from .config import load_config
from .models import Line

LOG_MAX_BYTES = 1_000_000


class ConsoleSink:
    def partial(self, uid: int, ja: str) -> None:
        if ja:
            print(f"  … {ja}")

    def final(self, line: Line) -> None:
        print(f"[JA] {line.ja}")
        if line.reading:
            print(f"[읽기] {line.reading}")

    def translated(self, line: Line) -> None:
        print(f"[KO] {line.ko or '(번역 없음)'}")

    def status(self, text: str) -> None:
        print(f"[상태] {text}")

    def level(self, speaker: str, value: float) -> None:
        pass


MAC_APP_DIR = Path.home() / "Library" / "Application Support" / "jp_subtitle"


def _default_config_path() -> Path:
    frozen = getattr(sys, "frozen", False)
    if frozen and sys.platform == "darwin":
        # .app 안이나 옆은 macOS 가 읽기 전용 위치로 옮겨 실행할 수 있어서 사용자 폴더에 둔다
        path = MAC_APP_DIR / "config.toml"
        if not path.exists():
            MAC_APP_DIR.mkdir(parents=True, exist_ok=True)
            example = Path(sys._MEIPASS) / "config.example.toml"
            if example.exists():
                path.write_text(example.read_text(encoding="utf-8"), encoding="utf-8")
        return path
    # exe 로 빌드했을 때는 exe 옆의 config.toml 을 쓴다
    base = Path(sys.executable).parent if frozen else Path.cwd()
    return base / "config.toml"


class _Tee:
    """검은 창을 숨기거나 콘솔이 없을 때도 실행 기록이 파일에 남게 한다."""

    def __init__(self, path: Path, original):
        too_big = path.exists() and path.stat().st_size > LOG_MAX_BYTES
        self._file = path.open("w" if too_big else "a", encoding="utf-8", buffering=1)
        self._original = original

    def write(self, text: str) -> int:
        self._file.write(text)
        if self._original is not None:
            try:
                self._original.write(text)
            except (OSError, ValueError):
                pass
        return len(text)

    def flush(self) -> None:
        self._file.flush()


def _utf8_console() -> None:
    # 한국어 Windows 콘솔(cp949)이나 파이프로 출력할 때 일본어 때문에 죽지 않게
    for stream in (sys.stdout, sys.stderr):
        if stream is not None and hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8", errors="replace")


def _disable_quick_edit() -> None:
    """검은 창을 클릭하면 Windows 가 '선택' 모드로 프로그램을 멈추는 것을 막는다."""
    if sys.platform != "win32":
        return
    import ctypes

    kernel32 = ctypes.windll.kernel32
    handle = kernel32.GetStdHandle(-10)  # STD_INPUT_HANDLE
    mode = ctypes.c_uint32()
    if kernel32.GetConsoleMode(handle, ctypes.byref(mode)):
        ENABLE_QUICK_EDIT_MODE, ENABLE_EXTENDED_FLAGS = 0x0040, 0x0080
        kernel32.SetConsoleMode(handle, (mode.value & ~ENABLE_QUICK_EDIT_MODE) | ENABLE_EXTENDED_FLAGS)


def _set_console_visible(visible: bool) -> None:
    """프로그램 창이 있으니 검은 창은 숨긴다. 시작하다 오류가 나면 다시 보여 준다."""
    if sys.platform != "win32" or not getattr(sys, "frozen", False):
        return
    import ctypes

    window = ctypes.windll.kernel32.GetConsoleWindow()
    if window:
        ctypes.windll.user32.ShowWindow(window, 5 if visible else 0)


def _run_console(args, cfg, config_path: Path, log_dir: Path) -> int:
    """창 없이 콘솔에만 출력한다 (시험과 자동 검사용)."""
    from .audio_capture import FileSource, LoopbackSource, MacAudioSource
    from .pipeline import ConversationLog, Pipeline
    from .reading import create_reader
    from .segmenter import Segmenter, SileroSpeechProb
    from .transcriber import create_transcriber
    from .translators import NoTranslator, create_translator

    sink = ConsoleSink()
    try:
        translator = create_translator(cfg.translator)
    except ValueError as e:
        # 키가 없어도 일본어 자막은 볼 수 있게 번역만 끈다
        print(f"[번역] {e}")
        sink.status(f"번역 꺼짐: {e}")
        translator = NoTranslator()
    transcriber = create_transcriber(cfg.whisper)
    if transcriber.notice:
        sink.status(transcriber.notice)
    reader = create_reader()
    segmenter = Segmenter(cfg.segmenter, SileroSpeechProb())
    if args.input:
        source = FileSource(args.input, realtime=not args.fast)
    elif sys.platform == "darwin":
        source = MacAudioSource(cfg.audio.app)
    else:
        source = LoopbackSource(cfg.audio.device)
    log = ConversationLog(log_dir) if cfg.log.enabled else None

    def transcribe(audio, final: bool):
        # 임시 자막은 빠르게(beam 1), 확정본은 정확하게
        if final:
            return transcriber.transcribe(audio, beam_size=None)
        return transcriber.transcribe(audio, beam_size=1, words=cfg.segmenter.sentence_cut)

    pipeline = Pipeline(
        source, segmenter, transcribe, translator, sink, cfg.translator.context_lines, log,
        reader.subtitle_reading if reader else None,
    )
    pipeline.start()
    print("[시작] Ctrl+C 로 종료")
    try:
        while not pipeline.wait(0.5):
            pass
    except KeyboardInterrupt:
        pipeline.stop()
    return 0


def main(argv: list[str] | None = None) -> int:
    _utf8_console()
    _disable_quick_edit()
    parser = argparse.ArgumentParser(
        prog="jp_subtitle", description="일본인과 통화할 때 자막, 답변 도우미, 복습을 제공합니다."
    )
    parser.add_argument("--config", type=Path, default=None, help="설정 파일 (기본: config.toml)")
    parser.add_argument("--input", type=Path, help="스피커 대신 wav 파일로 시험")
    parser.add_argument("--list-devices", action="store_true", help="캡처 가능한 출력 장치 목록")
    parser.add_argument("--list-mics", action="store_true", help="쓸 수 있는 마이크 목록")
    parser.add_argument("--no-gui", action="store_true", help="창 없이 콘솔에만 출력")
    parser.add_argument("--fast", action="store_true", help="파일 입력을 실시간 속도로 기다리지 않음")
    args = parser.parse_args(argv)

    if args.list_mics:
        from .audio_capture import list_input_devices

        for dev in list_input_devices():
            print(f"- {dev['name']}")
        return 0

    if args.list_devices:
        if sys.platform == "darwin":
            from .audio_capture import list_mac_apps

            print("소리를 낼 수 있는 실행 중인 앱 ([audio] app 에 이름 일부를 적으세요):")
            for name in list_mac_apps():
                print(f"- {name}")
            return 0
        from .audio_capture import list_loopback_devices

        for dev in list_loopback_devices():
            print(f"- {dev['name']}  ({int(dev['defaultSampleRate'])}Hz)")
        return 0

    config_path = args.config or _default_config_path()
    if getattr(sys, "frozen", False) and not args.no_gui:
        log_file = config_path.parent / "jp_subtitle.log"
        sys.stdout = _Tee(log_file, sys.stdout)
        sys.stderr = _Tee(log_file, sys.stderr)
    cfg = load_config(config_path)
    print(f"[설정] {config_path if config_path.exists() else '기본값 (config.toml 없음)'}")
    # 대화 기록 폴더는 설정 파일 옆 (상대 경로일 때)
    log_dir = config_path.parent / cfg.log.directory

    if args.no_gui:
        return _run_console(args, cfg, config_path, log_dir)

    from .ui.app import run_app

    _set_console_visible(False)
    return run_app(cfg, config_path, log_dir, args.input)


def run() -> int:
    """exe 를 더블클릭해서 실행했을 때 오류가 나도 창이 바로 닫히지 않게 한다."""
    try:
        return main()
    except SystemExit as e:
        return e.code if isinstance(e.code, int) else 0
    except Exception:
        if not getattr(sys, "frozen", False):
            raise
        import traceback

        traceback.print_exc()
        if sys.platform == "darwin":
            _mac_alert(traceback.format_exc())
            return 1
        _set_console_visible(True)
        print("\n오류가 발생했습니다. 위 메시지를 복사해서 알려주세요.")
        input("엔터를 누르면 종료합니다...")
        return 1


def _mac_alert(details: str) -> None:
    """콘솔이 없는 .app 에서 오류를 대화상자로 보여준다."""
    import subprocess

    log = MAC_APP_DIR / "jp_subtitle.log"
    message = f"자세한 내용은 {log} 에 저장되었습니다.\n\n{details[-800:]}"
    script = 'display alert "일본어 자막: 오류가 발생했습니다" message (system attribute "JP_MSG") as critical'
    subprocess.run(["/usr/bin/osascript", "-e", script], env={"JP_MSG": message}, check=False)


if __name__ == "__main__":
    sys.exit(run())
