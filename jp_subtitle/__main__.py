"""실행: python -m jp_subtitle [--config config.toml] [--input test.wav] [--no-gui]"""

from __future__ import annotations

import argparse
import signal
import sys
from pathlib import Path

from .config import load_config


class ConsoleSink:
    def partial(self, uid: int, ja: str) -> None:
        if ja:
            print(f"  … {ja}")

    def final(self, uid: int, ja: str) -> None:
        print(f"[JA] {ja}")

    def translated(self, uid: int, ja: str, ko: str) -> None:
        print(f"[KO] {ko or '(번역 없음)'}")

    def status(self, text: str) -> None:
        print(f"[상태] {text}")


def _default_config_path() -> Path:
    # exe 로 빌드했을 때는 exe 옆의 config.toml 을 쓴다
    base = Path(sys.executable).parent if getattr(sys, "frozen", False) else Path.cwd()
    return base / "config.toml"


def _utf8_console() -> None:
    # 한국어 Windows 콘솔(cp949)이나 파이프로 출력할 때 일본어 때문에 죽지 않게
    for stream in (sys.stdout, sys.stderr):
        if stream is not None and hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8", errors="replace")


def main(argv: list[str] | None = None) -> int:
    _utf8_console()
    parser = argparse.ArgumentParser(
        prog="jp_subtitle", description="일본어 음성을 실시간 자막 + 한국어 번역으로 보여줍니다."
    )
    parser.add_argument("--config", type=Path, default=None, help="설정 파일 (기본: config.toml)")
    parser.add_argument("--input", type=Path, help="스피커 대신 wav 파일로 시험")
    parser.add_argument("--list-devices", action="store_true", help="캡처 가능한 출력 장치 목록")
    parser.add_argument("--no-gui", action="store_true", help="자막 창 없이 콘솔에만 출력")
    parser.add_argument("--fast", action="store_true", help="파일 입력을 실시간 속도로 기다리지 않음")
    args = parser.parse_args(argv)

    if args.list_devices:
        from .audio_capture import list_loopback_devices

        for dev in list_loopback_devices():
            print(f"- {dev['name']}  ({int(dev['defaultSampleRate'])}Hz)")
        return 0

    config_path = args.config or _default_config_path()
    cfg = load_config(config_path)
    print(f"[설정] {config_path if config_path.exists() else '기본값 (config.toml 없음)'}")

    from .audio_capture import FileSource, LoopbackSource
    from .pipeline import ConversationLog, Pipeline
    from .segmenter import Segmenter, SileroSpeechProb
    from .transcriber import Transcriber
    from .translators import create_translator

    notices: list[str] = []
    try:
        translator = create_translator(cfg.translator)
    except ValueError as e:
        # 키가 없어도 일본어 자막은 볼 수 있게 번역만 끈다
        from .translators import NoTranslator

        print(f"[번역] {e}")
        notices.append(f"번역 꺼짐: {e}")
        translator = NoTranslator()
    transcriber = Transcriber(cfg.whisper)
    if transcriber.notice:
        notices.append(transcriber.notice)
    segmenter = Segmenter(cfg.segmenter, SileroSpeechProb())
    if args.input:
        source = FileSource(args.input, realtime=not args.fast)
    else:
        source = LoopbackSource(cfg.audio.device)
    log = ConversationLog(cfg.log.directory) if cfg.log.enabled else None

    def transcribe(audio, final: bool) -> str:
        # 임시 자막은 빠르게(beam 1), 확정본은 정확하게
        return transcriber.transcribe(audio, beam_size=None if final else 1)

    if args.no_gui:
        sink = ConsoleSink()
        pipeline = Pipeline(source, segmenter, transcribe, translator, sink,
                            cfg.translator.context_lines, log)
        pipeline.start()
        for notice in notices:
            sink.status(notice)
        print("[시작] Ctrl+C 로 종료")
        try:
            while not pipeline.wait(0.5):
                pass
        except KeyboardInterrupt:
            pipeline.stop()
        return 0

    from PySide6.QtWidgets import QApplication

    from .overlay import SinkBridge, SubtitleOverlay

    app = QApplication(sys.argv[:1])
    app.setQuitOnLastWindowClosed(False)
    bridge = SinkBridge()
    overlay = SubtitleOverlay(cfg.overlay, bridge)
    pipeline = Pipeline(source, segmenter, transcribe, translator, bridge,
                        cfg.translator.context_lines, log)
    overlay.show()
    pipeline.start()
    bridge.status(" / ".join(notices) or "듣는 중… 상대방이 말하면 자막이 나타납니다 (우클릭: 메뉴)")
    signal.signal(signal.SIGINT, lambda *_: app.quit())
    app.aboutToQuit.connect(pipeline.stop)
    app.aboutToQuit.connect(overlay.close)
    return app.exec()


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
        print("\n오류가 발생했습니다. 위 메시지를 복사해서 알려주세요.")
        input("엔터를 누르면 종료합니다...")
        return 1


if __name__ == "__main__":
    sys.exit(run())
