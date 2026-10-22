"""실제 Whisper 모델로 처음부터 끝까지 돌려보는 검사 (CI 의 Windows 에서 사용).

    python scripts/smoke_test.py make-wav ja.wav          # 일본어 TTS 음성 만들기
    python scripts/smoke_test.py check ja.wav -- python -m jp_subtitle
    python scripts/smoke_test.py check ja.wav -- dist/jp_subtitle/jp_subtitle.exe
    python scripts/smoke_test.py check ja.wav --device cuda -- ...   # GPU 없을 때 CPU 전환 확인
"""

from __future__ import annotations

import argparse
import asyncio
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np

SENTENCES = [
    "こんにちは、初めまして。",
    "韓国のどこに住んでいますか？",
]
# 인식 결과에 이 단어들이 나오면 통과 (TTS/모델에 따라 표기가 조금 달라질 수 있어 핵심어만 확인)
EXPECTED = ["こんにちは", "韓国"]

CONFIG = """\
[whisper]
model = "small"
device = "{device}"
compute_type = "{compute_type}"
beam_size = 1
[translator]
engine = "none"
[segmenter]
partial_interval_sec = 0
[log]
enabled = false
"""


async def _tts(text: str, path: Path) -> None:
    import edge_tts

    await edge_tts.Communicate(text, "ja-JP-NanamiNeural").save(str(path))


def make_wav(out: Path) -> None:
    from faster_whisper import decode_audio
    from scipy.io import wavfile

    pieces = [np.zeros(8000, dtype=np.float32)]
    with tempfile.TemporaryDirectory() as tmp:
        for i, text in enumerate(SENTENCES):
            mp3 = Path(tmp) / f"{i}.mp3"
            asyncio.run(_tts(text, mp3))
            pieces += [decode_audio(str(mp3), sampling_rate=16000), np.zeros(24000, np.float32)]
    audio = np.concatenate(pieces)
    wavfile.write(out, 16000, (audio * 32767).astype(np.int16))
    print(f"{out}: {len(audio) / 16000:.1f}초")


def check(wav: Path, command: list[str], device: str) -> int:
    compute_type = "float16" if device == "cuda" else "int8"
    with tempfile.TemporaryDirectory() as tmp:
        config = Path(tmp) / "config.toml"
        config.write_text(CONFIG.format(device=device, compute_type=compute_type), encoding="utf-8")
        proc = subprocess.run(
            [*command, "--config", str(config), "--input", str(wav), "--no-gui", "--fast"],
            stdin=subprocess.DEVNULL,
            capture_output=True,
            timeout=900,
        )
    out = proc.stdout.decode("utf-8", "replace")
    err = proc.stderr.decode("utf-8", "replace")
    print(out)
    if err.strip():
        print("--- stderr ---\n" + err)
    recognized = "".join(line for line in out.splitlines() if line.startswith("[JA]"))
    missing = [w for w in EXPECTED if w not in recognized]
    if proc.returncode != 0 or missing:
        print(f"실패: 종료코드 {proc.returncode}, 인식되지 않은 단어 {missing}")
        return 1
    if device == "cuda" and "CPU 로 실행 중" not in out:
        print("실패: GPU 가 없는데 CPU 전환 안내가 나오지 않았습니다")
        return 1
    print("통과: 일본어 음성이 정상적으로 인식되었습니다")
    return 0


def main() -> int:
    # "--" 뒤는 실행할 명령으로 그대로 넘긴다
    argv = sys.argv[1:]
    command: list[str] = []
    if "--" in argv:
        i = argv.index("--")
        argv, command = argv[:i], argv[i + 1 :]
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("make-wav")
    p.add_argument("out", type=Path)
    p = sub.add_parser("check")
    p.add_argument("wav", type=Path)
    p.add_argument(
        "--device", default="cpu",
        help="cuda 로 주면 GPU 가 없는 곳에서 CPU 자동 전환이 되는지 확인할 수 있다",
    )
    args = parser.parse_args(argv)
    if args.cmd == "make-wav":
        make_wav(args.out)
        return 0
    if not command:
        parser.error("check 뒤에 -- 와 실행할 명령을 적으세요")
    return check(args.wav, command, args.device)


if __name__ == "__main__":
    sys.exit(main())
