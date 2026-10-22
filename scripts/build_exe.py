"""PyInstaller 로 Windows 실행 폴더를 만든다.

    python scripts/build_exe.py

결과: dist/jp_subtitle/jp_subtitle.exe (+ config.toml, 사용법.txt)
pip 로 설치한 CUDA 라이브러리(nvidia-cublas/cudnn)도 함께 묶어서
받는 사람이 CUDA 를 따로 설치하지 않아도 GPU 로 돌아가게 한다.
"""

from __future__ import annotations

import os
import shutil
import site
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DIST = ROOT / "dist" / "jp_subtitle"

USAGE = """\
일본어 실시간 자막 + 한국어 번역
================================

1. config.toml 을 메모장으로 열어 deepl_api_key 에 DeepL 무료 API 키를 넣습니다.
   (키가 없으면 일본어 자막만 나옵니다)
2. jp_subtitle.exe 를 더블클릭합니다.
   처음 한 번은 음성인식 모델(약 1.6GB)을 내려받느라 몇 분 걸립니다.
3. 화면 아래 자막 창이 뜨면 블루스택에서 통화를 시작하세요. (이어폰 사용 권장)

- 자막 창 드래그: 이동 / Ctrl+휠: 글자 크기 / 우클릭: 메뉴(종료 등)
- 대화 기록은 logs 폴더에 저장됩니다.
- 자세한 설명: https://github.com/a78072500-coder/-
"""


def nvidia_binaries() -> list[str]:
    args = []
    roots = [Path(p) for p in site.getsitepackages() + [site.getusersitepackages()]]
    for root in roots:
        for bin_dir in sorted(root.glob("nvidia/*/bin")):
            dest = bin_dir.relative_to(root).as_posix()
            for dll in bin_dir.glob("*.dll"):
                args += ["--add-binary", f"{dll}{os.pathsep}{dest}"]
    return args


def main() -> int:
    nvidia = nvidia_binaries()
    print(f"CUDA DLL {len(nvidia) // 2}개 포함")
    cmd = [
        sys.executable, "-m", "PyInstaller",
        "--noconfirm", "--clean",
        "--name", "jp_subtitle",
        "--collect-all", "faster_whisper",
        "--collect-all", "ctranslate2",
        "--collect-all", "onnxruntime",
        "--collect-all", "pyaudiowpatch",
        *nvidia,
        str(ROOT / "run.py"),
    ]
    subprocess.run(cmd, check=True, cwd=ROOT)

    example = ROOT / "config.example.toml"
    shutil.copy(example, DIST / "config.example.toml")
    if not (DIST / "config.toml").exists():
        shutil.copy(example, DIST / "config.toml")
    (DIST / "사용법.txt").write_text(USAGE, encoding="utf-8-sig")
    print(f"완료: {DIST / 'jp_subtitle.exe'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
