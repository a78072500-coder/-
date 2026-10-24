"""PyInstaller 로 실행 파일을 만든다.

    python scripts/build_exe.py

Windows: dist/jp_subtitle/jp_subtitle.exe (+ config.toml, 사용법.txt)
  pip 로 설치한 CUDA 라이브러리(nvidia-cublas/cudnn)도 함께 묶어서
  받는 사람이 CUDA 를 따로 설치하지 않아도 GPU 로 돌아가게 한다.
macOS:   dist/jp_subtitle-mac.zip (jp_subtitle.app + 사용법.txt)
  ScreenCaptureKit 오디오 도우미(Swift)를 arm64/x86_64 공용으로 빌드해 함께 넣는다.
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


USAGE_MAC = """\
일본어 실시간 자막 + 한국어 번역 (macOS)
=======================================

1. jp_subtitle.app 을 실행합니다.
   - "확인되지 않은 개발자" 경고가 뜨면: 시스템 설정 → 개인정보 보호 및 보안 → 아래쪽 "그래도 열기"
2. 처음 실행하면 '화면 및 시스템 오디오 녹음' 권한을 묻습니다. 허용한 뒤 앱을 다시 실행하세요.
   (다른 앱의 소리를 듣기 위해 필요합니다. 화면은 녹화하지 않습니다)
3. 자막 창 우클릭 → "설정 파일 열기" 에서 deepl_api_key 를 넣고 앱을 다시 실행합니다.
   - maum 소리만 듣고 싶으면 [audio] app = "maum" 처럼 앱 이름 일부를 적으세요.
4. 처음 한 번은 음성인식 모델(약 1.6GB)을 내려받느라 몇 분 걸립니다.

- 자막 창 드래그: 이동 / Ctrl+휠: 글자 크기 / 우클릭: 메뉴
- 설정·대화 기록·로그: ~/Library/Application Support/jp_subtitle
- 자세한 설명: https://github.com/a78072500-coder/-
"""


def build_mac_helper() -> Path:
    """sck_audio 를 Apple Silicon / Intel 공용(universal) 바이너리로 빌드한다."""
    build = ROOT / "build"
    build.mkdir(exist_ok=True)
    source = ROOT / "mac" / "sck_audio.swift"
    parts = []
    for arch in ("arm64", "x86_64"):
        out = build / f"sck_audio-{arch}"
        subprocess.run(
            ["swiftc", "-O", "-target", f"{arch}-apple-macos13", str(source), "-o", str(out)],
            check=True,
        )
        parts.append(str(out))
    helper = build / "sck_audio"
    subprocess.run(["lipo", "-create", *parts, "-output", str(helper)], check=True)
    return helper


def main_mac() -> int:
    helper = build_mac_helper()
    collect = ["faster_whisper", "ctranslate2", "onnxruntime"]
    try:
        import mlx_whisper  # noqa: F401  (Apple Silicon 에서만 설치됨)

        collect += ["mlx", "mlx_whisper"]
    except ImportError:
        pass
    cmd = [
        sys.executable, "-m", "PyInstaller",
        "--noconfirm", "--clean", "--windowed",
        "--name", "jp_subtitle",
        "--osx-bundle-identifier", "io.github.a78072500-coder.jp-subtitle",
        *[arg for name in collect for arg in ("--collect-all", name)],
        # mlx-whisper 가 설치하지만 실행에는 쓰지 않는 torch 는 빼서 앱 크기를 줄인다
        "--exclude-module", "torch",
        "--add-binary", f"{helper}:.",
        "--add-data", f"{ROOT / 'config.example.toml'}:.",
        str(ROOT / "run.py"),
    ]
    subprocess.run(cmd, check=True, cwd=ROOT)

    package = ROOT / "dist" / "jp_subtitle-mac"
    shutil.rmtree(package, ignore_errors=True)
    package.mkdir()
    # 심볼릭 링크와 서명을 보존하려고 ditto 로 복사/압축한다
    subprocess.run(["ditto", str(ROOT / "dist" / "jp_subtitle.app"), str(package / "jp_subtitle.app")], check=True)
    (package / "사용법.txt").write_text(USAGE_MAC, encoding="utf-8")
    archive = ROOT / "dist" / "jp_subtitle-mac.zip"
    archive.unlink(missing_ok=True)
    subprocess.run(["ditto", "-c", "-k", "--keepParent", str(package), str(archive)], check=True)
    print(f"완료: {archive}")
    return 0


def main() -> int:
    if sys.platform == "darwin":
        return main_mac()
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
