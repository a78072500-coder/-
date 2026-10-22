# 일본어 실시간 자막 + 한국어 번역

블루스택에서 헬로우톡·maum 등으로 일본인과 통화할 때, **상대방이 하는 말**을
화면 위에 일본어 자막과 한국어 번역으로 실시간으로 띄워주는 Windows 프로그램입니다.

```
[PC 스피커 출력 캡처] → [말 끊기(VAD)] → [Whisper 일본어 인식(GPU)] → [DeepL/Claude 번역] → [자막 창]
```

- 상대가 말하는 동안 회색 **임시 자막**이 먼저 뜨고, 말을 멈추면 1초 안팎으로 확정 자막과 번역이 나타납니다.
- 내 목소리는 마이크로 들어가므로 자막에 잡히지 않습니다. 상대방 목소리만 나옵니다.
- 대화 내용은 `logs/` 폴더에 txt로 저장되어 일본어 복습에 쓸 수 있습니다.

## 1. 설치 (처음 한 번)

1. **Python 3.11 이상** 설치: <https://www.python.org/downloads/> (설치할 때 "Add python.exe to PATH" 체크)
2. **NVIDIA 그래픽 드라이버**를 최신으로 업데이트합니다.
3. 이 폴더에서 명령 프롬프트(cmd)를 열고:
   ```bat
   python -m venv .venv
   .venv\Scripts\activate
   pip install -r requirements.txt
   ```
   CUDA와 cuDNN 라이브러리도 pip로 같이 설치되므로 따로 설치하지 않아도 됩니다.
4. `config.example.toml`을 복사해 `config.toml`을 만들고, 번역 키를 넣습니다.
   - **DeepL (기본, 무료)**: <https://www.deepl.com/pro-api>에서 "API Free"에 가입하고
     키(끝이 `:fx`)를 `deepl_api_key`에 넣습니다. 월 50만 자까지 무료입니다.
   - **Claude (선택, 유료)**: 말투와 뉘앙스를 더 자연스럽게 옮깁니다.
     `engine = "claude"`로 바꾸고 환경변수 `ANTHROPIC_API_KEY`를 설정합니다.
     거절 응답에 대비해 서버 측 대체 모델(fallback)을 자동으로 쓰도록 설정되어 있습니다.

처음 실행할 때 Whisper 모델(약 1.6GB)을 자동으로 내려받으므로 몇 분 걸립니다.

## 2. 실행

```bat
.venv\Scripts\activate
python -m jp_subtitle
```

화면 아래에 자막 창이 뜹니다.

| 조작 | 동작 |
|---|---|
| 창 드래그 | 위치 이동 (다음 실행 때도 기억) |
| 오른쪽 아래 모서리 | 창 너비 조절 |
| `Ctrl` + 마우스 휠 | 글자 크기 |
| 우클릭 / 트레이 아이콘 | 일본어 원문 켜기·끄기, 기록 지우기, 숨기기, 종료 |

### 먼저 시험해 보기
블루스택이나 브라우저에서 **일본어 유튜브**를 틀어 놓고 실행해 보세요. 자막이 뜨면 준비가 끝난 겁니다.

그 밖의 실행 옵션:
```bat
python -m jp_subtitle --list-devices        :: 캡처할 수 있는 출력 장치 목록
python -m jp_subtitle --no-gui              :: 자막 창 없이 콘솔에만 출력
python -m jp_subtitle --input sample.wav    :: 스피커 대신 wav 파일로 시험
```

## 3. 블루스택에서 쓸 때

- **통화할 때는 이어폰이나 헤드셋을 쓰세요.** 스피커로 들으면 상대 목소리가 마이크로 다시 들어가 에코가 생깁니다.
  이어폰을 써도 자막 캡처는 정상으로 됩니다.
- 블루스택 설정 → 오디오에서 마이크와 스피커 장치를 확인하세요.
- 기본 설정은 **PC 전체 소리**를 캡처하므로, 통화 중에는 유튜브나 알림 소리를 꺼 두세요.
- **블루스택 소리만 받고 싶다면**:
  1. Windows 설정 → 시스템 → 소리 → **볼륨 믹서**에서 블루스택(`HD-Player`)의 출력 장치를 헤드셋으로 지정하고,
     다른 프로그램은 스피커로 둡니다.
  2. `config.toml`의 `[audio] device`에 헤드셋 이름 일부를 적습니다(예: `"Headset"`).
     정확한 이름은 `--list-devices`로 확인하세요.

## 4. 문제 해결

| 증상 | 해결 |
|---|---|
| `cublas64_12.dll`·`cudnn` 관련 오류 | `pip install -r requirements.txt`를 다시 실행하고 그래픽 드라이버를 업데이트하세요. 그래도 안 되면 `[whisper] device = "cpu"`, `compute_type = "int8"`, `model = "small"`로 바꾸세요(느리지만 동작합니다). |
| 그래픽카드 메모리 부족 | `model = "medium"` 또는 `"small"` |
| 잡음이 자막으로 뜸 | `[segmenter] vad_threshold`를 0.6~0.7로 올리세요 |
| 문장이 중간에 자주 끊김 | `silence_end_sec`를 0.8~1.0으로 올리세요 |
| 자막이 늦게 뜸 | `silence_end_sec`를 0.4~0.5로 내리고 `beam_size`를 1~2로 낮추세요 |
| "ご視聴ありがとうございました" 같은 엉뚱한 문장 | 무음에서 Whisper가 지어내는 문구입니다. 자주 나오는 문구는 `jp_subtitle/transcriber.py`의 `HALLUCINATIONS`에 추가하세요. |

## 5. exe로 만들기 (선택)

```bat
pip install pyinstaller
build_exe.bat
```

`dist\jp_subtitle\jp_subtitle.exe`가 생성됩니다. `config.toml`을 exe와 같은 폴더에 두세요.

## 개발

```bash
pip install -r requirements.txt pytest
python -m pytest
```

| 파일 | 역할 |
|---|---|
| `jp_subtitle/audio_capture.py` | WASAPI 루프백 캡처 / wav 파일 입력, 16kHz 모노 변환 |
| `jp_subtitle/segmenter.py` | silero VAD로 발화 단위 분리, 임시 조각 생성 |
| `jp_subtitle/transcriber.py` | faster-whisper 인식, 환각 문구 필터 |
| `jp_subtitle/translators/` | DeepL / Claude / 번역 안 함 |
| `jp_subtitle/pipeline.py` | 캡처 → 인식 → 번역 스레드 연결, 대화 기록 |
| `jp_subtitle/overlay.py` | PySide6 자막 창 |
