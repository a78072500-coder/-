# 일본어 통화 도우미

블루스택에서 헬로우톡·maum 등으로 일본인과 통화할 때 **듣기, 대답하기, 복습**을 돕는 프로그램입니다.
상대방의 말을 실시간 자막과 한국어 번역으로 보여 주고, 하고 싶은 말을 일본어로 바꿔 주며,
통화가 끝나면 복습 자료까지 만듭니다.

```
[PC 스피커 출력 캡처] → [말 끊기(VAD)] → [Whisper 일본어 인식(GPU)] → [DeepL/Claude 번역] → [메인 창 + 자막 창]
[마이크 (선택)] ─────→ [말 끊기(VAD)] → [Whisper 인식] ──→ 번역 문맥과 대화 기록
```

| 화면 | 할 수 있는 일 | 필요한 것 |
|---|---|---|
| **통화** | 실시간 자막과 번역, 읽는 법, 잘못 들었을 수 있는 문장 표시 | 번역에 DeepL, Claude, Gemini 중 하나의 키 |
| | 되묻기 문구 (`もう一度お願いします` 등) | 없음 |
| | 하고 싶은 말을 반말·존댓말 일본어로 바꾸기, 읽는 법과 한글 발음 | DeepL, Claude, Gemini 중 하나의 키 |
| | 추천 답변 3개 | Claude 또는 Gemini 키 |
| **기록** | 지난 통화 다시 보기, 기록 삭제 | 없음 |
| | AI 복습: 요약, 외워 둘 표현, 내 문장 고치기, 다음 통화 주제 | Claude 또는 Gemini 키 |
| **복습** | 저장한 문장을 간격 반복 카드로 익히기, Anki CSV 내보내기 | 없음 |
| **진단** | 자막이 왜 늦는지 구간별 시간 보기, 실행 기록 | 없음 |
| **설정** | API 키, 음성인식, 소리 장치, 자주 나오는 단어 | 없음 |

- 상대가 말하는 동안 회색 **임시 자막**이 먼저 뜨고, 문장이 끝나면 확정 자막과 번역이 나타납니다. 걸리는 시간은 [진단] 화면에서 볼 수 있습니다.
- **문장 길이 슬라이더** ([통화] 화면 위쪽): 쉬지 않고 이어 말해도 문장 끝을 찾아 한 문장씩 번역합니다. 왼쪽일수록 빨리 끊고, 오른쪽일수록 길게 묶습니다. 통화 중에 움직여도 바로 적용됩니다.
- `うん`, `へえ` 같은 **맞장구**는 번역 사용량을 쓰지 않고 작게 표시해서 앞 문장이 밀려나지 않습니다.
- **내 목소리**는 기본으로 인식하지 않습니다. [설정]에서 켜면 대화 기록과 번역 문맥에 들어갑니다 (이어폰 필수).
- 대화 내용은 `logs/` 폴더의 `history.db`와 txt에 저장됩니다. [설정]에서 끌 수 있고 [기록] 화면에서 지울 수 있습니다.
- 번역과 AI 기능을 쓰면 상대방 말이나 입력한 문장이 해당 서비스(DeepL, Anthropic, Google)로 전송됩니다. 음성은 저장하지도 보내지도 않습니다.
- **Gemini 무료 키 주의:** 무료 등급에서는 보낸 내용을 Google이 제품 개선에 쓰고 사람이 검토할 수 있으며, 약관은 개인정보를 보내지 말라고 안내합니다. 통화 상대의 개인정보가 오갈 수 있다면 결제를 연결한 유료 키를 쓰세요. 이 프로그램은 Gemini에 저장하지 않음(`store=false`)으로 요청합니다.

### 전역 단축키

통화 앱을 보고 있을 때도 동작합니다.

| Windows | macOS | 동작 |
|---|---|---|
| `Ctrl`+`Alt`+`R` | `Control`+`Option`+`R` | 메인 창을 띄우고 하고 싶은 말 입력 |
| `Ctrl`+`Alt`+`Q` | `Control`+`Option`+`Q` | 되묻기 문구를 자막 창에 하나씩 표시 |
| `Ctrl`+`Alt`+`S` | `Control`+`Option`+`S` | 추천 답변 만들기 (Claude 또는 Gemini 키 필요) |

## 0. 가장 쉬운 방법: exe 받기 (Python 설치 필요 없음)

1. **[jp_subtitle-windows.zip 내려받기](https://github.com/a78072500-coder/-/releases/latest/download/jp_subtitle-windows.zip)**
   (로그인 필요 없음. 그래픽카드용 라이브러리가 들어 있어서 1GB가 넘습니다)
2. 압축을 풀고 `jp_subtitle.exe`를 더블클릭합니다.
   **`jp_subtitle.exe`와 `_internal` 폴더는 꼭 같은 폴더에 있어야 합니다.** 옮길 때는 폴더째 옮기세요.
3. [설정] 화면에서 DeepL 무료 키를 넣고, [통화] 화면에서 [통화 시작]을 누릅니다.
   키가 없어도 일본어 자막은 나옵니다.

그래픽카드를 쓸 수 없는 상황이면 자동으로 CPU 모드로 바뀌고, [통화] 화면에 안내가 뜹니다.

### 업데이트
**통화 도우미 버전으로 처음 올릴 때는 zip 전체를 다시 받으세요** (`_internal`에 새 파일이 추가됐습니다).
그 뒤로는 PowerShell에서 아래 한 줄이면 됩니다.
(바탕화면의 `jp_subtitle` 폴더 기준이며, `config.toml`은 그대로 유지됩니다)

```powershell
$d=[Environment]::GetFolderPath('Desktop')+'\jp_subtitle'; taskkill /im jp_subtitle.exe /f 2>$null; $ProgressPreference='SilentlyContinue'; Invoke-WebRequest https://github.com/a78072500-coder/-/releases/latest/download/jp_subtitle.exe -OutFile "$d\jp_subtitle.exe"; Start-Process "$d\jp_subtitle.exe" -WorkingDirectory $d
```

## macOS (맥북)

Windows와 같은 화면과 기능을 그대로 쓸 수 있습니다. **Apple Silicon(M1 이후) 맥과 macOS 13 이상**이 필요합니다.
음성인식은 Apple GPU(MLX)로 돌아갑니다. 배포하는 앱은 Apple Silicon 전용이라, Intel 맥은 아래 "직접 설치"로 실행해야 하고 CPU로 돌아가서 느립니다.

### 설치 / 업데이트
터미널(응용 프로그램 → 유틸리티 → 터미널)에 붙여 넣으세요. 같은 명령으로 업데이트도 됩니다.

```bash
mkdir -p ~/Applications && cd ~/Applications && curl -fL -o /tmp/jp_subtitle-mac.zip https://github.com/a78072500-coder/-/releases/latest/download/jp_subtitle-mac.zip && rm -rf jp_subtitle-mac && ditto -x -k /tmp/jp_subtitle-mac.zip . && open jp_subtitle-mac/jp_subtitle.app
```

브라우저로 zip을 받았다면, 서명이 없는 앱이라 처음 열 때 막힙니다. 이때는
**시스템 설정 → 개인정보 보호 및 보안 → 아래쪽 "그래도 열기"**를 누르세요.

### 처음 실행할 때
1. **설정** 화면에서 API 키를 넣습니다.
2. **통화 앱 소리만 듣고 싶으면** 설정 화면의 **상대방 소리**에 앱 이름 일부를 적습니다(예: `maum`). **실행 중인 앱 불러오기**를 누르면 목록에서 고를 수 있습니다.
3. **통화 시작**을 누르면 **'화면 및 시스템 오디오 녹음' 권한**을 묻습니다. 허용한 뒤 앱을 다시 실행합니다.
   다른 앱(maum 등)의 소리를 듣는 데만 쓰고, 화면은 녹화하지 않습니다. 앱을 새 버전으로 바꾸면 다시 허용해야 할 수 있습니다.
4. 내 목소리도 인식하려면 **설정**에서 켭니다. 처음 쓸 때 **마이크 권한**을 묻습니다.

설정, 대화 기록, 로그(`jp_subtitle.log`)는 `~/Library/Application Support/jp_subtitle`에 저장됩니다.
Windows에서 쓰던 키와 기록을 옮기려면 `config.toml`과 `logs` 폴더를 이 위치에 복사하세요.
Apple Silicon 맥에서는 maum·헬로우톡 iPhone 앱이 App Store에서 그대로 설치되는 경우가 많아서, 블루스택 없이 쓸 수 있을 수도 있습니다.

## 1. 직접 설치 (Python으로 실행)

1. **Python 3.11 이상** 설치: <https://www.python.org/downloads/> (설치할 때 "Add python.exe to PATH" 체크)
2. **NVIDIA 그래픽 드라이버**를 최신으로 업데이트합니다.
3. 이 폴더에서 명령 프롬프트(cmd)를 열고:
   ```bat
   python -m venv .venv
   .venv\Scripts\activate
   pip install -r requirements.txt
   ```
   CUDA와 cuDNN 라이브러리도 pip로 같이 설치되므로 따로 설치하지 않아도 됩니다.
4. 프로그램의 [설정] 화면에서 API 키를 넣습니다 (`config.toml`에 저장됩니다).
   - **DeepL (무료)**: <https://www.deepl.com/pro-api>에서 "API Free"에 가입하고
     키(끝이 `:fx`)를 넣습니다. 월 50만 자까지 무료입니다. 자막 번역과 한국어→일본어 바꾸기에 씁니다.
   - **Claude (유료)**: <https://console.anthropic.com>. 추천 답변과 AI 복습에 쓰고, 자막 번역 엔진으로도 고를 수 있습니다.
     키를 비워 두면 환경변수 `ANTHROPIC_API_KEY`를 씁니다.
     거절 응답에 대비해 서버 측 대체 모델(fallback)을 자동으로 쓰도록 설정되어 있습니다.
   - **Gemini (무료 등급 있음)**: <https://aistudio.google.com/apikey>. Claude와 같은 기능을 쓸 수 있습니다.
     키를 비워 두면 환경변수 `GEMINI_API_KEY`를 씁니다. 기본 모델은 `gemini-3.5-flash-lite`입니다.
   - Claude와 Gemini 키가 모두 있으면 [설정]의 **답변·복습 AI**에서 고를 수 있습니다.
     자동으로 두면 자막 번역에 쓰는 AI를 먼저 쓰고, 그다음 Claude, Gemini 순으로 키가 있는 쪽을 씁니다.

처음 실행할 때 Whisper 모델(약 1.6GB)을 자동으로 내려받으므로 몇 분 걸립니다.

## 2. 실행

```bat
.venv\Scripts\activate
python -m jp_subtitle
```

메인 창이 뜨고, [통화 시작]을 누르면 화면 아래에 자막 창도 뜹니다. 자막 창 조작은 다음과 같습니다.

| 조작 | 동작 |
|---|---|
| 창 드래그 | 위치 이동 (다음 실행 때도 기억) |
| 오른쪽 아래 모서리 | 창 너비 조절 |
| `Ctrl` + 마우스 휠 | 글자 크기 |
| 우클릭 | 메인 창 열기, 일본어 원문·읽는 법 켜고 끄기, 자막 지우기, 숨기기 |

### 먼저 시험해 보기
블루스택이나 브라우저에서 **일본어 유튜브**를 틀어 놓고 [통화 시작]을 눌러 보세요. 자막이 뜨면 준비가 끝난 겁니다.

그 밖의 실행 옵션:
```bat
python -m jp_subtitle --list-devices        :: 캡처할 수 있는 출력 장치 목록
python -m jp_subtitle --list-mics           :: 쓸 수 있는 마이크 목록
python -m jp_subtitle --no-gui              :: 창 없이 콘솔에만 출력
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
  2. [설정] 화면의 **상대방 소리**에서 헤드셋을 고릅니다.

## 4. 문제 해결

| 증상 | 해결 |
|---|---|
| exe가 바로 꺼지고 `Failed to load Python DLL ... python311.dll` 또는 `python312.dll` 오류 | `jp_subtitle.exe` 옆에 `_internal` 폴더가 없는 경우입니다. 두 개를 같은 폴더에 두세요. |
| "CPU 로 실행 중" 안내 | `pip install -r requirements.txt`를 다시 실행하고 그래픽 드라이버를 업데이트하세요. 그래도 안 되면 [설정]에서 실행 장치를 CPU로, 모델을 `small`로 바꾸세요(느리지만 동작합니다). |
| 그래픽카드 메모리 부족 | [설정]에서 모델을 `medium` 또는 `small`로 바꾸세요 |
| 자막이 전혀 안 뜸 | [통화] 화면 위쪽의 **상대** 음량 막대가 움직이는지 보세요. 안 움직이면 소리가 다른 장치로 나가고 있는 것이니 [설정]의 **상대방 소리**를 확인하세요. |
| 잡음이 자막으로 뜸 | [설정]에서 말소리 판단 기준을 0.6~0.7로 올리세요 |
| 문장이 너무 길게 묶임 | [통화] 화면의 **문장 길이**를 왼쪽으로 옮기세요 |
| 문장이 중간에 자주 끊김 | **문장 길이**를 오른쪽으로 옮기세요. [설정]에서 문장 끝 대기와 최대 길이를 직접 정할 수도 있습니다 |
| 자막이 늦게 뜸 | [진단] 화면에서 가장 오래 걸리는 구간을 보고 안내를 따르세요 |
| "잘못 들었을 수 있어요"가 너무 자주 뜸 | `config.toml`의 `low_confidence_logprob`를 `-0.8` 정도로 낮추세요 |
| 단축키가 안 먹음 | [진단] 화면에 다른 프로그램이 쓰고 있는 단축키가 표시됩니다 |
| "ご視聴ありがとうございました" 같은 엉뚱한 문장 | 무음에서 Whisper가 지어내는 문구입니다. 자주 나오는 문구는 `jp_subtitle/transcriber.py`의 `HALLUCINATIONS`에 추가하세요. |

## 5. exe 직접 만들기 (선택)

```bat
pip install pyinstaller
python scripts\build_exe.py
```

`dist\jp_subtitle\jp_subtitle.exe`가 생성됩니다. GitHub에 push하면 Actions가 같은 빌드를 자동으로 수행합니다.
이때 일본어 음성(TTS)을 만들어 실제 Whisper로 인식되는지도 함께 검사합니다.

## 개발

```bash
pip install -r requirements.txt pytest
python -m pytest
```

| 파일 | 역할 |
|---|---|
| `jp_subtitle/audio_capture.py` | WASAPI 루프백·마이크(Windows) / ScreenCaptureKit 도우미·마이크(macOS) / wav 파일 입력, 16kHz 모노 변환 |
| `mac/sck_audio.swift` | macOS 시스템·앱 소리를 받아 넘겨주는 도우미 |
| `jp_subtitle/segmenter.py` | silero VAD로 발화 단위 분리, 임시 조각 생성, 문장 끝이나 숨 고른 곳에서 자르기 |
| `jp_subtitle/sentences.py` | 인식 결과에서 일본어 문장 끝 찾기 |
| `jp_subtitle/transcriber.py` | faster-whisper 인식, 단어별 시간, 신뢰도 계산, 환각 문구 필터 |
| `jp_subtitle/backchannel.py` | 맞장구 판별 |
| `jp_subtitle/reading.py` | 읽는 법(히라가나)과 한글 발음 (janome) |
| `jp_subtitle/translators/` | DeepL / Claude / Gemini / 번역 안 함 |
| `jp_subtitle/pipeline.py` | 캡처 → 인식 → 번역 스레드 연결, 지연 시간 측정 |
| `jp_subtitle/store.py` | 통화 기록, 복습 카드, 지연 통계 (SQLite) |
| `jp_subtitle/assistant.py` | 답변 만들기, 추천 답변, AI 복습 (Claude / Gemini / DeepL) |
| `jp_subtitle/engine.py` | 모델 준비, 통화 시작·종료, 설정 적용 |
| `jp_subtitle/hotkeys.py` | Windows·macOS 전역 단축키 |
| `jp_subtitle/overlay.py` | 항상 위에 뜨는 자막 창 |
| `jp_subtitle/ui/` | 메인 창과 통화·기록·복습·진단·설정 화면 |
