import sys
import threading
import time

import numpy as np
import pytest

from jp_subtitle.audio_capture import BLOCK_SAMPLES, SAMPLE_RATE, to_mono_16k
from jp_subtitle.config import Config, SegmenterConfig, load_config
from jp_subtitle.pipeline import ConversationLog, Pipeline
from jp_subtitle.segmenter import Segmenter, energy_speech_prob
from jp_subtitle.transcriber import is_hallucination
from jp_subtitle.translators import NoTranslator, Translator, create_translator
from jp_subtitle.config import TranslatorConfig

BLOCK_SEC = BLOCK_SAMPLES / SAMPLE_RATE


def tone(sec: float) -> list[np.ndarray]:
    n = int(sec / BLOCK_SEC)
    t = np.arange(BLOCK_SAMPLES) / SAMPLE_RATE
    return [(0.3 * np.sin(2 * np.pi * 220 * t)).astype(np.float32) for _ in range(n)]


def silence(sec: float) -> list[np.ndarray]:
    return [np.zeros(BLOCK_SAMPLES, dtype=np.float32) for _ in range(int(sec / BLOCK_SEC))]


def run_segmenter(cfg, blocks):
    seg = Segmenter(cfg, energy_speech_prob)
    out = [u for b in blocks for u in seg.feed(b)]
    return out + list(seg.flush())


# --- segmenter -----------------------------------------------------------------

def test_two_sentences_split_on_silence():
    cfg = SegmenterConfig(partial_interval_sec=0)
    out = run_segmenter(cfg, silence(0.5) + tone(1.0) + silence(1.0) + tone(2.0) + silence(1.0))
    assert [u.final for u in out] == [True, True]
    assert [u.utterance_id for u in out] == [0, 1]
    # 앞뒤 무음 없이 대략 말한 길이 + pre-roll
    assert 1.0 <= len(out[0].audio) / SAMPLE_RATE <= 1.3
    assert 2.0 <= len(out[1].audio) / SAMPLE_RATE <= 2.3


def test_short_noise_ignored():
    cfg = SegmenterConfig(partial_interval_sec=0, min_utterance_sec=0.4)
    out = run_segmenter(cfg, silence(0.5) + tone(0.1) + silence(1.0))
    assert out == []


def test_long_speech_forced_cut():
    cfg = SegmenterConfig(partial_interval_sec=0, max_utterance_sec=3.0)
    out = run_segmenter(cfg, tone(7.0) + silence(1.0))
    assert len(out) == 3
    assert all(u.final for u in out)


def test_partials_emitted_while_speaking():
    cfg = SegmenterConfig(partial_interval_sec=1.0)
    out = run_segmenter(cfg, tone(3.5) + silence(1.0))
    partials = [u for u in out if not u.final]
    assert len(partials) == 3
    assert all(u.utterance_id == 0 for u in out)
    assert out[-1].final
    # 임시 조각은 점점 길어진다
    lengths = [len(u.audio) for u in partials]
    assert lengths == sorted(lengths)


def test_flush_finishes_open_utterance():
    cfg = SegmenterConfig(partial_interval_sec=0)
    out = run_segmenter(cfg, tone(1.0))
    assert len(out) == 1 and out[0].final


# --- 기타 --------------------------------------------------------------------------

def test_resample_to_16k_mono():
    stereo_48k = np.ones(48000 * 2, dtype=np.float32)
    out = to_mono_16k(stereo_48k, 48000, 2)
    assert len(out) == 16000
    assert out.dtype == np.float32


@pytest.mark.parametrize(
    "text,expected",
    [
        ("ご視聴ありがとうございました。", True),
        ("チャンネル登録よろしくお願いします！", True),
        ("", True),
        ("。", True),
        ("あああああああああ", True),
        ("ははははははは", True),
        ("ありがとうございました", False),
        ("今日は何してたの？", False),
        ("韓国語勉強してるんですか", False),
    ],
)
def test_hallucination_filter(text, expected):
    assert is_hallucination(text) is expected


def test_config_partial_override(tmp_path):
    p = tmp_path / "config.toml"
    p.write_text('[whisper]\ndevice = "cpu"\n[segmenter]\nsilence_end_sec = 1\n', encoding="utf-8")
    cfg = load_config(p)
    assert cfg.whisper.device == "cpu"
    assert cfg.whisper.model == "large-v3-turbo"
    assert cfg.segmenter.silence_end_sec == 1.0


def test_config_rejects_unknown_key(tmp_path):
    p = tmp_path / "config.toml"
    p.write_text("[whisper]\nmodle = 'small'\n", encoding="utf-8")
    with pytest.raises(ValueError, match="modle"):
        load_config(p)


def test_example_config_loads():
    cfg = load_config("config.example.toml")
    assert cfg == Config()


def test_translator_factory():
    assert isinstance(create_translator(TranslatorConfig(engine="none")), NoTranslator)
    with pytest.raises(ValueError):
        create_translator(TranslatorConfig(engine="papago"))
    with pytest.raises(ValueError, match="deepl_api_key"):
        create_translator(TranslatorConfig(engine="deepl", deepl_api_key=""))


# --- pipeline (가짜 오디오/인식/번역) -----------------------------------------------

class FakeSource:
    def __init__(self, blocks):
        self._blocks = blocks
        self._stop = threading.Event()

    def blocks(self):
        for b in self._blocks:
            if self._stop.is_set():
                return
            yield b

    def stop(self):
        self._stop.set()


class RecordingSink:
    def __init__(self):
        self.events = []

    def partial(self, uid, ja):
        self.events.append(("partial", uid, ja))

    def final(self, uid, ja):
        self.events.append(("final", uid, ja))

    def translated(self, uid, ja, ko):
        self.events.append(("translated", uid, ja, ko))

    def status(self, text):
        self.events.append(("status", text))


class EchoTranslator(Translator):
    def __init__(self):
        self.histories = []

    def translate(self, ja, history):
        self.histories.append(list(history))
        return f"KO({ja})"


def test_pipeline_end_to_end(tmp_path):
    blocks = silence(0.3) + tone(1.0) + silence(1.0) + tone(1.5) + silence(1.0) + tone(0.1)
    seg = Segmenter(SegmenterConfig(partial_interval_sec=0), energy_speech_prob)
    texts = iter(["こんにちは", "元気ですか"])
    translator = EchoTranslator()
    sink = RecordingSink()
    log = ConversationLog(tmp_path)
    p = Pipeline(FakeSource(blocks), seg, lambda a, final: next(texts), translator, sink, 4, log)
    p.start()
    assert p.wait(10)

    translated = [e for e in sink.events if e[0] == "translated"]
    assert translated == [
        ("translated", 0, "こんにちは", "KO(こんにちは)"),
        ("translated", 1, "元気ですか", "KO(元気ですか)"),
    ]
    # 두 번째 번역에는 첫 번째 문장이 문맥으로 들어간다
    assert translator.histories == [[], [("こんにちは", "KO(こんにちは)")]]
    content = log.path.read_text(encoding="utf-8")
    assert "こんにちは" in content and "KO(元気ですか)" in content


def test_pipeline_survives_translator_error():
    class Broken(Translator):
        def translate(self, ja, history):
            raise RuntimeError("network down")

    seg = Segmenter(SegmenterConfig(partial_interval_sec=0), energy_speech_prob)
    sink = RecordingSink()
    p = Pipeline(FakeSource(tone(1.0) + silence(1.0)), seg, lambda a, f: "テスト", Broken(), sink)
    p.start()
    assert p.wait(10)
    assert ("final", 0, "テスト") in sink.events
    assert ("translated", 0, "テスト", "") in sink.events
    assert any(e[0] == "status" and "network down" in e[1] for e in sink.events)


def test_pipeline_drops_stale_partials():
    """인식이 느려도 임시 조각이 쌓이지 않고 확정본은 빠짐없이 처리된다."""
    seg = Segmenter(SegmenterConfig(partial_interval_sec=0.2), energy_speech_prob)
    calls = []

    def slow_transcribe(audio, final):
        calls.append(final)
        time.sleep(0.05)
        return "テスト"

    sink = RecordingSink()
    blocks = (tone(3.0) + silence(1.0)) * 2
    p = Pipeline(FakeSource(blocks), seg, slow_transcribe, NoTranslator(), sink)
    p.start()
    assert p.wait(10)
    assert calls.count(True) == 2
    assert calls.count(False) < 2 * int(3.0 / 0.2)


# --- 실행 진입점 / Claude 번역기 (모델·네트워크 없이) ------------------------------

def test_main_with_wav_file(tmp_path, monkeypatch, capsys):
    from scipy.io import wavfile

    import jp_subtitle.segmenter as segmenter_mod
    import jp_subtitle.transcriber as transcriber_mod
    from jp_subtitle.__main__ import main

    class StubTranscriber:
        notice = ""

        def __init__(self, cfg):
            self.n = 0

        def transcribe(self, audio, beam_size=None):
            self.n += 1
            return f"文{self.n}"

    monkeypatch.setattr(transcriber_mod, "Transcriber", StubTranscriber)
    monkeypatch.setattr(segmenter_mod, "SileroSpeechProb", lambda: energy_speech_prob)

    audio = np.concatenate(silence(0.3) + tone(1.0) + silence(1.0) + tone(1.0))
    wav = tmp_path / "in.wav"
    wavfile.write(wav, 16000, (audio * 32767).astype(np.int16))
    cfg = tmp_path / "config.toml"
    cfg.write_text(
        f'[translator]\nengine = "deepl"\n[segmenter]\npartial_interval_sec = 0\n'
        f'[log]\ndirectory = "{(tmp_path / "logs").as_posix()}"\n',
        encoding="utf-8",
    )
    assert main(["--config", str(cfg), "--input", str(wav), "--no-gui", "--fast"]) == 0
    out = capsys.readouterr().out
    assert "[JA] 文1" in out and "[JA] 文2" in out
    # DeepL 키가 없으면 번역만 끄고 자막은 계속 나온다
    assert "번역 꺼짐" in out
    assert len(list((tmp_path / "logs").glob("*.txt"))) == 1


def test_claude_translator_request(monkeypatch):
    from types import SimpleNamespace

    from jp_subtitle.translators.claude_tr import ClaudeTranslator

    monkeypatch.setenv("ANTHROPIC_API_KEY", "test")
    tr = ClaudeTranslator("claude-opus-5-5")
    captured = {}

    def fake_create(**kwargs):
        captured.update(kwargs)
        return SimpleNamespace(
            stop_reason="end_turn",
            content=[SimpleNamespace(type="thinking"), SimpleNamespace(type="text", text=" 안녕 \n")],
        )

    monkeypatch.setattr(tr._client.beta.messages, "create", fake_create)
    assert tr.translate("やあ", [("こんにちは", "안녕하세요")]) == "안녕"
    assert captured["model"] == "claude-opus-5-5"
    assert captured["fallbacks"] == "default"
    assert "こんにちは → 안녕하세요" in captured["messages"][0]["content"]
    assert captured["messages"][0]["content"].endswith("やあ")

    monkeypatch.setattr(
        tr._client.beta.messages, "create",
        lambda **kw: SimpleNamespace(stop_reason="refusal", content=[]),
    )
    assert tr.translate("やあ", []) == ""


def test_transcriber_falls_back_to_cpu(monkeypatch):
    import jp_subtitle.transcriber as transcriber_mod
    from jp_subtitle.config import WhisperConfig

    loads = []

    def fake_load(self, model, device, compute_type):
        loads.append((model, device, compute_type))
        if device == "cuda":
            raise RuntimeError("Library cublas64_12.dll is not found")

    monkeypatch.setattr(transcriber_mod.Transcriber, "_load", fake_load)
    t = transcriber_mod.Transcriber(WhisperConfig())
    assert loads == [("large-v3-turbo", "cuda", "float16"), ("small", "cpu", "int8")]
    assert "CPU" in t.notice

    loads.clear()
    with pytest.raises(RuntimeError):
        def broken(self, *a):
            raise RuntimeError("x")
        monkeypatch.setattr(transcriber_mod.Transcriber, "_load", broken)
        transcriber_mod.Transcriber(WhisperConfig(device="cpu", compute_type="int8"))


# --- 실시간 캡처 (가짜 WASAPI 장치) -----------------------------------------------

def test_stream_resampler_matches_one_shot():
    from jp_subtitle.audio_capture import StreamResampler

    rng = np.random.default_rng(0)
    for rate in (48000, 44100):
        t = np.arange(rate) / rate
        sig = np.sin(2 * np.pi * 440 * t).astype(np.float32)
        stereo = np.stack([sig, sig], 1).reshape(-1)
        whole = StreamResampler(rate, 2).process(stereo)
        r = StreamResampler(rate, 2)
        parts, i = [], 0
        while i < len(sig):
            n = int(rng.integers(1, 1500))
            parts.append(r.process(stereo[2 * i : 2 * (i + n)]))
            i += n
        chunked = np.concatenate(parts)
        assert abs(len(chunked) - SAMPLE_RATE) <= 1
        np.testing.assert_allclose(chunked, whole[: len(chunked)], atol=1e-5)


def test_loopback_reads_small_chunks_without_gaps(monkeypatch):
    """장치가 조금씩만 데이터를 줘도 소리가 빠짐없이 이어지고 가짜 무음이 끼지 않는다."""
    import sys
    import types

    import jp_subtitle.audio_capture as ac

    rate = 48000
    t = np.arange(rate * 2) / rate
    tone_48k = (0.5 * np.sin(2 * np.pi * 300 * t)).astype(np.float32)
    stereo = np.stack([tone_48k, tone_48k], 1).reshape(-1)

    class FakeStream:
        def __init__(self):
            self.pos = 0
            self.calls = 0

        def get_read_available(self):
            self.calls += 1
            if self.calls % 3 == 0:  # 가끔 아직 데이터가 없는 척
                return 0
            return min(300, len(tone_48k) - self.pos)

        def read(self, n, exception_on_overflow=True):
            chunk = stereo[2 * self.pos : 2 * (self.pos + n)]
            self.pos += n
            return chunk.tobytes()

        def stop_stream(self):
            pass

        def close(self):
            pass

    stream = FakeStream()

    class FakePyAudio:
        def get_default_wasapi_loopback(self):
            return {"name": "fake", "defaultSampleRate": rate, "maxInputChannels": 2, "index": 0}

        def open(self, **kwargs):
            return stream

        def terminate(self):
            pass

    fake = types.SimpleNamespace(PyAudio=FakePyAudio, paFloat32=1)
    monkeypatch.setitem(sys.modules, "pyaudiowpatch", fake)
    monkeypatch.setattr(sys, "platform", "win32")

    src = ac.LoopbackSource()
    blocks = []
    for block in src.blocks():
        blocks.append(block)
        if stream.pos >= len(tone_48k) or stream.calls > 200_000:  # 회귀 시 무한루프 방지
            src.stop()
    audio = np.concatenate(blocks)
    # 2초 분량이 (블록 단위 자투리만 빼고) 모두 들어오고
    assert len(audio) >= 2 * SAMPLE_RATE - 2 * BLOCK_SAMPLES
    # 중간에 무음이 끼어들지 않았다 (필터 시작 구간 제외)
    body = audio[200 : 2 * SAMPLE_RATE - BLOCK_SAMPLES]
    rms = np.sqrt(np.convolve(body**2, np.ones(160) / 160, mode="valid"))
    assert rms.min() > 0.2


# --- macOS (Linux 에서 가짜 도우미/플랫폼으로 확인) -----------------------------------

FAKE_HELPER = r'''
import sys, time, numpy as np
mode = sys.argv[1] if len(sys.argv) > 1 else "ok"
if mode == "perm":
    sys.stderr.write("PERMISSION\n"); sys.exit(3)
rate = 48000
t = np.arange(rate) / rate
tone = (0.5 * np.sin(2 * np.pi * 300 * t)).astype(np.float32)
sys.stderr.write("READY\n"); sys.stderr.write("FORMAT 48000\n"); sys.stderr.flush()
out = sys.stdout.buffer
for i in range(0, len(tone), 1000):          # 1초 분량을 잘게 나눠서
    out.write(tone[i:i + 1000].tobytes()[:3999] if i == 0 else tone[i:i + 1000].tobytes())
    if i == 0:
        out.write(tone[:1000].tobytes()[3999:])  # 4바이트 경계가 어긋난 조각도 처리해야 한다
    out.flush()
time.sleep(30)
'''


def _mac_source(tmp_path, *args):
    from jp_subtitle.audio_capture import MacAudioSource

    script = tmp_path / "fake_helper.py"
    script.write_text(FAKE_HELPER, encoding="utf-8")
    return MacAudioSource(helper=[sys.executable, str(script), *args])


def test_mac_source_reads_helper_and_fills_silence(tmp_path):
    src = _mac_source(tmp_path)
    blocks = []
    started = time.monotonic()
    for block in src.blocks():
        blocks.append(block)
        if len(blocks) * BLOCK_SAMPLES >= int(1.8 * SAMPLE_RATE) or time.monotonic() - started > 20:
            src.stop()
    audio = np.concatenate(blocks)
    # 1초짜리 톤이 끊김 없이 들어오고
    tone_part = audio[200 : SAMPLE_RATE - BLOCK_SAMPLES]
    rms = np.sqrt(np.convolve(tone_part**2, np.ones(160) / 160, mode="valid"))
    assert rms.min() > 0.2
    # 도우미가 조용해진 뒤에는 무음이 채워져 문장이 끝날 수 있다
    assert len(audio) >= int(1.5 * SAMPLE_RATE)
    assert np.abs(audio[int(1.3 * SAMPLE_RATE) :]).max() < 0.01


def test_mac_source_permission_error(tmp_path):
    from jp_subtitle.audio_capture import MAC_PERMISSION_HELP

    src = _mac_source(tmp_path, "perm")
    with pytest.raises(RuntimeError) as err:
        for _ in src.blocks():
            pass
    assert str(err.value) == MAC_PERMISSION_HELP


def test_mlx_repo_names():
    from jp_subtitle.transcriber import mlx_repo

    assert mlx_repo("large-v3-turbo") == "mlx-community/whisper-large-v3-turbo"
    assert mlx_repo("small") == "mlx-community/whisper-small-mlx"
    assert mlx_repo("someone/custom") == "someone/custom"


@pytest.mark.parametrize(
    "machine,mlx_ok,expected",
    [
        ("arm64", True, ("mlx", "")),
        ("arm64", False, ("cpu", "Apple GPU")),
        ("x86_64", True, ("cpu", "Intel")),
    ],
)
def test_create_transcriber_on_mac(monkeypatch, machine, mlx_ok, expected):
    import jp_subtitle.transcriber as tm
    from jp_subtitle.config import WhisperConfig

    class FakeMLX:
        def __init__(self, cfg):
            if not mlx_ok:
                raise RuntimeError("no Metal device")
            self.kind, self.notice = "mlx", ""

    class FakeCPU:
        def __init__(self, cfg):
            assert (cfg.device, cfg.model, cfg.compute_type) == ("cpu", "small", "int8")
            self.kind, self.notice = "cpu", ""

    monkeypatch.setattr(tm.sys, "platform", "darwin")
    monkeypatch.setattr(tm.platform, "machine", lambda: machine)
    monkeypatch.setattr(tm, "MLXTranscriber", FakeMLX)
    monkeypatch.setattr(tm, "Transcriber", FakeCPU)
    t = tm.create_transcriber(WhisperConfig())
    assert t.kind == expected[0]
    assert expected[1] in t.notice
