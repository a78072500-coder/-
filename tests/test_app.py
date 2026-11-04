"""통화 도우미 기능 검사. 모델, 네트워크, 실제 화면 없이 돌아간다."""

import csv
import ctypes
import json
import os
import sys
import threading
import time
import types
from datetime import date, timedelta
from types import SimpleNamespace

import numpy as np
import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from jp_subtitle.audio_capture import BLOCK_SAMPLES, SAMPLE_RATE
from jp_subtitle.backchannel import backchannel_gloss
from jp_subtitle.config import Config, SegmenterConfig, WhisperConfig, load_config, save_config
from jp_subtitle.models import ME, THEM, Line, Timing, Transcript
from jp_subtitle.pipeline import ME_PREFIX, MicChannel, Pipeline, _AsrQueue
from jp_subtitle.reading import Reader, has_kanji, kana_to_hangul
from jp_subtitle.segmenter import Segmenter, Utterance, energy_speech_prob
from jp_subtitle.store import HistoryStore, SessionRecorder, export_anki_csv
from jp_subtitle.transcriber import build_transcript
from jp_subtitle.translators import NoTranslator, Translator

BLOCK_SEC = BLOCK_SAMPLES / SAMPLE_RATE


def tone(sec):
    t = np.arange(BLOCK_SAMPLES) / SAMPLE_RATE
    return [(0.3 * np.sin(2 * np.pi * 220 * t)).astype(np.float32) for _ in range(int(sec / BLOCK_SEC))]


def silence(sec):
    return [np.zeros(BLOCK_SAMPLES, dtype=np.float32) for _ in range(int(sec / BLOCK_SEC))]


# --- 맞장구 ------------------------------------------------------------------------


@pytest.mark.parametrize(
    "text,expected",
    [
        ("うん", "응"),
        ("うんうんうん", "응"),
        ("はい、はい。", "네"),
        ("ハイ", "네"),
        ("なるほどね", "그렇구나"),
        ("そうそうそう", "맞아"),
        ("へえ、そうなんだ", "오~ 그렇구나"),
        ("マジで？", "진짜?"),
        ("今日は", None),
        ("はい、行きます", None),
        ("そうですねえ、たぶん大丈夫だと思います", None),
        ("", None),
        ("ね", None),
    ],
)
def test_backchannel(text, expected):
    assert backchannel_gloss(text) == expected


# --- 읽는 법 ------------------------------------------------------------------------


def test_kana_to_hangul():
    assert kana_to_hangul("コンニチワ") == "콘니치와"
    assert kana_to_hangul("きょう") == "쿄우"
    assert kana_to_hangul("がっこう") == "갓코우"
    assert kana_to_hangul("コーヒー") == "코-히-"
    assert kana_to_hangul("すきな んです") == "스키난데스"  # 낱말 경계의 ん 도 받침으로
    assert kana_to_hangul("ABC 1") == "ABC 1"


def test_reader_annotates_japanese():
    reader = Reader()
    note = reader.annotate("今日は何してたの？")
    assert note.hiragana.startswith("きょう は")
    assert note.hangul.startswith("쿄- 와")  # 조사 は 는 발음대로 '와'
    assert reader.subtitle_reading("ありがとう") == ""  # 한자가 없으면 읽는 법이 필요 없다
    assert "かんこく" in reader.subtitle_reading("韓国に行きました")
    assert has_kanji("韓国") and not has_kanji("かんこく")


# --- 인식 결과 합치기 ----------------------------------------------------------------


def test_build_transcript_marks_low_confidence():
    cfg = WhisperConfig(low_confidence_logprob=-0.6)
    good = build_transcript([("こんにちは", -0.2, 0.01)], "ja", cfg)
    assert good.text == "こんにちは" and not good.low_confidence
    bad = build_transcript([("こんにちは", -1.1, 0.01)], "ja", cfg)
    assert bad.text == "こんにちは" and bad.low_confidence
    # 긴 조각의 확신도를 더 크게 반영한다
    mixed = build_transcript([("あ", -2.0, 0.0), ("いいいいいいいいいいいいいいい", -0.1, 0.0)], "ja", cfg)
    assert not mixed.low_confidence


def test_build_transcript_drops_noise_and_prompt_echo():
    cfg = WhisperConfig(vocabulary=["田中", "ヨルシカ"])
    assert build_transcript([("ご視聴ありがとうございました", -0.2, 0.0)], "ja", cfg).text == ""
    assert build_transcript([("田中、ヨルシカ", -0.2, 0.0)], "ja", cfg).text == ""
    assert build_transcript([("田中", -0.2, 0.0)], "ja", cfg).text == "田中"
    assert build_transcript([("だれか", -0.2, 0.9)], "ja", cfg).text == ""


# --- 문장 나누기 --------------------------------------------------------------------


def run_segmenter(cfg, blocks):
    seg = Segmenter(cfg, energy_speech_prob)
    out = [u for b in blocks for u in seg.feed(b)]
    return out + list(seg.flush())


def test_long_speech_cut_at_pause_and_continued():
    """길이 제한에 걸리면 숨 고른 곳에서 나누고 나머지 말은 잃지 않는다."""
    cfg = SegmenterConfig(partial_interval_sec=0, max_utterance_sec=6.0, silence_end_sec=0.6)
    out = run_segmenter(cfg, tone(4.5) + silence(0.3) + tone(3.0) + silence(1.0))
    assert [u.final for u in out] == [True, True]
    assert [u.utterance_id for u in out] == [0, 1]
    assert 4.3 <= len(out[0].audio) / SAMPLE_RATE <= 4.7
    assert 2.9 <= len(out[1].audio) / SAMPLE_RATE <= 3.4


def test_utterance_records_silence_wait():
    cfg = SegmenterConfig(partial_interval_sec=0, silence_end_sec=0.6)
    out = run_segmenter(cfg, tone(1.0) + silence(1.0))
    assert 0.55 <= out[0].silence_wait <= 0.7


# --- 문장 끝 찾기 --------------------------------------------------------------------


@pytest.mark.parametrize(
    "text,expected",
    [
        # 문장 부호
        ("映画を見ました。とても面白かった", ["映画を見ました。", "とても面白かった"]),
        ("好きですか?私は好きです", ["好きですか?", "私は好きです"]),
        ("本当？うそでしょ！", ["本当？", "うそでしょ！"]),
        # 부호가 없어도 존댓말 끝맺음이면 나눈다
        ("映画を見ましたとても面白かったです", ["映画を見ました", "とても面白かったです"]),
        ("練習しませんか今度行きましょう", ["練習しませんか", "今度行きましょう"]),
        ("教えてくださいお願いします", ["教えてください", "お願いします"]),
        ("そうですねでも高いです", ["そうですね", "でも高いです"]),
        ("そうですでも高いです", ["そうです", "でも高いです"]),
        ("行きましたしかし遅れました", ["行きました", "しかし遅れました"]),
        # 뒤에 이어지는 말이면 나누지 않는다
        ("雨ですから行きません", ["雨ですから行きません"]),
        ("行きますけど遅れます", ["行きますけど遅れます"]),
        ("好きですが高いです", ["好きですが高いです"]),
        ("行きましたら連絡します", ["行きましたら連絡します"]),
        ("ますます好きになりました", ["ますます好きになりました"]),
        ("そうです、それでいいです", ["そうです、それでいいです"]),
        # 나눌 것이 없는 경우
        ("来月ソウルに行く予定なので", ["来月ソウルに行く予定なので"]),
        ("映画を見ました。", ["映画を見ました。"]),
        ("映画を見た人", ["映画を見た人"]),
        ("", []),
    ],
)
def test_split_sentences(text, expected):
    from jp_subtitle.sentences import split_sentences

    assert split_sentences(text) == expected


def test_first_sentence_end_needs_following_words():
    from jp_subtitle.sentences import first_sentence_end

    # 끝에 찍힌 부호는 말이 아직 이어질 수 있어 믿지 않는다
    assert first_sentence_end("映画を見ました。") is None
    assert first_sentence_end("映画を見ました。と") is None
    assert first_sentence_end("映画を見ました。とても") == len("映画を見ました。")


def test_sentence_end_time_uses_word_timestamps():
    from jp_subtitle.transcriber import sentence_end_time

    words = [("映画", 0.0, 0.5), ("を", 0.5, 0.6), ("見", 0.6, 0.8), ("ました。", 0.8, 1.2),
             ("と", 1.4, 1.5), ("ても", 1.5, 1.8)]
    assert sentence_end_time(words) == 1.2
    assert sentence_end_time(words[:4]) is None
    assert sentence_end_time([]) is None


def test_build_transcript_reports_sentence_end():
    cfg = WhisperConfig()
    words = [("見ました。", 0.0, 1.2), ("とても", 1.4, 1.9)]
    result = build_transcript([("見ました。とても", -0.2, 0.0)], "ja", cfg, words)
    assert result.sentence_end_sec == 1.2
    assert build_transcript([("見ました。とても", -0.2, 0.0)], "ja", cfg).sentence_end_sec is None


# --- 문장 길이 단계 ------------------------------------------------------------------


def test_sentence_presets_find_and_default():
    from dataclasses import replace

    from jp_subtitle.config import SENTENCE_PRESETS, find_preset, nearest_preset_index

    assert find_preset(SegmenterConfig()).key == "short"  # 기본값은 '짧게'
    for index, preset in enumerate(SENTENCE_PRESETS):
        cfg = replace(SegmenterConfig(), **preset.values())
        assert find_preset(cfg) is preset
        assert nearest_preset_index(cfg) == index
    custom = SegmenterConfig(silence_end_sec=0.6, max_utterance_sec=12.0)
    assert find_preset(custom) is None
    assert SENTENCE_PRESETS[nearest_preset_index(custom)].key == "long"
    # 짧은 쪽일수록 빨리 끊는다
    assert [p.max_utterance_sec for p in SENTENCE_PRESETS] == sorted(
        p.max_utterance_sec for p in SENTENCE_PRESETS
    )
    assert [p.silence_end_sec for p in SENTENCE_PRESETS] == sorted(
        p.silence_end_sec for p in SENTENCE_PRESETS
    )


def test_segmenter_cuts_where_sentence_ended_without_a_long_pause():
    """쉬지 않고 이어 말해도 문장 끝을 알려 주면 그 자리에서 나누고 나머지를 잃지 않는다."""
    cfg = SegmenterConfig(
        partial_interval_sec=1.0, silence_end_sec=0.6, max_utterance_sec=20.0, sentence_cut=True
    )
    seg = Segmenter(cfg, energy_speech_prob)
    out = []
    # 2초 말하고 0.1초만 쉰 뒤 이어서 2초 더 말한다
    first = tone(2.0) + silence(0.1)
    for block in first:
        out += list(seg.feed(block))
    partial = next(u for u in out if not u.final)
    assert not any(u.final for u in out)
    seg.request_cut(partial.utterance_id, 2.0)
    for block in tone(2.0):
        out += list(seg.feed(block))
    finals = [u for u in out if u.final]
    assert [u.utterance_id for u in finals] == [0]
    assert 1.8 <= len(finals[0].audio) / SAMPLE_RATE <= 2.2
    assert finals[0].silence_wait == 0.0

    rest = [u for block in silence(1.0) for u in seg.feed(block) if u.final]
    assert [u.utterance_id for u in rest] == [1]
    assert 1.9 <= len(rest[0].audio) / SAMPLE_RATE <= 2.4


def test_segmenter_ignores_a_cut_for_a_sentence_already_finished():
    cfg = SegmenterConfig(partial_interval_sec=0)
    seg = Segmenter(cfg, energy_speech_prob)
    out = [u for b in tone(1.0) + silence(1.0) for u in seg.feed(b)]
    assert len(out) == 1
    seg.request_cut(0, 0.5)
    out = [u for b in tone(1.0) + silence(1.0) for u in seg.feed(b)]
    assert [u.utterance_id for u in out] == [1]
    assert 0.9 <= len(out[0].audio) / SAMPLE_RATE <= 1.3


def test_segmenter_applies_new_settings_during_a_call():
    from dataclasses import replace

    seg = Segmenter(SegmenterConfig(partial_interval_sec=0, max_utterance_sec=20.0), energy_speech_prob)
    assert [u for b in tone(3.0) for u in seg.feed(b)] == []
    seg.cfg = replace(seg.cfg, max_utterance_sec=2.0)
    out = list(seg.feed(tone(0.1)[0]))
    assert len(out) == 1 and out[0].final


def test_hard_length_limit_keeps_the_rest_of_the_speech():
    """쉬는 곳이 없어도 앞 조각만 내보내고 뒷말은 다음 조각으로 이어 간다."""
    cfg = SegmenterConfig(partial_interval_sec=0, max_utterance_sec=3.0)
    out = run_segmenter(cfg, tone(7.0) + silence(1.0))
    total = sum(len(u.audio) for u in out) / SAMPLE_RATE
    assert total >= 6.9


# --- 파이프라인 ---------------------------------------------------------------------


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
        self.finals, self.translated_lines, self.statuses, self.levels = [], [], [], []

    def partial(self, uid, ja):
        pass

    def final(self, line):
        self.finals.append(line)

    def translated(self, line):
        self.translated_lines.append(line)

    def status(self, text):
        self.statuses.append(text)

    def level(self, speaker, value):
        self.levels.append((speaker, value))


class EchoTranslator(Translator):
    def __init__(self):
        self.calls = []

    def translate(self, ja, history):
        self.calls.append((ja, list(history)))
        return f"KO({ja})"


class ListRecorder:
    def __init__(self):
        self.rows = []

    def record(self, line, timing):
        self.rows.append((line, timing))


def sentences_pipeline(texts, translator, sink, **kwargs):
    blocks = []
    for _ in texts:
        blocks += tone(0.8) + silence(1.0)
    seg = Segmenter(SegmenterConfig(partial_interval_sec=0), energy_speech_prob)
    iterator = iter(texts)
    return Pipeline(
        FakeSource(blocks), seg, lambda a, f: next(iterator), translator, sink, **kwargs
    )


def test_backchannel_skips_translator_and_context():
    translator = EchoTranslator()
    sink = RecordingSink()
    recorder = ListRecorder()
    texts = [Transcript("今日は暑いね"), Transcript("うんうん"), Transcript("明日も暑いかな")]
    p = sentences_pipeline(texts, translator, sink, log=recorder)
    p.start()
    assert p.wait(10)
    assert [ja for ja, _ in translator.calls] == ["今日は暑いね", "明日も暑いかな"]
    # 맞장구는 번역 문맥에도 들어가지 않는다
    assert translator.calls[1][1] == [("今日は暑いね", "KO(今日は暑いね)")]
    middle = sink.translated_lines[1]
    assert middle.backchannel and middle.ko == "응"
    assert len(recorder.rows) == 3


def test_timing_is_measured():
    sink = RecordingSink()
    recorder = ListRecorder()

    class SlowTranslator(Translator):
        def translate(self, ja, history):
            time.sleep(0.05)
            return "뜻"

    p = sentences_pipeline([Transcript("こんにちは")], SlowTranslator(), sink, log=recorder)
    p.start()
    assert p.wait(10)
    _, timing = recorder.rows[0]
    assert timing.translate >= 0.04  # Windows 의 sleep 은 조금 짧을 수 있다
    assert timing.silence_wait >= 0.3
    assert timing.to_translation >= timing.to_subtitle >= timing.silence_wait
    assert 0.7 <= timing.audio_sec <= 1.1


def test_low_confidence_and_reading_reach_the_line():
    sink = RecordingSink()
    texts = [Transcript("韓国", low_confidence=True, avg_logprob=-1.2)]
    p = sentences_pipeline(texts, NoTranslator(), sink, reading=lambda text: "かんこく")
    p.start()
    assert p.wait(10)
    line = sink.finals[0]
    assert line.low_confidence and line.reading == "かんこく" and line.avg_logprob == -1.2


def test_broken_reading_does_not_stop_subtitles():
    def broken(_text):
        raise RuntimeError("dictionary")

    sink = RecordingSink()
    p = sentences_pipeline([Transcript("韓国")], NoTranslator(), sink, reading=broken)
    p.start()
    assert p.wait(10)
    assert sink.finals[0].ja == "韓国" and sink.finals[0].reading == ""


def test_broken_recorder_warns_once_and_keeps_going():
    class Broken:
        def record(self, line, timing):
            raise OSError("disk full")

    sink = RecordingSink()
    texts = [Transcript("一番目"), Transcript("二番目")]
    p = sentences_pipeline(texts, NoTranslator(), sink, log=Broken())
    p.start()
    assert p.wait(10)
    assert len(sink.translated_lines) == 2
    assert sum("저장 오류" in s for s in sink.statuses) == 1


def test_final_with_several_sentences_is_translated_one_by_one():
    translator = EchoTranslator()
    sink = RecordingSink()
    recorder = ListRecorder()
    texts = [Transcript("映画を見ました。とても面白かったです。また行きます")]
    p = sentences_pipeline(texts, translator, sink, log=recorder, reading=lambda t: f"R({t})")
    p.start()
    assert p.wait(10)
    assert [ja for ja, _ in translator.calls] == [
        "映画を見ました。", "とても面白かったです。", "また行きます",
    ]
    # 앞 문장이 뒤 문장의 번역 문맥이 된다
    assert translator.calls[2][1][-1] == ("とても面白かったです。", "KO(とても面白かったです。)")
    uids = [line.uid for line in sink.finals]
    assert uids[0] == 0 and len(set(uids)) == 3  # 첫 문장은 임시 자막 줄을 이어받는다
    assert sink.finals[1].reading == "R(とても面白かったです。)"
    assert len(recorder.rows) == 3


def test_long_setting_keeps_final_as_one_line():
    translator = EchoTranslator()
    sink = RecordingSink()
    text = "映画を見ました。とても面白かったです"
    seg = Segmenter(
        SegmenterConfig(partial_interval_sec=0, sentence_cut=False), energy_speech_prob
    )
    p = Pipeline(
        FakeSource(tone(0.8) + silence(1.0)), seg, lambda a, f: Transcript(text), translator, sink
    )
    p.start()
    assert p.wait(10)
    assert [line.ja for line in sink.finals] == [text]


def test_several_backchannels_stay_one_line():
    sink = RecordingSink()
    p = sentences_pipeline([Transcript("はい。うん。そうですね")], NoTranslator(), sink)
    p.start()
    assert p.wait(10)
    assert len(sink.finals) == 1 and sink.finals[0].backchannel


def test_partial_with_sentence_end_cuts_right_away():
    """임시 자막에서 문장 끝이 보이면 쉬지 않아도 앞 문장을 확정한다."""
    seen_partial = threading.Event()

    class Source(FakeSource):
        def blocks(self):
            for index, block in enumerate(self._blocks):
                if index == int(1.3 / BLOCK_SEC):
                    assert seen_partial.wait(10)  # 인식이 끝난 뒤 다음 블록을 보낸다
                yield block

    sink = RecordingSink()
    finals = iter(["映画を見ました。", "続き"])
    first_partial = threading.Event()

    def transcribe(audio, final):
        if final:
            return Transcript(next(finals))
        if first_partial.is_set():
            return Transcript("続き")  # 잘린 뒤의 임시 자막에는 문장 끝이 아직 없다
        first_partial.set()
        seen_partial.set()
        return Transcript("映画を見ました。とても", sentence_end_sec=1.0)

    seg = Segmenter(
        SegmenterConfig(partial_interval_sec=1.2, silence_end_sec=0.6, max_utterance_sec=20.0),
        energy_speech_prob,
    )
    p = Pipeline(Source(tone(3.0) + silence(1.0)), seg, transcribe, NoTranslator(), sink)
    p.start()
    assert p.wait(10)
    assert [line.ja for line in sink.finals] == ["映画を見ました。", "続き"]
    assert [line.uid for line in sink.finals] == [0, 1]


def test_asr_queue_prefers_other_person():
    q = _AsrQueue()
    audio = np.zeros(1, dtype=np.float32)
    q.put(Utterance(audio, True, 1_000_000, ME))
    q.put(Utterance(audio, True, 0, THEM))
    q.put(Utterance(audio, False, 1, THEM))
    q.close()
    assert [u.utterance_id for u in iter(q.get, None)] == [0, 1, 1_000_000]


def test_mic_lines_are_labelled_and_used_as_context():
    translator = EchoTranslator()
    mine_done = threading.Event()

    class Sink(RecordingSink):
        def translated(self, line):
            super().translated(line)
            if line.speaker == ME:
                mine_done.set()

    class AfterMine(FakeSource):
        """내 말이 끝난 뒤에야 상대방이 말한다."""

        def blocks(self):
            assert mine_done.wait(10)
            yield from super().blocks()

    sink = Sink()
    me_blocks = tone(0.8) + silence(1.0)
    p = Pipeline(
        AfterMine(tone(0.8) + silence(1.0)),
        Segmenter(SegmenterConfig(partial_interval_sec=0), energy_speech_prob),
        lambda a, f: Transcript("そうなんだ、いつ？"),
        translator,
        sink,
        mic=MicChannel(
            FakeSource(me_blocks),
            Segmenter(SegmenterConfig(partial_interval_sec=0), energy_speech_prob, ME, 1_000_000),
            lambda a, f: Transcript("韓国に行きます"),
        ),
    )
    p.start()
    assert p.wait(10)
    mine = next(line for line in sink.finals if line.speaker == ME)
    assert mine.uid == 1_000_000 and mine.ko == ""
    # 상대방 말을 번역할 때 내 말이 문맥에 들어간다
    assert translator.calls == [
        ("そうなんだ、いつ？", [(ME_PREFIX + "韓国に行きます", "")])
    ]


def test_capture_failure_is_reported():
    class Broken:
        def blocks(self):
            raise RuntimeError("장치 없음")
            yield

        def stop(self):
            pass

    sink = RecordingSink()
    p = Pipeline(
        Broken(),
        Segmenter(SegmenterConfig(), energy_speech_prob),
        lambda a, f: Transcript(),
        NoTranslator(),
        sink,
    )
    p.start()
    assert p.wait(10)
    assert p.capture_error == "장치 없음"


def test_mic_failure_keeps_subtitles_running():
    class BrokenMic:
        def blocks(self):
            raise RuntimeError("no mic")
            yield

        def stop(self):
            pass

    sink = RecordingSink()
    p = Pipeline(
        FakeSource(tone(0.8) + silence(1.0)),
        Segmenter(SegmenterConfig(partial_interval_sec=0), energy_speech_prob),
        lambda a, f: Transcript("こんにちは"),
        NoTranslator(),
        sink,
        mic=MicChannel(BrokenMic(), Segmenter(SegmenterConfig(), energy_speech_prob), None),
    )
    p.start()
    assert p.wait(10)
    assert any("마이크" in s for s in sink.statuses)
    assert sink.finals[0].ja == "こんにちは"


def test_mic_source_reads_blocks(monkeypatch):
    import jp_subtitle.audio_capture as ac

    rate = 48000
    tone_48k = (0.5 * np.sin(2 * np.pi * 300 * np.arange(rate) / rate)).astype(np.float32)

    class FakeStream:
        pos = 0

        def get_read_available(self):
            return min(480, len(tone_48k) - self.pos)

        def read(self, n, exception_on_overflow=True):
            chunk = tone_48k[self.pos : self.pos + n]
            self.pos += n
            return chunk.tobytes()

        def stop_stream(self):
            pass

        def close(self):
            pass

    stream = FakeStream()
    mic = {
        "index": 1, "hostApi": 0, "name": "USB Mic", "maxInputChannels": 1,
        "defaultSampleRate": rate, "isLoopbackDevice": False,
    }

    class FakePyAudio:
        def get_host_api_info_by_type(self, _kind):
            return {"index": 0, "defaultInputDevice": 1}

        def get_device_count(self):
            return 2

        def get_device_info_by_index(self, i):
            return mic if i == 1 else {**mic, "index": 0, "name": "Speakers [Loopback]", "isLoopbackDevice": True}

        def open(self, **kwargs):
            assert kwargs["input_device_index"] == 1
            return stream

        def terminate(self):
            pass

    fake = types.SimpleNamespace(PyAudio=FakePyAudio, paFloat32=1, paWASAPI=13)
    monkeypatch.setitem(sys.modules, "pyaudiowpatch", fake)
    monkeypatch.setattr(sys, "platform", "win32")

    source = ac.MicSource("usb")
    blocks = []
    for block in source.blocks():
        blocks.append(block)
        if stream.pos >= len(tone_48k):
            source.stop()
    assert len(blocks) * BLOCK_SAMPLES >= SAMPLE_RATE - 2 * BLOCK_SAMPLES

    missing = ac.MicSource("없는 장치")
    with pytest.raises(RuntimeError, match="없는 장치"):
        list(missing.blocks())


# --- 설정 저장 ---------------------------------------------------------------------


def test_save_config_keeps_comments_and_updates_values(tmp_path):
    path = tmp_path / "config.toml"
    path.write_text(
        "﻿# 맨 위 주석\n[translator]\n"
        'engine = "deepl"  # 엔진 설명\n'
        'deepl_api_key = ""\n'
        "\n[whisper]\n"
        'vocabulary = [\n  "田中",\n  "佐藤",\n]\n'
        "beam_size = 5 # 빔\n",
        encoding="utf-8",
    )
    save_config(
        path,
        {
            "translator": {"engine": "claude", "deepl_api_key": 'k"ey:fx', "anthropic_api_key": "abc"},
            "whisper": {"vocabulary": ["ヨルシカ"], "beam_size": 3},
            "overlay": {"enabled": False, "opacity": 0.5},
        },
    )
    text = path.read_text(encoding="utf-8")
    assert "# 맨 위 주석" in text and "# 엔진 설명" in text and "# 빔" in text
    cfg = load_config(path)
    assert cfg.translator.engine == "claude"
    assert cfg.translator.deepl_api_key == 'k"ey:fx'
    assert cfg.translator.anthropic_api_key == "abc"
    assert cfg.whisper.vocabulary == ["ヨルシカ"]
    assert cfg.whisper.beam_size == 3
    assert cfg.overlay.enabled is False and cfg.overlay.opacity == 0.5


def test_old_default_sentence_values_become_new_defaults(tmp_path):
    path = tmp_path / "config.toml"
    old = "[segmenter]\nvad_threshold = 0.7\nsilence_end_sec = 0.6\nmax_utterance_sec = 12.0\npartial_interval_sec = 1.5\n"
    path.write_text(old, encoding="utf-8")
    cfg = load_config(path).segmenter
    assert (cfg.silence_end_sec, cfg.max_utterance_sec, cfg.partial_interval_sec) == (0.35, 6.0, 1.0)
    assert cfg.sentence_cut is True
    assert cfg.vad_threshold == 0.7  # 다른 값은 건드리지 않는다
    assert path.read_text(encoding="utf-8") == old  # 파일도 바꾸지 않는다

    # 직접 고친 값은 그대로 둔다
    path.write_text(old.replace("0.6", "0.8"), encoding="utf-8")
    cfg = load_config(path).segmenter
    assert (cfg.silence_end_sec, cfg.max_utterance_sec) == (0.8, 12.0)

    # '길게' 를 고른 뒤에는 새 항목이 파일에 있으므로 되돌리지 않는다
    path.write_text(old + "sentence_cut = true\n", encoding="utf-8")
    assert load_config(path).segmenter.max_utterance_sec == 12.0


def test_save_config_creates_file_and_rejects_bad_values(tmp_path):
    path = tmp_path / "config.toml"
    save_config(path, {"segmenter": {"silence_end_sec": 0.8}})
    assert load_config(path).segmenter.silence_end_sec == 0.8
    before = path.read_text(encoding="utf-8")
    with pytest.raises(ValueError):
        save_config(path, {"segmenter": {"없는항목": 1}})
    assert path.read_text(encoding="utf-8") == before


# --- 기록과 복습 카드 ---------------------------------------------------------------


@pytest.fixture
def store(tmp_path):
    db = HistoryStore(tmp_path / "logs" / "history.db")
    yield db
    db.close()


def _timing(**kwargs):
    return Timing(audio_sec=1.0, silence_wait=0.6, asr=0.3, translate=0.2, **kwargs)


def test_session_lifecycle(store):
    empty = store.create_session()
    store.end_session(empty)
    assert store.list_sessions() == []  # 아무 말도 없던 통화는 남기지 않는다

    sid = store.create_session()
    store.add_line(sid, Line(0, "こんにちは", ko="안녕하세요", reading=""), _timing())
    store.add_line(sid, Line(1, "うん", ko="응", backchannel=True), _timing())
    store.add_line(sid, Line(2, "元気です", speaker=ME), _timing())
    store.end_session(sid)

    info = store.list_sessions()[0]
    assert info.line_count == 3 and info.preview == "안녕하세요" and not info.has_review
    lines = store.lines(sid)
    assert [line.speaker for line in lines] == [THEM, THEM, ME]
    assert lines[1].backchannel and lines[1].ko == "응"

    store.save_review(sid, {"summary": "요약"})
    assert store.review(sid) == {"summary": "요약"}
    assert store.list_sessions()[0].has_review

    store.add_card("こんにちは", "안녕하세요", session_id=sid)
    store.delete_session(sid)
    assert store.list_sessions() == [] and store.lines(sid) == []
    assert len(store.all_cards()) == 1  # 카드는 기록을 지워도 남는다


def test_card_intervals(store):
    today = date(2026, 11, 5)
    assert store.add_card("文", "문장")
    assert not store.add_card("文", "다른 뜻")  # 같은 문장은 한 장만
    card = store.all_cards()[0]
    assert store.due_cards(today + timedelta(days=9999))[0].id == card.id

    store.grade_card(card.id, True, today)
    assert store.due_cards(today) == []
    assert store.due_cards(today + timedelta(days=1))[0].box == 1

    store.grade_card(card.id, True, today + timedelta(days=1))
    assert store.due_cards(today + timedelta(days=3)) == []
    assert store.due_cards(today + timedelta(days=4))[0].box == 2

    store.grade_card(card.id, False, today + timedelta(days=4))
    again = store.due_cards(today + timedelta(days=4))[0]
    assert again.box == 0  # 몰랐으면 처음부터

    store.delete_card(card.id)
    assert store.all_cards() == []


def test_latency_stats_ignore_backchannel_and_me(store):
    sid = store.create_session()
    store.add_line(sid, Line(0, "一"), Timing(silence_wait=0.6, asr=0.2, translate=0.1))
    store.add_line(sid, Line(1, "二"), Timing(silence_wait=0.6, asr=0.4, translate=0.3))
    store.add_line(sid, Line(2, "うん", backchannel=True), Timing(silence_wait=9, asr=9))
    store.add_line(sid, Line(3, "三", speaker=ME), Timing(silence_wait=9, asr=9))
    stats = {s.name: s for s in store.latency_stats(sid)}
    assert stats["음성인식"].count == 2
    assert stats["음성인식"].typical == pytest.approx(0.3)
    assert stats["한국어까지"].typical == pytest.approx(1.1)
    assert stats["음성인식"].slow == pytest.approx(0.38)
    assert store.latency_stats(sid + 1) == []


def test_export_anki_csv(tmp_path):
    path = tmp_path / "cards.csv"
    assert export_anki_csv(path, [("文,1", "ぶん", "문장", ""), ("二", "に", "둘", "메모")]) == 2
    with path.open(encoding="utf-8-sig", newline="") as f:
        assert list(csv.reader(f))[0] == ["文,1", "ぶん", "문장", ""]


def test_session_recorder_writes_both(store, tmp_path):
    from jp_subtitle.pipeline import ConversationLog

    sid = store.create_session()
    log = ConversationLog(tmp_path / "txt")
    recorder = SessionRecorder(log, store, sid)
    recorder.record(Line(0, "聞こえますか", ko="들려요?", low_confidence=True), _timing())
    recorder.record(Line(1, "はい", speaker=ME), _timing())
    text = log.path.read_text(encoding="utf-8")
    assert "(?) 聞こえますか" in text and "들려요?" in text and "(나) はい" in text
    assert len(store.lines(sid)) == 2


# --- 답변 도우미 --------------------------------------------------------------------


@pytest.fixture(autouse=True)
def no_api_keys_in_environment(monkeypatch):
    for name in ("ANTHROPIC_API_KEY", "GEMINI_API_KEY", "GOOGLE_API_KEY"):
        monkeypatch.delenv(name, raising=False)


def _assistant(claude="", deepl="", gemini="", **options):
    from jp_subtitle.assistant import Assistant
    from jp_subtitle.config import TranslatorConfig

    cfg = TranslatorConfig(
        anthropic_api_key=claude, deepl_api_key=deepl, gemini_api_key=gemini, **options
    )
    return Assistant(cfg, Reader())


def _fake_claude(assistant, monkeypatch, payload, stop_reason="end_turn"):
    captured = {}

    def create(**kwargs):
        captured.update(kwargs)
        return SimpleNamespace(
            stop_reason=stop_reason,
            content=[SimpleNamespace(type="thinking"), SimpleNamespace(type="text", text=json.dumps(payload))],
        )

    messages = SimpleNamespace(create=create)
    client = SimpleNamespace(
        with_options=lambda timeout: SimpleNamespace(beta=SimpleNamespace(messages=messages))
    )
    assistant._claude = client
    return captured


def test_reply_with_claude(monkeypatch):
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    assistant = _assistant(claude="key")
    captured = _fake_claude(
        assistant, monkeypatch, {"casual": "今日は暇だよ", "polite": "今日は暇です"}
    )
    recent = [Line(0, "今日は何するの？", ko="오늘 뭐 해?"), Line(1, "韓国語の勉強", speaker=ME)]
    options = assistant.reply("오늘 한가해", recent)
    assert [o.label for o in options] == ["반말", "존댓말"]
    assert options[1].ja == "今日は暇です"
    assert options[1].reading.startswith("きょう は")
    assert options[1].hangul.startswith("쿄-")
    prompt = captured["messages"][0]["content"]
    assert "[상대] 今日は何するの？" in prompt and "[나] 韓国語の勉強" in prompt
    assert "오늘 한가해" in prompt
    assert captured["model"] == "claude-opus-5-5"
    assert captured["output_config"]["format"]["type"] == "json_schema"
    assert captured["fallbacks"] == "default"


def test_reply_with_deepl_only(monkeypatch):
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    assistant = _assistant(deepl="key:fx")
    calls = []

    def translate_text(text, **kwargs):
        calls.append(kwargs)
        return SimpleNamespace(text="暇です" if kwargs["formality"] == "prefer_more" else "暇だよ")

    assistant._deepl = SimpleNamespace(translate_text=translate_text)
    options = assistant.reply("한가해")
    assert [o.ja for o in options] == ["暇だよ", "暇です"]
    assert calls[0]["source_lang"] == "KO" and calls[0]["target_lang"] == "JA"
    assert assistant.can_reply and not assistant.has_claude


def test_features_explain_missing_keys(monkeypatch):
    from jp_subtitle.assistant import AssistantError

    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    assistant = _assistant()
    assert not assistant.can_reply
    with pytest.raises(AssistantError, match="API 키"):
        assistant.reply("안녕")
    with pytest.raises(AssistantError, match="Claude 또는 Gemini 키"):
        assistant.suggest([Line(0, "やあ")])
    with pytest.raises(AssistantError, match="Claude 또는 Gemini 키"):
        assistant.review([Line(0, "やあ")])


def test_suggest_and_review_with_claude(monkeypatch):
    from jp_subtitle.assistant import AssistantError

    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    assistant = _assistant(claude="key")
    with pytest.raises(AssistantError, match="대화 내용"):
        assistant.suggest([])
    _fake_claude(
        assistant,
        monkeypatch,
        {"options": [{"ja": f"案{i}", "ko": f"안{i}"} for i in range(5)]},
    )
    options = assistant.suggest([Line(0, "やあ")])
    assert [o.ja for o in options] == ["案0", "案1", "案2"]
    assert options[0].ko == "안0" and options[0].reading

    review = {"summary": "s", "expressions": [], "corrections": [], "next_topics": ["t"]}
    captured = _fake_claude(assistant, monkeypatch, review)
    assert assistant.review([Line(0, "やあ")]) == review
    assert captured["output_config"]["effort"] == "medium"


def test_claude_refusal_and_truncation_are_explained(monkeypatch):
    from jp_subtitle.assistant import AssistantError

    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    assistant = _assistant(claude="key")
    _fake_claude(assistant, monkeypatch, {}, stop_reason="refusal")
    with pytest.raises(AssistantError, match="답하지"):
        assistant.reply("안녕")
    _fake_claude(assistant, monkeypatch, {}, stop_reason="max_tokens")
    with pytest.raises(AssistantError, match="끝내지"):
        assistant.reply("안녕")


def test_api_errors_become_friendly_messages(monkeypatch):
    import anthropic
    import httpx2 as httpx

    from jp_subtitle.assistant import AssistantError

    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    assistant = _assistant(claude="key")
    request = httpx.Request("POST", "https://api.anthropic.com/v1/messages")

    def fail(error):
        def create(**_kwargs):
            raise error

        messages = SimpleNamespace(create=create)
        assistant._claude = SimpleNamespace(
            with_options=lambda timeout: SimpleNamespace(beta=SimpleNamespace(messages=messages))
        )

    fail(anthropic.AuthenticationError("bad", response=httpx.Response(401, request=request), body=None))
    with pytest.raises(AssistantError, match="키를 확인"):
        assistant.reply("안녕")
    fail(anthropic.RateLimitError("slow", response=httpx.Response(429, request=request), body=None))
    with pytest.raises(AssistantError, match="잠시 뒤"):
        assistant.suggest([Line(0, "やあ")])
    fail(anthropic.APIConnectionError(request=request))
    with pytest.raises(AssistantError, match="인터넷"):
        assistant.review([Line(0, "やあ")])


def test_claude_translator_marks_my_lines(monkeypatch):
    from jp_subtitle.translators.claude_tr import ClaudeTranslator

    tr = ClaudeTranslator("claude-opus-5-5", api_key="test")
    captured = {}

    def fake_create(**kwargs):
        captured.update(kwargs)
        return SimpleNamespace(stop_reason="end_turn", content=[SimpleNamespace(type="text", text="네")])

    monkeypatch.setattr(tr._client.beta.messages, "create", fake_create)
    tr.translate("はい", [(ME_PREFIX + "元気？", ""), ("うん", "응")])
    content = captured["messages"][0]["content"]
    assert f"- {ME_PREFIX}元気？\n" in content and "- うん → 응" in content


# --- Gemini (진짜 SDK 를 PC 안의 가짜 서버에 연결한다. 외부로 나가지 않는다) ---------------


@pytest.fixture
def gemini_server(monkeypatch):
    from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

    state = SimpleNamespace(
        requests=[], http_status=200, status="completed", text="안녕하세요", delay=0.0, error="오류"
    )

    class Handler(BaseHTTPRequestHandler):
        def do_POST(self):
            length = int(self.headers.get("content-length", 0))
            state.requests.append(json.loads(self.rfile.read(length) or b"{}"))
            time.sleep(state.delay)
            if state.http_status == 200:
                body = {"id": "i1", "status": state.status, "model": "m"}
                if state.text:
                    body["steps"] = [
                        {"type": "model_output", "content": [{"type": "text", "text": state.text}]}
                    ]
            else:
                body = {"error": {"code": state.http_status, "message": state.error}}
            data = json.dumps(body).encode()
            self.send_response(state.http_status)
            self.send_header("content-type", "application/json")
            self.send_header("content-length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def log_message(self, *args):
            pass

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    server.daemon_threads = True
    threading.Thread(target=server.serve_forever, daemon=True).start()
    monkeypatch.setenv("GOOGLE_GEMINI_BASE_URL", f"http://127.0.0.1:{server.server_port}")
    yield state
    server.shutdown()
    server.server_close()


def test_gemini_translator_sends_private_low_latency_request(gemini_server):
    from jp_subtitle.translators.gemini_tr import GeminiTranslator

    tr = GeminiTranslator("gemini-3.5-flash-lite", "key")
    assert tr.translate("はい", [(ME_PREFIX + "元気？", ""), ("うん", "응")]) == "안녕하세요"
    request = gemini_server.requests[-1]
    assert request["store"] is False  # 통화 내용을 Google 서버에 저장하지 않는다
    assert request["model"] == "gemini-3.5-flash-lite"
    assert request["generation_config"]["thinking_level"] == "minimal"
    assert "번역" in request["system_instruction"]
    assert f"- {ME_PREFIX}元気？\n" in request["input"] and request["input"].endswith("はい")
    assert "response_format" not in request


def test_gemini_translator_empty_answer_is_not_an_error(gemini_server):
    from jp_subtitle.translators.gemini_tr import GeminiTranslator

    gemini_server.text = ""
    gemini_server.status = "failed"
    assert GeminiTranslator("gemini-3.5-flash-lite", "key").translate("はい", []) == ""


@pytest.mark.parametrize(
    "http_status,error,expected",
    [
        (401, "unauthenticated", "키를 확인"),
        (400, "API key not valid. Please pass a valid API key.", "키를 확인"),
        (403, "permission denied", "키를 확인"),
        (404, "model not found", "모델"),
        (429, "quota exceeded", "한도"),
        (400, "bad schema", "거절"),
        (500, "internal", "서버 오류"),
    ],
)
def test_gemini_errors_become_friendly_messages(gemini_server, http_status, error, expected):
    from jp_subtitle.translators.gemini_tr import GeminiTranslator

    gemini_server.http_status = http_status
    gemini_server.error = error
    tr = GeminiTranslator("gemini-3.5-flash-lite", "key")
    with pytest.raises(RuntimeError, match=expected):
        tr.translate("はい", [])


def test_gemini_timeout_is_in_seconds_and_slow_answers_fail_cleanly(gemini_server):
    from jp_subtitle.translators.gemini_tr import generate, gemini_error_message, make_client

    client = make_client("key")
    gemini_server.delay = 0.3
    result = generate(client, "gemini-3.8-flash", "s", "p", "low", 5.0)  # 초 단위면 통과한다
    assert result.output_text == "안녕하세요"

    gemini_server.delay = 1.5
    with pytest.raises(Exception) as caught:
        generate(client, "gemini-3.8-flash", "s", "p", "low", 0.2)
    assert "연결하지 못했어요" in gemini_error_message(caught.value)


def test_gemini_thinking_level_matches_what_each_model_accepts():
    from jp_subtitle.translators.gemini_tr import fastest_thinking

    assert fastest_thinking("gemini-3.5-flash-lite") == "minimal"
    assert fastest_thinking("gemini-3.6-flash") == "minimal"
    assert fastest_thinking("gemini-3.8-flash") == "low"
    assert fastest_thinking("gemini-4-argon") == "low"
    assert fastest_thinking("다른-모델") == "low"


def test_translator_factory_makes_gemini():
    from jp_subtitle.config import TranslatorConfig
    from jp_subtitle.translators import create_translator
    from jp_subtitle.translators.gemini_tr import GeminiTranslator

    with pytest.raises(ValueError, match="gemini_api_key"):
        create_translator(TranslatorConfig(engine="gemini"))
    made = create_translator(TranslatorConfig(engine="gemini", gemini_api_key="key"))
    assert isinstance(made, GeminiTranslator)


def test_reply_with_gemini(gemini_server):
    gemini_server.text = json.dumps({"casual": "今日は暇だよ", "polite": "今日は暇です"})
    assistant = _assistant(gemini="key")
    assert assistant.ai_name == "gemini" and assistant.ai_label == "Gemini"
    assert assistant.can_reply and assistant.has_ai and not assistant.has_claude
    recent = [Line(0, "今日は何するの？", ko="오늘 뭐 해?"), Line(1, "勉強", speaker=ME)]
    options = assistant.reply("오늘 한가해", recent)
    assert [option.label for option in options] == ["반말", "존댓말"]
    assert options[1].ja == "今日は暇です"
    assert options[1].reading.startswith("きょう は") and options[1].hangul.startswith("쿄-")

    request = gemini_server.requests[-1]
    assert request["store"] is False
    assert request["model"] == "gemini-3.5-flash-lite"
    assert request["generation_config"]["thinking_level"] == "minimal"
    assert "[나] 勉強" in request["input"] and "오늘 한가해" in request["input"]
    assert request["response_format"]["mime_type"] == "application/json"
    assert request["response_format"]["schema"]["required"] == ["casual", "polite"]


def test_suggest_and_review_with_gemini(gemini_server):
    assistant = _assistant(gemini="key", gemini_model="gemini-3.8-flash")
    gemini_server.text = json.dumps({"options": [{"ja": f"案{i}", "ko": f"안{i}"} for i in range(5)]})
    options = assistant.suggest([Line(0, "やあ")])
    assert [option.ja for option in options] == ["案0", "案1", "案2"]
    assert gemini_server.requests[-1]["generation_config"]["thinking_level"] == "low"

    review = {"summary": "s", "expressions": [], "corrections": [], "next_topics": ["t"]}
    gemini_server.text = json.dumps(review)
    assert assistant.review([Line(0, "やあ")]) == review
    assert gemini_server.requests[-1]["generation_config"]["thinking_level"] == "medium"
    assert gemini_server.requests[-1]["store"] is False


@pytest.mark.parametrize(
    "status,text,expected",
    [
        ("incomplete", "{", "끝내지"),
        ("failed", "", "답하지"),
        ("completed", "", "답하지"),
        ("completed", "JSON 아님", "형식"),
    ],
)
def test_gemini_unusable_answers_are_explained(gemini_server, status, text, expected):
    from jp_subtitle.assistant import AssistantError

    gemini_server.status = status
    gemini_server.text = text
    with pytest.raises(AssistantError, match=expected):
        _assistant(gemini="key").reply("안녕")


def test_gemini_api_error_reaches_assistant_user_friendly(gemini_server):
    from jp_subtitle.assistant import AssistantError

    gemini_server.http_status = 400
    gemini_server.error = "API key not valid."
    with pytest.raises(AssistantError, match="Gemini API 키를 확인"):
        _assistant(gemini="key").reply("안녕")


@pytest.mark.parametrize(
    "engine,ai_engine,claude,gemini,expected",
    [
        ("deepl", "auto", "", "", None),
        ("deepl", "auto", "c", "", "claude"),
        ("deepl", "auto", "", "g", "gemini"),
        ("deepl", "auto", "c", "g", "claude"),
        ("gemini", "auto", "c", "g", "gemini"),  # 자막에 쓰는 AI 를 먼저 고른다
        ("claude", "auto", "c", "g", "claude"),
        ("gemini", "auto", "c", "", "claude"),   # 그 AI 키가 없으면 있는 쪽을 쓴다
        ("deepl", "gemini", "c", "g", "gemini"),
        ("deepl", "claude", "c", "g", "claude"),
        ("deepl", "gemini", "c", "", None),      # 직접 고른 AI 의 키가 없으면 쓰지 않는다
    ],
)
def test_ai_selection(engine, ai_engine, claude, gemini, expected):
    assistant = _assistant(claude=claude, gemini=gemini, engine=engine, ai_engine=ai_engine)
    assert assistant.ai_name == expected


def test_gemini_key_from_environment(monkeypatch):
    monkeypatch.setenv("GEMINI_API_KEY", "from-env")
    assert _assistant().ai_name == "gemini"


# --- 화면 ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def qapp():
    from PySide6.QtWidgets import QApplication

    from jp_subtitle.ui.theme import STYLESHEET

    app = QApplication.instance() or QApplication([])
    app.setStyleSheet(STYLESHEET)
    return app


def pump(qapp, until=None, timeout=15.0):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        qapp.processEvents()
        if until is not None and until():
            return True
        time.sleep(0.01)
    return until is None


@pytest.fixture
def app_parts(qapp, tmp_path, monkeypatch):
    from PySide6.QtCore import QSettings

    from jp_subtitle.ui.app import build_app

    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    QSettings.setPath(QSettings.Format.NativeFormat, QSettings.Scope.UserScope, str(tmp_path))
    config_path = tmp_path / "config.toml"
    cfg = Config()
    cfg.log.directory = "logs"
    ctx, window = build_app(cfg, config_path, tmp_path / "logs")
    yield ctx, window
    window.hotkeys.unregister_all()
    ctx.engine.shutdown()
    window.close()
    ctx.overlay.close()
    ctx.store.close()


def test_overlay_keeps_sentences_through_backchannels(qapp):
    from jp_subtitle.config import OverlayConfig
    from jp_subtitle.overlay import SinkBridge, SubtitleOverlay

    bridge = SinkBridge()
    overlay = SubtitleOverlay(OverlayConfig(history_lines=1), bridge)
    bridge.final(Line(0, "昨日映画を見た", ko="어제 영화 봤어", reading="きのう"))
    for i in range(1, 6):
        bridge.partial(i, "うん")
        bridge.final(Line(i, "うん", ko="응", backchannel=True))
    pump(qapp, timeout=0.05)
    assert list(overlay._lines) == [0]  # 맞장구가 앞 문장을 밀어내지 않는다
    text = overlay.label.text()
    assert "昨日映画を見た" in text and "きのう" in text and "응" in text

    bridge.final(Line(6, "面白かった", low_confidence=True))
    pump(qapp, timeout=0.05)
    text = overlay.label.text()
    assert "잘못 들었을" in text and "うん" not in text  # 새 문장이 오면 맞장구는 사라진다

    bridge.final(Line(1_000_000, "私の言葉", speaker=ME))
    pump(qapp, timeout=0.05)
    assert "私の言葉" not in overlay.label.text()  # 내 말은 자막 창에 띄우지 않는다

    overlay.show_hint("もう一度", "もういちど", "한 번 더")
    assert "もういちど" in overlay.label.text()
    overlay.clear_hint()
    assert "もういちど" not in overlay.label.text()

    # 번역 키가 없으면 "번역 중" 이 계속 남거나 안내 문구가 줄마다 뜨지 않는다
    bridge.translated(Line(6, "面白かった"))
    line = Line(7, "翻訳なし")
    bridge.final(line)
    pump(qapp, timeout=0.05)
    assert "번역 중" in overlay.label.text()
    bridge.translated(line)
    pump(qapp, timeout=0.05)
    assert "번역 중" not in overlay.label.text() and "번역 없음" not in overlay.label.text()
    overlay.close()


def test_reading_toggle_applies_to_existing_lines(app_parts, qapp):
    ctx, window = app_parts
    live = window.live
    line = Line(0, "韓国", ko="한국", reading="かんこく")
    ctx.bridge.final(line)
    ctx.bridge.translated(line)
    pump(qapp, timeout=0.05)

    def texts():
        labels = live.view._rows[0].findChildren(type(live.status_label))
        return [label.text() for label in labels]

    assert "かんこく" in texts()
    live.reading_check.setChecked(False)
    pump(qapp, timeout=0.05)
    assert "かんこく" not in texts() and "한국" in texts()
    live.reading_check.setChecked(True)
    pump(qapp, timeout=0.05)
    assert "かんこく" in texts()


def test_history_rereads_a_call_in_progress(app_parts, qapp):
    ctx, window = app_parts
    sid = ctx.store.create_session()
    ctx.store.add_line(sid, Line(0, "一番目", ko="첫째"), _timing())
    window.go_to(1)
    assert len(window.history.view._rows) == 1
    ctx.store.add_line(sid, Line(1, "二番目", ko="둘째"), _timing())
    window.go_to(0)
    window.go_to(1)
    assert len(window.history.view._rows) == 2


def test_engine_applies_settings_without_needless_reload(qapp, tmp_path, monkeypatch):
    import jp_subtitle.engine as engine_mod
    import jp_subtitle.transcriber as transcriber_mod
    from jp_subtitle.engine import FAILED, READY, CallEngine
    from jp_subtitle.overlay import SinkBridge

    loads = []
    updates = []

    class Stub:
        notice = ""

        def __init__(self, cfg):
            loads.append(cfg.model)
            if cfg.model == "broken":
                raise RuntimeError("no model")

        def update_settings(self, cfg):
            updates.append(cfg)

    monkeypatch.setattr(transcriber_mod, "create_transcriber", Stub)
    store = HistoryStore(tmp_path / "history.db")
    engine = CallEngine(Config(), tmp_path / "config.toml", tmp_path, SinkBridge(), store)
    try:
        engine.prepare()
        assert pump(qapp, lambda: engine.state == READY and loads == ["large-v3-turbo"])

        # 단어 목록과 정확도는 모델을 다시 불러오지 않고 바로 적용된다
        changed = Config()
        changed.whisper.vocabulary = ["田中"]
        changed.whisper.beam_size = 1
        engine.apply_config(changed)
        assert loads == ["large-v3-turbo"]
        assert updates[-1].vocabulary == ["田中"] and updates[-1].beam_size == 1

        # 모델이 바뀌면 다시 불러온다
        broken = Config()
        broken.whisper.model = "broken"
        engine.apply_config(broken)
        assert pump(qapp, lambda: engine.state == FAILED)

        # 실패한 뒤 설정을 고치면 다시 시도한다
        engine.apply_config(Config())
        assert pump(qapp, lambda: engine.state == READY and loads[-1] == "large-v3-turbo")
    finally:
        store.close()


def test_hiding_overlay_from_its_menu_unticks_main_window(app_parts, qapp):
    ctx, window = app_parts
    assert window.live.overlay_check.isChecked()
    ctx.overlay._hide_by_user()
    assert not window.live.overlay_check.isChecked()


def test_backchannel_is_never_flagged_as_misheard():
    sink = RecordingSink()
    texts = [Transcript("うんうん", low_confidence=True)]
    p = sentences_pipeline(texts, NoTranslator(), sink)
    p.start()
    assert p.wait(10)
    assert sink.finals[0].backchannel and not sink.finals[0].low_confidence


def test_live_page_flow(app_parts, qapp):
    ctx, window = app_parts
    live = window.live
    ctx.bridge.partial(0, "昨日")
    pump(qapp, timeout=0.05)
    assert live.view._rows[0].line is None

    line = Line(0, "昨日映画を見た", reading="きのう えいが", low_confidence=True)
    ctx.bridge.final(line)
    line.ko = "어제 영화 봤어"
    ctx.bridge.translated(line)
    ctx.bridge.final(Line(1, "うん", ko="응", backchannel=True))
    ctx.bridge.final(Line(2, "韓国語です", speaker=ME))
    ctx.bridge.partial(3, "ノイズ")
    ctx.bridge.partial(3, "")
    pump(qapp, timeout=0.05)
    assert sorted(live.view._rows) == [0, 1, 2]
    assert live.view._rows[0]._ko.text() == "어제 영화 봤어"

    # 복습 추가
    assert live._save_line(live.view._rows[0].line)
    assert ctx.store.all_cards()[0].ko == "어제 영화 봤어"

    # 키가 없으면 안내가 보이고 버튼은 꺼진다
    assert not live.reply_button.isEnabled() and not live.assist_notice.isHidden()
    live.next_phrase()
    assert live.results.count() == 1
    assert "もう一度" in ctx.overlay.label.text()


def test_reply_uses_background_thread_and_shows_result(app_parts, qapp):
    from jp_subtitle.assistant import AssistantError, Option

    ctx, window = app_parts
    live = window.live
    main_thread = threading.get_ident()
    threads = []

    class FakeAssistant:
        can_reply = True
        has_ai = True

        def reply(self, text, recent):
            threads.append(threading.get_ident())
            if text == "실패":
                raise AssistantError("키를 확인해 주세요")
            return [Option("暇です", label="존댓말", reading="ひまです", hangul="히마데스")]

    ctx.assistant = FakeAssistant()
    live.refresh_assist_state()
    live.reply_input.setPlainText("한가해요")
    live.submit_reply()
    assert pump(qapp, lambda: live.reply_button.isEnabled())
    assert threads and threads[0] != main_thread
    assert live.results.count() == 1

    live.reply_input.setPlainText("실패")
    live.submit_reply()
    assert pump(qapp, lambda: live.reply_button.isEnabled())
    assert "키를 확인" in live.results.itemAt(0).widget().text()


def test_history_page_shows_and_deletes_sessions(app_parts, qapp, monkeypatch):
    from PySide6.QtWidgets import QMessageBox

    ctx, window = app_parts
    sid = ctx.store.create_session()
    ctx.store.add_line(sid, Line(0, "昨日映画を見た", ko="어제 영화 봤어"), _timing())
    ctx.store.add_line(sid, Line(1, "うん", ko="응", backchannel=True), _timing())
    ctx.store.end_session(sid)
    ctx.store.save_review(
        sid,
        {
            "summary": "영화 이야기",
            "expressions": [{"ja": "映画を見た", "ko": "영화를 봤다", "note": "회화체"}],
            "corrections": [{"original": "行くました", "corrected": "行きました", "note": "활용"}],
            "next_topics": ["좋아하는 영화"],
        },
    )
    window.go_to(1)
    history = window.history
    assert history.list.count() == 1
    assert len(history.view._rows) == 2
    assert "1문장" in history.heading.text()
    assert history.review_body.count() > 0

    history._add_all([{"ja": "映画を見た", "ko": "영화를 봤다", "note": "회화체"}])
    assert window.nav_buttons[2].text() == "복습  1"

    monkeypatch.setattr(QMessageBox, "question", lambda *a, **k: QMessageBox.StandardButton.No)
    history._delete()
    assert history.list.count() == 1
    monkeypatch.setattr(QMessageBox, "question", lambda *a, **k: QMessageBox.StandardButton.Yes)
    history._delete()
    assert history.list.count() == 0
    assert ctx.store.list_sessions() == []


def test_review_page_runs_a_deck(app_parts, qapp):
    ctx, window = app_parts
    ctx.engine.reader = Reader()
    ctx.store.add_card("映画を見た", "영화를 봤다", "회화체")
    ctx.store.add_card("聞こえますか", "")
    window.go_to(2)
    review = window.review
    assert review.front.text() == "영화를 봤다"
    assert review.answer_ja.isHidden() and not review.reveal_button.isHidden()

    review._reveal()
    assert review.answer_ja.text() == "映画を見た"
    assert "えいが" in review.answer_reading.text()
    assert not review.known_button.isHidden()
    review._grade(True)

    # 뜻이 없는 카드는 일본어를 보여 주고 뜻을 떠올리게 한다
    assert review.front.text() == "聞こえますか"
    review._reveal()
    assert review.answer_ja.isHidden()
    review._grade(False)
    assert review.front.text() == "聞こえますか"  # 몰랐던 카드는 다시 나온다
    review._reveal()
    review._grade(True)
    assert review.card_frame.isHidden()
    assert "모두 끝냈어요" in review.empty.text()

    window._refresh_badge()
    assert window.nav_buttons[2].text() == "복습"
    assert review.table.rowCount() == 2
    review.table.selectRow(0)
    review._delete_selected()
    assert len(ctx.store.all_cards()) == 1


def test_diagnostics_page_shows_latency(app_parts, qapp):
    from jp_subtitle.store import LatencyStat
    from jp_subtitle.ui.diagnostics_page import latency_advice

    ctx, window = app_parts
    window.go_to(3)
    assert "아직" in window.diagnostics.advice.text()
    sid = ctx.store.create_session()
    ctx.store.add_line(sid, Line(0, "一", ko="일"), _timing())
    ctx.store.end_session(sid)
    window.go_to(3)
    assert window.diagnostics.table.rowCount() > 0
    assert "1.1초" in window.diagnostics.advice.text()

    def stats(**typical):
        return [LatencyStat(name, value, value, 1) for name, value in typical.items()]

    assert "번역" in latency_advice(stats(번역=2.0, 한국어까지=3.0))
    assert "밀리고" in latency_advice(stats(**{"인식 대기": 1.0, "한국어까지": 3.0}))


def test_settings_page_saves_and_applies(app_parts, qapp):
    ctx, window = app_parts
    settings = window.settings
    assert not window.live.reply_button.isEnabled()
    settings.deepl_key.setText(" abc:fx ")
    settings.vocabulary.setPlainText("田中\n\n ヨルシカ \n")
    settings.silence_end.setValue(0.8)
    settings.device.setCurrentIndex(settings.device.findData("cpu"))
    settings.overlay_enabled.setChecked(False)
    settings.save()

    saved = load_config(ctx.config_path)
    assert saved.translator.deepl_api_key == "abc:fx"
    assert saved.whisper.vocabulary == ["田中", "ヨルシカ"]
    assert saved.whisper.device == "cpu" and saved.whisper.compute_type == "int8"
    assert saved.segmenter.silence_end_sec == 0.8
    assert ctx.cfg is not None and ctx.cfg.translator.deepl_api_key == "abc:fx"
    assert window.live.reply_button.isEnabled()  # 저장하면 바로 쓸 수 있다
    assert not window.live.overlay_check.isChecked()
    assert "저장했어요" in settings.message.text()

    settings.load_values()
    assert settings.vocabulary.toPlainText() == "田中\nヨルシカ"


def test_length_slider_applies_during_call_and_saves(app_parts, qapp):
    from jp_subtitle.config import SENTENCE_PRESETS

    ctx, window = app_parts
    live, settings = window.live, window.settings
    assert live.length_label.text() == "짧게"
    assert settings.sentence_preset.currentData() == "short"

    # 통화 중이라고 가정: 돌고 있는 분할기에 바로 반영된다
    running = Segmenter(ctx.cfg.segmenter, energy_speech_prob)
    ctx.engine._segmenter = running
    live.length_slider.setValue(0)
    assert live.length_label.text() == "아주 짧게"
    assert running.cfg.max_utterance_sec == SENTENCE_PRESETS[0].max_utterance_sec
    assert running.cfg.silence_end_sec == SENTENCE_PRESETS[0].silence_end_sec

    # 파일에 저장되어 다음 실행에도 남고, 설정 화면도 따라온다
    saved = load_config(ctx.config_path).segmenter
    assert saved.max_utterance_sec == SENTENCE_PRESETS[0].max_utterance_sec
    assert saved.partial_interval_sec == SENTENCE_PRESETS[0].partial_interval_sec
    assert settings.sentence_preset.currentData() == "very_short"
    assert settings.max_length.value() == SENTENCE_PRESETS[0].max_utterance_sec
    ctx.engine._segmenter = None


def test_settings_page_custom_sentence_values_sync_with_slider(app_parts, qapp):
    from jp_subtitle.config import SENTENCE_PRESETS

    ctx, window = app_parts
    live, settings = window.live, window.settings

    # 단계를 고르면 값이 채워진다
    index = settings.sentence_preset.findData("long")
    settings.sentence_preset.setCurrentIndex(index)
    settings._on_preset_chosen(index)
    assert settings.max_length.value() == SENTENCE_PRESETS[3].max_utterance_sec
    assert not settings.sentence_cut.isChecked()
    settings.save()
    assert live.length_slider.value() == 3 and live.length_label.text() == "길게"
    assert load_config(ctx.config_path).segmenter.partial_interval_sec == 1.5

    # 값을 직접 바꾸면 '직접 설정' 이 되고 슬라이더에도 표시된다
    settings.max_length.setValue(5.5)
    settings.silence_end.setValue(0.3)
    assert settings.sentence_preset.currentData() == "custom"
    settings.save()
    saved = load_config(ctx.config_path).segmenter
    assert saved.max_utterance_sec == 5.5 and saved.silence_end_sec == 0.3
    assert live.length_label.text() == "직접 설정"
    assert live.length_slider.value() == 1  # 가장 가까운 단계


def test_settings_page_enables_ai_features_with_gemini_key(app_parts, qapp):
    ctx, window = app_parts
    settings = window.settings
    live = window.live
    assert not live.suggest_button.isEnabled() and not live.reply_button.isEnabled()

    settings.gemini_key.setText(" gem-key ")
    settings.gemini_model.setEditText("gemini-3.8-flash")
    settings.engine.setCurrentIndex(settings.engine.findData("gemini"))
    settings.save()

    saved = load_config(ctx.config_path)
    assert saved.translator.gemini_api_key == "gem-key"
    assert saved.translator.gemini_model == "gemini-3.8-flash"
    assert saved.translator.engine == "gemini"
    assert saved.translator.ai_engine == "auto"
    assert ctx.assistant.ai_name == "gemini"
    assert live.suggest_button.isEnabled() and live.reply_button.isEnabled()

    # 복습 화면도 어떤 AI 로 보내는지 알려 준다
    sid = ctx.store.create_session()
    ctx.store.add_line(sid, Line(0, "一", ko="일"), _timing())
    window.go_to(1)
    assert window.history.review_button.isEnabled()
    assert "Gemini" in window.history.review_info.text()

    settings.load_values()
    assert settings.gemini_key.text() == "gem-key"
    settings.ai_engine.setCurrentIndex(settings.ai_engine.findData("claude"))
    settings.save()
    # 직접 고른 Claude 의 키가 없으니 AI 기능은 꺼진다
    assert ctx.assistant.ai_name is None and not live.suggest_button.isEnabled()


def test_engine_runs_two_calls_end_to_end(qapp, tmp_path, monkeypatch):
    """화면, 엔진, 기록이 이어진 실제 흐름: 준비 → 통화 → 기록 → 다시 통화."""
    from PySide6.QtCore import QSettings
    from scipy.io import wavfile

    import jp_subtitle.engine as engine_mod
    import jp_subtitle.transcriber as transcriber_mod
    from jp_subtitle.engine import LISTENING, READY
    from jp_subtitle.ui.app import build_app

    class StubTranscriber:
        notice = ""

        def __init__(self, cfg):
            self.texts = iter(["今日は暑いね", "うんうん", "明日は雨かな", "もう一回", "二回目です"])

        def transcribe(self, audio, beam_size=None, language="ja"):
            if len(audio) == 16000 and not np.any(audio):
                return Transcript()
            return Transcript(next(self.texts))

    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    monkeypatch.setattr(transcriber_mod, "create_transcriber", StubTranscriber)
    monkeypatch.setattr(engine_mod, "SileroSpeechProb", lambda: energy_speech_prob)
    QSettings.setPath(QSettings.Format.NativeFormat, QSettings.Scope.UserScope, str(tmp_path))

    audio = np.concatenate(tone(0.5) + silence(0.8) + tone(0.5) + silence(0.8) + tone(0.5))
    wav = tmp_path / "in.wav"
    wavfile.write(wav, 16000, (audio * 32767).astype(np.int16))

    cfg = Config()
    cfg.segmenter.partial_interval_sec = 0.0
    ctx, window = build_app(cfg, tmp_path / "config.toml", tmp_path / "logs", wav)
    try:
        engine = ctx.engine
        states = []
        engine.state_changed.connect(lambda state, _m: states.append(state))

        # 준비 중에 눌러도 끝나면 바로 시작한다
        window.live.toggle_call()
        engine.prepare()
        window.live.toggle_call()
        assert pump(qapp, lambda: LISTENING in states)
        assert not ctx.overlay.isHidden()  # 통화 중에는 자막 창이 뜬다
        assert window.live.start_button.text() == "통화 끝내기"
        assert pump(qapp, lambda: engine.state == READY and len(ctx.store.list_sessions()) == 1)
        pump(qapp, timeout=0.05)
        assert ctx.overlay.isHidden()  # 통화가 끝나면 자막 창도 닫는다
        assert window.live.start_button.text() == "통화 시작"
        assert engine.reader is not None

        session = ctx.store.list_sessions()[0]
        lines = ctx.store.lines(session.id)
        assert [line.ja for line in lines] == ["今日は暑いね", "うんうん", "明日は雨かな"]
        assert lines[1].backchannel
        assert lines[0].reading.startswith("きょう")
        assert ctx.store.latency_stats(session.id)
        assert list((tmp_path / "logs").glob("*.txt"))

        # 같은 모델로 두 번째 통화도 시작하면 화면이 새로 비워진다
        states.clear()
        window.live.toggle_call()
        assert pump(qapp, lambda: LISTENING in states)
        assert pump(qapp, lambda: engine.state == READY and len(ctx.store.list_sessions()) == 2)
    finally:
        window.hotkeys.unregister_all()
        ctx.engine.shutdown()
        window.close()
        ctx.overlay.close()
        ctx.store.close()


# --- macOS (Windows 에서도 가짜 모듈로 동작을 확인한다) --------------------------------


def _fake_mlx_whisper(monkeypatch):
    calls = []

    def transcribe(audio, **options):
        calls.append((threading.get_ident(), options))
        segment = {"text": "映画を見ました。とても", "avg_logprob": -0.1, "no_speech_prob": 0.0}
        if options.get("word_timestamps"):
            segment["words"] = [
                {"word": "映画を見ました。", "start": 0.0, "end": 1.2},
                {"word": "とても", "start": 1.3, "end": 1.8},
            ]
        return {"language": "ja", "segments": [segment]}

    monkeypatch.setitem(sys.modules, "mlx_whisper", types.SimpleNamespace(transcribe=transcribe))
    return calls


def test_mlx_runs_loading_and_recognition_on_one_thread(monkeypatch):
    """MLX 는 스레드마다 GPU 스트림이 달라서 모든 작업을 전용 스레드 하나에서 해야 한다."""
    from jp_subtitle.transcriber import MLXTranscriber

    calls = _fake_mlx_whisper(monkeypatch)
    results = []
    # 앱처럼 모델 로딩과 인식을 서로 다른 스레드에서 부른다
    holder = {}
    loader = threading.Thread(target=lambda: holder.update(t=MLXTranscriber(WhisperConfig())))
    loader.start()
    loader.join()
    audio = np.zeros(1600, dtype=np.float32)
    worker = threading.Thread(target=lambda: results.append(holder["t"].transcribe(audio)))
    worker.start()
    worker.join()
    results.append(holder["t"].transcribe(audio))

    assert len(calls) == 3
    threads = {thread for thread, _ in calls}
    assert len(threads) == 1 and threading.get_ident() not in threads
    assert loader.ident not in threads and worker.ident not in threads
    assert results[0].text == "映画を見ました。とても"


def test_mlx_passes_word_timing_and_vocabulary(monkeypatch):
    from jp_subtitle.transcriber import MLXTranscriber

    calls = _fake_mlx_whisper(monkeypatch)
    t = MLXTranscriber(WhisperConfig(vocabulary=["田中"]))
    audio = np.zeros(1600, dtype=np.float32)
    assert t.transcribe(audio).sentence_end_sec is None
    assert calls[-1][1]["word_timestamps"] is False
    assert calls[-1][1]["initial_prompt"] == "田中"

    result = t.transcribe(audio, beam_size=1, words=True)
    assert calls[-1][1]["word_timestamps"] is True
    assert result.sentence_end_sec == 1.2

    t.update_settings(WhisperConfig(vocabulary=["佐藤"]))
    t.transcribe(audio, language=None)
    assert calls[-1][1]["initial_prompt"] == "佐藤" and calls[-1][1]["language"] is None


def _fake_sounddevice(monkeypatch, rate=48000):
    devices = [
        {"name": "MacBook Pro Speakers", "max_input_channels": 0, "default_samplerate": rate},
        {"name": "MacBook Pro Microphone", "max_input_channels": 1, "default_samplerate": rate},
        {"name": "AirPods Pro", "max_input_channels": 1, "default_samplerate": rate},
    ]
    opened = []

    class InputStream:
        def __init__(self, **options):
            opened.append(options)
            self.callback = options["callback"]

        def __enter__(self):
            tone_48k = (0.5 * np.sin(2 * np.pi * 300 * np.arange(rate) / rate)).astype(np.float32)
            for start in range(0, rate, 960):
                chunk = tone_48k[start : start + 960].reshape(-1, 1)
                self.callback(chunk, len(chunk), None, None)
            return self

        def __exit__(self, *args):
            return False

    def query_devices(device=None, kind=None):
        if kind == "input":
            return devices[1 if device is None else device]
        return devices

    fake = types.SimpleNamespace(query_devices=query_devices, InputStream=InputStream)
    monkeypatch.setitem(sys.modules, "sounddevice", fake)
    return opened


def test_mac_microphone_reads_blocks(monkeypatch):
    import jp_subtitle.audio_capture as ac

    opened = _fake_sounddevice(monkeypatch)
    monkeypatch.setattr(sys, "platform", "darwin")
    source = ac.create_mic_source("airpods")
    assert isinstance(source, ac.MacMicSource)
    blocks = []
    for block in source.blocks():
        blocks.append(block)
        if len(blocks) * BLOCK_SAMPLES >= 0.9 * SAMPLE_RATE:
            source.stop()
    assert opened[0]["device"] == 2 and opened[0]["channels"] == 1
    assert np.sqrt(np.mean(np.concatenate(blocks[2:]) ** 2)) > 0.2

    assert [d["name"] for d in ac.list_input_devices()] == ["MacBook Pro Microphone", "AirPods Pro"]
    with pytest.raises(RuntimeError, match="없는 장치"):
        list(ac.MacMicSource("없는 장치").blocks())


def test_mic_source_for_each_platform(monkeypatch):
    import jp_subtitle.audio_capture as ac

    monkeypatch.setattr(sys, "platform", "win32")
    assert isinstance(ac.create_mic_source(), ac.MicSource)
    monkeypatch.setattr(sys, "platform", "linux")
    assert ac.create_mic_source() is None


def test_settings_page_on_mac_chooses_app_instead_of_device(app_parts, qapp, monkeypatch):
    import jp_subtitle.audio_capture as ac
    from jp_subtitle.ui.settings_page import MAC_ALL_SOUND, SettingsPage

    ctx, _window = app_parts
    _fake_sounddevice(monkeypatch)
    monkeypatch.setattr(sys, "platform", "darwin")
    monkeypatch.setattr(ac, "list_mac_apps", lambda: ["maum", "Safari", "maum"])
    page = SettingsPage(ctx, lambda: None)

    assert page.output_device.currentText() == MAC_ALL_SOUND
    assert page.mic_enabled.isEnabled() and page.mic_device.findData("AirPods Pro") >= 0
    assert page.device.itemText(0) == "Apple GPU"

    # 모든 소리는 빈 값으로 저장한다
    page.save()
    assert load_config(ctx.config_path).audio.app == ""

    # 앱 목록은 눌렀을 때만 읽고, 적어 둔 값을 지우지 않는다
    page.output_device.setEditText("ma")
    page._load_mac_apps()
    assert pump(qapp, lambda: page.load_apps_button.isEnabled())
    assert page.output_device.findData("Safari") >= 0
    assert page.output_device.count() == 3
    assert page.output_device.currentText() == "ma"

    page.output_device.setEditText(" maum ")
    page.save()
    saved = load_config(ctx.config_path).audio
    assert saved.app == "maum" and saved.device == ""
    page.load_values()
    assert page.output_device.currentText() == "maum"

    # 권한이 없어 목록을 못 읽으면 이유를 보여 준다
    def denied():
        raise RuntimeError("권한이 필요합니다")

    monkeypatch.setattr(ac, "list_mac_apps", denied)
    page._load_mac_apps()
    assert pump(qapp, lambda: page.load_apps_button.isEnabled())
    assert "권한" in page.message.text()


def test_hotkey_names_follow_the_platform(monkeypatch):
    from jp_subtitle.hotkeys import hotkey_name

    monkeypatch.setattr(sys, "platform", "win32")
    assert hotkey_name("R") == "Ctrl+Alt+R"
    monkeypatch.setattr(sys, "platform", "darwin")
    assert hotkey_name("R") == "⌃⌥R"


def test_mac_hotkeys_with_fake_carbon():
    from jp_subtitle.hotkeys import MAC_CONTROL_KEY, MAC_KEY_CODES, MAC_OPTION_KEY, _MacHotkeys

    class Function:
        def __init__(self, implementation):
            self.implementation = implementation

        def __call__(self, *args):
            return self.implementation(*args)

    registered = []
    state = {"status": 0, "pressed": 0}

    def install(target, handler, count, spec, user, out):
        state["handler"] = handler
        return 0

    def register(code, modifiers, hot_id, target, options, out):
        registered.append((code, modifiers, hot_id.id))
        out._obj.value = 1234
        return state["status"]

    def get_parameter(event, name, kind, actual, size, out_size, out):
        from jp_subtitle.hotkeys import _MAC_SIGNATURE

        out._obj.signature = _MAC_SIGNATURE
        out._obj.id = state["pressed"]
        return 0

    carbon = types.SimpleNamespace(
        GetApplicationEventTarget=Function(lambda: 1),
        InstallEventHandler=Function(install),
        GetEventParameter=Function(get_parameter),
        RegisterEventHotKey=Function(register),
        UnregisterEventHotKey=Function(lambda ref: registered.append("해제")),
    )
    pressed = []
    keys = _MacHotkeys(pressed.append, carbon=carbon)
    assert keys.register(1, "R") is True
    assert registered == [(MAC_KEY_CODES["R"], MAC_CONTROL_KEY | MAC_OPTION_KEY, 1)]
    state["pressed"] = 1
    assert keys._handle(None, None, None) == 0
    assert pressed == [1]

    # 단축키 처리 중 오류가 나도 프로그램은 죽지 않는다
    keys._on_hotkey = lambda _id: 1 / 0
    assert keys._handle(None, None, None) == 0

    state["status"] = -9878
    assert keys.register(2, "Q") is False
    keys.unregister_all()
    assert registered[-1] == "해제" and not keys._refs


@pytest.mark.skipif(sys.platform != "darwin", reason="실제 macOS 전역 단축키")
def test_mac_hotkeys_real_registration(qapp):
    from jp_subtitle.hotkeys import GlobalHotkeys

    hotkeys = GlobalHotkeys()
    hotkeys.register("R", lambda: None)
    assert hotkeys.available
    assert len(hotkeys._callbacks) + len(hotkeys.failed) == 1
    hotkeys.unregister_all()


def test_info_plist_gets_microphone_text(tmp_path):
    import importlib.util
    import plistlib
    from pathlib import Path

    script = Path(__file__).parent.parent / "scripts" / "build_exe.py"
    spec = importlib.util.spec_from_file_location("build_exe_for_test", script)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    path = tmp_path / "Info.plist"
    path.write_bytes(plistlib.dumps({"CFBundleName": "jp_subtitle"}))
    module.patch_info_plist(path)
    info = plistlib.loads(path.read_bytes())
    assert info["CFBundleName"] == "jp_subtitle"
    assert "마이크" in info["NSMicrophoneUsageDescription"]
    assert info["CFBundleDisplayName"] == "일본어 통화 도우미"


@pytest.mark.skipif(sys.platform != "win32", reason="Windows 전역 단축키")
def test_hotkey_filter_calls_callback(qapp):
    from ctypes import wintypes

    from jp_subtitle.hotkeys import WM_HOTKEY, GlobalHotkeys

    hotkeys = GlobalHotkeys()
    called = []
    hotkeys._callbacks[7] = lambda: called.append(True)
    msg = wintypes.MSG()
    msg.message = WM_HOTKEY
    msg.wParam = 7
    assert hotkeys.nativeEventFilter(b"windows_generic_MSG", ctypes.addressof(msg)) == (True, 0)
    assert called == [True]
    msg.wParam = 8
    assert hotkeys.nativeEventFilter(b"windows_generic_MSG", ctypes.addressof(msg)) == (False, 0)
    hotkeys._callbacks.clear()
