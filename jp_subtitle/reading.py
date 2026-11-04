"""일본어 문장의 읽는 법(히라가나)과 한글 발음을 만든다.

형태소 분석기로 만든 추정값이라 이름이나 신조어는 틀릴 수 있다.
"""

from __future__ import annotations

import threading
from dataclasses import dataclass

_DIGRAPHS = {
    "きゃ": "캬", "きゅ": "큐", "きょ": "쿄", "しゃ": "샤", "しゅ": "슈", "しょ": "쇼",
    "ちゃ": "차", "ちゅ": "추", "ちょ": "초", "にゃ": "냐", "にゅ": "뉴", "にょ": "뇨",
    "ひゃ": "햐", "ひゅ": "휴", "ひょ": "효", "みゃ": "먀", "みゅ": "뮤", "みょ": "묘",
    "りゃ": "랴", "りゅ": "류", "りょ": "료", "ぎゃ": "갸", "ぎゅ": "규", "ぎょ": "교",
    "じゃ": "자", "じゅ": "주", "じょ": "조", "ぢゃ": "자", "ぢゅ": "주", "ぢょ": "조",
    "びゃ": "뱌", "びゅ": "뷰", "びょ": "뵤", "ぴゃ": "퍄", "ぴゅ": "퓨", "ぴょ": "표",
    "ふぁ": "파", "ふぃ": "피", "ふぇ": "페", "ふぉ": "포", "てぃ": "티", "でぃ": "디",
    "うぃ": "위", "うぇ": "웨", "うぉ": "워", "ちぇ": "체", "しぇ": "셰", "じぇ": "제",
    "つぁ": "차", "とぅ": "투", "どぅ": "두",
}
_SINGLES = dict(zip(
    "あいうえおかきくけこさしすせそたちつてとなにぬねのはひふへほまみむめも"
    "やゆよらりるれろわをがぎぐげござじずぜぞだぢづでどばびぶべぼぱぴぷぺぽ"
    "ゔぁぃぅぇぉゃゅょゎ",
    "아이우에오카키쿠케코사시스세소타치츠테토나니누네노하히후헤호마미무메모"
    "야유요라리루레로와오가기구게고자지즈제조다지즈데도바비부베보파피푸페포"
    "부아이우에오야유요와",
))
_JONG_N = 4   # ㄴ 받침
_JONG_S = 19  # ㅅ 받침


def katakana_to_hiragana(text: str) -> str:
    return "".join(chr(ord(c) - 0x60) if "ァ" <= c <= "ヶ" else c for c in text)


def _add_final(out: list[str], jong: int, alone: str) -> None:
    """앞 글자에 받침을 붙인다. 붙일 글자가 없으면 자모만 적는다."""
    if len(out) >= 2 and out[-1] == " ":
        out.pop()  # 낱말 경계에 걸린 ん 도 앞 글자에 붙여 읽는다
    if out and "가" <= out[-1] <= "힣" and (ord(out[-1]) - 0xAC00) % 28 == 0:
        out[-1] = chr(ord(out[-1]) + jong)
    else:
        out.append(alone)


def kana_to_hangul(text: str) -> str:
    """가나를 한글 발음으로 옮긴다. 가나가 아닌 글자는 그대로 둔다."""
    kana = katakana_to_hiragana(text)
    out: list[str] = []
    i = 0
    while i < len(kana):
        pair = kana[i : i + 2]
        char = kana[i]
        if pair in _DIGRAPHS:
            out.append(_DIGRAPHS[pair])
            i += 2
            continue
        if char == "ん":
            _add_final(out, _JONG_N, "ㄴ")
        elif char == "っ":
            _add_final(out, _JONG_S, "ㅅ")
        elif char == "ー":
            out.append("-")
        else:
            out.append(_SINGLES.get(char, char))
        i += 1
    return "".join(out)


def has_kanji(text: str) -> bool:
    return any("一" <= c <= "鿿" or "㐀" <= c <= "䶿" or c == "々" for c in text)


@dataclass
class Annotation:
    hiragana: str
    hangul: str


# 앞 낱말에 붙여 읽는 품사 (띄어 쓰면 오히려 읽기 어렵다)
_ATTACH = (
    "助動詞", "記号", "助詞,接続助詞", "動詞,非自立", "動詞,接尾", "名詞,接尾", "名詞,非自立",
)


class Reader:
    def __init__(self) -> None:
        from janome.tokenizer import Tokenizer

        self._tokenizer = Tokenizer()
        self._lock = threading.Lock()

    def annotate(self, text: str) -> Annotation:
        hiragana: list[str] = []
        phonetics: list[str] = []
        with self._lock:
            tokens = list(self._tokenizer.tokenize(text))
        for token in tokens:
            reading = token.reading if token.reading != "*" else token.surface
            phonetic = token.phonetic if token.phonetic != "*" else reading
            gap = " " if hiragana and not token.part_of_speech.startswith(_ATTACH) else ""
            hiragana.append(gap + katakana_to_hiragana(reading))
            phonetics.append(gap + phonetic)
        return Annotation("".join(hiragana), kana_to_hangul("".join(phonetics)))

    def subtitle_reading(self, text: str) -> str:
        """자막에 붙일 읽는 법. 한자가 없으면 읽을 수 있으니 비워 둔다."""
        return self.annotate(text).hiragana if has_kanji(text) else ""


def create_reader() -> Reader | None:
    """분석기를 쓸 수 없어도 자막은 나오도록 None 을 돌려준다."""
    try:
        return Reader()
    except Exception as e:
        print(f"[읽는 법] 분석기를 불러오지 못했습니다: {e}")
        return None
