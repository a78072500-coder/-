"""맞장구(相槌)를 알아본다.

통화에서는 うん, へえ 같은 짧은 말이 자주 나온다. 이것까지 문장으로 처리하면 앞 문장이
화면에서 밀려나고 번역 사용량도 줄어든다. 사전에 있는 말로만 이루어진 발화만 맞장구로 본다.
"""

from __future__ import annotations

import re

_GLOSS = {
    "うん": "응",
    "ううん": "아니",
    "うーん": "음…",
    "はい": "네",
    "ええ": "네",
    "えー": "에이",
    "えっ": "어?",
    "え": "어?",
    "ああ": "아",
    "あー": "아",
    "あっ": "앗",
    "おお": "오",
    "おー": "오",
    "へえ": "오~",
    "へぇ": "오~",
    "へー": "오~",
    "ふーん": "흠",
    "ふん": "응",
    "ほう": "호오",
    "ほー": "호오",
    "なるほど": "그렇구나",
    "そう": "맞아",
    "そうだ": "그렇네",
    "そうです": "맞아요",
    "そうですね": "그렇네요",
    "そうなんだ": "그렇구나",
    "そうなんですね": "그렇군요",
    "そうなんですか": "그래요?",
    "そうですか": "그래요?",
    "そうか": "그렇구나",
    "そっか": "그렇구나",
    "たしかに": "그러네",
    "確かに": "그러네",
    "ほんと": "진짜",
    "本当": "진짜",
    "ほんとに": "진짜로",
    "本当に": "진짜로",
    "まじ": "진짜",
    "まじで": "진짜?",
    "わかった": "알겠어",
    "分かった": "알겠어",
    "わかりました": "알겠습니다",
    "分かりました": "알겠습니다",
}
_MAX_CHARS = 16
_PUNCT = re.compile(r"[\s。、，,！？!?．.…・~〜「」『』()（）]+")
_WORDS = sorted(_GLOSS, key=len, reverse=True)


def _to_hiragana(text: str) -> str:
    return "".join(chr(ord(c) - 0x60) if "ァ" <= c <= "ヶ" else c for c in text)


def backchannel_gloss(text: str) -> str | None:
    """맞장구면 짧은 한국어 뜻을, 아니면 None 을 돌려준다."""
    rest = _to_hiragana(_PUNCT.sub("", text))
    if rest not in _GLOSS:
        rest = rest.rstrip("ねよな")  # なるほどね, そうだよ 처럼 끝에 붙는 말은 뜻을 바꾸지 않는다
    if not rest or len(rest) > _MAX_CHARS:
        return None
    glosses: list[str] = []
    while rest:
        word = next((w for w in _WORDS if rest.startswith(w)), None)
        if word is None:
            return None
        gloss = _GLOSS[word]
        if gloss not in glosses:
            glosses.append(gloss)
        rest = rest[len(word) :]
    return " ".join(glosses)
