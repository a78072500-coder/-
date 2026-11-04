"""음성인식 결과에서 일본어 문장의 끝을 찾는다.

Whisper 는 문장 부호를 찍기도 하고 빠뜨리기도 하므로, 부호가 없어도 존댓말 끝맺음(ます, です 등)이면
문장 끝으로 본다. 반말은 뒤에 명사가 이어지는 경우와 구분하기 어려워 부호가 있을 때만 끝으로 본다.
"""

from __future__ import annotations

import re

_TERMINATOR = re.compile(r"[。．！？!?]+[」』）)]*")
# 뒤에 이런 말이 이어지면 문장이 아직 끝나지 않았다
_POLITE = re.compile(
    r"(?<!ます)(?:ませんでした|ました|でした|ません|です|ます|ください|でしょう|ましょう)"
    r"(?!から|が|けど|けれど|ので|のに|のは|のが|し(?!か)|と[、,]|って|ま|ら|よう)"
)
_FINAL_PARTICLES = ("かね", "よね", "ねえ", "ねー", "よー", "か", "ね", "よ", "わ", "な")
_NOT_WORD = re.compile(r"[\s。．！？!?、,，…・「」『』（）()]")
MIN_CHARS = 2  # 이보다 짧은 조각은 문장으로 나누지 않는다


def _word_chars(text: str) -> int:
    return len(_NOT_WORD.sub("", text))


def _candidates(text: str):
    for match in _TERMINATOR.finditer(text):
        yield match.end()
    for match in _POLITE.finditer(text):
        end = match.end()
        for particle in _FINAL_PARTICLES:
            if text.startswith(particle, end):
                end += len(particle)
                break
        following = text[end : end + 1]
        if following in ("、", ",", "，"):
            continue  # 쉼표가 뒤따르면 문장이 이어진다
        terminator = _TERMINATOR.match(text, end)
        yield terminator.end() if terminator else end


def first_sentence_end(text: str) -> int | None:
    """첫 문장이 끝나는 글자 위치. 문장이 끝난 뒤 이어지는 말이 없으면 None."""
    for end in sorted(set(_candidates(text))):
        if _word_chars(text[:end]) >= MIN_CHARS and _word_chars(text[end:]) >= MIN_CHARS:
            return end
    return None


def split_sentences(text: str) -> list[str]:
    """여러 문장이 붙어 있으면 나눈다."""
    parts: list[str] = []
    rest = text.strip()
    while (end := first_sentence_end(rest)) is not None:
        parts.append(rest[:end].strip())
        rest = rest[end:].strip()
    if rest:
        parts.append(rest)
    return parts
