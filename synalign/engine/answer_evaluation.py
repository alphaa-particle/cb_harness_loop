"""Grading short answers without a judge model: numbers, refusals and yes/no verdicts.

Every check here is a plain rule that can be read and argued with:

    numbers_in(text)        every number an answer states, as plain integers
                            ("Rs 2 lakh" -> 200000, "३,०००" -> 3000, "1.5 crore" -> 15000000)
    states_value(...)       does the answer state the expected value (each accepted form matched whole)?
    is_refusal(text)        does the answer say it could not find the information?
    verdict(text)           does the answer open with yes, no or not sure?

They are imperfect by nature (a long answer listing many numbers can contain the
right one by luck), so results built on them should be read together with a hand
check of a sample of answers.
"""

from __future__ import annotations

import re

_DEVANAGARI_DIGITS = str.maketrans("०१२३४५६७८९", "0123456789")
_SCALE = {"lakh": 100_000, "lakhs": 100_000, "lac": 100_000, "लाख": 100_000,
          "crore": 10_000_000, "crores": 10_000_000, "करोड़": 10_000_000, "करोड": 10_000_000,
          "thousand": 1_000, "hazar": 1_000, "hazaar": 1_000, "हज़ार": 1_000, "हजार": 1_000}
_NUMBER = re.compile(r"(\d+(?:,\d+)*(?:\.\d+)?)\s*(" + "|".join(sorted(_SCALE, key=len, reverse=True)) + r")?",
                     re.IGNORECASE)

REFUSAL_PHRASES = (
    "could not find", "couldn't find", "can not find", "cannot find", "not found in", "no information",
    "does not contain", "doesn't contain", "not mentioned", "not available in", "i don't know", "i do not know",
    "not in the scheme information", "unable to find", "no details",
    "नहीं मिली", "नहीं मिला", "जानकारी नहीं", "उपलब्ध नहीं", "नहीं पता", "पता नहीं",
    "nahi mili", "nahin mili", "jankari nahi", "pata nahi",
)
_YES = ("yes", "हाँ", "हां", "जी हाँ", "जी हां", "haan", "han ji", "ji haan")
_NO = ("no", "नहीं", "nahi", "nahin", "जी नहीं")
_UNSURE = ("not sure", "cannot say", "can't say", "it depends", "maybe", "पक्का नहीं", "निश्चित नहीं",
           "कह नहीं सकते", "pakka nahi")
# When the answer does not open with yes or no, its first sentence usually still says which.
_SAYS_NO = ("cannot apply", "can't apply", "can not apply", "not eligible", "do not qualify", "does not qualify",
            "cannot join", "cannot get", "are not allowed", "नहीं कर सकते", "नहीं कर सकती", "नहीं कर सकता",
            "पात्र नहीं", "नहीं मिल सकती", "नहीं मिल सकता", "nahi kar sakte", "patra nahi", "nahi mil sakti")
_SAYS_YES = ("you can apply", "you are eligible", "you can join", "you qualify", "you can get",
             "कर सकते हैं", "कर सकती हैं", "कर सकता है", "पात्र हैं", "मिल सकती है", "मिल सकता है",
             "kar sakte hain", "patra hain", "mil sakti hai")

# A word ends where no letter, digit or Indian-script vowel sign follows ("नहीं," but not "Nobody").
# The Hindi full stops । and ॥ sit inside the Devanagari block but do end a word ("नहीं।").
_WORD_END = r"(?![\w\u0900-\u0963\u0966-\u0DFF])"


def numbers_in(text: str) -> set[int]:
    """Whole-number values stated in the text, after removing digit-group commas and scaling words."""
    found = set()
    for digits, scale in _NUMBER.findall(text.translate(_DEVANAGARI_DIGITS)):
        value = float(digits.replace(",", ""))
        if scale:
            value *= _SCALE[scale.lower()]
        if value == int(value):
            found.add(int(value))
    return found


def _plain(text: str) -> str:
    return re.sub(r"\s+", " ", text.translate(_DEVANAGARI_DIGITS).lower()).strip()


# Time units: "3 weeks" must not pass for "3 days". English, Hindi and Hindi in Latin letters.
_UNITS = {
    "hour": ("hour", "hours", "hr", "hrs", "घंटे", "घंटा", "ghante", "ghanta"),
    "day": ("day", "days", "दिन", "din"),
    "week": ("week", "weeks", "सप्ताह", "हफ्ते", "हफ़्ते", "hafte", "saptah"),
    "month": ("month", "months", "monthly", "महीने", "महीना", "माह", "mahine", "mahina", "maah"),
    "year": ("year", "years", "yearly", "annual", "annually", "साल", "वर्ष", "saal", "varsh"),
}


def _units(text: str) -> set[str]:
    words = set(re.findall(r"[\w\u0900-\u097f]+", text.lower()))
    return {unit for unit, forms in _UNITS.items() if words & set(forms)}


def states_value(answer: str, value: str, accept: list[str] | None = None) -> bool:
    """True when the answer states the expected value in one of its accepted forms, matched whole.

    A form with numbers matches when every one of its numbers is in the answer ("18 to 40" needs
    both 18 and 40; "Rs 2 lakh" matches "200000"), and when it names a time unit, the answer names
    that unit too ("72 hours" is not "3 weeks"). A form without numbers must appear as text, so
    "free" is found in "free of cost". Numbers are compared as numbers: "60" is not found in "600".
    """
    options = [str(o) for o in [value, *(accept or [])] if str(o).strip()]
    stated = numbers_in(answer)
    plain = _plain(answer)
    said_units = _units(answer)
    for option in options:
        wanted = numbers_in(option) if re.search(r"\d", option.translate(_DEVANAGARI_DIGITS)) else set()
        if wanted:
            if wanted <= stated and _units(option) <= said_units:
                return True
        elif _plain(option) in plain:
            return True
    return False


def is_refusal(answer: str) -> bool:
    low = answer.lower()
    return any(phrase in low for phrase in REFUSAL_PHRASES)


def verdict(answer: str) -> str:
    """'yes', 'no', 'unsure' or 'none', read from how the answer opens (first sentence)."""
    first = re.split(r"(?<=[.!?।])\s+|\n", answer.strip(), maxsplit=1)[0].lower().strip(" *#:-\"'")
    if any(first.startswith(p) for p in _UNSURE):
        return "unsure"
    if any(re.match(rf"{re.escape(p)}{_WORD_END}", first) for p in _NO):
        return "no"
    if any(re.match(rf"{re.escape(p)}{_WORD_END}", first) for p in _YES):
        return "yes"
    if any(p in first for p in _SAYS_NO):          # checked first: "नहीं कर सकते हैं" contains "कर सकते हैं"
        return "no"
    if any(p in first for p in _SAYS_YES):
        return "yes"
    return "none"
