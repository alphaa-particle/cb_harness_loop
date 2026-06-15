import numpy as np

from engine.config import RANDOM_SEED
from engine.domain_pack import DomainPack
from engine.schemas import TestCase

# Neighboring keys on a QWERTY keyboard, for realistic typos.
_QWERTY = {
    "a": "qsz", "b": "vgn", "c": "xdv", "d": "sfce", "e": "wrd", "f": "dgr",
    "g": "fhv", "h": "gjb", "i": "uok", "j": "hkn", "k": "jli", "l": "ko",
    "m": "nj", "n": "bmh", "o": "ipl", "p": "ol", "q": "wa", "r": "etf",
    "s": "adw", "t": "ryg", "u": "yij", "v": "cbf", "w": "qes", "x": "zc",
    "y": "tuh", "z": "xa",
}


def _typo_word(word: str, rng) -> str:
    """Apply one keyboard-style mistake to a word."""
    if len(word) < 3:
        return word
    i = int(rng.integers(0, len(word)))
    ch = word[i].lower()
    op = rng.random()
    if op < 0.4 and ch in _QWERTY:                       # substitution
        repl = _QWERTY[ch][int(rng.integers(0, len(_QWERTY[ch])))]
        return word[:i] + repl + word[i + 1:]
    if op < 0.7:                                          # deletion
        return word[:i] + word[i + 1:]
    return word[:i] + word[i] + word[i:]                  # duplication


def add_typos(text: str, rate: float, rng) -> str:
    words = text.split(" ")
    out = [(_typo_word(w, rng) if rng.random() < rate else w) for w in words]
    return " ".join(out)


def build_case_for_condition(
    pack: DomainPack, profile: dict, condition: str, split: str, rng
) -> TestCase:
    schema = pack.profile_schema
    cfg = pack.perturbations["conditions"][condition]
    question_fields = schema["question_fields"]

    # --- Decide what the question reveals -------------------------------
    if cfg.get("use_vague_question"):
        visible = {f: None for f in question_fields}
        question = schema["vague_question"]
    else:
        hidden = set(cfg.get("hide_fields", []))
        visible = {}
        sentences = []
        for f in question_fields:
            value = profile.get(f)
            if f in hidden or value is None:
                visible[f] = None        # hidden by perturbation, or user doesn't know it
            else:
                visible[f] = value
                sentences.append(schema["field_phrases"][f].format(**{f: value}))
        question = " ".join(sentences + [schema["base_question"]])

        wrapper = cfg.get("wrapper")
        if wrapper:
            entity_ids = list(pack.entity_aliases().keys())
            entity_id = entity_ids[int(rng.integers(0, len(entity_ids)))]
            entity_name = pack.entity_aliases()[entity_id][0]
            question = wrapper.format(question=question, entity=entity_name)

        typo_rate = cfg.get("typo_rate", 0.0)
        if typo_rate > 0:
            question = add_typos(question, typo_rate, rng)

    # --- Ground truth follows the VISIBLE information --------------------
    ground_truth = pack.build_ground_truth(profile, visible)

    return TestCase(
        case_id=f"user_{profile['user_id']}_{condition}",
        user_id=profile["user_id"],
        split=split,
        condition=condition,
        question=question,
        profile=profile,
        visible=visible,
        ground_truth=ground_truth,
    )


def make_test_cases(pack: DomainPack, users: list[dict], splits: dict[int, str]) -> list[TestCase]:
    rng = np.random.default_rng(RANDOM_SEED + 1)
    conditions = list(pack.perturbations["conditions"].keys())
    cases = []
    for profile in users:
        split = splits[profile["user_id"]]
        for condition in conditions:
            cases.append(build_case_for_condition(pack, profile, condition, split, rng))
    return cases
