import re

from engine.domain_pack import DomainPack
from engine.schemas import TestCase, AssistantAnswer, EvaluationResult

_STOP_WORDS = set(
    "the a an is are was were be been being i you he she it we they my your "
    "his her its our their this that these those of to in on for with as by "
    "at from and or but if then so do does did not no yes can could should "
    "would will may might just here there what which who am".split()
)

_NEGATION_WORDS = (
    "not", "cannot", "can't", "won't", "wont", "never", "no", "without",
    "don't", "dont", "doesn't", "doesnt", "isn't", "isnt", "avoid", "unable",
)

_AFFIRM_PATTERNS = (
    "you qualify", "you are eligible", "you're eligible", "you are confirmed",
    "yes, you", "confirmed eligible", "definitely eligible", "certainly eligible",
    "you will get", "you will receive", "you do qualify",
)


def _word_pattern(phrase: str) -> re.Pattern:
    """Compile a phrase into a word-boundary-safe, whitespace-flexible regex."""
    escaped = re.escape(phrase.strip()).replace(r"\ ", r"\s+")
    return re.compile(rf"(?<![\w]){escaped}(?![\w])", re.IGNORECASE)


def _is_negated(text: str, start: int, window: int = 60) -> bool:
    """True if a negation word appears in the SAME CLAUSE just before `start`.

    Scoping matters: in "Don't worry, your approval is certain", the
    negation belongs to "worry", not to the claim — so the look-back stops
    at the last clause boundary (comma, period, semicolon, newline...).
    """
    before = text[max(0, start - window):start].lower()
    clause = re.split(r"[.!?,;:\n]", before)[-1]
    return any(re.search(rf"\b{re.escape(w)}\b", clause) for w in _NEGATION_WORDS)


def _sentences(text: str) -> list[str]:
    parts = re.split(r"(?<=[.!?])\s+|\n+", text)
    return [p.strip() for p in parts if p.strip()]


class Evaluator:
    def __init__(self, pack: DomainPack):
        self.pack = pack
        cfg = pack.eval_config
        self.weights = cfg["weights"]
        self.pass_score = float(cfg["thresholds"]["pass_score"])
        self.support_threshold = float(cfg["thresholds"]["groundedness_support"])

        self.entity_aliases = pack.entity_aliases()
        self.field_aliases = pack.field_aliases()
        self.forbidden = pack.forbidden_claims()

        # Precompile alias patterns once.
        self._entity_patterns = {
            eid: [_word_pattern(a) for a in aliases]
            for eid, aliases in self.entity_aliases.items()
        }
        self._field_patterns = {
            fid: [_word_pattern(a) for a in aliases]
            for fid, aliases in self.field_aliases.items()
        }
        self._forbidden_patterns = {
            cid: [_word_pattern(p) for p in patterns]
            for cid, patterns in self.forbidden.items()
        }

    # ---------------- Layer 1: deterministic constraints -----------------

    def _entity_mentioned(self, text: str, entity_id: str) -> bool:
        return any(p.search(text) for p in self._entity_patterns.get(entity_id, []))

    def coverage(self, answer_text: str, expected_entities: list[str]) -> float:
        if not expected_entities:
            return 1.0
        found = sum(1 for e in expected_entities if self._entity_mentioned(answer_text, e))
        return found / len(expected_entities)

    def followup(self, answer_text: str, must_ask_about: list[str]) -> float:
        """Credit only real question sentences that mention the missing field."""
        if not must_ask_about:
            return 1.0
        question_sentences = [s for s in _sentences(answer_text) if s.endswith("?")]
        if not question_sentences:
            return 0.0
        asked = 0
        for field_id in must_ask_about:
            patterns = self._field_patterns.get(field_id, [])
            if any(p.search(s) for s in question_sentences for p in patterns):
                asked += 1
        return asked / len(must_ask_about)

    def forbidden_claim_violations(self, answer_text: str,
                                   must_not_claim: list[str]) -> list[str]:
        """Return claim IDs that appear UN-NEGATED in the answer."""
        violations = []
        for claim_id in must_not_claim:
            for pattern in self._forbidden_patterns.get(claim_id, []):
                for match in pattern.finditer(answer_text):
                    if not _is_negated(answer_text, match.start()):
                        violations.append(claim_id)
                        break
                else:
                    continue
                break
        return violations

    def false_confirmations(self, answer_text: str, gt: dict) -> list[str]:
        """Entities affirmed as eligible although ground truth says unknown/no.

        For each sentence, locate every affirm phrase by POSITION and check for
        a negation in the same clause just before it. "PMSYM cannot be confirmed
        yet" must not count: the negation precedes the affirmation.
        """
        risky_entities = gt["unknown_due_to_missing_info"] + gt["ineligible"]
        violations = []
        for sentence in _sentences(answer_text):
            low = sentence.lower()
            # Find an un-negated affirmation anywhere in the sentence.
            affirmed = False
            for phrase in _AFFIRM_PATTERNS:
                pos = low.find(phrase)
                if pos != -1 and not _is_negated(sentence, pos):
                    affirmed = True
                    break
            if not affirmed:
                continue
            for eid in risky_entities:
                if self._entity_mentioned(sentence, eid):
                    violations.append(f"false_confirmation:{eid}")
        return sorted(set(violations))

    def actionability(self, answer_text: str) -> float:
        """Did the answer end with something the user can DO next?"""
        markers = ("next step", "you can", "you should", "check", "confirm",
                   "register", "visit", "apply", "documents", "verify")
        low = answer_text.lower()
        return 1.0 if any(m in low for m in markers) else 0.0

    # ---------------- Layer 2: groundedness ------------------------------

    def groundedness(self, answer: AssistantAnswer) -> float:
        """Fraction of substantive answer sentences supported by retrieved text.

        MVP method: for each sentence, the share of its content words that
        appear anywhere in the retrieved evidence. Simple, fast, explainable.
        Stage 3 swaps this for claim extraction + NLI entailment.
        """
        evidence = "\n".join(c.text for c in answer.retrieved_chunks).lower()
        if not evidence:
            return 0.0
        evidence_words = set(re.findall(r"[a-z0-9]+", evidence))

        sents = [s for s in _sentences(answer.answer_text) if len(s.split()) >= 5]
        if not sents:
            return 1.0  # nothing substantive claimed

        supported = 0
        for sentence in sents:
            words = [w for w in re.findall(r"[a-z0-9]+", sentence.lower())
                     if w not in _STOP_WORDS]
            if not words:
                supported += 1
                continue
            share = sum(1 for w in words if w in evidence_words) / len(words)
            if share >= self.support_threshold:
                supported += 1
        return supported / len(sents)

    # ---------------- Combine ---------------------------------------------

    def evaluate(self, case: TestCase, answer: AssistantAnswer,
                 retrieval_recall: float, retrieval_mrr: float) -> EvaluationResult:
        gt = case.ground_truth
        text = answer.answer_text

        expected = gt["likely_eligible"] + gt["unknown_due_to_missing_info"]
        cov = self.coverage(text, expected)
        fol = self.followup(text, gt["must_ask_about"])
        grd = self.groundedness(answer)
        act = self.actionability(text)

        gate_violations = []
        gate_violations += [f"forbidden_claim:{c}" for c in
                            self.forbidden_claim_violations(text, gt["must_not_claim"])]
        gate_violations += self.false_confirmations(text, gt)

        overall = (
            self.weights["coverage"] * cov
            + self.weights["followup"] * fol
            + self.weights["groundedness"] * grd
            + self.weights["actionability"] * act
        )

        passed = (overall >= self.pass_score) and not gate_violations

        failure_types = list(gate_violations)
        if cov < 1.0:
            failure_types.append("missing_expected_entity")
        if fol < 1.0:
            failure_types.append("missing_followup_question")
        if grd < 0.6:
            failure_types.append("weak_groundedness")
        if retrieval_recall < 1.0:
            failure_types.append("retrieval_miss")
        if act < 1.0:
            failure_types.append("weak_next_action")

        return EvaluationResult(
            case_id=case.case_id,
            condition=case.condition,
            split=case.split,
            passed=passed,
            gate_violations=gate_violations,
            overall_score=round(overall, 3),
            coverage_score=round(cov, 3),
            followup_score=round(fol, 3),
            groundedness_score=round(grd, 3),
            actionability_score=round(act, 3),
            retrieval_recall=round(retrieval_recall, 3),
            retrieval_mrr=round(retrieval_mrr, 3),
            failure_types=failure_types,
        )
