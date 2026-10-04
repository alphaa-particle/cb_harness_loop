from dataclasses import dataclass, field, asdict
from typing import Any


@dataclass
class TestCase:
    case_id: str
    user_id: int
    split: str                      # train / dev / test
    condition: str                  # clean / vague / ...
    question: str
    profile: dict[str, Any]         # full synthetic user
    visible: dict[str, Any]         # what the question revealed (None = hidden/unknown)
    ground_truth: dict[str, Any]

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class RetrievedChunk:
    chunk_id: str
    text: str
    score: float
    title: str = ""
    scheme_id: str = ""             # sections of one scheme share this
    section: str = ""               # "rules" marks the eligibility-rule section
    source: str = ""                # the official page the section was written from, if known


@dataclass
class AssistantAnswer:
    answer_text: str
    retrieved_chunks: list[RetrievedChunk] = field(default_factory=list)
    # answered / no_evidence / unclear / blocked — see engine/enforcement.py
    mode: str = "answered"


@dataclass
class EvaluationResult:
    case_id: str
    condition: str
    split: str
    passed: bool                    # score >= pass threshold AND no gate fired
    gate_violations: list[str]
    overall_score: float
    coverage_score: float
    followup_score: float
    groundedness_score: float
    actionability_score: float
    retrieval_recall: float         # gold chunks found in top-k
    retrieval_mrr: float
    failure_types: list[str]

    def to_dict(self) -> dict:
        return asdict(self)
