"""Portable corpus loading, independent of synthetic profiles and gold labels."""

from dataclasses import dataclass, field
import hashlib
import json
from pathlib import Path
import re


_ID_PATTERN = re.compile(r"^\s*[A-Za-z ]*ID:\s*(\S+)\s*$", re.MULTILINE)
_CHUNK_ID_PATTERN = re.compile(r"^\s*Chunk ID:\s*(\S+)\s*$", re.MULTILINE)
_SCHEME_ID_PATTERN = re.compile(r"^\s*Scheme ID:\s*(\S+)\s*$", re.MULTILINE)
_SECTION_PATTERN = re.compile(r"^\s*Section:\s*(\S+)\s*$", re.MULTILINE)
RULES_SECTION = "rules"


@dataclass(frozen=True)
class CorpusDocument:
    chunk_id: str
    text: str
    title: str = ""
    aliases: tuple[str, ...] = ()
    scheme_id: str = ""
    source: str = ""
    section: str = ""

    @property
    def search_text(self) -> str:
        return "\n".join(part for part in (self.title, *self.aliases, self.text) if part)

    @property
    def scheme_key(self) -> str:
        """Sections sharing a scheme_id belong together; otherwise each stands alone."""
        return self.scheme_id or self.chunk_id

    @property
    def is_rules(self) -> bool:
        return self.section.strip().lower() == RULES_SECTION


@dataclass
class CorpusPack:
    """Retrieval-only adapter; never invents ground truth for an external corpus."""

    corpus_path: Path
    top_k: int = 3
    method: str = "char_tfidf"
    min_score: float = 0.0
    eval_config: dict = field(init=False)

    def __post_init__(self):
        self.corpus_path = Path(self.corpus_path).expanduser().resolve()
        self.eval_config = {"retrieval": {"top_k": self.top_k, "method": self.method,
                                           "min_score": self.min_score}}


def load_corpus(source: str | Path) -> list[CorpusDocument]:
    """Read a JSONL file, a Markdown file, or a directory of Markdown files.

    A directory's corpus.jsonl takes precedence, allowing BEIR-format datasets
    to include separate queries/qrels without indexing those as evidence.
    """
    path = Path(source).expanduser().resolve()
    if not path.exists():
        raise ValueError(f"Corpus path does not exist: {path}")
    if path.is_dir():
        files = [path / "corpus.jsonl"] if (path / "corpus.jsonl").is_file() else sorted(path.rglob("*.md"))
    else:
        files = [path]
    documents = []
    seen = {}
    for file in files:
        if file.suffix.lower() == ".jsonl":
            entries = []
            with file.open(encoding="utf-8") as stream:
                for line_number, line in enumerate(stream, 1):
                    if not line.strip():
                        continue
                    location = f"{file}:{line_number}"
                    try:
                        record = json.loads(line)
                    except json.JSONDecodeError as exc:
                        raise ValueError(f"Invalid JSON at {location}: {exc.msg}") from exc
                    if not isinstance(record, dict):
                        raise ValueError(f"Expected an object at {location}")
                    cid = record.get("chunk_id", record.get("_id"))
                    if "chunk_id" in record and "_id" in record and record["chunk_id"] != record["_id"]:
                        raise ValueError(f"Conflicting chunk_id and _id at {location}")
                    body = record.get("text")
                    if not isinstance(cid, str) or not cid.strip():
                        raise ValueError(f"A nonempty string chunk_id (or BEIR _id) is required at {location}")
                    if not isinstance(body, str) or not body.strip():
                        raise ValueError(f"Nonempty source text is required at {location}")
                    aliases = record.get("aliases", [])
                    if not isinstance(aliases, list) or not all(isinstance(a, str) for a in aliases):
                        raise ValueError(f"aliases must be a list of strings at {location}")
                    for key in ("title", "scheme_id", "source", "section"):
                        if not isinstance(record.get(key, ""), str):
                            raise ValueError(f"{key} must be a string at {location}")
                    entries.append((CorpusDocument(cid, body, record.get("title", ""), tuple(aliases),
                                                   record.get("scheme_id", ""), record.get("source", ""),
                                                   record.get("section", "")), location))
        elif file.suffix.lower() == ".md":
            entries = []
            relative = file.relative_to(path) if path.is_dir() else Path(file.name)
            # Retain legacy IDs for root-level documents; avoid nested stem collisions.
            stem = relative.with_suffix("").as_posix().replace("/", "::")
            for index, raw in enumerate(file.read_text(encoding="utf-8").split("---")):
                section = raw.strip()
                if section:
                    # A lone ID line names the section (legacy). A section that carries
                    # both "Chunk ID:" and "Scheme ID:" is one part of a larger scheme.
                    chunk, scheme = _CHUNK_ID_PATTERN.search(section), _SCHEME_ID_PATTERN.search(section)
                    match = chunk or _ID_PATTERN.search(section)
                    cid = match.group(1) if match else f"{stem}_{index}"
                    kind = _SECTION_PATTERN.search(section)
                    entries.append((CorpusDocument(cid, section, scheme_id=scheme.group(1) if chunk and scheme else "",
                                                   section=kind.group(1) if kind else ""),
                                    f"{file}:section {index}"))
        else:
            raise ValueError(f"Unsupported corpus format {file.suffix!r}; use Markdown or JSONL")
        for document, location in entries:
            if document.chunk_id in seen:
                raise ValueError(f"Duplicate chunk ID {document.chunk_id!r}: {seen[document.chunk_id]} and {location}")
            seen[document.chunk_id] = location
            documents.append(document)
    if not documents:
        raise ValueError(f"No document sections found in {path}")
    return documents


def corpus_fingerprint(documents: list[CorpusDocument]) -> str:
    digest = hashlib.sha256()
    for doc in documents:
        # The section label is appended only when present, so corpora that do
        # not use it keep the fingerprint they had before it existed.
        fields = [doc.chunk_id, doc.text, doc.title, doc.aliases, doc.scheme_id, doc.source]
        digest.update(json.dumps(fields + ([doc.section] if doc.section else []),
                                 ensure_ascii=False, separators=(",", ":")).encode("utf-8"))
        digest.update(b"\n")
    return digest.hexdigest()
