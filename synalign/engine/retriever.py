import re

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from engine.domain_pack import DomainPack
from engine.schemas import RetrievedChunk

_ID_PATTERN = re.compile(r"^\s*[A-Za-z ]*ID:\s*(\S+)\s*$", re.MULTILINE)


def _chunk_documents(pack: DomainPack) -> list[tuple[str, str]]:
    """Return (chunk_id, chunk_text) pairs for every section in every document.

    Sections are separated by '---'. Each section should contain an ID line
    such as 'Scheme ID: scheme_pmsym'. Sections without one get a positional ID.
    """
    chunks: list[tuple[str, str]] = []
    for doc_path in sorted(pack.documents_dir.glob("**/*.md")):
        text = doc_path.read_text(encoding="utf-8")
        for i, raw in enumerate(text.split("---")):
            section = raw.strip()
            if not section:
                continue
            match = _ID_PATTERN.search(section)
            chunk_id = match.group(1) if match else f"{doc_path.stem}_{i}"
            chunks.append((chunk_id, section))
    return chunks


class TfidfRetriever:
    """Beginner-friendly retriever. Swap for embeddings in Stage 2 —
    the interface (retrieve(query, top_k) -> list[RetrievedChunk]) stays the same."""

    def __init__(self, pack: DomainPack):
        pairs = _chunk_documents(pack)
        self.chunk_ids = [cid for cid, _ in pairs]
        self.chunks = [text for _, text in pairs]
        # No stop-word removal: real user queries are messy and multilingual;
        # character n-grams give some tolerance to typos.
        self.vectorizer = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5))
        self.matrix = self.vectorizer.fit_transform(self.chunks)
        self.top_k = int(pack.eval_config["retrieval"]["top_k"])

    def retrieve(self, query: str, top_k: int | None = None) -> list[RetrievedChunk]:
        k = top_k or self.top_k
        qv = self.vectorizer.transform([query])
        sims = cosine_similarity(qv, self.matrix).flatten()
        order = sims.argsort()[::-1][:k]
        return [
            RetrievedChunk(chunk_id=self.chunk_ids[i], text=self.chunks[i], score=float(sims[i]))
            for i in order
        ]

    def get_chunks_by_ids(self, chunk_ids: list[str]) -> list[RetrievedChunk]:
        """Fetch specific chunks directly — used for oracle-retrieval diagnosis."""
        out = []
        for cid in chunk_ids:
            if cid in self.chunk_ids:
                i = self.chunk_ids.index(cid)
                out.append(RetrievedChunk(chunk_id=cid, text=self.chunks[i], score=1.0))
        return out
