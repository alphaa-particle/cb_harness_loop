"""Search over scheme sections: by words, by letters, by meaning, or all three merged.

Methods (retrieval.method in eval_config.yaml, or SYNALIGN_RETRIEVAL_METHOD):

    char_tfidf   letter runs of 3-5 characters; tolerant of typos (the original default)
    word_tfidf   words and word pairs
    hybrid       the average of the two scores above
    bm25         the standard word-ranking formula, with Hindi words kept whole
    dense        meaning: cosine similarity of Qwen3-Embedding vectors
    fusion       bm25, letter runs and meaning merged by rank (reciprocal rank fusion)

Whatever the method, retrieve_evidence() then puts each found scheme's rules first. Three
evidence settings (retrieval.* in eval_config.yaml, all off by default) refine which sections
fill the prompt:

    add_named_schemes  a scheme the question names but the evidence lacks gets its rules in
                       place of the least useful slot; nothing else changes (see _named)
    name_rules         1: the first name rules (round 4); 2: the corrected ones (round 5)
    rules_count_once   a scheme's rules take one slot however many parts they were cut into
    named_schemes      schemes the question names get all the slots
    overview_last      for a named scheme, its overview (a summary of its other sections)
                       only fills slots its specific sections leave free

The last three improved search but were undone at the end-to-end answer check (they change the
evidence of many questions, and the small answer model's answers move with any change); they stay
here, off, so the rounds in docs/results/SYNALIGN_RETRIEVAL_ROUNDS.md can be reproduced.
"""

from __future__ import annotations

import hashlib
import re
from pathlib import Path

import numpy as np
from scipy import sparse
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer

from engine.corpus import CorpusPack, load_corpus, corpus_fingerprint
from engine.domain_pack import DomainPack
from engine.schemas import RetrievedChunk

METHODS = ("char_tfidf", "word_tfidf", "hybrid", "bm25", "dense", "fusion")
DENSE_METHODS = ("dense", "fusion")
FUSION_WEIGHTS = {"bm25": 0.5, "char": 0.5, "dense": 1.0}   # word and letter matching share one vote
# Letters, digits and the combining vowel signs of Indian scripts, so that a Hindi word such
# as "किसान" stays one word instead of being cut at every vowel sign.
_WORD = re.compile(r"[\wऀ-෿]+")


def tokens(text: str) -> list[str]:
    return _WORD.findall(text.lower())


class BM25:
    """Okapi BM25 (k1=1.2, b=0.75) as one sparse matrix, so a query is a single product."""

    def __init__(self, texts: list[str], k1: float = 1.2, b: float = 0.75):
        self.vectorizer = CountVectorizer(tokenizer=tokens, lowercase=False, token_pattern=None)
        counts = self.vectorizer.fit_transform(texts).astype(np.float64).tocoo()
        lengths = np.asarray(counts.sum(axis=1)).ravel()
        n = counts.shape[0]
        df = np.bincount(counts.col, minlength=counts.shape[1])
        idf = np.log(1 + (n - df + 0.5) / (df + 0.5))
        tf = counts.data
        norm = k1 * (1 - b + b * lengths[counts.row] / max(lengths.mean(), 1e-9))
        weights = idf[counts.col] * tf * (k1 + 1) / (tf + norm)
        self.matrix = sparse.csr_matrix((weights, (counts.col, counts.row)), shape=(counts.shape[1], n))

    def scores(self, query: str) -> np.ndarray:
        return (self.vectorizer.transform([query]) @ self.matrix).toarray().ravel()


class DenseIndex:
    """Meaning vectors for every section, cached on disk by content so only changed sections are re-embedded."""

    def __init__(self, texts: list[str], embedder, cache_dir: str | Path | None = None):
        self.embedder = embedder
        model = embedder.model_id
        keys = [hashlib.sha1(f"{model}\n{t}".encode("utf-8")).hexdigest() for t in texts]
        cached: dict[str, np.ndarray] = {}
        path = None
        if cache_dir is not None:
            path = Path(cache_dir) / f"{re.sub(r'[^A-Za-z0-9_.-]', '_', model)}.npz"
            if path.is_file():
                stored = np.load(path)
                cached = dict(zip(stored["keys"].tolist(), stored["vectors"]))
        missing = [i for i, key in enumerate(keys) if key not in cached]
        # Embed in parts and save after each one, so a long first build that is interrupted
        # resumes where it stopped. The cache only ever grows: other corpora's vectors stay.
        for start in range(0, len(missing), 2048):
            part = missing[start:start + 2048]
            fresh = embedder.embed_documents([texts[i] for i in part])
            cached.update({keys[i]: fresh[n] for n, i in enumerate(part)})
            if path is not None:
                path.parent.mkdir(parents=True, exist_ok=True)
                every = list(cached)
                np.savez(path, keys=np.array(every), vectors=np.stack([cached[k] for k in every]).astype(np.float32))
        self.matrix = np.stack([cached[key] for key in keys]).astype(np.float32)
        self.embedded_now = len(missing)

    def scores(self, query: str) -> np.ndarray:
        return self.matrix @ self.embedder.embed_query(query)


def rank_of(i: int, order: np.ndarray) -> int:
    hits = np.flatnonzero(order == i)
    return int(hits[0]) if len(hits) else len(order)


class Retriever:
    """In-memory search over the sections at pack.corpus_path, built once per process.

    Chunk IDs, source text and scores keep the same meaning for the assistant,
    gold-context diagnosis and training pipeline.
    """

    def __init__(self, pack: DomainPack | CorpusPack, embedder=None, cache_dir: str | Path | None = None):
        self.documents = load_corpus(pack.corpus_path)
        self.corpus_fingerprint = corpus_fingerprint(self.documents)
        self.chunk_ids = [doc.chunk_id for doc in self.documents]
        self.chunks = [doc.text for doc in self.documents]
        self._chunk_index: dict[str, int] = {}
        for i, cid in enumerate(self.chunk_ids):
            if cid in self._chunk_index:
                raise ValueError(f"Duplicate chunk ID {cid!r}; give every section a unique ID")
            self._chunk_index[cid] = i
        # Rule sections of every scheme, so they can be put in front of the model
        # whenever any part of that scheme is retrieved.
        self._rules: dict[str, list[int]] = {}
        self._sections: dict[str, list[int]] = {}
        for i, doc in enumerate(self.documents):
            self._sections.setdefault(doc.scheme_key, []).append(i)
            if doc.is_rules:
                self._rules.setdefault(doc.scheme_key, []).append(i)

        cfg = pack.eval_config["retrieval"]
        self.top_k = int(cfg["top_k"])
        if self.top_k <= 0:
            raise ValueError("retrieval.top_k must be positive")
        self.method = cfg.get("method", "char_tfidf")
        self.min_score = float(cfg.get("min_score", 0.0))
        if self.method not in METHODS:
            raise ValueError(f"Unknown retrieval method: {self.method}")
        if not 0 <= self.min_score <= 1:
            raise ValueError("retrieval.min_score must be between 0 and 1")
        self.fusion_weights = {**FUSION_WEIGHTS, **(cfg.get("fusion_weights") or {})}
        self.rrf_k = int(cfg.get("rrf_k", 60))
        self.depth = int(cfg.get("fusion_depth", 50))
        self.rules_count_once = bool(cfg.get("rules_count_once", False))
        self.named_schemes = bool(cfg.get("named_schemes", False))
        self.overview_last = bool(cfg.get("overview_last", False))
        self.add_named_schemes = bool(cfg.get("add_named_schemes", False))
        self.name_rules = int(cfg.get("name_rules", 1))
        if self.name_rules not in (1, 2):
            raise ValueError("retrieval.name_rules must be 1 or 2")
        self._names, self._name_pairs = [], {}
        if self.named_schemes or self.add_named_schemes:
            self._names, self._name_pairs = (self._scheme_names() if self.name_rules == 1 else self._scheme_names_v2())
        # Hindi and Hinglish words mapped to the English words the documents use; word and
        # letter matching only (meaning search needs no help).
        self._glossary = {k.lower(): v for k, v in (getattr(pack, "glossary", None) or {}).items()}

        search_text = [doc.search_text for doc in self.documents]
        # No stop-word removal: real user queries are messy and multilingual;
        # character n-grams give some tolerance to typos.
        self.vectorizer = None
        if self.method in ("char_tfidf", "hybrid", "fusion"):
            self.vectorizer = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5))
        elif self.method == "word_tfidf":
            self.vectorizer = TfidfVectorizer(ngram_range=(1, 2), sublinear_tf=True)
        # TF-IDF rows already have L2 norm 1, so their dot product is cosine
        # similarity. Only the transpose is kept, in CSR form, so a query is one
        # sparse product with no per-request normalizing or transposing.
        self._search_matrix = (self.vectorizer.fit_transform(search_text).T.tocsr()
                               if self.vectorizer is not None else None)
        self._word_vectorizer = None
        self._word_search_matrix = None
        if self.method == "hybrid":
            self._word_vectorizer = TfidfVectorizer(ngram_range=(1, 2), sublinear_tf=True)
            self._word_search_matrix = self._word_vectorizer.fit_transform(search_text).T.tocsr()
        self._bm25 = BM25(search_text) if self.method in ("bm25", "fusion") else None
        self.dense = None
        if self.method in DENSE_METHODS:
            if embedder is None:
                from engine.llama_cpp import MODELS_DIR, LlamaEmbedder
                embedder = LlamaEmbedder()
                cache_dir = cache_dir or MODELS_DIR / "index"
            self.dense = DenseIndex(search_text, embedder, cache_dir)

    def _chunk(self, i: int, score: float) -> RetrievedChunk:
        doc = self.documents[i]
        return RetrievedChunk(chunk_id=doc.chunk_id, text=doc.text, score=score, title=doc.title,
                              scheme_id=doc.scheme_id, section=doc.section, source=doc.source)

    def _expand(self, query: str) -> str:
        extra = [self._glossary[t] for t in dict.fromkeys(tokens(query)) if t in self._glossary]
        return " ".join([query, *extra])

    def _char_scores(self, query: str) -> np.ndarray:
        return (self.vectorizer.transform([query]) @ self._search_matrix).toarray().ravel()

    @staticmethod
    def _top(scores: np.ndarray, k: int) -> np.ndarray:
        """Indices of the k highest positive scores, best first; ties broken the same way every time."""
        k = min(k, int(np.count_nonzero(scores > 0)))
        if k == 0:
            return np.empty(0, dtype=int)
        cutoff = np.partition(scores, len(scores) - k)[len(scores) - k]
        above = np.flatnonzero(scores > cutoff)
        tied = np.flatnonzero(scores == cutoff)[::-1][:k - len(above)]
        candidates = np.concatenate((above, tied))
        return candidates[np.lexsort((-candidates, -scores[candidates]))]

    def _rank(self, query: str, top_k: int | None):
        """Return (score per section, indices of the best k sections in order)."""
        k = self.top_k if top_k is None else top_k
        if k < 0:
            raise ValueError("top_k must not be negative")
        none = np.empty(0, dtype=int)
        if k == 0 or not query.strip():
            return None, none
        lexical = self._expand(query) if self._glossary else query

        if self.method == "fusion":
            # Merge by rank, not by score: the three scores are on different scales. The score
            # reported (and checked against min_score) is meaning similarity, which is comparable
            # across questions; the merged rank score carries no confidence information.
            meaning = self.dense.scores(query)
            if meaning.max() < self.min_score:
                return meaning, none
            fused = np.zeros(len(self.documents))
            for name, scores in (("bm25", self._bm25.scores(lexical)), ("char", self._char_scores(lexical)),
                                 ("dense", meaning)):
                for rank, i in enumerate(self._top(scores, self.depth), 1):
                    fused[i] += self.fusion_weights[name] / (self.rrf_k + rank)
            return meaning, self._top(fused, k)

        if self.method == "dense":
            sims = self.dense.scores(query)
        elif self.method == "bm25":
            sims = self._bm25.scores(lexical)
            return sims, self._top(sims, k)          # BM25 is unbounded, so min_score does not apply
        else:
            qv = self.vectorizer.transform([lexical])
            if qv.nnz == 0 and self._word_vectorizer is None:
                return None, none
            sims = (qv @ self._search_matrix).toarray().ravel()
            if self._word_vectorizer is not None:
                word_scores = (self._word_vectorizer.transform([lexical]) @ self._word_search_matrix).toarray().ravel()
                sims = 0.5 * sims + 0.5 * word_scores
        raw = sims.copy()
        sims[sims < self.min_score] = 0.0
        # Do not pad results with unrelated sections that have zero overlap.
        return raw, self._top(sims, k)

    def retrieve(self, query: str, top_k: int | None = None) -> list[RetrievedChunk]:
        """The k sections most similar to the query, best first."""
        sims, order = self._rank(query, top_k)
        return [self._chunk(i, float(sims[i])) for i in order]

    def retrieve_evidence(self, query: str, top_k: int | None = None) -> list[RetrievedChunk]:
        """The sections to put in the model's context: at most k, rules first.

        Search finds the scheme; this makes sure its rules come with it. A
        scheme's other sections (how to apply, payment...) often match a
        question better than its rule section does, and would push the rules
        out of a short context. So for every scheme found, its rule section is
        placed first, and no other section of a scheme is included without it.
        Corpora that do not label a rule section behave exactly like retrieve().
        """
        budget = self.top_k if top_k is None else top_k
        deep = self.rules_count_once or self.named_schemes
        sims, order = self._rank(query, max(budget * 10, 30) if deep else top_k)
        chosen: list[int] = []
        slots = 0

        def take(indices) -> None:
            nonlocal slots
            fresh = [j for j in indices if j not in chosen]
            if fresh and slots < budget:
                chosen.extend(fresh)
                slots += 1

        def take_rules(key: str) -> None:
            parts = self._rules.get(key, ())
            if self.rules_count_once:
                take(parts)         # cut into parts only to keep sections short; still one set of rules
            else:
                for j in parts:
                    take([j])

        named = self._named(query, order) if self.named_schemes and len(order) else []
        if named and self.documents[order[0]].scheme_key not in named:
            named.append(self.documents[order[0]].scheme_key)     # a wrong name match never pushes it out
        if named:
            named.sort(key=lambda key: (min(rank_of(i, order) for i in self._sections[key]), key))
            for key in named:
                take_rules(key)
            rank = {i: n for n, i in enumerate(order)}
            pool = sorted((i for key in named for i in self._sections[key] if not self.documents[i].is_rules),
                          key=lambda i: (rank.get(i, len(rank)), i))
            if self.overview_last:
                pool = ([i for i in pool if self.documents[i].section != "overview"]
                        + [i for i in pool if self.documents[i].section == "overview"])
            for i in pool:
                take([i])
        else:
            for i in order[:budget * 3] if deep else order:
                if slots >= budget:
                    break
                take_rules(self.documents[i].scheme_key)
                take([i])
        if self.add_named_schemes and chosen and self.name_rules == 2:
            chosen = self._add_missing(query, order, chosen, budget)
        elif self.add_named_schemes and chosen:
            # The smallest change: only a named scheme that is missing comes in, with one rules part,
            # in place of the last slot(s). Questions whose evidence already has every named scheme,
            # or that name none, keep exactly the evidence they had.
            top = self.documents[order[0]].scheme_key
            present = {self.documents[i].scheme_key for i in chosen}
            missing = [key for key in self._named(query, order) if key != top and key not in present]
            add = [self._rules[key][0] for key in missing if self._rules.get(key)][:budget - 1]
            if add:
                chosen = chosen[:budget - len(add)] + add
        return [self._chunk(i, float(sims[i])) for i in chosen]

    # --- schemes named in the question ------------------------------------------------------

    @staticmethod
    def _name_key(text: str) -> str:
        return " ".join(tokens(re.sub(r"[-/_.,()'’]", " ", text)))

    def _scheme_names(self) -> tuple[list[tuple[str, str]], dict]:
        """(name, scheme) pairs, longest first, from every section's title and aliases; and the
        consecutive word pairs of every scheme's names.

        A section's title is "<scheme name> — <heading>"; "Name (SHORT)" also gives "Name" and
        "SHORT". A name counts only if it belongs to exactly one scheme ("Disability Pension",
        shared by several, identifies none), and a one-word name only if it is not an ordinary
        word in the text of two or more other schemes.
        """
        owners: dict[str, set[str]] = {}
        pairs: dict[tuple[str, str], set[str]] = {}
        for doc in self.documents:
            full = doc.title.split(" — ")[0].strip()
            for form in (full, *doc.aliases):
                words = self._name_key(form).split()
                for pair in zip(words, words[1:]):
                    pairs.setdefault(pair, set()).add(doc.scheme_key)
            forms = {full, re.sub(r"\s*\([^)]*\)\s*$", "", full), *doc.aliases}
            short = re.search(r"\(([^)]*)\)\s*$", full)
            if short and len(short.group(1).split()) <= 4:
                forms.add(short.group(1))
            for form in forms:
                key = self._name_key(form)
                if len(key) >= 3:
                    owners.setdefault(key, set()).add(doc.scheme_key)
        in_text: dict[str, set[str]] = {}
        for doc in self.documents:
            for word in set(tokens(doc.text)):
                in_text.setdefault(word, set()).add(doc.scheme_key)
        names = [(key, next(iter(keys))) for key, keys in owners.items()
                 if len(keys) == 1 and (" " in key or len(in_text.get(key, set()) - keys) < 2)]
        return sorted(names, key=lambda pair: (-len(pair[0]), pair[0])), pairs

    def _named(self, query: str, order: np.ndarray) -> list[str]:
        """Schemes the question names: whole words, longest name first, no overlaps, best-ranked first.

        A name does not count when it is the start or end of a longer name of another scheme: if
        the question's next word (or previous word) continues it into a name of a different
        scheme ("PM Kisan" followed by "Maandhan" is part of "Pradhan Mantri Kisan Maandhan Yojana").
        """
        words = self._name_key(query).split()
        text = f" {' '.join(words)} "
        spans: list[tuple[int, int]] = []
        found: list[str] = []
        for name, key in self._names:
            start = text.find(f" {name} ")
            while start >= 0:
                end = start + len(name) + 1
                if not any(s < end and start < e for s, e in spans):
                    spans.append((start, end))
                    first = len(text[:start + 1].split())
                    parts = name.split()
                    before = words[first - 1] if first > 0 else None
                    after = words[first + len(parts)] if first + len(parts) < len(words) else None
                    longer = any(None not in pair and self._name_pairs.get(pair, set()) - {key}
                                 for pair in ((parts[-1], after), (before, parts[0])))
                    if not longer and key not in found:
                        found.append(key)
                start = text.find(f" {name} ", start + 1)
        return sorted(found, key=lambda key: (min(rank_of(i, order) for i in self._sections[key]), key))

    # --- name rules 2 (round 5) ------------------------------------------------------------

    # Words that join names rather than tell them apart ("for", "scheme", "yojana"...): a name next
    # to one of them is not the start of another scheme's longer name.
    JOINING_WORDS = {"for", "and", "of", "the", "a", "an", "in", "to", "on", "or", "with", "under", "from",
                     "me", "mein", "ki", "ka", "ke", "aur", "se", "ko", "scheme", "schemes", "yojana", "yojna"}

    @staticmethod
    def _name_key2(text: str) -> str:
        return " ".join(tokens(re.sub(r"[-/_.,()'’।॥]", " ", text)))

    def _scheme_names_v2(self) -> tuple[list[tuple[str, str]], dict]:
        """Names, longest first, from every section's title and aliases, each read the same way.

        "Name (SHORT)" and "Name (old name)" give "Name"; a parenthesised single word ("PMJJBY",
        "DAY-NRLM") is also a short form, while a parenthesised description ("Medical Treatment",
        "old name") is not a name; a " - note" tail is dropped. A name counts only if it belongs to
        one scheme. A short form counts even when other schemes mention it; any other one-word name
        only if no other scheme's text uses it.
        """
        owners: dict[str, set[str]] = {}
        shorts: dict[str, set[str]] = {}
        name_words: dict[str, set[str]] = {}
        for doc in self.documents:
            full = doc.title.split(" — ")[0].strip()
            for form in (full, *doc.aliases):
                base = re.split(r"\s+[-–]\s+", re.sub(r"\s*\([^)]*\)\s*$", "", form))[0]
                forms = {base, full} if form == full else {base}
                short = re.search(r"\(([^)\s]+)\)\s*$", form)
                if short:
                    forms.add(short.group(1))
                    shorts.setdefault(self._name_key2(short.group(1)), set()).add(doc.scheme_key)
                for f in forms:
                    key = self._name_key2(f)
                    if len(key) >= 3:
                        owners.setdefault(key, set()).add(doc.scheme_key)
                        for word in key.split():
                            name_words.setdefault(word, set()).add(doc.scheme_key)
        in_text: dict[str, set[str]] = {}
        for doc in self.documents:
            for word in set(tokens(re.sub(r"[।॥]", " ", doc.text))):
                in_text.setdefault(word, set()).add(doc.scheme_key)
        names = []
        for key, keys in owners.items():
            if len(keys) != 1:
                continue
            if " " not in key and not (shorts.get(key) == keys or not in_text.get(key, set()) - keys):
                continue
            names.append((key, next(iter(keys))))
        # pairs inside every scheme's names (ambiguous names too), for the longer-name check, and the
        # words used in many schemes' names, which join names rather than tell them apart
        all_pairs: dict[tuple[str, str], set[str]] = {}
        for key, keys in owners.items():
            for pair in zip(key.split(), key.split()[1:]):
                all_pairs.setdefault(pair, set()).update(keys)
        self._common_name_words = {w for w, keys in name_words.items() if len(keys) > 3} | self.JOINING_WORDS
        return sorted(names, key=lambda pair: (-len(pair[0]), pair[0])), all_pairs

    def _named_v2(self, query: str) -> list[str]:
        """Schemes the question names, in the order it names them: whole words, longest first, no overlaps.

        A name does not count when the next (or previous) word of the question is a distinctive word
        that, together with the name's edge word, belongs to a name of a different scheme ("PM Kisan"
        followed by "Maandhan"); joining words ("for", "scheme") never decide this. A name that does not
        count leaves its words free for shorter names.
        """
        words = self._name_key2(query).split()
        text = f" {' '.join(words)} "
        spans: list[tuple[int, int]] = []
        found: list[tuple[int, str]] = []
        for name, key in self._names:
            start = text.find(f" {name} ")
            while start >= 0:
                end = start + len(name) + 1
                if not any(s < end and start < e for s, e in spans):
                    first = len(text[:start + 1].split())
                    parts = name.split()
                    before = words[first - 1] if first > 0 else None
                    after = words[first + len(parts)] if first + len(parts) < len(words) else None
                    longer = any(word and word not in self._common_name_words
                                 and self._name_pairs.get(pair, set()) - {key}
                                 for word, pair in ((after, (parts[-1], after)), (before, (before, parts[0]))))
                    if not longer:
                        spans.append((start, end))
                        found.append((start, key))
                start = text.find(f" {name} ", start + 1)
        return list(dict.fromkeys(key for _, key in sorted(found)))

    def _add_missing(self, query: str, order: np.ndarray, chosen: list[int], budget: int) -> list[int]:
        """Bring in each named scheme the evidence lacks, in the least useful slot; change nothing else.

        The slot given up is, in turn: a section of a scheme that is neither named nor ranked first;
        an overview (a summary of sections that are there); another section that is not rules; a
        second rules part. The rules of a scheme that is named or ranked first are never all removed.
        A named scheme without a rules section brings its best-ranked section instead.
        """
        top = self.documents[order[0]].scheme_key
        named = self._named_v2(query)
        wanted = set(named) | {top}
        chosen = list(chosen)
        for key in [k for k in named if k != top and k not in {self.documents[i].scheme_key for i in chosen}]:
            parts = self._rules.get(key) or sorted(self._sections[key], key=lambda i: (rank_of(i, order), i))
            new = parts[0]
            if len(chosen) < budget:
                chosen.append(new)
                continue
            victim = self._victim(chosen, wanted)
            if victim is None:
                break
            chosen[victim] = new
        return chosen

    def _victim(self, chosen: list[int], wanted: set[str]) -> int | None:
        doc = lambda n: self.documents[chosen[n]]  # noqa: E731
        last_first = range(len(chosen) - 1, -1, -1)
        for n in last_first:
            if doc(n).scheme_key not in wanted:
                return n
        for n in last_first:
            if doc(n).section == "overview" and sum(doc(m).scheme_key == doc(n).scheme_key for m in range(len(chosen))) > 1:
                return n
        for n in last_first:
            if not doc(n).is_rules:
                return n
        for n in last_first:
            if sum(doc(m).scheme_key == doc(n).scheme_key and doc(m).is_rules for m in range(len(chosen))) > 1:
                return n
        return None

    def get_chunks_by_ids(self, chunk_ids: list[str]) -> list[RetrievedChunk]:
        """Fetch specific chunks directly — used for oracle-retrieval diagnosis."""
        out = []
        for cid in chunk_ids:
            i = self._chunk_index.get(cid)
            if i is not None:
                out.append(self._chunk(i, 1.0))
        return out
