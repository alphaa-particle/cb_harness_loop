"""Single-node API. Index the configured corpus once; load a model only for /ask.

/ui        the web page
/search    the sections most similar to a question
/context   the exact prompt the model would be given (rules first), without running it
/ask       the answer, with the evidence it was built from
/feedback  thumbs up or down from the page, appended to data/feedback/feedback.jsonl
"""

from contextlib import asynccontextmanager
from dataclasses import asdict
from datetime import datetime, timezone
import json
from threading import Lock

from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from pydantic import BaseModel, Field

from engine.assistant import BACKENDS, NO_MODEL_BACKENDS, make_assistant
from engine.config import ACTIVE_DOMAIN, ASSISTANT_BACKEND, ROOT_DIR
from engine.domain_pack import DomainPack
from engine.enforcement import NO_EVIDENCE, build_messages
from engine.retriever import Retriever


class AskRequest(BaseModel):
    question: str = Field(max_length=20000)


class SearchRequest(AskRequest):
    top_k: int | None = Field(default=None, ge=1, le=50)


class FeedbackRequest(BaseModel):
    question: str = Field(max_length=20000)
    answer: str = Field(max_length=50000)
    mode: str = Field(max_length=40)
    rating: str = Field(pattern="^(up|down)$")
    note: str = Field(default="", max_length=2000)
    evidence_ids: list[str] = Field(default_factory=list, max_length=50)


UI_PAGE = ROOT_DIR / "ui" / "index.html"
FEEDBACK_FILE = ROOT_DIR / "data" / "feedback" / "feedback.jsonl"


def create_app(corpus_path=None, backend=None, assistant=None, feedback_file=None):
    backend = backend or ASSISTANT_BACKEND
    if backend not in BACKENDS:
        raise ValueError(f"Unknown assistant backend: {backend}")
    assistant_lock, feedback_lock = Lock(), Lock()
    feedback_file = feedback_file or FEEDBACK_FILE

    @asynccontextmanager
    async def lifespan(app):
        app.state.pack = DomainPack(ACTIVE_DOMAIN, corpus_path=corpus_path)
        app.state.retriever = Retriever(app.state.pack)
        app.state.assistant = assistant
        yield

    app = FastAPI(title="SYNALIGN Assistant API", lifespan=lifespan)

    @app.get("/")
    def home():
        model_ready, model = backend not in NO_MODEL_BACKENDS, None
        if backend == "llama_cpp":
            from engine.llama_cpp import DECODER_FILE, DECODER_URL, is_ready
            model_ready, model = is_ready(DECODER_URL), DECODER_FILE.removesuffix(".gguf")
        return {"status": "running", "domain": app.state.pack.name, "assistant_backend": backend,
                "model": model, "model_ready": model_ready,
                "chunks": len(app.state.retriever.chunk_ids),
                "retrieval_method": app.state.retriever.method,
                "top_k": app.state.retriever.top_k, "min_score": app.state.retriever.min_score,
                "corpus_sha256": app.state.retriever.corpus_fingerprint,
                "routes": ["/ui", "/search", "/context", "/ask", "/feedback"]}

    @app.get("/ui", response_class=HTMLResponse)
    def ui():
        return HTMLResponse(UI_PAGE.read_text(encoding="utf-8"))

    @app.post("/feedback")
    def feedback(req: FeedbackRequest):
        # Kept on this machine only (the folder is git-ignored); it can hold what people typed about themselves.
        row = {"time": datetime.now(timezone.utc).isoformat(timespec="seconds"), "domain": app.state.pack.name,
               "corpus_sha256": app.state.retriever.corpus_fingerprint, **req.model_dump()}
        with feedback_lock:
            feedback_file.parent.mkdir(parents=True, exist_ok=True)
            with feedback_file.open("a", encoding="utf-8") as stream:
                stream.write(json.dumps(row, ensure_ascii=False) + "\n")
        return {"saved": True}

    @app.post("/search")
    def search(req: SearchRequest):
        chunks = app.state.retriever.retrieve(req.question, top_k=req.top_k)
        return {"mode": "retrieved" if chunks else "no_match",
                "corpus_sha256": app.state.retriever.corpus_fingerprint,
                "evidence": [asdict(chunk) for chunk in chunks]}

    @app.post("/context")
    def context(req: SearchRequest):
        question = req.question.strip()
        chunks = app.state.retriever.retrieve_evidence(question, top_k=req.top_k)
        return {"mode": "evidence" if chunks else NO_EVIDENCE,
                "corpus_sha256": app.state.retriever.corpus_fingerprint,
                "evidence": [asdict(chunk) for chunk in chunks],
                "messages": build_messages(app.state.pack, question, chunks) if chunks else []}

    def get_assistant():
        with assistant_lock:
            if app.state.assistant is None:
                app.state.assistant = make_assistant(backend, app.state.pack, app.state.retriever)
            return app.state.assistant

    @app.post("/ask")
    def ask(req: AskRequest):
        question = req.question.strip()
        if not question:
            return {"mode": "empty", "answer": "Please ask a question.", "evidence": []}
        try:
            selected = get_assistant()
        except (ImportError, OSError) as exc:
            raise HTTPException(status_code=503, detail="Assistant dependencies or local model are unavailable; /search and /context remain available.") from exc
        answer = selected.answer(question)
        return {"mode": answer.mode, "backend": backend, "answer": answer.answer_text,
                "evidence": [asdict(chunk) for chunk in answer.retrieved_chunks],
                "corpus_sha256": app.state.retriever.corpus_fingerprint}

    return app


app = create_app()
