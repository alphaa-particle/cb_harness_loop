from fastapi import FastAPI
from pydantic import BaseModel

from engine.audit import build_components

app = FastAPI(title="SYNALIGN-wrapped Assistant API")
pack, retriever, assistant, evaluator = build_components()


class AskRequest(BaseModel):
    question: str


@app.get("/")
def home():
    return {"message": "Assistant API is running", "domain": pack.name}


@app.post("/ask")
def ask(req: AskRequest):
    answer = assistant.answer(req.question)
    return {
        "answer": answer.answer_text,
        "evidence": [
            {"chunk_id": c.chunk_id, "score": c.score}
            for c in answer.retrieved_chunks
        ],
    }
