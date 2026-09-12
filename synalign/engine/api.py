from fastapi import FastAPI
from pydantic import BaseModel

from audit import build_components

app = FastAPI(title="SYNALIGN Assistant API")

pack, retriever, assistant, evaluator = build_components(backend="qwen")

LOW_CONFIDENCE_THRESHOLD = 0.10
RAG_THRESHOLD = 0.15


class AskRequest(BaseModel):
    question: str


@app.get("/")
def home():
    return {
        "status": "running",
        "domain": pack.name,
        "assistant": type(assistant).__name__,
        "routes": ["/ask"],
    }


@app.post("/ask")
def ask(req: AskRequest):
    question = req.question.strip()

    if not question:
        return {
            "mode": "empty",
            "answer": "Please ask a question.",
            "evidence": [],
        }

    # Retrieve first only to estimate whether this is a domain question.
    chunks = retriever.retrieve(question)
    top_score = chunks[0].score if chunks else 0.0

    # Clearly not a welfare/domain query.
    if top_score < LOW_CONFIDENCE_THRESHOLD:
        return {
            "mode": "non_domain",
            "top_score": top_score,
            "answer": (
                "I can help with welfare-scheme eligibility questions from the available documents. "
                "Please ask about a scheme, pension, eShram, PMSYM, eligibility, income, age, or worker status."
            ),
            "evidence": [],
PY  }   ],  for c in answer.retrieved_chunks: c.score}ore}rk type, and EPFO/ESIC status."Shram? "
(venv) (base) pangea@VeronaRig:~/vaibhav/cb_harness_loop/cb_harness_loop/synalign$ PYTHONPATH="$PWD:$PWD/engine" python -m py_compile engine/api.py
(venv) (base) pangea@VeronaRig:~/vaibhav/cb_harness_loop/cb_harness_loop/synalign$ cd ~/vaibhav/cb_harness_loop/cb_harness_loop/synalign
source ../../venv/bin/activate

PYTHONPATH="$PWD:$PWD/engine" python - <<'PY'
from audit import build_components

pack, retriever, assistant, evaluator = build_components(backend="qwen")

tests = [
    "hi",
    "thanks",
    "who are you",
    "what is linux",
    "explain machine learning",
    "can I get pension",
    "am I eligible for pension scheme",
    "I am 35 and earn 14000",
    "I am 35 and earn 14000. I am an unorganised worker. No EPFO. Can I apply for PMSYM?",
    "Can I register for eShram?",
    "I am a self employed worker age 45. Can I get eShram?",
]

for q in tests:
    chunks = retriever.retrieve(q)
    top = chunks[0].score if chunks else 0
    print(f"{top:.3f} | {q}")
