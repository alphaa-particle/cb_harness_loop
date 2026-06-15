from engine.schemas import RetrievedChunk


def recall_at_k(retrieved: list[RetrievedChunk], gold_chunk_ids: list[str]) -> float:
    if not gold_chunk_ids:
        return 1.0
    got = {c.chunk_id for c in retrieved}
    return sum(1 for g in gold_chunk_ids if g in got) / len(gold_chunk_ids)


def mrr(retrieved: list[RetrievedChunk], gold_chunk_ids: list[str]) -> float:
    if not gold_chunk_ids:
        return 1.0
    for rank, chunk in enumerate(retrieved, start=1):
        if chunk.chunk_id in gold_chunk_ids:
            return 1.0 / rank
    return 0.0
