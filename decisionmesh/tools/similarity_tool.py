"""
DecisionMesh — Similarity Tool
Vector similarity search over decision embeddings.
"""
from pydantic import BaseModel, Field


class SimilaritySearchInput(BaseModel):
    query_text: str = Field(description="The text to search for semantically similar decisions.")
    limit: int = Field(default=5, ge=1, le=20, description="Maximum number of similar decisions to return.")
    min_similarity: float = Field(default=0.5, ge=0.0, le=1.0, description="Minimum cosine similarity threshold.")


SIMILARITY_SEARCH_TOOL: dict = {
    "name": "semantic_search_decisions",
    "description": (
        "Search for decisions semantically similar to a query text using local embeddings. "
        "Useful for finding decisions built on similar premises that may be affected by the "
        "same divergence (compounding risk analysis)."
    ),
    "input_schema": SimilaritySearchInput.model_json_schema(),
}


async def semantic_search_decisions(
    query_text: str,
    limit: int,
    min_similarity: float,
    db_session,
) -> list[dict]:
    """Find decisions semantically similar to the query text."""
    from sqlalchemy import select
    from decisionmesh.models.decision import DecisionORM, DecisionStatus
    from decisionmesh.tools.embedding_tool import embed_text, cosine_similarity

    query_embedding = embed_text(query_text)
    if query_embedding is None:
        return []

    result = await db_session.execute(
        select(DecisionORM).where(
            DecisionORM.status.in_([DecisionStatus.ACTIVE.value, DecisionStatus.DRIFTED.value])
        )
    )
    decisions = result.scalars().all()

    scored = []
    for d in decisions:
        if d.embedding is None:
            continue
        emb = d.embedding if isinstance(d.embedding, list) else None
        if emb is None:
            continue
        sim = cosine_similarity(query_embedding, emb)
        if sim >= min_similarity:
            scored.append((sim, d))

    scored.sort(key=lambda x: x[0], reverse=True)
    return [
        {**d.to_pydantic().model_dump(mode="json"), "similarity_score": round(sim, 4)}
        for sim, d in scored[:limit]
    ]
