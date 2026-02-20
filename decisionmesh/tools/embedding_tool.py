"""
DecisionMesh — Embedding Tool
Local sentence-transformer embeddings. Never calls any external API.
Model: all-MiniLM-L6-v2 (384 dimensions)
"""
from typing import Optional
from pydantic import BaseModel, Field

from decisionmesh.config import settings


class EmbedTextInput(BaseModel):
    text: str = Field(description="The text to embed using the local sentence-transformer model.")


EMBED_TEXT_TOOL: dict = {
    "name": "embed_text",
    "description": (
        "Generate a local embedding vector for a text string. "
        "Uses the all-MiniLM-L6-v2 model locally — no external API calls. "
        "Returns a 384-dimensional float vector."
    ),
    "input_schema": EmbedTextInput.model_json_schema(),
}

# Module-level model cache
_model = None


def get_embedding_model():
    """Lazy-load the sentence-transformer model (cached after first load)."""
    global _model
    if _model is None:
        try:
            from sentence_transformers import SentenceTransformer
            _model = SentenceTransformer(settings.EMBEDDING_MODEL)
        except ImportError:
            _model = None
    return _model


def embed_text(text: str) -> Optional[list[float]]:
    """Embed text using the local sentence-transformer model."""
    model = get_embedding_model()
    if model is None:
        return None
    try:
        embedding = model.encode(text, convert_to_numpy=True)
        return embedding.tolist()
    except Exception:
        return None


def cosine_similarity(vec_a: list[float], vec_b: list[float]) -> float:
    """Compute cosine similarity between two embedding vectors."""
    try:
        import numpy as np
        a = np.array(vec_a)
        b = np.array(vec_b)
        norm_a = np.linalg.norm(a)
        norm_b = np.linalg.norm(b)
        if norm_a == 0 or norm_b == 0:
            return 0.0
        return float(np.dot(a, b) / (norm_a * norm_b))
    except Exception:
        return 0.0
