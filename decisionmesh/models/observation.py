"""
DecisionMesh — Observation model.
What the Condition Monitor found on each run.
"""
import json
import uuid
from datetime import datetime
from typing import Optional

from pydantic import BaseModel, Field
from sqlalchemy import Boolean, DateTime, Float, ForeignKey, JSON, String, Text
from sqlalchemy.orm import Mapped, mapped_column, relationship

from decisionmesh.database import Base


# ── Pydantic model ────────────────────────────────────────────────────────────

class Observation(BaseModel):
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    decision_id: str
    premise_id: str
    observed_at: datetime = Field(default_factory=datetime.utcnow)
    observation_text: str
    source_urls: list[str] = Field(default_factory=list)
    relevance_score: float          # 0.0-1.0
    supports_premise: Optional[bool] = None  # True/False/None (ambiguous)
    agent_reasoning: str            # Full think-tool output

    model_config = {"from_attributes": True}


# ── SQLAlchemy ORM model ──────────────────────────────────────────────────────

class ObservationORM(Base):
    __tablename__ = "observations"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    decision_id: Mapped[str] = mapped_column(String(36), ForeignKey("decisions.id"), nullable=False)
    premise_id: Mapped[str] = mapped_column(String(36), ForeignKey("premises.id"), nullable=False)
    observed_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)
    observation_text: Mapped[str] = mapped_column(Text, nullable=False)
    source_urls: Mapped[Optional[str]] = mapped_column(JSON, nullable=True)
    relevance_score: Mapped[float] = mapped_column(Float, nullable=False)
    supports_premise: Mapped[Optional[bool]] = mapped_column(Boolean, nullable=True)
    agent_reasoning: Mapped[str] = mapped_column(Text, nullable=False)

    # Relationships
    decision: Mapped["DecisionORM"] = relationship("DecisionORM", back_populates="observations_rel")  # noqa: F821
    premise: Mapped["PremiseORM"] = relationship("PremiseORM", back_populates="observations")  # noqa: F821

    def to_pydantic(self) -> Observation:
        return Observation(
            id=self.id,
            decision_id=self.decision_id,
            premise_id=self.premise_id,
            observed_at=self.observed_at,
            observation_text=self.observation_text,
            source_urls=self.source_urls if isinstance(self.source_urls, list) else (json.loads(self.source_urls) if self.source_urls else []),
            relevance_score=self.relevance_score,
            supports_premise=self.supports_premise,
            agent_reasoning=self.agent_reasoning,
        )
