"""
DecisionMesh — Divergence Event model.
A drift event: when reality has moved away from the decision's premises.
"""
import json
import uuid
from datetime import datetime
from enum import Enum
from typing import Optional

from pydantic import BaseModel, Field
from sqlalchemy import Boolean, DateTime, Float, ForeignKey, JSON, String, Table, Text, Column
from sqlalchemy.orm import Mapped, mapped_column, relationship

from decisionmesh.database import Base


class DivergenceSeverity(str, Enum):
    LOW = "low"           # Worth noting, not urgent
    MEDIUM = "medium"     # Consider reviewing
    HIGH = "high"         # Premises significantly undermined
    CRITICAL = "critical" # Original decision logic is broken


# Junction table for divergence_events ↔ premises
divergence_premises_table = Table(
    "divergence_premises",
    Base.metadata,
    Column("divergence_event_id", String(36), ForeignKey("divergence_events.id"), primary_key=True),
    Column("premise_id", String(36), ForeignKey("premises.id"), primary_key=True),
)


# ── Pydantic model ────────────────────────────────────────────────────────────

class DivergenceEvent(BaseModel):
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    decision_id: str
    detected_at: datetime = Field(default_factory=datetime.utcnow)
    severity: DivergenceSeverity
    affected_premise_ids: list[str]
    summary: str
    evidence: list[str]
    divergence_score: float     # 0.0-1.0 composite drift magnitude
    acknowledged_by_user: bool = False
    acknowledged_at: Optional[datetime] = None

    model_config = {"from_attributes": True}


# ── SQLAlchemy ORM model ──────────────────────────────────────────────────────

class DivergenceEventORM(Base):
    __tablename__ = "divergence_events"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    decision_id: Mapped[str] = mapped_column(String(36), ForeignKey("decisions.id"), nullable=False)
    detected_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)
    severity: Mapped[str] = mapped_column(String(20), nullable=False)
    affected_premise_ids: Mapped[Optional[str]] = mapped_column(JSON, nullable=True)
    summary: Mapped[str] = mapped_column(Text, nullable=False)
    evidence: Mapped[Optional[str]] = mapped_column(JSON, nullable=True)
    divergence_score: Mapped[float] = mapped_column(Float, nullable=False)
    acknowledged_by_user: Mapped[bool] = mapped_column(Boolean, default=False)
    acknowledged_at: Mapped[Optional[datetime]] = mapped_column(DateTime, nullable=True)

    # Relationships
    decision: Mapped["DecisionORM"] = relationship("DecisionORM", back_populates="divergence_events_rel")  # noqa: F821
    counterfactuals: Mapped[list] = relationship("CounterfactualORM", back_populates="divergence_event", cascade="all, delete-orphan")

    def to_pydantic(self) -> DivergenceEvent:
        return DivergenceEvent(
            id=self.id,
            decision_id=self.decision_id,
            detected_at=self.detected_at,
            severity=DivergenceSeverity(self.severity),
            affected_premise_ids=self.affected_premise_ids if isinstance(self.affected_premise_ids, list) else (json.loads(self.affected_premise_ids) if self.affected_premise_ids else []),
            summary=self.summary,
            evidence=self.evidence if isinstance(self.evidence, list) else (json.loads(self.evidence) if self.evidence else []),
            divergence_score=self.divergence_score,
            acknowledged_by_user=self.acknowledged_by_user,
            acknowledged_at=self.acknowledged_at,
        )
