"""
DecisionMesh — Individual Premise model.
Premises are tracked separately for granular drift detection.
"""
import json
import uuid
from datetime import datetime
from enum import Enum
from typing import Optional

from pydantic import BaseModel, Field
from sqlalchemy import DateTime, Float, ForeignKey, JSON, String, Text
from sqlalchemy.orm import Mapped, mapped_column, relationship

from decisionmesh.database import Base


class PremiseStatus(str, Enum):
    VALID = "valid"
    UNCERTAIN = "uncertain"
    INVALIDATED = "invalidated"
    UNVERIFIABLE = "unverifiable"


class PremiseTestability(str, Enum):
    EMPIRICALLY_TESTABLE = "empirically_testable"
    MEASURABLE = "measurable"
    SUBJECTIVE_UNTESTABLE = "subjective_untestable"


# ── Pydantic model ────────────────────────────────────────────────────────────

class Premise(BaseModel):
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    decision_id: str
    text: str
    confidence_score: float = 1.0
    status: PremiseStatus = PremiseStatus.VALID
    testability: PremiseTestability = PremiseTestability.EMPIRICALLY_TESTABLE
    weight: float = 1.0       # 0.1-3.0, used in divergence score formula
    last_checked_at: Optional[datetime] = None
    invalidated_at: Optional[datetime] = None
    invalidation_evidence: Optional[str] = None
    embedding: Optional[list[float]] = None
    requires_web_monitoring: bool = False

    model_config = {"from_attributes": True}


# ── SQLAlchemy ORM model ──────────────────────────────────────────────────────

class PremiseORM(Base):
    __tablename__ = "premises"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    decision_id: Mapped[str] = mapped_column(String(36), ForeignKey("decisions.id"), nullable=False)
    text: Mapped[str] = mapped_column(Text, nullable=False)
    confidence_score: Mapped[float] = mapped_column(Float, default=1.0)
    status: Mapped[str] = mapped_column(String(30), default=PremiseStatus.VALID.value)
    testability: Mapped[str] = mapped_column(String(40), default=PremiseTestability.EMPIRICALLY_TESTABLE.value)
    weight: Mapped[float] = mapped_column(Float, default=1.0)
    last_checked_at: Mapped[Optional[datetime]] = mapped_column(DateTime, nullable=True)
    invalidated_at: Mapped[Optional[datetime]] = mapped_column(DateTime, nullable=True)
    invalidation_evidence: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    embedding: Mapped[Optional[str]] = mapped_column(JSON, nullable=True)
    requires_web_monitoring: Mapped[bool] = mapped_column(default=False)

    # Relationships
    decision: Mapped["DecisionORM"] = relationship("DecisionORM", back_populates="premises_rel")  # noqa: F821
    observations: Mapped[list] = relationship("ObservationORM", back_populates="premise", cascade="all, delete-orphan")

    def to_pydantic(self) -> Premise:
        return Premise(
            id=self.id,
            decision_id=self.decision_id,
            text=self.text,
            confidence_score=self.confidence_score,
            status=PremiseStatus(self.status),
            testability=PremiseTestability(self.testability),
            weight=self.weight,
            last_checked_at=self.last_checked_at,
            invalidated_at=self.invalidated_at,
            invalidation_evidence=self.invalidation_evidence,
            embedding=self.embedding if isinstance(self.embedding, list) else (json.loads(self.embedding) if self.embedding else None),
            requires_web_monitoring=self.requires_web_monitoring,
        )
