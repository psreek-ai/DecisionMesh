"""
DecisionMesh — Core Decision DNA model.
The central entity of the entire system.
"""
import json
import uuid
from datetime import datetime
from enum import Enum
from typing import Optional

from pydantic import BaseModel, Field
from sqlalchemy import JSON, Boolean, DateTime, Integer, String, Text
from sqlalchemy.orm import Mapped, mapped_column, relationship

from decisionmesh.database import Base


class DecisionStatus(str, Enum):
    ACTIVE = "active"          # Premises being monitored
    DRIFTED = "drifted"        # Significant premise divergence detected
    REVISED = "revised"        # User has acknowledged and updated
    CLOSED = "closed"          # Decision lifecycle complete
    PAUSED = "paused"          # Monitoring paused by user


# ── Pydantic model ────────────────────────────────────────────────────────────

class DecisionDNA(BaseModel):
    """The complete genetic blueprint of a decision."""
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    title: str
    full_narrative: str
    premises: list[str]
    predicted_outcomes: list[str]
    triggering_conditions: list[str]
    decision_date: datetime
    review_interval_days: int = 30
    status: DecisionStatus = DecisionStatus.ACTIVE
    domain: str = "general"
    tags: list[str] = Field(default_factory=list)
    embedding: Optional[list[float]] = None  # Local embedding of full_narrative
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)
    web_monitoring_enabled: bool = False  # Explicit opt-in for web search

    model_config = {"from_attributes": True}


# ── SQLAlchemy ORM model ──────────────────────────────────────────────────────

class DecisionORM(Base):
    __tablename__ = "decisions"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    title: Mapped[str] = mapped_column(String(500), nullable=False)
    full_narrative: Mapped[str] = mapped_column(Text, nullable=False)
    premises: Mapped[str] = mapped_column(JSON, nullable=False, default=list)   # JSON array
    predicted_outcomes: Mapped[str] = mapped_column(JSON, nullable=False, default=list)
    triggering_conditions: Mapped[str] = mapped_column(JSON, nullable=False, default=list)
    decision_date: Mapped[datetime] = mapped_column(DateTime, nullable=False)
    review_interval_days: Mapped[int] = mapped_column(Integer, default=30)
    status: Mapped[str] = mapped_column(String(20), default=DecisionStatus.ACTIVE.value)
    domain: Mapped[str] = mapped_column(String(100), default="general")
    tags: Mapped[Optional[str]] = mapped_column(JSON, default=list)
    # Embedding stored as JSON array of floats (SQLite-compatible)
    # On PostgreSQL, migrate to pgvector VECTOR(384)
    embedding: Mapped[Optional[str]] = mapped_column(JSON, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)
    updated_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    web_monitoring_enabled: Mapped[bool] = mapped_column(Boolean, default=False)

    # Relationships
    premises_rel: Mapped[list] = relationship("PremiseORM", back_populates="decision", cascade="all, delete-orphan")
    observations_rel: Mapped[list] = relationship("ObservationORM", back_populates="decision", cascade="all, delete-orphan")
    divergence_events_rel: Mapped[list] = relationship("DivergenceEventORM", back_populates="decision", cascade="all, delete-orphan")
    counterfactuals_rel: Mapped[list] = relationship("CounterfactualORM", back_populates="decision", cascade="all, delete-orphan")
    agent_logs_rel: Mapped[list] = relationship("AgentLogORM", back_populates="decision", cascade="all, delete-orphan")

    def to_pydantic(self) -> DecisionDNA:
        return DecisionDNA(
            id=self.id,
            title=self.title,
            full_narrative=self.full_narrative,
            premises=self.premises if isinstance(self.premises, list) else json.loads(self.premises),
            predicted_outcomes=self.predicted_outcomes if isinstance(self.predicted_outcomes, list) else json.loads(self.predicted_outcomes),
            triggering_conditions=self.triggering_conditions if isinstance(self.triggering_conditions, list) else json.loads(self.triggering_conditions),
            decision_date=self.decision_date,
            review_interval_days=self.review_interval_days,
            status=DecisionStatus(self.status),
            domain=self.domain,
            tags=self.tags if isinstance(self.tags, list) else (json.loads(self.tags) if self.tags else []),
            embedding=self.embedding if isinstance(self.embedding, list) else (json.loads(self.embedding) if self.embedding else None),
            created_at=self.created_at,
            updated_at=self.updated_at,
            web_monitoring_enabled=self.web_monitoring_enabled,
        )
