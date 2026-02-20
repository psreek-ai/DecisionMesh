"""
DecisionMesh — Monitoring Condition model.
User-defined conditions that trigger reviews.
"""
import uuid
from datetime import datetime
from enum import Enum
from typing import Optional

from pydantic import BaseModel, Field
from sqlalchemy import DateTime, ForeignKey, String, Text
from sqlalchemy.orm import Mapped, mapped_column, relationship

from decisionmesh.database import Base


class ConditionType(str, Enum):
    SCHEDULE = "schedule"         # Run on a fixed interval
    EVENT_BASED = "event_based"   # Run when a keyword/topic is detected
    MANUAL = "manual"             # User triggers manually


# ── Pydantic model ────────────────────────────────────────────────────────────

class Condition(BaseModel):
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    decision_id: str
    description: str
    condition_type: ConditionType = ConditionType.SCHEDULE
    search_keywords: list[str] = Field(default_factory=list)
    created_at: datetime = Field(default_factory=datetime.utcnow)
    last_triggered_at: Optional[datetime] = None

    model_config = {"from_attributes": True}


# ── SQLAlchemy ORM model ──────────────────────────────────────────────────────

class ConditionORM(Base):
    __tablename__ = "conditions"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    decision_id: Mapped[str] = mapped_column(String(36), ForeignKey("decisions.id"), nullable=False)
    description: Mapped[str] = mapped_column(Text, nullable=False)
    condition_type: Mapped[str] = mapped_column(String(30), default=ConditionType.SCHEDULE.value)
    search_keywords: Mapped[Optional[str]] = mapped_column(String(2000), nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)
    last_triggered_at: Mapped[Optional[datetime]] = mapped_column(DateTime, nullable=True)

    def to_pydantic(self) -> Condition:
        import json
        return Condition(
            id=self.id,
            decision_id=self.decision_id,
            description=self.description,
            condition_type=ConditionType(self.condition_type),
            search_keywords=json.loads(self.search_keywords) if self.search_keywords else [],
            created_at=self.created_at,
            last_triggered_at=self.last_triggered_at,
        )
