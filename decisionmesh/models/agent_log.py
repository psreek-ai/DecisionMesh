"""
DecisionMesh — Agent Log model.
Full audit trail of every agent iteration, tool call, and reasoning output.
"""
import uuid
from datetime import datetime
from typing import Optional

from pydantic import BaseModel, Field
from sqlalchemy import DateTime, ForeignKey, Integer, JSON, String, Text
from sqlalchemy.orm import Mapped, mapped_column, relationship

from decisionmesh.database import Base


# ── Pydantic model ────────────────────────────────────────────────────────────

class AgentLog(BaseModel):
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    decision_id: Optional[str] = None
    agent_name: str
    iteration: int
    tool_name: Optional[str] = None
    tool_input: Optional[dict] = None
    tool_output: Optional[str] = None
    thinking_content: Optional[str] = None  # Extended thinking output
    text_content: Optional[str] = None
    stop_reason: Optional[str] = None
    input_tokens: int = 0
    output_tokens: int = 0
    logged_at: datetime = Field(default_factory=datetime.utcnow)

    model_config = {"from_attributes": True}


# ── SQLAlchemy ORM model ──────────────────────────────────────────────────────

class AgentLogORM(Base):
    __tablename__ = "agent_logs"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    decision_id: Mapped[Optional[str]] = mapped_column(String(36), ForeignKey("decisions.id"), nullable=True)
    agent_name: Mapped[str] = mapped_column(String(100), nullable=False)
    iteration: Mapped[int] = mapped_column(Integer, nullable=False)
    tool_name: Mapped[Optional[str]] = mapped_column(String(100), nullable=True)
    tool_input: Mapped[Optional[str]] = mapped_column(JSON, nullable=True)
    tool_output: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    thinking_content: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    text_content: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    stop_reason: Mapped[Optional[str]] = mapped_column(String(50), nullable=True)
    input_tokens: Mapped[int] = mapped_column(Integer, default=0)
    output_tokens: Mapped[int] = mapped_column(Integer, default=0)
    logged_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)

    # Relationships
    decision: Mapped[Optional["DecisionORM"]] = relationship("DecisionORM", back_populates="agent_logs_rel")  # noqa: F821

    def to_pydantic(self) -> AgentLog:
        return AgentLog(
            id=self.id,
            decision_id=self.decision_id,
            agent_name=self.agent_name,
            iteration=self.iteration,
            tool_name=self.tool_name,
            tool_input=self.tool_input,
            tool_output=self.tool_output,
            thinking_content=self.thinking_content,
            text_content=self.text_content,
            stop_reason=self.stop_reason,
            input_tokens=self.input_tokens,
            output_tokens=self.output_tokens,
            logged_at=self.logged_at,
        )
