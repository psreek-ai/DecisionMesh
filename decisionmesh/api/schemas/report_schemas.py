"""API schemas for reports (divergence, counterfactuals)."""
from datetime import datetime
from typing import Optional
from pydantic import BaseModel


class DivergenceEventResponse(BaseModel):
    id: str
    decision_id: str
    decision_title: str
    detected_at: datetime
    severity: str
    divergence_score: float
    summary: str
    evidence: list[str]
    affected_premise_ids: list[str]
    acknowledged_by_user: bool
    acknowledged_at: Optional[datetime] = None
    counterfactual_id: Optional[str] = None


class CounterfactualResponse(BaseModel):
    id: str
    decision_id: str
    divergence_event_id: str
    generated_at: datetime
    alternative_scenario: str
    projected_outcome_delta: str
    confidence: float
    causal_chain: list[str]


class AgentLogResponse(BaseModel):
    id: str
    agent_name: str
    iteration: int
    tool_name: Optional[str] = None
    tool_input: Optional[dict] = None
    tool_output: Optional[str] = None
    thinking_content: Optional[str] = None
    text_content: Optional[str] = None
    stop_reason: Optional[str] = None
    input_tokens: int
    output_tokens: int
    logged_at: datetime


class InboxResponse(BaseModel):
    total: int
    items: list[DivergenceEventResponse]
