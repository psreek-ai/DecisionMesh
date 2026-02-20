"""API schemas for decisions."""
from datetime import datetime
from typing import Optional
from pydantic import BaseModel, Field


class CaptureDecisionRequest(BaseModel):
    narrative: str = Field(description="Free-form decision narrative from the user.")
    web_monitoring_enabled: bool = Field(default=False, description="Enable web monitoring for this decision.")


class CaptureDecisionResponse(BaseModel):
    decision_id: str
    title: str
    premises_count: int
    status: str
    web_monitoring_enabled: bool
    message: str


class DecisionListItem(BaseModel):
    id: str
    title: str
    status: str
    domain: str
    decision_date: datetime
    review_interval_days: int
    premises_count: int
    web_monitoring_enabled: bool
    created_at: datetime


class DecisionDetailResponse(BaseModel):
    id: str
    title: str
    full_narrative: str
    premises: list[str]
    predicted_outcomes: list[str]
    triggering_conditions: list[str]
    decision_date: datetime
    review_interval_days: int
    status: str
    domain: str
    tags: list[str]
    web_monitoring_enabled: bool
    created_at: datetime
    updated_at: datetime


class UpdateDecisionRequest(BaseModel):
    field: str = Field(description="Field to update: title|status|domain|tags|premises|predicted_outcomes|triggering_conditions")
    value: str = Field(description="New value (JSON string for list fields).")


class PauseDecisionRequest(BaseModel):
    pause: bool = Field(description="True to pause, False to resume.")
