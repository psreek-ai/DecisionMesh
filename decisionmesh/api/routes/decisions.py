"""Decision CRUD and lifecycle endpoints."""
import json
from typing import Optional
from fastapi import APIRouter, HTTPException, Query

from decisionmesh.agents.orchestrator import DecisionMeshOrchestrator
from decisionmesh.database import get_session
from decisionmesh.api.schemas.decision_schemas import (
    CaptureDecisionRequest, CaptureDecisionResponse,
    DecisionListItem, DecisionDetailResponse,
    UpdateDecisionRequest, PauseDecisionRequest,
)

router = APIRouter(prefix="/decisions", tags=["decisions"])


@router.post("/", response_model=CaptureDecisionResponse, status_code=201)
async def capture_decision(request: CaptureDecisionRequest):
    """Capture a new decision from a free-form narrative."""
    orchestrator = DecisionMeshOrchestrator.get_instance()
    try:
        decision_id = await orchestrator.capture_decision(
            narrative=request.narrative,
            web_monitoring=request.web_monitoring_enabled,
        )
    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e))

    # Fetch the created decision
    from sqlalchemy import select
    from decisionmesh.models.decision import DecisionORM
    from decisionmesh.models.premise import PremiseORM

    async with get_session() as session:
        result = await session.execute(select(DecisionORM).where(DecisionORM.id == decision_id))
        decision = result.scalar_one_or_none()
        premises_result = await session.execute(
            select(PremiseORM).where(PremiseORM.decision_id == decision_id)
        )
        premises = premises_result.scalars().all()

    if not decision:
        raise HTTPException(status_code=404, detail="Decision not found after capture")

    return CaptureDecisionResponse(
        decision_id=decision.id,
        title=decision.title,
        premises_count=len(premises),
        status=decision.status,
        web_monitoring_enabled=decision.web_monitoring_enabled,
        message="Decision captured successfully. Monitoring schedule registered.",
    )


@router.get("/", response_model=list[DecisionListItem])
async def list_decisions(
    status: Optional[str] = Query(None, description="Filter by status"),
    domain: Optional[str] = Query(None),
    limit: int = Query(50, ge=1, le=200),
):
    """List all decisions with status indicators."""
    from sqlalchemy import select
    from decisionmesh.models.decision import DecisionORM
    from decisionmesh.models.premise import PremiseORM

    async with get_session() as session:
        query = select(DecisionORM)
        if status:
            query = query.where(DecisionORM.status == status)
        if domain:
            query = query.where(DecisionORM.domain == domain)
        query = query.limit(limit).order_by(DecisionORM.created_at.desc())

        result = await session.execute(query)
        decisions = result.scalars().all()

        items = []
        for d in decisions:
            p_result = await session.execute(
                select(PremiseORM).where(PremiseORM.decision_id == d.id)
            )
            premise_count = len(p_result.scalars().all())
            items.append(DecisionListItem(
                id=d.id,
                title=d.title,
                status=d.status,
                domain=d.domain,
                decision_date=d.decision_date,
                review_interval_days=d.review_interval_days,
                premises_count=premise_count,
                web_monitoring_enabled=d.web_monitoring_enabled,
                created_at=d.created_at,
            ))

    return items


@router.get("/{decision_id}", response_model=DecisionDetailResponse)
async def get_decision(decision_id: str):
    """Get full Decision DNA for a specific decision."""
    from sqlalchemy import select
    from decisionmesh.models.decision import DecisionORM

    async with get_session() as session:
        result = await session.execute(select(DecisionORM).where(DecisionORM.id == decision_id))
        decision = result.scalar_one_or_none()

    if not decision:
        raise HTTPException(status_code=404, detail="Decision not found")

    d = decision.to_pydantic()
    return DecisionDetailResponse(**d.model_dump(mode="json"))


@router.patch("/{decision_id}")
async def update_decision(decision_id: str, request: UpdateDecisionRequest):
    """Update a field in a decision's DNA."""
    from sqlalchemy import select
    from decisionmesh.models.decision import DecisionORM
    from datetime import datetime

    async with get_session() as session:
        result = await session.execute(select(DecisionORM).where(DecisionORM.id == decision_id))
        decision = result.scalar_one_or_none()
        if not decision:
            raise HTTPException(status_code=404, detail="Decision not found")

        if request.field in ("premises", "predicted_outcomes", "triggering_conditions", "tags"):
            try:
                parsed = json.loads(request.value)
                setattr(decision, request.field, parsed)
            except json.JSONDecodeError:
                raise HTTPException(status_code=422, detail="Invalid JSON for list field")
        elif request.field in ("status", "domain", "title"):
            setattr(decision, request.field, request.value)
        else:
            raise HTTPException(status_code=422, detail=f"Field '{request.field}' is not updatable")

        decision.updated_at = datetime.utcnow()

    return {"status": "updated", "field": request.field, "decision_id": decision_id}


@router.post("/{decision_id}/pause")
async def pause_decision(decision_id: str, request: PauseDecisionRequest):
    """Pause or resume monitoring for a decision."""
    orchestrator = DecisionMeshOrchestrator.get_instance()
    if request.pause:
        await orchestrator.pause_decision(decision_id)
        return {"status": "paused", "decision_id": decision_id}
    else:
        await orchestrator.resume_decision(decision_id)
        return {"status": "resumed", "decision_id": decision_id}


@router.post("/{decision_id}/close")
async def close_decision(decision_id: str, reason: str = "User closed decision"):
    """Close a decision — end of lifecycle."""
    from sqlalchemy import select
    from decisionmesh.models.decision import DecisionORM, DecisionStatus
    from datetime import datetime

    async with get_session() as session:
        result = await session.execute(select(DecisionORM).where(DecisionORM.id == decision_id))
        decision = result.scalar_one_or_none()
        if not decision:
            raise HTTPException(status_code=404, detail="Decision not found")
        decision.status = DecisionStatus.CLOSED.value
        decision.updated_at = datetime.utcnow()

    orchestrator = DecisionMeshOrchestrator.get_instance()
    if orchestrator._scheduler:
        await orchestrator._scheduler.remove_decision(decision_id)

    return {"status": "closed", "decision_id": decision_id}


@router.get("/{decision_id}/export")
async def export_decision(decision_id: str):
    """Export a decision as JSON."""
    from sqlalchemy import select
    from decisionmesh.models.decision import DecisionORM
    from decisionmesh.models.premise import PremiseORM

    async with get_session() as session:
        result = await session.execute(select(DecisionORM).where(DecisionORM.id == decision_id))
        decision = result.scalar_one_or_none()
        if not decision:
            raise HTTPException(status_code=404, detail="Decision not found")

        p_result = await session.execute(
            select(PremiseORM).where(PremiseORM.decision_id == decision_id)
        )
        premises = [p.to_pydantic().model_dump(mode="json") for p in p_result.scalars().all()]

    d = decision.to_pydantic().model_dump(mode="json")
    d.pop("embedding", None)  # Don't export embeddings
    return {"decision": d, "premises": premises}
