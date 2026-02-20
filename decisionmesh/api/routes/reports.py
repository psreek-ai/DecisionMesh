"""Reports: divergence events, counterfactuals, agent logs."""
from typing import Optional
from fastapi import APIRouter, HTTPException, Query
from decisionmesh.database import get_session
from decisionmesh.api.schemas.report_schemas import (
    DivergenceEventResponse, CounterfactualResponse,
    AgentLogResponse, InboxResponse,
)

router = APIRouter(prefix="/reports", tags=["reports"])


@router.get("/inbox", response_model=InboxResponse)
async def get_inbox():
    """Get all unacknowledged divergence events."""
    from decisionmesh.agents.orchestrator import DecisionMeshOrchestrator
    orchestrator = DecisionMeshOrchestrator.get_instance()
    items = await orchestrator.get_pending_inbox()

    formatted = [
        DivergenceEventResponse(
            id=item["divergence_event_id"],
            decision_id=item["decision_id"],
            decision_title=item["decision_title"],
            detected_at=item["detected_at"],
            severity=item["severity"],
            divergence_score=item["divergence_score"],
            summary=item["summary"],
            evidence=[],
            affected_premise_ids=[],
            acknowledged_by_user=False,
        )
        for item in items
    ]
    return InboxResponse(total=len(formatted), items=formatted)


@router.get("/divergence/{decision_id}", response_model=list[DivergenceEventResponse])
async def get_divergence_events(decision_id: str, limit: int = Query(10, ge=1, le=100)):
    """Get divergence events for a decision."""
    from sqlalchemy import select, desc
    from decisionmesh.models.divergence import DivergenceEventORM
    from decisionmesh.models.decision import DecisionORM
    from decisionmesh.models.counterfactual import CounterfactualORM

    async with get_session() as session:
        result = await session.execute(
            select(DivergenceEventORM)
            .where(DivergenceEventORM.decision_id == decision_id)
            .order_by(desc(DivergenceEventORM.detected_at))
            .limit(limit)
        )
        events = result.scalars().all()

        d_result = await session.execute(
            select(DecisionORM).where(DecisionORM.id == decision_id)
        )
        decision = d_result.scalar_one_or_none()
        title = decision.title if decision else "Unknown"

        items = []
        for e in events:
            cf_result = await session.execute(
                select(CounterfactualORM).where(CounterfactualORM.divergence_event_id == e.id)
            )
            cf = cf_result.scalars().first()
            pydantic_e = e.to_pydantic()
            items.append(DivergenceEventResponse(
                id=pydantic_e.id,
                decision_id=pydantic_e.decision_id,
                decision_title=title,
                detected_at=pydantic_e.detected_at,
                severity=pydantic_e.severity.value,
                divergence_score=pydantic_e.divergence_score,
                summary=pydantic_e.summary,
                evidence=pydantic_e.evidence,
                affected_premise_ids=pydantic_e.affected_premise_ids,
                acknowledged_by_user=pydantic_e.acknowledged_by_user,
                acknowledged_at=pydantic_e.acknowledged_at,
                counterfactual_id=cf.id if cf else None,
            ))

    return items


@router.get("/counterfactuals/{decision_id}", response_model=list[CounterfactualResponse])
async def get_counterfactuals(decision_id: str):
    """Get all counterfactuals generated for a decision."""
    from sqlalchemy import select
    from decisionmesh.models.counterfactual import CounterfactualORM

    async with get_session() as session:
        result = await session.execute(
            select(CounterfactualORM).where(CounterfactualORM.decision_id == decision_id)
        )
        cfs = result.scalars().all()

    return [
        CounterfactualResponse(**cf.to_pydantic().model_dump(mode="json"))
        for cf in cfs
    ]


@router.get("/history/{decision_id}", response_model=list[AgentLogResponse])
async def get_agent_history(decision_id: str, limit: int = Query(50, ge=1, le=500)):
    """Get the full agent reasoning audit trail for a decision."""
    from sqlalchemy import select, desc
    from decisionmesh.models.agent_log import AgentLogORM

    async with get_session() as session:
        result = await session.execute(
            select(AgentLogORM)
            .where(AgentLogORM.decision_id == decision_id)
            .order_by(desc(AgentLogORM.logged_at))
            .limit(limit)
        )
        logs = result.scalars().all()

    return [AgentLogResponse(**log.to_pydantic().model_dump(mode="json")) for log in logs]
