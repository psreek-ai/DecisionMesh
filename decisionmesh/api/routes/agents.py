"""Endpoints to manually trigger agent runs."""
from fastapi import APIRouter, HTTPException
from decisionmesh.agents.orchestrator import DecisionMeshOrchestrator

router = APIRouter(prefix="/agents", tags=["agents"])


@router.post("/monitor/{decision_id}")
async def trigger_monitor(decision_id: str):
    """Manually trigger a monitoring run for a specific decision."""
    orchestrator = DecisionMeshOrchestrator.get_instance()
    try:
        result = await orchestrator.run_monitor(decision_id)
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/revision/{divergence_event_id}")
async def trigger_revision(divergence_event_id: str, decision_id: str, user_choice: str = None):
    """Trigger the Revision Advisor for a divergence event."""
    orchestrator = DecisionMeshOrchestrator.get_instance()
    try:
        response = await orchestrator.present_revision(
            divergence_event_id=divergence_event_id,
            decision_id=decision_id,
            user_choice=user_choice,
        )
        return {"response": response}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
