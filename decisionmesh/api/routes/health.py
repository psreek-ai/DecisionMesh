"""Health check and scheduler status endpoints."""
from fastapi import APIRouter
from decisionmesh.agents.orchestrator import DecisionMeshOrchestrator

router = APIRouter(prefix="/health", tags=["health"])


@router.get("/")
async def health_check():
    return {"status": "ok", "service": "decisionmesh"}


@router.get("/scheduler")
async def scheduler_status():
    orchestrator = DecisionMeshOrchestrator.get_instance()
    if orchestrator._scheduler:
        return orchestrator._scheduler.get_scheduler_status()
    return {"running": False, "job_count": 0, "jobs": []}
