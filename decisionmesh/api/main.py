"""
DecisionMesh FastAPI application factory.
"""
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from decisionmesh.api.routes import decisions, agents, reports, health
from decisionmesh.database import init_db
from decisionmesh.agents.orchestrator import DecisionMeshOrchestrator


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Startup and shutdown lifecycle."""
    # Initialize database
    await init_db()

    # Start orchestrator (registers scheduler jobs)
    orchestrator = DecisionMeshOrchestrator.get_instance()
    await orchestrator.startup()

    yield

    # Shutdown
    await orchestrator.shutdown()


def create_app() -> FastAPI:
    app = FastAPI(
        title="DecisionMesh API",
        description="Temporal Decision Intelligence Engine",
        version="0.1.0",
        lifespan=lifespan,
    )

    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_methods=["*"],
        allow_headers=["*"],
    )

    app.include_router(health.router)
    app.include_router(decisions.router)
    app.include_router(agents.router)
    app.include_router(reports.router)

    return app


app = create_app()
