"""
Tests for the DecisionMesh HTTP API.

Tests:
- GET /health/               → 200
- GET /decisions/            → 200, empty list initially
- POST /decisions/           → 201 (orchestrator mocked)
- GET /decisions/{id}        → 404 for unknown id

The routes use `get_session()` as a direct async context manager (not FastAPI
Depends), so we patch `decisionmesh.database.get_session` to redirect all DB
calls to an in-memory per-test SQLite engine.
"""
import uuid
from contextlib import asynccontextmanager
from datetime import datetime
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from httpx import AsyncClient, ASGITransport
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

from decisionmesh.database import Base
from decisionmesh.models import (  # noqa: F401  — populate Base.metadata
    agent_log, condition, counterfactual, decision, divergence, observation, premise
)


# ── Per-test DB engine + session factory ──────────────────────────────────────

@pytest.fixture
async def test_engine():
    engine = create_async_engine("sqlite+aiosqlite:///:memory:", echo=False)
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    yield engine
    await engine.dispose()


@pytest.fixture
def session_factory(test_engine):
    return async_sessionmaker(test_engine, class_=AsyncSession, expire_on_commit=False)


# ── Patch get_session to use the test engine ──────────────────────────────────

@pytest.fixture
def patched_get_session(session_factory):
    """
    Replace decisionmesh.database.get_session (and its import in each route
    module) with a version that uses the in-memory test database.
    """
    @asynccontextmanager
    async def _test_get_session():
        async with session_factory() as session:
            try:
                yield session
                await session.commit()
            except Exception:
                await session.rollback()
                raise

    targets = [
        "decisionmesh.database.get_session",
        "decisionmesh.api.routes.decisions.get_session",
        "decisionmesh.api.routes.health.get_session",
    ]
    patchers = [patch(t, new=_test_get_session) for t in targets if True]
    # Patch only the ones that exist
    active_patches = []
    for target in [
        "decisionmesh.api.routes.decisions.get_session",
    ]:
        try:
            p = patch(target, new=_test_get_session)
            p.start()
            active_patches.append(p)
        except AttributeError:
            pass

    yield _test_get_session

    for p in active_patches:
        p.stop()


# ── App fixture that avoids the full lifespan (no scheduler) ─────────────────

@pytest.fixture
def test_app():
    """
    Create a minimal FastAPI app with only the routes under test.
    The orchestrator singleton is reset between tests.
    """
    from fastapi import FastAPI
    from fastapi.middleware.cors import CORSMiddleware
    from decisionmesh.api.routes import decisions, health

    @asynccontextmanager
    async def noop_lifespan(app):
        yield  # No DB init, no scheduler — handled by fixture

    app = FastAPI(lifespan=noop_lifespan)
    app.add_middleware(
        CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"]
    )
    app.include_router(health.router)
    app.include_router(decisions.router)
    return app


@pytest.fixture
async def client(test_app, patched_get_session):
    """HTTP test client with the session patch active."""
    async with AsyncClient(
        transport=ASGITransport(app=test_app),
        base_url="http://testserver",
    ) as ac:
        yield ac


# ── GET /health/ ──────────────────────────────────────────────────────────────

class TestHealthEndpoint:
    async def test_health_returns_200(self, test_app):
        """Health check should return 200 even without DB or orchestrator."""
        # Override get_instance to avoid scheduler startup
        with patch(
            "decisionmesh.api.routes.health.DecisionMeshOrchestrator.get_instance"
        ) as mock_orch:
            mock_orch.return_value._scheduler = None

            async with AsyncClient(
                transport=ASGITransport(app=test_app),
                base_url="http://testserver",
            ) as ac:
                response = await ac.get("/health/")

        assert response.status_code == 200

    async def test_health_returns_ok_status(self, test_app):
        with patch(
            "decisionmesh.api.routes.health.DecisionMeshOrchestrator.get_instance"
        ) as mock_orch:
            mock_orch.return_value._scheduler = None

            async with AsyncClient(
                transport=ASGITransport(app=test_app),
                base_url="http://testserver",
            ) as ac:
                response = await ac.get("/health/")

        data = response.json()
        assert data["status"] == "ok"

    async def test_health_returns_service_name(self, test_app):
        with patch(
            "decisionmesh.api.routes.health.DecisionMeshOrchestrator.get_instance"
        ) as mock_orch:
            mock_orch.return_value._scheduler = None

            async with AsyncClient(
                transport=ASGITransport(app=test_app),
                base_url="http://testserver",
            ) as ac:
                response = await ac.get("/health/")

        data = response.json()
        assert data["service"] == "decisionmesh"


# ── GET /decisions/ ───────────────────────────────────────────────────────────

class TestListDecisionsEndpoint:
    async def test_empty_list_initially(self, client):
        """A fresh in-memory database must return an empty list."""
        response = await client.get("/decisions/")
        assert response.status_code == 200
        assert response.json() == []

    async def test_response_is_list(self, client):
        response = await client.get("/decisions/")
        assert isinstance(response.json(), list)

    async def test_list_decisions_after_insert(self, client, session_factory):
        """After directly inserting a decision, GET /decisions/ must list it."""
        from decisionmesh.models.decision import DecisionORM

        decision_id = str(uuid.uuid4())
        async with session_factory() as session:
            orm = DecisionORM(
                id=decision_id,
                title="API list test decision",
                full_narrative="Narrative.",
                premises=["Premise 1"],
                predicted_outcomes=["Outcome"],
                triggering_conditions=["Condition"],
                decision_date=datetime(2024, 3, 1),
                domain="business",
            )
            session.add(orm)
            await session.commit()

        response = await client.get("/decisions/")
        assert response.status_code == 200
        items = response.json()
        assert len(items) == 1
        assert items[0]["id"] == decision_id
        assert items[0]["title"] == "API list test decision"
        assert items[0]["domain"] == "business"

    async def test_filter_by_status(self, client, session_factory):
        """?status=active should only return active decisions."""
        from decisionmesh.models.decision import DecisionORM, DecisionStatus

        async with session_factory() as session:
            active = DecisionORM(
                id=str(uuid.uuid4()),
                title="Active decision",
                full_narrative="x",
                premises=[], predicted_outcomes=[], triggering_conditions=[],
                decision_date=datetime.utcnow(),
                status=DecisionStatus.ACTIVE.value,
            )
            closed = DecisionORM(
                id=str(uuid.uuid4()),
                title="Closed decision",
                full_narrative="x",
                premises=[], predicted_outcomes=[], triggering_conditions=[],
                decision_date=datetime.utcnow(),
                status=DecisionStatus.CLOSED.value,
            )
            session.add_all([active, closed])
            await session.commit()

        response = await client.get("/decisions/?status=active")
        assert response.status_code == 200
        items = response.json()
        assert len(items) >= 1
        assert all(item["status"] == "active" for item in items)
        titles = [item["title"] for item in items]
        assert "Active decision" in titles
        assert "Closed decision" not in titles


# ── POST /decisions/ ──────────────────────────────────────────────────────────

class TestCreateDecisionEndpoint:
    async def test_post_with_mocked_orchestrator_returns_201(
        self, test_app, patched_get_session, session_factory
    ):
        """
        POST /decisions/ must call the orchestrator and return 201.
        The orchestrator is mocked; we pre-insert the resulting DecisionORM
        so the route's post-capture DB fetch succeeds.
        """
        from decisionmesh.models.decision import DecisionORM

        new_decision_id = str(uuid.uuid4())

        # Pre-insert so the route's DB lookup after capture_decision() succeeds
        async with session_factory() as session:
            session.add(DecisionORM(
                id=new_decision_id,
                title="Mocked captured decision",
                full_narrative="Some narrative.",
                premises=["A premise"],
                predicted_outcomes=["An outcome"],
                triggering_conditions=["A condition"],
                decision_date=datetime(2024, 1, 15),
                domain="career",
            ))
            await session.commit()

        with patch(
            "decisionmesh.api.routes.decisions.DecisionMeshOrchestrator.get_instance"
        ) as mock_get_instance:
            mock_orchestrator = MagicMock()
            mock_orchestrator.capture_decision = AsyncMock(return_value=new_decision_id)
            mock_get_instance.return_value = mock_orchestrator

            async with AsyncClient(
                transport=ASGITransport(app=test_app),
                base_url="http://testserver",
            ) as ac:
                response = await ac.post(
                    "/decisions/",
                    json={
                        "narrative": "I decided to invest in index funds for long-term growth."
                    },
                )

        assert response.status_code == 201
        data = response.json()
        assert data["decision_id"] == new_decision_id
        assert data["title"] == "Mocked captured decision"

    async def test_post_without_narrative_returns_422(self, client):
        """Missing 'narrative' field should return 422 Unprocessable Entity."""
        response = await client.post("/decisions/", json={})
        assert response.status_code == 422

    async def test_post_calls_orchestrator_with_correct_narrative(self, test_app, patched_get_session):
        """The orchestrator must receive the exact narrative string."""
        narrative = "Testing that narrative string is forwarded verbatim."

        with patch(
            "decisionmesh.api.routes.decisions.DecisionMeshOrchestrator.get_instance"
        ) as mock_get_instance:
            mock_orchestrator = MagicMock()
            # Raise ValueError → route returns 422 (no need to set up DB)
            mock_orchestrator.capture_decision = AsyncMock(
                side_effect=ValueError("Clarification needed: date missing")
            )
            mock_get_instance.return_value = mock_orchestrator

            async with AsyncClient(
                transport=ASGITransport(app=test_app),
                base_url="http://testserver",
            ) as ac:
                await ac.post("/decisions/", json={"narrative": narrative})

        mock_orchestrator.capture_decision.assert_called_once_with(
            narrative=narrative,
            web_monitoring=False,
        )

    async def test_post_orchestrator_value_error_returns_422(self, test_app, patched_get_session):
        """A ValueError from the orchestrator should produce a 422 response."""
        with patch(
            "decisionmesh.api.routes.decisions.DecisionMeshOrchestrator.get_instance"
        ) as mock_get_instance:
            mock_orchestrator = MagicMock()
            mock_orchestrator.capture_decision = AsyncMock(
                side_effect=ValueError("Clarification needed: ambiguous date")
            )
            mock_get_instance.return_value = mock_orchestrator

            async with AsyncClient(
                transport=ASGITransport(app=test_app),
                base_url="http://testserver",
            ) as ac:
                response = await ac.post(
                    "/decisions/",
                    json={"narrative": "Some narrative."},
                )

        assert response.status_code == 422


# ── GET /decisions/{id} ───────────────────────────────────────────────────────

class TestGetDecisionEndpoint:
    async def test_unknown_id_returns_404(self, client):
        """A random UUID not in the DB should return 404."""
        fake_id = str(uuid.uuid4())
        response = await client.get(f"/decisions/{fake_id}")
        assert response.status_code == 404

    async def test_known_id_returns_200(self, client, session_factory):
        """After inserting a decision, GET by ID must return 200 with full DNA."""
        from decisionmesh.models.decision import DecisionORM

        decision_id = str(uuid.uuid4())
        async with session_factory() as session:
            orm = DecisionORM(
                id=decision_id,
                title="Get by ID test",
                full_narrative="Testing the get-by-ID endpoint.",
                premises=["Market is growing"],
                predicted_outcomes=["10% revenue gain"],
                triggering_conditions=["Market contracts"],
                decision_date=datetime(2024, 6, 15),
                domain="business",
                tags=["api-test"],
            )
            session.add(orm)
            await session.commit()

        response = await client.get(f"/decisions/{decision_id}")
        assert response.status_code == 200

        data = response.json()
        assert data["id"] == decision_id
        assert data["title"] == "Get by ID test"
        assert data["domain"] == "business"
        assert "Market is growing" in data["premises"]
        assert "10% revenue gain" in data["predicted_outcomes"]
        assert "api-test" in data["tags"]

    async def test_known_id_response_includes_required_fields(self, client, session_factory):
        """The detail response must include all expected DNA fields."""
        from decisionmesh.models.decision import DecisionORM

        decision_id = str(uuid.uuid4())
        async with session_factory() as session:
            session.add(DecisionORM(
                id=decision_id,
                title="Field check decision",
                full_narrative="Full narrative text.",
                premises=[],
                predicted_outcomes=[],
                triggering_conditions=[],
                decision_date=datetime.utcnow(),
            ))
            await session.commit()

        response = await client.get(f"/decisions/{decision_id}")
        data = response.json()

        required_fields = {
            "id", "title", "full_narrative", "premises",
            "predicted_outcomes", "triggering_conditions",
            "decision_date", "status", "domain", "tags",
            "review_interval_days", "web_monitoring_enabled",
            "created_at", "updated_at",
        }
        assert required_fields.issubset(set(data.keys()))

    async def test_404_error_body_has_detail_key(self, client):
        """FastAPI 404 responses include a 'detail' key in the response body."""
        response = await client.get(f"/decisions/{uuid.uuid4()}")
        assert "detail" in response.json()
