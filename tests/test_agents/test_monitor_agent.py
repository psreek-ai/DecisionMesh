"""
Tests for decisionmesh.agents.monitor_agent.MonitorAgent

Tests:
- db_compute_divergence produces correct scores (integration with real DB)
- MonitorAgent._update_premise_status correctly updates a premise in the DB
"""
import uuid
from datetime import datetime

import pytest
from sqlalchemy import select

from decisionmesh.agents.monitor_agent import MonitorAgent
from decisionmesh.models.decision import DecisionORM, DecisionStatus
from decisionmesh.models.premise import PremiseORM, PremiseStatus
from decisionmesh.tools.db_tools import db_compute_divergence


# ── Fixtures ──────────────────────────────────────────────────────────────────

@pytest.fixture
def monitor_agent(async_session, mock_anthropic_client):
    from decisionmesh.config import Settings
    config = Settings(
        ANTHROPIC_API_KEY="test-key",
        MAX_AGENT_ITERATIONS=5,
        DIVERGENCE_ALERT_THRESHOLD=0.3,
        DISABLE_ALL_WEB_MONITORING=True,
    )
    return MonitorAgent(db_session=async_session, config=config)


# ── DB helpers ────────────────────────────────────────────────────────────────

async def _create_decision_with_premises(
    session,
    premise_specs: list[dict],
) -> tuple[str, list[str]]:
    """
    Insert a DecisionORM and one PremiseORM per spec dict.
    spec keys: status (str), weight (float, optional)
    Returns (decision_id, [premise_id, ...]).
    """
    decision_id = str(uuid.uuid4())
    decision = DecisionORM(
        id=decision_id,
        title="Monitor agent test decision",
        full_narrative="Test narrative for monitor agent.",
        premises=[s.get("text", "A premise") for s in premise_specs],
        predicted_outcomes=["Good outcome"],
        triggering_conditions=["Some condition"],
        decision_date=datetime.utcnow(),
        status=DecisionStatus.ACTIVE.value,
    )
    session.add(decision)
    await session.flush()

    premise_ids = []
    for spec in premise_specs:
        pid = str(uuid.uuid4())
        session.add(
            PremiseORM(
                id=pid,
                decision_id=decision_id,
                text=spec.get("text", f"Premise {pid[:8]}"),
                status=spec["status"],
                weight=spec.get("weight", 1.0),
            )
        )
        premise_ids.append(pid)

    await session.flush()
    return decision_id, premise_ids


# ── Divergence score computation via db_compute_divergence ────────────────────

class TestMonitorDivergenceScoreComputation:
    """
    These tests verify the divergence formula through the real DB layer.
    Formula: score = Σ(weight_i × invalidation_score_i) / Σ(weight_i)
    Scores: valid=0.0, uncertain=0.4, invalidated=1.0, unverifiable=0.2
    """

    async def test_all_valid_premises_zero_divergence(self, async_session):
        decision_id, _ = await _create_decision_with_premises(
            async_session,
            [
                {"status": PremiseStatus.VALID.value, "weight": 1.0},
                {"status": PremiseStatus.VALID.value, "weight": 1.5},
            ],
        )
        result = await db_compute_divergence(decision_id, async_session)
        assert result["divergence_score"] == pytest.approx(0.0)

    async def test_single_invalidated_is_full_divergence(self, async_session):
        decision_id, _ = await _create_decision_with_premises(
            async_session,
            [{"status": PremiseStatus.INVALIDATED.value, "weight": 1.0}],
        )
        result = await db_compute_divergence(decision_id, async_session)
        assert result["divergence_score"] == pytest.approx(1.0)

    async def test_uncertain_premise_produces_partial_score(self, async_session):
        """1 uncertain + 1 valid with equal weights → 0.4/2 = 0.2"""
        decision_id, _ = await _create_decision_with_premises(
            async_session,
            [
                {"status": PremiseStatus.UNCERTAIN.value, "weight": 1.0},
                {"status": PremiseStatus.VALID.value, "weight": 1.0},
            ],
        )
        result = await db_compute_divergence(decision_id, async_session)
        assert result["divergence_score"] == pytest.approx(0.2)

    async def test_high_weight_invalidated_dominates(self, async_session):
        """
        invalidated (weight=3) + valid (weight=1):
        score = (3×1.0 + 1×0.0) / 4 = 0.75
        """
        decision_id, _ = await _create_decision_with_premises(
            async_session,
            [
                {"status": PremiseStatus.INVALIDATED.value, "weight": 3.0},
                {"status": PremiseStatus.VALID.value, "weight": 1.0},
            ],
        )
        result = await db_compute_divergence(decision_id, async_session)
        assert result["divergence_score"] == pytest.approx(0.75)

    async def test_score_updates_after_status_change(self, async_session, monitor_agent):
        """
        Insert two valid premises, then update one to 'invalidated' via the
        MonitorAgent._update_premise_status method, and verify the score changes.
        """
        decision_id, premise_ids = await _create_decision_with_premises(
            async_session,
            [
                {"status": PremiseStatus.VALID.value, "weight": 1.0},
                {"status": PremiseStatus.VALID.value, "weight": 1.0},
            ],
        )

        # Verify initial score is 0
        initial = await db_compute_divergence(decision_id, async_session)
        assert initial["divergence_score"] == pytest.approx(0.0)

        # Update the first premise to invalidated
        await monitor_agent._update_premise_status({
            "premise_id": premise_ids[0],
            "status": "invalidated",
            "confidence_score": 0.9,
            "invalidation_evidence": "Evidence found in Q3 report.",
        })

        # Recompute — should now be (1×1.0 + 1×0.0) / 2 = 0.5
        updated = await db_compute_divergence(decision_id, async_session)
        assert updated["divergence_score"] == pytest.approx(0.5)


# ── MonitorAgent._update_premise_status ──────────────────────────────────────

class TestUpdatePremiseStatus:
    async def test_updates_status_to_uncertain(self, async_session, monitor_agent):
        decision_id, premise_ids = await _create_decision_with_premises(
            async_session,
            [{"status": PremiseStatus.VALID.value}],
        )
        pid = premise_ids[0]

        result = await monitor_agent._update_premise_status({
            "premise_id": pid,
            "status": "uncertain",
            "confidence_score": 0.6,
        })

        assert result["new_status"] == "uncertain"
        assert result["premise_id"] == pid

        fetched = await async_session.execute(
            select(PremiseORM).where(PremiseORM.id == pid)
        )
        orm = fetched.scalar_one()
        assert orm.status == PremiseStatus.UNCERTAIN.value

    async def test_updates_status_to_invalidated_and_records_evidence(
        self, async_session, monitor_agent
    ):
        decision_id, premise_ids = await _create_decision_with_premises(
            async_session,
            [{"status": PremiseStatus.VALID.value}],
        )
        pid = premise_ids[0]

        await monitor_agent._update_premise_status({
            "premise_id": pid,
            "status": "invalidated",
            "confidence_score": 0.95,
            "invalidation_evidence": "GDP fell 2% — contradicts growth premise.",
        })

        fetched = await async_session.execute(
            select(PremiseORM).where(PremiseORM.id == pid)
        )
        orm = fetched.scalar_one()
        assert orm.status == PremiseStatus.INVALIDATED.value
        assert orm.invalidation_evidence == "GDP fell 2% — contradicts growth premise."
        assert orm.invalidated_at is not None

    async def test_updates_confidence_score(self, async_session, monitor_agent):
        decision_id, premise_ids = await _create_decision_with_premises(
            async_session,
            [{"status": PremiseStatus.VALID.value}],
        )
        pid = premise_ids[0]

        await monitor_agent._update_premise_status({
            "premise_id": pid,
            "status": "uncertain",
            "confidence_score": 0.75,
        })

        fetched = await async_session.execute(
            select(PremiseORM).where(PremiseORM.id == pid)
        )
        orm = fetched.scalar_one()
        assert orm.confidence_score == pytest.approx(0.75)

    async def test_sets_last_checked_at(self, async_session, monitor_agent):
        """_update_premise_status must always stamp last_checked_at."""
        decision_id, premise_ids = await _create_decision_with_premises(
            async_session,
            [{"status": PremiseStatus.VALID.value}],
        )
        pid = premise_ids[0]

        await monitor_agent._update_premise_status({
            "premise_id": pid,
            "status": "valid",
            "confidence_score": 1.0,
        })

        fetched = await async_session.execute(
            select(PremiseORM).where(PremiseORM.id == pid)
        )
        orm = fetched.scalar_one()
        assert orm.last_checked_at is not None

    async def test_returns_error_for_nonexistent_premise(self, async_session, monitor_agent):
        """A missing premise ID must return an error dict, not raise."""
        result = await monitor_agent._update_premise_status({
            "premise_id": str(uuid.uuid4()),  # Unknown ID
            "status": "valid",
            "confidence_score": 1.0,
        })
        assert "error" in result

    async def test_valid_status_does_not_set_invalidated_at(self, async_session, monitor_agent):
        """
        When status is set to 'valid', invalidated_at should NOT be set
        (only 'invalidated' status triggers that timestamp).
        """
        decision_id, premise_ids = await _create_decision_with_premises(
            async_session,
            [{"status": PremiseStatus.UNCERTAIN.value}],
        )
        pid = premise_ids[0]

        await monitor_agent._update_premise_status({
            "premise_id": pid,
            "status": "valid",
            "confidence_score": 1.0,
        })

        fetched = await async_session.execute(
            select(PremiseORM).where(PremiseORM.id == pid)
        )
        orm = fetched.scalar_one()
        # The premise was never invalidated, so this should remain None
        assert orm.invalidated_at is None
