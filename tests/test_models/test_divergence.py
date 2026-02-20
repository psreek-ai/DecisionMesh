"""
Tests for decisionmesh.models.divergence and db_compute_divergence.

Divergence score formula:
    score = Σ(weight_i × invalidation_score_i) / Σ(weight_i)
    where:
        valid        → 0.0
        uncertain    → 0.4
        invalidated  → 1.0
        unverifiable → 0.2
"""
import uuid
from datetime import datetime

import pytest
from pydantic import ValidationError
from sqlalchemy import select

from decisionmesh.models.decision import DecisionORM
from decisionmesh.models.divergence import DivergenceEvent, DivergenceEventORM, DivergenceSeverity
from decisionmesh.models.premise import PremiseORM, PremiseStatus
from decisionmesh.tools.db_tools import db_compute_divergence


# ── Helpers ───────────────────────────────────────────────────────────────────

async def _insert_decision(session) -> str:
    decision_id = str(uuid.uuid4())
    decision = DecisionORM(
        id=decision_id,
        title="Divergence test decision",
        full_narrative="Test narrative.",
        premises=[],
        predicted_outcomes=[],
        triggering_conditions=[],
        decision_date=datetime.utcnow(),
    )
    session.add(decision)
    await session.flush()
    return decision_id


async def _insert_premise(
    session,
    decision_id: str,
    status: str,
    weight: float = 1.0,
) -> str:
    premise_id = str(uuid.uuid4())
    session.add(
        PremiseORM(
            id=premise_id,
            decision_id=decision_id,
            text=f"Premise with status={status} weight={weight}",
            status=status,
            weight=weight,
        )
    )
    await session.flush()
    return premise_id


# ── DivergenceSeverity enum ───────────────────────────────────────────────────

class TestDivergenceSeverityEnum:
    def test_all_expected_values_exist(self):
        values = {s.value for s in DivergenceSeverity}
        assert values == {"low", "medium", "high", "critical"}

    def test_is_string_enum(self):
        assert DivergenceSeverity.HIGH == "high"
        assert DivergenceSeverity.CRITICAL == "critical"

    def test_invalid_severity_raises(self):
        with pytest.raises(ValueError):
            DivergenceSeverity("extreme")


# ── DivergenceEvent pydantic model ────────────────────────────────────────────

class TestDivergenceEventPydantic:
    def _valid_payload(self, **overrides) -> dict:
        base = {
            "decision_id": str(uuid.uuid4()),
            "severity": DivergenceSeverity.MEDIUM,
            "affected_premise_ids": [str(uuid.uuid4())],
            "summary": "Revenue growth has slowed to 8% MoM vs 15% expected.",
            "evidence": ["Q3 earnings show 8.2% MoM growth", "CFO confirmed slower pipeline"],
            "divergence_score": 0.45,
        }
        base.update(overrides)
        return base

    def test_valid_construction(self):
        event = DivergenceEvent(**self._valid_payload())
        assert event.divergence_score == pytest.approx(0.45)
        assert event.severity == DivergenceSeverity.MEDIUM

    def test_id_auto_generated(self):
        event = DivergenceEvent(**self._valid_payload())
        parsed = uuid.UUID(event.id)
        assert str(parsed) == event.id

    def test_detected_at_auto_set(self):
        event = DivergenceEvent(**self._valid_payload())
        assert isinstance(event.detected_at, datetime)

    def test_default_not_acknowledged(self):
        event = DivergenceEvent(**self._valid_payload())
        assert event.acknowledged_by_user is False
        assert event.acknowledged_at is None

    def test_missing_decision_id_raises(self):
        payload = self._valid_payload()
        del payload["decision_id"]
        with pytest.raises(ValidationError):
            DivergenceEvent(**payload)

    def test_missing_severity_raises(self):
        payload = self._valid_payload()
        del payload["severity"]
        with pytest.raises(ValidationError):
            DivergenceEvent(**payload)

    def test_score_boundaries(self):
        """Score of 0.0 and 1.0 are both valid."""
        low = DivergenceEvent(**self._valid_payload(divergence_score=0.0))
        high = DivergenceEvent(**self._valid_payload(divergence_score=1.0))
        assert low.divergence_score == 0.0
        assert high.divergence_score == 1.0

    def test_from_attributes_config_enabled(self):
        assert DivergenceEvent.model_config.get("from_attributes") is True


# ── Divergence score formula edge cases ──────────────────────────────────────

class TestDivergenceScoreFormula:
    """
    Tests for db_compute_divergence, which implements:
        score = Σ(weight_i × invalidation_score_i) / Σ(weight_i)
    These directly test the scoring logic using the live in-memory DB.
    """

    async def test_no_premises_returns_zero(self, async_session):
        """A decision with no premises has zero divergence."""
        decision_id = await _insert_decision(async_session)
        result = await db_compute_divergence(decision_id, async_session)
        assert result["divergence_score"] == pytest.approx(0.0)
        assert result["details"] == []

    async def test_all_valid_premises_score_is_zero(self, async_session):
        """
        All premises valid → all invalidation_scores = 0.0
        score = (1.0×0.0 + 2.0×0.0 + 1.0×0.0) / 4.0 = 0.0
        """
        decision_id = await _insert_decision(async_session)
        await _insert_premise(async_session, decision_id, PremiseStatus.VALID.value, weight=1.0)
        await _insert_premise(async_session, decision_id, PremiseStatus.VALID.value, weight=2.0)
        await _insert_premise(async_session, decision_id, PremiseStatus.VALID.value, weight=1.0)

        result = await db_compute_divergence(decision_id, async_session)
        assert result["divergence_score"] == pytest.approx(0.0)

    async def test_all_invalidated_premises_score_is_one(self, async_session):
        """
        All premises invalidated → all invalidation_scores = 1.0
        score = (1.0×1.0 + 1.0×1.0) / 2.0 = 1.0
        """
        decision_id = await _insert_decision(async_session)
        await _insert_premise(async_session, decision_id, PremiseStatus.INVALIDATED.value, weight=1.0)
        await _insert_premise(async_session, decision_id, PremiseStatus.INVALIDATED.value, weight=1.0)

        result = await db_compute_divergence(decision_id, async_session)
        assert result["divergence_score"] == pytest.approx(1.0)

    async def test_mixed_uncertain_valid_equal_weights(self, async_session):
        """
        1 uncertain (weight=1) + 1 valid (weight=1):
        score = (1×0.4 + 1×0.0) / 2 = 0.2
        """
        decision_id = await _insert_decision(async_session)
        await _insert_premise(async_session, decision_id, PremiseStatus.UNCERTAIN.value, weight=1.0)
        await _insert_premise(async_session, decision_id, PremiseStatus.VALID.value, weight=1.0)

        result = await db_compute_divergence(decision_id, async_session)
        assert result["divergence_score"] == pytest.approx(0.2)

    async def test_weighted_invalidated_vs_valid(self, async_session):
        """
        1 invalidated (weight=3) + 1 valid (weight=1):
        score = (3×1.0 + 1×0.0) / (3+1) = 3.0 / 4.0 = 0.75
        """
        decision_id = await _insert_decision(async_session)
        await _insert_premise(async_session, decision_id, PremiseStatus.INVALIDATED.value, weight=3.0)
        await _insert_premise(async_session, decision_id, PremiseStatus.VALID.value, weight=1.0)

        result = await db_compute_divergence(decision_id, async_session)
        assert result["divergence_score"] == pytest.approx(0.75)

    async def test_single_uncertain_premise(self, async_session):
        """
        1 uncertain (weight=1):
        score = (1×0.4) / 1 = 0.4
        """
        decision_id = await _insert_decision(async_session)
        await _insert_premise(async_session, decision_id, PremiseStatus.UNCERTAIN.value, weight=1.0)

        result = await db_compute_divergence(decision_id, async_session)
        assert result["divergence_score"] == pytest.approx(0.4)

    async def test_unverifiable_premise_score(self, async_session):
        """
        1 unverifiable (weight=1):
        score = (1×0.2) / 1 = 0.2
        """
        decision_id = await _insert_decision(async_session)
        await _insert_premise(async_session, decision_id, PremiseStatus.UNVERIFIABLE.value, weight=1.0)

        result = await db_compute_divergence(decision_id, async_session)
        assert result["divergence_score"] == pytest.approx(0.2)

    async def test_complex_weighted_mix(self, async_session):
        """
        Premises:
          - valid       weight=2  → 2×0.0 = 0.0
          - uncertain   weight=1  → 1×0.4 = 0.4
          - invalidated weight=1  → 1×1.0 = 1.0
        Total weight = 4, weighted sum = 1.4
        score = 1.4 / 4.0 = 0.35
        """
        decision_id = await _insert_decision(async_session)
        await _insert_premise(async_session, decision_id, PremiseStatus.VALID.value, weight=2.0)
        await _insert_premise(async_session, decision_id, PremiseStatus.UNCERTAIN.value, weight=1.0)
        await _insert_premise(async_session, decision_id, PremiseStatus.INVALIDATED.value, weight=1.0)

        result = await db_compute_divergence(decision_id, async_session)
        assert result["divergence_score"] == pytest.approx(0.35)

    async def test_details_contains_all_premises(self, async_session):
        """Result details must include one entry per premise."""
        decision_id = await _insert_decision(async_session)
        await _insert_premise(async_session, decision_id, PremiseStatus.VALID.value, weight=1.0)
        await _insert_premise(async_session, decision_id, PremiseStatus.UNCERTAIN.value, weight=2.0)

        result = await db_compute_divergence(decision_id, async_session)
        assert len(result["details"]) == 2

    async def test_details_contain_expected_keys(self, async_session):
        decision_id = await _insert_decision(async_session)
        await _insert_premise(async_session, decision_id, PremiseStatus.VALID.value)

        result = await db_compute_divergence(decision_id, async_session)
        detail = result["details"][0]
        assert "premise_id" in detail
        assert "status" in detail
        assert "weight" in detail
        assert "invalidation_score" in detail

    async def test_unknown_decision_returns_zero_score(self, async_session):
        """A nonexistent decision_id should return an empty result."""
        result = await db_compute_divergence(str(uuid.uuid4()), async_session)
        assert result["divergence_score"] == pytest.approx(0.0)
        assert result["details"] == []


# ── DivergenceEventORM round-trip ─────────────────────────────────────────────

class TestDivergenceEventORMRoundTrip:
    async def test_insert_and_retrieve(self, async_session):
        decision_id = await _insert_decision(async_session)
        event_id = str(uuid.uuid4())
        premise_id = str(uuid.uuid4())

        orm = DivergenceEventORM(
            id=event_id,
            decision_id=decision_id,
            severity=DivergenceSeverity.HIGH.value,
            affected_premise_ids=[premise_id],
            summary="Revenue growth dropped below threshold.",
            evidence=["Q3 earnings show 6% MoM"],
            divergence_score=0.72,
        )
        async_session.add(orm)
        await async_session.flush()

        result = await async_session.execute(
            select(DivergenceEventORM).where(DivergenceEventORM.id == event_id)
        )
        fetched = result.scalar_one()
        pydantic_model = fetched.to_pydantic()

        assert pydantic_model.id == event_id
        assert pydantic_model.decision_id == decision_id
        assert pydantic_model.severity == DivergenceSeverity.HIGH
        assert pydantic_model.divergence_score == pytest.approx(0.72)
        assert pydantic_model.affected_premise_ids == [premise_id]
        assert pydantic_model.evidence == ["Q3 earnings show 6% MoM"]
        assert pydantic_model.acknowledged_by_user is False
