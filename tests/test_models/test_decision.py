"""
Tests for decisionmesh.models.decision
Covers: DecisionDNA pydantic validation, DecisionStatus enum,
        DecisionORM round-trip persistence, and field defaults.
"""
import uuid
from datetime import datetime

import pytest
from pydantic import ValidationError
from sqlalchemy import select

from decisionmesh.models.decision import DecisionDNA, DecisionORM, DecisionStatus


# ── DecisionStatus enum ───────────────────────────────────────────────────────

class TestDecisionStatusEnum:
    def test_all_expected_values_exist(self):
        values = {s.value for s in DecisionStatus}
        assert values == {"active", "drifted", "revised", "closed", "paused"}

    def test_is_string_enum(self):
        # DecisionStatus should be usable as a plain string (str, Enum)
        assert DecisionStatus.ACTIVE == "active"
        assert DecisionStatus.DRIFTED == "drifted"

    def test_comparison_with_raw_string(self):
        assert DecisionStatus("active") is DecisionStatus.ACTIVE
        assert DecisionStatus("closed") is DecisionStatus.CLOSED

    def test_invalid_value_raises(self):
        with pytest.raises(ValueError):
            DecisionStatus("nonexistent_status")


# ── DecisionDNA pydantic model ────────────────────────────────────────────────

class TestDecisionDNA:
    def _valid_payload(self, **overrides) -> dict:
        base = {
            "title": "Switch to remote-first work policy",
            "full_narrative": "We decided to go remote-first because office costs are too high.",
            "premises": ["Office rental costs exceed 20% of revenue", "Remote tools have matured"],
            "predicted_outcomes": ["30% reduction in overhead within 12 months"],
            "triggering_conditions": ["Team productivity drops > 15%"],
            "decision_date": datetime(2024, 3, 1),
        }
        base.update(overrides)
        return base

    def test_valid_construction(self):
        dna = DecisionDNA(**self._valid_payload())
        assert dna.title == "Switch to remote-first work policy"
        assert len(dna.premises) == 2

    def test_id_auto_generated_as_uuid(self):
        dna = DecisionDNA(**self._valid_payload())
        # Should parse without raising
        parsed = uuid.UUID(dna.id)
        assert str(parsed) == dna.id

    def test_two_instances_have_different_ids(self):
        dna1 = DecisionDNA(**self._valid_payload())
        dna2 = DecisionDNA(**self._valid_payload())
        assert dna1.id != dna2.id

    def test_default_status_is_active(self):
        dna = DecisionDNA(**self._valid_payload())
        assert dna.status == DecisionStatus.ACTIVE

    def test_default_review_interval_days(self):
        dna = DecisionDNA(**self._valid_payload())
        assert dna.review_interval_days == 30

    def test_default_domain_is_general(self):
        dna = DecisionDNA(**self._valid_payload())
        assert dna.domain == "general"

    def test_default_tags_empty_list(self):
        dna = DecisionDNA(**self._valid_payload())
        assert dna.tags == []

    def test_default_embedding_is_none(self):
        dna = DecisionDNA(**self._valid_payload())
        assert dna.embedding is None

    def test_default_web_monitoring_disabled(self):
        dna = DecisionDNA(**self._valid_payload())
        assert dna.web_monitoring_enabled is False

    def test_created_at_and_updated_at_auto_set(self):
        dna = DecisionDNA(**self._valid_payload())
        assert isinstance(dna.created_at, datetime)
        assert isinstance(dna.updated_at, datetime)

    def test_explicit_id_preserved(self):
        fixed_id = str(uuid.uuid4())
        dna = DecisionDNA(**self._valid_payload(), id=fixed_id)
        assert dna.id == fixed_id

    def test_explicit_status_accepted(self):
        dna = DecisionDNA(**self._valid_payload(), status=DecisionStatus.PAUSED)
        assert dna.status == DecisionStatus.PAUSED

    def test_embedding_field_accepts_float_list(self):
        vec = [0.1, 0.2, 0.3]
        dna = DecisionDNA(**self._valid_payload(), embedding=vec)
        assert dna.embedding == vec

    def test_missing_title_raises_validation_error(self):
        payload = self._valid_payload()
        del payload["title"]
        with pytest.raises(ValidationError) as exc_info:
            DecisionDNA(**payload)
        assert "title" in str(exc_info.value)

    def test_missing_decision_date_raises(self):
        payload = self._valid_payload()
        del payload["decision_date"]
        with pytest.raises(ValidationError):
            DecisionDNA(**payload)

    def test_premises_must_be_list(self):
        payload = self._valid_payload()
        payload["premises"] = "not a list"
        with pytest.raises(ValidationError):
            DecisionDNA(**payload)

    def test_model_config_from_attributes_true(self):
        # Verify ORM-mode is enabled — needed for to_pydantic()
        assert DecisionDNA.model_config.get("from_attributes") is True


# ── DecisionORM database round-trip ──────────────────────────────────────────

class TestDecisionORMRoundTrip:
    """
    These tests write an ORM row to the in-memory SQLite DB, then fetch it
    back and verify the data survives serialization without corruption.
    """

    async def test_insert_and_retrieve_basic_fields(self, async_session):
        decision_id = str(uuid.uuid4())
        orm = DecisionORM(
            id=decision_id,
            title="Invest in index funds",
            full_narrative="Long-term passive investing strategy.",
            premises=["Markets trend upward long-term"],
            predicted_outcomes=["Outperform active funds over 10 years"],
            triggering_conditions=["Personal financial emergency"],
            decision_date=datetime(2023, 6, 1),
        )
        async_session.add(orm)
        await async_session.flush()

        result = await async_session.execute(
            select(DecisionORM).where(DecisionORM.id == decision_id)
        )
        fetched = result.scalar_one()

        assert fetched.id == decision_id
        assert fetched.title == "Invest in index funds"
        assert fetched.full_narrative == "Long-term passive investing strategy."

    async def test_default_status_is_active_in_orm(self, async_session):
        orm = DecisionORM(
            id=str(uuid.uuid4()),
            title="Test decision",
            full_narrative="Test narrative.",
            premises=[],
            predicted_outcomes=[],
            triggering_conditions=[],
            decision_date=datetime.utcnow(),
        )
        async_session.add(orm)
        await async_session.flush()

        result = await async_session.execute(
            select(DecisionORM).where(DecisionORM.id == orm.id)
        )
        fetched = result.scalar_one()
        assert fetched.status == DecisionStatus.ACTIVE.value

    async def test_json_list_fields_survive_roundtrip(self, async_session):
        premises_list = ["Premise A", "Premise B", "Premise C"]
        outcomes_list = ["Outcome X", "Outcome Y"]
        conditions_list = ["Condition 1"]

        orm = DecisionORM(
            id=str(uuid.uuid4()),
            title="JSON round-trip test",
            full_narrative="Testing JSON serialization.",
            premises=premises_list,
            predicted_outcomes=outcomes_list,
            triggering_conditions=conditions_list,
            decision_date=datetime(2024, 1, 1),
        )
        async_session.add(orm)
        await async_session.flush()

        result = await async_session.execute(
            select(DecisionORM).where(DecisionORM.id == orm.id)
        )
        fetched = result.scalar_one()

        # After flush, SQLAlchemy should have materialized the JSON
        assert fetched.premises == premises_list
        assert fetched.predicted_outcomes == outcomes_list
        assert fetched.triggering_conditions == conditions_list

    async def test_to_pydantic_returns_correct_decision_dna(self, async_session):
        decision_id = str(uuid.uuid4())
        decision_date = datetime(2024, 5, 10)

        orm = DecisionORM(
            id=decision_id,
            title="Hire remote contractor",
            full_narrative="Needed extra capacity without headcount.",
            premises=["Contractor market has qualified candidates"],
            predicted_outcomes=["Project delivered on schedule"],
            triggering_conditions=["Contractor unavailable for > 1 week"],
            decision_date=decision_date,
            domain="business",
            tags=["hiring", "remote"],
            review_interval_days=14,
            web_monitoring_enabled=True,
        )
        async_session.add(orm)
        await async_session.flush()

        result = await async_session.execute(
            select(DecisionORM).where(DecisionORM.id == decision_id)
        )
        fetched = result.scalar_one()
        pydantic_model = fetched.to_pydantic()

        assert isinstance(pydantic_model, DecisionDNA)
        assert pydantic_model.id == decision_id
        assert pydantic_model.title == "Hire remote contractor"
        assert pydantic_model.domain == "business"
        assert pydantic_model.tags == ["hiring", "remote"]
        assert pydantic_model.review_interval_days == 14
        assert pydantic_model.web_monitoring_enabled is True
        assert pydantic_model.status == DecisionStatus.ACTIVE

    async def test_status_update_persists(self, async_session):
        orm = DecisionORM(
            id=str(uuid.uuid4()),
            title="Status mutation test",
            full_narrative="Will this status update persist?",
            premises=[],
            predicted_outcomes=[],
            triggering_conditions=[],
            decision_date=datetime.utcnow(),
        )
        async_session.add(orm)
        await async_session.flush()

        orm.status = DecisionStatus.DRIFTED.value
        await async_session.flush()

        result = await async_session.execute(
            select(DecisionORM).where(DecisionORM.id == orm.id)
        )
        fetched = result.scalar_one()
        assert fetched.status == DecisionStatus.DRIFTED.value

    async def test_tags_default_to_empty_list(self, async_session):
        orm = DecisionORM(
            id=str(uuid.uuid4()),
            title="No-tags decision",
            full_narrative="Decision with no tags.",
            premises=[],
            predicted_outcomes=[],
            triggering_conditions=[],
            decision_date=datetime.utcnow(),
        )
        async_session.add(orm)
        await async_session.flush()

        pydantic_model = orm.to_pydantic()
        assert pydantic_model.tags == []

    async def test_embedding_stored_and_retrieved(self, async_session):
        embedding = [0.1, 0.5, -0.3, 0.8]
        orm = DecisionORM(
            id=str(uuid.uuid4()),
            title="Embedded decision",
            full_narrative="Decision with embedding.",
            premises=[],
            predicted_outcomes=[],
            triggering_conditions=[],
            decision_date=datetime.utcnow(),
            embedding=embedding,
        )
        async_session.add(orm)
        await async_session.flush()

        result = await async_session.execute(
            select(DecisionORM).where(DecisionORM.id == orm.id)
        )
        fetched = result.scalar_one()
        pydantic_model = fetched.to_pydantic()
        assert pydantic_model.embedding == embedding
