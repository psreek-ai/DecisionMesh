"""
Tests for decisionmesh.models.premise
Covers: Premise pydantic validation, PremiseStatus/PremiseTestability enums,
        and PremiseORM database round-trip.
"""
import uuid
from datetime import datetime

import pytest
from pydantic import ValidationError
from sqlalchemy import select

from decisionmesh.models.decision import DecisionORM, DecisionStatus
from decisionmesh.models.premise import (
    Premise,
    PremiseORM,
    PremiseStatus,
    PremiseTestability,
)


# ── Enum tests ────────────────────────────────────────────────────────────────

class TestPremiseStatusEnum:
    def test_all_expected_values_exist(self):
        values = {s.value for s in PremiseStatus}
        assert values == {"valid", "uncertain", "invalidated", "unverifiable"}

    def test_is_string_enum(self):
        assert PremiseStatus.VALID == "valid"
        assert PremiseStatus.INVALIDATED == "invalidated"

    def test_round_trip_from_string(self):
        assert PremiseStatus("uncertain") is PremiseStatus.UNCERTAIN
        assert PremiseStatus("unverifiable") is PremiseStatus.UNVERIFIABLE

    def test_invalid_status_raises(self):
        with pytest.raises(ValueError):
            PremiseStatus("deleted")


class TestPremiseTestabilityEnum:
    def test_all_expected_values_exist(self):
        values = {t.value for t in PremiseTestability}
        assert values == {
            "empirically_testable",
            "measurable",
            "subjective_untestable",
        }

    def test_is_string_enum(self):
        assert PremiseTestability.MEASURABLE == "measurable"

    def test_round_trip_from_string(self):
        assert (
            PremiseTestability("subjective_untestable")
            is PremiseTestability.SUBJECTIVE_UNTESTABLE
        )

    def test_invalid_testability_raises(self):
        with pytest.raises(ValueError):
            PremiseTestability("unknown")


# ── Premise pydantic model ────────────────────────────────────────────────────

class TestPremisePydantic:
    def _valid_payload(self, **overrides) -> dict:
        base = {
            "decision_id": str(uuid.uuid4()),
            "text": "Remote work tools have matured sufficiently for async collaboration.",
        }
        base.update(overrides)
        return base

    def test_valid_construction(self):
        p = Premise(**self._valid_payload())
        assert p.text == "Remote work tools have matured sufficiently for async collaboration."

    def test_id_auto_generated_as_uuid(self):
        p = Premise(**self._valid_payload())
        parsed = uuid.UUID(p.id)
        assert str(parsed) == p.id

    def test_two_instances_have_different_ids(self):
        p1 = Premise(**self._valid_payload())
        p2 = Premise(**self._valid_payload())
        assert p1.id != p2.id

    def test_default_confidence_score_is_one(self):
        p = Premise(**self._valid_payload())
        assert p.confidence_score == 1.0

    def test_default_status_is_valid(self):
        p = Premise(**self._valid_payload())
        assert p.status == PremiseStatus.VALID

    def test_default_testability_is_empirically_testable(self):
        p = Premise(**self._valid_payload())
        assert p.testability == PremiseTestability.EMPIRICALLY_TESTABLE

    def test_default_weight_is_one(self):
        p = Premise(**self._valid_payload())
        assert p.weight == 1.0

    def test_default_optional_fields_are_none(self):
        p = Premise(**self._valid_payload())
        assert p.last_checked_at is None
        assert p.invalidated_at is None
        assert p.invalidation_evidence is None
        assert p.embedding is None

    def test_default_web_monitoring_disabled(self):
        p = Premise(**self._valid_payload())
        assert p.requires_web_monitoring is False

    def test_explicit_status_accepted(self):
        p = Premise(**self._valid_payload(), status=PremiseStatus.UNCERTAIN)
        assert p.status == PremiseStatus.UNCERTAIN

    def test_explicit_testability_accepted(self):
        p = Premise(
            **self._valid_payload(),
            testability=PremiseTestability.SUBJECTIVE_UNTESTABLE,
        )
        assert p.testability == PremiseTestability.SUBJECTIVE_UNTESTABLE

    def test_missing_decision_id_raises(self):
        with pytest.raises(ValidationError) as exc_info:
            Premise(text="Some premise")
        assert "decision_id" in str(exc_info.value)

    def test_missing_text_raises(self):
        with pytest.raises(ValidationError) as exc_info:
            Premise(decision_id=str(uuid.uuid4()))
        assert "text" in str(exc_info.value)

    def test_embedding_accepts_float_list(self):
        vec = [0.1, -0.2, 0.3, 0.4]
        p = Premise(**self._valid_payload(), embedding=vec)
        assert p.embedding == vec

    def test_from_attributes_config_enabled(self):
        assert Premise.model_config.get("from_attributes") is True


# ── PremiseORM database round-trip ────────────────────────────────────────────

class TestPremiseORMRoundTrip:
    """Insert and retrieve PremiseORM rows, verify data integrity."""

    async def _create_decision(self, session) -> str:
        """Helper: persist a minimal DecisionORM and return its ID."""
        decision_id = str(uuid.uuid4())
        decision = DecisionORM(
            id=decision_id,
            title="Parent decision for premise tests",
            full_narrative="Narrative.",
            premises=[],
            predicted_outcomes=[],
            triggering_conditions=[],
            decision_date=datetime.utcnow(),
        )
        session.add(decision)
        await session.flush()
        return decision_id

    async def test_insert_and_retrieve_basic_fields(self, async_session):
        decision_id = await self._create_decision(async_session)
        premise_id = str(uuid.uuid4())

        orm = PremiseORM(
            id=premise_id,
            decision_id=decision_id,
            text="The market for B2B SaaS is growing at 20% annually",
            confidence_score=0.85,
            status=PremiseStatus.VALID.value,
            testability=PremiseTestability.MEASURABLE.value,
            weight=2.0,
        )
        async_session.add(orm)
        await async_session.flush()

        result = await async_session.execute(
            select(PremiseORM).where(PremiseORM.id == premise_id)
        )
        fetched = result.scalar_one()

        assert fetched.id == premise_id
        assert fetched.decision_id == decision_id
        assert fetched.text == "The market for B2B SaaS is growing at 20% annually"
        assert fetched.confidence_score == pytest.approx(0.85)
        assert fetched.weight == pytest.approx(2.0)

    async def test_to_pydantic_returns_premise_model(self, async_session):
        decision_id = await self._create_decision(async_session)
        premise_id = str(uuid.uuid4())

        orm = PremiseORM(
            id=premise_id,
            decision_id=decision_id,
            text="Founding team has two successful exits",
            status=PremiseStatus.UNCERTAIN.value,
            testability=PremiseTestability.EMPIRICALLY_TESTABLE.value,
            weight=1.5,
            invalidation_evidence="One founder left the company",
        )
        async_session.add(orm)
        await async_session.flush()

        result = await async_session.execute(
            select(PremiseORM).where(PremiseORM.id == premise_id)
        )
        fetched = result.scalar_one()
        pydantic_model = fetched.to_pydantic()

        assert isinstance(pydantic_model, Premise)
        assert pydantic_model.id == premise_id
        assert pydantic_model.decision_id == decision_id
        assert pydantic_model.status == PremiseStatus.UNCERTAIN
        assert pydantic_model.testability == PremiseTestability.EMPIRICALLY_TESTABLE
        assert pydantic_model.weight == pytest.approx(1.5)
        assert pydantic_model.invalidation_evidence == "One founder left the company"

    async def test_default_status_in_orm(self, async_session):
        decision_id = await self._create_decision(async_session)
        orm = PremiseORM(
            id=str(uuid.uuid4()),
            decision_id=decision_id,
            text="Default status premise",
        )
        async_session.add(orm)
        await async_session.flush()

        result = await async_session.execute(
            select(PremiseORM).where(PremiseORM.id == orm.id)
        )
        fetched = result.scalar_one()
        assert fetched.status == PremiseStatus.VALID.value

    async def test_status_mutation_persists(self, async_session):
        decision_id = await self._create_decision(async_session)
        orm = PremiseORM(
            id=str(uuid.uuid4()),
            decision_id=decision_id,
            text="Mutable premise",
        )
        async_session.add(orm)
        await async_session.flush()

        orm.status = PremiseStatus.INVALIDATED.value
        orm.invalidated_at = datetime.utcnow()
        orm.invalidation_evidence = "Contradicted by Q3 earnings report"
        await async_session.flush()

        result = await async_session.execute(
            select(PremiseORM).where(PremiseORM.id == orm.id)
        )
        fetched = result.scalar_one()
        assert fetched.status == PremiseStatus.INVALIDATED.value
        assert fetched.invalidation_evidence == "Contradicted by Q3 earnings report"
        assert fetched.invalidated_at is not None

    async def test_embedding_survives_roundtrip(self, async_session):
        decision_id = await self._create_decision(async_session)
        embedding = [0.11, -0.22, 0.33, 0.44, -0.55]
        orm = PremiseORM(
            id=str(uuid.uuid4()),
            decision_id=decision_id,
            text="Embedded premise",
            embedding=embedding,
        )
        async_session.add(orm)
        await async_session.flush()

        result = await async_session.execute(
            select(PremiseORM).where(PremiseORM.id == orm.id)
        )
        fetched = result.scalar_one()
        pydantic_model = fetched.to_pydantic()
        assert pydantic_model.embedding == embedding

    async def test_multiple_premises_for_one_decision(self, async_session):
        decision_id = await self._create_decision(async_session)
        texts = ["Premise alpha", "Premise beta", "Premise gamma"]

        for text in texts:
            async_session.add(
                PremiseORM(
                    id=str(uuid.uuid4()),
                    decision_id=decision_id,
                    text=text,
                )
            )
        await async_session.flush()

        result = await async_session.execute(
            select(PremiseORM).where(PremiseORM.decision_id == decision_id)
        )
        fetched = result.scalars().all()
        assert len(fetched) == 3
        assert {f.text for f in fetched} == set(texts)
