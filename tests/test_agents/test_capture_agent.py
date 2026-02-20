"""
Tests for decisionmesh.agents.capture_agent.CaptureAgent

Tests:
- _classify_testability heuristic returns correct categories
- capture() extracts DecisionDNA from a narrative using the mocked API
- save_to_db() persists decision and premises to the database
"""
import uuid
from datetime import datetime
from unittest.mock import MagicMock, patch

import pytest
from sqlalchemy import select

from decisionmesh.agents.capture_agent import CaptureAgent, CaptureResult, ExtractDecisionDNAInput
from decisionmesh.models.decision import DecisionDNA, DecisionORM, DecisionStatus
from decisionmesh.models.premise import Premise, PremiseORM, PremiseStatus, PremiseTestability
from tests.conftest import make_end_turn_response, make_tool_use_response


# ── Fixtures ──────────────────────────────────────────────────────────────────

@pytest.fixture
def capture_agent(async_session, mock_anthropic_client):
    from decisionmesh.config import Settings
    config = Settings(
        ANTHROPIC_API_KEY="test-key",
        MAX_AGENT_ITERATIONS=10,
    )
    return CaptureAgent(db_session=async_session, config=config)


def _make_capture_result(decision_id: str | None = None) -> CaptureResult:
    """Build a ready-made CaptureResult without touching the API."""
    did = decision_id or str(uuid.uuid4())
    decision = DecisionDNA(
        id=did,
        title="Join Series A startup",
        full_narrative="I decided to join a startup because growth is 15% MoM.",
        premises=["Growth rate is 15% month-over-month", "Market gap exists"],
        predicted_outcomes=["Series B within 18 months"],
        triggering_conditions=["Growth drops below 8% MoM"],
        decision_date=datetime(2024, 1, 15),
        domain="career",
        tags=["startup"],
    )
    premises_detail = [
        Premise(
            decision_id=did,
            text="Growth rate is 15% month-over-month",
            status=PremiseStatus.VALID,
            testability=PremiseTestability.MEASURABLE,
            weight=1.0,
        ),
        Premise(
            decision_id=did,
            text="Market gap exists",
            status=PremiseStatus.VALID,
            testability=PremiseTestability.EMPIRICALLY_TESTABLE,
            weight=1.0,
        ),
    ]
    return CaptureResult(decision=decision, premises_detail=premises_detail)


# ── _classify_testability heuristic ──────────────────────────────────────────

class TestClassifyTestability:
    """
    The heuristic in CaptureAgent._classify_testability uses keyword signals.
    These tests confirm the correct bucket is chosen for representative inputs.
    """

    def test_measurable_signals_detected(self, capture_agent):
        cases = [
            "Revenue growth rate is 15%",
            "Operating costs will decrease by a fixed amount",
            "The price of raw materials is below $50/unit",
            "Customer metric will improve by 20%",
        ]
        for text in cases:
            result = capture_agent._classify_testability(text)
            assert result == PremiseTestability.MEASURABLE.value, (
                f"Expected 'measurable' for: {text!r}, got {result!r}"
            )

    def test_subjective_untestable_signals_detected(self, capture_agent):
        cases = [
            "I believe the team has the right culture",
            "I feel strongly about the product direction",
            "We trust our instincts on this market",
            "I hope the economy stabilises",
        ]
        for text in cases:
            result = capture_agent._classify_testability(text)
            assert result == PremiseTestability.SUBJECTIVE_UNTESTABLE.value, (
                f"Expected 'subjective_untestable' for: {text!r}, got {result!r}"
            )

    def test_empirically_testable_default(self, capture_agent):
        cases = [
            "The founding team has shipped two products to market",
            "Remote work infrastructure supports async collaboration",
            "The regulatory environment allows this product category",
        ]
        for text in cases:
            result = capture_agent._classify_testability(text)
            assert result == PremiseTestability.EMPIRICALLY_TESTABLE.value, (
                f"Expected 'empirically_testable' for: {text!r}, got {result!r}"
            )

    def test_measurable_takes_precedence_over_subjective(self, capture_agent):
        """
        A sentence with both measurable and subjective signals —
        the order in the if/elif means measurable is checked for subjective
        signals first. Verify the actual priority in the implementation.
        """
        # The implementation checks untestable first, then measurable
        text = "I believe the growth rate will reach 20%"
        result = capture_agent._classify_testability(text)
        # "believe" is an untestable signal → subjective_untestable wins
        assert result == PremiseTestability.SUBJECTIVE_UNTESTABLE.value

    def test_case_insensitive_matching(self, capture_agent):
        """Heuristic must be case-insensitive (lower() applied)."""
        assert capture_agent._classify_testability("GROWTH RATE IS HIGH") == PremiseTestability.MEASURABLE.value
        assert capture_agent._classify_testability("I BELIEVE IN THIS") == PremiseTestability.SUBJECTIVE_UNTESTABLE.value


# ── capture() end-to-end with mocked API ─────────────────────────────────────

class TestCaptureAgentCapture:
    async def test_capture_returns_capture_result(
        self, capture_agent, mock_anthropic_client, sample_narrative
    ):
        """
        Simulate the tool-use flow the agent would actually perform:
          1. think tool
          2. check_premise_testability (×3)
          3. extract_decision_dna
          4. end_turn
        The agent stores the extracted DNA in self._extracted_dna.
        """
        premise1 = "The startup addresses a genuine gap in the B2B SaaS market"
        premise2 = "Revenue growth rate is 15% month-over-month"
        premise3 = "Founding team has shipped two successful exits previously"

        mock_anthropic_client.messages.create.side_effect = [
            # Step 1: think
            make_tool_use_response("think", {"thought": "Let me extract DNA..."}),
            # Step 2: check_premise_testability for each premise
            make_tool_use_response("check_premise_testability", {"premise": premise1}),
            make_tool_use_response("check_premise_testability", {"premise": premise2}),
            make_tool_use_response("check_premise_testability", {"premise": premise3}),
            # Step 3: extract_decision_dna
            make_tool_use_response(
                "extract_decision_dna",
                {
                    "title": "Join Series A startup as Head of Engineering",
                    "full_narrative": sample_narrative,
                    "premises": [premise1, premise2, premise3],
                    "predicted_outcomes": ["Series B funding within 18 months"],
                    "triggering_conditions": ["Growth drops below 8% MoM for 2 months"],
                    "decision_date": "2024-01-15",
                    "domain": "career",
                    "tags": ["startup", "career-change"],
                    "web_monitoring_needed": False,
                },
            ),
            # Step 4: end_turn
            make_end_turn_response("Decision DNA extracted successfully."),
        ]

        # Patch the embedding tool so we don't need a real model
        with patch(
            "decisionmesh.tools.embedding_tool.embed_text",
            return_value=[0.1, 0.2, 0.3],
        ):
            result = await capture_agent.capture(sample_narrative)

        assert isinstance(result, CaptureResult)
        assert result.clarification_needed is None
        assert result.decision is not None
        assert result.decision.title == "Join Series A startup as Head of Engineering"
        assert result.decision.status == DecisionStatus.ACTIVE
        assert result.decision.domain == "career"
        assert len(result.decision.premises) == 3

    async def test_capture_builds_premise_objects(
        self, capture_agent, mock_anthropic_client, sample_narrative
    ):
        """Premises in the CaptureResult must be Premise objects with correct decision_id."""
        premise1 = "Revenue growth rate is 15% month-over-month"

        mock_anthropic_client.messages.create.side_effect = [
            make_tool_use_response("think", {"thought": "reasoning"}),
            make_tool_use_response("check_premise_testability", {"premise": premise1}),
            make_tool_use_response(
                "extract_decision_dna",
                {
                    "title": "Minimal decision",
                    "full_narrative": sample_narrative,
                    "premises": [premise1],
                    "predicted_outcomes": [],
                    "triggering_conditions": [],
                    "decision_date": "2024-01-15",
                    "domain": "career",
                    "tags": [],
                    "web_monitoring_needed": False,
                },
            ),
            make_end_turn_response("Done."),
        ]

        with patch("decisionmesh.tools.embedding_tool.embed_text", return_value=[0.0]):
            result = await capture_agent.capture(sample_narrative)

        assert len(result.premises_detail) == 1
        prem = result.premises_detail[0]
        assert isinstance(prem, Premise)
        assert prem.decision_id == result.decision.id
        assert prem.text == premise1
        # "rate" triggers measurable
        assert prem.testability == PremiseTestability.MEASURABLE

    async def test_capture_raises_if_no_dna_extracted(
        self, capture_agent, mock_anthropic_client, sample_narrative
    ):
        """If the agent never calls extract_decision_dna, capture() must raise ValueError."""
        mock_anthropic_client.messages.create.return_value = make_end_turn_response(
            "I couldn't extract anything."
        )

        with patch("decisionmesh.tools.embedding_tool.embed_text", return_value=[]):
            with pytest.raises(ValueError, match="CaptureAgent failed to extract"):
                await capture_agent.capture(sample_narrative)

    async def test_capture_returns_clarification_when_needed(
        self, capture_agent, mock_anthropic_client, sample_narrative
    ):
        """When ask_clarification tool is called, result has clarification_needed set."""
        mock_anthropic_client.messages.create.side_effect = [
            make_tool_use_response(
                "ask_clarification",
                {
                    "question": "When exactly was this decision made?",
                    "why_needed": "Date is ambiguous.",
                },
            ),
            make_end_turn_response("Waiting for clarification."),
        ]

        with patch("decisionmesh.tools.embedding_tool.embed_text", return_value=[]):
            result = await capture_agent.capture(sample_narrative)

        assert result.clarification_needed == "When exactly was this decision made?"
        assert result.decision.title == "[Pending clarification]"


# ── save_to_db() ──────────────────────────────────────────────────────────────

class TestCaptureAgentSaveToDb:
    async def test_save_persists_decision_orm(self, capture_agent, async_session):
        result = _make_capture_result()
        decision_id = await capture_agent.save_to_db(result)

        fetched = await async_session.execute(
            select(DecisionORM).where(DecisionORM.id == decision_id)
        )
        orm = fetched.scalar_one()

        assert orm.id == decision_id
        assert orm.title == "Join Series A startup"
        assert orm.domain == "career"

    async def test_save_returns_decision_id(self, capture_agent, async_session):
        result = _make_capture_result()
        returned_id = await capture_agent.save_to_db(result)
        assert returned_id == result.decision.id

    async def test_save_persists_all_premises(self, capture_agent, async_session):
        result = _make_capture_result()
        decision_id = await capture_agent.save_to_db(result)

        fetched_premises = await async_session.execute(
            select(PremiseORM).where(PremiseORM.decision_id == decision_id)
        )
        premises = fetched_premises.scalars().all()

        assert len(premises) == 2
        texts = {p.text for p in premises}
        assert "Growth rate is 15% month-over-month" in texts
        assert "Market gap exists" in texts

    async def test_save_premises_have_correct_decision_id(self, capture_agent, async_session):
        result = _make_capture_result()
        decision_id = await capture_agent.save_to_db(result)

        fetched_premises = await async_session.execute(
            select(PremiseORM).where(PremiseORM.decision_id == decision_id)
        )
        for p in fetched_premises.scalars().all():
            assert p.decision_id == decision_id

    async def test_save_preserves_premise_testability(self, capture_agent, async_session):
        result = _make_capture_result()
        await capture_agent.save_to_db(result)

        fetched_premises = await async_session.execute(
            select(PremiseORM).where(PremiseORM.decision_id == result.decision.id)
        )
        premises = {p.text: p for p in fetched_premises.scalars().all()}

        growth_premise = premises["Growth rate is 15% month-over-month"]
        assert growth_premise.testability == PremiseTestability.MEASURABLE.value

    async def test_save_preserves_decision_status(self, capture_agent, async_session):
        result = _make_capture_result()
        decision_id = await capture_agent.save_to_db(result)

        fetched = await async_session.execute(
            select(DecisionORM).where(DecisionORM.id == decision_id)
        )
        orm = fetched.scalar_one()
        assert orm.status == DecisionStatus.ACTIVE.value

    async def test_save_stores_tags(self, capture_agent, async_session):
        result = _make_capture_result()
        assert result.decision.tags == ["startup"]
        decision_id = await capture_agent.save_to_db(result)

        fetched = await async_session.execute(
            select(DecisionORM).where(DecisionORM.id == decision_id)
        )
        orm = fetched.scalar_one()
        pydantic_model = orm.to_pydantic()
        assert "startup" in pydantic_model.tags
