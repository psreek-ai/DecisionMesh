"""
DecisionMesh — Shared pytest fixtures.

All fixtures here are available to every test module automatically.
The in-memory SQLite engine is rebuilt per test to guarantee isolation.
"""
import uuid
from datetime import datetime
from typing import AsyncGenerator
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
import pytest_asyncio
from sqlalchemy.ext.asyncio import (
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)

# ── Import all ORM models so metadata is populated before create_all ─────────
from decisionmesh.database import Base
from decisionmesh.models import (  # noqa: F401
    agent_log,
    condition,
    counterfactual,
    decision,
    divergence,
    observation,
    premise,
)
from decisionmesh.models.decision import DecisionDNA, DecisionStatus
from decisionmesh.models.premise import Premise, PremiseStatus, PremiseTestability


# ── Database fixtures ─────────────────────────────────────────────────────────

@pytest_asyncio.fixture
async def async_engine():
    """
    Per-test async SQLite in-memory engine.
    Tables are created fresh and dropped after each test — full isolation.
    """
    engine = create_async_engine(
        "sqlite+aiosqlite:///:memory:",
        echo=False,
        connect_args={"check_same_thread": False},
    )
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)

    yield engine

    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.drop_all)
    await engine.dispose()


@pytest_asyncio.fixture
async def async_session(async_engine) -> AsyncGenerator[AsyncSession, None]:
    """
    Per-test async database session bound to the in-memory engine.
    Automatically rolled back after each test so no state leaks.
    """
    factory = async_sessionmaker(
        async_engine,
        class_=AsyncSession,
        expire_on_commit=False,
    )
    async with factory() as session:
        yield session
        await session.rollback()


# ── Anthropic mock helpers ────────────────────────────────────────────────────

def _make_usage(input_tokens: int = 100, output_tokens: int = 50):
    """Return a usage object matching anthropic.types.Usage shape."""
    usage = MagicMock()
    usage.input_tokens = input_tokens
    usage.output_tokens = output_tokens
    return usage


def make_text_block(text: str):
    """Return a mock TextBlock."""
    block = MagicMock()
    block.type = "text"
    block.text = text
    return block


def make_tool_use_block(name: str, tool_input: dict, tool_id: str | None = None):
    """Return a mock ToolUseBlock."""
    block = MagicMock()
    block.type = "tool_use"
    block.name = name
    block.input = tool_input
    block.id = tool_id or f"toolu_{uuid.uuid4().hex[:24]}"
    return block


def make_end_turn_response(text: str = "Done.") -> MagicMock:
    """
    Return a mock anthropic.types.Message that represents a finished response
    (stop_reason == 'end_turn') containing a single text block.
    """
    response = MagicMock()
    response.stop_reason = "end_turn"
    response.content = [make_text_block(text)]
    response.usage = _make_usage()
    return response


def make_tool_use_response(
    tool_name: str,
    tool_input: dict,
    tool_id: str | None = None,
) -> MagicMock:
    """
    Return a mock anthropic.types.Message that requires a tool call
    (stop_reason == 'tool_use') containing a single tool_use block.
    """
    response = MagicMock()
    response.stop_reason = "tool_use"
    response.content = [make_tool_use_block(tool_name, tool_input, tool_id)]
    response.usage = _make_usage()
    return response


@pytest.fixture
def mock_anthropic_client():
    """
    Fixture that patches the Anthropic client used inside BaseAgent.
    Returns a MagicMock whose `.messages.create` side_effect can be configured
    per test to drive the agent loop.

    Usage in a test:
        mock_anthropic_client.messages.create.side_effect = [
            make_tool_use_response("think", {"thought": "reasoning..."}),
            make_end_turn_response("final answer"),
        ]
    """
    with patch("decisionmesh.agents.base.anthropic.Anthropic") as MockAnthropicClass:
        mock_client = MagicMock()
        MockAnthropicClass.return_value = mock_client
        yield mock_client


# ── Domain data fixtures ──────────────────────────────────────────────────────

SAMPLE_NARRATIVE = (
    "On January 15, 2024, I decided to leave my stable corporate job to join "
    "a Series A startup as Head of Engineering. My reasoning: the startup's "
    "product addresses a genuine gap in the B2B SaaS market (premise), their "
    "revenue growth rate is 15% month-over-month (premise), and the founding "
    "team has shipped two successful exits previously (premise). "
    "I predict that within 18 months the company will reach Series B funding "
    "and my equity stake will be worth at least 3x my foregone salary. "
    "I would revisit this decision if the monthly growth rate drops below 8% "
    "for two consecutive months, or if key founders depart."
)


@pytest.fixture
def sample_narrative() -> str:
    """A realistic multi-premise decision narrative for testing capture."""
    return SAMPLE_NARRATIVE


@pytest.fixture
def sample_decision() -> DecisionDNA:
    """
    A fully-populated DecisionDNA instance for use in tests that need
    a pre-existing decision without hitting the database.
    """
    return DecisionDNA(
        id=str(uuid.uuid4()),
        title="Join Series A startup as Head of Engineering",
        full_narrative=SAMPLE_NARRATIVE,
        premises=[
            "The startup's product addresses a genuine gap in the B2B SaaS market",
            "Revenue growth rate is 15% month-over-month",
            "Founding team has shipped two successful exits previously",
        ],
        predicted_outcomes=[
            "Series B funding within 18 months",
            "Equity worth at least 3x foregone salary",
        ],
        triggering_conditions=[
            "Monthly growth rate drops below 8% for two consecutive months",
            "Key founders depart",
        ],
        decision_date=datetime(2024, 1, 15),
        review_interval_days=30,
        status=DecisionStatus.ACTIVE,
        domain="career",
        tags=["startup", "career-change", "equity"],
        web_monitoring_enabled=False,
    )


@pytest.fixture
def sample_premises(sample_decision) -> list[Premise]:
    """Three Premise objects tied to the sample decision."""
    return [
        Premise(
            decision_id=sample_decision.id,
            text="The startup's product addresses a genuine gap in the B2B SaaS market",
            status=PremiseStatus.VALID,
            testability=PremiseTestability.EMPIRICALLY_TESTABLE,
            weight=1.0,
        ),
        Premise(
            decision_id=sample_decision.id,
            text="Revenue growth rate is 15% month-over-month",
            status=PremiseStatus.VALID,
            testability=PremiseTestability.MEASURABLE,
            weight=2.0,
        ),
        Premise(
            decision_id=sample_decision.id,
            text="Founding team has shipped two successful exits previously",
            status=PremiseStatus.VALID,
            testability=PremiseTestability.EMPIRICALLY_TESTABLE,
            weight=1.0,
        ),
    ]
