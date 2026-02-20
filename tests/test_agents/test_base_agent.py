"""
Tests for decisionmesh.agents.base.BaseAgent

Tests:
- _extract_final_text correctly extracts text from a response
- run() returns final text on end_turn stop_reason
- run() handles tool_use stop reason and continues the loop
- MaxIterationsExceeded is raised when the loop runs out of iterations
"""
import uuid
from datetime import datetime
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from decisionmesh.agents.base import BaseAgent, MaxIterationsExceeded
from decisionmesh.models.decision import DecisionORM
from tests.conftest import (
    make_end_turn_response,
    make_text_block,
    make_tool_use_block,
    make_tool_use_response,
)


# ── Concrete subclass for testing (BaseAgent is abstract) ─────────────────────

class ConcreteAgent(BaseAgent):
    """Minimal concrete subclass that exposes BaseAgent behaviour."""
    agent_name = "test_agent"

    def _register_tools(self):
        return []  # No tools needed for base tests

    async def _dispatch_tool(self, tool_name, tool_input, decision_id=None):
        # Echo back the input so we can see it was called
        return {"echoed": tool_name, "input": tool_input}


# ── Fixtures ──────────────────────────────────────────────────────────────────

@pytest.fixture
def agent(async_session, mock_anthropic_client):
    """
    Return a ConcreteAgent whose Anthropic client is mocked.
    The mock_anthropic_client fixture patches anthropic.Anthropic at the
    module level so the agent constructor picks up the mock.
    """
    from decisionmesh.config import Settings
    config = Settings(
        ANTHROPIC_API_KEY="test-key",
        MAX_AGENT_ITERATIONS=5,
    )
    return ConcreteAgent(db_session=async_session, config=config)


# ── _extract_final_text ───────────────────────────────────────────────────────

class TestExtractFinalText:
    def test_single_text_block(self, agent):
        response = MagicMock()
        response.content = [make_text_block("Hello, world!")]
        assert agent._extract_final_text(response) == "Hello, world!"

    def test_multiple_text_blocks_joined(self, agent):
        response = MagicMock()
        response.content = [
            make_text_block("First paragraph."),
            make_text_block("Second paragraph."),
        ]
        result = agent._extract_final_text(response)
        assert "First paragraph." in result
        assert "Second paragraph." in result

    def test_non_text_blocks_ignored(self, agent):
        response = MagicMock()
        tool_block = make_tool_use_block("think", {"thought": "reasoning"})
        text_block = make_text_block("Final answer.")
        response.content = [tool_block, text_block]
        result = agent._extract_final_text(response)
        assert result == "Final answer."

    def test_empty_content_returns_empty_string(self, agent):
        response = MagicMock()
        response.content = []
        assert agent._extract_final_text(response) == ""

    def test_only_tool_blocks_returns_empty_string(self, agent):
        response = MagicMock()
        response.content = [make_tool_use_block("think", {"thought": "x"})]
        assert agent._extract_final_text(response) == ""

    def test_leading_trailing_whitespace_stripped(self, agent):
        response = MagicMock()
        response.content = [make_text_block("  trimmed  ")]
        result = agent._extract_final_text(response)
        assert result == "trimmed"


# ── run() — end_turn path ─────────────────────────────────────────────────────

class TestRunEndTurn:
    async def test_immediate_end_turn_returns_text(self, agent, mock_anthropic_client):
        """
        When the API responds with end_turn on the first call,
        run() should return the text content immediately.
        """
        mock_anthropic_client.messages.create.return_value = make_end_turn_response(
            "The answer is 42."
        )

        result = await agent.run(
            system_prompt="You are a test agent.",
            user_message="What is the answer?",
        )

        assert result == "The answer is 42."
        # Verify the API was called exactly once
        mock_anthropic_client.messages.create.assert_called_once()

    async def test_run_passes_system_and_user_message(self, agent, mock_anthropic_client):
        """
        The messages passed to the API must include the user_message
        and the system prompt.
        """
        mock_anthropic_client.messages.create.return_value = make_end_turn_response("ok")

        await agent.run(
            system_prompt="Test system prompt.",
            user_message="Test user message.",
        )

        call_kwargs = mock_anthropic_client.messages.create.call_args[1]
        assert call_kwargs["system"] == "Test system prompt."
        assert call_kwargs["messages"][0]["role"] == "user"
        assert call_kwargs["messages"][0]["content"] == "Test user message."


# ── run() — tool_use loop ─────────────────────────────────────────────────────

class TestRunToolUseLoop:
    async def test_tool_use_then_end_turn(self, agent, mock_anthropic_client):
        """
        API first returns tool_use, then end_turn.
        run() should continue the loop, execute the tool, and return the
        final text from the second response.
        """
        mock_anthropic_client.messages.create.side_effect = [
            make_tool_use_response("think", {"thought": "Let me reason..."}),
            make_end_turn_response("After thinking, the answer is X."),
        ]

        result = await agent.run(
            system_prompt="sys",
            user_message="user",
        )

        assert result == "After thinking, the answer is X."
        assert mock_anthropic_client.messages.create.call_count == 2

    async def test_messages_grow_with_tool_results(self, agent, mock_anthropic_client):
        """
        After a tool call, the messages list must contain the assistant's
        tool_use response AND the tool_result — so the next API call has context.
        """
        tool_response = make_tool_use_response(
            "think",
            {"thought": "reasoning"},
            tool_id="toolu_abc123",
        )
        end_response = make_end_turn_response("Done.")

        mock_anthropic_client.messages.create.side_effect = [tool_response, end_response]

        await agent.run(system_prompt="sys", user_message="user")

        # Second call should have 3 messages: user, assistant (tool_use), user (tool_result)
        second_call_kwargs = mock_anthropic_client.messages.create.call_args_list[1][1]
        messages = second_call_kwargs["messages"]
        assert len(messages) == 3
        assert messages[0]["role"] == "user"
        assert messages[1]["role"] == "assistant"
        assert messages[2]["role"] == "user"
        # The tool result message should contain a tool_result block
        tool_result_content = messages[2]["content"]
        assert isinstance(tool_result_content, list)
        assert tool_result_content[0]["type"] == "tool_result"
        assert tool_result_content[0]["tool_use_id"] == "toolu_abc123"

    async def test_multiple_tool_loops_before_end(self, agent, mock_anthropic_client):
        """
        Three tool_use rounds followed by end_turn — the agent must
        continue until the end_turn is received.
        """
        mock_anthropic_client.messages.create.side_effect = [
            make_tool_use_response("think", {"thought": "step 1"}),
            make_tool_use_response("think", {"thought": "step 2"}),
            make_tool_use_response("think", {"thought": "step 3"}),
            make_end_turn_response("Final answer after 3 tools."),
        ]

        result = await agent.run(
            system_prompt="sys",
            user_message="user",
            max_iterations=10,  # Plenty of room
        )

        assert result == "Final answer after 3 tools."
        assert mock_anthropic_client.messages.create.call_count == 4


# ── MaxIterationsExceeded ─────────────────────────────────────────────────────

class TestMaxIterationsExceeded:
    async def test_raises_when_loop_exhausted(self, agent, mock_anthropic_client):
        """
        If the agent keeps getting tool_use responses beyond max_iterations,
        MaxIterationsExceeded must be raised.
        """
        # Always return a tool_use — the loop will never converge
        mock_anthropic_client.messages.create.return_value = make_tool_use_response(
            "think", {"thought": "still thinking..."}
        )

        with pytest.raises(MaxIterationsExceeded) as exc_info:
            await agent.run(
                system_prompt="sys",
                user_message="user",
                max_iterations=3,
            )

        assert "test_agent" in str(exc_info.value)
        assert "3" in str(exc_info.value)

    async def test_api_called_exactly_max_iterations_times(self, agent, mock_anthropic_client):
        """
        The API must be called exactly max_iterations times before giving up.
        """
        mock_anthropic_client.messages.create.return_value = make_tool_use_response(
            "think", {"thought": "loop"}
        )

        with pytest.raises(MaxIterationsExceeded):
            await agent.run(
                system_prompt="sys",
                user_message="user",
                max_iterations=4,
            )

        assert mock_anthropic_client.messages.create.call_count == 4

    async def test_exception_message_includes_agent_name(self, agent, mock_anthropic_client):
        mock_anthropic_client.messages.create.return_value = make_tool_use_response(
            "think", {"thought": "x"}
        )

        with pytest.raises(MaxIterationsExceeded) as exc_info:
            await agent.run(
                system_prompt="sys",
                user_message="user",
                max_iterations=2,
            )

        assert "test_agent" in str(exc_info.value)
