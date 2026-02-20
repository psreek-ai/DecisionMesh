"""
DecisionMesh — Base Agent
All agents inherit from this class.
Handles: Anthropic client, think tool, tool-use loop (while-loop, NOT recursion),
transparent logging to agent_log table, exponential backoff on API errors.
"""
import asyncio
import json
import logging
from abc import abstractmethod
from datetime import datetime
from typing import Any, Optional

import anthropic
from sqlalchemy.ext.asyncio import AsyncSession

from decisionmesh.config import settings
from decisionmesh.tools.think_tool import THINK_TOOL

logger = logging.getLogger(__name__)


class MaxIterationsExceeded(Exception):
    """Raised when an agent exceeds its maximum iteration count."""
    pass


class BaseAgent:
    """Base class for all DecisionMesh agents."""

    agent_name: str = "base_agent"

    def __init__(self, db_session: AsyncSession, config=None):
        self.config = config or settings
        self.client = anthropic.Anthropic(api_key=self.config.ANTHROPIC_API_KEY)
        self.model = self.config.DEFAULT_MODEL
        self.db = db_session
        self.tools = self._register_tools()

    def _register_tools(self) -> list[dict]:
        """Subclasses extend this to register their specific tools. Think tool always first."""
        return [THINK_TOOL]

    async def run(
        self,
        system_prompt: str,
        user_message: str,
        decision_id: Optional[str] = None,
        max_iterations: Optional[int] = None,
        extra_headers: Optional[dict] = None,
    ) -> str:
        """
        Core agentic loop — simple while-loop, not recursive.
        Returns the final text response from the agent.
        """
        max_iter = max_iterations or self.config.MAX_AGENT_ITERATIONS
        messages = [{"role": "user", "content": user_message}]

        for iteration in range(max_iter):
            response = await self._call_api_with_backoff(
                system_prompt, messages, extra_headers=extra_headers
            )

            # Log this iteration
            await self._log_iteration(iteration, response, decision_id)

            if response.stop_reason == "end_turn":
                return self._extract_final_text(response)

            if response.stop_reason == "tool_use":
                tool_results = await self._execute_tools(response, decision_id)
                messages.append({"role": "assistant", "content": response.content})
                messages.append({"role": "user", "content": tool_results})
                continue

            # Unexpected stop reason
            logger.warning(f"Unexpected stop_reason: {response.stop_reason}")
            return self._extract_final_text(response)

        raise MaxIterationsExceeded(
            f"Agent '{self.agent_name}' did not complete in {max_iter} iterations."
        )

    async def _call_api_with_backoff(
        self,
        system_prompt: str,
        messages: list[dict],
        extra_headers: Optional[dict] = None,
    ) -> anthropic.types.Message:
        """Call the Anthropic API with exponential backoff on errors."""
        backoff_seconds = [2, 4, 8, 16]
        last_error = None

        for attempt, wait in enumerate([0] + backoff_seconds):
            if wait > 0:
                logger.info(f"Retrying API call in {wait}s (attempt {attempt + 1})")
                await asyncio.sleep(wait)

            try:
                kwargs: dict[str, Any] = {
                    "model": self.model,
                    "max_tokens": 8192,
                    "system": system_prompt,
                    "messages": messages,
                    "tools": self.tools,
                }
                if extra_headers:
                    kwargs["extra_headers"] = extra_headers

                # Use sync client in executor to avoid blocking event loop
                loop = asyncio.get_event_loop()
                response = await loop.run_in_executor(
                    None,
                    lambda: self.client.messages.create(**kwargs),
                )
                return response

            except anthropic.RateLimitError as e:
                last_error = e
                logger.warning(f"Rate limit hit: {e}")
                continue
            except anthropic.APIStatusError as e:
                if e.status_code >= 500:
                    last_error = e
                    logger.warning(f"Server error {e.status_code}: {e}")
                    continue
                raise  # Client errors (4xx except 429) are not retried

        raise last_error or RuntimeError("API call failed after retries")

    async def _execute_tools(
        self,
        response: anthropic.types.Message,
        decision_id: Optional[str] = None,
    ) -> list[dict]:
        """Execute all tool calls in the response and return tool results."""
        tool_results = []

        for block in response.content:
            if block.type != "tool_use":
                continue

            tool_name = block.name
            tool_input = block.input
            tool_id = block.id

            try:
                result = await self._dispatch_tool(tool_name, tool_input, decision_id)
                result_str = json.dumps(result) if not isinstance(result, str) else result
            except Exception as e:
                logger.error(f"Tool '{tool_name}' failed: {e}")
                result_str = json.dumps({"error": str(e)})

            tool_results.append({
                "type": "tool_result",
                "tool_use_id": tool_id,
                "content": result_str,
            })

        return tool_results

    async def _dispatch_tool(
        self,
        tool_name: str,
        tool_input: dict,
        decision_id: Optional[str] = None,
    ) -> Any:
        """
        Dispatch tool calls to their implementations.
        Subclasses override this to add agent-specific tools.
        """
        if tool_name == "think":
            # Think tool is a no-op — its value is the logging side-effect
            return {"status": "thought_recorded", "thought": tool_input.get("thought", "")[:200]}

        raise ValueError(f"Unknown tool: {tool_name}")

    async def _log_iteration(
        self,
        iteration: int,
        response: anthropic.types.Message,
        decision_id: Optional[str] = None,
    ) -> None:
        """Log agent iteration to the audit trail."""
        try:
            from decisionmesh.models.agent_log import AgentLogORM

            # Extract text and thinking content
            text_parts = []
            thinking_parts = []
            tool_calls = []

            for block in response.content:
                if hasattr(block, "type"):
                    if block.type == "text":
                        text_parts.append(block.text)
                    elif block.type == "thinking":
                        thinking_parts.append(block.thinking)
                    elif block.type == "tool_use":
                        tool_calls.append({
                            "tool": block.name,
                            "input": block.input,
                        })

            # Log one entry per tool call, or one entry for the turn
            if tool_calls:
                for tc in tool_calls:
                    log_entry = AgentLogORM(
                        decision_id=decision_id,
                        agent_name=self.agent_name,
                        iteration=iteration,
                        tool_name=tc["tool"],
                        tool_input=tc["input"],
                        thinking_content="\n".join(thinking_parts) or None,
                        text_content="\n".join(text_parts) or None,
                        stop_reason=response.stop_reason,
                        input_tokens=response.usage.input_tokens,
                        output_tokens=response.usage.output_tokens,
                    )
                    self.db.add(log_entry)
            else:
                log_entry = AgentLogORM(
                    decision_id=decision_id,
                    agent_name=self.agent_name,
                    iteration=iteration,
                    thinking_content="\n".join(thinking_parts) or None,
                    text_content="\n".join(text_parts) or None,
                    stop_reason=response.stop_reason,
                    input_tokens=response.usage.input_tokens,
                    output_tokens=response.usage.output_tokens,
                )
                self.db.add(log_entry)

            await self.db.flush()
        except Exception as e:
            logger.error(f"Failed to log agent iteration: {e}")

    def _extract_final_text(self, response: anthropic.types.Message) -> str:
        """Extract the final text content from a response."""
        parts = []
        for block in response.content:
            if hasattr(block, "type") and block.type == "text":
                parts.append(block.text)
        return "\n".join(parts).strip()
