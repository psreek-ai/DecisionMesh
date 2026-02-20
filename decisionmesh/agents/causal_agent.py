"""
DecisionMesh — Layer 3: Causal Inference Agent
Deep causal reasoning + counterfactual simulation.
Uses extended thinking (budget: 8000 tokens) for maximum depth.
"""
import asyncio
import json
import logging
from typing import Any, Optional

import anthropic
from sqlalchemy.ext.asyncio import AsyncSession

from decisionmesh.agents.base import BaseAgent, MaxIterationsExceeded
from decisionmesh.config import Settings, settings
from decisionmesh.tools.think_tool import THINK_TOOL
from decisionmesh.tools.db_tools import (
    GET_ALL_ACTIVE_DECISIONS_TOOL, SAVE_COUNTERFACTUAL_TOOL,
    SaveCounterfactualInput,
    db_get_decision, db_get_all_active_decisions, db_save_counterfactual,
)
from decisionmesh.tools.similarity_tool import SIMILARITY_SEARCH_TOOL, semantic_search_decisions

logger = logging.getLogger(__name__)


CAUSAL_SYSTEM_PROMPT = """You are the Causal Inference Agent for DecisionMesh.
A divergence event has been detected: the real world has moved away from the premises of a decision.

Your job is deep causal reasoning:
1. Trace the CAUSAL CHAIN from premise failure to predicted outcome failure
2. Construct a COUNTERFACTUAL: if the opposite decision had been made at the time, what would the state of the world likely be NOW?
3. Identify COMPOUNDING RISKS: are any other active decisions by this user built on the same now-invalidated premises?
4. Produce a REVISION BRIEF: a structured set of options for the user

EPISTEMIC STANDARDS:
- Clearly distinguish between what you KNOW (from evidence), what you INFER (from reasoning), and what you SPECULATE (from patterns)
- Every step in the causal chain must be labeled: [KNOWN] / [INFERRED] / [SPECULATED]
- Include confidence intervals on all predictions
- Acknowledge alternative causal explanations

Use your full extended thinking budget. This reasoning will be preserved permanently as part of the decision's audit trail.

Call save_counterfactual when you have completed your analysis.
Call create_revision_brief to produce actionable options for the Revision Advisor."""


GET_DIVERGENCE_EVENT_TOOL = {
    "name": "get_divergence_event_with_context",
    "description": "Retrieve a divergence event with full decision context, premises, and observations.",
    "input_schema": {
        "type": "object",
        "properties": {
            "divergence_event_id": {"type": "string"},
            "decision_id": {"type": "string"},
        },
        "required": ["divergence_event_id", "decision_id"],
    },
}

CREATE_REVISION_BRIEF_TOOL = {
    "name": "create_revision_brief",
    "description": "Store a structured revision brief with options for the Revision Advisor Agent.",
    "input_schema": {
        "type": "object",
        "properties": {
            "decision_id": {"type": "string"},
            "divergence_event_id": {"type": "string"},
            "compounding_risk_decision_ids": {
                "type": "array",
                "items": {"type": "string"},
                "description": "IDs of other decisions that may be affected by the same premise failure.",
            },
            "revision_options": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "option": {"type": "string"},
                        "description": {"type": "string"},
                        "consequence": {"type": "string"},
                    },
                },
                "description": "List of revision options for the user.",
            },
            "primary_recommendation": {
                "type": "string",
                "description": "The option that most aligns with the original decision intent.",
            },
        },
        "required": ["decision_id", "divergence_event_id", "revision_options", "primary_recommendation"],
    },
}


class CausalAgent(BaseAgent):
    agent_name = "causal_agent"

    def __init__(self, db_session: AsyncSession, config: Settings = None):
        super().__init__(db_session, config)
        self._counterfactual_id: Optional[str] = None
        self._revision_brief: Optional[dict] = None

    def _register_tools(self) -> list[dict]:
        return [
            THINK_TOOL,
            GET_DIVERGENCE_EVENT_TOOL,
            GET_ALL_ACTIVE_DECISIONS_TOOL,
            SIMILARITY_SEARCH_TOOL,
            SAVE_COUNTERFACTUAL_TOOL,
            CREATE_REVISION_BRIEF_TOOL,
        ]

    async def _call_api_with_backoff(self, system_prompt, messages, extra_headers=None):
        """Override to use extended thinking."""
        backoff_seconds = [2, 4, 8, 16]
        last_error = None

        for attempt, wait in enumerate([0] + backoff_seconds):
            if wait > 0:
                await asyncio.sleep(wait)

            try:
                loop = asyncio.get_event_loop()
                response = await loop.run_in_executor(
                    None,
                    lambda: self.client.messages.create(
                        model=self.model,
                        max_tokens=16000,
                        thinking={
                            "type": "enabled",
                            "budget_tokens": self.config.CAUSAL_THINKING_BUDGET,
                        },
                        system=system_prompt,
                        messages=messages,
                        tools=self.tools,
                        betas=["interleaved-thinking-2025-05-14"],
                    ),
                )
                return response
            except anthropic.RateLimitError as e:
                last_error = e
                continue
            except anthropic.APIStatusError as e:
                if e.status_code >= 500:
                    last_error = e
                    continue
                raise

        raise last_error or RuntimeError("Causal agent API call failed after retries")

    async def _dispatch_tool(self, tool_name: str, tool_input: dict, decision_id: Optional[str] = None) -> Any:
        if tool_name == "think":
            return {"status": "thought_recorded"}

        if tool_name == "get_divergence_event_with_context":
            return await self._get_divergence_with_context(
                tool_input["divergence_event_id"],
                tool_input["decision_id"],
            )

        if tool_name == "get_all_active_decisions":
            return await db_get_all_active_decisions(tool_input.get("limit", 50), self.db)

        if tool_name == "semantic_search_decisions":
            return await semantic_search_decisions(
                tool_input["query_text"],
                tool_input.get("limit", 5),
                tool_input.get("min_similarity", 0.5),
                self.db,
            )

        if tool_name == "save_counterfactual":
            data = SaveCounterfactualInput(**tool_input)
            result = await db_save_counterfactual(data, self.db)
            self._counterfactual_id = result.get("counterfactual_id")
            return result

        if tool_name == "create_revision_brief":
            self._revision_brief = tool_input
            return {"status": "revision_brief_stored"}

        return await super()._dispatch_tool(tool_name, tool_input, decision_id)

    async def _get_divergence_with_context(self, divergence_event_id: str, decision_id: str) -> dict:
        """Fetch full context: divergence event + decision + premises + observations."""
        from sqlalchemy import select
        from decisionmesh.models.divergence import DivergenceEventORM
        from decisionmesh.models.observation import ObservationORM

        # Decision with premises
        decision_data = await db_get_decision(decision_id, self.db)

        # Divergence event
        result = await self.db.execute(
            select(DivergenceEventORM).where(DivergenceEventORM.id == divergence_event_id)
        )
        event = result.scalar_one_or_none()
        divergence_data = event.to_pydantic().model_dump(mode="json") if event else {}

        # Recent observations for affected premises
        affected_premise_ids = divergence_data.get("affected_premise_ids", [])
        observations = []
        for pid in affected_premise_ids[:5]:
            obs_result = await self.db.execute(
                select(ObservationORM)
                .where(ObservationORM.premise_id == pid)
                .order_by(ObservationORM.observed_at.desc())
                .limit(3)
            )
            for obs in obs_result.scalars().all():
                observations.append(obs.to_pydantic().model_dump(mode="json"))

        return {
            "decision": decision_data,
            "divergence_event": divergence_data,
            "recent_observations": observations,
        }

    async def analyze(self, divergence_event_id: str, decision_id: str) -> dict:
        """
        Main entry point: run deep causal analysis on a divergence event.
        Returns summary with counterfactual_id and revision_brief.
        """
        user_message = f"""A divergence event has been detected for decision {decision_id}.
Divergence event ID: {divergence_event_id}

Please conduct a full causal analysis:

1. Call get_divergence_event_with_context to get the full picture
2. Use your extended thinking to reason deeply about:
   - What causal chain led from premise failure to outcome divergence?
   - What would have happened if the opposite decision had been made?
   - Are there other active decisions that share the same invalidated premises?
3. Call semantic_search_decisions to find related decisions that may be at compounding risk
4. Call save_counterfactual with your full causal reasoning
5. Call create_revision_brief with structured options for the user

Remember to label every inference step: [KNOWN] / [INFERRED] / [SPECULATED]
Include confidence estimates for your counterfactual projections."""

        await self.run(
            system_prompt=CAUSAL_SYSTEM_PROMPT,
            user_message=user_message,
            decision_id=decision_id,
        )

        return {
            "decision_id": decision_id,
            "divergence_event_id": divergence_event_id,
            "counterfactual_id": self._counterfactual_id,
            "revision_brief": self._revision_brief,
            "status": "analysis_complete" if self._counterfactual_id else "analysis_incomplete",
        }
