"""
DecisionMesh — Layer 2: Condition Monitor Agent
THE HEART OF THE SYSTEM. Runs on a schedule to check whether premises still hold.
Produces Observations, computes divergence scores, creates DivergenceEvents.
"""
import json
import logging
from datetime import datetime
from typing import Any, Optional

from sqlalchemy.ext.asyncio import AsyncSession

from decisionmesh.agents.base import BaseAgent
from decisionmesh.config import Settings, settings
from decisionmesh.models.premise import PremiseStatus
from decisionmesh.tools.think_tool import THINK_TOOL
from decisionmesh.tools.search_tool import SEARCH_TOOL, execute_web_search
from decisionmesh.tools.db_tools import (
    GET_DECISION_TOOL, GET_PRIOR_OBSERVATIONS_TOOL,
    SAVE_OBSERVATION_TOOL, COMPUTE_DIVERGENCE_TOOL,
    CREATE_DIVERGENCE_EVENT_TOOL,
    GetDecisionInput, GetPriorObservationsInput, SaveObservationInput,
    ComputeDivergenceInput, CreateDivergenceEventInput,
    db_get_decision, db_get_prior_observations, db_save_observation,
    db_compute_divergence, db_create_divergence_event,
)

logger = logging.getLogger(__name__)


MONITOR_SYSTEM_PROMPT = """You are the Condition Monitor Agent for DecisionMesh.
You are running a scheduled check on a specific decision's premises.

Your job:
1. For each premise in the decision, assess whether current evidence supports or undermines it
2. If web_monitoring_enabled=True, use web_search to find relevant evidence
3. If web_monitoring_enabled=False, reason from information already in the database (prior observations, the decision's own narrative)
4. For each premise, produce an Observation record using the save_observation tool
5. Compute a divergence_score using compute_divergence_score
6. If divergence_score > threshold, create a DivergenceEvent using create_divergence_event

CRITICAL RULES:
- Never fabricate evidence. If you cannot verify a premise, mark it "unverifiable" — do not guess.
- Always cite sources when using web search.
- Your reasoning must be logged in full via the think tool before drawing any conclusions.
- Be conservative: prefer "uncertain" over "invalidated" unless evidence is strong.
- The think tool MUST be called before classifying any premise status.

Divergence score formula:
  score = Σ(weight_i × invalidation_score_i) / Σ(weight_i)
  where: valid=0.0, uncertain=0.4, invalidated=1.0, unverifiable=0.2"""


# ── Additional tool schemas ───────────────────────────────────────────────────

TRIGGER_CAUSAL_TOOL = {
    "name": "trigger_causal_agent",
    "description": "Dispatch the Causal Inference Agent asynchronously to analyze a divergence event.",
    "input_schema": {
        "type": "object",
        "properties": {
            "divergence_event_id": {"type": "string", "description": "The UUID of the divergence event."},
            "decision_id": {"type": "string", "description": "The UUID of the decision."},
        },
        "required": ["divergence_event_id", "decision_id"],
    },
}

UPDATE_PREMISE_STATUS_TOOL = {
    "name": "update_premise_status",
    "description": "Update the status of a specific premise after monitoring.",
    "input_schema": {
        "type": "object",
        "properties": {
            "premise_id": {"type": "string"},
            "status": {
                "type": "string",
                "enum": ["valid", "uncertain", "invalidated", "unverifiable"],
                "description": "New status based on monitoring findings.",
            },
            "confidence_score": {
                "type": "number",
                "minimum": 0.0,
                "maximum": 1.0,
                "description": "Agent confidence in the status classification.",
            },
            "invalidation_evidence": {
                "type": "string",
                "description": "If invalidated or uncertain, describe the evidence.",
            },
        },
        "required": ["premise_id", "status", "confidence_score"],
    },
}


class MonitorAgent(BaseAgent):
    agent_name = "monitor_agent"

    def __init__(
        self,
        db_session: AsyncSession,
        config: Settings = None,
        causal_trigger_callback=None,
    ):
        super().__init__(db_session, config)
        self._causal_trigger_callback = causal_trigger_callback
        self._divergence_event_id: Optional[str] = None

    def _register_tools(self) -> list[dict]:
        return [
            THINK_TOOL,
            SEARCH_TOOL,
            GET_DECISION_TOOL,
            GET_PRIOR_OBSERVATIONS_TOOL,
            SAVE_OBSERVATION_TOOL,
            COMPUTE_DIVERGENCE_TOOL,
            CREATE_DIVERGENCE_EVENT_TOOL,
            UPDATE_PREMISE_STATUS_TOOL,
            TRIGGER_CAUSAL_TOOL,
        ]

    async def _dispatch_tool(self, tool_name: str, tool_input: dict, decision_id: Optional[str] = None) -> Any:
        if tool_name == "think":
            return {"status": "thought_recorded"}

        if tool_name == "web_search":
            if self.config.DISABLE_ALL_WEB_MONITORING:
                return {"error": "Web monitoring is globally disabled."}
            results = await execute_web_search(
                query=tool_input["query"],
                max_results=tool_input.get("max_results", 5),
                search_depth=tool_input.get("search_depth", "basic"),
            )
            return [r.model_dump() for r in results]

        if tool_name == "get_decision_with_premises":
            return await db_get_decision(tool_input["decision_id"], self.db)

        if tool_name == "get_prior_observations":
            return await db_get_prior_observations(
                tool_input["premise_id"], tool_input.get("limit", 5), self.db
            )

        if tool_name == "save_observation":
            data = SaveObservationInput(**tool_input)
            return await db_save_observation(data, self.db)

        if tool_name == "compute_divergence_score":
            return await db_compute_divergence(tool_input["decision_id"], self.db)

        if tool_name == "create_divergence_event":
            data = CreateDivergenceEventInput(**tool_input)
            result = await db_create_divergence_event(data, self.db)
            self._divergence_event_id = result.get("divergence_event_id")
            return result

        if tool_name == "update_premise_status":
            return await self._update_premise_status(tool_input)

        if tool_name == "trigger_causal_agent":
            return await self._trigger_causal(
                tool_input["divergence_event_id"],
                tool_input["decision_id"],
            )

        return await super()._dispatch_tool(tool_name, tool_input, decision_id)

    async def _update_premise_status(self, tool_input: dict) -> dict:
        from sqlalchemy import select
        from decisionmesh.models.premise import PremiseORM

        result = await self.db.execute(
            select(PremiseORM).where(PremiseORM.id == tool_input["premise_id"])
        )
        premise = result.scalar_one_or_none()
        if not premise:
            return {"error": f"Premise {tool_input['premise_id']} not found"}

        premise.status = tool_input["status"]
        premise.confidence_score = tool_input.get("confidence_score", 1.0)
        premise.last_checked_at = datetime.utcnow()

        if tool_input["status"] == PremiseStatus.INVALIDATED.value:
            premise.invalidated_at = datetime.utcnow()
            premise.invalidation_evidence = tool_input.get("invalidation_evidence", "")

        await self.db.flush()
        return {"status": "updated", "premise_id": premise.id, "new_status": premise.status}

    async def _trigger_causal(self, divergence_event_id: str, decision_id: str) -> dict:
        """Async dispatch to Layer 3. Callback registered by orchestrator."""
        if self._causal_trigger_callback:
            try:
                await self._causal_trigger_callback(divergence_event_id, decision_id)
                return {"status": "causal_agent_triggered", "divergence_event_id": divergence_event_id}
            except Exception as e:
                logger.error(f"Failed to trigger causal agent: {e}")
                return {"status": "trigger_failed", "error": str(e)}
        return {"status": "no_callback_registered"}

    async def run_monitor(self, decision_id: str) -> dict:
        """
        Main entry point: run a monitoring cycle for a specific decision.
        Returns summary of what was found.
        """
        # Fetch decision first to build context
        decision_data = await db_get_decision(decision_id, self.db)
        if "error" in decision_data:
            return decision_data

        web_enabled = decision_data.get("web_monitoring_enabled", False)
        if self.config.DISABLE_ALL_WEB_MONITORING:
            web_enabled = False

        premises = decision_data.get("premises_detail", [])
        premises_summary = "\n".join(
            f"- [{p['id'][:8]}] {p['text']} (status: {p['status']}, weight: {p['weight']})"
            for p in premises
        )

        user_message = f"""Please monitor this decision's premises:

Decision ID: {decision_id}
Title: {decision_data.get('title', 'Unknown')}
Domain: {decision_data.get('domain', 'general')}
Web monitoring enabled: {web_enabled}

Premises to check:
{premises_summary}

Decision narrative:
{decision_data.get('full_narrative', '')[:2000]}

Steps:
1. Use the think tool to reason about what monitoring approach to use for each premise
2. For each premise:
   a. Think about what evidence would confirm or deny it
   b. If web_monitoring_enabled=True, use web_search to find relevant current information
   c. Reason from prior observations and the narrative if web monitoring is off
   d. Call update_premise_status with the new classification
   e. Call save_observation to record what you found
3. Call compute_divergence_score to get the overall drift score
4. If divergence_score > {self.config.DIVERGENCE_ALERT_THRESHOLD}, call create_divergence_event
5. If a divergence event was created, call trigger_causal_agent"""

        await self.run(
            system_prompt=MONITOR_SYSTEM_PROMPT,
            user_message=user_message,
            decision_id=decision_id,
        )

        # Return monitoring summary
        summary = await db_compute_divergence(decision_id, self.db)
        return {
            "decision_id": decision_id,
            "divergence_score": summary.get("divergence_score", 0.0),
            "divergence_event_created": self._divergence_event_id is not None,
            "divergence_event_id": self._divergence_event_id,
            "premise_details": summary.get("details", []),
        }
