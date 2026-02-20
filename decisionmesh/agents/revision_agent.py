"""
DecisionMesh — Layer 4: Decision Revision Advisor
The ONLY agent that interacts synchronously with the user.
Presents divergence evidence, walks through counterfactuals,
and helps the user choose a revision path — without deciding for them.
"""
import json
import logging
from typing import Any, Optional

from sqlalchemy.ext.asyncio import AsyncSession

from decisionmesh.agents.base import BaseAgent
from decisionmesh.config import Settings, settings
from decisionmesh.tools.think_tool import THINK_TOOL
from decisionmesh.tools.db_tools import (
    UPDATE_DECISION_DNA_TOOL, CLOSE_DECISION_TOOL, PAUSE_MONITORING_TOOL,
    UpdateDecisionDNAInput, CloseDecisionInput, PauseMonitoringInput,
)

logger = logging.getLogger(__name__)


REVISION_SYSTEM_PROMPT = """You are the Decision Revision Advisor for DecisionMesh.
The user has a decision that has drifted from its original premises.

You are NOT here to tell them what to do.

Your role:
1. Present the divergence evidence clearly and objectively
2. Walk through the counterfactual analysis produced by the Causal Agent
3. Offer a structured menu of revision options:
   a. Update premise (it was wrong from the start)
   b. Acknowledge premise drift (world changed, not the original logic)
   c. Revise predicted outcomes (adjust expectations given new reality)
   d. Add new triggering conditions (expand monitoring scope)
   e. Close the decision (it has run its course)
   f. Pause monitoring (user is aware, will review manually)
4. Execute whichever option the user chooses using the available tools
5. Update the DecisionDNA accordingly

Tone: calm, non-judgmental, analytical. Like a trusted advisor who has been watching events unfold.
- Never say "I told you so."
- Never imply a decision was wrong — only that circumstances changed.
- Always present options as choices, not recommendations.
- Use plain language. Avoid jargon.

Begin by presenting the situation, then ask what the user would like to do."""


class RevisionAgent(BaseAgent):
    agent_name = "revision_agent"

    def __init__(self, db_session: AsyncSession, config: Settings = None):
        super().__init__(db_session, config)
        self._revision_action_taken: Optional[str] = None

    def _register_tools(self) -> list[dict]:
        return [
            THINK_TOOL,
            UPDATE_DECISION_DNA_TOOL,
            CLOSE_DECISION_TOOL,
            PAUSE_MONITORING_TOOL,
        ]

    async def _dispatch_tool(self, tool_name: str, tool_input: dict, decision_id: Optional[str] = None) -> Any:
        if tool_name == "think":
            return {"status": "thought_recorded"}

        if tool_name == "update_decision_dna":
            return await self._update_dna(UpdateDecisionDNAInput(**tool_input))

        if tool_name == "close_decision":
            return await self._close_decision(CloseDecisionInput(**tool_input))

        if tool_name == "pause_monitoring":
            return await self._pause_monitoring(PauseMonitoringInput(**tool_input))

        return await super()._dispatch_tool(tool_name, tool_input, decision_id)

    async def _update_dna(self, data: UpdateDecisionDNAInput) -> dict:
        from sqlalchemy import select
        from decisionmesh.models.decision import DecisionORM
        from datetime import datetime

        result = await self.db.execute(
            select(DecisionORM).where(DecisionORM.id == data.decision_id)
        )
        decision = result.scalar_one_or_none()
        if not decision:
            return {"error": f"Decision {data.decision_id} not found"}

        try:
            if data.field in ("premises", "predicted_outcomes", "triggering_conditions", "tags"):
                parsed = json.loads(data.value)
                setattr(decision, data.field, parsed)
            elif data.field in ("status", "domain", "title"):
                setattr(decision, data.field, data.value)
            else:
                return {"error": f"Field '{data.field}' is not updatable"}

            decision.updated_at = datetime.utcnow()
            await self.db.flush()
            self._revision_action_taken = f"updated_{data.field}"
            return {"status": "updated", "field": data.field}
        except Exception as e:
            return {"error": str(e)}

    async def _close_decision(self, data: CloseDecisionInput) -> dict:
        from sqlalchemy import select
        from decisionmesh.models.decision import DecisionORM, DecisionStatus
        from datetime import datetime

        result = await self.db.execute(
            select(DecisionORM).where(DecisionORM.id == data.decision_id)
        )
        decision = result.scalar_one_or_none()
        if not decision:
            return {"error": "Decision not found"}

        decision.status = DecisionStatus.CLOSED.value
        decision.updated_at = datetime.utcnow()
        await self.db.flush()
        self._revision_action_taken = "closed"
        return {"status": "closed", "reason": data.reason}

    async def _pause_monitoring(self, data: PauseMonitoringInput) -> dict:
        from sqlalchemy import select
        from decisionmesh.models.decision import DecisionORM, DecisionStatus
        from datetime import datetime

        result = await self.db.execute(
            select(DecisionORM).where(DecisionORM.id == data.decision_id)
        )
        decision = result.scalar_one_or_none()
        if not decision:
            return {"error": "Decision not found"}

        if data.pause:
            decision.status = DecisionStatus.PAUSED.value
            self._revision_action_taken = "paused"
        else:
            decision.status = DecisionStatus.ACTIVE.value
            self._revision_action_taken = "resumed"

        decision.updated_at = datetime.utcnow()
        await self.db.flush()
        return {"status": "paused" if data.pause else "resumed"}

    async def present_divergence(
        self,
        decision_id: str,
        divergence_event_id: str,
        user_choice: Optional[str] = None,
    ) -> str:
        """
        Main entry point: present a divergence to the user and handle their response.
        If user_choice is None, presents the situation and options.
        If user_choice is provided, executes that revision action.
        """
        # Fetch full context
        from sqlalchemy import select
        from decisionmesh.models.decision import DecisionORM
        from decisionmesh.models.divergence import DivergenceEventORM
        from decisionmesh.models.counterfactual import CounterfactualORM

        decision_result = await self.db.execute(
            select(DecisionORM).where(DecisionORM.id == decision_id)
        )
        decision = decision_result.scalar_one_or_none()

        divergence_result = await self.db.execute(
            select(DivergenceEventORM).where(DivergenceEventORM.id == divergence_event_id)
        )
        divergence = divergence_result.scalar_one_or_none()

        cf_result = await self.db.execute(
            select(CounterfactualORM).where(CounterfactualORM.divergence_event_id == divergence_event_id)
        )
        counterfactual = cf_result.scalars().first()

        context = f"""Decision: {decision.title if decision else 'Unknown'}
Decision ID: {decision_id}
Divergence Severity: {divergence.severity if divergence else 'unknown'}
Divergence Score: {divergence.divergence_score if divergence else 0.0:.2%}
Summary: {divergence.summary if divergence else 'No summary available'}

Evidence:
{chr(10).join(f'- {e}' for e in (divergence.evidence if divergence else []))}

Counterfactual Analysis:
{counterfactual.alternative_scenario if counterfactual else 'Not yet generated'}

Projected Outcome Delta:
{counterfactual.projected_outcome_delta if counterfactual else 'Not yet generated'}"""

        if user_choice:
            user_message = f"""{context}

The user has chosen to: {user_choice}
Decision ID: {decision_id}
Divergence Event ID: {divergence_event_id}

Please execute this revision action using the appropriate tool, then confirm what was changed."""
        else:
            user_message = f"""{context}

Please present this situation to the user in clear, calm language and offer the revision options.
Do not make a recommendation — present all options equally.
The user will respond with their choice in the next message."""

        return await self.run(
            system_prompt=REVISION_SYSTEM_PROMPT,
            user_message=user_message,
            decision_id=decision_id,
        )

    async def acknowledge_divergence(self, divergence_event_id: str) -> None:
        """Mark a divergence event as acknowledged by the user."""
        from sqlalchemy import select
        from decisionmesh.models.divergence import DivergenceEventORM
        from datetime import datetime

        result = await self.db.execute(
            select(DivergenceEventORM).where(DivergenceEventORM.id == divergence_event_id)
        )
        event = result.scalar_one_or_none()
        if event:
            event.acknowledged_by_user = True
            event.acknowledged_at = datetime.utcnow()
            await self.db.flush()
