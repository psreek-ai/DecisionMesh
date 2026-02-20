"""
DecisionMesh — Layer 1: Decision Capture Agent
Takes a user's free-form narrative and extracts structured Decision DNA.
Epistemic approach: extract only what the user actually stated or clearly implied.
"""
import json
import logging
from typing import Any, Optional

from pydantic import BaseModel, Field
from sqlalchemy.ext.asyncio import AsyncSession

from decisionmesh.agents.base import BaseAgent
from decisionmesh.config import Settings, settings
from decisionmesh.models.decision import DecisionDNA, DecisionStatus
from decisionmesh.models.premise import Premise, PremiseStatus, PremiseTestability
from decisionmesh.tools.think_tool import THINK_TOOL

logger = logging.getLogger(__name__)


CAPTURE_SYSTEM_PROMPT = """You are the Decision Capture Agent for DecisionMesh.
Your job is to extract the complete "Decision DNA" from a user's narrative about a decision they've made.

You must be epistemically careful:
- Extract only what the user actually stated or clearly implied
- Never invent premises or outcomes not present in the narrative
- Ask clarifying questions (via the ask_clarification tool) if critical information is ambiguous
- Flag premises that are testable vs. untestable (philosophical beliefs cannot be monitored)

Use the think tool to reason through the extraction before producing output.
Use the extract_decision_dna tool to return structured output.
Use the check_premise_testability tool to classify each premise.

IMPORTANT: If web_monitoring_enabled=False (the default), flag any premises that would require
web search as requiring_web_monitoring=True and note this in your extraction.

Always use the think tool first to reason through what you're extracting and why."""


# ── Additional tool schemas for CaptureAgent ─────────────────────────────────

class ExtractDecisionDNAInput(BaseModel):
    title: str = Field(description="Short human-readable label for this decision (< 100 chars).")
    full_narrative: str = Field(description="The user's full narrative, preserved verbatim.")
    premises: list[str] = Field(description="Explicit beliefs the decision relies on. Extracted directly from narrative.")
    predicted_outcomes: list[str] = Field(description="What success looks like, as stated by the user.")
    triggering_conditions: list[str] = Field(description="Conditions that would require this decision to be revised.")
    decision_date: str = Field(description="ISO 8601 date string of when the decision was made.")
    domain: str = Field(description="Domain: business|health|finance|personal|policy|relationship|other")
    tags: list[str] = Field(default_factory=list, description="Short descriptive tags.")
    web_monitoring_needed: bool = Field(default=False, description="True if any premises require external data to monitor.")
    requires_clarification: Optional[str] = Field(default=None, description="If set, a question that needs answering before extraction is complete.")


class CheckPremiseTestabilityInput(BaseModel):
    premise: str = Field(description="The premise text to classify.")


class AskClarificationInput(BaseModel):
    question: str = Field(description="The clarifying question to ask the user.")
    why_needed: str = Field(description="Why this information is needed for the extraction.")


EXTRACT_DNA_TOOL = {
    "name": "extract_decision_dna",
    "description": (
        "Return the fully structured Decision DNA extracted from the user's narrative. "
        "Call this AFTER using the think tool to reason through the extraction. "
        "Be precise: only include premises explicitly stated or clearly implied."
    ),
    "input_schema": ExtractDecisionDNAInput.model_json_schema(),
}

CHECK_TESTABILITY_TOOL = {
    "name": "check_premise_testability",
    "description": (
        "Classify a premise as empirically_testable, measurable, or subjective_untestable. "
        "Empirically testable: can be verified via data or observation. "
        "Measurable: has a quantitative indicator. "
        "Subjective/untestable: philosophical, personal belief, or unverifiable claim."
    ),
    "input_schema": CheckPremiseTestabilityInput.model_json_schema(),
}

ASK_CLARIFICATION_TOOL = {
    "name": "ask_clarification",
    "description": (
        "Ask the user a clarifying question when critical information is missing or ambiguous. "
        "Calling this tool will pause the extraction and return the question to the user."
    ),
    "input_schema": AskClarificationInput.model_json_schema(),
}


class CaptureResult(BaseModel):
    """Result of the Capture Agent — a fully populated DecisionDNA."""
    decision: DecisionDNA
    premises_detail: list[Premise]
    clarification_needed: Optional[str] = None


class CaptureAgent(BaseAgent):
    agent_name = "capture_agent"

    def __init__(self, db_session: AsyncSession, config: Settings = None):
        super().__init__(db_session, config)
        self._extracted_dna: Optional[ExtractDecisionDNAInput] = None
        self._clarification_question: Optional[str] = None
        self._premise_classifications: dict[str, str] = {}

    def _register_tools(self) -> list[dict]:
        return [THINK_TOOL, EXTRACT_DNA_TOOL, CHECK_TESTABILITY_TOOL, ASK_CLARIFICATION_TOOL]

    async def _dispatch_tool(self, tool_name: str, tool_input: dict, decision_id: Optional[str] = None) -> Any:
        if tool_name == "think":
            return {"status": "thought_recorded"}

        if tool_name == "extract_decision_dna":
            validated = ExtractDecisionDNAInput(**tool_input)
            self._extracted_dna = validated
            return {"status": "extraction_recorded", "title": validated.title}

        if tool_name == "check_premise_testability":
            validated = CheckPremiseTestabilityInput(**tool_input)
            classification = self._classify_testability(validated.premise)
            self._premise_classifications[validated.premise] = classification
            return {"premise": validated.premise[:80], "testability": classification}

        if tool_name == "ask_clarification":
            validated = AskClarificationInput(**tool_input)
            self._clarification_question = validated.question
            return {"status": "clarification_queued", "question": validated.question}

        return await super()._dispatch_tool(tool_name, tool_input, decision_id)

    def _classify_testability(self, premise: str) -> str:
        """Simple heuristic classification — agent provides richer reasoning."""
        measurable_signals = ["%", "rate", "growth", "cost", "price", "metric", "number", "amount"]
        untestable_signals = ["believe", "feel", "prefer", "love", "trust", "hope", "spiritual"]

        lower = premise.lower()
        if any(s in lower for s in untestable_signals):
            return PremiseTestability.SUBJECTIVE_UNTESTABLE.value
        if any(s in lower for s in measurable_signals):
            return PremiseTestability.MEASURABLE.value
        return PremiseTestability.EMPIRICALLY_TESTABLE.value

    async def capture(self, narrative: str) -> CaptureResult:
        """
        Main entry point: extract Decision DNA from a free-form narrative.
        Returns a CaptureResult with the full DecisionDNA and premise details.
        """
        user_message = f"""Please extract the Decision DNA from this decision narrative:

---
{narrative}
---

Start by using the think tool to reason through:
1. What was actually decided?
2. What premises (beliefs) does this decision rely on?
3. What outcomes did the person predict?
4. What conditions would make them revise the decision?
5. When was the decision made?
6. Which premises require real-world monitoring vs. are internal/subjective?

Then check testability for each premise, and finally call extract_decision_dna with your structured result."""

        await self.run(
            system_prompt=CAPTURE_SYSTEM_PROMPT,
            user_message=user_message,
        )

        if self._clarification_question:
            # Return partial result with clarification needed
            return CaptureResult(
                decision=DecisionDNA(
                    title="[Pending clarification]",
                    full_narrative=narrative,
                    premises=[],
                    predicted_outcomes=[],
                    triggering_conditions=[],
                    decision_date=__import__("datetime").datetime.utcnow(),
                ),
                premises_detail=[],
                clarification_needed=self._clarification_question,
            )

        if not self._extracted_dna:
            raise ValueError("CaptureAgent failed to extract Decision DNA from the narrative.")

        # Build DecisionDNA
        from datetime import datetime
        try:
            decision_date = datetime.fromisoformat(self._extracted_dna.decision_date.replace("Z", "+00:00"))
        except Exception:
            decision_date = datetime.utcnow()

        # Generate local embedding
        from decisionmesh.tools.embedding_tool import embed_text
        embedding = embed_text(self._extracted_dna.full_narrative)

        decision = DecisionDNA(
            title=self._extracted_dna.title,
            full_narrative=self._extracted_dna.full_narrative,
            premises=self._extracted_dna.premises,
            predicted_outcomes=self._extracted_dna.predicted_outcomes,
            triggering_conditions=self._extracted_dna.triggering_conditions,
            decision_date=decision_date,
            status=DecisionStatus.ACTIVE,
            domain=self._extracted_dna.domain,
            tags=self._extracted_dna.tags,
            embedding=embedding,
            web_monitoring_enabled=False,  # Always opt-in explicitly
        )

        # Build Premise objects with testability info
        premises_detail = []
        for p_text in self._extracted_dna.premises:
            testability_val = self._premise_classifications.get(p_text, PremiseTestability.EMPIRICALLY_TESTABLE.value)
            requires_web = (
                self._extracted_dna.web_monitoring_needed
                and testability_val != PremiseTestability.SUBJECTIVE_UNTESTABLE.value
            )
            premise = Premise(
                decision_id=decision.id,
                text=p_text,
                status=PremiseStatus.VALID,
                testability=PremiseTestability(testability_val),
                requires_web_monitoring=requires_web,
                embedding=embed_text(p_text),
            )
            premises_detail.append(premise)

        return CaptureResult(
            decision=decision,
            premises_detail=premises_detail,
        )

    async def save_to_db(self, result: CaptureResult) -> str:
        """Persist the captured decision and premises to the database. Returns decision ID."""
        from decisionmesh.models.decision import DecisionORM
        from decisionmesh.models.premise import PremiseORM

        decision = result.decision
        decision_orm = DecisionORM(
            id=decision.id,
            title=decision.title,
            full_narrative=decision.full_narrative,
            premises=decision.premises,
            predicted_outcomes=decision.predicted_outcomes,
            triggering_conditions=decision.triggering_conditions,
            decision_date=decision.decision_date,
            review_interval_days=decision.review_interval_days,
            status=decision.status.value,
            domain=decision.domain,
            tags=decision.tags,
            embedding=decision.embedding,
            web_monitoring_enabled=decision.web_monitoring_enabled,
        )
        self.db.add(decision_orm)

        for premise in result.premises_detail:
            premise_orm = PremiseORM(
                id=premise.id,
                decision_id=decision.id,
                text=premise.text,
                confidence_score=premise.confidence_score,
                status=premise.status.value,
                testability=premise.testability.value,
                weight=premise.weight,
                embedding=premise.embedding,
                requires_web_monitoring=premise.requires_web_monitoring,
            )
            self.db.add(premise_orm)

        await self.db.flush()
        return decision.id
