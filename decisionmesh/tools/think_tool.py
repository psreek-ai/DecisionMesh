"""
DecisionMesh — Think Tool
The scratchpad reasoning tool. Always registered first for all agents.
Thoughts are logged to the audit trail but never shown to users.
"""
from pydantic import BaseModel, Field


class ThinkInput(BaseModel):
    thought: str = Field(
        description=(
            "Your internal reasoning, analysis, or scratchpad thinking. "
            "This is never shown to the user. Think step by step before "
            "making any classification or taking any consequential action."
        )
    )


THINK_TOOL: dict = {
    "name": "think",
    "description": (
        "Use this tool to think through complex problems before taking action. "
        "Your thoughts are logged but not shown to users. "
        "ALWAYS use this before making any classification or taking any "
        "consequential action. Reason through implications, edge cases, "
        "and epistemic uncertainty before committing to output."
    ),
    "input_schema": ThinkInput.model_json_schema(),
}
