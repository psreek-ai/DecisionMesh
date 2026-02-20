# CLAUDE.md — DecisionMesh: Open Decision Intelligence Infrastructure

## PROJECT MISSION
You are building **DecisionMesh**, an open-source agentic system that treats decisions as living, temporal objects. Every significant decision a human makes has premises, predicted outcomes, and triggering conditions. This system autonomously monitors whether those premises remain true over time, detects drift, simulates counterfactuals, and closes the feedback loop back to the decision-maker — with zero ongoing user prompting required.

This is NOT a task manager, NOT a journal, NOT a chatbot. It is a **Temporal Decision Intelligence Engine** — the first of its kind.

License: AGPL-3.0. Build for global accessibility. Self-hosted first. No telemetry. No data exfiltration. Privacy is non-negotiable.

---

## ABSOLUTE CONSTRAINTS
- Never make a decision FOR the user. Surface divergence and counterfactuals only.
- Never optimize for engagement. No nudges, no streaks, no dark patterns.
- All agent reasoning must be logged transparently in the database.
- Every agent action must be reversible or inspectable.
- The system must run entirely offline except when the Condition Monitor
  explicitly performs web searches (which the user must opt into per decision).
- Use the "think" tool pattern for all multi-step agent reasoning chains.
- Use Pydantic v2 for ALL data models and tool input/output validation.
- Use SQLite (via SQLAlchemy async) as the default database — zero infra deps.
- Provide a PostgreSQL+pgvector adapter as an optional drop-in for production.

---

## TECH STACK
- **Python 3.12+**
- **Anthropic SDK** (`anthropic>=0.40.0`) — claude-3-7-sonnet-20250219 as default
- **LangGraph 0.2+** — for agent state machines and checkpointing
- **SQLAlchemy 2.0 async** + **aiosqlite** — persistence layer
- **Pydantic v2** — all data models and validation
- **APScheduler 3.10+** — background scheduler for monitoring agents
- **Tavily Python client** — web search for condition monitoring (optional)
- **Sentence-transformers** (`all-MiniLM-L6-v2`) — local embeddings, no API call
- **FastAPI + Uvicorn** — REST API layer
- **Rich** + **Typer** — beautiful CLI interface
- **Pytest + pytest-asyncio** — full test suite
- **Alembic** — database migrations

Do NOT use LangChain. Use LangGraph directly with the Anthropic SDK.
Do NOT use OpenAI. This system is built on Anthropic's API exclusively.

---

## REPOSITORY STRUCTURE

```
decisionmesh/
├── CLAUDE.md
├── README.md
├── LICENSE
├── pyproject.toml
├── .env.example
├── docker-compose.yml
├── Dockerfile
├── alembic.ini
├── alembic/
│   └── versions/
├── decisionmesh/
│   ├── __init__.py
│   ├── config.py
│   ├── database.py
│   ├── models/
│   │   ├── __init__.py
│   │   ├── decision.py
│   │   ├── premise.py
│   │   ├── condition.py
│   │   ├── observation.py
│   │   ├── divergence.py
│   │   ├── counterfactual.py
│   │   └── agent_log.py
│   ├── agents/
│   │   ├── __init__.py
│   │   ├── base.py
│   │   ├── capture_agent.py
│   │   ├── monitor_agent.py
│   │   ├── causal_agent.py
│   │   ├── revision_agent.py
│   │   └── orchestrator.py
│   ├── tools/
│   │   ├── __init__.py
│   │   ├── think_tool.py
│   │   ├── search_tool.py
│   │   ├── db_tools.py
│   │   ├── embedding_tool.py
│   │   └── similarity_tool.py
│   ├── scheduler/
│   │   ├── __init__.py
│   │   └── monitor_scheduler.py
│   ├── api/
│   │   ├── __init__.py
│   │   ├── main.py
│   │   ├── routes/
│   │   │   ├── decisions.py
│   │   │   ├── agents.py
│   │   │   ├── reports.py
│   │   │   └── health.py
│   │   └── schemas/
│   │       ├── decision_schemas.py
│   │       └── report_schemas.py
│   └── cli/
│       ├── __init__.py
│       └── main.py
└── tests/
    ├── conftest.py
    ├── test_models/
    ├── test_agents/
    ├── test_api/
    └── test_scheduler/
```

---

## DATA MODELS

### `models/decision.py`

```python
class DecisionStatus(str, Enum):
    ACTIVE = "active"
    DRIFTED = "drifted"
    REVISED = "revised"
    CLOSED = "closed"
    PAUSED = "paused"

class DecisionDNA(BaseModel):
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    title: str
    full_narrative: str
    premises: list[str]
    predicted_outcomes: list[str]
    triggering_conditions: list[str]
    decision_date: datetime
    review_interval_days: int = 30
    status: DecisionStatus = DecisionStatus.ACTIVE
    domain: str = "general"
    tags: list[str] = Field(default_factory=list)
    embedding: Optional[list[float]] = None
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)
    web_monitoring_enabled: bool = False
```

### Divergence Score Formula

```
divergence_score = Σ (premise_weight_i × invalidation_score_i) / Σ premise_weight_i

where:
  invalidation_score: valid=0.0, uncertain=0.4, invalidated=1.0
  premise_weight: defaults to 1.0, user can manually set 0.1-3.0
```

---

## BUILD ORDER

1. pyproject.toml, .env.example, config.py, database.py
2. All Pydantic models (models/) + SQLAlchemy ORM + Alembic migrations
3. All tool definitions (tools/)
4. BaseAgent with the core agentic while-loop
5. CaptureAgent
6. MonitorAgent
7. CausalAgent (extended thinking, budget=8000)
8. RevisionAdvisor
9. Orchestrator
10. APScheduler integration
11. FastAPI routes
12. CLI (typer)
13. Dockerfile + docker-compose.yml
14. Full README.md
15. Full test suite — achieve >80% coverage
