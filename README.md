# DecisionMesh

**Temporal Decision Intelligence Engine** — the first open-source system that treats decisions as living, temporal objects.

Every significant decision you make rests on a set of premises about the world. DecisionMesh autonomously monitors whether those premises remain true over time, detects drift, simulates counterfactuals, and closes the feedback loop — with zero ongoing prompting required from you.

---

## Architecture

```
┌──────────────────────────────────────────────────────────────────┐
│                     DecisionMesh System                          │
│                                                                  │
│  ┌──────────────┐     ┌──────────────┐     ┌──────────────┐     │
│  │   Layer 1    │     │   Layer 2    │     │   Layer 3    │     │
│  │   Capture    │────▶│   Monitor    │────▶│   Causal     │     │
│  │   Agent      │     │   Agent      │     │   Inference  │     │
│  │              │     │  (Scheduled) │     │  (Extended   │     │
│  │ Extracts DNA │     │ Checks       │     │   Thinking)  │     │
│  │ from text    │     │ premises     │     │ Counterfact. │     │
│  └──────────────┘     └──────────────┘     └──────┬───────┘     │
│         │                    │                     │             │
│         ▼                    ▼                     ▼             │
│  ┌──────────────────────────────────────────────────────────┐    │
│  │              SQLite Database (zero infra)                │    │
│  │  decisions │ premises │ observations │ divergence_events │    │
│  │  counterfactuals │ agent_logs │ scheduler_jobs           │    │
│  └──────────────────────────────────────────────────────────┘    │
│         │                    │                     │             │
│         ▼                    ▼                     ▼             │
│  ┌──────────────┐     ┌──────────────┐     ┌──────────────┐     │
│  │   Layer 4    │     │  APScheduler │     │  Local LLM   │     │
│  │   Revision   │     │  (Persistent │     │  Embeddings  │     │
│  │   Advisor    │     │   Job Store) │     │ all-MiniLM   │     │
│  │ Non-directive│     │              │     │  (offline)   │     │
│  │ conversatio. │     │              │     │              │     │
│  └──────────────┘     └──────────────┘     └──────────────┘     │
│                                                                  │
│  ┌────────────────────┐     ┌─────────────────────────────┐     │
│  │    REST API        │     │    CLI (dm)                  │     │
│  │    FastAPI         │     │    Typer + Rich              │     │
│  └────────────────────┘     └─────────────────────────────┘     │
└──────────────────────────────────────────────────────────────────┘
```

---

## Quick Start

```bash
# 1. Install
pip install -e .

# 2. Configure
cp .env.example .env
# Edit .env — add your ANTHROPIC_API_KEY

# 3. Initialize the database
dm db migrate

# 4. Capture your first decision
dm add
```

That's it. DecisionMesh will begin monitoring your decision's premises on the schedule you configure (default: every 30 days).

---

## Full CLI Reference

```bash
# Capture a new decision (interactive wizard)
dm add
dm add --narrative "I decided to..." --web   # with web monitoring enabled

# List decisions
dm list
dm list --status active
dm list --domain business

# View a decision's full DNA
dm show <decision-id>

# Check your inbox (unacknowledged divergence alerts)
dm inbox

# Review a divergence event interactively
dm review <divergence-id>

# Manually trigger a monitoring run
dm check <decision-id>

# View the full agent reasoning audit trail
dm history <decision-id>

# View a timeline of a decision's life
dm timeline <decision-id>

# Export all decisions to JSON
dm export --output decisions_backup.json

# Pause/resume monitoring
dm pause <decision-id>
dm resume <decision-id>

# Close a decision
dm close <decision-id>

# Database
dm db migrate
dm db reset

# Scheduler
dm scheduler status
dm scheduler logs
```

---

## Configuration Reference

Copy `.env.example` to `.env` and configure:

| Variable | Default | Description |
|---|---|---|
| `ANTHROPIC_API_KEY` | *(required)* | Your Anthropic API key |
| `TAVILY_API_KEY` | *(optional)* | For web monitoring (opt-in per decision) |
| `DATABASE_URL` | `sqlite+aiosqlite:///./decisionmesh.db` | SQLite (default) or PostgreSQL |
| `DEFAULT_MODEL` | `claude-3-7-sonnet-20250219` | Claude model to use |
| `MAX_AGENT_ITERATIONS` | `10` | Max tool-use iterations per agent run |
| `CAUSAL_THINKING_BUDGET` | `8000` | Extended thinking token budget for Causal Agent |
| `DIVERGENCE_ALERT_THRESHOLD` | `0.3` | Drift score that triggers an alert (30%) |
| `DIVERGENCE_CRITICAL_THRESHOLD` | `0.7` | Drift score considered critical (70%) |
| `DEFAULT_REVIEW_INTERVAL_DAYS` | `30` | How often to check each decision |
| `SCHEDULER_TIMEZONE` | `UTC` | Timezone for scheduled jobs |
| `EMBEDDING_MODEL` | `all-MiniLM-L6-v2` | Local embedding model (never calls API) |
| `DISABLE_ALL_WEB_MONITORING` | `false` | Global kill-switch for web search |

---

## Self-Hosting Guide

### Docker (recommended)

```bash
git clone https://github.com/your-org/decisionmesh
cd decisionmesh
cp .env.example .env
# Edit .env with your API key

docker-compose up -d

# Access the API
curl http://localhost:8000/health/

# Use the CLI against the API
dm list
```

### Bare Metal

```bash
# Python 3.12+ required
git clone https://github.com/your-org/decisionmesh
cd decisionmesh

python -m venv .venv
source .venv/bin/activate

pip install -e .

cp .env.example .env
# Edit .env

# Initialize DB
dm db migrate

# Start API server (optional)
uvicorn decisionmesh.api.main:app --reload

# Or just use the CLI
dm add
```

### PostgreSQL + pgvector (production)

```bash
# Start postgres with pgvector
docker run -d --name pgvector \
  -e POSTGRES_DB=decisionmesh \
  -e POSTGRES_PASSWORD=yourpassword \
  -p 5432:5432 \
  ankane/pgvector

# Update .env
DATABASE_URL=postgresql+asyncpg://postgres:yourpassword@localhost:5432/decisionmesh

dm db migrate
```

---

## Philosophy

**Decisions are not moments — they are processes.**

When you make a significant decision, you are not just choosing an action. You are making a bet on how the world works. That bet rests on premises: beliefs about facts, trends, and conditions that you expect to remain stable.

The problem is that the world doesn't stay still. Markets shift. Circumstances change. The premises you relied on in January may be wrong by June. But most tools for thinking about decisions treat them as past events — journal entries, logged in a notebook, filed away.

DecisionMesh treats decisions differently: as living objects with premises that can be monitored, outcomes that can be tracked, and causal chains that can be simulated. When reality diverges from your premises, the system surfaces this — not to tell you what to do, but to give you the information you need to decide for yourself.

This is the feedback loop that most decision-making lacks: **the connection between what you believed when you decided, and what turned out to be true.**

We do not optimize for engagement. We do not push notifications, create streaks, or nudge behavior. The system exists to surface information and then get out of your way.

---

## Privacy Guarantees

- **No telemetry.** No data ever leaves your machine except for two explicit, opt-in cases:
  1. API calls to Anthropic for agent reasoning (required)
  2. Tavily web searches for premise monitoring (opt-in per decision, can be disabled globally)
- **Local embeddings.** All vector embeddings use `all-MiniLM-L6-v2` running entirely on your machine.
- **Local database.** SQLite by default — your decisions live in a single file on your disk.
- **Inspectable.** Every agent reasoning step is logged in the database. Nothing happens in a black box.
- **Reversible.** Every agent action can be undone or inspected. No silent mutations.

---

## Contributing

DecisionMesh is open-source infrastructure. Contributions are welcome.

### Process

1. Open an issue describing the change you want to make
2. For significant changes, write an RFC (Request for Comment) in the issue — describe the problem, your proposed solution, and alternatives you considered
3. Fork the repository and create a branch: `git checkout -b feature/your-feature`
4. Write tests before writing code where possible
5. Ensure the test suite passes: `pytest --cov=decisionmesh`
6. Submit a pull request referencing the issue

### Development Setup

```bash
pip install -e ".[dev]"
pytest
```

### Code Standards

- Python 3.12+
- Pydantic v2 for all data models
- Async SQLAlchemy 2.0 for all DB access
- All agent tool inputs/outputs must be Pydantic-validated
- Every new agent behavior must have a corresponding test with a mocked Anthropic client
- No telemetry, no tracking, no engagement optimization

---

## License

AGPL-3.0 — see [LICENSE](LICENSE).

This means: you can use, modify, and distribute DecisionMesh freely. If you modify it and offer it as a service to others, you must release your modifications under the same license.
