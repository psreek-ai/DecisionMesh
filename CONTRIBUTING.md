# Contributing to DecisionMesh

Thank you for your interest in contributing. DecisionMesh is open-source infrastructure for human decision intelligence, and every contribution matters.

---

## Before You Start

Please read the [README](README.md) and [Philosophy](#philosophy) section first. DecisionMesh has hard constraints that all contributions must respect.

**Non-negotiable constraints:**
- Never add features that make decisions for users
- Never add engagement mechanics (streaks, push notifications, gamification)
- All agent reasoning must be logged and inspectable
- No telemetry or data exfiltration, ever
- System must work entirely offline (except explicitly opted-in web search)

If your contribution conflicts with these principles, it will not be merged regardless of technical quality.

---

## Development Setup

```bash
# Python 3.12+ required
git clone https://github.com/your-org/decisionmesh
cd decisionmesh

python -m venv .venv
source .venv/bin/activate  # or .venv\Scripts\activate on Windows

pip install -e ".[dev]"

# Set up pre-commit hooks
pip install pre-commit
pre-commit install

# Copy and configure environment
cp .env.example .env
# Add your ANTHROPIC_API_KEY to .env

# Initialize database
dm db migrate

# Run the test suite
pytest
```

---

## Code Style

We use [ruff](https://github.com/astral-sh/ruff) for linting and formatting.

```bash
# Lint
ruff check decisionmesh/ tests/

# Format
ruff format decisionmesh/ tests/

# Type check
mypy decisionmesh/ --ignore-missing-imports
```

All CI checks must pass before a PR is merged.

**Style principles:**
- Prefer explicit over implicit
- Prefer simple over clever
- Every public function and class needs a docstring
- Async all the way down (no `asyncio.run()` in library code)
- Pydantic v2 for all data boundaries

---

## Testing Requirements

**All new behavior must have tests.** The test suite must not make real Anthropic API calls.

### Mocking the Anthropic client

Use the `mock_anthropic_client` fixture from `conftest.py`:

```python
async def test_my_agent_does_something(async_session, mock_anthropic_client, monkeypatch):
    # Configure the mock to return controlled responses
    mock_anthropic_client.configure_responses([
        make_tool_use_response("think", {"thought": "I am thinking..."}),
        make_text_response("Here is my conclusion."),
    ])
    monkeypatch.setattr("anthropic.Anthropic", lambda **_: mock_anthropic_client)

    agent = MyAgent(db_session=async_session)
    result = await agent.run(...)
    assert "conclusion" in result
```

### Test categories

| Category | Location | What to test |
|---|---|---|
| Model validation | `tests/test_models/` | Pydantic schemas, ORM roundtrips, enum values |
| Agent behavior | `tests/test_agents/` | Tool dispatch, loop termination, DB side-effects |
| API endpoints | `tests/test_api/` | Status codes, response schemas, error handling |
| Scheduler | `tests/test_scheduler/` | Job registration, persistence, URL conversion |

### Coverage requirement

The CI requires **≥ 70% coverage**. New code should not drop coverage.

```bash
pytest --cov=decisionmesh --cov-report=term-missing
```

---

## Commit Message Conventions

We use [Conventional Commits](https://www.conventionalcommits.org/):

```
<type>(<scope>): <short description>

[optional body]

[optional footer]
```

Types:
- `feat` — new feature
- `fix` — bug fix
- `docs` — documentation only
- `test` — adding or improving tests
- `refactor` — code change without behavior change
- `chore` — tooling, dependencies, CI

Examples:
```
feat(monitor): add confidence decay for stale observations
fix(cli): handle missing decision ID gracefully in dm show
docs(readme): add PostgreSQL setup instructions
test(capture): add tests for premise testability classification
```

---

## RFC Process

For significant changes — new agents, new data models, new CLI commands, changes to the agent loop — please write an RFC before implementing.

1. Open an issue titled `RFC: <what you're proposing>`
2. Describe: the problem, your proposed solution, alternatives you considered, trade-offs
3. Wait for discussion (typically 1 week for minor, 2 weeks for significant changes)
4. Once consensus is reached, implement and reference the RFC in your PR

This prevents wasted effort on large changes that might not be accepted.

---

## Adding a New Agent

1. Create `decisionmesh/agents/my_agent.py`
2. Inherit from `BaseAgent`
3. Override `_register_tools()` and `_dispatch_tool()`
4. Register all tool schemas with Pydantic-validated inputs
5. Add `agent_name = "my_agent"` class variable (used in audit logs)
6. Write tests in `tests/test_agents/test_my_agent.py`
7. Wire into `DecisionMeshOrchestrator` if it needs to be called from the main flow

---

## Adding a New CLI Command

```python
@app.command()
def my_command(arg: str = typer.Argument(...)):
    """One-line description shown in dm --help."""
    # Use Rich for all output
    # Use run_async() for any async work
    # Always call _ensure_db() at the start
```

Add a corresponding test using Typer's `CliRunner`.

---

## Submitting a Pull Request

1. Fork the repository
2. Create a branch: `git checkout -b feat/my-feature`
3. Write tests first where possible
4. Implement the change
5. Ensure all checks pass: `pytest && ruff check . && mypy decisionmesh/`
6. Update `CHANGELOG.md` under `[Unreleased]`
7. Submit a PR using the PR template
8. Respond to review feedback

---

## Philosophy

> "Decisions are not moments — they are processes."

Every feature contribution should ask: *does this help users understand their decisions better, or does it try to make decisions for them?* The former is always welcome. The latter is never acceptable.

We are building infrastructure for human judgment, not a replacement for it.
