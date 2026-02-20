# Security Policy

## Supported Versions

| Version | Supported |
|---------|-----------|
| 0.1.x   | ✅ Yes    |

## Reporting a Vulnerability

**Please do not report security vulnerabilities via public GitHub issues.**

If you discover a security vulnerability, please report it privately:

1. Go to the repository's **Security** tab
2. Click **"Report a vulnerability"**
3. Fill in the details

We will acknowledge receipt within **48 hours** and aim to provide a fix within **7 days** for critical issues.

Please include:
- Description of the vulnerability
- Steps to reproduce
- Potential impact
- Any suggested mitigations

---

## Security Model

DecisionMesh is designed with security and privacy as first principles.

### API Keys

- `ANTHROPIC_API_KEY` and `TAVILY_API_KEY` are read from environment variables or `.env` file
- Keys are **never** logged, stored in the database, or included in exports
- The `.gitignore` explicitly excludes `.env` files
- Never hardcode keys in code — the CI will reject PRs containing API key patterns

### Data at Rest

- Default storage is a local SQLite file on the user's machine
- The database contains decision narratives, premises, and agent reasoning
- There is no encryption at rest by default — if this is required, use filesystem-level encryption (e.g., LUKS, FileVault) or the PostgreSQL adapter with encrypted storage
- `dm export` strips embedding vectors but preserves all decision content — treat exports as sensitive

### Data in Transit

- Agent API calls go to **Anthropic's API** — your decision content is sent to Anthropic for processing. Review [Anthropic's privacy policy](https://www.anthropic.com/privacy) and [data usage policy](https://www.anthropic.com/legal/privacy)
- Web search queries go to **Tavily** — only when `web_monitoring_enabled=True` per decision. Review [Tavily's privacy policy](https://tavily.com/privacy)
- All other network traffic is blocked by design

### No Telemetry

DecisionMesh collects zero telemetry. No usage statistics, error reports, or analytics are sent anywhere. This is a hard constraint of the project and will never change.

### Dependency Security

We use `bandit` for static security analysis in CI. Dependencies are pinned in `pyproject.toml` and should be kept up to date. We recommend running:

```bash
pip install pip-audit
pip-audit
```

to check for known vulnerabilities in dependencies.

### SQL Injection

All database queries use SQLAlchemy's parameterized queries. Raw SQL strings are never constructed from user input.

### Agent Tool Inputs

All agent tool inputs are validated with Pydantic v2 before execution. Invalid inputs raise `ValidationError` and are logged, never executed.
