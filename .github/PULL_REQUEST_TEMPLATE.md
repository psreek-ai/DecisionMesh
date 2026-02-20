## Summary

<!-- 1-3 bullet points describing what this PR does -->

-
-

Closes #<!-- issue number -->

## Type of Change

- [ ] Bug fix (non-breaking)
- [ ] New feature
- [ ] Breaking change
- [ ] Documentation update
- [ ] Refactor (no behavior change)
- [ ] Test improvement

## Changes

<!-- Describe the key changes made -->

## Testing

- [ ] Tests added for all new behavior
- [ ] Existing tests still pass (`pytest`)
- [ ] Tested with a real Anthropic API key (if agent behavior changed)
- [ ] Mocked Anthropic client used in unit tests (no real API calls in test suite)

## Principles Checklist

- [ ] Does **not** make decisions for the user
- [ ] Does **not** add engagement mechanics (streaks, nudges, notifications)
- [ ] All agent reasoning is logged to `agent_logs` table
- [ ] No hardcoded API keys, credentials, or secrets
- [ ] No telemetry or data sent to third-party services without explicit user opt-in
- [ ] Works offline (except for optional Tavily web search)

## Migration Required?

- [ ] No database changes
- [ ] New migration added to `alembic/versions/`

## Documentation

- [ ] README updated (if user-facing change)
- [ ] CHANGELOG.md updated under `[Unreleased]`
- [ ] Docstrings added for new public functions
