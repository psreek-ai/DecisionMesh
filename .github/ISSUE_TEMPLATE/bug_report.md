---
name: Bug Report
about: Something isn't working as expected
labels: bug
---

## Description

A clear, concise description of what the bug is.

## Steps to Reproduce

```bash
dm add
# ... what you typed
dm check <id>
# ... what happened
```

## Expected Behavior

What you expected to happen.

## Actual Behavior

What actually happened. Include the full error output.

```
paste error here
```

## Environment

- OS: [e.g. macOS 14, Ubuntu 22.04]
- Python version: [e.g. 3.12.2]
- DecisionMesh version: [e.g. 0.1.0 — run `pip show decisionmesh`]
- Database backend: [SQLite / PostgreSQL]

## Agent Log (if relevant)

If the bug involves an agent run, please include the relevant agent log output:

```bash
dm history <decision-id>
```

```
paste log here
```

## Additional Context

Any other context, screenshots, or information about the problem.

---

**Privacy note:** Please do not include your actual decision content, API keys, or personal data in bug reports. Sanitize any logs before pasting.
