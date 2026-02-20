FROM python:3.12-slim

LABEL maintainer="DecisionMesh Contributors"
LABEL description="Temporal Decision Intelligence Engine"

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    && rm -rf /var/lib/apt/lists/*

# Copy dependency files first (for layer caching)
COPY pyproject.toml ./

# Install Python dependencies
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -e ".[dev]"

# Pre-download the embedding model (offline-first)
RUN python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('all-MiniLM-L6-v2')" || true

# Copy application code
COPY . .

# Create data directory for SQLite database
RUN mkdir -p /data
ENV DATABASE_URL=sqlite+aiosqlite:////data/decisionmesh.db

# Expose API port
EXPOSE 8000

# Run migrations then start API server
CMD ["sh", "-c", "alembic upgrade head && uvicorn decisionmesh.api.main:app --host 0.0.0.0 --port 8000"]
