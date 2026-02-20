"""
DecisionMesh Configuration
All settings via environment variables with sensible defaults.
"""
from typing import Optional
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        case_sensitive=True,
        extra="ignore",
    )

    # Required
    ANTHROPIC_API_KEY: str = ""

    # Optional — web monitoring
    TAVILY_API_KEY: Optional[str] = None

    # Database
    DATABASE_URL: str = "sqlite+aiosqlite:///./decisionmesh.db"

    # Agent behavior
    DEFAULT_MODEL: str = "claude-3-7-sonnet-20250219"
    MAX_AGENT_ITERATIONS: int = 10
    CAUSAL_THINKING_BUDGET: int = 8000
    DIVERGENCE_ALERT_THRESHOLD: float = 0.3   # 30% drift triggers alert
    DIVERGENCE_CRITICAL_THRESHOLD: float = 0.7  # 70% drift = critical

    # Scheduler
    DEFAULT_REVIEW_INTERVAL_DAYS: int = 30
    SCHEDULER_TIMEZONE: str = "UTC"

    # Privacy
    EMBEDDING_MODEL: str = "all-MiniLM-L6-v2"  # Always local, never API
    DISABLE_ALL_WEB_MONITORING: bool = False    # Global kill switch


def get_settings() -> Settings:
    """Return the application settings singleton."""
    return Settings()


# Module-level singleton
settings = get_settings()
