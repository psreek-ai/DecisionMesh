"""
DecisionMesh — Background Scheduler
APScheduler 3.10+ with SQLAlchemy job store.
Each decision gets an independent schedule. Jobs survive process restarts.
"""
import asyncio
import logging
from datetime import datetime, timedelta
from typing import TYPE_CHECKING, Optional

from apscheduler.schedulers.asyncio import AsyncIOScheduler
from apscheduler.jobstores.sqlalchemy import SQLAlchemyJobStore
from apscheduler.executors.asyncio import AsyncIOExecutor
from apscheduler.triggers.interval import IntervalTrigger

from decisionmesh.config import settings

if TYPE_CHECKING:
    from decisionmesh.agents.orchestrator import DecisionMeshOrchestrator

logger = logging.getLogger(__name__)


def _make_sync_db_url(async_url: str) -> str:
    """Convert async DB URL to sync for APScheduler's job store."""
    return (
        async_url
        .replace("sqlite+aiosqlite://", "sqlite:///")
        .replace("postgresql+asyncpg://", "postgresql://")
    )


class DecisionScheduler:
    """
    Wraps APScheduler for DecisionMesh monitoring jobs.
    Uses SQLAlchemyJobStore so jobs persist across restarts.
    """

    def __init__(self, orchestrator: "DecisionMeshOrchestrator"):
        self.orchestrator = orchestrator
        sync_db_url = _make_sync_db_url(settings.DATABASE_URL)

        jobstores = {
            "default": SQLAlchemyJobStore(url=sync_db_url, tablename="scheduler_jobs"),
        }
        executors = {
            "default": AsyncIOExecutor(),
        }
        job_defaults = {
            "coalesce": True,       # If a job missed its fire time, run once (not multiple)
            "max_instances": 1,     # Never run the same decision check twice simultaneously
            "misfire_grace_time": 3600,  # 1 hour grace for missed jobs
        }

        self._scheduler = AsyncIOScheduler(
            jobstores=jobstores,
            executors=executors,
            job_defaults=job_defaults,
            timezone=settings.SCHEDULER_TIMEZONE,
        )

    async def start(self) -> None:
        """Start the scheduler."""
        self._scheduler.start()
        logger.info("DecisionMesh scheduler started.")

    async def stop(self) -> None:
        """Stop the scheduler gracefully."""
        self._scheduler.shutdown(wait=False)
        logger.info("DecisionMesh scheduler stopped.")

    async def register_decision(self, decision_id: str, interval_days: int) -> None:
        """
        Register or update a monitoring job for a decision.
        If a job already exists for this decision, it is replaced.
        """
        job_id = f"monitor_{decision_id}"

        # Remove existing job if present
        existing = self._scheduler.get_job(job_id)
        if existing:
            existing.remove()

        self._scheduler.add_job(
            func=self._run_monitor_job,
            trigger=IntervalTrigger(days=interval_days),
            id=job_id,
            args=[decision_id],
            name=f"Monitor decision {decision_id[:8]}",
            replace_existing=True,
        )
        logger.info(f"Registered monitoring job for decision {decision_id[:8]} (every {interval_days} days)")

    async def remove_decision(self, decision_id: str) -> None:
        """Remove a monitoring job (when decision is paused or closed)."""
        job_id = f"monitor_{decision_id}"
        job = self._scheduler.get_job(job_id)
        if job:
            job.remove()
            logger.info(f"Removed monitoring job for decision {decision_id[:8]}")

    async def _run_monitor_job(self, decision_id: str) -> None:
        """The actual job function called by APScheduler."""
        logger.info(f"Scheduler: running monitor for decision {decision_id[:8]}")
        try:
            await self.orchestrator.run_monitor(decision_id)
        except Exception as e:
            logger.error(f"Scheduler: monitor job failed for {decision_id}: {e}")

    def get_jobs(self) -> list[dict]:
        """Return status of all registered jobs."""
        jobs = []
        for job in self._scheduler.get_jobs():
            next_run = job.next_run_time
            jobs.append({
                "id": job.id,
                "name": job.name,
                "next_run": next_run.isoformat() if next_run else None,
                "trigger": str(job.trigger),
            })
        return jobs

    def get_scheduler_status(self) -> dict:
        """Return overall scheduler status."""
        return {
            "running": self._scheduler.running,
            "job_count": len(self._scheduler.get_jobs()),
            "jobs": self.get_jobs(),
            "timezone": str(settings.SCHEDULER_TIMEZONE),
        }
