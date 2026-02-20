"""
DecisionMesh — Master Orchestrator
Coordinates all agents, manages scheduler job registry,
surfaces pending divergence inbox on user session start.
Runs as a singleton.
"""
import asyncio
import logging
from typing import Optional

from sqlalchemy.ext.asyncio import AsyncSession

from decisionmesh.config import Settings, settings
from decisionmesh.database import get_session

logger = logging.getLogger(__name__)


class DecisionMeshOrchestrator:
    """
    Stateful coordinator. Runs as a singleton.
    On startup: loads all ACTIVE decisions, re-registers their monitor schedules.
    On new decision: runs Capture Agent, stores result, registers monitor job.
    On divergence: runs Causal Agent async, writes to pending inbox.
    On user session start: checks pending inbox, surfaces unacknowledged alerts.
    """

    _instance: Optional["DecisionMeshOrchestrator"] = None

    def __init__(self, config: Settings = None):
        self.config = config or settings
        self._scheduler = None
        self._started = False

    @classmethod
    def get_instance(cls, config: Settings = None) -> "DecisionMeshOrchestrator":
        if cls._instance is None:
            cls._instance = cls(config)
        return cls._instance

    async def startup(self) -> None:
        """Initialize orchestrator: load decisions, register scheduler jobs."""
        if self._started:
            return

        from decisionmesh.scheduler.monitor_scheduler import DecisionScheduler
        self._scheduler = DecisionScheduler(self)
        await self._scheduler.start()

        # Re-register all active and drifted decisions
        await self._restore_scheduler_jobs()
        self._started = True
        logger.info("DecisionMesh Orchestrator started.")

    async def shutdown(self) -> None:
        """Graceful shutdown."""
        if self._scheduler:
            await self._scheduler.stop()
        self._started = False
        logger.info("DecisionMesh Orchestrator stopped.")

    async def _restore_scheduler_jobs(self) -> None:
        """On startup, re-register all active decisions with the scheduler."""
        from sqlalchemy import select
        from decisionmesh.models.decision import DecisionORM, DecisionStatus

        async with get_session() as session:
            result = await session.execute(
                select(DecisionORM).where(
                    DecisionORM.status.in_([
                        DecisionStatus.ACTIVE.value,
                        DecisionStatus.DRIFTED.value,
                    ])
                )
            )
            decisions = result.scalars().all()

        for decision in decisions:
            if self._scheduler:
                await self._scheduler.register_decision(
                    decision_id=decision.id,
                    interval_days=decision.review_interval_days,
                )

        logger.info(f"Restored {len(decisions)} scheduler jobs.")

    async def capture_decision(self, narrative: str, web_monitoring: bool = False) -> str:
        """
        Entry point for new decisions from CLI/API.
        Runs Capture Agent, saves to DB, registers monitor schedule.
        Returns the new decision ID.
        """
        from decisionmesh.agents.capture_agent import CaptureAgent

        async with get_session() as session:
            agent = CaptureAgent(db_session=session, config=self.config)
            result = await agent.capture(narrative)

            if result.clarification_needed:
                raise ValueError(f"Clarification needed: {result.clarification_needed}")

            if web_monitoring:
                result.decision.web_monitoring_enabled = True

            decision_id = await agent.save_to_db(result)

        # Register with scheduler
        if self._scheduler:
            await self._scheduler.register_decision(
                decision_id=decision_id,
                interval_days=result.decision.review_interval_days,
            )

        logger.info(f"Captured decision {decision_id}: {result.decision.title}")
        return decision_id

    async def run_monitor(self, decision_id: str) -> dict:
        """
        Run a monitoring cycle for a specific decision.
        Called by the scheduler or manually by the user.
        """
        from decisionmesh.agents.monitor_agent import MonitorAgent

        async with get_session() as session:
            agent = MonitorAgent(
                db_session=session,
                config=self.config,
                causal_trigger_callback=self._on_divergence_detected,
            )
            result = await agent.run_monitor(decision_id)

        logger.info(
            f"Monitor run for {decision_id}: divergence_score={result.get('divergence_score', 0):.2f}"
        )
        return result

    async def _on_divergence_detected(self, divergence_event_id: str, decision_id: str) -> None:
        """Callback: triggered by MonitorAgent when divergence threshold is exceeded."""
        logger.info(f"Divergence detected for {decision_id}, triggering Causal Agent...")
        # Run causal analysis in background to not block monitor
        asyncio.create_task(self._run_causal_analysis(divergence_event_id, decision_id))

    async def _run_causal_analysis(self, divergence_event_id: str, decision_id: str) -> None:
        """Run Causal Agent asynchronously."""
        from decisionmesh.agents.causal_agent import CausalAgent

        try:
            async with get_session() as session:
                agent = CausalAgent(db_session=session, config=self.config)
                result = await agent.analyze(divergence_event_id, decision_id)
                logger.info(
                    f"Causal analysis complete for {decision_id}: "
                    f"counterfactual_id={result.get('counterfactual_id')}"
                )
        except Exception as e:
            logger.error(f"Causal analysis failed for {decision_id}: {e}")

    async def get_pending_inbox(self) -> list[dict]:
        """Return all unacknowledged divergence events — the user's inbox."""
        from sqlalchemy import select
        from decisionmesh.models.divergence import DivergenceEventORM
        from decisionmesh.models.decision import DecisionORM

        async with get_session() as session:
            result = await session.execute(
                select(DivergenceEventORM, DecisionORM)
                .join(DecisionORM, DivergenceEventORM.decision_id == DecisionORM.id)
                .where(DivergenceEventORM.acknowledged_by_user == False)  # noqa: E712
                .order_by(DivergenceEventORM.detected_at.desc())
            )
            rows = result.all()

        inbox = []
        for event, decision in rows:
            inbox.append({
                "divergence_event_id": event.id,
                "decision_id": event.decision_id,
                "decision_title": decision.title,
                "severity": event.severity,
                "divergence_score": event.divergence_score,
                "summary": event.summary,
                "detected_at": event.detected_at.isoformat(),
            })
        return inbox

    async def present_revision(
        self,
        divergence_event_id: str,
        decision_id: str,
        user_choice: Optional[str] = None,
    ) -> str:
        """Run the Revision Advisor for a specific divergence event."""
        from decisionmesh.agents.revision_agent import RevisionAgent

        async with get_session() as session:
            agent = RevisionAgent(db_session=session, config=self.config)
            response = await agent.present_divergence(
                decision_id=decision_id,
                divergence_event_id=divergence_event_id,
                user_choice=user_choice,
            )
            if user_choice:
                await agent.acknowledge_divergence(divergence_event_id)

        return response

    async def pause_decision(self, decision_id: str) -> None:
        """Pause monitoring for a decision."""
        from decisionmesh.models.decision import DecisionORM, DecisionStatus
        from sqlalchemy import select

        async with get_session() as session:
            result = await session.execute(
                select(DecisionORM).where(DecisionORM.id == decision_id)
            )
            decision = result.scalar_one_or_none()
            if decision:
                decision.status = DecisionStatus.PAUSED.value

        if self._scheduler:
            await self._scheduler.remove_decision(decision_id)

    async def resume_decision(self, decision_id: str) -> None:
        """Resume monitoring for a decision."""
        from decisionmesh.models.decision import DecisionORM, DecisionStatus
        from sqlalchemy import select

        async with get_session() as session:
            result = await session.execute(
                select(DecisionORM).where(DecisionORM.id == decision_id)
            )
            decision = result.scalar_one_or_none()
            if decision:
                decision.status = DecisionStatus.ACTIVE.value
                interval = decision.review_interval_days

        if self._scheduler:
            await self._scheduler.register_decision(decision_id, interval)
