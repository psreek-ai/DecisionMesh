"""
DecisionMesh CLI — Typer + Rich interface.
All user-facing commands. Entry point: `dm`
"""
import asyncio
import json
from datetime import datetime
from typing import Optional

import typer
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.text import Text
from rich import box
from rich.prompt import Prompt, Confirm

app = typer.Typer(
    name="dm",
    help="DecisionMesh — Temporal Decision Intelligence Engine",
    add_completion=False,
)
console = Console()


def run_async(coro):
    """Helper to run async functions in the CLI."""
    return asyncio.run(coro)


def _status_color(status: str) -> str:
    return {
        "active": "green",
        "drifted": "yellow",
        "revised": "blue",
        "closed": "dim",
        "paused": "dim cyan",
    }.get(status, "white")


def _severity_color(severity: str) -> str:
    return {
        "low": "cyan",
        "medium": "yellow",
        "high": "red",
        "critical": "bold red",
    }.get(severity, "white")


async def _ensure_db():
    from decisionmesh.database import init_db
    await init_db()


# ── dm add ────────────────────────────────────────────────────────────────────

@app.command()
def add(
    web: bool = typer.Option(False, "--web", help="Enable web monitoring for this decision"),
    narrative: Optional[str] = typer.Option(None, "--narrative", "-n", help="Decision narrative (skips interactive wizard)"),
):
    """Capture a new decision (interactive wizard)."""
    console.print(Panel.fit(
        "[bold cyan]DecisionMesh — Decision Capture[/bold cyan]\n"
        "[dim]Tell me about a significant decision you've made.[/dim]",
        border_style="cyan",
    ))

    if not narrative:
        console.print("\n[dim]Describe your decision — what you decided, why, and what you believe to be true:[/dim]\n")
        lines = []
        console.print("[dim](Enter your narrative. Type a blank line when done.)[/dim]\n")
        while True:
            line = typer.prompt("", prompt_suffix="")
            if not line:
                break
            lines.append(line)
        narrative = "\n".join(lines)

    if not narrative.strip():
        console.print("[red]No narrative provided. Aborted.[/red]")
        raise typer.Exit(1)

    if web:
        confirmed = Confirm.ask(
            "[yellow]Enable web monitoring?[/yellow] This allows the system to search the web "
            "when checking your premises. Web searches are logged.",
            default=False,
        )
        web = confirmed

    async def _capture():
        await _ensure_db()
        from decisionmesh.agents.orchestrator import DecisionMeshOrchestrator
        orchestrator = DecisionMeshOrchestrator.get_instance()
        # Start without full scheduler for CLI use
        decision_id = await orchestrator.capture_decision(narrative, web_monitoring=web)
        return decision_id

    with console.status("[cyan]Analyzing your decision narrative...[/cyan]"):
        try:
            decision_id = run_async(_capture())
        except ValueError as e:
            console.print(f"\n[yellow]Clarification needed:[/yellow] {e}")
            raise typer.Exit(1)
        except Exception as e:
            console.print(f"\n[red]Error:[/red] {e}")
            raise typer.Exit(1)

    console.print(f"\n[bold green]✓ Decision captured![/bold green]")
    console.print(f"  ID: [cyan]{decision_id}[/cyan]")
    console.print(f"\nRun [bold]dm show {decision_id[:8]}[/bold] to view your Decision DNA.")
    console.print(f"Run [bold]dm check {decision_id[:8]}[/bold] to run a manual monitoring check.")


# ── dm list ───────────────────────────────────────────────────────────────────

@app.command(name="list")
def list_decisions(
    status: Optional[str] = typer.Option(None, "--status", "-s", help="Filter by status"),
    domain: Optional[str] = typer.Option(None, "--domain", "-d"),
):
    """List all active decisions with status indicators."""
    async def _list():
        await _ensure_db()
        from sqlalchemy import select
        from decisionmesh.database import get_session
        from decisionmesh.models.decision import DecisionORM
        from decisionmesh.models.premise import PremiseORM

        async with get_session() as session:
            query = select(DecisionORM)
            if status:
                query = query.where(DecisionORM.status == status)
            if domain:
                query = query.where(DecisionORM.domain == domain)
            query = query.order_by(DecisionORM.created_at.desc())
            result = await session.execute(query)
            decisions = result.scalars().all()

            rows = []
            for d in decisions:
                p_result = await session.execute(
                    select(PremiseORM).where(PremiseORM.decision_id == d.id)
                )
                pc = len(p_result.scalars().all())
                rows.append((d, pc))
            return rows

    rows = run_async(_list())

    if not rows:
        console.print("[dim]No decisions found.[/dim]")
        return

    table = Table(box=box.ROUNDED, show_header=True, header_style="bold cyan")
    table.add_column("ID", style="dim", width=10)
    table.add_column("Title", min_width=30)
    table.add_column("Status", width=10)
    table.add_column("Domain", width=12)
    table.add_column("Premises", justify="right", width=9)
    table.add_column("Date", width=12)
    table.add_column("Web", width=4)

    for d, pc in rows:
        color = _status_color(d.status)
        table.add_row(
            d.id[:8],
            d.title[:50] + ("..." if len(d.title) > 50 else ""),
            f"[{color}]{d.status}[/{color}]",
            d.domain,
            str(pc),
            d.decision_date.strftime("%Y-%m-%d"),
            "🌐" if d.web_monitoring_enabled else "—",
        )

    console.print(table)
    console.print(f"[dim]{len(rows)} decision(s)[/dim]")


# ── dm show ───────────────────────────────────────────────────────────────────

@app.command()
def show(decision_id: str = typer.Argument(help="Decision ID or prefix")):
    """Show full Decision DNA for a decision."""
    async def _show():
        await _ensure_db()
        from sqlalchemy import select
        from decisionmesh.database import get_session
        from decisionmesh.models.decision import DecisionORM
        from decisionmesh.models.premise import PremiseORM

        async with get_session() as session:
            result = await session.execute(
                select(DecisionORM).where(DecisionORM.id.startswith(decision_id))
            )
            decision = result.scalars().first()
            if not decision:
                return None, []
            p_result = await session.execute(
                select(PremiseORM).where(PremiseORM.decision_id == decision.id)
            )
            premises = p_result.scalars().all()
            return decision, premises

    decision, premises = run_async(_show())
    if not decision:
        console.print(f"[red]Decision '{decision_id}' not found.[/red]")
        raise typer.Exit(1)

    color = _status_color(decision.status)
    console.print(Panel(
        f"[bold]{decision.title}[/bold]\n"
        f"[dim]ID: {decision.id}[/dim]\n"
        f"Status: [{color}]{decision.status}[/{color}]  |  Domain: {decision.domain}  |  "
        f"Date: {decision.decision_date.strftime('%Y-%m-%d')}\n"
        f"Review interval: every {decision.review_interval_days} days  |  "
        f"Web monitoring: {'enabled' if decision.web_monitoring_enabled else 'disabled'}",
        border_style=color,
        title="[bold cyan]Decision DNA[/bold cyan]",
    ))

    console.print("\n[bold]Narrative:[/bold]")
    console.print(f"  {decision.full_narrative}\n")

    if premises:
        console.print("[bold]Premises:[/bold]")
        for i, p in enumerate(premises, 1):
            p_color = {"valid": "green", "uncertain": "yellow", "invalidated": "red", "unverifiable": "dim"}.get(p.status, "white")
            console.print(f"  [{p_color}]{i}. [{p.status}][/{p_color}] {p.text}")
        console.print()

    if decision.predicted_outcomes:
        console.print("[bold]Predicted Outcomes:[/bold]")
        for o in (decision.predicted_outcomes if isinstance(decision.predicted_outcomes, list) else []):
            console.print(f"  • {o}")
        console.print()

    if decision.triggering_conditions:
        console.print("[bold]Triggering Conditions:[/bold]")
        for c in (decision.triggering_conditions if isinstance(decision.triggering_conditions, list) else []):
            console.print(f"  ⚡ {c}")


# ── dm inbox ──────────────────────────────────────────────────────────────────

@app.command()
def inbox():
    """Show pending divergence alerts."""
    async def _inbox():
        await _ensure_db()
        from decisionmesh.agents.orchestrator import DecisionMeshOrchestrator
        orchestrator = DecisionMeshOrchestrator.get_instance()
        return await orchestrator.get_pending_inbox()

    items = run_async(_inbox())

    if not items:
        console.print(Panel.fit(
            "[green]✓ Inbox clear[/green] — No unacknowledged divergence events.",
            border_style="green",
        ))
        return

    console.print(Panel.fit(
        f"[bold yellow]⚠  {len(items)} Unacknowledged Divergence Alert(s)[/bold yellow]",
        border_style="yellow",
    ))

    for item in items:
        color = _severity_color(item["severity"])
        console.print(Panel(
            f"Decision: [bold]{item['decision_title']}[/bold]\n"
            f"Severity: [{color}]{item['severity'].upper()}[/{color}]  |  "
            f"Score: {item['divergence_score']:.0%}\n"
            f"Detected: {item['detected_at'][:10]}\n\n"
            f"[italic]{item['summary']}[/italic]\n\n"
            f"Run: [bold]dm review {item['divergence_event_id'][:8]}[/bold]",
            border_style=color,
        ))


# ── dm review ─────────────────────────────────────────────────────────────────

@app.command()
def review(divergence_id: str = typer.Argument(help="Divergence event ID or prefix")):
    """Review a divergence event interactively."""
    async def _get_divergence():
        await _ensure_db()
        from sqlalchemy import select
        from decisionmesh.database import get_session
        from decisionmesh.models.divergence import DivergenceEventORM

        async with get_session() as session:
            result = await session.execute(
                select(DivergenceEventORM).where(DivergenceEventORM.id.startswith(divergence_id))
            )
            return result.scalars().first()

    divergence = run_async(_get_divergence())
    if not divergence:
        console.print(f"[red]Divergence event '{divergence_id}' not found.[/red]")
        raise typer.Exit(1)

    async def _present():
        from decisionmesh.agents.orchestrator import DecisionMeshOrchestrator
        orchestrator = DecisionMeshOrchestrator.get_instance()
        return await orchestrator.present_revision(
            divergence_event_id=divergence.id,
            decision_id=divergence.decision_id,
        )

    with console.status("[cyan]Preparing revision briefing...[/cyan]"):
        presentation = run_async(_present())

    console.print(Panel(presentation, title="[bold cyan]Revision Briefing[/bold cyan]", border_style="cyan"))

    # Interactive revision loop
    choice = Prompt.ask(
        "\nYour choice (or press Enter to exit without changes)",
        default="",
    )

    if choice.strip():
        async def _execute_choice():
            from decisionmesh.agents.orchestrator import DecisionMeshOrchestrator
            orchestrator = DecisionMeshOrchestrator.get_instance()
            return await orchestrator.present_revision(
                divergence_event_id=divergence.id,
                decision_id=divergence.decision_id,
                user_choice=choice,
            )

        with console.status("[cyan]Applying your revision...[/cyan]"):
            result = run_async(_execute_choice())

        console.print(Panel(result, title="[bold green]Revision Applied[/bold green]", border_style="green"))


# ── dm check ──────────────────────────────────────────────────────────────────

@app.command()
def check(decision_id: str = typer.Argument(help="Decision ID or prefix")):
    """Manually trigger a monitor run for a decision."""
    async def _check():
        await _ensure_db()
        from decisionmesh.agents.orchestrator import DecisionMeshOrchestrator
        orchestrator = DecisionMeshOrchestrator.get_instance()
        return await orchestrator.run_monitor(decision_id)

    with console.status(f"[cyan]Running monitor for {decision_id[:8]}...[/cyan]"):
        result = run_async(_check())

    score = result.get("divergence_score", 0)
    color = "green" if score < 0.3 else ("yellow" if score < 0.7 else "red")

    console.print(Panel(
        f"[bold]Divergence Score:[/bold] [{color}]{score:.0%}[/{color}]\n"
        f"[bold]Divergence Event:[/bold] {'Created — check inbox!' if result.get('divergence_event_created') else 'None (below threshold)'}\n"
        + (f"Event ID: {result.get('divergence_event_id', '')[:8]}" if result.get("divergence_event_id") else ""),
        title="[bold]Monitor Run Complete[/bold]",
        border_style=color,
    ))


# ── dm history ────────────────────────────────────────────────────────────────

@app.command()
def history(decision_id: str = typer.Argument(help="Decision ID or prefix")):
    """Show the full agent reasoning audit trail for a decision."""
    async def _history():
        await _ensure_db()
        from sqlalchemy import select, desc
        from decisionmesh.database import get_session
        from decisionmesh.models.agent_log import AgentLogORM
        from decisionmesh.models.decision import DecisionORM

        async with get_session() as session:
            d_result = await session.execute(
                select(DecisionORM).where(DecisionORM.id.startswith(decision_id))
            )
            decision = d_result.scalars().first()
            if not decision:
                return None, []

            result = await session.execute(
                select(AgentLogORM)
                .where(AgentLogORM.decision_id == decision.id)
                .order_by(AgentLogORM.logged_at)
            )
            logs = result.scalars().all()
            return decision, logs

    decision, logs = run_async(_history())

    if not decision:
        console.print(f"[red]Decision '{decision_id}' not found.[/red]")
        raise typer.Exit(1)

    console.print(f"\n[bold cyan]Agent History — {decision.title}[/bold cyan]")
    console.print(f"[dim]{len(logs)} log entries[/dim]\n")

    for log in logs:
        agent_color = {"capture_agent": "blue", "monitor_agent": "green", "causal_agent": "magenta", "revision_agent": "cyan"}.get(log.agent_name, "white")
        console.print(f"[{agent_color}]{log.agent_name}[/{agent_color}] iter={log.iteration}  [{log.logged_at.strftime('%Y-%m-%d %H:%M:%S')}]")
        if log.tool_name:
            console.print(f"  Tool: [bold]{log.tool_name}[/bold]")
        if log.thinking_content:
            console.print(f"  [dim italic]Thinking: {log.thinking_content[:200]}...[/dim italic]")
        if log.text_content:
            console.print(f"  {log.text_content[:300]}")
        console.print()


# ── dm timeline ───────────────────────────────────────────────────────────────

@app.command()
def timeline(decision_id: str = typer.Argument(help="Decision ID or prefix")):
    """Show a timeline visualization of a decision's life."""
    async def _timeline():
        await _ensure_db()
        from sqlalchemy import select
        from decisionmesh.database import get_session
        from decisionmesh.models.decision import DecisionORM
        from decisionmesh.models.divergence import DivergenceEventORM
        from decisionmesh.models.observation import ObservationORM

        async with get_session() as session:
            d_result = await session.execute(
                select(DecisionORM).where(DecisionORM.id.startswith(decision_id))
            )
            decision = d_result.scalars().first()
            if not decision:
                return None, [], []

            e_result = await session.execute(
                select(DivergenceEventORM)
                .where(DivergenceEventORM.decision_id == decision.id)
                .order_by(DivergenceEventORM.detected_at)
            )
            events = e_result.scalars().all()

            o_result = await session.execute(
                select(ObservationORM)
                .where(ObservationORM.decision_id == decision.id)
                .order_by(ObservationORM.observed_at)
                .limit(20)
            )
            observations = o_result.scalars().all()

            return decision, events, observations

    decision, events, observations = run_async(_timeline())

    if not decision:
        console.print(f"[red]Decision '{decision_id}' not found.[/red]")
        raise typer.Exit(1)

    console.print(f"\n[bold cyan]Timeline — {decision.title}[/bold cyan]\n")

    # Decision captured
    console.print(f"  [green]●[/green] [bold]{decision.decision_date.strftime('%Y-%m-%d')}[/bold] — Decision made")
    console.print(f"  [dim]│[/dim]  Status: {decision.status}")

    # Observations
    for obs in observations:
        color = "green" if obs.supports_premise else ("red" if obs.supports_premise is False else "yellow")
        console.print(f"  [dim]│[/dim]")
        console.print(f"  [{color}]◆[/{color}] [bold]{obs.observed_at.strftime('%Y-%m-%d')}[/bold] — Observation")
        console.print(f"  [dim]│[/dim]  {obs.observation_text[:100]}")

    # Divergence events
    for event in events:
        color = _severity_color(event.severity)
        console.print(f"  [dim]│[/dim]")
        console.print(f"  [{color}]⚠ [/]{color}] [bold]{event.detected_at.strftime('%Y-%m-%d')}[/bold] — Divergence ({event.severity})")
        console.print(f"  [dim]│[/dim]  Score: {event.divergence_score:.0%} | {event.summary[:100]}")

    console.print(f"  [dim]│[/dim]")
    status_color = _status_color(decision.status)
    console.print(f"  [{status_color}]●[/{status_color}] Now — Status: {decision.status}\n")


# ── dm export ─────────────────────────────────────────────────────────────────

@app.command()
def export(
    output: str = typer.Option("decisions_backup.json", "--output", "-o"),
    fmt: str = typer.Option("json", "--format", "-f"),
):
    """Export all decisions as JSON."""
    async def _export():
        await _ensure_db()
        from sqlalchemy import select
        from decisionmesh.database import get_session
        from decisionmesh.models.decision import DecisionORM
        from decisionmesh.models.premise import PremiseORM

        async with get_session() as session:
            result = await session.execute(select(DecisionORM))
            decisions = result.scalars().all()

            export_data = []
            for d in decisions:
                d_dict = d.to_pydantic().model_dump(mode="json")
                d_dict.pop("embedding", None)

                p_result = await session.execute(
                    select(PremiseORM).where(PremiseORM.decision_id == d.id)
                )
                premises = []
                for p in p_result.scalars().all():
                    p_dict = p.to_pydantic().model_dump(mode="json")
                    p_dict.pop("embedding", None)
                    premises.append(p_dict)

                export_data.append({"decision": d_dict, "premises": premises})

            return export_data

    with console.status("[cyan]Exporting decisions...[/cyan]"):
        data = run_async(_export())

    export_doc = {
        "exported_at": datetime.utcnow().isoformat(),
        "version": "0.1.0",
        "count": len(data),
        "decisions": data,
    }

    with open(output, "w") as f:
        json.dump(export_doc, f, indent=2, default=str)

    console.print(f"[green]✓ Exported {len(data)} decision(s) to {output}[/green]")


# ── dm pause / resume ─────────────────────────────────────────────────────────

@app.command()
def pause(decision_id: str = typer.Argument(help="Decision ID or prefix")):
    """Pause monitoring for a decision."""
    async def _pause():
        await _ensure_db()
        from sqlalchemy import select
        from decisionmesh.database import get_session
        from decisionmesh.models.decision import DecisionORM, DecisionStatus

        async with get_session() as session:
            result = await session.execute(
                select(DecisionORM).where(DecisionORM.id.startswith(decision_id))
            )
            decision = result.scalars().first()
            if not decision:
                return None
            decision.status = DecisionStatus.PAUSED.value
            return decision.id

    did = run_async(_pause())
    if not did:
        console.print(f"[red]Decision '{decision_id}' not found.[/red]")
        raise typer.Exit(1)
    console.print(f"[yellow]⏸ Monitoring paused for {did[:8]}[/yellow]")


@app.command()
def resume(decision_id: str = typer.Argument(help="Decision ID or prefix")):
    """Resume monitoring for a decision."""
    async def _resume():
        await _ensure_db()
        from sqlalchemy import select
        from decisionmesh.database import get_session
        from decisionmesh.models.decision import DecisionORM, DecisionStatus

        async with get_session() as session:
            result = await session.execute(
                select(DecisionORM).where(DecisionORM.id.startswith(decision_id))
            )
            decision = result.scalars().first()
            if not decision:
                return None
            decision.status = DecisionStatus.ACTIVE.value
            return decision.id

    did = run_async(_resume())
    if not did:
        console.print(f"[red]Decision '{decision_id}' not found.[/red]")
        raise typer.Exit(1)
    console.print(f"[green]▶ Monitoring resumed for {did[:8]}[/green]")


# ── dm close ──────────────────────────────────────────────────────────────────

@app.command()
def close(
    decision_id: str = typer.Argument(help="Decision ID or prefix"),
    reason: str = typer.Option("User closed decision", "--reason", "-r"),
):
    """Close a decision (end of lifecycle)."""
    confirmed = Confirm.ask(f"Close decision [cyan]{decision_id[:8]}[/cyan]?", default=False)
    if not confirmed:
        console.print("[dim]Aborted.[/dim]")
        return

    async def _close():
        await _ensure_db()
        from sqlalchemy import select
        from decisionmesh.database import get_session
        from decisionmesh.models.decision import DecisionORM, DecisionStatus

        async with get_session() as session:
            result = await session.execute(
                select(DecisionORM).where(DecisionORM.id.startswith(decision_id))
            )
            decision = result.scalars().first()
            if not decision:
                return None
            decision.status = DecisionStatus.CLOSED.value
            return decision.id

    did = run_async(_close())
    if not did:
        console.print(f"[red]Decision '{decision_id}' not found.[/red]")
        raise typer.Exit(1)
    console.print(f"[dim]✓ Decision {did[:8]} closed.[/dim]")


# ── dm db ─────────────────────────────────────────────────────────────────────

db_app = typer.Typer(help="Database management commands.")
app.add_typer(db_app, name="db")


@db_app.command("migrate")
def db_migrate():
    """Run database migrations."""
    import subprocess
    console.print("[cyan]Running Alembic migrations...[/cyan]")
    result = subprocess.run(["alembic", "upgrade", "head"], capture_output=True, text=True)
    if result.returncode == 0:
        console.print("[green]✓ Migrations complete.[/green]")
    else:
        console.print(f"[red]Migration failed:[/red]\n{result.stderr}")
        raise typer.Exit(1)


@db_app.command("reset")
def db_reset():
    """Reset the database (drops all tables). Use with caution."""
    confirmed = Confirm.ask(
        "[bold red]WARNING: This will delete all data. Are you sure?[/bold red]",
        default=False,
    )
    if not confirmed:
        console.print("[dim]Aborted.[/dim]")
        return

    async def _reset():
        from decisionmesh.database import drop_db, init_db
        await drop_db()
        await init_db()

    run_async(_reset())
    console.print("[green]✓ Database reset complete.[/green]")


# ── dm scheduler ─────────────────────────────────────────────────────────────

scheduler_app = typer.Typer(help="Scheduler management commands.")
app.add_typer(scheduler_app, name="scheduler")


@scheduler_app.command("status")
def scheduler_status():
    """Show scheduler status and registered jobs."""
    async def _status():
        await _ensure_db()
        from decisionmesh.agents.orchestrator import DecisionMeshOrchestrator
        orchestrator = DecisionMeshOrchestrator.get_instance()
        await orchestrator.startup()
        if orchestrator._scheduler:
            return orchestrator._scheduler.get_scheduler_status()
        return {"running": False, "job_count": 0, "jobs": []}

    status = run_async(_status())

    console.print(f"\n[bold]Scheduler Status:[/bold] {'[green]running[/green]' if status['running'] else '[red]stopped[/red]'}")
    console.print(f"[bold]Active jobs:[/bold] {status['job_count']}\n")

    if status["jobs"]:
        table = Table(box=box.SIMPLE)
        table.add_column("Job ID")
        table.add_column("Name")
        table.add_column("Next Run")
        for job in status["jobs"]:
            table.add_row(job["id"][:20], job["name"], job["next_run"] or "—")
        console.print(table)


if __name__ == "__main__":
    app()
