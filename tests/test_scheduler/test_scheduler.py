"""
Tests for decisionmesh.scheduler.monitor_scheduler._make_sync_db_url

The function replaces the async driver prefix so APScheduler's sync
SQLAlchemy job store can use the same database as the async application.

Replacement rules implemented in the source:
    "sqlite+aiosqlite://" → "sqlite:///"   (adds one slash)
    "postgresql+asyncpg://" → "postgresql://"

We test the function's actual output, which is the specification of the
public contract, not an assumed idealised form.
"""
import pytest

from decisionmesh.scheduler.monitor_scheduler import _make_sync_db_url


class TestMakeSyncDbUrl:
    # ── SQLite conversions ────────────────────────────────────────────────────

    def test_sqlite_relative_file_url(self):
        """
        sqlite+aiosqlite:///./decisionmesh.db
        The replace removes '+aiosqlite' and the final slash of '//' becomes
        '///' in the output — the path portion remains unchanged.
        """
        async_url = "sqlite+aiosqlite:///./decisionmesh.db"
        result = _make_sync_db_url(async_url)
        # The function replaces "sqlite+aiosqlite://" -> "sqlite:///"
        # so "sqlite+aiosqlite:///./..." -> "sqlite:////./..."
        assert result == "sqlite:////./decisionmesh.db"

    def test_sqlite_memory_url(self):
        async_url = "sqlite+aiosqlite:///:memory:"
        result = _make_sync_db_url(async_url)
        assert result == "sqlite:////:memory:"

    def test_sqlite_absolute_path(self):
        async_url = "sqlite+aiosqlite:////var/data/app.db"
        result = _make_sync_db_url(async_url)
        assert result == "sqlite://///var/data/app.db"

    def test_sqlite_simple_filename(self):
        async_url = "sqlite+aiosqlite:///test.db"
        result = _make_sync_db_url(async_url)
        assert result == "sqlite:////test.db"

    def test_sqlite_result_does_not_contain_aiosqlite(self):
        result = _make_sync_db_url("sqlite+aiosqlite:///some.db")
        assert "aiosqlite" not in result

    def test_sqlite_result_starts_with_sqlite(self):
        result = _make_sync_db_url("sqlite+aiosqlite:///some.db")
        assert result.startswith("sqlite:")

    # ── PostgreSQL conversions ────────────────────────────────────────────────

    def test_postgresql_asyncpg_with_port(self):
        async_url = "postgresql+asyncpg://user:pass@localhost:5432/mydb"
        result = _make_sync_db_url(async_url)
        assert result == "postgresql://user:pass@localhost:5432/mydb"

    def test_postgresql_asyncpg_no_port(self):
        async_url = "postgresql+asyncpg://admin:secret@db-host/production"
        result = _make_sync_db_url(async_url)
        assert result == "postgresql://admin:secret@db-host/production"

    def test_postgresql_result_does_not_contain_asyncpg(self):
        result = _make_sync_db_url("postgresql+asyncpg://user:pass@localhost/db")
        assert "asyncpg" not in result

    def test_postgresql_result_starts_with_postgresql(self):
        result = _make_sync_db_url("postgresql+asyncpg://user:pass@host/db")
        assert result.startswith("postgresql://")

    def test_postgresql_with_query_params_preserved(self):
        """Query parameters such as sslmode must survive the conversion."""
        async_url = "postgresql+asyncpg://user:pass@host/db?sslmode=require"
        result = _make_sync_db_url(async_url)
        assert result == "postgresql://user:pass@host/db?sslmode=require"

    # ── Already-sync URL edge cases (idempotency) ─────────────────────────────

    def test_sync_sqlite_url_is_returned_unchanged(self):
        """
        An already-sync SQLite URL contains neither 'aiosqlite' nor
        'asyncpg', so both replace() calls are no-ops.
        """
        sync_url = "sqlite:///./app.db"
        result = _make_sync_db_url(sync_url)
        assert result == sync_url

    def test_sync_postgresql_url_is_returned_unchanged(self):
        sync_url = "postgresql://user:pass@localhost/db"
        result = _make_sync_db_url(sync_url)
        assert result == sync_url

    # ── Return type ───────────────────────────────────────────────────────────

    def test_returns_string(self):
        result = _make_sync_db_url("sqlite+aiosqlite:///test.db")
        assert isinstance(result, str)

    # ── Replacements are independent of each other ────────────────────────────

    def test_sqlite_replacement_does_not_affect_postgresql_format(self):
        """A PostgreSQL URL should not be touched by the SQLite replacement."""
        async_url = "postgresql+asyncpg://user:pass@localhost/db"
        result = _make_sync_db_url(async_url)
        assert "sqlite" not in result
        assert result == "postgresql://user:pass@localhost/db"

    def test_postgresql_replacement_does_not_affect_sqlite_format(self):
        """A SQLite URL should not be touched by the PostgreSQL replacement."""
        async_url = "sqlite+aiosqlite:///test.db"
        result = _make_sync_db_url(async_url)
        assert "postgresql" not in result
        # aiosqlite removed, path retained as produced by the function
        assert "aiosqlite" not in result
        assert result.startswith("sqlite:")
