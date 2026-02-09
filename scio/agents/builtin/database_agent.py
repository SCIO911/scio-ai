"""
SCIO Database Agent

Agent fuer Datenbankoperationen (SQLite, PostgreSQL, MySQL).
"""

import sqlite3
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Generator, Optional
from urllib.parse import urlparse

from pydantic import Field

from scio.agents.base import Agent, AgentConfig, AgentContext
from scio.agents.registry import register_agent
from scio.core.exceptions import AgentError
from scio.core.logging import get_logger

logger = get_logger(__name__)


class DatabaseConfig(AgentConfig):
    """Konfiguration fuer den Database Agent."""

    name: str = Field(default="Database Agent")
    description: str = Field(default="Agent fuer Datenbankoperationen")

    # Connection settings
    connection_string: Optional[str] = Field(
        default=None,
        description="Datenbank-Verbindungsstring (z.B. sqlite:///data.db)"
    )
    db_type: str = Field(
        default="sqlite",
        description="Datenbanktyp: sqlite, postgresql, mysql"
    )

    # SQLite specific
    sqlite_path: Optional[str] = Field(
        default=None,
        description="Pfad zur SQLite-Datei"
    )

    # Connection pool settings
    pool_size: int = Field(default=5, ge=1, le=20)
    timeout_seconds: int = Field(default=30, ge=1)

    # Query settings
    max_rows: int = Field(default=10000, ge=1)
    allow_write: bool = Field(default=False, description="Erlaubt INSERT/UPDATE/DELETE")
    allow_ddl: bool = Field(default=False, description="Erlaubt CREATE/DROP/ALTER")


@register_agent("database")
class DatabaseAgent(Agent[dict[str, Any], dict[str, Any]]):
    """
    Agent fuer Datenbankoperationen.

    Unterstuetzt:
    - SQLite (eingebaut)
    - PostgreSQL (optional, benoetigt psycopg2)
    - MySQL (optional, benoetigt mysql-connector-python)

    Operationen:
    - SELECT: Daten abfragen
    - INSERT: Daten einfuegen (wenn allow_write=True)
    - UPDATE: Daten aktualisieren (wenn allow_write=True)
    - DELETE: Daten loeschen (wenn allow_write=True)
    - DDL: Schema-Aenderungen (wenn allow_ddl=True)
    """

    agent_type = "database"
    version = "1.0"

    def __init__(self, config: DatabaseConfig | dict[str, Any]):
        if isinstance(config, dict):
            config = DatabaseConfig(**config)
        super().__init__(config)
        self.config: DatabaseConfig = config
        self._connection = None

    def _get_connection(self) -> Any:
        """Gibt eine Datenbankverbindung zurueck."""
        # For SQLite in-memory, reuse connection
        if self.config.db_type == "sqlite" and self._connection is not None:
            return self._connection

        if self.config.db_type == "sqlite":
            db_path = self.config.sqlite_path or ":memory:"
            conn = sqlite3.connect(
                db_path,
                timeout=self.config.timeout_seconds
            )
            conn.row_factory = sqlite3.Row
            # Store connection for in-memory databases
            if db_path == ":memory:":
                self._connection = conn
        elif self.config.db_type == "postgresql":
            conn = self._connect_postgresql()
        elif self.config.db_type == "mysql":
            conn = self._connect_mysql()
        else:
            raise AgentError(
                f"Unterstuetzter Datenbanktyp: {self.config.db_type}",
                agent_id=self.agent_id
            )
        return conn

    @contextmanager
    def _connection_context(self) -> Generator[Any, None, None]:
        """Context manager for database connection."""
        conn = self._get_connection()
        try:
            yield conn
        finally:
            # Don't close in-memory SQLite connections
            if not (self.config.db_type == "sqlite" and
                    (self.config.sqlite_path or ":memory:") == ":memory:"):
                conn.close()

    def _connect_postgresql(self) -> Any:
        """Verbindet zu PostgreSQL."""
        try:
            import psycopg2
            import psycopg2.extras
        except ImportError:
            raise AgentError(
                "psycopg2 nicht installiert. Installiere mit: pip install psycopg2-binary",
                agent_id=self.agent_id
            )

        conn_str = self.config.connection_string
        if not conn_str:
            raise AgentError(
                "connection_string erforderlich fuer PostgreSQL",
                agent_id=self.agent_id
            )

        conn = psycopg2.connect(
            conn_str,
            connect_timeout=self.config.timeout_seconds
        )
        return conn

    def _connect_mysql(self) -> Any:
        """Verbindet zu MySQL."""
        try:
            import mysql.connector
        except ImportError:
            raise AgentError(
                "mysql-connector-python nicht installiert. Installiere mit: pip install mysql-connector-python",
                agent_id=self.agent_id
            )

        conn_str = self.config.connection_string
        if not conn_str:
            raise AgentError(
                "connection_string erforderlich fuer MySQL",
                agent_id=self.agent_id
            )

        # Parse connection string
        parsed = urlparse(conn_str)
        conn = mysql.connector.connect(
            host=parsed.hostname or "localhost",
            port=parsed.port or 3306,
            user=parsed.username,
            password=parsed.password,
            database=parsed.path.lstrip("/"),
            connection_timeout=self.config.timeout_seconds
        )
        return conn

    async def execute(
        self, input_data: dict[str, Any], context: AgentContext
    ) -> dict[str, Any]:
        """Fuehrt Datenbankoperationen aus."""

        operation = input_data.get("operation", "query")
        query = input_data.get("query", "")
        params = input_data.get("params", [])

        # Operations that don't require a query
        if operation in ["tables", "schema"]:
            pass
        elif not query:
            raise AgentError("Keine Query angegeben", agent_id=self.agent_id)

        # Validiere Operation wenn Query vorhanden
        if query:
            query_upper = query.strip().upper()
            is_select = query_upper.startswith("SELECT")
            is_write = any(query_upper.startswith(op) for op in ["INSERT", "UPDATE", "DELETE"])
            is_ddl = any(query_upper.startswith(op) for op in ["CREATE", "DROP", "ALTER", "TRUNCATE"])

            if is_write and not self.config.allow_write:
                raise AgentError(
                    "Schreiboperationen nicht erlaubt. Setze allow_write=True",
                    agent_id=self.agent_id
                )

            if is_ddl and not self.config.allow_ddl:
                raise AgentError(
                    "DDL-Operationen nicht erlaubt. Setze allow_ddl=True",
                    agent_id=self.agent_id
                )

            self.logger.info(
                "Executing database operation",
                operation=operation,
                query_type="SELECT" if is_select else "WRITE" if is_write else "DDL"
            )
        else:
            self.logger.info(
                "Executing database operation",
                operation=operation,
            )

        with self._connection_context() as conn:
            cursor = conn.cursor()

            try:
                if operation == "query":
                    return await self._execute_query(cursor, query, params, conn)
                elif operation == "execute":
                    return await self._execute_command(cursor, query, params, conn)
                elif operation == "execute_many":
                    return await self._execute_many(cursor, query, params, conn)
                elif operation == "tables":
                    return await self._get_tables(cursor)
                elif operation == "schema":
                    return await self._get_schema(cursor, input_data.get("table"))
                else:
                    raise AgentError(
                        f"Unbekannte Operation: {operation}",
                        agent_id=self.agent_id
                    )
            except AgentError:
                raise
            except Exception as e:
                conn.rollback()
                raise AgentError(f"Datenbankfehler: {e}", agent_id=self.agent_id)

    async def _execute_query(
        self, cursor: Any, query: str, params: list, conn: Any
    ) -> dict[str, Any]:
        """Fuehrt eine SELECT-Query aus."""
        cursor.execute(query, params)
        rows = cursor.fetchmany(self.config.max_rows)

        if self.config.db_type == "sqlite":
            columns = [desc[0] for desc in cursor.description]
            data = [dict(zip(columns, row)) for row in rows]
        else:
            columns = [desc[0] for desc in cursor.description]
            data = [dict(zip(columns, row)) for row in rows]

        return {
            "data": data,
            "columns": columns,
            "row_count": len(data),
            "truncated": len(rows) >= self.config.max_rows,
        }

    async def _execute_command(
        self, cursor: Any, query: str, params: list, conn: Any
    ) -> dict[str, Any]:
        """Fuehrt einen INSERT/UPDATE/DELETE-Befehl aus."""
        cursor.execute(query, params)
        conn.commit()

        return {
            "affected_rows": cursor.rowcount,
            "last_row_id": cursor.lastrowid if hasattr(cursor, "lastrowid") else None,
        }

    async def _execute_many(
        self, cursor: Any, query: str, params: list, conn: Any
    ) -> dict[str, Any]:
        """Fuehrt einen Befehl fuer mehrere Datensaetze aus."""
        cursor.executemany(query, params)
        conn.commit()

        return {
            "affected_rows": cursor.rowcount,
        }

    async def _get_tables(self, cursor: Any) -> dict[str, Any]:
        """Gibt alle Tabellen der Datenbank zurueck."""
        if self.config.db_type == "sqlite":
            cursor.execute(
                "SELECT name FROM sqlite_master WHERE type='table' ORDER BY name"
            )
        elif self.config.db_type == "postgresql":
            cursor.execute(
                "SELECT table_name FROM information_schema.tables "
                "WHERE table_schema = 'public' ORDER BY table_name"
            )
        elif self.config.db_type == "mysql":
            cursor.execute("SHOW TABLES")

        tables = [row[0] for row in cursor.fetchall()]
        return {"tables": tables, "count": len(tables)}

    async def _get_schema(
        self, cursor: Any, table: Optional[str]
    ) -> dict[str, Any]:
        """Gibt das Schema einer Tabelle zurueck."""
        if not table:
            raise AgentError("Tabellenname erforderlich", agent_id=self.agent_id)

        if self.config.db_type == "sqlite":
            cursor.execute(f"PRAGMA table_info({table})")
            columns = []
            for row in cursor.fetchall():
                columns.append({
                    "name": row[1],
                    "type": row[2],
                    "nullable": not row[3],
                    "default": row[4],
                    "primary_key": bool(row[5]),
                })
        elif self.config.db_type == "postgresql":
            cursor.execute(
                """
                SELECT column_name, data_type, is_nullable, column_default
                FROM information_schema.columns
                WHERE table_name = %s
                ORDER BY ordinal_position
                """,
                [table]
            )
            columns = []
            for row in cursor.fetchall():
                columns.append({
                    "name": row[0],
                    "type": row[1],
                    "nullable": row[2] == "YES",
                    "default": row[3],
                })
        elif self.config.db_type == "mysql":
            cursor.execute(f"DESCRIBE {table}")
            columns = []
            for row in cursor.fetchall():
                columns.append({
                    "name": row[0],
                    "type": row[1],
                    "nullable": row[2] == "YES",
                    "default": row[4],
                    "primary_key": row[3] == "PRI",
                })
        else:
            columns = []

        return {"table": table, "columns": columns}
