"""
Tests for SCIO Database and API Agents

Tests for:
- DatabaseAgent with SQLite
- APIAgent for REST and GraphQL
- WebhookAgent
"""

import asyncio
import json
import pytest
import tempfile
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

from scio.agents.base import AgentContext, AgentState
from scio.agents.builtin.database_agent import DatabaseAgent, DatabaseConfig
from scio.agents.builtin.api_agent import APIAgent, APIConfig, WebhookAgent


# =============================================================================
# DATABASE AGENT TESTS
# =============================================================================

class TestDatabaseConfig:
    """Tests for DatabaseConfig."""

    def test_default_config(self):
        """Test default configuration."""
        config = DatabaseConfig(name="TestDB")
        assert config.db_type == "sqlite"
        assert config.pool_size == 5
        assert config.max_rows == 10000
        assert config.allow_write is False
        assert config.allow_ddl is False

    def test_custom_config(self):
        """Test custom configuration."""
        config = DatabaseConfig(
            name="TestDB",
            db_type="postgresql",
            connection_string="postgresql://user:pass@localhost/db",
            allow_write=True,
            max_rows=1000,
        )
        assert config.db_type == "postgresql"
        assert config.allow_write is True
        assert config.max_rows == 1000


class TestDatabaseAgent:
    """Tests for DatabaseAgent."""

    @pytest.fixture
    def agent(self):
        """Create a test agent with in-memory SQLite."""
        config = DatabaseConfig(
            name="TestDB",
            db_type="sqlite",
            sqlite_path=":memory:",
            allow_write=True,
            allow_ddl=True,
        )
        return DatabaseAgent(config)

    @pytest.fixture
    def context(self):
        """Create test context."""
        return AgentContext(
            agent_id="test-agent",
            execution_id="test-exec",
        )

    def test_create_agent(self, agent):
        """Test creating agent."""
        assert agent.agent_type == "database"
        assert agent.version == "1.0"
        assert agent.config.db_type == "sqlite"

    @pytest.mark.asyncio
    async def test_create_table(self, agent, context):
        """Test creating a table."""
        result = await agent.execute(
            {
                "operation": "execute",
                "query": """
                    CREATE TABLE users (
                        id INTEGER PRIMARY KEY,
                        name TEXT NOT NULL,
                        email TEXT
                    )
                """,
            },
            context,
        )
        assert "affected_rows" in result

    @pytest.mark.asyncio
    async def test_insert_and_query(self, agent, context):
        """Test inserting and querying data."""
        # Create table
        await agent.execute(
            {
                "operation": "execute",
                "query": "CREATE TABLE test (id INTEGER PRIMARY KEY, value TEXT)",
            },
            context,
        )

        # Insert data
        result = await agent.execute(
            {
                "operation": "execute",
                "query": "INSERT INTO test (value) VALUES (?)",
                "params": ["test_value"],
            },
            context,
        )
        assert result["affected_rows"] == 1

        # Query data
        result = await agent.execute(
            {
                "operation": "query",
                "query": "SELECT * FROM test",
            },
            context,
        )
        assert result["row_count"] == 1
        assert result["data"][0]["value"] == "test_value"

    @pytest.mark.asyncio
    async def test_get_tables(self, agent, context):
        """Test getting table list."""
        # Create some tables
        await agent.execute(
            {"operation": "execute", "query": "CREATE TABLE table1 (id INTEGER)"},
            context,
        )
        await agent.execute(
            {"operation": "execute", "query": "CREATE TABLE table2 (id INTEGER)"},
            context,
        )

        result = await agent.execute(
            {"operation": "tables"},
            context,
        )
        assert "tables" in result
        assert "table1" in result["tables"]
        assert "table2" in result["tables"]

    @pytest.mark.asyncio
    async def test_get_schema(self, agent, context):
        """Test getting table schema."""
        await agent.execute(
            {
                "operation": "execute",
                "query": """
                    CREATE TABLE schema_test (
                        id INTEGER PRIMARY KEY,
                        name TEXT NOT NULL,
                        value REAL
                    )
                """,
            },
            context,
        )

        result = await agent.execute(
            {"operation": "schema", "table": "schema_test"},
            context,
        )
        assert result["table"] == "schema_test"
        assert len(result["columns"]) == 3

        column_names = [c["name"] for c in result["columns"]]
        assert "id" in column_names
        assert "name" in column_names
        assert "value" in column_names

    @pytest.mark.asyncio
    async def test_write_protection(self, context):
        """Test that write operations are blocked when allow_write is False."""
        config = DatabaseConfig(
            name="ReadOnly",
            db_type="sqlite",
            sqlite_path=":memory:",
            allow_write=False,
        )
        agent = DatabaseAgent(config)

        with pytest.raises(Exception) as exc_info:
            await agent.execute(
                {
                    "operation": "execute",
                    "query": "INSERT INTO test VALUES (1, 'test')",
                },
                context,
            )
        assert "nicht erlaubt" in str(exc_info.value).lower() or "not allowed" in str(exc_info.value).lower()

    @pytest.mark.asyncio
    async def test_ddl_protection(self, context):
        """Test that DDL operations are blocked when allow_ddl is False."""
        config = DatabaseConfig(
            name="NoDDL",
            db_type="sqlite",
            sqlite_path=":memory:",
            allow_write=True,
            allow_ddl=False,
        )
        agent = DatabaseAgent(config)

        with pytest.raises(Exception) as exc_info:
            await agent.execute(
                {
                    "operation": "execute",
                    "query": "CREATE TABLE test (id INTEGER)",
                },
                context,
            )
        assert "nicht erlaubt" in str(exc_info.value).lower() or "not allowed" in str(exc_info.value).lower()

    @pytest.mark.asyncio
    async def test_execute_many(self, agent, context):
        """Test executing multiple inserts."""
        await agent.execute(
            {"operation": "execute", "query": "CREATE TABLE bulk_test (id INTEGER, name TEXT)"},
            context,
        )

        result = await agent.execute(
            {
                "operation": "execute_many",
                "query": "INSERT INTO bulk_test (id, name) VALUES (?, ?)",
                "params": [(1, "one"), (2, "two"), (3, "three")],
            },
            context,
        )
        assert result["affected_rows"] == 3

    @pytest.mark.asyncio
    async def test_parameterized_query(self, agent, context):
        """Test parameterized queries."""
        await agent.execute(
            {"operation": "execute", "query": "CREATE TABLE params_test (id INTEGER, name TEXT)"},
            context,
        )
        await agent.execute(
            {
                "operation": "execute",
                "query": "INSERT INTO params_test (id, name) VALUES (?, ?)",
                "params": [1, "test"],
            },
            context,
        )

        result = await agent.execute(
            {
                "operation": "query",
                "query": "SELECT * FROM params_test WHERE id = ?",
                "params": [1],
            },
            context,
        )
        assert result["row_count"] == 1
        assert result["data"][0]["name"] == "test"


# =============================================================================
# API AGENT TESTS
# =============================================================================

class TestAPIConfig:
    """Tests for APIConfig."""

    def test_default_config(self):
        """Test default configuration."""
        config = APIConfig(name="TestAPI")
        assert config.timeout_seconds == 30
        assert config.max_retries == 3
        assert "Content-Type" in config.default_headers

    def test_auth_config(self):
        """Test authentication configuration."""
        config = APIConfig(
            name="TestAPI",
            base_url="https://api.example.com",
            auth_type="bearer",
            auth_token="test-token",
        )
        assert config.auth_type == "bearer"
        assert config.auth_token == "test-token"


class TestAPIAgent:
    """Tests for APIAgent."""

    @pytest.fixture
    def agent(self):
        """Create a test agent."""
        config = APIConfig(
            name="TestAPI",
            base_url="https://api.example.com",
        )
        return APIAgent(config)

    @pytest.fixture
    def context(self):
        """Create test context."""
        return AgentContext(
            agent_id="test-agent",
            execution_id="test-exec",
        )

    def test_create_agent(self, agent):
        """Test creating agent."""
        assert agent.agent_type == "api"
        assert agent.version == "1.0"

    def test_build_url(self, agent):
        """Test URL building."""
        assert agent._build_url("/users") == "https://api.example.com/users"
        assert agent._build_url("users") == "https://api.example.com/users"
        assert agent._build_url("https://other.com/api") == "https://other.com/api"

    def test_get_headers_bearer(self):
        """Test Bearer token authentication."""
        config = APIConfig(
            name="TestAPI",
            auth_type="bearer",
            auth_token="test-token",
        )
        agent = APIAgent(config)
        headers = agent._get_headers()
        assert headers["Authorization"] == "Bearer test-token"

    def test_get_headers_api_key(self):
        """Test API key authentication."""
        config = APIConfig(
            name="TestAPI",
            auth_type="api_key",
            auth_token="my-api-key",
            auth_header="X-API-Key",
        )
        agent = APIAgent(config)
        headers = agent._get_headers()
        assert headers["X-API-Key"] == "my-api-key"

    def test_get_headers_extra(self):
        """Test extra headers are added."""
        config = APIConfig(name="TestAPI")
        agent = APIAgent(config)
        headers = agent._get_headers({"X-Custom": "value"})
        assert headers["X-Custom"] == "value"

    @pytest.mark.asyncio
    async def test_rest_get_mock(self, agent, context):
        """Test REST GET request with mock."""
        mock_response = AsyncMock()
        mock_response.status = 200
        mock_response.headers = {"Content-Type": "application/json"}
        mock_response.json = AsyncMock(return_value={"id": 1, "name": "test"})
        mock_response.url = "https://api.example.com/users/1"

        with patch("aiohttp.ClientSession") as mock_session:
            mock_context = AsyncMock()
            mock_context.__aenter__ = AsyncMock(return_value=mock_response)
            mock_context.__aexit__ = AsyncMock(return_value=None)

            session_instance = AsyncMock()
            session_instance.request = MagicMock(return_value=mock_context)
            session_instance.__aenter__ = AsyncMock(return_value=session_instance)
            session_instance.__aexit__ = AsyncMock(return_value=None)
            mock_session.return_value = session_instance

            result = await agent.execute(
                {
                    "type": "rest",
                    "method": "GET",
                    "endpoint": "/users/1",
                },
                context,
            )

            assert result["status_code"] == 200
            assert result["success"] is True
            assert result["data"]["id"] == 1

    @pytest.mark.asyncio
    async def test_rest_post_mock(self, agent, context):
        """Test REST POST request with mock."""
        mock_response = AsyncMock()
        mock_response.status = 201
        mock_response.headers = {"Content-Type": "application/json"}
        mock_response.json = AsyncMock(return_value={"id": 2, "name": "new"})
        mock_response.url = "https://api.example.com/users"

        with patch("aiohttp.ClientSession") as mock_session:
            mock_context = AsyncMock()
            mock_context.__aenter__ = AsyncMock(return_value=mock_response)
            mock_context.__aexit__ = AsyncMock(return_value=None)

            session_instance = AsyncMock()
            session_instance.request = MagicMock(return_value=mock_context)
            session_instance.__aenter__ = AsyncMock(return_value=session_instance)
            session_instance.__aexit__ = AsyncMock(return_value=None)
            mock_session.return_value = session_instance

            result = await agent.execute(
                {
                    "type": "rest",
                    "method": "POST",
                    "endpoint": "/users",
                    "body": {"name": "new"},
                },
                context,
            )

            assert result["status_code"] == 201
            assert result["success"] is True

    @pytest.mark.asyncio
    async def test_graphql_mock(self, agent, context):
        """Test GraphQL request with mock."""
        mock_response = AsyncMock()
        mock_response.status = 200
        mock_response.json = AsyncMock(return_value={
            "data": {"user": {"id": "1", "name": "Test"}},
            "errors": [],
        })
        mock_response.url = "https://api.example.com/graphql"

        with patch("aiohttp.ClientSession") as mock_session:
            mock_context = AsyncMock()
            mock_context.__aenter__ = AsyncMock(return_value=mock_response)
            mock_context.__aexit__ = AsyncMock(return_value=None)

            session_instance = AsyncMock()
            session_instance.post = MagicMock(return_value=mock_context)
            session_instance.__aenter__ = AsyncMock(return_value=session_instance)
            session_instance.__aexit__ = AsyncMock(return_value=None)
            mock_session.return_value = session_instance

            result = await agent.execute(
                {
                    "type": "graphql",
                    "query": "query { user(id: 1) { id name } }",
                },
                context,
            )

            assert result["success"] is True
            assert result["data"]["user"]["id"] == "1"


# =============================================================================
# WEBHOOK AGENT TESTS
# =============================================================================

class TestWebhookAgent:
    """Tests for WebhookAgent."""

    @pytest.fixture
    def agent(self):
        """Create a test agent."""
        config = APIConfig(
            name="TestWebhook",
            base_url="https://webhook.example.com/hook",
        )
        return WebhookAgent(config)

    @pytest.fixture
    def context(self):
        """Create test context."""
        return AgentContext(
            agent_id="test-agent",
            execution_id="test-exec",
        )

    def test_create_agent(self, agent):
        """Test creating agent."""
        assert agent.agent_type == "webhook"
        assert agent.version == "1.0"

    @pytest.mark.asyncio
    async def test_send_webhook_mock(self, agent, context):
        """Test sending webhook with mock."""
        mock_response = AsyncMock()
        mock_response.status = 200
        mock_response.json = AsyncMock(return_value={"received": True})
        mock_response.url = "https://webhook.example.com/hook"

        with patch("aiohttp.ClientSession") as mock_session:
            mock_context = AsyncMock()
            mock_context.__aenter__ = AsyncMock(return_value=mock_response)
            mock_context.__aexit__ = AsyncMock(return_value=None)

            session_instance = AsyncMock()
            session_instance.post = MagicMock(return_value=mock_context)
            session_instance.__aenter__ = AsyncMock(return_value=session_instance)
            session_instance.__aexit__ = AsyncMock(return_value=None)
            mock_session.return_value = session_instance

            result = await agent.execute(
                {
                    "payload": {"event": "test", "data": {"key": "value"}},
                },
                context,
            )

            assert result["status_code"] == 200
            assert result["success"] is True

    @pytest.mark.asyncio
    async def test_webhook_with_signature(self, agent, context):
        """Test webhook with signature."""
        mock_response = AsyncMock()
        mock_response.status = 200
        mock_response.json = AsyncMock(return_value={"ok": True})
        mock_response.url = "https://webhook.example.com/hook"

        with patch("aiohttp.ClientSession") as mock_session:
            mock_context = AsyncMock()
            mock_context.__aenter__ = AsyncMock(return_value=mock_response)
            mock_context.__aexit__ = AsyncMock(return_value=None)

            session_instance = AsyncMock()
            session_instance.post = MagicMock(return_value=mock_context)
            session_instance.__aenter__ = AsyncMock(return_value=session_instance)
            session_instance.__aexit__ = AsyncMock(return_value=None)
            mock_session.return_value = session_instance

            result = await agent.execute(
                {
                    "payload": {"event": "signed_test"},
                    "secret": "my-secret-key",
                },
                context,
            )

            assert result["success"] is True


# =============================================================================
# INTEGRATION TESTS
# =============================================================================

class TestAgentIntegration:
    """Integration tests for Database and API agents."""

    @pytest.mark.asyncio
    async def test_database_full_workflow(self):
        """Test complete database workflow."""
        config = DatabaseConfig(
            name="Integration",
            db_type="sqlite",
            sqlite_path=":memory:",
            allow_write=True,
            allow_ddl=True,
        )
        agent = DatabaseAgent(config)
        context = AgentContext(agent_id="int-test", execution_id="int-exec")

        # Create table
        await agent.execute(
            {
                "operation": "execute",
                "query": """
                    CREATE TABLE products (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        name TEXT NOT NULL,
                        price REAL
                    )
                """,
            },
            context,
        )

        # Insert products
        await agent.execute(
            {
                "operation": "execute_many",
                "query": "INSERT INTO products (name, price) VALUES (?, ?)",
                "params": [
                    ("Widget", 9.99),
                    ("Gadget", 19.99),
                    ("Doodad", 29.99),
                ],
            },
            context,
        )

        # Query products
        result = await agent.execute(
            {
                "operation": "query",
                "query": "SELECT * FROM products WHERE price > ?",
                "params": [15.0],
            },
            context,
        )

        assert result["row_count"] == 2
        names = [r["name"] for r in result["data"]]
        assert "Gadget" in names
        assert "Doodad" in names

        # Update product
        await agent.execute(
            {
                "operation": "execute",
                "query": "UPDATE products SET price = ? WHERE name = ?",
                "params": [24.99, "Gadget"],
            },
            context,
        )

        # Verify update
        result = await agent.execute(
            {
                "operation": "query",
                "query": "SELECT price FROM products WHERE name = ?",
                "params": ["Gadget"],
            },
            context,
        )
        assert result["data"][0]["price"] == 24.99

    @pytest.mark.asyncio
    async def test_agent_registry_contains_new_agents(self):
        """Test that new agents are registered."""
        from scio.agents.registry import AgentRegistry

        # Ensure builtins are registered
        AgentRegistry.register_builtins()

        agent_types = AgentRegistry.list_types()

        assert "database" in agent_types
        assert "api" in agent_types
        assert "webhook" in agent_types


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
