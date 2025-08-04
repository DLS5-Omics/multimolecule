# MultiMolecule
# Copyright (C) 2024-Present  MultiMolecule

# This file is part of MultiMolecule.

# MultiMolecule is free software: you can redistribute it and/or modify
# it under the terms of the GNU Affero General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# any later version.

# MultiMolecule is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU Affero General Public License for more details.

# You should have received a copy of the GNU Affero General Public License
# along with this program.  If not, see <http://www.gnu.org/licenses/>.

# For additional terms and clarifications, please refer to our License FAQ at:
# <https://multimolecule.danling.org/about/license-faq>.

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

import pandas as pd
from sqlalchemy.exc import OperationalError

from multimolecule.datasets.database_utils import (
    DatabaseConfig,
    DatabaseConnection,
    close_connections,
    get_config_path,
    get_connection,
    query,
    query_dataset,
)


class TestDatabaseConfig(unittest.TestCase):
    """Test DatabaseConfig functionality."""

    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()
        self.config_path = Path(self.temp_dir) / "test_database.yaml"

    def test_config_creation_with_all_fields(self):
        """Test creating config with all required fields."""
        config_data = {
            "dialect": "postgresql",
            "driver": "psycopg2",
            "host": "localhost",
            "database": "testdb",
            "user": "testuser",
            "password": "testpass",
            "port": 5432,
        }

        import yaml

        with open(self.config_path, "w") as f:
            yaml.dump(config_data, f)

        config = DatabaseConfig.from_yaml(str(self.config_path))
        self.assertEqual(config.dialect, "postgresql")
        self.assertEqual(config.host, "localhost")
        self.assertEqual(config.database, "testdb")
        self.assertEqual(config.user, "testuser")
        self.assertEqual(config.password, "testpass")
        self.assertEqual(config.port, 5432)

    def test_config_with_empty_password(self):
        """Test config with empty password (public databases)."""
        config_data = {
            "dialect": "mysql",
            "driver": "pymysql",
            "host": "public-db.example.com",
            "database": "publicdb",
            "user": "readonly",
            "password": "",
            "port": 3306,
        }

        import yaml

        with open(self.config_path, "w") as f:
            yaml.dump(config_data, f)

        config = DatabaseConfig.from_yaml(str(self.config_path))
        url = config.build_url()
        expected = "mysql+pymysql://readonly:@public-db.example.com:3306/publicdb"
        self.assertEqual(url, expected)

    def test_build_url_postgresql(self):
        """Test URL building for PostgreSQL."""
        config = DatabaseConfig(
            dialect="postgresql",
            driver="psycopg2",
            host="localhost",
            database="testdb",
            user="user",
            password="pass",
            port=5432,
        )

        url = config.build_url()
        expected = "postgresql+psycopg2://user:pass@localhost:5432/testdb"
        self.assertEqual(url, expected)

    def test_build_url_mysql(self):
        """Test URL building for MySQL."""
        config = DatabaseConfig(
            dialect="mysql",
            driver="pymysql",
            host="mysql.example.com",
            database="mydb",
            user="myuser",
            password="mypass",
            port=3306,
        )

        url = config.build_url()
        expected = "mysql+pymysql://myuser:mypass@mysql.example.com:3306/mydb"
        self.assertEqual(url, expected)

    def test_build_url_missing_required_fields(self):
        """Test URL building fails with missing required fields."""
        config = DatabaseConfig(
            dialect="postgresql",
            host="localhost",
        )

        with self.assertRaises(ValueError) as cm:
            config.build_url()

        self.assertIn("Missing database config fields", str(cm.exception))
        self.assertIn("database", str(cm.exception))
        self.assertIn("user", str(cm.exception))

    def test_direct_url_override(self):
        """Test that direct URL overrides individual fields."""
        config = DatabaseConfig(
            url="postgresql://direct:url@host:5432/db",
            dialect="mysql",
            host="ignored",
        )

        url = config.build_url()
        self.assertEqual(url, "postgresql://direct:url@host:5432/db")

    def test_pool_config_defaults(self):
        """Test default pool configuration."""
        config = DatabaseConfig(dialect="postgresql", host="localhost", database="db", user="user")
        pool_config = config.get_pool_config()

        self.assertEqual(pool_config["pool_size"], 5)
        self.assertEqual(pool_config["max_overflow"], 10)
        self.assertTrue(pool_config["pool_pre_ping"])

    def test_pool_config_override(self):
        """Test pool configuration override."""
        config = DatabaseConfig(
            dialect="postgresql",
            host="localhost",
            database="db",
            user="user",
            pool={"pool_size": 10, "max_overflow": 20},
        )

        pool_config = config.get_pool_config()
        self.assertEqual(pool_config["pool_size"], 10)
        self.assertEqual(pool_config["max_overflow"], 20)


class TestDatabaseConnection(unittest.TestCase):
    """Test DatabaseConnection functionality."""

    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()
        self.config_path = Path(self.temp_dir) / "test_database.yaml"

        config_data = {"dialect": "sqlite", "host": "", "database": ":memory:", "user": "", "password": ""}

        import yaml

        with open(self.config_path, "w") as f:
            yaml.dump(config_data, f)

    def test_connection_creation(self):
        """Test DatabaseConnection creation."""
        conn = DatabaseConnection(str(self.config_path))
        self.assertIsNotNone(conn.config)
        self.assertEqual(conn.config.dialect, "sqlite")

    def test_normalize_query(self):
        """Test query normalization."""
        query = "SELECT  *   FROM   table   WHERE  id = 1  ;"
        normalized = DatabaseConnection._normalize_query(query)
        expected = "SELECT * FROM table WHERE id = 1"
        self.assertEqual(normalized, expected)

        query = "SELECT * FROM table;"
        normalized = DatabaseConnection._normalize_query(query)
        expected = "SELECT * FROM table"
        self.assertEqual(normalized, expected)

        query = "SELECT * FROM table   "
        normalized = DatabaseConnection._normalize_query(query)
        expected = "SELECT * FROM table"
        self.assertEqual(normalized, expected)

    def test_extract_limit(self):
        """Test LIMIT clause extraction."""
        conn = DatabaseConnection(str(self.config_path))

        query = "SELECT * FROM table LIMIT 100"
        base_query, limit = conn._extract_limit(query)
        self.assertEqual(base_query, "SELECT * FROM table")
        self.assertEqual(limit, 100)

        query = "SELECT * FROM table LIMIT 50 OFFSET 10"
        base_query, limit = conn._extract_limit(query)
        self.assertEqual(base_query, "SELECT * FROM table")
        self.assertEqual(limit, 50)

        query = "SELECT * FROM table"
        base_query, limit = conn._extract_limit(query)
        self.assertEqual(base_query, "SELECT * FROM table")
        self.assertIsNone(limit)

    def test_extract_limit_negative(self):
        """Test LIMIT extraction with negative value should raise error."""
        conn = DatabaseConnection(str(self.config_path))

        query = "SELECT * FROM table LIMIT -1"
        with self.assertRaises(ValueError) as cm:
            conn._extract_limit(query)

        self.assertIn("LIMIT cannot be negative", str(cm.exception))

    @patch("multimolecule.datasets.database_utils.create_engine")
    def test_retry_mechanism(self, mock_create_engine):
        """Test retry mechanism for database operations."""
        conn = DatabaseConnection(str(self.config_path))

        mock_engine = Mock()
        mock_create_engine.return_value = mock_engine

        call_count = 0

        def failing_function():
            nonlocal call_count
            call_count += 1
            if call_count <= 2:
                raise OperationalError("Connection failed", None, None)
            return "success"

        result = conn._retry(failing_function)
        self.assertEqual(result, "success")
        self.assertEqual(call_count, 3)

    @patch("multimolecule.datasets.database_utils.create_engine")
    def test_retry_exhausted(self, mock_create_engine):
        """Test retry mechanism when all retries are exhausted."""
        conn = DatabaseConnection(str(self.config_path))
        conn._max_retries = 2

        mock_engine = Mock()
        mock_create_engine.return_value = mock_engine

        def always_failing_function():
            raise OperationalError("Always fails", None, None)

        with self.assertRaises(OperationalError):
            conn._retry(always_failing_function)


class TestQueryFunctions(unittest.TestCase):
    """Test high-level query functions."""

    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()
        self.config_path = Path(self.temp_dir) / "test_database.yaml"

        config_data = {"dialect": "sqlite", "host": "", "database": ":memory:", "user": "", "password": ""}

        import yaml

        with open(self.config_path, "w") as f:
            yaml.dump(config_data, f)

    @patch("multimolecule.datasets.database_utils.get_connection")
    def test_query_function(self, mock_get_connection):
        """Test the main query function."""
        mock_conn = Mock()
        mock_df = pd.DataFrame({"id": [1, 2, 3], "name": ["A", "B", "C"]})
        mock_conn.query.return_value = mock_df
        mock_get_connection.return_value = mock_conn

        result = query(str(self.config_path), "SELECT * FROM test")

        self.assertIsInstance(result, pd.DataFrame)
        mock_get_connection.assert_called_once_with(str(self.config_path))
        mock_conn.query.assert_called_once()

    @patch("multimolecule.datasets.database_utils.get_config_path")
    @patch("multimolecule.datasets.database_utils.query")
    def test_query_dataset_function(self, mock_query, mock_get_config_path):
        """Test the query_dataset convenience function."""
        mock_get_config_path.return_value = str(self.config_path)
        mock_df = pd.DataFrame({"result": [42]})
        mock_query.return_value = mock_df

        result = query_dataset("test_dataset", "SELECT 42 as result")

        self.assertIsInstance(result, pd.DataFrame)
        mock_get_config_path.assert_called_once_with("test_dataset", None)
        mock_query.assert_called_once()

    def test_get_config_path(self):
        """Test config path generation."""
        path = get_config_path("test_dataset")
        self.assertTrue(path.endswith("test_dataset/database.yaml"))

        custom_root = "/custom/path"
        path = get_config_path("test_dataset", custom_root)
        expected = "/custom/path/test_dataset/database.yaml"
        self.assertEqual(path, expected)

    def test_connection_pooling(self):
        """Test connection pooling works correctly."""
        close_connections()

        conn1 = get_connection(str(self.config_path))
        conn2 = get_connection(str(self.config_path))

        self.assertIs(conn1, conn2)

        config_path2 = Path(self.temp_dir) / "test_database2.yaml"
        config_data = {"dialect": "sqlite", "host": "", "database": ":memory:", "user": "", "password": ""}

        import yaml

        with open(config_path2, "w") as f:
            yaml.dump(config_data, f)

        conn3 = get_connection(str(config_path2))
        self.assertIsNot(conn1, conn3)

    def test_close_connections(self):
        """Test connection cleanup."""
        conn1 = get_connection(str(self.config_path))

        close_connections()

        conn2 = get_connection(str(self.config_path))
        self.assertIsNot(conn1, conn2)


class TestErrorHandling(unittest.TestCase):
    """Test error handling and edge cases."""

    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()
        self.config_path = Path(self.temp_dir) / "test_database.yaml"

    def test_empty_query_validation(self):
        """Test validation of empty queries."""
        config_data = {"dialect": "sqlite", "host": "", "database": ":memory:", "user": "", "password": ""}

        import yaml

        with open(self.config_path, "w") as f:
            yaml.dump(config_data, f)

        conn = DatabaseConnection(str(self.config_path))

        with self.assertRaises(ValueError) as cm:
            conn.query("")

        self.assertIn("Query cannot be empty", str(cm.exception))

    def test_invalid_batch_size(self):
        """Test validation of batch size."""
        config_data = {"dialect": "sqlite", "host": "", "database": ":memory:", "user": "", "password": ""}

        import yaml

        with open(self.config_path, "w") as f:
            yaml.dump(config_data, f)

        conn = DatabaseConnection(str(self.config_path))

        with self.assertRaises(ValueError) as cm:
            conn.query("SELECT 1", batch_size=0)

        self.assertIn("Batch size must be positive", str(cm.exception))

    def test_missing_config_file(self):
        """Test handling of missing config file."""
        non_existent_path = "/non/existent/path.yaml"

        with self.assertRaises(FileNotFoundError):
            DatabaseConnection(non_existent_path)


if __name__ == "__main__":
    unittest.main()
