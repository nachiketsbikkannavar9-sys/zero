"""Snowflake Client for DataMind.

Connects to Snowflake using credentials from environment variables / .env,
enforces session-level statement timeouts, and returns query results as a
pandas DataFrame along with Snowflake query ID (sfqid) and elapsed execution time.
Errors are explicitly surfaced and never swallowed.
"""

from __future__ import annotations

import os
import time
from dataclasses import dataclass
from typing import Any, Dict, Optional
import pandas as pd
from dotenv import load_dotenv

import snowflake.connector
from snowflake.connector.errors import Error as SnowflakeError

# Load environment variables from .env if present
load_dotenv()


class SnowflakeConfigurationError(Exception):
    """Raised when required Snowflake environment configuration is missing."""
    pass


class SnowflakeExecutionError(Exception):
    """Raised when query execution on Snowflake fails."""
    def __init__(self, message: str, query_id: Optional[str] = None, sql: Optional[str] = None):
        super().__init__(message)
        self.query_id = query_id
        self.sql = sql


@dataclass
class SnowflakeQueryResult:
    """Represents the execution outcome of a Snowflake query."""
    df: pd.DataFrame
    query_id: str
    elapsed_seconds: float
    row_count: int
    sql: str


class SnowflakeClient:
    """Manages connections and executions against Snowflake."""

    def __init__(
        self,
        account: Optional[str] = None,
        user: Optional[str] = None,
        password: Optional[str] = None,
        warehouse: Optional[str] = None,
        database: Optional[str] = None,
        schema: Optional[str] = None,
        role: Optional[str] = None,
        authenticator: Optional[str] = None,
        private_key_path: Optional[str] = None,
        query_timeout_s: Optional[int] = None,
    ):
        self.account = account or os.getenv("SNOWFLAKE_ACCOUNT")
        self.user = user or os.getenv("SNOWFLAKE_USER")
        self.password = password or os.getenv("SNOWFLAKE_PASSWORD")
        self.warehouse = warehouse or os.getenv("SNOWFLAKE_WAREHOUSE", "DATAMIND_WH")
        self.database = database or os.getenv("SNOWFLAKE_DATABASE", "SNOWFLAKE_SAMPLE_DATA")
        self.schema = schema or os.getenv("SNOWFLAKE_SCHEMA", "TPCH_SF1")
        self.role = role or os.getenv("SNOWFLAKE_ROLE", "DATAMIND_READONLY")
        self.authenticator = authenticator or os.getenv("SNOWFLAKE_AUTHENTICATOR")
        self.private_key_path = private_key_path or os.getenv("SNOWFLAKE_PRIVATE_KEY_PATH")

        timeout_str = os.getenv("DATAMIND_QUERY_TIMEOUT_S", "30")
        self.query_timeout_s = query_timeout_s if query_timeout_s is not None else int(timeout_str)

    def validate_config(self) -> None:
        """Ensures minimal required connection parameters are present."""
        missing = []
        if not self.account:
            missing.append("SNOWFLAKE_ACCOUNT")
        if not self.user:
            missing.append("SNOWFLAKE_USER")
        if not self.password and not self.authenticator and not self.private_key_path:
            missing.append("SNOWFLAKE_PASSWORD (or SNOWFLAKE_AUTHENTICATOR/SNOWFLAKE_PRIVATE_KEY_PATH)")

        if missing:
            raise SnowflakeConfigurationError(
                f"Missing required Snowflake configuration: {', '.join(missing)}. "
                f"Please define them in your environment or .env file."
            )

    def get_connection(self) -> snowflake.connector.SnowflakeConnection:
        """Establishes and returns a new Snowflake connection."""
        self.validate_config()

        conn_params: Dict[str, Any] = {
            "account": self.account,
            "user": self.user,
            "warehouse": self.warehouse,
            "database": self.database,
            "schema": self.schema,
            "session_parameters": {
                "STATEMENT_TIMEOUT_IN_SECONDS": self.query_timeout_s,
            },
        }

        if self.role:
            conn_params["role"] = self.role

        if self.authenticator:
            conn_params["authenticator"] = self.authenticator
        elif self.private_key_path:
            from cryptography.hazmat.backends import default_backend
            from cryptography.hazmat.primitives import serialization

            passphrase = os.getenv("SNOWFLAKE_PRIVATE_KEY_PASSPHRASE")
            passphrase_bytes = passphrase.encode() if passphrase else None

            with open(self.private_key_path, "rb") as key_file:
                p_key = serialization.load_pem_private_key(
                    key_file.read(),
                    password=passphrase_bytes,
                    backend=default_backend(),
                )
            pkb = p_key.private_bytes(
                encoding=serialization.Encoding.DER,
                format=serialization.PrivateFormat.PKCS8,
                encryption_algorithm=serialization.NoEncryption(),
            )
            conn_params["private_key"] = pkb
        else:
            conn_params["password"] = self.password

        try:
            return snowflake.connector.connect(**conn_params)
        except SnowflakeError as exc:
            raise SnowflakeExecutionError(f"Failed to connect to Snowflake: {exc}") from exc

    def execute_query(self, sql: str, params: Optional[Dict[str, Any]] = None) -> SnowflakeQueryResult:
        """Executes a SQL query on Snowflake and returns a SnowflakeQueryResult.

        Sets STATEMENT_TIMEOUT_IN_SECONDS, records elapsed time, and captures cursor.sfqid.
        Never swallows failures.
        """
        start_time = time.perf_counter()
        conn = None
        cursor = None
        sfqid = ""

        try:
            conn = self.get_connection()
            cursor = conn.cursor()
            cursor.execute(sql, params)
            sfqid = cursor.sfqid or ""

            try:
                df = cursor.fetch_pandas_all()
            except Exception:
                # Fallback to standard fetchall if fetch_pandas_all is not available
                rows = cursor.fetchall()
                cols = [desc[0] for desc in cursor.description] if cursor.description else []
                df = pd.DataFrame(rows, columns=cols)

            elapsed = time.perf_counter() - start_time
            return SnowflakeQueryResult(
                df=df,
                query_id=sfqid,
                elapsed_seconds=elapsed,
                row_count=len(df),
                sql=sql,
            )
        except SnowflakeConfigurationError:
            raise
        except SnowflakeError as exc:
            sfqid = getattr(cursor, "sfqid", "") or sfqid
            elapsed = time.perf_counter() - start_time
            raise SnowflakeExecutionError(
                message=f"Snowflake query execution failed: {exc} (Query ID: {sfqid})",
                query_id=sfqid,
                sql=sql,
            ) from exc
        except Exception as exc:
            raise SnowflakeExecutionError(
                message=f"Unexpected error executing Snowflake query: {exc}",
                query_id=sfqid,
                sql=sql,
            ) from exc
        finally:
            if cursor:
                try:
                    cursor.close()
                except Exception:
                    pass
            if conn:
                try:
                    conn.close()
                except Exception:
                    pass
