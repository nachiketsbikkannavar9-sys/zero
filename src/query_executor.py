"""Query Executor for DataMind.

Executes only validated, read-only SQL queries on Snowflake.
Guarantees that unvalidated or unsafe SQL statements never reach Snowflake.
"""

from __future__ import annotations

from typing import Optional
from src.snowflake_client import SnowflakeClient, SnowflakeQueryResult
from src.sql_validator import SQLValidator


class UnsafeQueryError(Exception):
    """Raised when an unvalidated or prohibited SQL query attempts execution."""
    pass


class QueryExecutor:
    """Executes validated SQL statements against Snowflake."""

    def __init__(
        self,
        client: Optional[SnowflakeClient] = None,
        validator: Optional[SQLValidator] = None,
    ):
        self.client = client or SnowflakeClient()
        self.validator = validator or SQLValidator()

    def execute(self, sql: str) -> SnowflakeQueryResult:
        """Validates SQL statement and executes it on Snowflake.

        Raises UnsafeQueryError if validation fails.
        """
        is_valid, reason = self.validator.validate(sql)
        if not is_valid:
            raise UnsafeQueryError(f"Execution rejected: {reason}")

        return self.client.execute_query(sql)
