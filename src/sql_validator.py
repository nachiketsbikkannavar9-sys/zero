"""SQL Validator for DataMind.

Enforces read-only safety using sqlglot AST parsing in the Snowflake dialect.
Validates:
1. Exactly one SQL statement.
2. Must strictly be a SELECT statement (rejects DML, DDL, admin commands, procedures).
3. Every table referenced exists in the catalog/schema.
4. Every column referenced exists in the referenced catalog tables or is a defined SELECT alias.
5. LIMIT clause is present and does not exceed the allowed max rows.
Returns (is_valid, failure_reason).
"""

from __future__ import annotations

import os
from typing import Optional, Set, Tuple
import sqlglot
from sqlglot import exp

from src.semantic_catalog import DatasetConfig, get_catalog


class SQLValidator:
    """Validates SQL statements against semantic catalog and security rules."""

    def __init__(self, dataset: Optional[DatasetConfig] = None, max_rows: Optional[int] = None):
        self.dataset = dataset or get_catalog().get_dataset()
        self.max_rows = max_rows or int(os.getenv("DATAMIND_MAX_ROWS", "500"))

    def validate(self, sql: str) -> Tuple[bool, str]:
        """Validates a SQL string. Returns (True, 'Valid') or (False, reason)."""
        if not sql or not sql.strip():
            return False, "Query string is empty."

        # Parse with sqlglot using Snowflake dialect
        try:
            statements = sqlglot.parse(sql, read="snowflake")
        except Exception as exc:
            return False, f"SQL syntax error or unparseable query: {exc}"

        if not statements:
            return False, "No valid SQL statements found."

        # Rule 1: Exactly one statement
        if len(statements) > 1:
            return False, f"Multiple statements detected ({len(statements)} statements). Only single statements are permitted."

        stmt = statements[0]

        # Rule 2: Must strictly be a SELECT statement
        # Reject DDL / DML / Admin AST types explicitly
        forbidden_types = (
            exp.Insert,
            exp.Update,
            exp.Delete,
            exp.Drop,
            exp.Create,
            exp.Alter,
            exp.Command,
            exp.Transaction,
            exp.Commit,
            exp.Rollback,
        )
        if isinstance(stmt, forbidden_types) or not isinstance(stmt, (exp.Select, exp.Union)):
            return False, f"Forbidden statement type '{type(stmt).__name__}'. Only read-only SELECT statements are allowed."

        # Also inspect entire AST for nested DML/DDL or multiple sub-commands
        for node in stmt.walk():
            if isinstance(node, forbidden_types):
                return False, f"Nested forbidden statement '{type(node).__name__}' found within query."

        # Rule 3: Tables must exist in the catalog
        known_tables = {t.upper() for t in self.dataset.tables.keys()}
        referenced_tables: Set[str] = set()

        for table_node in stmt.find_all(exp.Table):
            t_name = table_node.name.upper()
            if not t_name:
                continue
            if t_name not in known_tables:
                return False, (
                    f"Forbidden or unknown table '{t_name}'. "
                    f"Allowed tables in {self.dataset.name}: {', '.join(sorted(known_tables))}."
                )
            referenced_tables.add(t_name)

        if not referenced_tables:
            return False, "Query must reference at least one valid catalog table."

        # Rule 4: Column validation
        # Gather all valid columns across the dataset's referenced tables
        valid_columns: Set[str] = set()
        for t_name in referenced_tables:
            tbl_cfg = self.dataset.tables.get(t_name)
            if tbl_cfg:
                for c in tbl_cfg.columns.keys():
                    valid_columns.add(c.upper())

        # Also collect aliases defined in SELECT
        select_aliases: Set[str] = set()
        for alias_node in stmt.find_all(exp.Alias):
            if alias_node.alias:
                select_aliases.add(alias_node.alias.upper())

        # Also common pseudo-columns or special tokens
        allowed_extras = {"*", "NULL", "TRUE", "FALSE"}

        for col_node in stmt.find_all(exp.Column):
            c_name = col_node.name.upper()
            if not c_name or c_name in allowed_extras:
                continue
            if c_name not in valid_columns and c_name not in select_aliases:
                return False, (
                    f"Unknown or uncataloged column '{c_name}'. "
                    f"Columns must belong to referenced tables ({', '.join(sorted(referenced_tables))})."
                )

        # Rule 5: LIMIT clause enforcement
        limit_node = stmt.find(exp.Limit)
        if not limit_node or not limit_node.expression:
            return False, f"Missing required LIMIT clause. Every query must enforce LIMIT <= {self.max_rows}."

        try:
            limit_val = int(str(limit_node.expression).strip())
            if limit_val <= 0:
                return False, f"Invalid LIMIT value: {limit_val}. Must be >= 1."
            if limit_val > self.max_rows:
                return False, f"LIMIT {limit_val} exceeds maximum allowed limit of {self.max_rows}."
        except ValueError:
            return False, f"Unparseable non-integer LIMIT clause: '{limit_node.expression}'."

        return True, "Query is valid and safe."
