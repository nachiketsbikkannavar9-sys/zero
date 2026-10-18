"""Query Planner and SQL Compiler for DataMind.

Defines the Pydantic QueryPlan schema, validates plans against the semantic catalog,
and deterministically compiles plans into safe, read-only Snowflake SQL.
Contains zero LLM calls; completely deterministic and unit-testable.
"""

from __future__ import annotations

import os
from typing import Any, List, Literal, Optional, Set, Union
from pydantic import BaseModel, Field, field_validator

from src.semantic_catalog import DatasetConfig, get_catalog


class FilterCondition(BaseModel):
    dimension: str
    op: Literal["=", "!=", ">", ">=", "<", "<=", "in", "between", "contains"]
    value: Union[str, int, float, List[Union[str, int, float]]]


class OrderBy(BaseModel):
    field: str
    direction: Literal["asc", "desc", "ASC", "DESC"] = "desc"


class QueryPlan(BaseModel):
    status: Literal["ok", "needs_clarification", "unsupported"] = "ok"
    metric: Optional[str] = None
    dimensions: List[str] = Field(default_factory=list)
    filters: List[FilterCondition] = Field(default_factory=list)
    time_grain: Optional[Literal["year", "quarter", "month", "day"]] = None
    order_by: Optional[OrderBy] = None
    limit: int = Field(default=10, ge=1)
    clarification_question: Optional[str] = None
    unsupported_reason: Optional[str] = None
    assumptions: List[str] = Field(default_factory=list)

    @field_validator("metric", mode="before")
    def normalize_metric(cls, v: Any) -> Optional[str]:
        if isinstance(v, str):
            v_strip = v.strip().lower()
            return v_strip if v_strip and v_strip != "null" and v_strip != "none" else None
        return v

    @field_validator("dimensions", mode="before")
    def normalize_dimensions(cls, v: Any) -> List[str]:
        if isinstance(v, list):
            return [str(d).strip().lower() for d in v if str(d).strip()]
        return []

    @field_validator("time_grain", mode="before")
    def normalize_time_grain(cls, v: Any) -> Optional[str]:
        if isinstance(v, str):
            v_norm = v.strip().lower()
            return v_norm if v_norm in ["year", "quarter", "month", "day"] else None
        return None


class PlanValidationError(Exception):
    """Raised when a query plan fails semantic catalog validation."""
    pass


class QueryCompilationResult(BaseModel):
    sql: str
    plan: QueryPlan
    assumptions: List[str]
    tables_used: List[str]


class QueryPlanner:
    """Validates QueryPlan instances and compiles them to deterministic SQL."""

    def __init__(self, dataset_config: Optional[DatasetConfig] = None):
        if dataset_config is None:
            self.dataset = get_catalog().get_dataset()
        else:
            self.dataset = dataset_config
        self.max_rows = int(os.getenv("DATAMIND_MAX_ROWS", "500"))

    def validate_plan(self, plan: QueryPlan, question_text: str = "") -> QueryPlan:
        """Validates the plan against catalog metrics, dimensions, and constraints."""
        # 1. If plan is already non-ok, preserve status
        if plan.status in ("needs_clarification", "unsupported"):
            return plan

        # Check for unavailable concepts in question or plan
        combined_text = f"{question_text} {plan.metric or ''} {' '.join(plan.dimensions)}".lower()
        for concept, reason in self.dataset.unavailable_concepts.items():
            if concept in combined_text:
                plan.status = "unsupported"
                plan.unsupported_reason = f"Concept '{concept}' is unavailable: {reason}"
                return plan

        # Check vague questions without explicit metrics (e.g. 'best', 'top', 'performance')
        vague_indicators = ["best", "top performing", "performance", "ranking", "most successful"]
        if not plan.metric and any(w in question_text.lower() for w in vague_indicators):
            plan.status = "needs_clarification"
            available = ", ".join(self.dataset.metrics.keys())
            plan.clarification_question = (
                f"Which metric would you like to rank by? Available metrics in {self.dataset.name}: {available}."
            )
            return plan

        # 2. Metric validation
        if plan.metric:
            metric_cfg = self.dataset.get_metric(plan.metric)
            if not metric_cfg:
                # Check if it was an unavailable concept
                unavail = self.dataset.get_unavailable_concept_reason(plan.metric)
                if unavail:
                    plan.status = "unsupported"
                    plan.unsupported_reason = unavail
                    return plan
                plan.status = "unsupported"
                available = ", ".join(self.dataset.metrics.keys())
                plan.unsupported_reason = f"Unknown metric '{plan.metric}'. Available metrics: {available}."
                return plan
        elif not plan.dimensions and not plan.time_grain:
            plan.status = "needs_clarification"
            plan.clarification_question = "Please specify a metric or dimension to query."
            return plan

        # 3. Dimension validation
        for dim in plan.dimensions:
            dim_cfg = self.dataset.get_dimension(dim)
            if not dim_cfg:
                plan.status = "unsupported"
                available = ", ".join(self.dataset.dimensions.keys())
                plan.unsupported_reason = f"Unknown dimension '{dim}'. Available dimensions: {available}."
                return plan

        # 4. Filter validation
        for f in plan.filters:
            dim_name = f.dimension.lower()
            dim_cfg = self.dataset.get_dimension(dim_name)
            if not dim_cfg:
                plan.status = "unsupported"
                plan.unsupported_reason = f"Filter references unknown dimension '{dim_name}'."
                return plan

            # Check allowed values if defined
            if dim_cfg.allowed_values and f.op == "=" and isinstance(f.value, str):
                allowed_upper = [str(v).upper() for v in dim_cfg.allowed_values]
                if f.value.upper() not in allowed_upper:
                    plan.status = "needs_clarification"
                    plan.clarification_question = (
                        f"Value '{f.value}' for dimension '{dim_name}' is not recognized. "
                        f"Allowed values are: {', '.join(dim_cfg.allowed_values)}."
                    )
                    return plan

        # 5. Time grain validation
        if plan.time_grain:
            if not self.dataset.date_dimension:
                plan.status = "unsupported"
                plan.unsupported_reason = f"Dataset {self.dataset.name} does not have a configured date dimension."
                return plan
            if plan.time_grain not in self.dataset.date_dimension.allowed_grains:
                plan.status = "unsupported"
                plan.unsupported_reason = (
                    f"Time grain '{plan.time_grain}' is not allowed. "
                    f"Allowed grains: {', '.join(self.dataset.date_dimension.allowed_grains)}."
                )
                return plan

        # 6. Clamp limit
        if plan.limit <= 0 or plan.limit > self.max_rows:
            plan.assumptions.append(f"Clamped query limit to max allowed ({self.max_rows}).")
            plan.limit = self.max_rows

        return plan

    def _sanitize_literal(self, value: Union[str, int, float]) -> str:
        """Safely format and escape literal values for SQL injection prevention."""
        if isinstance(value, (int, float)):
            return str(value)
        # String escaping: replace ' with ''
        clean_str = str(value).replace("'", "''")
        return f"'{clean_str}'"

    def compile_sql(self, plan: QueryPlan) -> QueryCompilationResult:
        """Compiles a validated QueryPlan into Snowflake SQL."""
        if plan.status != "ok":
            raise PlanValidationError(
                f"Cannot compile plan with status '{plan.status}': "
                f"{plan.unsupported_reason or plan.clarification_question or 'Plan invalid'}"
            )

        db = self.dataset.database
        schema = self.dataset.schema_name

        select_clauses: List[str] = []
        group_by_indices: List[str] = []
        tables_needed: Set[str] = set()
        assumptions: List[str] = list(plan.assumptions)
        col_index = 1

        # 1. Time grain / date grouping
        if plan.time_grain:
            date_cfg = self.dataset.date_dimension
            if not date_cfg:
                raise PlanValidationError("Missing date dimension in dataset config.")
            tables_needed.add(date_cfg.table.upper())
            grain_str = plan.time_grain.upper()
            select_clauses.append(f"DATE_TRUNC('{grain_str}', {date_cfg.table}.{date_cfg.column}) AS {date_cfg.column}_{plan.time_grain}")
            group_by_indices.append(str(col_index))
            col_index += 1
            assumptions.append(f"Aggregated by {plan.time_grain} using {date_cfg.table}.{date_cfg.column}.")

        # 2. Dimensions
        for dim_name in plan.dimensions:
            dim_cfg = self.dataset.get_dimension(dim_name)
            if not dim_cfg:
                raise PlanValidationError(f"Unknown dimension '{dim_name}'.")
            tables_needed.add(dim_cfg.table.upper())
            select_clauses.append(f"{dim_cfg.expression} AS {dim_name.upper()}")
            group_by_indices.append(str(col_index))
            col_index += 1

        # 3. Metric
        metric_alias = "VALUE"
        if plan.metric:
            metric_cfg = self.dataset.get_metric(plan.metric)
            if not metric_cfg:
                raise PlanValidationError(f"Unknown metric '{plan.metric}'.")
            for t in metric_cfg.required_tables:
                tables_needed.add(t.upper())
            metric_alias = plan.metric.upper()
            select_clauses.append(f"{metric_cfg.expression} AS {metric_alias}")
            assumptions.append(f"Metric '{plan.metric}' computed as: {metric_cfg.expression}.")

        # 4. Filters & Where conditions
        where_clauses: List[str] = []
        for f in plan.filters:
            dim_cfg = self.dataset.get_dimension(f.dimension)
            if not dim_cfg:
                raise PlanValidationError(f"Filter references unknown dimension '{f.dimension}'.")
            tables_needed.add(dim_cfg.table.upper())
            expr = dim_cfg.expression

            if f.op == "contains":
                clean = str(f.value).replace("'", "''").upper()
                where_clauses.append(f"UPPER({expr}) ILIKE '%{clean}%'")
            elif f.op == "in":
                if isinstance(f.value, list):
                    vals = ", ".join(self._sanitize_literal(v) for v in f.value)
                    where_clauses.append(f"{expr} IN ({vals})")
                else:
                    where_clauses.append(f"{expr} = {self._sanitize_literal(f.value)}")
            elif f.op == "between" and isinstance(f.value, list) and len(f.value) == 2:
                v1 = self._sanitize_literal(f.value[0])
                v2 = self._sanitize_literal(f.value[1])
                where_clauses.append(f"{expr} BETWEEN {v1} AND {v2}")
            else:
                lit = self._sanitize_literal(f.value)
                where_clauses.append(f"{expr} {f.op} {lit}")

        if not select_clauses:
            raise PlanValidationError("Query plan must specify at least one metric or dimension.")

        # 5. Build FROM and JOINs
        sorted_tables = sorted(list(tables_needed))
        if not sorted_tables:
            sorted_tables = ["ORDERS"]

        # Base table
        if "LINEITEM" in sorted_tables:
            base_table = "LINEITEM"
        elif "ORDERS" in sorted_tables:
            base_table = "ORDERS"
        else:
            base_table = sorted_tables[0]

        from_clause = f"FROM {db}.{schema}.{base_table} AS {base_table}"
        joins = self.dataset.resolve_joins(tables_needed)

        # Build join sequence
        joined_tables = {base_table}
        join_sql_parts = []
        pending_joins = list(joins)

        # Join traversal
        while pending_joins:
            progress = False
            for edge in list(pending_joins):
                u = edge.from_table.upper()
                v = edge.to_table.upper()
                if u in joined_tables and v not in joined_tables:
                    join_sql_parts.append(f"JOIN {db}.{schema}.{v} AS {v} ON {edge.on}")
                    joined_tables.add(v)
                    pending_joins.remove(edge)
                    progress = True
                    break
                elif v in joined_tables and u not in joined_tables:
                    join_sql_parts.append(f"JOIN {db}.{schema}.{u} AS {u} ON {edge.on}")
                    joined_tables.add(u)
                    pending_joins.remove(edge)
                    progress = True
                    break
            if not progress and pending_joins:
                # If disconnected edges exist, join directly
                edge = pending_joins.pop(0)
                target = edge.to_table.upper() if edge.from_table.upper() in joined_tables else edge.from_table.upper()
                join_sql_parts.append(f"JOIN {db}.{schema}.{target} AS {target} ON {edge.on}")
                joined_tables.add(target)

        # Combine SQL parts
        sql_lines = [
            f"SELECT\n    " + ",\n    ".join(select_clauses),
            from_clause,
        ]
        if join_sql_parts:
            sql_lines.extend(join_sql_parts)

        if where_clauses:
            sql_lines.append("WHERE " + " AND ".join(where_clauses))

        if group_by_indices and plan.metric:
            sql_lines.append(f"GROUP BY {', '.join(group_by_indices)}")

        # Order by
        if plan.order_by:
            direction = plan.order_by.direction.upper()
            fld = plan.order_by.field.upper()
            sql_lines.append(f"ORDER BY {fld} {direction}")
        elif plan.metric and (plan.dimensions or plan.time_grain):
            # Default ranking
            sql_lines.append(f"ORDER BY {metric_alias} DESC")

        # Limit
        effective_limit = min(plan.limit or 10, self.max_rows)
        sql_lines.append(f"LIMIT {effective_limit}")

        final_sql = "\n".join(sql_lines)
        return QueryCompilationResult(
            sql=final_sql,
            plan=plan,
            assumptions=assumptions,
            tables_used=list(joined_tables),
        )
