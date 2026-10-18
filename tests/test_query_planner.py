"""Unit tests for QueryPlanner (offline).

Verifies plan validation and deterministic SQL compilation:
- Ranking queries (metric + dimension + limit)
- Filtering queries (operators =, >=, in, contains)
- Join resolution (multi-hop graph joins)
- Time grouping (date_trunc with grain)
- Rejection of unknown metrics and dimensions
- Ambiguity detection (vague questions without metric)
- Unsupported concepts (profit, cost, prediction)
"""

import pytest
from src.query_planner import FilterCondition, OrderBy, PlanValidationError, QueryPlan, QueryPlanner
from src.semantic_catalog import get_catalog


@pytest.fixture
def planner():
    cat = get_catalog()
    ds = cat.get_dataset("tpch")
    return QueryPlanner(dataset_config=ds)


def test_ranking_compilation(planner):
    plan = QueryPlan(
        metric="revenue",
        dimensions=["part_name"],
        order_by=OrderBy(field="revenue", direction="desc"),
        limit=5,
    )
    validated = planner.validate_plan(plan)
    assert validated.status == "ok"

    compiled = planner.compile_sql(validated)
    sql = compiled.sql.upper()

    assert "SELECT" in sql
    assert "PART.P_NAME AS PART_NAME" in sql
    assert "SUM(LINEITEM.L_EXTENDEDPRICE * (1 - LINEITEM.L_DISCOUNT)) AS REVENUE" in sql
    assert "JOIN SNOWFLAKE_SAMPLE_DATA.TPCH_SF1.PART AS PART" in sql
    assert "ORDER BY REVENUE DESC" in sql
    assert "LIMIT 5" in sql
    assert any("revenue" in a.lower() for a in compiled.assumptions)


def test_filter_compilation(planner):
    plan = QueryPlan(
        metric="order_count",
        filters=[
            FilterCondition(dimension="order_priority", op="=", value="1-URGENT"),
            FilterCondition(dimension="market_segment", op="in", value=["BUILDING", "MACHINERY"]),
        ],
        limit=10,
    )
    validated = planner.validate_plan(plan)
    assert validated.status == "ok"

    compiled = planner.compile_sql(validated)
    sql = compiled.sql.upper()

    assert "ORDERS.O_ORDERPRIORITY = '1-URGENT'" in sql
    assert "CUSTOMER.C_MKTSEGMENT IN ('BUILDING', 'MACHINERY')" in sql
    assert "JOIN SNOWFLAKE_SAMPLE_DATA.TPCH_SF1.CUSTOMER AS CUSTOMER" in sql
    assert "LIMIT 10" in sql


def test_multi_hop_join_compilation(planner):
    plan = QueryPlan(
        metric="revenue",
        dimensions=["region"],
        limit=10,
    )
    validated = planner.validate_plan(plan)
    assert validated.status == "ok"

    compiled = planner.compile_sql(validated)
    sql = compiled.sql.upper()

    # LINEITEM -> SUPPLIER -> NATION -> REGION
    assert "LINEITEM" in sql
    assert "SUPPLIER" in sql
    assert "NATION" in sql
    assert "REGION" in sql
    assert "REGION.R_NAME AS REGION" in sql


def test_time_grouping_compilation(planner):
    plan = QueryPlan(
        metric="revenue",
        time_grain="month",
        filters=[
            FilterCondition(dimension="order_date", op=">=", value="1995-01-01"),
            FilterCondition(dimension="order_date", op="<=", value="1995-12-31"),
        ],
        limit=12,
    )
    validated = planner.validate_plan(plan)
    assert validated.status == "ok"

    compiled = planner.compile_sql(validated)
    sql = compiled.sql.upper()

    assert "DATE_TRUNC('MONTH', ORDERS.O_ORDERDATE)" in sql
    assert "ORDERS.O_ORDERDATE >= '1995-01-01'" in sql
    assert "GROUP BY 1" in sql
    assert "LIMIT 12" in sql


def test_rejects_unknown_metric(planner):
    plan = QueryPlan(metric="unknown_random_metric")
    validated = planner.validate_plan(plan)
    assert validated.status == "unsupported"
    assert "Unknown metric" in validated.unsupported_reason


def test_rejects_unknown_dimension(planner):
    plan = QueryPlan(metric="revenue", dimensions=["unknown_dimension"])
    validated = planner.validate_plan(plan)
    assert validated.status == "unsupported"
    assert "Unknown dimension" in validated.unsupported_reason


def test_detects_ambiguity_needs_clarification(planner):
    plan = QueryPlan(metric=None, dimensions=["part_name"])
    validated = planner.validate_plan(plan, question_text="Show me the top performing parts")
    assert validated.status == "needs_clarification"
    assert validated.clarification_question is not None


def test_detects_unavailable_concept_profit(planner):
    plan = QueryPlan(metric="revenue")
    validated = planner.validate_plan(plan, question_text="What is our total profit?")
    assert validated.status == "unsupported"
    assert "profit" in validated.unsupported_reason.lower()


def test_clamps_excessive_limit(planner):
    plan = QueryPlan(metric="revenue", limit=99999)
    validated = planner.validate_plan(plan)
    assert validated.limit == planner.max_rows
    assert any("Clamped" in a for a in validated.assumptions)
