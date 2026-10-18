"""Failure cases test suite (offline).

Mocking Ollama and Snowflake ONLY in these offline tests to verify resilient error handling:
- Model down (OllamaConnectionError)
- Malformed JSON from model (ModelPlanningError)
- Empty query result from Snowflake (0 rows returned)
- Snowflake execution error (SnowflakeExecutionError with sfqid)
- Unsafe / destructive requests (blocked before execution)
"""

from unittest.mock import MagicMock
import pandas as pd
import pytest

from src.answer_generator import AnswerGenerator
from src.gemma_model import GemmaPlanner, ModelPlanningError, OllamaConnectionError
from src.pipeline import DataMindPipeline
from src.query_executor import QueryExecutor
from src.query_planner import QueryPlan, QueryPlanner
from src.snowflake_client import SnowflakeClient, SnowflakeExecutionError, SnowflakeQueryResult
from src.sql_validator import SQLValidator


def test_model_down_handled_gracefully():
    mock_gemma = MagicMock(spec=GemmaPlanner)
    mock_gemma.model = "gemma3:1b"
    mock_gemma.plan_query.side_effect = OllamaConnectionError("Could not connect to Ollama at http://localhost:11434")

    pipeline = DataMindPipeline(planner_model=mock_gemma)
    result = pipeline.ask("What is total revenue?")

    assert not result.is_success
    assert result.evidence.status == "error"
    assert "Could not connect to Ollama" in result.evidence.error_message
    assert "Model Connection Error" in result.answer


def test_bad_json_handled_gracefully():
    mock_gemma = MagicMock(spec=GemmaPlanner)
    mock_gemma.model = "gemma3:1b"
    mock_gemma.plan_query.side_effect = ModelPlanningError("Gemma output invalid JSON")

    pipeline = DataMindPipeline(planner_model=mock_gemma)
    result = pipeline.ask("What is total revenue?")

    assert not result.is_success
    assert result.evidence.status == "error"
    assert "invalid JSON" in result.evidence.error_message
    assert "Model Planning Failure" in result.answer


def test_empty_result_reported_cleanly():
    mock_gemma = MagicMock(spec=GemmaPlanner)
    mock_gemma.model = "gemma3:1b"
    mock_gemma.plan_query.return_value = QueryPlan(
        status="ok",
        metric="revenue",
        dimensions=["nation"],
        limit=5,
    )

    mock_client = MagicMock(spec=SnowflakeClient)
    # Empty DataFrame with expected columns
    empty_df = pd.DataFrame(columns=["NATION", "REVENUE"])
    mock_client.execute_query.return_value = SnowflakeQueryResult(
        df=empty_df,
        query_id="sfqid-empty-12345",
        elapsed_seconds=0.12,
        row_count=0,
        sql="SELECT ...",
    )

    executor = QueryExecutor(client=mock_client, validator=SQLValidator())
    pipeline = DataMindPipeline(planner_model=mock_gemma, query_executor=executor)

    result = pipeline.ask("Show revenue for nation Atlantis")
    assert result.is_success
    assert result.evidence.status == "empty"
    assert result.evidence.row_count == 0
    assert "No matching rows found in Snowflake" in result.answer
    assert result.evidence.snowflake_query_id == "sfqid-empty-12345"


def test_snowflake_error_captured_and_surfaced():
    mock_gemma = MagicMock(spec=GemmaPlanner)
    mock_gemma.model = "gemma3:1b"
    mock_gemma.plan_query.return_value = QueryPlan(
        status="ok",
        metric="revenue",
        limit=5,
    )

    mock_client = MagicMock(spec=SnowflakeClient)
    mock_client.execute_query.side_effect = SnowflakeExecutionError(
        message="Warehouse DATAMIND_WH suspended or unavailable",
        query_id="sfqid-err-99999",
        sql="SELECT ...",
    )

    executor = QueryExecutor(client=mock_client, validator=SQLValidator())
    pipeline = DataMindPipeline(planner_model=mock_gemma, query_executor=executor)

    result = pipeline.ask("What is total revenue?")
    assert not result.is_success
    assert result.evidence.status == "error"
    assert result.evidence.snowflake_query_id == "sfqid-err-99999"
    assert "DATAMIND_WH suspended" in result.evidence.error_message
    assert "Query Execution Failed" in result.answer


def test_unsafe_query_never_reaches_model_or_snowflake():
    mock_gemma = MagicMock(spec=GemmaPlanner)
    mock_client = MagicMock(spec=SnowflakeClient)
    executor = QueryExecutor(client=mock_client, validator=SQLValidator())

    pipeline = DataMindPipeline(planner_model=mock_gemma, query_executor=executor)
    result = pipeline.ask("DROP TABLE ORDERS")

    assert not result.is_success
    assert result.evidence.status == "unsupported"
    assert "Destructive" in result.evidence.error_message
    mock_gemma.plan_query.assert_not_called()
    mock_client.execute_query.assert_not_called()
