"""End-to-End Pipeline for DataMind.

Orchestrates the entire query answering lifecycle:
User Question
  -> Pre-flight Safety Guard
  -> Gemma (Ollama) Structured Plan
  -> Semantic Catalog Plan Validation
  -> Deterministic SQL Compilation
  -> sqlglot AST Safety Validation
  -> Real Snowflake Execution
  -> Deterministic Result Row Answer Generation
  -> Complete Evidence Provenance
"""

from __future__ import annotations

import os
import re
import time
from dataclasses import dataclass
from typing import Optional
import pandas as pd

from src.answer_generator import AnswerGenerator
from src.evidence import Evidence
from src.gemma_model import GemmaPlanner, ModelPlanningError, OllamaConnectionError
from src.query_executor import QueryExecutor, UnsafeQueryError
from src.query_planner import PlanValidationError, QueryPlan, QueryPlanner
from src.semantic_catalog import DatasetConfig, get_catalog
from src.snowflake_client import SnowflakeConfigurationError, SnowflakeExecutionError
from src.sql_validator import SQLValidator


@dataclass
class PipelineResult:
    """End-to-end outcome of a DataMind question."""
    question: str
    answer: str
    evidence: Evidence
    is_success: bool


class DataMindPipeline:
    """The central pipeline coordinating planning, validation, execution, and answering."""

    def __init__(
        self,
        dataset_id: Optional[str] = None,
        planner_model: Optional[GemmaPlanner] = None,
        query_planner: Optional[QueryPlanner] = None,
        sql_validator: Optional[SQLValidator] = None,
        query_executor: Optional[QueryExecutor] = None,
        answer_generator: Optional[AnswerGenerator] = None,
    ):
        catalog = get_catalog()
        self.dataset: DatasetConfig = catalog.get_dataset(dataset_id)
        self.planner_model = planner_model or GemmaPlanner()
        self.query_planner = query_planner or QueryPlanner(dataset_config=self.dataset)
        self.sql_validator = sql_validator or SQLValidator(dataset=self.dataset)
        self.query_executor = query_executor or QueryExecutor(validator=self.sql_validator)
        self.answer_generator = answer_generator or AnswerGenerator()

    def _check_unsafe_question(self, question: str) -> Optional[str]:
        """Pre-flight check to block dangerous or destructive intentions immediately."""
        lower = question.lower()
        destructive_patterns = [
            r"\bdelete\b",
            r"\bdrop\b",
            r"\btruncate\b",
            r"\balter\b",
            r"\binsert\b",
            r"\bupdate\b",
            r"\bgrant\b",
            r"\brevoke\b",
            r"\bshutdown\b",
            r"\bkill\b",
        ]
        for pattern in destructive_patterns:
            if re.search(pattern, lower):
                return (
                    f"Destructive or state-modifying requests ('{pattern.strip(chr(92) + 'b')}') "
                    f"are prohibited. DataMind operates exclusively in read-only mode."
                )
        return None

    def ask(self, question: str) -> PipelineResult:
        """Executes the full pipeline for a natural language question."""
        clean_q = question.strip()
        if not clean_q:
            plan_obj = QueryPlan(status="needs_clarification", clarification_question="Please enter a non-empty question.")
            ans = self.answer_generator.build_deterministic_answer(None, plan_obj)
            ev = Evidence(question=question, plan_json=plan_obj.model_dump(), status="needs_clarification")
            return PipelineResult(question=clean_q, answer=ans, evidence=ev, is_success=False)

        # 1. Pre-flight Safety Guard
        unsafe_reason = self._check_unsafe_question(clean_q)
        if unsafe_reason:
            plan_obj = QueryPlan(status="unsupported", unsupported_reason=unsafe_reason)
            ans = self.answer_generator.build_deterministic_answer(None, plan_obj)
            ev = Evidence(
                question=clean_q,
                plan_json=plan_obj.model_dump(),
                status="unsupported",
                assumptions=["Operation rejected by pre-execution safety validator."],
                error_message=unsafe_reason,
            )
            return PipelineResult(question=clean_q, answer=ans, evidence=ev, is_success=False)

        # 2. Plan Generation with Gemma (or offline bypass if mocked)
        start_time = time.perf_counter()
        try:
            raw_plan = self.planner_model.plan_query(clean_q, dataset=self.dataset)
        except OllamaConnectionError as e:
            err_msg = str(e)
            plan_obj = QueryPlan(status="unsupported", unsupported_reason=err_msg)
            ev = Evidence(
                question=clean_q,
                plan_json={},
                status="error",
                error_message=err_msg,
                model_name=self.planner_model.model,
            )
            ans = f"🔌 **Model Connection Error**\n\n{err_msg}"
            return PipelineResult(question=clean_q, answer=ans, evidence=ev, is_success=False)
        except ModelPlanningError as e:
            err_msg = str(e)
            plan_obj = QueryPlan(status="unsupported", unsupported_reason=err_msg)
            ev = Evidence(
                question=clean_q,
                plan_json={},
                status="error",
                error_message=err_msg,
                model_name=self.planner_model.model,
            )
            ans = f"⚠️ **Model Planning Failure**\n\n{err_msg}"
            return PipelineResult(question=clean_q, answer=ans, evidence=ev, is_success=False)

        # 3. Catalog Validation
        validated_plan = self.query_planner.validate_plan(raw_plan, question_text=clean_q)
        plan_dict = validated_plan.model_dump()

        if validated_plan.status in ("needs_clarification", "unsupported"):
            ans = self.answer_generator.build_deterministic_answer(None, validated_plan)
            ev = Evidence(
                question=clean_q,
                plan_json=plan_dict,
                status=validated_plan.status,
                model_name=self.planner_model.model,
            )
            return PipelineResult(question=clean_q, answer=ans, evidence=ev, is_success=False)

        # 4. Deterministic SQL Compilation
        try:
            compilation = self.query_planner.compile_sql(validated_plan)
        except PlanValidationError as e:
            err_msg = str(e)
            validated_plan.status = "unsupported"
            validated_plan.unsupported_reason = err_msg
            ans = self.answer_generator.build_deterministic_answer(None, validated_plan)
            ev = Evidence(
                question=clean_q,
                plan_json=plan_dict,
                status="unsupported",
                error_message=err_msg,
                model_name=self.planner_model.model,
            )
            return PipelineResult(question=clean_q, answer=ans, evidence=ev, is_success=False)

        sql = compilation.sql
        assumptions = compilation.assumptions

        # 5. AST SQL Validation
        is_safe, safety_reason = self.sql_validator.validate(sql)
        if not is_safe:
            err_msg = f"SQL rejected by safety engine: {safety_reason}"
            ans = f"🛡️ **Security Alert**\n\n{err_msg}"
            ev = Evidence(
                question=clean_q,
                plan_json=plan_dict,
                executed_sql=sql,
                status="error",
                assumptions=assumptions,
                error_message=err_msg,
                model_name=self.planner_model.model,
            )
            return PipelineResult(question=clean_q, answer=ans, evidence=ev, is_success=False)

        # 6. Real Snowflake Execution
        try:
            query_res = self.query_executor.execute(sql)
        except SnowflakeConfigurationError as e:
            err_msg = f"Snowflake configuration missing: {e}"
            ans = (
                f"⚙️ **Snowflake Configuration Required**\n\n{err_msg}\n\n"
                f"Generated SQL:\n```sql\n{sql}\n```"
            )
            ev = Evidence(
                question=clean_q,
                plan_json=plan_dict,
                executed_sql=sql,
                status="error",
                assumptions=assumptions,
                error_message=err_msg,
                model_name=self.planner_model.model,
            )
            return PipelineResult(question=clean_q, answer=ans, evidence=ev, is_success=False)
        except SnowflakeExecutionError as e:
            err_msg = str(e)
            ans = self.answer_generator.build_deterministic_answer(
                None, validated_plan, assumptions=assumptions, error=err_msg
            )
            ev = Evidence(
                question=clean_q,
                plan_json=plan_dict,
                executed_sql=sql,
                snowflake_query_id=e.query_id,
                status="error",
                assumptions=assumptions,
                error_message=err_msg,
                model_name=self.planner_model.model,
            )
            return PipelineResult(question=clean_q, answer=ans, evidence=ev, is_success=False)
        except UnsafeQueryError as e:
            err_msg = str(e)
            ans = f"🛡️ **Security Alert**\n\n{err_msg}"
            ev = Evidence(
                question=clean_q,
                plan_json=plan_dict,
                executed_sql=sql,
                status="error",
                assumptions=assumptions,
                error_message=err_msg,
                model_name=self.planner_model.model,
            )
            return PipelineResult(question=clean_q, answer=ans, evidence=ev, is_success=False)

        # 7. Answer Generation from Real Rows
        df = query_res.df
        row_status = "empty" if df.empty else "ok"
        ans = self.answer_generator.build_deterministic_answer(
            df=df,
            plan=validated_plan,
            assumptions=assumptions,
        )

        ev = Evidence(
            question=clean_q,
            plan_json=plan_dict,
            executed_sql=sql,
            snowflake_query_id=query_res.query_id,
            row_count=query_res.row_count,
            elapsed_time_seconds=query_res.elapsed_seconds,
            result_df=df,
            assumptions=assumptions,
            model_name=self.planner_model.model,
            status=row_status,
        )

        return PipelineResult(question=clean_q, answer=ans, evidence=ev, is_success=True)
