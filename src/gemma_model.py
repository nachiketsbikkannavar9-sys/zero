"""Gemma model interface via Ollama for DataMind.

Converts plain-English user questions into structured QueryPlan JSON objects using
local Google Gemma (via Ollama). Temperature is set to 0.0 with schema enforcement.
Sends only the semantic catalog (metrics, dimensions, unavailable concepts), NEVER raw database tables.
Includes few-shot examples and automatic single retry on invalid JSON.
"""

from __future__ import annotations

import json
import os
from typing import Any, Dict, List, Optional
import requests

from src.query_planner import QueryPlan
from src.semantic_catalog import DatasetConfig, get_catalog


class OllamaConnectionError(Exception):
    """Raised when the Ollama host is unreachable."""
    pass


class ModelPlanningError(Exception):
    """Raised when Gemma fails to produce a valid QueryPlan after retries."""
    pass


class GemmaPlanner:
    """Interacts with local Gemma via Ollama /api/chat to generate structured query plans."""

    def __init__(
        self,
        host: Optional[str] = None,
        model: Optional[str] = None,
        timeout_s: int = 45,
    ):
        self.host = (host or os.getenv("OLLAMA_HOST", "http://localhost:11434")).rstrip("/")
        self.model = model or os.getenv("OLLAMA_MODEL", "gemma3:1b")
        self.timeout_s = timeout_s

    def _build_catalog_prompt(self, dataset: DatasetConfig) -> str:
        """Serializes ONLY the semantic catalog (no raw tables) into the system prompt."""
        metrics_desc = [
            f"  - {name}: {m.description}"
            for name, m in dataset.metrics.items()
        ]
        dims_desc = [
            f"  - {name}: {d.description}" + (f" (Allowed: {d.allowed_values})" if d.allowed_values else "")
            for name, d in dataset.dimensions.items()
        ]
        unavail_desc = [
            f"  - {concept}: {reason}"
            for concept, reason in dataset.unavailable_concepts.items()
        ]

        grains = dataset.date_dimension.allowed_grains if dataset.date_dimension else ["year", "quarter", "month"]

        return f"""You are DataMind, a semantic query planner.
Your goal is to translate a user's natural language question into a strict QueryPlan JSON object according to the semantic catalog.

DATASET: {dataset.name} ({dataset.description})

AVAILABLE METRICS:
{chr(10).join(metrics_desc)}

AVAILABLE DIMENSIONS:
{chr(10).join(dims_desc)}

DATE TIME GRAINS:
Allowed time grains for date grouping: {', '.join(grains)}

UNAVAILABLE CONCEPTS (mark status='unsupported' with unsupported_reason):
{chr(10).join(unavail_desc)}

RULES:
1. Always return a valid JSON object matching the QueryPlan schema.
2. If the user asks for unavailable concepts (profit, cost, margin, churn, returns) or asks for predictive forecasting, return status='unsupported' with an explicit unsupported_reason explaining why.
3. If the user question is vague (e.g. 'show best items', 'top performing') and doesn't specify which metric to use, return status='needs_clarification' with clarification_question.
4. If the question is valid, return status='ok' with metric, dimensions, filters, time_grain, order_by, and limit.
5. Limit defaults to 10 unless specified.
6. NEVER make up metrics or dimensions not in the catalog above.
"""

    def _get_few_shot_messages(self) -> List[Dict[str, str]]:
        """Provides 6 canonical few-shot examples."""
        return [
            # 1. Ranking
            {
                "role": "user",
                "content": "What are the top 5 parts by revenue?",
            },
            {
                "role": "assistant",
                "content": json.dumps({
                    "status": "ok",
                    "metric": "revenue",
                    "dimensions": ["part_name"],
                    "filters": [],
                    "time_grain": None,
                    "order_by": {"field": "revenue", "direction": "desc"},
                    "limit": 5,
                    "clarification_question": None,
                    "unsupported_reason": None,
                    "assumptions": ["Ranked by net revenue descending (top 5)."]
                })
            },
            # 2. Time grouping
            {
                "role": "user",
                "content": "Show monthly revenue for 1995.",
            },
            {
                "role": "assistant",
                "content": json.dumps({
                    "status": "ok",
                    "metric": "revenue",
                    "dimensions": [],
                    "filters": [
                        {"dimension": "order_date", "op": ">=", "value": "1995-01-01"},
                        {"dimension": "order_date", "op": "<=", "value": "1995-12-31"}
                    ],
                    "time_grain": "month",
                    "order_by": {"field": "order_date_month", "direction": "asc"},
                    "limit": 12,
                    "clarification_question": None,
                    "unsupported_reason": None,
                    "assumptions": ["Aggregated monthly across 1995."]
                })
            },
            # 3. Join / Multi-dimension
            {
                "role": "user",
                "content": "What is the total revenue by region?",
            },
            {
                "role": "assistant",
                "content": json.dumps({
                    "status": "ok",
                    "metric": "revenue",
                    "dimensions": ["region"],
                    "filters": [],
                    "time_grain": None,
                    "order_by": {"field": "revenue", "direction": "desc"},
                    "limit": 10,
                    "clarification_question": None,
                    "unsupported_reason": None,
                    "assumptions": ["Aggregated across all 5 macro-regions."]
                })
            },
            # 4. Ambiguous / Vague question
            {
                "role": "user",
                "content": "Show me our best performing items.",
            },
            {
                "role": "assistant",
                "content": json.dumps({
                    "status": "needs_clarification",
                    "metric": None,
                    "dimensions": ["part_name"],
                    "filters": [],
                    "time_grain": None,
                    "order_by": None,
                    "limit": 10,
                    "clarification_question": "Which metric should be used to measure performance? (e.g., revenue, order_count, or total_quantity?)",
                    "unsupported_reason": None,
                    "assumptions": []
                })
            },
            # 5. Missing / Unavailable field (Profit)
            {
                "role": "user",
                "content": "What was our total profit in 1995?",
            },
            {
                "role": "assistant",
                "content": json.dumps({
                    "status": "unsupported",
                    "metric": None,
                    "dimensions": [],
                    "filters": [],
                    "time_grain": None,
                    "order_by": None,
                    "limit": 10,
                    "clarification_question": None,
                    "unsupported_reason": "Profit cannot be calculated because manufacturing overhead and fulfillment costs are not part of the standard TPC-H schema.",
                    "assumptions": []
                })
            },
            # 6. Forecasting / Prediction question
            {
                "role": "user",
                "content": "Predict customer order volume for the next 2 quarters.",
            },
            {
                "role": "assistant",
                "content": json.dumps({
                    "status": "unsupported",
                    "metric": None,
                    "dimensions": [],
                    "filters": [],
                    "time_grain": None,
                    "order_by": None,
                    "limit": 10,
                    "clarification_question": None,
                    "unsupported_reason": "DataMind executes deterministic historical queries on Snowflake. Predictive forecasting is not supported.",
                    "assumptions": []
                })
            },
        ]

    def plan_query(self, question: str, dataset: Optional[DatasetConfig] = None) -> QueryPlan:
        """Sends the question to Gemma and returns a parsed QueryPlan."""
        ds = dataset or get_catalog().get_dataset()
        system_prompt = self._build_catalog_prompt(ds)
        schema_format = QueryPlan.model_json_schema()

        messages = [
            {"role": "system", "content": system_prompt},
            *self._get_few_shot_messages(),
            {"role": "user", "content": question},
        ]

        payload = {
            "model": self.model,
            "messages": messages,
            "stream": False,
            "format": schema_format,
            "options": {
                "temperature": 0.0,
            },
        }

        # Attempt 1
        try:
            resp = requests.post(
                f"{self.host}/api/chat",
                json=payload,
                timeout=self.timeout_s,
            )
        except requests.exceptions.ConnectionError as exc:
            raise OllamaConnectionError(
                f"Could not connect to Ollama at '{self.host}'. "
                f"Ensure Ollama is running (`ollama serve`) and model '{self.model}' is pulled."
            ) from exc
        except Exception as exc:
            raise OllamaConnectionError(f"Error calling Ollama API: {exc}") from exc

        if resp.status_code != 200:
            raise ModelPlanningError(f"Ollama returned HTTP {resp.status_code}: {resp.text}")

        data = resp.json()
        content = data.get("message", {}).get("content", "").strip()

        # Parse JSON
        try:
            return QueryPlan.model_validate_json(content)
        except Exception as first_parse_err:
            # Retry once with explicit feedback
            retry_messages = list(messages)
            retry_messages.append({"role": "assistant", "content": content})
            retry_messages.append({
                "role": "user",
                "content": (
                    f"The previous output could not be parsed according to the QueryPlan schema. "
                    f"Error: {first_parse_err}. "
                    f"Please output ONLY a valid JSON object matching the schema."
                ),
            })
            payload["messages"] = retry_messages

            try:
                retry_resp = requests.post(
                    f"{self.host}/api/chat",
                    json=payload,
                    timeout=self.timeout_s,
                )
                if retry_resp.status_code == 200:
                    retry_content = retry_resp.json().get("message", {}).get("content", "").strip()
                    return QueryPlan.model_validate_json(retry_content)
            except Exception as second_err:
                raise ModelPlanningError(
                    f"Gemma model failed to output valid QueryPlan JSON after retry. "
                    f"First error: {first_parse_err}. Second error: {second_err}."
                ) from second_err

            raise ModelPlanningError(
                f"Gemma model failed to output valid QueryPlan JSON. Error: {first_parse_err}"
            )
