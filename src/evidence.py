"""Evidence data model for DataMind.

Captures the full audit trail and provenance for every question:
Question -> Plan JSON -> Generated SQL -> Snowflake Query ID -> Returned Rows -> Assumptions.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional
import pandas as pd


@dataclass
class Evidence:
    """Complete provenance and audit record for a single question."""
    question: str
    plan_json: Dict[str, Any]
    executed_sql: Optional[str] = None
    snowflake_query_id: Optional[str] = None
    row_count: int = 0
    elapsed_time_seconds: float = 0.0
    result_df: Optional[pd.DataFrame] = None
    assumptions: List[str] = field(default_factory=list)
    model_name: str = "gemma3:1b"
    status: str = "ok"  # "ok", "needs_clarification", "unsupported", "error", "empty"
    error_message: Optional[str] = None

    def to_summary_dict(self) -> Dict[str, Any]:
        """Returns a JSON-serializable dictionary summary of the execution."""
        return {
            "question": self.question,
            "status": self.status,
            "model_name": self.model_name,
            "plan": self.plan_json,
            "executed_sql": self.executed_sql,
            "snowflake_query_id": self.snowflake_query_id,
            "row_count": self.row_count,
            "elapsed_time_seconds": round(self.elapsed_time_seconds, 4),
            "assumptions": self.assumptions,
            "error_message": self.error_message,
        }
