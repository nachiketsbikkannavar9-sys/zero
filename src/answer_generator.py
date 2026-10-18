"""Deterministic Answer Generator for DataMind.

Builds answers strictly from verified Snowflake query result rows.
Numbers come ONLY from Snowflake results, NEVER from language models.
Optionally verifies any LLM explanation to ensure 100% numerical fidelity against
result rows; discards LLM text if any unverified number appears.
"""

from __future__ import annotations

import re
from typing import Any, List, Optional, Set
import pandas as pd

from src.query_planner import QueryPlan


class AnswerGenerator:
    """Generates traceable, deterministic natural language answers from query results."""

    def format_number(self, val: Any) -> str:
        """Formats numbers cleanly (currency/commas if applicable)."""
        if pd.isna(val) or val is None:
            return "N/A"
        if isinstance(val, (int, float)):
            if isinstance(val, float):
                # If large or decimal
                if abs(val) >= 1:
                    return f"{val:,.2f}"
                return f"{val:.4f}"
            return f"{val:,}"
        return str(val)

    def extract_numbers_from_text(self, text: str) -> Set[str]:
        """Extracts numerical tokens from text (integers and floats)."""
        raw_tokens = re.findall(r"[-+]?\b\d+(?:,\d{3})*(?:\.\d+)?\b", text)
        cleaned = set()
        for tok in raw_tokens:
            normalized = tok.replace(",", "")
            # Ignore purely year-like numbers or tiny limits if context dictates, or check strictly
            try:
                val = float(normalized)
                cleaned.add(f"{val:.2f}")
                cleaned.add(f"{int(val)}" if val.is_integer() else f"{val}")
            except ValueError:
                continue
        return cleaned

    def verify_numbers_against_df(self, text: str, df: pd.DataFrame) -> bool:
        """Verifies that every numerical claim in text exists in the dataframe rows."""
        if df.empty:
            return False

        text_numbers = self.extract_numbers_from_text(text)
        if not text_numbers:
            return True

        df_numbers: Set[str] = set()
        for col in df.columns:
            for val in df[col]:
                if pd.notna(val):
                    try:
                        num = float(str(val).replace(",", ""))
                        df_numbers.add(f"{num:.2f}")
                        df_numbers.add(f"{int(num)}" if num.is_integer() else f"{num}")
                        df_numbers.add(str(val).strip())
                    except (ValueError, TypeError):
                        pass

        # Also add row count
        df_numbers.add(str(len(df)))

        # Every number extracted from text must be in df_numbers
        for num in text_numbers:
            if num not in df_numbers:
                return False
        return True

    def build_deterministic_answer(
        self,
        df: Optional[pd.DataFrame],
        plan: QueryPlan,
        assumptions: Optional[List[str]] = None,
        error: Optional[str] = None,
    ) -> str:
        """Builds a deterministic summary directly from result rows."""
        # 1. Error handling
        if error:
            return f"❌ **Query Execution Failed**\n\n{error}"

        # 2. Clarification status
        if plan.status == "needs_clarification":
            return (
                f"❓ **Clarification Needed**\n\n"
                f"{plan.clarification_question or 'Please provide additional details for your question.'}"
            )

        # 3. Unsupported status
        if plan.status == "unsupported":
            return (
                f"⚠️ **Unsupported Question**\n\n"
                f"{plan.unsupported_reason or 'This question cannot be answered with the current dataset.'}"
            )

        # 4. Empty result
        if df is None or df.empty:
            answer = "ℹ️ **No matching rows found in Snowflake** for the specified criteria."
            if assumptions:
                answer += "\n\n**Assumptions & Definitions:**\n" + "\n".join(f"- {a}" for a in assumptions)
            return answer

        # 5. Build answer from rows
        row_count = len(df)
        cols = list(df.columns)
        lines: List[str] = []

        # Single scalar result (e.g., total count or revenue across whole dataset)
        if row_count == 1 and len(cols) == 1:
            col_name = cols[0]
            val = df.iloc[0, 0]
            formatted_val = self.format_number(val)
            lines.append(f"**Result:** {col_name} is **{formatted_val}**.")

        # Single row with multiple dimensions/metrics
        elif row_count == 1 and len(cols) > 1:
            row = df.iloc[0]
            summary_items = [f"**{col}**: {self.format_number(row[col])}" for col in cols]
            lines.append(" | ".join(summary_items))

        # Multi-row result (ranking or grouped data)
        else:
            metric_col = plan.metric.upper() if plan.metric and plan.metric.upper() in cols else cols[-1]
            dim_cols = [c for c in cols if c != metric_col]

            headline = f"Found **{row_count}** result{'s' if row_count > 1 else ''} from Snowflake:"
            lines.append(headline)
            lines.append("")

            # Top results list
            for i, (_, row) in enumerate(df.iterrows(), start=1):
                if dim_cols:
                    dim_str = ", ".join(f"{c}: {row[c]}" for c in dim_cols)
                    metric_str = f"{metric_col}: {self.format_number(row[metric_col])}"
                    lines.append(f"{i}. {dim_str} → **{metric_str}**")
                else:
                    lines.append(f"{i}. **{self.format_number(row[metric_col])}**")

        # Include assumptions
        all_assumptions = list(assumptions or [])
        if all_assumptions:
            lines.append("")
            lines.append("---")
            lines.append("**Assumptions & Catalog Definitions:**")
            for a in all_assumptions:
                lines.append(f"- {a}")

        return "\n".join(lines)
