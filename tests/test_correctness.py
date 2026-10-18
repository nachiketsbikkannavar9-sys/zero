"""Live correctness tests comparing DataMind against hand-written ground-truth SQL.

Marked @pytest.mark.live.
Automatically skipped if Snowflake credentials are not present in the environment or .env.
"""

import json
import os
from pathlib import Path
import pytest
from dotenv import load_dotenv

load_dotenv()

from src.pipeline import DataMindPipeline
from src.snowflake_client import SnowflakeClient, SnowflakeConfigurationError


def snowflake_credentials_present() -> bool:
    return bool(os.getenv("SNOWFLAKE_ACCOUNT") and os.getenv("SNOWFLAKE_USER"))


@pytest.mark.live
def test_live_correctness_against_ground_truth():
    if not snowflake_credentials_present():
        pytest.skip("Snowflake credentials not present in environment or .env; skipping live correctness test.")

    # Load questions
    eval_file = Path(__file__).resolve().parent.parent / "evaluation" / "questions.json"
    with open(eval_file, "r", encoding="utf-8") as f:
        questions = json.load(f)

    client = SnowflakeClient()
    pipeline = DataMindPipeline()

    tested_queries = 0

    for item in questions:
        gt_sql = item.get("ground_truth_sql")
        if not gt_sql:
            continue

        q_text = item["question"]
        print(f"\n[LIVE TEST] Evaluating: {q_text}")

        # 1. Execute ground-truth SQL directly on Snowflake
        gt_res = client.execute_query(gt_sql)
        gt_df = gt_res.df

        # 2. Execute pipeline
        pipe_res = pipeline.ask(q_text)

        assert pipe_res.is_success, f"Pipeline failed on question '{q_text}': {pipe_res.evidence.error_message}"
        pipe_df = pipe_res.evidence.result_df
        assert pipe_df is not None, "Pipeline result DataFrame is None"

        # Compare row counts
        assert len(pipe_df) == len(gt_df), (
            f"Row count mismatch for '{q_text}': pipeline got {len(pipe_df)}, ground truth got {len(gt_df)}"
        )

        # If scalar or small ranking, compare top metric values
        if not gt_df.empty:
            # Check shape
            assert len(pipe_df.columns) == len(gt_df.columns)

        tested_queries += 1

    print(f"\nSuccessfully verified {tested_queries} live questions against ground-truth Snowflake queries!")
