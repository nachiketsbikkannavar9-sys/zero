#!/usr/bin/env python3
"""Smoke test script for Snowflake connection.

Runs a simple SELECT COUNT(*) on ORDERS table in the configured schema
and prints the resulting count, Snowflake query ID (sfqid), and elapsed time.
"""

import sys
import os
from pathlib import Path

# Add project root to sys.path
root_dir = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(root_dir))

from src.snowflake_client import SnowflakeClient, SnowflakeConfigurationError, SnowflakeExecutionError


def main() -> int:
    print("=" * 60)
    print("DataMind Snowflake Smoke Test")
    print("=" * 60)

    try:
        client = SnowflakeClient()
        db = client.database
        schema = client.schema
        table = "ORDERS"
        sql = f"SELECT COUNT(*) AS ORDER_COUNT FROM {db}.{schema}.{table}"

        print(f"Connecting to Snowflake account: {client.account or '[Not configured]'}")
        print(f"Target DB.SCHEMA: {db}.{schema}")
        print(f"Warehouse: {client.warehouse} | Role: {client.role}")
        print(f"Executing: {sql}")
        print("-" * 60)

        result = client.execute_query(sql)

        print("Query executed successfully!")
        print(f"Snowflake Query ID (sfqid) : {result.query_id}")
        print(f"Elapsed Time               : {result.elapsed_seconds:.3f} s")
        print(f"Row Count                  : {result.row_count}")
        print("Result Data:")
        print(result.df.to_string(index=False))
        print("=" * 60)
        return 0

    except SnowflakeConfigurationError as e:
        print("\n[CONFIGURATION ERROR]")
        print(f"  {e}")
        print("\nPlease create a .env file based on .env.example with your Snowflake credentials.")
        return 1
    except SnowflakeExecutionError as e:
        print("\n[EXECUTION ERROR]")
        print(f"  {e}")
        if e.query_id:
            print(f"  Snowflake Query ID: {e.query_id}")
        return 1
    except Exception as e:
        print(f"\n[UNEXPECTED ERROR]: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
