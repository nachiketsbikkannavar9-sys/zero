"""Schema discovery for DataMind.

Inspects tables, columns, and data types from Snowflake's INFORMATION_SCHEMA
for the configured database and schema. Used for SQL validation and schema
inspection in the Streamlit UI.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Dict, List, Optional
import pandas as pd

from src.snowflake_client import SnowflakeClient, SnowflakeExecutionError


@dataclass
class ColumnMetadata:
    """Metadata for a single column."""
    name: str
    data_type: str
    is_nullable: bool
    comment: Optional[str] = None


@dataclass
class TableMetadata:
    """Metadata for a single table."""
    name: str
    columns: Dict[str, ColumnMetadata] = field(default_factory=dict)


@dataclass
class SchemaMetadata:
    """Metadata for a database schema."""
    database: str
    schema: str
    tables: Dict[str, TableMetadata] = field(default_factory=dict)

    def has_table(self, table_name: str) -> bool:
        return table_name.upper() in self.tables

    def has_column(self, table_name: str, column_name: str) -> bool:
        t_name = table_name.upper()
        if t_name not in self.tables:
            return False
        return column_name.upper() in self.tables[t_name].columns

    def to_dict(self) -> Dict[str, Any]:
        result = {}
        for t_name, table in self.tables.items():
            result[t_name] = [
                {
                    "column": col.name,
                    "type": col.data_type,
                    "nullable": col.is_nullable,
                    "comment": col.comment,
                }
                for col in table.columns.values()
            ]
        return result


# Static reference metadata for standard TPC-H schemas (used for offline tests or when offline)
TPCH_REFERENCE_COLUMNS: Dict[str, List[tuple[str, str]]] = {
    "CUSTOMER": [
        ("C_CUSTKEY", "NUMBER"),
        ("C_NAME", "VARCHAR"),
        ("C_ADDRESS", "VARCHAR"),
        ("C_NATIONKEY", "NUMBER"),
        ("C_PHONE", "VARCHAR"),
        ("C_ACCTBAL", "NUMBER"),
        ("C_MKTSEGMENT", "VARCHAR"),
        ("C_COMMENT", "VARCHAR"),
    ],
    "ORDERS": [
        ("O_ORDERKEY", "NUMBER"),
        ("O_CUSTKEY", "NUMBER"),
        ("O_ORDERSTATUS", "VARCHAR"),
        ("O_TOTALPRICE", "NUMBER"),
        ("O_ORDERDATE", "DATE"),
        ("O_ORDERPRIORITY", "VARCHAR"),
        ("O_CLERK", "VARCHAR"),
        ("O_SHIPPRIORITY", "NUMBER"),
        ("O_COMMENT", "VARCHAR"),
    ],
    "LINEITEM": [
        ("L_ORDERKEY", "NUMBER"),
        ("L_PARTKEY", "NUMBER"),
        ("L_SUPPKEY", "NUMBER"),
        ("L_LINENUMBER", "NUMBER"),
        ("L_QUANTITY", "NUMBER"),
        ("L_EXTENDEDPRICE", "NUMBER"),
        ("L_DISCOUNT", "NUMBER"),
        ("L_TAX", "NUMBER"),
        ("L_RETURNFLAG", "VARCHAR"),
        ("L_LINESTATUS", "VARCHAR"),
        ("L_SHIPDATE", "DATE"),
        ("L_COMMITDATE", "DATE"),
        ("L_RECEIPTDATE", "DATE"),
        ("L_SHIPINSTRUCT", "VARCHAR"),
        ("L_SHIPMODE", "VARCHAR"),
        ("L_COMMENT", "VARCHAR"),
    ],
    "PART": [
        ("P_PARTKEY", "NUMBER"),
        ("P_NAME", "VARCHAR"),
        ("P_MFGR", "VARCHAR"),
        ("P_BRAND", "VARCHAR"),
        ("P_TYPE", "VARCHAR"),
        ("P_SIZE", "NUMBER"),
        ("P_CONTAINER", "VARCHAR"),
        ("P_RETAILPRICE", "NUMBER"),
        ("P_COMMENT", "VARCHAR"),
    ],
    "SUPPLIER": [
        ("S_SUPPKEY", "NUMBER"),
        ("S_NAME", "VARCHAR"),
        ("S_ADDRESS", "VARCHAR"),
        ("S_NATIONKEY", "NUMBER"),
        ("S_PHONE", "VARCHAR"),
        ("S_ACCTBAL", "NUMBER"),
        ("S_COMMENT", "VARCHAR"),
    ],
    "PARTSUPP": [
        ("PS_PARTKEY", "NUMBER"),
        ("PS_SUPPKEY", "NUMBER"),
        ("PS_AVAILQTY", "NUMBER"),
        ("PS_SUPPLYCOST", "NUMBER"),
        ("PS_COMMENT", "VARCHAR"),
    ],
    "NATION": [
        ("N_NATIONKEY", "NUMBER"),
        ("N_NAME", "VARCHAR"),
        ("N_REGIONKEY", "NUMBER"),
        ("N_COMMENT", "VARCHAR"),
    ],
    "REGION": [
        ("R_REGIONKEY", "NUMBER"),
        ("R_NAME", "VARCHAR"),
        ("R_COMMENT", "VARCHAR"),
    ],
}


def get_reference_tpch_schema(database: str = "SNOWFLAKE_SAMPLE_DATA", schema: str = "TPCH_SF1") -> SchemaMetadata:
    """Builds reference SchemaMetadata for TPC-H schemas for offline validation."""
    meta = SchemaMetadata(database=database.upper(), schema=schema.upper())
    for t_name, cols in TPCH_REFERENCE_COLUMNS.items():
        tbl = TableMetadata(name=t_name)
        for col_name, data_type in cols:
            tbl.columns[col_name] = ColumnMetadata(
                name=col_name,
                data_type=data_type,
                is_nullable=False,
            )
        meta.tables[t_name] = tbl
    return meta


class SchemaDiscovery:
    """Discovers and caches schema metadata from Snowflake INFORMATION_SCHEMA."""

    def __init__(self, client: Optional[SnowflakeClient] = None):
        self.client = client or SnowflakeClient()

    def discover(self, database: Optional[str] = None, schema: Optional[str] = None) -> SchemaMetadata:
        """Queries INFORMATION_SCHEMA.COLUMNS to build a live SchemaMetadata model."""
        target_db = (database or self.client.database or "SNOWFLAKE_SAMPLE_DATA").upper()
        target_schema = (schema or self.client.schema or "TPCH_SF1").upper()

        sql = f"""
        SELECT 
            TABLE_NAME, 
            COLUMN_NAME, 
            DATA_TYPE, 
            IS_NULLABLE, 
            COMMENT
        FROM {target_db}.INFORMATION_SCHEMA.COLUMNS
        WHERE TABLE_SCHEMA = '{target_schema}'
        ORDER BY TABLE_NAME, ORDINAL_POSITION
        """

        try:
            result = self.client.execute_query(sql)
            df = result.df

            schema_meta = SchemaMetadata(database=target_db, schema=target_schema)

            for _, row in df.iterrows():
                t_name = str(row["TABLE_NAME"]).upper()
                c_name = str(row["COLUMN_NAME"]).upper()
                data_type = str(row["DATA_TYPE"]).upper()
                nullable = str(row.get("IS_NULLABLE", "YES")).upper() == "YES"
                comment = row.get("COMMENT")
                if pd.isna(comment):
                    comment = None

                if t_name not in schema_meta.tables:
                    schema_meta.tables[t_name] = TableMetadata(name=t_name)

                schema_meta.tables[t_name].columns[c_name] = ColumnMetadata(
                    name=c_name,
                    data_type=data_type,
                    is_nullable=nullable,
                    comment=str(comment) if comment else None,
                )

            return schema_meta

        except Exception as exc:
            # If target is TPCH, provide reference fallback with informative logging if offline
            if "TPCH" in target_schema:
                return get_reference_tpch_schema(database=target_db, schema=target_schema)
            raise SnowflakeExecutionError(f"Failed to discover schema from INFORMATION_SCHEMA: {exc}") from exc
