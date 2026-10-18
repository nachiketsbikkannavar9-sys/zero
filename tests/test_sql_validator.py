"""Unit tests for SQLValidator (offline).

Verifies AST-based read-only enforcement:
- Accepts valid SELECT statements with allowed tables/columns and valid LIMIT.
- Rejects DML (INSERT, UPDATE, DELETE).
- Rejects DDL (DROP, CREATE, ALTER).
- Rejects multi-statement attacks.
- Rejects uncataloged tables and columns.
- Rejects queries missing LIMIT or with LIMIT > max_rows.
"""

import pytest
from src.sql_validator import SQLValidator
from src.semantic_catalog import get_catalog


@pytest.fixture
def validator():
    cat = get_catalog()
    ds = cat.get_dataset("tpch")
    return SQLValidator(dataset=ds, max_rows=500)


def test_accepts_valid_select(validator):
    sql = """
    SELECT 
        O_ORDERKEY, 
        O_TOTALPRICE 
    FROM SNOWFLAKE_SAMPLE_DATA.TPCH_SF1.ORDERS AS ORDERS 
    WHERE O_TOTALPRICE > 1000 
    LIMIT 10
    """
    is_valid, reason = validator.validate(sql)
    assert is_valid, f"Expected valid query, got: {reason}"


def test_accepts_valid_join_aggregation(validator):
    sql = """
    SELECT 
        PART.P_NAME AS PART_NAME,
        SUM(LINEITEM.L_EXTENDEDPRICE * (1 - LINEITEM.L_DISCOUNT)) AS REVENUE
    FROM SNOWFLAKE_SAMPLE_DATA.TPCH_SF1.LINEITEM AS LINEITEM
    JOIN SNOWFLAKE_SAMPLE_DATA.TPCH_SF1.PART AS PART 
      ON LINEITEM.L_PARTKEY = PART.P_PARTKEY
    GROUP BY 1
    ORDER BY REVENUE DESC
    LIMIT 5
    """
    is_valid, reason = validator.validate(sql)
    assert is_valid, f"Expected valid join query, got: {reason}"


@pytest.mark.parametrize("dml", [
    "INSERT INTO ORDERS (O_ORDERKEY) VALUES (1);",
    "UPDATE ORDERS SET O_TOTALPRICE = 0 WHERE O_ORDERKEY = 1;",
    "DELETE FROM CUSTOMER WHERE C_CUSTKEY = 10;",
    "DROP TABLE CUSTOMER;",
    "CREATE TABLE HACKED (ID INT);",
    "ALTER TABLE ORDERS DROP COLUMN O_COMMENT;",
    "TRUNCATE TABLE LINEITEM;",
])
def test_rejects_dml_and_ddl(validator, dml):
    is_valid, reason = validator.validate(dml)
    assert not is_valid
    assert "Forbidden" in reason or "SELECT" in reason or "Multiple" in reason


def test_rejects_multi_statement(validator):
    sql = "SELECT * FROM ORDERS LIMIT 10; DROP TABLE ORDERS;"
    is_valid, reason = validator.validate(sql)
    assert not is_valid
    assert "Multiple statements" in reason or "Forbidden" in reason


def test_rejects_unknown_table(validator):
    sql = "SELECT ID, SECRET FROM USERS_SECRET_TABLE LIMIT 10;"
    is_valid, reason = validator.validate(sql)
    assert not is_valid
    assert "Forbidden or unknown table" in reason or "USERS_SECRET_TABLE" in reason


def test_rejects_unknown_column(validator):
    sql = "SELECT NON_EXISTENT_COLUMN FROM ORDERS LIMIT 10;"
    is_valid, reason = validator.validate(sql)
    assert not is_valid
    assert "Unknown or uncataloged column" in reason or "NON_EXISTENT_COLUMN" in reason


def test_rejects_missing_limit(validator):
    sql = "SELECT O_ORDERKEY FROM ORDERS;"
    is_valid, reason = validator.validate(sql)
    assert not is_valid
    assert "Missing required LIMIT" in reason


def test_rejects_excessive_limit(validator):
    sql = "SELECT O_ORDERKEY FROM ORDERS LIMIT 99999;"
    is_valid, reason = validator.validate(sql)
    assert not is_valid
    assert "exceeds maximum allowed limit" in reason
