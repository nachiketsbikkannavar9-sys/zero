-- ============================================================================
-- DataMind: Snowflake Setup Script
-- Role, Warehouse, and Read-Only Privilege Configuration
-- ============================================================================

-- 1. Switch to ACCOUNTADMIN or SECURITYADMIN to configure roles and privileges
USE ROLE ACCOUNTADMIN;

-- 2. Create dedicated read-only role
CREATE ROLE IF NOT EXISTS DATAMIND_READONLY
    COMMENT = 'Read-only role for DataMind semantic assistant';

-- 3. Create dedicated lightweight warehouse with auto-suspend
CREATE WAREHOUSE IF NOT EXISTS DATAMIND_WH
    WITH 
    WAREHOUSE_SIZE = 'XSMALL'
    AUTO_SUSPEND = 60
    AUTO_RESUME = TRUE
    INITIALLY_SUSPENDED = TRUE
    COMMENT = 'Warehouse for DataMind analytical query execution';

-- 4. Grant warehouse usage to the read-only role
GRANT USAGE ON WAREHOUSE DATAMIND_WH TO ROLE DATAMIND_READONLY;
GRANT OPERATE ON WAREHOUSE DATAMIND_WH TO ROLE DATAMIND_READONLY;

-- 5. Grant read-only imported privileges on SNOWFLAKE_SAMPLE_DATA
-- This allows querying TPCH_SF1, TPCH_SF10, etc., with zero write access
GRANT IMPORTED PRIVILEGES ON DATABASE SNOWFLAKE_SAMPLE_DATA TO ROLE DATAMIND_READONLY;

-- 6. Grant role to your service or user account (replace <YOUR_SNOWFLAKE_USER>)
-- GRANT ROLE DATAMIND_READONLY TO USER YOUR_SNOWFLAKE_USER;
-- ALTER USER YOUR_SNOWFLAKE_USER SET DEFAULT_ROLE = DATAMIND_READONLY;
-- ALTER USER YOUR_SNOWFLAKE_USER SET DEFAULT_WAREHOUSE = DATAMIND_WH;

-- ============================================================================
-- Verification:
-- Run with ROLE DATAMIND_READONLY:
--   USE ROLE DATAMIND_READONLY;
--   USE WAREHOUSE DATAMIND_WH;
--   SELECT COUNT(*) FROM SNOWFLAKE_SAMPLE_DATA.TPCH_SF1.ORDERS;
-- ============================================================================
