# DataMind – Talk to Your Data

> **ZeroDay OSS Hacktoberfest 2026 — Problem Statement 1**  
> *A deterministic, zero-hallucination semantic data assistant powered by local Google Gemma and Snowflake.*

---

## 📖 What it is

**DataMind** is an enterprise data assistant that bridges natural language and cloud data warehouses with mathematical rigor:
1. **User asks a plain-English question** (e.g., *"What are the top 5 parts by revenue?"*).
2. **Google Gemma (local, via Ollama)** translates the question into a validated **Structured Query Plan (JSON)**.
3. **Python validates the plan** against a config-defined semantic catalog and compiles SQL deterministically.
4. **SQL is validated again** with `sqlglot` AST parsing (single statement, `SELECT` only, catalog-only tables/columns, strict `LIMIT`).
5. **Snowflake executes the query** at question time in `SNOWFLAKE_SAMPLE_DATA`.
6. **The answer is built strictly from real returned rows**, surfaced alongside the executed SQL, result table, Snowflake Query ID (`sfqid`), elapsed execution time, and stated metric assumptions.

**It is NOT a chatbot:** Numbers come **only** from real Snowflake query results, never from the model.

---

## ⚖️ Division of Responsibilities: Gemma vs. Snowflake

| Component | Responsibility | Constraints |
|---|---|---|
| **Google Gemma** *(via Ollama)* | Semantic understanding & plan extraction (NL → QueryPlan JSON). | • Sees **only** the semantic catalog (metrics, dimensions, unavailable concepts).<br>• Never sees database tables, raw DDL, or credentials.<br>• Never computes or formats final numbers.<br>• Temperature 0.0 with schema enforcement. |
| **Python Core** | Deterministic SQL compilation & AST validation. | • Validates plan against catalog metrics & dimensions.<br>• Rejects unknown fields & flags ambiguous queries.<br>• Compiles SQL from verified definitions & join paths.<br>• `sqlglot` parses AST to ensure read-only safety. |
| **Snowflake** | High-performance execution on cloud warehouse. | • Real query execution at question time.<br>• Zero mocked data, zero local copies of data.<br>• Runs under `DATAMIND_READONLY` role.<br>• Enforced session timeout (`STATEMENT_TIMEOUT_IN_SECONDS=30`). |

---

## 🚀 Quickstart & Setup on Ubuntu

### Prerequisites
- Ubuntu 22.04 LTS or 24.04 LTS
- Python 3.11 or 3.12
- [Ollama](https://ollama.com/) installed and running
- A Snowflake account with access to `SNOWFLAKE_SAMPLE_DATA`

### 1. Clone & Set Up Python Environment
```bash
git clone <repo-url> datamind
cd datamind

# Create virtual environment
python3.12 -m venv .venv
source .venv/bin/activate

# Install pinned dependencies
pip install -r requirements.txt
```

### 2. Set Up Local Gemma Model (Ollama)
```bash
# Start Ollama service (if not running)
ollama serve

# Pull the default Gemma model
ollama pull gemma3:1b
```

### 3. Configure Snowflake Role & Warehouse
Log in to your Snowflake web console or CLI and execute [`scripts/snowflake_setup.sql`](file:///home/abhilash/zeroday/scripts/snowflake_setup.sql):
```sql
USE ROLE ACCOUNTADMIN;

-- 1. Create dedicated read-only role
CREATE ROLE IF NOT EXISTS DATAMIND_READONLY;

-- 2. Create lightweight warehouse
CREATE WAREHOUSE IF NOT EXISTS DATAMIND_WH
    WITH WAREHOUSE_SIZE = 'XSMALL'
    AUTO_SUSPEND = 60
    AUTO_RESUME = TRUE
    INITIALLY_SUSPENDED = TRUE;

-- 3. Grant privileges
GRANT USAGE ON WAREHOUSE DATAMIND_WH TO ROLE DATAMIND_READONLY;
GRANT IMPORTED PRIVILEGES ON DATABASE SNOWFLAKE_SAMPLE_DATA TO ROLE DATAMIND_READONLY;

-- 4. Grant role to your user
GRANT ROLE DATAMIND_READONLY TO USER <YOUR_SNOWFLAKE_USER>;
```

### 4. Configure Environment Variables
Copy `.env.example` to `.env`:
```bash
cp .env.example .env
```
Edit `.env` with your Snowflake account details:
```ini
SNOWFLAKE_ACCOUNT=xy12345.us-east-1
SNOWFLAKE_USER=datamind_user
SNOWFLAKE_PASSWORD=your_secure_password
SNOWFLAKE_WAREHOUSE=DATAMIND_WH
SNOWFLAKE_DATABASE=SNOWFLAKE_SAMPLE_DATA
SNOWFLAKE_SCHEMA=TPCH_SF1
SNOWFLAKE_ROLE=DATAMIND_READONLY

OLLAMA_HOST=http://localhost:11434
OLLAMA_MODEL=gemma3:1b

DATAMIND_DATASET=tpch
DATAMIND_MAX_ROWS=500
DATAMIND_QUERY_TIMEOUT_S=30
```

### 5. Verify Connection with Smoke Test
Run the smoke test to verify connectivity to Snowflake:
```bash
python scripts/smoke_test.py
```
Output confirms:
- Connection to Snowflake
- Database, schema, and warehouse target
- Snowflake Query ID (`sfqid`)
- Elapsed time and real row count from `ORDERS` table

---

## 🖥️ Running the Application

Launch the Streamlit web interface:
```bash
streamlit run app.py
```
Open your browser at `http://localhost:8501`.

### Interactive Features:
- **Ask Natural Language Questions**: Type questions or click preset benchmark prompts.
- **Traceable Answers**: Clear markdown answer built strictly from result rows.
- **Traceability Expander**: Inspect the compiled SQL, Snowflake Query ID, elapsed time, and row count.
- **Result Data Table**: Interactive table displaying exact rows returned from Snowflake.
- **Dynamic Visualizations**: Auto-generates bar charts for ranking and time-series aggregations.
- **Query Plan Inspection**: Inspect the underlying structured JSON plan emitted by Gemma.

---

## 🧪 Testing & Evaluation

### Offline Unit Test Suite (28 Tests)
Runs without requiring Snowflake credentials or a running Ollama server:
```bash
pytest -v -m "not live"
```
Covers:
- `test_sql_validator.py`: Accepts valid `SELECT`; blocks DML (`INSERT`, `UPDATE`, `DELETE`), DDL (`DROP`, `CREATE`, `ALTER`), multi-statement attacks, unknown tables/columns, missing `LIMIT`.
- `test_query_planner.py`: Validates deterministic SQL compilation for ranking, filtering, multi-hop joins, time grouping, unknown field rejection, ambiguity handling, and unavailable concept handling.
- `test_failure_cases.py`: Mocks offline failure modes (Ollama down, invalid JSON, empty results, Snowflake errors, unsafe requests).

### Live Ground-Truth Correctness Tests
Runs against your live Snowflake instance to compare pipeline results against hand-written SQL:
```bash
pytest -v -m "live"
```

### Benchmark Evaluation Suite
Evaluates the benchmark suite across all 10 required question categories:
```bash
python scripts/run_eval.py
```
Outputs execution results and Snowflake Query IDs to `evaluation/results.json`.

---

## 🧩 Adding a New Dataset (Config-Only)

DataMind is completely config-driven. To add a new dataset (e.g. Our World in Data, internal sales), add a new entry to [`config/datasets.yaml`](file:///home/abhilash/zeroday/config/datasets.yaml) under `datasets:`:

```yaml
datasets:
  my_new_dataset:
    name: "My New Dataset"
    description: "Enterprise operational reporting"
    database: "MY_DATABASE"
    schema: "PUBLIC"
    date_dimension:
      column: "TRANSACTION_DATE"
      table: "TRANSACTIONS"
      allowed_grains: ["year", "quarter", "month", "day"]
    tables:
      TRANSACTIONS:
        description: "Sales transactions"
        columns:
          TRANSACTION_ID: "Unique identifier"
          AMOUNT: "Total transaction amount"
    join_paths: []
    metrics:
      total_sales:
        description: "Sum of transaction amounts: SUM(AMOUNT)"
        expression: "SUM(TRANSACTIONS.AMOUNT)"
        required_tables: ["TRANSACTIONS"]
    dimensions:
      transaction_id:
        description: "Identifier"
        expression: "TRANSACTIONS.TRANSACTION_ID"
        table: "TRANSACTIONS"
    unavailable_concepts:
      profit: "Cost of goods sold is not tracked in this table."
```
Select the new dataset directly from the Streamlit sidebar dropdown without writing any Python code.

---

## 📋 Honest Known Limitations

1. **Aggregation Scope**: Currently supports single metric aggregations per query (or single metric grouped by dimensions). Multi-metric pivot tables are not supported in v1.
2. **Date Granularity**: Date grouping is restricted to the dataset's configured date dimension (`O_ORDERDATE` in TPC-H) with supported grains (`year`, `quarter`, `month`, `day`).
3. **Historical Queries Only**: DataMind queries existing historical data. It does not perform machine-learning forecasting, scenario simulations, or predictive modeling.
4. **Row Limit**: All queries enforce a hard limit (`DATAMIND_MAX_ROWS`, default 500) to prevent unbounded memory consumption or runaway queries.
5. **No Cross-Dataset Federation**: Queries execute against one dataset at a time. Joins across multiple datasets/catalogs are not supported.

---

## ⚖️ TPC-H Benchmark License Note

This repository queries the pre-installed standard sample data in Snowflake (`SNOWFLAKE_SAMPLE_DATA.TPCH_SF1`) in-place. **No TPC-H benchmark data files are redistributed or stored within this repository.**

---

## 📄 License

Licensed under the Apache License, Version 2.0. See [`LICENSE`](file:///home/abhilash/zeroday/LICENSE) for details.
