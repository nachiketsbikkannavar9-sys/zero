# DataMind Architecture & Traceability

"DataMind – Talk to Your Data" is a zero-hallucination semantic data assistant designed for ZeroDay OSS Hacktoberfest 2026 (Problem Statement 1).

The core tenet is strict separation between probabilistic reasoning (language models) and deterministic data operations (SQL compilation and real database execution). Numbers come **only** from Snowflake query result rows, never from language model generations.

---

## 1. End-to-End Pipeline Workflow

```mermaid
sequenceDiagram
    autonumber
    actor User
    participant Streamlit as Streamlit UI (app.py)
    participant Pipeline as DataMind Pipeline (pipeline.py)
    participant Guard as Pre-flight Safety Guard
    participant Gemma as Google Gemma via Ollama (gemma_model.py)
    participant Planner as Deterministic Planner (query_planner.py)
    participant Validator as sqlglot AST Validator (sql_validator.py)
    participant Snowflake as Snowflake Data Cloud (snowflake_client.py)
    participant Answer as Answer Generator (answer_generator.py)

    User->>Streamlit: Enters natural language question
    Streamlit->>Pipeline: ask(question)
    
    Pipeline->>Guard: Check for destructive commands (DROP, DELETE, TRUNCATE)
    alt Destructive Request
        Guard-->>Pipeline: Reject immediately
        Pipeline-->>Streamlit: Return security rejection (Never touches Snowflake)
    end

    Pipeline->>Gemma: Send Catalog Prompt + Question + JSON Schema (format)
    Gemma-->>Pipeline: Return structured QueryPlan JSON
    
    Pipeline->>Planner: validate_plan(plan, question)
    alt Ambiguous Question
        Planner-->>Pipeline: Return status='needs_clarification' + question
    else Unavailable Concept (profit, cost, churn)
        Planner-->>Pipeline: Return status='unsupported' + reason
    else Valid Plan
        Pipeline->>Planner: compile_sql(plan)
        Planner-->>Pipeline: Return deterministic SQL + assumptions
        
        Pipeline->>Validator: validate(sql)
        alt Invalid SQL or Non-SELECT
            Validator-->>Pipeline: Reject with AST safety reason
        else Safe SELECT
            Pipeline->>Snowflake: execute_query(sql) [STATEMENT_TIMEOUT_IN_SECONDS=30]
            Snowflake-->>Pipeline: Return real rows (DataFrame) + sfqid (Query ID)
            
            Pipeline->>Answer: build_deterministic_answer(df, plan, assumptions)
            Answer-->>Pipeline: Traceable text answer
        end
    end

    Pipeline-->>Streamlit: PipelineResult (Answer + Evidence audit trail)
    Streamlit-->>User: Displays Answer, Verified Rows, Query ID, Executed SQL, Chart
```

---

## 2. The Claim -> SQL -> Rows Traceability Chain

Every insight displayed to the user follows a verifiable audit trail:

```mermaid
graph TD
    A["Natural Language Claim<br/><i>'Top 5 parts by revenue'</i>"] --> B["Structured QueryPlan<br/><code>metric='revenue', dimensions=['part_name'], limit=5</code>"]
    B --> C["Deterministic SQL<br/><code>SELECT PART.P_NAME, SUM(...) FROM LINEITEM JOIN PART ... LIMIT 5</code>"]
    C --> D["Snowflake Query Execution<br/><b>Snowflake Query ID (sfqid): 01ba...</b>"]
    D --> E["Verified Result Rows<br/><b>5 rows returned from SNOWFLAKE_SAMPLE_DATA</b>"]
    E --> F["Deterministic Answer<br/><i>'1. Part#123 → REVENUE: 45,210.50'</i>"]
    
    style A fill:#e1f5fe,stroke:#0288d1
    style B fill:#e8eaf6,stroke:#3f51b5
    style C fill:#fff8e1,stroke:#fbc02d
    style D fill:#ede7f6,stroke:#512da8
    style E fill:#e8f5e9,stroke:#388e3c
    style F fill:#e0f2f1,stroke:#00796b
```

### Traceability Guarantees:
1. **Provenance Dataclass (`Evidence`)**:
   Contains `question`, `plan_json`, `executed_sql`, `snowflake_query_id`, `row_count`, `elapsed_time_seconds`, `result_df`, `assumptions`, and `model_name`.
2. **Zero Model Number Generation**:
   The model never formats, computes, or fabricates numbers. All aggregations are performed inside Snowflake's analytical engine.
3. **Double Verification**:
   If an optional natural language explanation is requested from Gemma, the `AnswerGenerator.verify_numbers_against_df()` checks every extracted number against the returned dataframe. If any number fails to match, the LLM text is discarded and only deterministic rows are shown.

---

## 3. Defense-in-Depth Security Architecture

DataMind enforces safety across five distinct layers:

| Layer | Component | Mechanism |
|---|---|---|
| **1. Pre-flight Guard** | `pipeline.py` | Regex pattern matching rejecting DDL/DML intentions before any LLM or DB call. |
| **2. Catalog Isolation** | `gemma_model.py` | Prompt receives **only** the semantic catalog (metrics, dimensions, unavailable concepts). No table names, raw DDL, or database connection strings are ever sent to the model. |
| **3. Deterministic Compilation** | `query_planner.py` | Code generates SQL using predefined expressions from `config/datasets.yaml` and safe graph-based joins. Parameterized literals and string-escaped filters prevent injection. |
| **4. AST SQL Validator** | `sql_validator.py` | `sqlglot` parses Snowflake AST. Strictly requires: exactly one statement; must be a `SELECT`; forbids all DDL/DML; verifies every table and column against catalog; enforces mandatory `LIMIT <= max_rows`. |
| **5. Database Hardening** | Snowflake Role | Snowflake role `DATAMIND_READONLY` granted with `IMPORTED PRIVILEGES` on `SNOWFLAKE_SAMPLE_DATA`. Session statement timeout `STATEMENT_TIMEOUT_IN_SECONDS=30`. |

---

## 4. Semantic Catalog Extensibility

The semantic catalog in `config/datasets.yaml` defines:
- **`database` & `schema`**: Snowflake physical location.
- **`tables` & `columns`**: Physical schema structure.
- **`join_paths`**: Explicit foreign-key relationships.
- **`metrics`**: Standard mathematical definitions and required tables (e.g. `revenue = SUM(L_EXTENDEDPRICE * (1 - L_DISCOUNT))`).
- **`dimensions`**: Categorical and geographical attributes.
- **`date_dimension`**: Date tracking with allowed grains (`year`, `quarter`, `month`, `day`).
- **`unavailable_concepts`**: Explicit list of concepts that cannot be computed (e.g. `profit`, `cost`, `churn`), allowing DataMind to gracefully refuse without hallucinating answers.

New datasets (such as Our World in Data) are added purely through YAML configuration without altering a single line of application code.
