"""DataMind – Talk to Your Data (Streamlit Application).

Zero-hallucination semantic data assistant.
Natural language -> Gemma structured query plan -> Catalog validation -> Deterministic SQL
-> Snowflake execution -> Traceable answer with query ID & result rows.
"""

import os
import streamlit as st
import pandas as pd
from dotenv import load_dotenv

load_dotenv()

from src.pipeline import DataMindPipeline
from src.semantic_catalog import get_catalog
from src.snowflake_client import SnowflakeClient

st.set_page_config(
    page_title="DataMind – Talk to Your Data",
    page_icon="🧠",
    layout="wide",
)

# Initialize catalog and datasets
catalog = get_catalog()
available_datasets = list(catalog.datasets.keys())
default_dataset = os.getenv("DATAMIND_DATASET", "tpch").lower()
default_idx = available_datasets.index(default_dataset) if default_dataset in available_datasets else 0

# --- Sidebar ---
st.sidebar.title("🧠 DataMind Engine")
st.sidebar.caption("Deterministic Natural Language to Snowflake")

selected_dataset_id = st.sidebar.selectbox(
    "Active Dataset",
    available_datasets,
    index=default_idx,
    format_func=lambda x: catalog.get_dataset(x).name,
)

current_dataset = catalog.get_dataset(selected_dataset_id)

st.sidebar.markdown("---")
st.sidebar.subheader("Dataset Configuration")
st.sidebar.text(f"Database: {current_dataset.database}")
st.sidebar.text(f"Schema:   {current_dataset.schema_name}")

with st.sidebar.expander("Available Metrics", expanded=False):
    for m_name, m_cfg in current_dataset.metrics.items():
        st.markdown(f"**`{m_name}`**\n*{m_cfg.description}*")

with st.sidebar.expander("Available Dimensions", expanded=False):
    for d_name, d_cfg in current_dataset.dimensions.items():
        st.markdown(f"**`{d_name}`**\n*{d_cfg.description}*")

with st.sidebar.expander("Unavailable Concepts", expanded=False):
    for u_name, u_reason in current_dataset.unavailable_concepts.items():
        st.markdown(f"**`{u_name}`**\n*{u_reason}*")

st.sidebar.markdown("---")
st.sidebar.subheader("System Info")
model_name = os.getenv("OLLAMA_MODEL", "gemma3:1b")
ollama_host = os.getenv("OLLAMA_HOST", "http://localhost:11434")
st.sidebar.text(f"Planner Model: {model_name}")
st.sidebar.text(f"Ollama Host:   {ollama_host}")
sf_account = os.getenv("SNOWFLAKE_ACCOUNT", "[Not Configured]")
st.sidebar.text(f"Snowflake Acct: {sf_account}")
st.sidebar.text(f"Warehouse:     {os.getenv('SNOWFLAKE_WAREHOUSE', 'DATAMIND_WH')}")
st.sidebar.text(f"Role:          {os.getenv('SNOWFLAKE_ROLE', 'DATAMIND_READONLY')}")

# --- Main Page ---
st.title("DataMind – Talk to Your Data")
st.markdown(
    "Ask a plain-English question. Google Gemma creates a structured query plan, "
    "DataMind validates it and generates deterministic SQL, executes on Snowflake, "
    "and returns answers verified directly against Snowflake rows."
)

# Preset prompt buttons
st.markdown("**Sample Questions:**")
sample_cols = st.columns(3)
preset_question = None

with sample_cols[0]:
    if st.button("Top 5 parts by revenue", use_container_width=True):
        preset_question = "What are the top 5 parts by revenue?"
    if st.button("Revenue by region", use_container_width=True):
        preset_question = "What is the total revenue by region?"

with sample_cols[1]:
    if st.button("Monthly revenue for 1995", use_container_width=True):
        preset_question = "Show monthly revenue for 1995"
    if st.button("Best performing items (Ambiguity test)", use_container_width=True):
        preset_question = "Show me our best performing items"

with sample_cols[2]:
    if st.button("Total profit in 1995 (Unavailable field)", use_container_width=True):
        preset_question = "What was our total profit in 1995?"
    if st.button("Delete all customers (Safety test)", use_container_width=True):
        preset_question = "Delete all customers"

# Input text box
default_val = preset_question if preset_question else ""
question_input = st.text_input(
    "Ask a question about your data:",
    value=default_val,
    placeholder="e.g. What are the top 5 parts by revenue?",
    key="question_input_box",
)

if st.button("Execute Query", type="primary") or preset_question:
    if not question_input.strip():
        st.warning("Please enter a question.")
    else:
        with st.spinner("Analyzing catalog, planning query, and executing on Snowflake..."):
            pipeline = DataMindPipeline(dataset_id=selected_dataset_id)
            result = pipeline.ask(question_input)

        st.markdown("---")
        st.subheader("Answer")
        st.markdown(result.answer)

        ev = result.evidence

        # Display SQL & Snowflake traceability
        if ev.executed_sql:
            with st.expander("🔍 Executed SQL & Snowflake Query ID", expanded=True):
                st.code(ev.executed_sql, language="sql")
                cols = st.columns(3)
                with cols[0]:
                    st.metric("Snowflake Query ID", ev.snowflake_query_id or "N/A")
                with cols[1]:
                    st.metric("Elapsed Time", f"{ev.elapsed_time_seconds:.3f} s")
                with cols[2]:
                    st.metric("Rows Returned", ev.row_count)

        # Display Data Table and optional Bar Chart
        if ev.result_df is not None and not ev.result_df.empty:
            df = ev.result_df
            with st.expander("📊 Snowflake Result Table", expanded=True):
                st.dataframe(df, use_container_width=True)

            # Auto-chart if applicable
            numeric_cols = df.select_dtypes(include=["number"]).columns.tolist()
            non_numeric_cols = [c for c in df.columns if c not in numeric_cols]

            if numeric_cols and len(df) > 1 and len(df) <= 100:
                with st.expander("📈 Chart Visualization", expanded=True):
                    val_col = numeric_cols[0]
                    cat_col = non_numeric_cols[0] if non_numeric_cols else df.columns[0]
                    try:
                        chart_data = df[[cat_col, val_col]].set_index(cat_col)
                        st.bar_chart(chart_data)
                    except Exception:
                        pass

        # Query Plan Inspection
        if ev.plan_json:
            with st.expander("📋 Structured Query Plan (JSON)", expanded=False):
                st.json(ev.plan_json)

        # Assumptions
        if ev.assumptions:
            with st.expander("ℹ️ Assumptions & Metric Definitions", expanded=False):
                for a in ev.assumptions:
                    st.markdown(f"- {a}")
