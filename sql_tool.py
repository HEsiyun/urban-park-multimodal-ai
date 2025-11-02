# sql_tool.py
from __future__ import annotations
import os, time
from datetime import datetime
from typing import Any, Dict, List, Optional, Callable
import torch

import duckdb, pandas as pd
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoModelForSeq2SeqLM, pipeline
from config import DUCK_FILE, LABOR_XLSX, LABOR_SHEET
from rag import RAG

try:
    from transformers import T5ForConditionalGeneration, T5Tokenizer, AutoTokenizer
except Exception:
    T5ForConditionalGeneration = None
    T5Tokenizer = None

_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
_t5_model: Optional[T5ForConditionalGeneration] = None
_t5_tokenizer: Optional[T5Tokenizer] = None
# -----------------------------
# Config (adjust paths as needed)
# -----------------------------
DATA_DIR = os.path.abspath("data")
DUCK_FILE = os.path.join(DATA_DIR, "mowing.duckdb")
LABOR_XLSX = os.path.join(DATA_DIR, "6 Mowing Reports to Jun 20 2025.xlsx")
LABOR_SHEET = 0  # or sheet name string

# -----------------------------
# DuckDB bootstrap
# -----------------------------
def _ensure_duck() -> duckdb.DuckDBPyConnection:
    """Create a DuckDB connection and load the latest Excel into a table."""
    con = duckdb.connect(DUCK_FILE)

    # Read Excel fresh (simple & explicit for MVP)
    df = pd.read_excel(LABOR_XLSX, sheet_name=LABOR_SHEET)

    # Load field size data
    field_size_path = os.path.join(DATA_DIR, "3 vsfs_master_inventory_fieldsizes.xlsx")
    diamond_field_size_df = pd.read_excel(field_size_path, sheet_name=1)
    diamond_field_size_df = diamond_field_size_df.fillna("None")
    rectangular_field_size_df = pd.read_excel(field_size_path, sheet_name=0)
    rectangular_field_size_df = rectangular_field_size_df.fillna("None")
    # Normalize column names
    df.columns = [str(c).strip() for c in df.columns]

    # Parse dates robustly (avoid out-of-bounds errors)
    if "Posting Date" in df.columns:
        # try a fast path, then a tolerant path
        try:
            df["Posting Date"] = pd.to_datetime(df["Posting Date"], format="%m/%d/%y", errors="coerce")
        except Exception:
            pass
        df["Posting Date"] = pd.to_datetime(df["Posting Date"], errors="coerce")
        df = df.dropna(subset=["Posting Date"])

    # Ensure numeric cost
    if "Val.in rep.cur." in df.columns:
        df["Val.in rep.cur."] = pd.to_numeric(df["Val.in rep.cur."], errors="coerce").fillna(0.0)

    # Re-create table
    con.execute("DROP TABLE IF EXISTS labor_data")
    con.execute("DROP TABLE IF EXISTS diamond_field_size_data")
    con.execute("DROP TABLE IF EXISTS rectangular_field_size_data")
    con.register("labor_df", df)
    con.register("diamond_field_size_data", diamond_field_size_df)
    con.register("rectangular_field_size_data", rectangular_field_size_df)
    con.execute("CREATE TABLE labor_data AS SELECT * FROM labor_df")
    con.execute("CREATE TABLE diamond_field_size_data AS SELECT * FROM diamond_field_size_data")
    con.execute("CREATE TABLE rectangular_field_size_data AS SELECT * FROM rectangular_field_size_data")
    return con

# -----------------------------
# Template implementations
# -----------------------------
def _tpl_mowing_labor_cost_month_top1(con: duckdb.DuckDBPyConnection, params: Dict[str, Any]) -> Dict[str, Any]:
    """
    Returns the park with the highest total mowing labor cost in the given month/year.
    Expects params: { "month": int (1-12), "year": int }
    """
    month = params.get("month")
    year = params.get("year")

    # Validate defaults if missing
    if not isinstance(month, int) or month < 1 or month > 12:
        month = datetime.utcnow().month
    if not isinstance(year, int) or year < 2000 or year > 2100:
        year = datetime.utcnow().year

    sql = f"""
    WITH month_data AS (
      SELECT
        "CO Object Name" AS park,
        CAST("Val.in rep.cur." AS DOUBLE) AS cost,
        "Posting Date"::TIMESTAMP AS posting_dt
      FROM labor_data
    )
    SELECT park, SUM(cost) AS total_cost
    FROM month_data
    WHERE year(posting_dt) = {year}
      AND month(posting_dt) = {month}
    GROUP BY park
    ORDER BY total_cost DESC
    LIMIT 1;
    """
    t0 = time.time()
    rows = con.execute(sql).fetchdf().to_dict(orient="records")
    elapsed = int((time.time() - t0) * 1000)
    return {"rows": rows, "rowcount": len(rows), "elapsed_ms": elapsed}


def _tpl_mowing_last_date_by_park(con: duckdb.DuckDBPyConnection, params: Dict[str, Any]) -> Dict[str, Any]:
    """
    Returns the most recent mowing date for a specific park or all parks.
    Expects params: { "park_name": str (optional) }
    
    Example queries:
    - "When was the last mowing at Cambridge Park?"
    - "Show me the most recent mowing date for each park"
    """
    park_name = params.get("park_name")
    
    if park_name:
        # Specific park query
        sql = f"""
        SELECT 
            "CO Object Name" AS park,
            MAX("Posting Date") AS last_mowing_date,
            COUNT(*) AS total_mowing_sessions,
            SUM(CAST("Val.in rep.cur." AS DOUBLE)) AS total_cost
        FROM labor_data
        WHERE LOWER("CO Object Name") LIKE LOWER('%{park_name}%')
        GROUP BY "CO Object Name"
        ORDER BY last_mowing_date DESC
        LIMIT 1;
        """
    else:
        # All parks query
        sql = """
        SELECT 
            "CO Object Name" AS park,
            MAX("Posting Date") AS last_mowing_date,
            COUNT(*) AS total_sessions,
            SUM(CAST("Val.in rep.cur." AS DOUBLE)) AS total_cost
        FROM labor_data
        GROUP BY "CO Object Name"
        ORDER BY last_mowing_date DESC;
        """
    
    t0 = time.time()
    rows = con.execute(sql).fetchdf().to_dict(orient="records")
    elapsed = int((time.time() - t0) * 1000)
    return {"rows": rows, "rowcount": len(rows), "elapsed_ms": elapsed}


def _tpl_mowing_cost_trend(con: duckdb.DuckDBPyConnection, params: Dict[str, Any]) -> Dict[str, Any]:
    """
    Returns monthly mowing cost trend for a date range.
    Expects params: { 
        "start_month": int (1-12), 
        "end_month": int (1-12), 
        "year": int,
        "park_name": str (optional - for specific park trend)
    }
    
    Example queries:
    - "Show mowing cost trend from January to June 2025"
    - "How did costs change from April to June 2025 for Cambridge Park?"
    """
    start_month = params.get("start_month")
    end_month = params.get("end_month")
    year = params.get("year") or params.get("range_year")
    park_name = params.get("park_name")
    
    # Defaults
    if not isinstance(year, int) or year < 2000:
        year = datetime.utcnow().year
    if not isinstance(start_month, int) or start_month < 1 or start_month > 12:
        start_month = 1
    if not isinstance(end_month, int) or end_month < 1 or end_month > 12:
        end_month = 12
    
    # Build WHERE clause for park filter
    park_filter = ""
    if park_name:
        park_filter = f"AND LOWER(\"CO Object Name\") LIKE LOWER('%{park_name}%')"
    
    sql = f"""
    WITH monthly_costs AS (
        SELECT 
            year("Posting Date") AS year,
            month("Posting Date") AS month,
            "CO Object Name" AS park,
            SUM(CAST("Val.in rep.cur." AS DOUBLE)) AS monthly_cost,
            COUNT(*) AS session_count
        FROM labor_data
        WHERE year("Posting Date") = {year}
          AND month("Posting Date") BETWEEN {start_month} AND {end_month}
          {park_filter}
        GROUP BY year("Posting Date"), month("Posting Date"), "CO Object Name"
    )
    SELECT 
        year,
        month,
        park,
        monthly_cost,
        session_count
    FROM monthly_costs
    ORDER BY year, month, park;
    """
    
    t0 = time.time()
    rows = con.execute(sql).fetchdf().to_dict(orient="records")
    elapsed = int((time.time() - t0) * 1000)
    return {"rows": rows, "rowcount": len(rows), "elapsed_ms": elapsed, "chart_type": "line"}


def _tpl_mowing_cost_by_park_month(con: duckdb.DuckDBPyConnection, params: Dict[str, Any]) -> Dict[str, Any]:
    """
    Returns cost comparison across all parks for a specific month.
    Expects params: { "month": int (1-12), "year": int }
    
    Example queries:
    - "Compare mowing costs across all parks in March 2025"
    - "Show total mowing costs breakdown by park in April"
    """
    month = params.get("month")
    year = params.get("year")
    
    # Defaults
    if not isinstance(month, int) or month < 1 or month > 12:
        month = datetime.utcnow().month
    if not isinstance(year, int) or year < 2000 or year > 2100:
        year = datetime.utcnow().year
    
    sql = f"""
    SELECT 
        "CO Object Name" AS park,
        SUM(CAST("Val.in rep.cur." AS DOUBLE)) AS total_cost,
        COUNT(*) AS mowing_sessions,
        AVG(CAST("Val.in rep.cur." AS DOUBLE)) AS avg_cost_per_session,
        SUM("Total quantity") AS total_quantity
    FROM labor_data
    WHERE year("Posting Date") = {year}
      AND month("Posting Date") = {month}
    GROUP BY "CO Object Name"
    ORDER BY total_cost DESC;
    """
    
    t0 = time.time()
    rows = con.execute(sql).fetchdf().to_dict(orient="records")
    elapsed = int((time.time() - t0) * 1000)
    return {"rows": rows, "rowcount": len(rows), "elapsed_ms": elapsed, "chart_type": "bar"}


def _tpl_mowing_cost_breakdown_by_park(con: duckdb.DuckDBPyConnection, params: Dict[str, Any]) -> Dict[str, Any]:
    """
    Returns detailed monthly cost breakdown for a specific park or all parks.
    Expects params: { 
        "park_name": str (optional),
        "month": int (optional - if not provided, shows all months),
        "year": int
    }
    
    Example queries:
    - "Show total mowing costs breakdown by park in March 2025"
    - "What's the cost breakdown for Cambridge Park?"
    """
    park_name = params.get("park_name")
    month = params.get("month")
    year = params.get("year")
    
    # Defaults
    if not isinstance(year, int) or year < 2000 or year > 2100:
        year = datetime.utcnow().year
    
    # Build WHERE clause
    where_parts = [f"year(\"Posting Date\") = {year}"]
    
    if park_name:
        where_parts.append(f"LOWER(\"CO Object Name\") LIKE LOWER('%{park_name}%')")
    
    if isinstance(month, int) and 1 <= month <= 12:
        where_parts.append(f"month(\"Posting Date\") = {month}")
    
    where_clause = " AND ".join(where_parts)
    
    sql = f"""
    SELECT 
        "CO Object Name" AS park,
        month("Posting Date") AS month,
        "ParActivity" AS activity_type,
        SUM(CAST("Val.in rep.cur." AS DOUBLE)) AS cost,
        COUNT(*) AS sessions,
        SUM("Total quantity") AS total_quantity
    FROM labor_data
    WHERE {where_clause}
    GROUP BY "CO Object Name", month("Posting Date"), "ParActivity"
    ORDER BY park, month, cost DESC;
    """
    
    t0 = time.time()
    rows = con.execute(sql).fetchdf().to_dict(orient="records")
    elapsed = int((time.time() - t0) * 1000)
    return {"rows": rows, "rowcount": len(rows), "elapsed_ms": elapsed}

def _tpl_get_diamond_dimensions(con: duckdb.DuckDBPyConnection, params: Dict[str, Any]) -> Dict[str, Any]:
    """
    Returns diamond field dimensions from the diamond_field_size_data table.
    """
    sql = """
    SELECT "Name of Field ", "Diamonds: Dimension Home to Pitchers Plate - m ", "Diamonds: Home to First Base Path - m "
    FROM diamond_field_size_data
    LIMIT 10;
    """
    t0 = time.time()
    rows = con.execute(sql).fetchdf().to_dict(orient="records")
    elapsed = int((time.time() - t0) * 1000)
    return {"rows": rows, "rowcount": len(rows), "elapsed_ms": elapsed}

def _tpl_get_rectangular_dimensions(con: duckdb.DuckDBPyConnection, params: Dict[str, Any]) -> Dict[str, Any]:
    """
    Returns rectangular field dimensions from the rectangular_field_size_data table.
    """
    sql = """
    SELECT "Name of Field ", "Rectangular Field Dimension: Length - m ", "Rectangular Field Dimension: Width - m "
    FROM rectangular_field_size_data
    LIMIT 10;
    """
    t0 = time.time()
    rows = con.execute(sql).fetchdf().to_dict(orient="records")
    elapsed = int((time.time() - t0) * 1000)
    return {"rows": rows, "rowcount": len(rows), "elapsed_ms": elapsed}

# -----------------------------
# Dispatcher registry
# -----------------------------
TEMPLATE_REGISTRY: Dict[str, Callable[[duckdb.DuckDBPyConnection, Dict[str, Any]], Dict[str, Any]]] = {
    # Original template
    "mowing.labor_cost_month_top1": _tpl_mowing_labor_cost_month_top1,
    
    # NEW: 最近除草时间
    "mowing.last_mowing_date": _tpl_mowing_last_date_by_park,
    
    # NEW: 成本趋势图
    "mowing.cost_trend": _tpl_mowing_cost_trend,
    
    # NEW: 公园对比（单月）
    "mowing.cost_by_park_month": _tpl_mowing_cost_by_park_month,
    
    # NEW: 月度总览（详细分解）
    "mowing.cost_breakdown": _tpl_mowing_cost_breakdown_by_park,
    # NEW: 运动场地尺寸查询
    "field_dimension.rectangular": _tpl_get_rectangular_dimensions,
    "field_dimension.diamond": _tpl_get_diamond_dimensions,
}

# -----------------------------
# Public entry point
# -----------------------------
def run_sql_template(template: str, params: Dict[str, Any]) -> Dict[str, Any]:
    """
    Unified SQL template executor.
    Usage: run_sql_template("mowing.labor_cost_month_top1", {"month": 3, "year": 2025})
    Returns: {"rows": [...], "rowcount": int, "elapsed_ms": int}
    """
    if template not in TEMPLATE_REGISTRY:
        return {
            "rows": [{"error": f"Unknown SQL template: {template}"}],
            "rowcount": 1,
            "elapsed_ms": 0,
        }

    con = _ensure_duck()
    try:
        func = TEMPLATE_REGISTRY[template]
        out = func(con, params or {})
        return out
    finally:
        try:
            con.close()
        except Exception:
            pass

# -----------------------------
# RAG-compatible wrapper (for executor.py)
# -----------------------------
def sql_query_rag(template: str = None, params: Dict[str, Any] = None, 
                  support: list = None, **kwargs) -> Dict[str, Any]:
    """
    Wrapper for RAG executor compatibility.
    
    Args:
        template: SQL template name (e.g., "mowing.labor_cost_month_top1")
        params: Template parameters
        support: Optional support documents (for future use)
        **kwargs: Additional arguments (merged into params)
    
    Returns:
        Dict with keys: rows, rowcount, elapsed_ms, support
    """
    # Merge kwargs into params
    if params is None:
        params = {}
    params.update(kwargs)
    
    # Default template if not specified
    if template is None:
        template = "mowing.labor_cost_month_top1"
    
    # Run the template
    result = run_sql_template(template, params)
    
    # Add support field for RAG compatibility
    result["support"] = support or []
    
    return result


# ✅ 最近除草时间
"When was the last mowing at Cambridge Park?"
"Show me the most recent mowing date for each park"

# ✅ 成本趋势
"Show mowing cost trend from January to June 2025"
"How did costs change from April to June for Garden Park?"

# ✅ 公园对比
"Compare mowing costs across all parks in March 2025"
"Show me all parks ranked by mowing cost in April"

# ✅ 详细分解
"Show total mowing costs breakdown by park in March 2025"
"What's the cost breakdown for Cambridge Park?"

# ✅ 原有功能
"Which park had the highest mowing cost in March 2025?"
"What are the mowing safety requirements?"

def _init_t5():
    global _t5_model, _t5_tokenizer
    if _t5_model is not None and _t5_tokenizer is not None:
        return
    if T5ForConditionalGeneration is None or (T5Tokenizer is None and AutoTokenizer is None):
        raise RuntimeError("transformers not available; pip install transformers[torch]")

    model_id = "suriya7/t5-base-text-to-sql"
    # simple load using AutoTokenizer (fallback to T5Tokenizer supported by HF)
    if AutoTokenizer is not None:
        _t5_tokenizer = AutoTokenizer.from_pretrained(model_id, use_fast=False)
    else:
        _t5_tokenizer = T5Tokenizer.from_pretrained(model_id)

    _t5_model = T5ForConditionalGeneration.from_pretrained(model_id).to(_device)
    _t5_model.eval()
    try:
        setattr(_t5_model, "name_or_path", model_id)
    except Exception:
        pass

def generate_sql_with_t5(
    nl_query: str,
    table_schema: Optional[str] = None,
    max_length: int = 256,
    num_beams: int = 3,
    num_return_sequences: int = 1,
    model_name: str = "suriya7/t5-base-text-to-sql"
) -> List[str]:
    """
    Convert a natural-language query to SQL using T5.
    """
    # allow switching model if needed (re-init if different)

    _init_t5()
    global _t5_model, _t5_tokenizer
    if _t5_model is None or getattr(_t5_model, "name_or_path", None) != model_name:
        # re-init with requested model
        _t5_model = None
        _t5_tokenizer = None
        # load the requested model
        if T5ForConditionalGeneration is None or T5Tokenizer is None:
            raise RuntimeError("transformers not available; pip install transformers[torch]")
        _t5_tokenizer = T5Tokenizer.from_pretrained(model_name)
        _t5_model = T5ForConditionalGeneration.from_pretrained(model_name).to(_device)
        _t5_model.eval()
        setattr(_t5_model, "name_or_path", model_name)
    print("i am here")
    # build few-shot prompt
    examples = [
        ("Which park had the highest mowing cost in January 2025?",
         """
        SELECT park, mowing_cost
        FROM labor_data
        WHERE year(posting_dt) = 2025
          AND month(posting_dt) = 1
        GROUP BY park
        ORDER BY mowing_cost DESC
        LIMIT 1;
        """),
        ("Which park had the highest mowing cost in February 2025?",
         """
        SELECT park, mowing_cost
        FROM labor_data
        WHERE year(posting_dt) = 2025
          AND month(posting_dt) = 2
        GROUP BY park
        ORDER BY mowing_cost DESC
        LIMIT 1;
        """),
        ("Which park had the highest mowing cost in January 2026?",
         """
        SELECT park, mowing_cost
        FROM labor_data
        WHERE year(posting_dt) = 2026
          AND month(posting_dt) = 1
        GROUP BY park
        ORDER BY mowing_cost DESC
        LIMIT 1;
        """)
    ]
    few_shot = "\n\n".join([f"NL: {q}\nSQL: {s}" for q, s in examples])

    header = f"Schema: {table_schema}\n\n" if table_schema else ""
    prompt = f"{header}Translate the following natural language to executable SQL. Provide only the SQL (no explanation).\n\n{few_shot}\n\nNL: {nl_query}\nSQL:"

    # defensive checks / debug info
    if _t5_tokenizer is None:
        raise RuntimeError("tokenizer not initialized")

    try:
        # tokenize, then move tensors to device explicitly (avoid calling .to on BatchEncoding)
        inputs = _t5_tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512)
        inputs = {k: v.to(_device) for k, v in inputs.items()}
    except Exception as e:
        # provide clearer debug info
        raise RuntimeError(f"tokenizer failed: {e}; prompt type={type(prompt)} model_name={type(model_name)}") from e

    gen_kwargs = {
        "max_length": max_length,
        "num_beams": num_beams,
        "early_stopping": True,
        "num_return_sequences": max(1, num_return_sequences),
    }
    with torch.no_grad():
        outputs = _t5_model.generate(**inputs, **gen_kwargs)

    results = []
    for out in outputs:
        sql = _t5_tokenizer.decode(out, skip_special_tokens=True).strip()
        if nl_query.lower() in sql.lower() or sql.startswith(prompt[: min(200, len(prompt))]):
            results.append("") 
        else:
            results.append(sql)
    print("Generated SQL:", results)
    return results

# Example usage (in code):
# sqls = generate_sql_with_t5("Which park had the highest mowing cost in March 2025?",
#                            table_schema="labor_data(posting_dt TIMESTAMP, CO Object Name TEXT, Val.in rep.cur. DOUBLE)")
# print(sqls[0])

def generate_with_gaussalgo():
    model_path = 'gaussalgo/T5-LM-Large-text2sql-spider'
    model = AutoModelForSeq2SeqLM.from_pretrained(model_path)
    tokenizer = AutoTokenizer.from_pretrained(model_path)

    question = "Which park has most mowing cost in March 2025?"
    schema = """
    "park_mowing" "Park_ID" int , "mowing_cost" double , "Park_Name" text , "Posting_Date" date
    """

    input_text = " ".join(["Question: ",question, "Schema:", schema])

    model_inputs = tokenizer(input_text, return_tensors="pt")
    outputs = model.generate(**model_inputs, max_length=512)

    output_text = tokenizer.batch_decode(outputs, skip_special_tokens=True)
    return output_text
def get_tokenizer_model(model_name):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        trust_remote_code=True,
        torch_dtype=torch.float16,
        device_map="auto",
        use_cache=True,
    )
    return tokenizer, model

def generate_prompt(question, prompt_file="prompt.md", metadata_file="metadata.sql"):
    with open(prompt_file, "r") as f:
        prompt = f.read()
    
    with open(metadata_file, "r") as f:
        table_metadata_string = f.read()

    prompt = prompt.format_map({"user_question": question, "table_metadata_string_DDL": table_metadata_string})
    return prompt

def generate_sql_with_sqlcoder(question, prompt_file="prompt.md", metadata_file="metadata.sql"):
    tokenizer, model = get_tokenizer_model("defog/sqlcoder-7b-2")
    prompt = generate_prompt(question, prompt_file, metadata_file)
    print("Prompt:", prompt)
    # make sure the model stops generating at triple ticks
    # eos_token_id = tokenizer.convert_tokens_to_ids(["```"])[0]
    eos_token_id = tokenizer.eos_token_id
    pipe = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        max_new_tokens=300,
        do_sample=False,
        return_full_text=False, # added return_full_text parameter to prevent splitting issues with prompt
        num_beams=5, # do beam search with 5 beams for high quality results
    )
    generated_query = (
        pipe(
            prompt,
            num_return_sequences=1,
            eos_token_id=eos_token_id,
            pad_token_id=eos_token_id,
        )[0]["generated_text"]
        .split(";")[0]
        .split("```")[0]
        .strip()
        + ";"
    )
    return generated_query
