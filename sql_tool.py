# sql_tool.py
from __future__ import annotations
import os, time
from datetime import datetime
from typing import Any, Dict, List, Optional, Callable
import torch

import pandas as pd
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoModelForSeq2SeqLM, pipeline
from config import LABOR_XLSX, LABOR_SHEET
from rag import RAG
from Data_layer import DataLayer
import sqlite3

try:
    from transformers import T5ForConditionalGeneration, T5Tokenizer, AutoTokenizer
except Exception:
    T5ForConditionalGeneration = None
    T5Tokenizer = None

_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
_t5_model: Optional[T5ForConditionalGeneration] = None
_t5_tokenizer: Optional[T5Tokenizer] = None
DB = DataLayer().initialize_database()



# -----------------------------
# Template implementations
# -----------------------------
def _tpl_mowing_labor_cost_month_top1(con: sqlite3.Connection, params: Dict[str, Any]) -> Dict[str, Any]:
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
            CAST("Val.in rep.cur." AS REAL) AS cost,
            "Posting Date" AS posting_dt
        FROM labor_data
    )
    SELECT park, SUM(cost) AS total_cost
    FROM month_data
    WHERE strftime('%Y', posting_dt) = '{year}'
        AND strftime('%m', posting_dt) = '{month:02d}'
    GROUP BY park
    ORDER BY total_cost DESC
    LIMIT 1;
    """
    t0 = time.time()
    cur = con.execute(sql)
    cols = [d[0] for d in cur.description] if cur.description else []
    rows = [dict(zip(cols, r)) for r in cur.fetchall()]
    elapsed = int((time.time() - t0) * 1000)
    return {"rows": rows, "rowcount": len(rows), "elapsed_ms": elapsed}


def _tpl_mowing_last_date_by_park(con: sqlite3.Connection, params: Dict[str, Any]) -> Dict[str, Any]:
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
            SUM(CAST("Val.in rep.cur." AS REAL)) AS total_cost
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
            SUM(CAST("Val.in rep.cur." AS REAL)) AS total_cost
        FROM labor_data
        GROUP BY "CO Object Name"
        ORDER BY last_mowing_date DESC;
        """
    
    t0 = time.time()
    cur = con.execute(sql)
    cols = [d[0] for d in cur.description] if cur.description else []
    rows = [dict(zip(cols, r)) for r in cur.fetchall()]
    elapsed = int((time.time() - t0) * 1000)
    print(f"Elapsed time: {elapsed} ms")
    return {"rows": rows, "rowcount": len(rows), "elapsed_ms": elapsed}


def _tpl_mowing_cost_trend(con: sqlite3.Connection, params: Dict[str, Any]) -> Dict[str, Any]:
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
            strftime('%Y', "Posting Date") AS year,
            strftime('%m', "Posting Date") AS month,
            "CO Object Name" AS park,
            SUM(CAST("Val.in rep.cur." AS REAL)) AS monthly_cost,
            COUNT(*) AS session_count
        FROM labor_data
        WHERE strftime('%Y', "Posting Date") = '{year}'
          AND CAST(strftime('%m', "Posting Date") AS INTEGER) BETWEEN {start_month} AND {end_month}
          {park_filter}
        GROUP BY strftime('%Y', "Posting Date"), strftime('%m', "Posting Date"), "CO Object Name"
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
    cur = con.execute(sql)
    cols = [d[0] for d in cur.description] if cur.description else []
    rows = [dict(zip(cols, r)) for r in cur.fetchall()]
    elapsed = int((time.time() - t0) * 1000)
    return {"rows": rows, "rowcount": len(rows), "elapsed_ms": elapsed, "chart_type": "line"}


def _tpl_mowing_cost_by_park_month(con: sqlite3.Connection, params: Dict[str, Any]) -> Dict[str, Any]:
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
            SUM(CAST("Val.in rep.cur." AS REAL)) AS total_cost,
            COUNT(*) AS mowing_sessions,
            AVG(CAST("Val.in rep.cur." AS REAL)) AS avg_cost_per_session,
            SUM("Total quantity") AS total_quantity
    FROM labor_data
    WHERE strftime('%Y', "Posting Date") = '{year}'
        AND strftime('%m', "Posting Date") = '{month:02d}'
    GROUP BY "CO Object Name"
    ORDER BY total_cost DESC;
    """

    t0 = time.time()
    cur = con.execute(sql)
    cols = [d[0] for d in cur.description] if cur.description else []
    rows = [dict(zip(cols, r)) for r in cur.fetchall()]
    elapsed = int((time.time() - t0) * 1000)
    return {"rows": rows, "rowcount": len(rows), "elapsed_ms": elapsed, "chart_type": "bar"}


def _tpl_mowing_cost_breakdown_by_park(con: sqlite3.Connection, params: Dict[str, Any]) -> Dict[str, Any]:
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
    where_parts = [f"strftime('%Y', \"Posting Date\") = '{year}'"]
    
    if park_name:
        where_parts.append(f"LOWER(\"CO Object Name\") LIKE LOWER('%{park_name}%')")
    
    if isinstance(month, int) and 1 <= month <= 12:
        where_parts.append(f"strftime('%m', \"Posting Date\") = '{month:02d}'")
    
    where_clause = " AND ".join(where_parts)
    
    sql = f"""
    SELECT 
        "CO Object Name" AS park,
        strftime('%m', "Posting Date") AS month,
        "ParActivity" AS activity_type,
        SUM(CAST("Val.in rep.cur." AS REAL)) AS cost,
        COUNT(*) AS sessions,
        SUM("Total quantity") AS total_quantity
    FROM labor_data
    WHERE {where_clause}
    GROUP BY "CO Object Name", strftime('%m', "Posting Date"), "ParActivity"
    ORDER BY park, month, cost DESC;
    """
    
    t0 = time.time()
    cur = con.execute(sql)
    cols = [d[0] for d in cur.description] if cur.description else []
    rows = [dict(zip(cols, r)) for r in cur.fetchall()]
    elapsed = int((time.time() - t0) * 1000)
    return {"rows": rows, "rowcount": len(rows), "elapsed_ms": elapsed}

def _tpl_get_diamond_dimensions(con: sqlite3.Connection, params: Dict[str, Any]) -> Dict[str, Any]:
    """
    Returns diamond field dimensions from the diamond_field_size_data table.
    """
    sql = """
    SELECT "Name of Field ", "Diamonds: Dimension Home to Pitchers Plate - m ", "Diamonds: Home to First Base Path - m "
    FROM diamond_field_size_data
    LIMIT 10;
    """
    t0 = time.time()
    cur = con.execute(sql)
    cols = [d[0] for d in cur.description] if cur.description else []
    rows = [dict(zip(cols, r)) for r in cur.fetchall()]
    elapsed = int((time.time() - t0) * 1000)
    return {"rows": rows, "rowcount": len(rows), "elapsed_ms": elapsed}

def _tpl_get_rectangular_dimensions(con: sqlite3.Connection, params: Dict[str, Any]) -> Dict[str, Any]:
    """
    Returns rectangular field dimensions from the rectangular_field_size_data table.
    """
    sql = """
    SELECT "Name of Field ", "Rectangular Field Dimension: Length - m ", "Rectangular Field Dimension: Width - m "
    FROM rectangular_field_size_data
    LIMIT 10;
    """
    t0 = time.time()
    cur = con.execute(sql)
    cols = [d[0] for d in cur.description] if cur.description else []
    rows = [dict(zip(cols, r)) for r in cur.fetchall()]
    elapsed = int((time.time() - t0) * 1000)
    return {"rows": rows, "rowcount": len(rows), "elapsed_ms": elapsed}

def _tpl_mowing_cost_least_per_sqft(con: sqlite3.Connection, params: Dict[str, Any]) -> Dict[str, Any]:
    """
    Returns the park with the lowest mowing cost per square foot for a given period.
    """
    # Extract parameters
    start_month = params.get("month1")
    end_month = params.get("month2")
    start_year = params.get("year1")
    end_year = params.get("year2")
    
    # Defaults
    if not start_year or not end_year:
        current_year = datetime.utcnow().year
        start_year = start_year or current_year
        end_year = end_year or current_year
    
    if not start_month or not end_month:
        start_month = 1
        end_month = 12
    
    # Create start and end date strings for comparison
    start_date = f"{start_year}-{start_month:02d}-01"
    # End date: last day of end_month
    if end_month == 12:
        end_date = f"{end_year}-12-31"
    else:
        end_date = f"{end_year}-{end_month:02d}-31"  # Approximation
    
    sql = f"""
    WITH park_costs AS (
        SELECT 
            p.PARKNAME,
            p.Shape_Area,
            SUM(CAST(l."Val.in rep.cur." AS REAL)) AS total_cost,
            COUNT(*) AS mowing_sessions
        FROM labor_data l
        JOIN park_name_mapping m ON l."CO Object Name" = m.CO_Object_Name
        JOIN park_GIS_data p ON m.matched_PARKNAME = p.PARKNAME
        WHERE l."Posting Date" >= '{start_date}'
            AND l."Posting Date" <= '{end_date}'
            AND p.Shape_Area > 0
            AND m.matched_PARKNAME IS NOT NULL
            AND m.matched_PARKNAME != 'None'
        GROUP BY p.PARKNAME, p.Shape_Area
    )
    SELECT 
        PARKNAME as park_name,
        ROUND(total_cost, 2) as total_cost,
        ROUND(Shape_Area, 2) as area_sqft,
        ROUND(total_cost / Shape_Area, 4) AS cost_per_sqft
    FROM park_costs
    WHERE Shape_Area > 0
    ORDER BY cost_per_sqft ASC
    LIMIT 10;
    """
    
    t0 = time.time()
    cur = con.execute(sql)
    cols = [d[0] for d in cur.description] if cur.description else []
    rows = [dict(zip(cols, r)) for r in cur.fetchall()]
    elapsed = int((time.time() - t0) * 1000)
    
    return {
        "rows": rows, 
        "rowcount": len(rows), 
        "elapsed_ms": elapsed, 
        "chart_type": "bar",
        "period": f"{start_month}/{start_year} to {end_month}/{end_year}"
    }

# -----------------------------
# Dispatcher registry
# -----------------------------
TEMPLATE_REGISTRY: Dict[str, Callable[[sqlite3.Connection, Dict[str, Any]], Dict[str, Any]]] = {
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
    # 过去per sqft最低除草成本
    "mowing.cost_by_park_least_per_sqft": _tpl_mowing_cost_least_per_sqft,
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

    con = sqlite3.connect(DB.db_path)
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
