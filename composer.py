# composer.py - Response Composition Layer (Adapted for Refactored Architecture)
from __future__ import annotations
from typing import Any, Dict, List, Optional
import re

# ========== LLM Integration (Ollama Only) ==========
try:
    from openai import OpenAI
    LLM_AVAILABLE = True
except ImportError:
    LLM_AVAILABLE = False
    print("[WARN] OpenAI library not available. Install: pip install openai")
    print("[INFO] Note: We use OpenAI library to connect to Ollama (local LLM)")

# Configuration - Local Ollama Only
OLLAMA_BASE_URL = "http://localhost:11434/v1"
OLLAMA_MODEL = "llama3.2:3b"  # 或 "mistral", "phi3"
OPENAI_MODEL = "gpt-4o-mini"
OLLAMA_MODEL_FALL_BACK = "llama3:8b" # "wizard-math:7b"  "mistral" "llama3.2:3b"  或 "mistral", "phi3"


def _summarize_rag_context(
    rag_snippets: List[Dict[str, Any]], 
    query: str,
    sql_result_summary: str = ""
) -> str:
    """
    Use local Ollama LLM to summarize RAG document snippets into coherent context
    """
    if not LLM_AVAILABLE or not rag_snippets:
        return _format_rag_snippets_simple(rag_snippets)
    try:
        context_text = "\n\n".join([
            f"Source {i+1} (page {snippet.get('page', '?')}): {snippet.get('text', '')[:500]}"
            for i, snippet in enumerate(rag_snippets[:3])
        ])
        
        if sql_result_summary:
            prompt = f"""You are an assistant helping interpret park maintenance data.

User Question: {query}

SQL Query Result: {sql_result_summary}

Reference Documents:
{context_text}

Task: Based on the reference documents, provide 2-3 sentences of relevant context that helps interpret the SQL results above. Focus on:
- Relevant standards, procedures, or guidelines
- Cost factors or typical ranges mentioned
- Any important notes about the data

Keep it concise and directly relevant to the user's question. Use markdown formatting."""
        else:
            prompt = f"""You are an assistant helping answer questions about park maintenance procedures.

User Question: {query}

Reference Documents:
{context_text}

Task: Summarize the key information from the reference documents that answers the user's question. Provide:
- 2-3 key points or steps
- Relevant standards or guidelines
- Important safety notes if applicable

Use markdown formatting with bullet points."""

        client = OpenAI(base_url=OLLAMA_BASE_URL, api_key="ollama")
        
        response = client.chat.completions.create(
            model=OLLAMA_MODEL,
            messages=[
                {"role": "system", "content": "You are a helpful assistant that summarizes technical documentation clearly and concisely."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.3,
            max_tokens=300,
            timeout=10.0
        )
        
        summary = response.choices[0].message.content.strip()
        return summary
        
    except Exception as e:
        print(f"[WARN] LLM summarization failed: {e}")
        # 回退到简单格式化
        return _format_rag_snippets_simple(rag_snippets)

def _summarize_rag_context_dimension_comparison(
    rag_snippets: List[Dict[str, Any]], 
    query: str,
    sql_result_summary: str = "",
    sql: str = ""
) -> str:
    """
    使用 LLM 将 RAG 文档片段总结成连贯的上下文
    
    Args:
        rag_snippets: RAG 检索到的文档片段
        query: 用户原始查询
        sql_result_summary: SQL 查询结果的摘要（如果有）
    
    Returns:
        格式化的上下文说明
    """
    if not LLM_AVAILABLE or not rag_snippets:
        # 回退方案：简单格式化
        return _format_rag_snippets_simple(rag_snippets)
    
    try:
        # 准备上下文
        context_text = "\n\n".join([
            f"Source {i+1} (page {snippet.get('page', '?')}): {snippet.get('text', '')[:500]}"
            for i, snippet in enumerate(rag_snippets[:3])
        ])
        print("SQL Result Summary:", sql_result_summary)
        # 构建 prompt
        if sql_result_summary:
            prompt = f"""You are an assistant helping calculate the dimension differences.

User Question: {query}

SQL Query Result: {sql}

Reference Documents:
{context_text}

Task: You will find dimension data for a list of fields from the SQL Query Result. The reference document provides the criteria for the certain dimensions.
Compare the dimension data from the SQL results against the criteria mentioned in the reference documents.
List the differences for each criterion for each field.

Keep it concise and directly relevant to the user's question. Use markdown formatting."""
        else:
            prompt = f"""You are an assistant helping answer questions about park maintenance procedures.

User Question: {query}

Reference Documents:
{context_text}

Task: Summarize the key information from the reference documents that answers the user's question. Provide:
- 2-3 key points or steps
- Relevant standards or guidelines
- Important safety notes if applicable

Use markdown formatting with bullet points."""

        # 调用 LLM
        if LLM_AVAILABLE:
            client = OpenAI(base_url=OLLAMA_BASE_URL, api_key="ollama")
            model = OLLAMA_MODEL_FALL_BACK
        else:
            client = OpenAI()  # 需要设置 OPENAI_API_KEY 环境变量
            model = OPENAI_MODEL
        
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": "You are a helpful assistant that summarizes technical documentation clearly and concisely."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.3,
            max_tokens=128000
        )

        summary = response.choices[0].message.content.strip()
        return summary
        
    except Exception as e:
        print(f"[WARN] Ollama LLM summarization failed: {e}")
        print(f"[INFO] Make sure Ollama is running: open -a Ollama")
        print(f"[INFO] Check model is available: ollama list")
        return _format_rag_snippets_simple(rag_snippets)


def _format_rag_snippets_simple(snippets: List[Dict[str, Any]]) -> str:
    """Simple RAG snippet formatting (fallback when Ollama is unavailable)"""
    if not snippets:
        return ""
    
    output = "### Reference Context\n\n"
    for i, snippet in enumerate(snippets[:3], 1):
        text = snippet.get("text", "")
        text = re.sub(r'\s+', ' ', text).strip()
        text = text[:200] + "..." if len(text) > 200 else text
        
        page = snippet.get("page", "?")
        output += f"**Source {i}** (page {page}):\n{text}\n\n"
    
    return output


def _snip(txt: str, n: int = 150) -> str:
    """Truncate text and clean whitespace"""
    s = re.sub(r"\s+", " ", (txt or "")).strip()
    return (s[:n] + "...") if len(s) > n else s


def compose_answer(
    nlu: Dict[str, Any], 
    state: Dict[str, Any],
    plan_metadata: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """
    Compose final user-facing answer from execution state
    
    Args:
        nlu: NLU result dict with {intent, confidence, slots}
        state: ExecutionState.to_dict() with {evidence, logs, errors, slots}
        plan_metadata: Optional metadata from ExecutionPlan {workflow, template, ...}
    
    Returns:
        {
            "answer_md": str,
            "tables": list,
            "charts": list,
            "citations": list,
            "logs": list
        }
    """
    intent = nlu.get("intent", "")
    ev = state.get("evidence", {})
    tables: List[Dict[str, Any]] = []
    citations: List[Dict[str, Any]] = []
    charts: List[Dict[str, Any]] = []
    answer_md = ""
    
    # Get user's original query from multiple possible sources
    user_query = (
        nlu.get("raw_query", "") or 
        nlu.get("slots", {}).get("text", "") or
        state.get("slots", {}).get("text", "")
    )
    
    # Get template hint from plan_metadata or infer from logs
    template_hint = None
    if plan_metadata:
        template_hint = plan_metadata.get("template")
    
    # Fallback: try to extract from execution logs
    if not template_hint:
        template_hint = _extract_template_from_logs(state.get("logs", []))
    
    # Get slots for template context
    slots = nlu.get("slots", {}) or state.get("slots", {})

    # ========== RAG content handling ==========
    if intent in ("RAG", "RAG+SQL_tool", "RAG+CV_tool"):
        sop = ev.get("sop", {})
        kb_hits = ev.get("kb_hits", [])
        
        has_sop_content = any(sop.get(k) for k in ["steps", "materials", "tools", "safety"])
        
        # Detect query type
        is_mowing_query = False
        if kb_hits:
            combined_text = " ".join([h.get("text", "")[:200] for h in kb_hits[:2]]).lower()
            is_mowing_query = any(k in combined_text for k in ["mowing", "mow", "contractor", "equipment", "ppe"])
        
        if has_sop_content and is_mowing_query:
            # Display as Mowing SOP
            answer_md = "**Mowing SOP (Standard Operating Procedures)**\n\n"
            
            if sop.get("steps"):
                answer_md += "### Steps\n" + "\n".join([f"{i+1}. {s}" for i, s in enumerate(sop["steps"])])
            
            if sop.get("materials"):
                answer_md += "\n\n### Materials\n- " + "\n- ".join(sop["materials"])
            
            if sop.get("tools"):
                answer_md += "\n\n### Tools\n- " + "\n- ".join(sop["tools"])
            
            if sop.get("safety"):
                answer_md += "\n\n### Safety\n- " + "\n- ".join(sop["safety"])
            
            for h in kb_hits[:3]:
                citations.append({"title": "Mowing Standard/Manual", "source": h.get("source", "")})
                
        elif kb_hits:
            # Field dimensions or other RAG query
            answer_md = "### Field Standards Information\n\n"
            
            if LLM_AVAILABLE:
                try:
                    summary = _summarize_rag_context(
                        rag_snippets=kb_hits,
                        query=user_query or "Field standards query",
                        sql_result_summary=""
                    )
                    answer_md += summary
                except Exception as e:
                    print(f"[WARN] LLM summary failed: {e}")
                    answer_md += _format_rag_snippets_simple(kb_hits)
            else:
                answer_md += _format_rag_snippets_simple(kb_hits)
            
            for h in kb_hits[:3]:
                citations.append({"title": "Field Standards Reference", "source": h.get("source", "")})

    # ========== SQL content handling ==========
    if intent in ("SQL_tool", "RAG+SQL_tool"):
        sql = ev.get("sql", {})
        rows = sql.get("rows", [])
        
        # Generate chart
        chart_config = _detect_chart_type(rows, template_hint)
        if chart_config:
            charts.append(chart_config)
        
        # Table data
        if rows:
            tables.append({
                "name": _get_table_name(template_hint, slots),
                "columns": list(rows[0].keys()),
                "rows": rows
            })
        
        # Generate SQL summary
        sql_summary = _generate_sql_summary(rows, template_hint, slots)
        
        if answer_md:
            answer_md += "\n\n"
        
        answer_md += sql_summary
        answer_md += f"\n\n**Query Performance**: {sql.get('rowcount',0)} rows in {sql.get('elapsed_ms',0)}ms"
        
        # 伪造RAG hits以供后续处理
        if intent == "SQL_tool_2":
            ev["kb_hits"] = [{"page": "1", "text": "Criteria For Softball Female - U17: Dimension Home to Pitchers Plate should be greater than 12.9m and less than 13.42m; Home to First Base Path should be greater than 17.988m and less than 18.588m"}]
            rag_hits = ev.get("kb_hits", [])
            if rag_hits:
                answer_md += "\n\n---\n\n"
                # 使用 LLM 总结 RAG 上下文
                rag_context = _summarize_rag_context_dimension_comparison(
                    rag_snippets=rag_hits,
                    query=user_query,
                    sql_result_summary=sql_summary, sql=sql.get("rows", [])
                )
                answer_md += rag_context
                
                # 添加引用
                for h in rag_hits[:3]:
                    citations.append({
                        "title": "Reference Document", 
                        "source": h.get("source", "")
                    })
        # Add RAG context for hybrid queries
        if intent == "RAG+SQL_tool":
            rag_hits = ev.get("kb_hits", [])
            if rag_hits:
                answer_md += "\n\n---\n\n"
                if nlu.get("slots", {}).get("domain") == "field_dimension":
                    rag_context = _summarize_rag_context_dimension_comparison(
                        rag_snippets=rag_hits,
                        query=user_query,
                        sql_result_summary=sql_summary, sql=sql.get("rows", [])
                    )
                else:
                    rag_context = _summarize_rag_context(
                        rag_snippets=rag_hits,
                        query=user_query,
                        sql_result_summary=sql_summary
                    )
                answer_md += rag_context
                
                for h in rag_hits[:3]:
                    citations.append({"title": "Reference Document", "source": h.get("source", "")})

    # ========== CV content handling ==========
    if intent in ("CV_tool", "RAG+CV_tool"):
        cv = ev.get("cv", {})
        
        is_mock = any("VLM not configured" in str(label) for label in cv.get("labels", []))
        
        if answer_md:
            answer_md += "\n\n"
        
        if is_mock:
            answer_md += (
                "**Image Analysis (Not Configured)**\n\n"
                "To enable AI-powered image analysis:\n"
                "1. Get free API key from https://openrouter.ai/\n"
                "2. Set environment variable: `export OPENROUTER_API_KEY='your-key'`\n"
                "3. Restart backend\n\n"
                "Supported analysis:\n"
                "- Field condition assessment\n"
                "- Turf health evaluation\n"
                "- Maintenance recommendations\n"
                "- Safety hazard detection"
            )
        else:
            answer_md += (
                "**Image Assessment**\n\n"
                f"Condition: **{cv.get('condition','unknown')}** (score {cv.get('score',0):.2f})\n\n"
                f"Issues: {', '.join(cv.get('labels', []))}\n\n"
                f"Recommendations: {'; '.join(cv.get('explanations', []))}"
            )
            
            if cv.get("low_confidence"):
                answer_md = "> ⚠️ Low confidence - consider uploading a clearer image.\n\n" + answer_md
        
        # Add RAG context for hybrid CV queries
        if intent == "RAG+CV_tool":
            rag_hits = ev.get("kb_hits", [])
            if rag_hits and not is_mock:
                answer_md += "\n\n---\n\n"
                rag_context = _summarize_rag_context(
                    rag_snippets=rag_hits,
                    query=user_query,
                    sql_result_summary=""
                )
                answer_md += rag_context
        
        for h in ev.get("support", [])[:2]:
            citations.append({"title": "Reference Standards", "source": h.get("source", "")})

    # ========== Fallback ==========
    if not answer_md:
        answer_md = "I couldn't generate a response for this query."

    return {
        "answer_md": answer_md,
        "tables": tables,
        "charts": charts,
        "map_layer": None,
        "citations": citations,
        "logs": state.get("logs", [])
    }


# ========== Helper Functions ==========

def _extract_template_from_logs(logs: List[Dict[str, Any]]) -> Optional[str]:
    """
    Extract SQL template from execution logs
    
    This is a fallback method when plan_metadata is not provided
    """
    for log in logs:
        if log.get("tool") == "sql_query_rag":
            # Template would be in the args if we logged it
            # For now, we can't extract it from redacted args
            # This is why passing plan_metadata is preferred
            pass
    return None


def _get_table_name(template_hint: Optional[str], slots: Dict[str, Any]) -> str:
    """Generate table name based on template"""
    if not template_hint:
        return "Query Result"
    
    if template_hint == "mowing.labor_cost_month_top1":
        month = slots.get("month", "")
        year = slots.get("year", "")
        return f"Top Park by Mowing Cost ({month}/{year})"
    elif template_hint == "mowing.cost_trend":
        return "Mowing Cost Trend"
    elif template_hint == "mowing.cost_by_park_month":
        return "Cost Comparison by Park"
    elif template_hint == "mowing.last_mowing_date":
        return "Last Mowing Dates"
    elif template_hint == "mowing.cost_breakdown":
        return "Detailed Cost Breakdown"
    elif template_hint == "field_dimension.rectangular":
        return "Rectangular Field Dimension Comparison"
    elif template_hint == "field_dimension.diamond":
        return "Diamond Field Dimension Comparison"
    return "Query Result"


def _generate_sql_summary(rows: List[Dict], template_hint: Optional[str], slots: Dict[str, Any]) -> str:
    """Generate natural language summary of SQL results"""
    if not rows:
        return "No results found."
    
    if template_hint == "mowing.labor_cost_month_top1":
        if rows:
            park = rows[0].get("park", "Unknown")
            cost = rows[0].get("total_cost", 0)
            month = slots.get("month", "")
            year = slots.get("year", "")
            return f"### 🏆 Results\n\n**{park}** had the highest mowing cost of **${cost:,.2f}** in {month}/{year}."
    
    elif template_hint == "mowing.cost_trend":
        return f"### 📈 Trend Analysis\n\nCost trend data across **{len(rows)} time periods**."
    
    elif template_hint == "mowing.cost_by_park_month":
        total = sum(row.get("total_cost", 0) for row in rows)
        return f"### 📊 Cost Comparison\n\n**{len(rows)} parks** with combined costs of **${total:,.2f}**."
    
    elif template_hint == "mowing.last_mowing_date":
        return f"### 📅 Last Mowing Activity\n\nShowing data for **{len(rows)} park(s)**."
    
    elif template_hint == "mowing.cost_breakdown":
        return f"### 💰 Detailed Breakdown\n\n**{len(rows)} cost entries** by activity type."
    
    elif template_hint == "field_dimension.rectangular":
        return f"### 📏 Field Dimension Comparison\n\nComparing dimensions for **{len(rows)} rectangular fields**."
    elif template_hint == "field_dimension.diamond":
        return f"### 📐 Field Dimension Comparison\n\nComparing dimensions for **{len(rows)} diamond fields**."
    return f"### Results\n\nFound **{len(rows)} records**."


def _detect_chart_type(rows: List[Dict], template_hint: Optional[str] = None) -> Optional[Dict[str, Any]]:
    """Detect appropriate chart type based on data structure"""
    if not rows:
        return None
    
    columns = list(rows[0].keys())
    
    if template_hint == "mowing.cost_trend":
        if "month" in columns and "monthly_cost" in columns:
            parks = sorted(list(set(row.get("park") for row in rows if row.get("park"))))
            
            if len(parks) > 10:
                park_totals = {}
                for park in parks:
                    park_totals[park] = sum(
                        row["monthly_cost"] for row in rows 
                        if row.get("park") == park
                    )
                top_parks = sorted(park_totals.items(), key=lambda x: x[1], reverse=True)[:10]
                parks = [p[0] for p in top_parks]
            
            return {
                "type": "line",
                "title": "Mowing Cost Trend",
                "x_axis": {"field": "month", "label": "Month", "type": "category"},
                "y_axis": {"field": "monthly_cost", "label": "Cost ($)", "type": "value"},
                "series": [
                    {
                        "name": park,
                        "data": [
                            {"x": row["month"], "y": row["monthly_cost"]}
                            for row in rows if row.get("park") == park
                        ]
                    }
                    for park in parks
                ],
                "legend": True,
                "grid": True,
                "note": f"Showing top {len(parks)} parks by total cost" if len(parks) < len(set(row.get("park") for row in rows)) else None
            }
    
    elif template_hint in ["mowing.cost_by_park_month", "mowing.labor_cost_month_top1"]:
        if "park" in columns and "total_cost" in columns:
            return {
                "type": "bar",
                "title": "Mowing Cost by Park",
                "x_axis": {"field": "park", "label": "Park", "type": "category"},
                "y_axis": {"field": "total_cost", "label": "Total Cost ($)", "type": "value"},
                "series": [
                    {
                        "name": "Total Cost",
                        "data": [{"x": row["park"], "y": row["total_cost"]} for row in rows]
                    }
                ],
                "legend": False,
                "grid": True,
                "color": "#4CAF50"
            }
    
    elif template_hint == "mowing.last_mowing_date":
        if "park" in columns or "PARK" in columns:
            def get_field(row, *field_names):
                for field in field_names:
                    if field in row:
                        return row[field]
                    if field.upper() in row:
                        return row[field.upper()]
                    if field.lower() in row:
                        return row[field.lower()]
                return 0
            
            return {
                "type": "timeline",
                "title": "Last Mowing Date by Park",
                "data": [
                    {
                        "park": get_field(row, "park", "PARK"),
                        "date": get_field(row, "last_mowing_date", "LAST_MOWING_DATE"),
                        "sessions": get_field(row, "total_sessions", "total_mowing_sessions", "TOTAL_SESSIONS", "TOTAL_MOWING_SESSIONS"),
                        "cost": get_field(row, "total_cost", "TOTAL_COST")
                    }
                    for row in rows
                ],
                "sort_by": "date",
                "sort_order": "desc"
            }
    
    return None


def _generate_chart_description(chart_config: Dict[str, Any], rows: List[Dict]) -> str:
    """Generate chart description text"""
    if not chart_config or not rows:
        return ""
    
    chart_type = chart_config.get("type")
    
    if chart_type == "line":
        parks = list(set(row.get("park") for row in rows if row.get("park")))
        months = sorted(set(row.get("month") for row in rows if row.get("month")))
        return f"Line chart comparing {len(parks)} park(s) from month {min(months)} to {max(months)}"
    
    elif chart_type == "bar":
        return f"Bar chart comparing {len(rows)} park(s)"
    
    elif chart_type == "timeline":
        return f"Timeline of last mowing dates for {len(rows)} park(s)"
    
    return ""