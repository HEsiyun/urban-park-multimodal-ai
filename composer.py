# composer.py
from __future__ import annotations
from typing import Any, Dict, List, Optional
import re

# ========== LLM 集成 ==========
try:
    from openai import OpenAI
    LLM_AVAILABLE = True
except ImportError:
    LLM_AVAILABLE = False
    print("[WARN] OpenAI not available. Install: pip install openai")

# 配置：使用本地 Ollama 或 OpenAI
USE_LOCAL_LLM = True  # True = Ollama, False = OpenAI
OLLAMA_BASE_URL = "http://localhost:11434/v1"
OLLAMA_MODEL = "phi3:medium-128k" # "wizard-math:7b"  "mistral" "llama3.2:3b"  或 "mistral", "phi3"
OPENAI_MODEL = "gpt-4o-mini"


def _summarize_rag_context(
    rag_snippets: List[Dict[str, Any]], 
    query: str,
    sql_result_summary: str = ""
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
        
        # 构建 prompt
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

        # 调用 LLM
        if USE_LOCAL_LLM:
            client = OpenAI(base_url=OLLAMA_BASE_URL, api_key="ollama")
            model = OLLAMA_MODEL
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
            max_tokens=300
        )
        
        summary = response.choices[0].message.content.strip()
        return summary
        
    except Exception as e:
        print(f"[WARN] LLM summarization failed: {e}")
        # 回退到简单格式化
        return _format_rag_snippets_simple(rag_snippets)

def _summarize_rag_context2(
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
        if USE_LOCAL_LLM:
            client = OpenAI(base_url=OLLAMA_BASE_URL, api_key="ollama")
            model = OLLAMA_MODEL
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
        print(f"[WARN] LLM summarization failed: {e}")
        # 回退到简单格式化
        return _format_rag_snippets_simple(rag_snippets)


def _format_rag_snippets_simple(snippets: List[Dict[str, Any]]) -> str:
    """简单格式化 RAG 片段（无 LLM 时的回退方案）"""
    if not snippets:
        return ""
    
    output = "### 📚 Reference Context\n\n"
    for i, snippet in enumerate(snippets[:3], 1):
        text = snippet.get("text", "")
        # 清理文本
        text = re.sub(r'\s+', ' ', text).strip()
        text = text[:200] + "..." if len(text) > 200 else text
        
        page = snippet.get("page", "?")
        output += f"**Source {i}** (page {page}):\n{text}\n\n"
    
    return output


def _snip(txt: str, n: int = 150) -> str:
    """截断文本并清理空白字符"""
    s = re.sub(r"\s+", " ", (txt or "")).strip()
    return (s[:n] + "...") if len(s) > n else s


def compose_answer(nlu: Dict[str, Any], state: Dict[str, Any]) -> Dict[str, Any]:
    intent = nlu.get("intent", "")
    ev = state["evidence"]
    tables: List[Dict[str, Any]] = []
    citations: List[Dict[str, Any]] = []
    charts: List[Dict[str, Any]] = []
    answer_md = ""
    
    # 获取用户原始查询
    user_query = nlu.get("slots", {}).get("original_query", "")
    if not user_query and state.get("slots"):
        user_query = state["slots"].get("text", "")

    # ========== RAG 内容处理 ==========
    if intent in ("RAG", "RAG+SQL_tool"):
        sop = ev.get("sop", {})
        if sop:
            answer_md = (
                "**Mowing SOP (Standard Operating Procedures)**\n\n"
                "### Steps\n" + "\n".join([f"{i+1}. {s}" for i, s in enumerate(sop.get("steps", []))]) +
                ( "\n\n### Materials\n- " + "\n- ".join(sop.get("materials", [])) if sop.get("materials") else "" ) +
                ( "\n\n### Tools\n- " + "\n- ".join(sop.get("tools", [])) if sop.get("tools") else "" ) +
                ( "\n\n### Safety\n- " + "\n- ".join(sop.get("safety", [])) if sop.get("safety") else "" )
            )
            
            # 添加引用
            for h in ev.get("kb_hits", [])[:3]:
                citations.append({"title": "Mowing Standard/Manual", "source": h.get("source", "")})

    # ========== SQL 内容处理 + 图表生成 ==========
    if intent in ("SQL_tool", "RAG+SQL_tool", "SQL_tool_2"):
        sql = ev.get("sql", {})
        rows = sql.get("rows", [])
        
        # 获取模板提示
        template_hint = None
        if state.get("plan"):
            for step in state["plan"]:
                if step.get("tool") == "sql_query_rag":
                    template_hint = step.get("args", {}).get("template")
                    break
        
        # 生成图表配置
        chart_config = _detect_chart_type(rows, template_hint)
        if chart_config:
            charts.append(chart_config)
            chart_desc = _generate_chart_description(chart_config, rows)
            if chart_desc:
                if answer_md:
                    answer_md += "\n\n"
                answer_md += chart_desc + "\n\n"
        
        # 表格数据
        tables.append({
            "name": _get_table_name(template_hint, nlu.get("slots", {})),
            "columns": list(rows[0].keys()) if rows else [],
            "rows": rows
        })
        
        # 生成 SQL 结果摘要
        sql_summary = _generate_sql_summary(rows, template_hint, nlu.get("slots", {}))
        
        if answer_md:
            answer_md += "\n\n"
        
        # 主要结果
        answer_md += sql_summary
        answer_md += f"\n\n**Query Performance**: {sql.get('rowcount',0)} rows in {sql.get('elapsed_ms',0)}ms"
        
        # 伪造RAG hits以供后续处理
        if intent == "SQL_tool_2":
            ev["kb_hits"] = [{"page": "1", "text": "Criteria For Softball Female - U17: Dimension Home to Pitchers Plate should be greater than 12.9m and less than 13.42m; Home to First Base Path should be greater than 17.988m and less than 18.588m"}]
            rag_hits = ev.get("kb_hits", [])
            if rag_hits:
                answer_md += "\n\n---\n\n"
                # 使用 LLM 总结 RAG 上下文
                rag_context = _summarize_rag_context2(
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
        # ✅ 关键改进：使用 LLM 增强 RAG 内容
        if intent == "RAG+SQL_tool":
            rag_hits = ev.get("kb_hits", [])
            if rag_hits:
                answer_md += "\n\n---\n\n"
                # 使用 LLM 总结 RAG 上下文
                rag_context = _summarize_rag_context(
                    rag_snippets=rag_hits,
                    query=user_query,
                    sql_result_summary=sql_summary
                )
                answer_md += rag_context
                
                # 添加引用
                for h in rag_hits[:3]:
                    citations.append({
                        "title": "Reference Document", 
                        "source": h.get("source", "")
                    })

    # ========== CV 内容处理 ==========
    if intent in ("CV_tool", "RAG+CV_tool"):
        cv = ev.get("cv", {})
        if answer_md:
            answer_md += "\n\n"
        answer_md += (
            "**Image Assessment**\n\n"
            f"Condition: **{cv.get('condition','unknown')}** (score {cv.get('score',0):.2f})\n\n"
            f"Labels: {', '.join(cv.get('labels', []))}\n\n"
            f"Notes: {'; '.join(cv.get('explanations', []))}"
        )
        
        # RAG 上下文（如果有）
        if intent == "RAG+CV_tool":
            rag_hits = ev.get("kb_hits", [])
            if rag_hits:
                answer_md += "\n\n---\n\n"
                rag_context = _summarize_rag_context(
                    rag_snippets=rag_hits,
                    query=user_query,
                    sql_result_summary=""
                )
                answer_md += rag_context
        
        for h in ev.get("support", [])[:2]:
            citations.append({"title": "Inspection Guidance", "source": h.get("source", "")})
        
        if cv.get("low_confidence"):
            answer_md = "> ⚠️ Low confidence — consider another angle.\n\n" + answer_md

    # ========== 兜底 ==========
    if not answer_md:
        answer_md = "I couldn't generate a response for this query."

    return {
        "answer_md": answer_md,
        "tables": tables,
        "charts": charts,
        "map_layer": None,
        "citations": citations,
        "logs": state["logs"]
    }


# ========== 辅助函数 ==========

def _get_table_name(template_hint: str, slots: Dict[str, Any]) -> str:
    """根据模板生成表格名称"""
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
    elif template_hint == "field_dimension.compare_dimensions":
        return "Field Dimension Comparison"
    return "Query Result"


def _generate_sql_summary(rows: List[Dict], template_hint: str, slots: Dict[str, Any]) -> str:
    """生成 SQL 查询结果的自然语言摘要"""
    if not rows:
        return "❌ No results found."
    
    if template_hint == "mowing.labor_cost_month_top1":
        if len(rows) > 0:
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
    
    elif template_hint == "field_dimension.compare_dimensions":
        return f"### 📏 Field Dimension Comparison\n\nComparing dimensions for **{len(rows)} fields**."

    return f"### Results\n\nFound **{len(rows)} records**."


def _detect_chart_type(rows: List[Dict], template_hint: str = None) -> Optional[Dict[str, Any]]:
    """检测图表类型"""
    if not rows:
        return None
    
    columns = list(rows[0].keys())
    
    if template_hint == "mowing.cost_trend":
        if "month" in columns and "monthly_cost" in columns:
            parks = sorted(list(set(row.get("park") for row in rows if row.get("park"))))
            
            # ✅ 限制：如果公园数量超过 10 个，只显示成本最高的前 10 个
            if len(parks) > 10:
                # 计算每个公园的总成本
                park_totals = {}
                for park in parks:
                    park_totals[park] = sum(
                        row["monthly_cost"] for row in rows 
                        if row.get("park") == park
                    )
                # 取前 10 名
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
        if "park" in columns and "last_mowing_date" in columns:
            return {
                "type": "timeline",
                "title": "Last Mowing Date by Park",
                "data": [
                    {
                        "park": row["park"],
                        "date": row["last_mowing_date"],
                        "sessions": row.get("total_sessions", 0),
                        "cost": row.get("total_cost", 0)
                    }
                    for row in rows
                ],
                "sort_by": "date",
                "sort_order": "desc"
            }
    
    return None


def _generate_chart_description(chart_config: Dict[str, Any], rows: List[Dict]) -> str:
    """生成图表描述"""
    if not chart_config or not rows:
        return ""
    
    chart_type = chart_config.get("type")
    
    if chart_type == "line":
        parks = list(set(row.get("park") for row in rows if row.get("park")))
        months = sorted(set(row.get("month") for row in rows if row.get("month")))
        return f"📈 **Visualization**: Line chart comparing {len(parks)} park(s) from month {min(months)} to {max(months)}"
    
    elif chart_type == "bar":
        return f"📊 **Visualization**: Bar chart comparing {len(rows)} park(s)"
    
    elif chart_type == "timeline":
        return f"📅 **Visualization**: Timeline of last mowing dates for {len(rows)} park(s)"
    
    return ""