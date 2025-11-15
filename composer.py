# composer.py - Response Composition Layer (Adapted for Refactored Architecture)
from __future__ import annotations
from typing import Any, Dict, List, Optional, Tuple
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


# ========= Lightweight standards extraction (works well with short TXT standards) =========
_MOWING_METRIC_PATTERNS = [
    ("grass_length_cm", r"grass\s*length.*?(\d+\s*-\s*\d+)\s*cm", lambda s: s.replace(" ", "")),
    ("cutting_height_cm", r"cutting\s*height.*?(\d+(?:\.\d+)?)\s*cm", lambda s: s),
    ("drainage_max_hours", r"(?:percolation|standing\s*water).*?(\d+)\s*hour", lambda s: s),
    ("mowing_frequency", r"every\s+(\d+)\s+working\s+days", lambda s: s),
    ("weed_tolerance_pct", r"weed\s*tolerance.*?<\s*(\d+)\s*%", lambda s: s),
    ("bare_ground_pct", r"bare\s*ground.*?<\s*(\d+)\s*%", lambda s: s),
]
# extract mowing standards from text snippets
def _extract_mowing_standards_from_text(text: str) -> dict:
    found: Dict[str, str] = {}
    t = " ".join(line.strip() for line in (text or "").splitlines() if line.strip())
    for key, pat, norm in _MOWING_METRIC_PATTERNS:
        m = re.search(pat, t, flags=re.I)
        if m:
            try:
                found[key] = norm(m.group(1))
            except Exception:
                found[key] = m.group(1)
    return found

def _extract_mowing_standards_from_hits(hits: List[Dict[str, Any]]) -> dict:
    merged: Dict[str, str] = {}
    for h in hits[:3]:
        snippet = h.get("text", "") or ""
        if not snippet:
            continue
        cur = _extract_mowing_standards_from_text(snippet)
        for k, v in cur.items():
            if v and k not in merged:
                merged[k] = v
    return merged
# extract activity frequency from text snippets
def _extract_activity_frequency_from_hits(
    hits: List[Dict[str, Any]], activity_name: Optional[str]
) -> Dict[str, Any]:
    """
    Surface the first frequency line that matches the requested activity.
    """
    target = (activity_name or "").lower()
    best: Dict[str, Any] = {}

    for h in hits:
        text = h.get("text", "") or ""
        if not text:
            continue
        lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
        freq_line = None
        for ln in lines:
            if "frequency" in ln.lower():
                freq_line = ln
                break
        if not freq_line:
            continue
        freq_text = re.sub(r"(?i)^[-•]*\s*(\*\*frequency:\*\*|frequency:)\s*", "", freq_line).strip()
        entry = {
            "frequency": freq_text,
            "raw_line": freq_line,
            "source": h.get("source", ""),
            "snippet": text.strip()
        }
        if target and target in text.lower():
            return entry
        if not best:
            best = entry
    return best

def _estimate_cycle_days_from_text(text: Optional[str]) -> Optional[float]:
    """
    Convert textual frequency (e.g., '2–3 times per week') into approximate days per cycle.
    """
    if not text:
        return None
    t = text.lower()

    def _parse_range(m: re.Match) -> float:
        start = float(m.group(1))
        end = m.group(2)
        if end:
            return (start + float(end)) / 2.0
        return start

    # every X-Y weeks
    m = re.search(r"every\s+(\d+(?:\.\d+)?)\s*(?:-|–|—)?\s*(\d+(?:\.\d+)?)?\s*week", t)
    if m:
        return _parse_range(m) * 7.0

    # every X-Y days
    m = re.search(r"every\s+(\d+(?:\.\d+)?)\s*(?:-|–|—)?\s*(\d+(?:\.\d+)?)?\s*day", t)
    if m:
        return _parse_range(m)

    # X-Y times per week
    m = re.search(r"(\d+(?:\.\d+)?)\s*(?:-|–|—)?\s*(\d+(?:\.\d+)?)?\s*(?:times\s+)?per\s+week", t)
    if m:
        avg = _parse_range(m)
        if avg > 0:
            return 7.0 / avg

    if "biweekly" in t or "every other week" in t:
        return 14.0
    if "weekly" in t:
        return 7.0
    if "monthly" in t:
        return 30.0
    if "daily" in t or "every day" in t:
        return 1.0
    return None

    # （删除了重复的 mowing 标准提取代码段）
    # 这里原本是一段与前面完全相同的 _MOWING_METRIC_PATTERNS /
    # _extract_mowing_standards_from_text / _extract_mowing_standards_from_hits 实现，
    # 会导致函数和常量重复定义。保留文件前面那一份即可，这里安全删除。

# ========= RAG Context Summarization, with Fallback ==========
def _summarize_rag_context(
    rag_snippets: List[Dict[str, Any]],
    query: str,
    sql_result_summary: str = ""
) -> str:
    """
    Use local Ollama LLM to summarize RAG document snippets into coherent context.
    Falls back to simple formatting if LLM is unavailable or fails.
    """
    if not rag_snippets:
        return ""

    if not LLM_AVAILABLE:
        return _format_rag_snippets_simple(rag_snippets)
    try:
        context_text = "\n\n".join([
            f"Source {i+1} (page {snippet.get('page', '?')}): {snippet.get('text', '')[:500]}"
            for i, snippet in enumerate(rag_snippets[:3])
        ])

        common_tail = """
Guidelines:
- Never use placeholders like [insert ...] or [fill in ...]. If a value is unknown, omit it rather than inventing or leaving placeholders.
- Keep it concise and directly relevant. Use markdown formatting.
""".strip()

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

{common_tail}"""
        else:
            prompt = f"""You are an assistant helping answer questions about park maintenance procedures and standards.

User Question: {query}

Reference Documents:
{context_text}

Task: Summarize the key information from the reference documents that answers the user's question. Provide:
- 2-3 key points or thresholds
- Relevant standards or guidelines
- Important safety/operational notes if applicable

{common_tail}"""

        client = OpenAI(base_url=OLLAMA_BASE_URL, api_key="ollama")

        response = client.chat.completions.create(
            model=OLLAMA_MODEL,
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
        # Extra safety: strip leftover placeholders if the model still produced any
        summary = re.sub(r"\[(?:insert|fill in)[^\]]*\]", "", summary, flags=re.I).strip()
        return summary

    except Exception as e:
        print(f"[WARN] Ollama LLM summarization failed: {e}")
        print(f"[INFO] Make sure Ollama is running and model is available (e.g., `ollama list`).")
        return _format_rag_snippets_simple(rag_snippets)


def _format_rag_snippets_simple(snippets: List[Dict[str, Any]]) -> str:
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
    s = re.sub(r"\s+", " ", (txt or "")).strip()
    return (s[:n] + "...") if len(s) > n else s

# ========== Main Composition Function being called externally ==========
def compose_answer(
    nlu: Dict[str, Any],
    state: Dict[str, Any],
    plan_metadata: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    # --- Early returns for unsupported or clarification flows ---
    status = state.get("status") or (plan_metadata or {}).get("status")
    if status == "UNSUPPORTED":
        msg = state.get("message") or "This question is not supported yet"
        return {
            "answer_md": f"**Not Supported**\n\n{msg}",
            "tables": [],
            "charts": [],
            "map_layer": None,
            "citations": [],
            "logs": state.get("logs", [])
        }

    clarifications = state.get("clarifications") or []
    if status == "NEEDS_CLARIFICATION" and clarifications:
        bullets = "\n".join([f"- {q}" for q in clarifications])
        return {
            "answer_md": f"**I need a bit more info to proceed:**\n\n{bullets}",
            "tables": [],
            "charts": [],
            "map_layer": None,
            "citations": [],
            "logs": state.get("logs", [])
        }

    intent = nlu.get("intent", "")
    ev = state.get("evidence", {})
    tables: List[Dict[str, Any]] = []
    citations: List[Dict[str, Any]] = []
    charts: List[Dict[str, Any]] = []
    answer_md = ""

    user_query = (
        nlu.get("raw_query", "") or
        nlu.get("slots", {}).get("text", "") or
        state.get("slots", {}).get("text", "")
    )

    template_hint = (plan_metadata or {}).get("template")
    if not template_hint:
        template_hint = _extract_template_from_logs(state.get("logs", []))

    slots = nlu.get("slots", {}) or state.get("slots", {})
    explanation_requested = bool(slots.get("explanation_requested"))

    activity_context: Dict[str, Any] = {}
    activity_cycle_days: Optional[float] = None

    header = []
    if intent:
        header.append(f"**Workflow:** `{intent}`")
    if template_hint:
        header.append(f"**Template:** `{template_hint}`")
    if header:
        answer_md = "  •  ".join(header) + "\n\n"

    # ========== RAG content handling (updated for standards-style TXT) ==========
    if intent in ("RAG", "RAG+SQL_tool", "RAG+CV_tool"):
        kb_hits = ev.get("kb_hits", []) or []
        sop = ev.get("sop", {}) or {}

        combined_text = " ".join([h.get("text", "")[:300] for h in kb_hits[:2]]).lower()
        is_mowing_query = any(k in combined_text for k in [
            "mowing", "cutting height", "grass length", "drainage",
            "inspection", "frequency", "weed tolerance", "bare ground"
        ])

        standards = _extract_mowing_standards_from_hits(kb_hits) if kb_hits else {}
        rag_section_added = False

        if is_mowing_query and standards:
            answer_md += "**Maintenance Standards Summary**\n\n"
            lines = []
            if standards.get("grass_length_cm"):
                lines.append(f"- **Grass length**: {standards['grass_length_cm']}")
            if standards.get("cutting_height_cm"):
                lines.append(f"- **Cutting height**: {standards['cutting_height_cm']} cm")
            if standards.get("mowing_frequency"):
                lines.append(f"- **Mowing frequency**: every {standards['mowing_frequency']} working days")
            if standards.get("drainage_max_hours"):
                lines.append(f"- **Drainage**: no standing water within {standards['drainage_max_hours']} hour(s) after rain stops")
            if standards.get("weed_tolerance_pct"):
                lines.append(f"- **Weed tolerance**: < {standards['weed_tolerance_pct']}% coverage (typical)")
            if standards.get("bare_ground_pct"):
                lines.append(f"- **Bare ground**: < {standards['bare_ground_pct']}% of ground cover (typical)")

            if lines:
                answer_md += "\n".join(lines) + "\n"
            else:
                answer_md += "_Key metrics are not explicit in the current excerpt._\n"

            for h in kb_hits[:3]:
                citations.append({"title": "Maintenance Standards", "source": h.get("source", "")})
            rag_section_added = True

        if not rag_section_added and kb_hits and slots.get("domain") == "activity":
            activity_context = _extract_activity_frequency_from_hits(kb_hits, slots.get("activity_name"))
            freq_text = activity_context.get("frequency")
            if freq_text:
                activity_cycle_days = _estimate_cycle_days_from_text(freq_text)
                if activity_cycle_days:
                    activity_context["cycle_days"] = activity_cycle_days
                activity_label = (slots.get("activity_name") or "Requested activity").title()
                answer_md += "**Activity Frequency Reference**\n\n"
                freq_line = freq_text
                if activity_cycle_days:
                    freq_line += f" (~every {activity_cycle_days:.1f} days)"
                answer_md += f"- **{activity_label}**: {freq_line}\n"
                src = activity_context.get("source")
                if src:
                    citations.append({"title": "Activity Reference", "source": src})
                else:
                    for h in kb_hits[:2]:
                        citations.append({"title": "Activity Reference", "source": h.get("source", "")})
                rag_section_added = True

        if not rag_section_added and kb_hits and intent == "RAG":
            answer_md += "### Reference Summary\n\n"
            if LLM_AVAILABLE:
                try:
                    summary = _summarize_rag_context(
                        rag_snippets=kb_hits,
                        query=user_query or "Maintenance standards query",
                        sql_result_summary=""
                    )
                    answer_md += summary
                except Exception as e:
                    print(f"[WARN] LLM summary failed: {e}")
                    answer_md += _format_rag_snippets_simple(kb_hits)
            else:
                answer_md += _format_rag_snippets_simple(kb_hits)

            for h in kb_hits[:3]:
                citations.append({"title": "Maintenance Standards", "source": h.get("source", "")})

    # ========== SQL content handling ==========
    if intent in ("SQL_tool", "RAG+SQL_tool"):
        sql = ev.get("sql", {})
        rows = sql.get("rows") or []

        # Base data used by all SQL templates
        annotated_rows = rows
        due_groups: Dict[str, List[Dict[str, Any]]] = {}
        horizon_days: Optional[float] = None

        # ---- Template-specific enrichments / annotations ----
        # NOTE: When adding new SQL templates that need extra processing,
        #       follow this pattern and keep all template-specific logic
        #       in this section, before charts/tables/summary.
        if template_hint == "activity.maintenance_due_window":
            annotated_rows, due_groups, horizon_days = _annotate_due_rows(
                rows,
                activity_cycle_days,
                slots.get("weeks_ahead")
            )

        # TODO: add more template-specific post-processing here as needed
        # e.g.
        # elif template_hint == "some.new_template":
        #     annotated_rows = _post_process_new_template(rows, slots)

        # ---- Generic chart + table generation (template-agnostic) ----
        chart_config = _detect_chart_type(annotated_rows, template_hint)
        if chart_config:
            charts.append(chart_config)

        if annotated_rows:
            tables.append({
                "name": _get_table_name(template_hint, slots),
                "columns": list(annotated_rows[0].keys()),
                "rows": annotated_rows
            })

        # ---- Generic SQL summary generation ----
        sql_summary = _generate_sql_summary(annotated_rows, template_hint, slots)

        # ---- Template-specific summary augmentations ----
        if template_hint == "activity.maintenance_due_window":
            due_md = _format_due_window_summary(
                due_groups,
                horizon_days,
                activity_context,
                slots.get("weeks_ahead")
            )
            if due_md:
                sql_summary += "\n\n" + due_md

        # TODO: add more template-specific summary extensions here
        # e.g.
        # elif template_hint == "some.new_template":
        #     sql_summary += "\n\n" + _extra_summary_for_new_template(annotated_rows)

        # ---- Final answer composition for SQL part ----
        if answer_md:
            answer_md += "\n\n"

        answer_md += sql_summary
        answer_md += (
            f"\n\n**Query Performance**: "
            f"{sql.get('rowcount', 0)} rows in {sql.get('elapsed_ms', 0)}ms"
        )
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
                        query=user_query or "",
                        sql_result_summary=sql_summary
                    )
                answer_md += rag_context
                for h in rag_hits[:3]:
                    citations.append({"title": "Reference Document", "source": h.get("source", "")})

    # ========== CV content handling ==========
    if intent in ("CV_tool", "RAG+CV_tool"):
        cv = ev.get("cv", {})
        labels = cv.get("labels", [])
        is_mock = any("VLM not configured" in str(label) for label in labels) if isinstance(labels, list) else False

        if is_mock:
            answer_md += (
                ("\n\n" if answer_md else "") +
                "**Image Analysis (Not Configured)**\n\n"
                "To enable AI-powered image analysis:\n"
                "1. Get an API key from your provider (e.g., OpenRouter)\n"
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
                ("\n\n" if answer_md else "") +
                "**Image Assessment**\n\n"
                f"Condition: **{cv.get('condition','unknown')}** (score {cv.get('score',0):.2f})\n\n"
                f"Issues: {', '.join(labels) if isinstance(labels, list) else labels}\n\n"
                f"Recommendations: {'; '.join(cv.get('explanations', [])) if isinstance(cv.get('explanations', []), list) else cv.get('explanations', '')}"
            )
            if cv.get("low_confidence"):
                answer_md = "> ⚠️ Low confidence — consider uploading a clearer image.\n\n" + answer_md

        if intent == "RAG+CV_tool" and not is_mock:
            rag_hits = ev.get("kb_hits", [])
            if rag_hits:
                answer_md += "\n\n---\n\n"
                rag_context = _summarize_rag_context(
                    rag_snippets=rag_hits,
                    query=user_query or "",
                    sql_result_summary=""
                )
                answer_md += rag_context

        for h in ev.get("support", [])[:2]:
            citations.append({"title": "Reference Standards", "source": h.get("source", "")})

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
    return None

def _get_table_name(template_hint: Optional[str], slots: Dict[str, Any]) -> str:
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
    elif template_hint == "activity.maintenance_due_window":
        return "Maintenance Due Window"
    return "Query Result"

def _safe_get_field(row: Dict[str, Any], *names: str):
    """Return the first present field (case-insensitive), else None."""
    for name in names:
        if name in row:
            return row[name]
        up = name.upper()
        lo = name.lower()
        if up in row:
            return row[up]
        if lo in row:
            return row[lo]
    return None

def _generate_sql_summary(rows: List[Dict], template_hint: Optional[str], slots: Dict[str, Any]) -> str:
    """Generate natural language summary of SQL results."""
    if not rows:
        return "No results found."

    if template_hint == "mowing.labor_cost_month_top1":
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
        # Prefer the requested park if provided, else first row
        park_name = (slots.get("park_name") or "").strip().lower()
        target = None
        if park_name:
            for r in rows:
                p = _safe_get_field(r, "park", "PARK")
                if p and str(p).strip().lower() == park_name:
                    target = r
                    break
        if target is None:
            target = rows[0]

        park = _safe_get_field(target, "park", "PARK") or "Unknown"
        date = _safe_get_field(target, "last_mowing_date", "LAST_MOWING_DATE", "date")
        sessions = _safe_get_field(target, "total_sessions", "TOTAL_SESSIONS", "total_mowing_sessions", "TOTAL_MOWING_SESSIONS")
        cost = _safe_get_field(target, "total_cost", "TOTAL_COST")

        parts = []
        if date:
            parts.append(f"**{park}** was last mowed on **{date}**")
        else:
            parts.append(f"Latest mowing record for **{park}** is shown below")

        if sessions is not None:
            parts.append(f"sessions: {sessions}")
        if cost is not None:
            try:
                parts.append(f"total cost: ${float(cost):,.2f}")
            except Exception:
                parts.append(f"total cost: {cost}")

        return "### 📅 Last Mowing Activity\n\n" + "; ".join(parts) + "."

    elif template_hint == "mowing.cost_breakdown":
        return f"### 💰 Detailed Breakdown\n\n**{len(rows)} cost entries** by activity type."
    
    elif template_hint == "field_dimension.rectangular":
        return f"### 📏 Field Dimension Comparison\n\nComparing dimensions for **{len(rows)} rectangular fields**."
    elif template_hint == "field_dimension.diamond":
        return f"### 📐 Field Dimension Comparison\n\nComparing dimensions for **{len(rows)} diamond fields**."
    elif template_hint == "activity.maintenance_due_window":
        weeks = slots.get("weeks_ahead")
        activity_name = (slots.get("activity_name") or "maintenance").title()
        window_text = f"the next {weeks} week(s)" if weeks else "the requested window"
        return f"### 🛠 Maintenance Window Check\n\nEvaluated **{len(rows)}** park(s) for **{activity_name}** over {window_text}."
    return f"### Results\n\nFound **{len(rows)} records**."

def _detect_chart_type(rows: List[Dict], template_hint: Optional[str] = None) -> Optional[Dict[str, Any]]:
    if not rows:
        return None

    columns = list(rows[0].keys())

    if template_hint == "mowing.cost_trend":
        if "month" in columns and "monthly_cost" in columns:
            parks = sorted(list(set(row.get("park") for row in rows if row.get("park"))))
            if len(parks) > 10:
                park_totals: Dict[str, float] = {}
                for park in parks:
                    park_totals[park] = sum(
                        row.get("monthly_cost", 0) for row in rows
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
                "grid": True
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
                "grid": True
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
                return None

            return {
                "type": "timeline",
                "title": "Last Mowing Date by Park",
                "data": [
                    {
                        "park": get_field(row, "park", "PARK"),
                        "date": get_field(row, "last_mowing_date", "LAST_MOWING_DATE"),
                        "sessions": get_field(row, "total_sessions", "TOTAL_SESSIONS", "total_mowing_sessions", "TOTAL_MOWING_SESSIONS"),
                        "cost": get_field(row, "total_cost", "TOTAL_COST")
                    }
                    for row in rows
                ],
                "sort_by": "date",
                "sort_order": "desc"
            }
    elif template_hint == "activity.maintenance_due_window":
        if "location" in columns and "last_service_date" in columns:
            return {
                "type": "timeline",
                "title": "Last Service Date by Park",
                "data": [
                    {
                        "park": row.get("location"),
                        "date": row.get("last_service_date"),
                        "days_since": row.get("days_since_last"),
                        "status": row.get("due_status")
                    }
                    for row in rows
                ],
                "sort_by": "date",
                "sort_order": "desc"
            }

    return None

def _generate_chart_description(chart_config: Dict[str, Any], rows: List[Dict]) -> str:
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
# ========== Specific SQL template helpers: maintenance due window ==========
def _safe_float(value: Any) -> Optional[float]:
    try:
        if value is None:
            return None
        return float(value)
    except (TypeError, ValueError):
        return None

def _annotate_due_rows(
    rows: List[Dict[str, Any]],
    cycle_days: Optional[float],
    weeks_ahead: Optional[int]
) -> Tuple[List[Dict[str, Any]], Dict[str, List[Dict[str, Any]]], Optional[float]]:
    """
    Add due status metadata to each row for maintenance window analysis.
    """
    if not rows:
        return rows, {}, None

    annotated: List[Dict[str, Any]] = []
    groups = {"overdue": [], "due": [], "recent": [], "unknown": []}

    horizon_days = _safe_float(rows[0].get("horizon_days"))
    if horizon_days is None and weeks_ahead:
        try:
            horizon_days = float(weeks_ahead) * 7.0
        except (TypeError, ValueError):
            horizon_days = None

    for row in rows:
        record = dict(row)
        days_since = _safe_float(record.get("days_since_last"))
        status = "unknown"
        days_until_due = None
        days_overdue = None

        if cycle_days is not None and days_since is not None:
            gap = cycle_days - days_since
            if gap <= 0:
                status = "overdue"
                days_overdue = abs(gap)
                days_until_due = 0.0
            else:
                if horizon_days is not None and gap <= horizon_days:
                    status = "due soon"
                else:
                    status = "recent"
                days_until_due = gap
                days_overdue = 0.0

        if cycle_days is not None:
            record["recommended_cycle_days"] = round(cycle_days, 2)
        if days_until_due is not None:
            record["days_until_due"] = round(days_until_due, 1)
        if days_overdue is not None:
            record["days_overdue"] = round(days_overdue, 1)
        record["due_status"] = status
        record["horizon_days"] = horizon_days
        annotated.append(record)

        group_key = "unknown"
        if status == "overdue":
            group_key = "overdue"
        elif status == "due soon":
            group_key = "due"
        elif status == "recent":
            group_key = "recent"
        groups.setdefault(group_key, []).append(record)

    return annotated, groups, horizon_days

def _format_due_window_summary(
    groups: Dict[str, List[Dict[str, Any]]],
    horizon_days: Optional[float],
    activity_context: Dict[str, Any],
    weeks_ahead: Optional[int]
) -> str:
    if not groups:
        return ""

    lines: List[str] = []
    freq = activity_context.get("frequency")
    cycle = activity_context.get("cycle_days")
    if freq:
        freq_line = f"**Reference frequency:** {freq}"
        if cycle:
            freq_line += f" (~every {cycle:.1f} days)"
        lines.append(freq_line)
    elif cycle:
        lines.append(f"**Reference frequency:** ~every {cycle:.1f} days")

    if horizon_days:
        weeks = horizon_days / 7.0
        window = weeks_ahead or round(weeks, 1)
        lines.append(f"**Planning window:** next {window} week(s) (~{horizon_days:.0f} days)")

    due_sections: List[str] = []
    for key, label in (("overdue", "Overdue"), ("due", "Due within window")):
        entries = groups.get(key) or []
        if not entries:
            continue
        block = [f"**{label}:**"]
        for rec in entries[:6]:
            park = rec.get("location") or rec.get("park") or "Unknown park"
            last_dt = rec.get("last_service_date") or "unknown date"
            days_since = rec.get("days_since_last")
            if days_since is not None:
                try:
                    days_since = float(days_since)
                except (TypeError, ValueError):
                    pass
            detail = f"- {park}: last serviced {last_dt}"
            if isinstance(days_since, (int, float)):
                detail += f" ({days_since:.1f} days ago)"
            if key == "overdue":
                overdue = rec.get("days_overdue")
                if isinstance(overdue, (int, float)) and overdue > 0:
                    detail += f", overdue by {overdue:.1f} day(s)"
            else:
                until = rec.get("days_until_due")
                if isinstance(until, (int, float)):
                    detail += f", due in ~{until:.1f} day(s)"
            block.append(detail)
        due_sections.append("\n".join(block))

    if not due_sections and groups.get("recent"):
        lines.append("All tracked parks are outside the requested window based on current records.")
    elif not due_sections and groups.get("unknown"):
        lines.append("Frequency reference was found but existing records are insufficient to determine upcoming needs.")

    if due_sections:
        lines.append("\n\n".join(due_sections))

    return "\n\n".join(lines).strip()
