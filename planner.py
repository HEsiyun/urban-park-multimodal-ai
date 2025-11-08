# planner.py - Execution Planning Layer
# Responsibility: Convert NLU results into executable tool plans
from __future__ import annotations
import re
from typing import Dict, Any, List, Optional

from nlu import NLUResult


# ========== EXECUTION PLAN DATACLASS ==========
class ExecutionPlan:
    """Structured execution plan"""
    def __init__(
        self,
        tool_chain: List[Dict[str, Any]],
        clarifications: List[str],
        metadata: Optional[Dict[str, Any]] = None
    ):
        self.tool_chain = tool_chain
        self.clarifications = clarifications
        self.metadata = metadata or {}

    def is_ready(self) -> bool:
        """Check if plan is ready to execute (no clarifications needed)"""
        return len(self.clarifications) == 0 and self.metadata.get("status") != "UNSUPPORTED"

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization"""
        return {
            "tool_chain": self.tool_chain,
            "clarifications": self.clarifications,
            "metadata": self.metadata,
            "ready": self.is_ready()
        }


# ========== EXECUTION PLANNER ==========
class ExecutionPlanner:
    """
    Converts NLU results into concrete execution plans.
    - No intent rewriting here.
    - If a SQL intent cannot be mapped to a supported template -> mark UNSUPPORTED.
    - If required params are missing for a supported template -> ask clarifications.
    """

    def __init__(self):
        # SQL template registry
        self.sql_templates = {
            "mowing.labor_cost_month_top1": {
                "builder": self._build_top_cost_params,
                "required": ["month", "year"],
                "optional": ["park_name"]
            },
            "mowing.cost_trend": {
                "builder": self._build_trend_params,
                "required": ["start_month", "end_month", "year"],
                "optional": ["park_name"]
            },
            "mowing.last_mowing_date": {
                "builder": self._build_last_date_params,
                "required": [],
                "optional": ["park_name"]
            },
            "mowing.cost_by_park_month": {
                "builder": self._build_cost_by_park_params,
                "required": ["month", "year"],
                "optional": ["park_name"]
            },
            "mowing.cost_breakdown": {
                "builder": self._build_breakdown_params,
                "required": ["month", "year"],
                "optional": ["park_name"]
            },
            "mowing.cost_by_park_least_per_sqft": {
                "builder": self._build_cost_least_per_sqft_params,
                "required": ["month1", "month2", "year1", "year2"],
                "optional": []
            },
            "field_dimension.rectangular": {
                "builder": self._build_top_cost_params,  # Placeholder
                "required": ["sport", "age_group"],
                "optional": []
            },
            "field_dimension.diamond": {
                "builder": self._build_top_cost_params,  # Placeholder
                "required": ["sport", "age_group"],
                "optional": []
            }
        }

    # ========== MAIN PLANNING ENTRY ==========
    def plan(self, nlu_result: NLUResult) -> ExecutionPlan:
        """
        Generate execution plan from NLU result (no intent rewriting).

        Returns:
            ExecutionPlan with tool chain and clarifications
        """
        intent = nlu_result.intent

        if intent == "RAG":
            return self._plan_rag(nlu_result)

        if intent == "SQL_tool":
            return self._plan_sql(nlu_result)

        if intent == "RAG+SQL_tool":
            return self._plan_rag_sql(nlu_result)

        if intent == "CV_tool":
            return self._plan_cv(nlu_result)

        if intent == "RAG+CV_tool":
            return self._plan_rag_cv(nlu_result)

        return ExecutionPlan(
            tool_chain=[],
            clarifications=[],
            metadata={"status": "UNSUPPORTED", "reason": f"Unknown intent: {intent}"}
        )

    # ========== RAG WORKFLOW ==========
    def _plan_rag(self, nlu_result: NLUResult) -> ExecutionPlan:
        """Plan pure RAG workflow"""
        slots = nlu_result.slots
        # Domain-specific retrieval hints
        if slots.get("domain") == "field_standards":
            keywords = "field dimensions standards age group requirements length width pitching distance soccer baseball softball"
        else:
            keywords = "mowing maintenance safety equipment frequency inspection checklist"

        tool_chain = [
            {
                "tool": "kb_retrieve",
                "args": {"query": keywords, "top_k": 5}
            },
            {
                "tool": "sop_extract",
                "args": {"schema": ["steps", "materials", "tools", "safety"]}
            }
        ]
        return ExecutionPlan(
            tool_chain=tool_chain,
            clarifications=[],
            metadata={"workflow": "RAG", "domain": slots.get("domain")}
        )

    # ========== SQL WORKFLOW ==========
    def _plan_sql(self, nlu_result: NLUResult) -> ExecutionPlan:
        """Plan pure SQL workflow"""
        slots = nlu_result.slots
        query = nlu_result.raw_query

        # Step 1: Template routing
        template = self._route_sql_template(query, slots)
        if not template:
            # Not supported: no template match for this SQL intent
            return ExecutionPlan(
                tool_chain=[],
                clarifications=[],
                metadata={"workflow": "SQL", "status": "UNSUPPORTED", "reason": "NO_TEMPLATE_MATCH"}
            )

        # Step 2: Template lookup
        template_config = self.sql_templates.get(template)
        if not template_config:
            # Not supported: routed template not registered
            return ExecutionPlan(
                tool_chain=[],
                clarifications=[],
                metadata={"workflow": "SQL", "status": "UNSUPPORTED", "reason": f"UNREGISTERED_TEMPLATE:{template}"}
            )

        # Step 3: Build params & validate
        params = template_config["builder"](slots)
        clarifications = self._validate_params(template, params, template_config)

        tool_chain = []
        if not clarifications:
            tool_chain.append({
                "tool": "sql_query_rag",
                "args": {"template": template, "params": params}
            })

        return ExecutionPlan(
            tool_chain=tool_chain,
            clarifications=clarifications,
            metadata={"workflow": "SQL", "template": template, "status": "OK" if not clarifications else "NEEDS_CLARIFICATION"}
        )

    # ========== RAG+SQL WORKFLOW ==========
    def _plan_rag_sql(self, nlu_result: NLUResult) -> ExecutionPlan:
        if nlu_result.slots.get("domain") == "field_dimension":
            return self._plan_rag_sql_fields(nlu_result)
        else:
            return self._plan_rag_sql_mowing(nlu_result)
    def _plan_rag_sql_mowing(self, nlu_result: NLUResult) -> ExecutionPlan:
        """Plan hybrid RAG+SQL workflow"""
        slots = nlu_result.slots
        query = nlu_result.raw_query

        # SQL template routing first (if unsupported, we bail out)
        template = self._route_sql_template(query, slots)
        if not template:
            return ExecutionPlan(
                tool_chain=[],
                clarifications=[],
                metadata={"workflow": "RAG+SQL", "status": "UNSUPPORTED", "reason": "NO_TEMPLATE_MATCH"}
            )

        template_config = self.sql_templates.get(template)
        if not template_config:
            return ExecutionPlan(
                tool_chain=[],
                clarifications=[],
                metadata={"workflow": "RAG+SQL", "status": "UNSUPPORTED", "reason": f"UNREGISTERED_TEMPLATE:{template}"}
            )

        params = template_config["builder"](slots)
        clarifications = self._validate_params(template, params, template_config)

        # RAG retrieval keywords for context (kept lightweight)
        rag_keywords = "mowing maintenance cost frequency staffing inspection policy standards"

        tool_chain: List[Dict[str, Any]] = []
        if not clarifications:
            tool_chain = [
                {"tool": "kb_retrieve", "args": {"query": rag_keywords, "top_k": 3}},
                {"tool": "sql_query_rag", "args": {"template": template, "params": params}}
            ]

        return ExecutionPlan(
            tool_chain=tool_chain,
            clarifications=clarifications,
            metadata={
                "workflow": "RAG+SQL",
                "template": template,
                "status": "OK" if not clarifications else "NEEDS_CLARIFICATION",
                "explanation_requested": bool(slots.get("explanation_requested"))
            }
        )
    
    def _plan_rag_sql_fields(self, nlu_result: NLUResult) -> ExecutionPlan:
        """Plan hybrid RAG+SQL workflow for field dimension queries"""
        slots = nlu_result.slots
        query = nlu_result.raw_query
        
        # RAG component: retrieve context
        rag_keywords = self._build_rag_keywords(slots)
        
        # SQL component: determine template and params
        template = self._route_sql_template(query, slots)
        template_config = self.sql_templates.get(template)
        
        if not template_config:
            return ExecutionPlan(
                tool_chain=[],
                clarifications=[f"Unknown SQL template: {template}"]
            )
        
        
        # param_builder = template_config["builder"]
        # params = param_builder(slots)
        
        # Validate parameters
        # clarifications = self._validate_params(template, params, template_config)
        
        tool_chain = [
            {
                "tool": "kb_retrieve",
                "args": {
                    "query": rag_keywords,
                    "top_k": 3
                }
            },
            {
                "tool": "sql_query_rag",
                "args": {
                    "template": template,   
                }
            }
        ]
        
        print(f"[Planner] RAG+SQL plan: template={template}")
        
        return ExecutionPlan(
            tool_chain=tool_chain,
            clarifications="",
            metadata={"workflow": "RAG+SQL", "template": template}
        )
    # ========== CV WORKFLOW ==========
    def _plan_cv(self, nlu_result: NLUResult) -> ExecutionPlan:
        """Plan pure CV workflow"""
        slots = nlu_result.slots
        image_uri = slots.get("image_uri")

        if not image_uri:
            return ExecutionPlan(
                tool_chain=[],
                clarifications=["Please upload an image for computer vision analysis"],
                metadata={"workflow": "CV", "status": "NEEDS_CLARIFICATION"}
            )

        tool_chain = [
            {
                "tool": "cv_assess_rag",
                "args": {
                    "image_uri": image_uri,
                    "topic_hint": "turf wear disease inspection guidelines"
                }
            }
        ]
        return ExecutionPlan(
            tool_chain=tool_chain,
            clarifications=[],
            metadata={"workflow": "CV", "status": "OK"}
        )

    # ========== RAG+CV WORKFLOW ==========
    def _plan_rag_cv(self, nlu_result: NLUResult) -> ExecutionPlan:
        """Plan hybrid RAG+CV workflow"""
        slots = nlu_result.slots
        image_uri = slots.get("image_uri")

        if not image_uri:
            return ExecutionPlan(
                tool_chain=[],
                clarifications=["Please upload an image for analysis"],
                metadata={"workflow": "RAG+CV", "status": "NEEDS_CLARIFICATION"}
            )

        tool_chain = [
            {"tool": "kb_retrieve", "args": {"query": "turf inspection maintenance repair standards", "top_k": 3}},
            {"tool": "cv_assess_rag", "args": {"image_uri": image_uri, "topic_hint": "turf wear disease inspection guidelines"}}
        ]
        return ExecutionPlan(
            tool_chain=tool_chain,
            clarifications=[],
            metadata={"workflow": "RAG+CV", "status": "OK"}
        )

    # ========== SQL TEMPLATE ROUTING ==========
    def _route_sql_template(self, query: str, slots: Dict[str, Any]) -> Optional[str]:
        """
        Determine which SQL template to use based on query patterns.
        Return None when no supported template is identified (so executor can say "not supported").
        """
        lowq = (query or "").lower().strip()
        domain = slots.get("domain")
        
        if domain == "field_dimension":
            if slots.get("sport") in ["soccer", "cricket", "gaelic football", "cfl", "nfl", "rugby", "ultimate frisbee", "lacrosse", "frisbee", "ultimate"]:
                return "field_dimension.rectangular"
            return "field_dimension.diamond"  # placeholder for field dimension template
        if domain != "mowing":
            return None

        # Pattern 1: Superlative queries (highest, top, most expensive)
        if any(k in lowq for k in ["highest", "top", "max", "most expensive"]) and "cost" in lowq:
            return "mowing.labor_cost_month_top1"

        # Pattern 2: Recent/last queries
        if any(k in lowq for k in ["last", "recent", "latest", "when was"]) and any(k in lowq for k in ["mow", "mowing"]):
            return "mowing.last_mowing_date"

        # Pattern 3: Trend queries
        if "trend" in lowq and "cost" in lowq:
            return "mowing.cost_trend"

        # Pattern 4: Range queries (from X to Y)
        if re.search(r'\bfrom\s+\w+\s+to\s+\w+', lowq) and "cost" in lowq:
            return "mowing.cost_trend"

        # Pattern 5: Comparison queries
        if any(k in lowq for k in ["compare", "across", "all parks"]) and "cost" in lowq:
            return "mowing.cost_by_park_month"

        # Pattern 6: Breakdown queries
        if any(k in lowq for k in ["breakdown", "detail", "break down"]) and "cost" in lowq:
            return "mowing.cost_breakdown"

        # Pattern 7: By-park phrasing
        if "by park" in lowq or "each park" in lowq:
            return "mowing.cost_by_park_month"

        # Pattern 8: Parks having the least mowing cost for a period
        if "least" in lowq and "mowing cost" in lowq:
            return "mowing.cost_by_park_least_per_sqft"

        # No supported template matched
        return None

    # ========== SQL PARAMETER BUILDERS ==========
    def _build_top_cost_params(self, slots: Dict[str, Any]) -> Dict[str, Any]:
        """Build parameters for top cost query"""
        return {"month": slots.get("month"), "year": slots.get("year"), "park_name": slots.get("park_name")}

    def _build_trend_params(self, slots: Dict[str, Any]) -> Dict[str, Any]:
        """Build parameters for trend analysis query"""
        return {
            "year": slots.get("range_year") or slots.get("year"),
            "start_month": slots.get("start_month") or slots.get("month") or 1,
            "end_month": slots.get("end_month") or slots.get("month") or 12,
            "park_name": slots.get("park_name"),
        }

    def _build_last_date_params(self, slots: Dict[str, Any]) -> Dict[str, Any]:
        """Build parameters for last mowing date query"""
        return {"park_name": slots.get("park_name")}

    def _build_cost_by_park_params(self, slots: Dict[str, Any]) -> Dict[str, Any]:
        """Build parameters for cost-by-park query"""
        return {"month": slots.get("month"), "year": slots.get("year"), "park_name": slots.get("park_name")}

    def _build_breakdown_params(self, slots: Dict[str, Any]) -> Dict[str, Any]:
        """Build parameters for cost breakdown query"""
        return {"month": slots.get("month"), "year": slots.get("year"), "park_name": slots.get("park_name")}
    def _build_cost_least_per_sqft_params(self, slots: Dict[str, Any]) -> Dict[str, Any]:
        """Build parameters for least cost per sqft query"""
        return {
            "month1": slots.get("month1"), 
            "month2": slots.get("month2"), 
            "year1": slots.get("year1"),
            "year2": slots.get("year2")
        }

    # ========== PARAMETER VALIDATION ==========
    def _validate_params(self, template: str, params: Dict[str, Any], config: Dict[str, Any]) -> List[str]:
        """
        Validate required parameters and generate clarification prompts.
        If required params are missing -> return clarifications (NOT unsupported).
        """
        clarifications: List[str] = []
        required = config.get("required", [])
        missing = [k for k in required if not params.get(k)]

        if not missing:
            return clarifications

        # Template-specific clarification copy
        if template == "mowing.cost_trend":
            clarifications.append("Which time period would you like to see? (e.g., from January to June, and year)")
        elif template in ["mowing.cost_by_park_month", "mowing.cost_breakdown", "mowing.labor_cost_month_top1"]:
            clarifications.append("Which month and year would you like to query?")
        # mowing.last_mowing_date has no required fields; nothing to add
        elif template == "mowing.cost_by_park_least_per_sqft":
            clarifications.append("Please specify the start month, end month, start year, and end year for the period.")
        return clarifications
    
    def _build_rag_keywords(self, slots: Dict[str, Any]) -> str:
        """Build RAG retrieval keywords based on domain and query"""
        res = ""
        for key, value in slots.items():
            if key != "domain" and value is not None:
                res += f"{key}:{value} " 
        return res.strip()