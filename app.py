# app.py - FastAPI Backend with Refactored Architecture
from __future__ import annotations
import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"
from datetime import datetime
from typing import Any, Dict, Optional

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

# Import refactored modules
from nlu import parse_intent_and_slots, NLUResult
from planner import ExecutionPlanner
from executor import ToolExecutor
from composer import compose_answer
from rag import RAG
from sql_fall_back import sql_fall_back

# ========== Pydantic Models ==========
class NLURequest(BaseModel):
    """Request model for NLU parsing"""
    text: str
    image_uri: Optional[str] = None


class AgentRequest(BaseModel):
    """Request model for full agent workflow"""
    text: str
    image_uri: Optional[str] = None
    skip_nlu: Optional[bool] = False
    nlu_result: Optional[Dict[str, Any]] = None


# ========== FastAPI App ==========
app = FastAPI(
    title="Parks Maintenance Intelligence API",
    version="1.0.1",
    description="Modular NLU → Planner → Executor → Composer Architecture"
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",
        "http://127.0.0.1:5173",
        "http://localhost:3000",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ========== Global Instances ==========
planner = ExecutionPlanner()
executor = ToolExecutor()


# ========== Health Check ==========
@app.get("/health")
def health_check():
    """System health check"""
    return {
        "status": "ok",
        "timestamp": datetime.utcnow().isoformat(),
        "architecture": "NLU → Planner → Executor → Composer",
        "components": {
            "rag": {
                "status": "active",
                "mode": getattr(RAG, "mode", "unknown")
            },
            "planner": {
                "status": "active",
                "templates": len(planner.sql_templates)
            },
            "executor": {
                "status": "active",
                "tools": list(executor.tool_registry.keys())
            }
        }
    }


# ========== NLU Endpoint ==========
@app.post("/nlu/parse")
def nlu_parse_endpoint(req: NLURequest):
    """
    Parse user input to extract intent and slots

    Returns:
        {
            "intent": str,
            "confidence": float,
            "slots": dict,
            "raw_query": str
        }
    """
    try:
        nlu_result = parse_intent_and_slots(req.text, req.image_uri)
        return {
            "intent": nlu_result.intent,
            "confidence": nlu_result.confidence,
            "slots": nlu_result.slots,
            "raw_query": nlu_result.raw_query
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"NLU parsing failed: {str(e)}")


# ========== Planning Endpoint ==========
@app.post("/plan/generate")
def plan_generate_endpoint(req: Dict[str, Any]):
    """
    Generate execution plan from NLU result

    Input:
        {
            "intent": str,
            "confidence": float,
            "slots": dict,
            "raw_query": str
        }

    Returns:
        {
            "tool_chain": list,
            "clarifications": list,
            "metadata": dict,
            "ready": bool
        }
    """
    try:
        # Reconstruct NLUResult from dict
        nlu_result = NLUResult(
            intent=req.get("intent", ""),
            confidence=req.get("confidence", 0.0),
            slots=req.get("slots", {}),
            raw_query=req.get("raw_query", "")
        )
        execution_plan = planner.plan(nlu_result)
        return execution_plan.to_dict()
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Planning failed: {str(e)}")


# ========== Execution Endpoint ==========
@app.post("/execute/run")
def execute_run_endpoint(req: Dict[str, Any]):
    """
    Execute a tool chain

    Input:
        {
            "tool_chain": list,
            "slots": dict,
            "metadata": dict,           # optional; may carry {"status": "UNSUPPORTED", ...}
            "clarifications": list      # optional; if provided, executor will short-circuit
        }

    Returns:
        ExecutionState.to_dict()
    """
    try:
        tool_chain = req.get("tool_chain", [])
        slots = req.get("slots", {})
        metadata = req.get("metadata", {}) or {}
        clarifications = req.get("clarifications", []) or []

        execution_state = executor.execute(
            tool_chain=tool_chain,
            slots=slots,
            status=metadata.get("status", "OK"),
            message=metadata.get("message"),
            clarifications=clarifications,
            metadata=metadata
        )
        return execution_state.to_dict()
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Execution failed: {str(e)}")


# ========== Full Agent Workflow ==========
@app.post("/agent/answer")
def agent_answer_endpoint(req: AgentRequest):
    """
    Complete agent workflow: NLU → Planner → Executor → Composer

    Returns:
        {
            "answer_md": str,
            "tables": list,
            "charts": list,
            "citations": list,
            "clarifications": list,
            "status": str,
            "debug": {...}
        }
    """
    try:
        # ========== STAGE 1: NLU ==========
        print("\n" + "="*60)
        print("[Agent] STAGE 1: Natural Language Understanding")
        print("="*60)

        if req.skip_nlu and req.nlu_result:
            nlu_result = NLUResult(
                intent=req.nlu_result.get("intent", ""),
                confidence=req.nlu_result.get("confidence", 0.0),
                slots=req.nlu_result.get("slots", {}),
                raw_query=req.nlu_result.get("raw_query", req.text)
            )
        else:
            nlu_result = parse_intent_and_slots(req.text, req.image_uri)

        if (nlu_result.slots.get("domain", {}) == "generic"):
            return sql_fall_back(nlu_result.raw_query)

        print(f"[Agent] Intent: {nlu_result.intent} (confidence: {nlu_result.confidence})")

        # ========== STAGE 2: Planning ==========
        print("\n" + "="*60)
        print("[Agent] STAGE 2: Execution Planning")
        print("="*60)

        execution_plan = planner.plan(nlu_result)

        print(f"[Agent] Tool chain: {len(execution_plan.tool_chain)} steps")
        print(f"[Agent] Plan status: {execution_plan.metadata.get('status', 'OK')}")
        if execution_plan.clarifications:
            print(f"[Agent] Clarifications suggested: {execution_plan.clarifications}")

        # ========== STAGE 3: Execution ==========
        # Always call executor; it will short-circuit for UNSUPPORTED / NEEDS_CLARIFICATION.
        print("\n" + "="*60)
        print("[Agent] STAGE 3: Tool Execution")
        print("="*60)

        execution_state = executor.execute(
            tool_chain=execution_plan.tool_chain,
            slots=nlu_result.slots,
            status=execution_plan.metadata.get("status", "OK"),
            message=execution_plan.metadata.get("message"),
            clarifications=execution_plan.clarifications,
            metadata=execution_plan.metadata
        )

        print(f"[Agent] Execution complete: {len(execution_state.logs)} tools executed")
        print(f"[Agent] Success: {len(execution_state.errors) == 0}")
        if execution_state.errors:
            print(f"[Agent] Errors: {execution_state.errors}")

        # ========== STAGE 4: Response Composition ==========
        print("\n" + "="*60)
        print("[Agent] STAGE 4: Response Composition")
        print("="*60)

        # Prepare data for composer
        nlu_dict = {
            "intent": nlu_result.intent,
            "confidence": nlu_result.confidence,
            "slots": nlu_result.slots,
            "raw_query": nlu_result.raw_query
        }
        state_dict = execution_state.to_dict()
        plan_metadata = execution_plan.metadata  # carries template/status

        response = compose_answer(nlu_dict, state_dict, plan_metadata)

        # Add status and debug info
        response["status"] = state_dict.get("status", "OK")
        response["clarifications"] = state_dict.get("clarifications", [])
        response["debug"] = {
            "nlu": {
                "intent": nlu_result.intent,
                "confidence": nlu_result.confidence,
                "slots": nlu_result.slots
            },
            "plan": execution_plan.to_dict(),
            "execution": {
                "tools_executed": len(execution_state.logs),
                "success": len(execution_state.errors) == 0,
                "errors": execution_state.errors
            }
        }

        print(f"[Agent] Response generated (status={response['status']})")
        print("="*60 + "\n")
        return response

    except Exception as e:
        print(f"[Agent ERROR] {str(e)}")
        import traceback
        traceback.print_exc()
        raise HTTPException(
            status_code=500,
            detail={
                "error": str(e),
                "stage": "unknown",
                "traceback": traceback.format_exc()
            }
        )

@app.post("/sql/fall_back")
def sql_fall_back_endpoint(req: AgentRequest):
    """
    SQL Fall Back Endpoint
    """
    try:
        response = sql_fall_back(req.text)
        return response
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"SQL Fall Back failed: {str(e)}")

# ========== Debug Endpoint ==========
@app.post("/debug/pipeline")
def debug_pipeline_endpoint(req: AgentRequest):
    """
    Run full pipeline with detailed debug output at each stage
    """
    try:
        debug_output = {"stages": {}, "errors": []}

        # Stage 1: NLU
        try:
            nlu_result = parse_intent_and_slots(req.text, req.image_uri)
            debug_output["stages"]["nlu"] = {
                "status": "success",
                "output": {
                    "intent": nlu_result.intent,
                    "confidence": nlu_result.confidence,
                    "slots": nlu_result.slots,
                    "raw_query": nlu_result.raw_query
                }
            }
        except Exception as e:
            debug_output["stages"]["nlu"] = {"status": "failed", "error": str(e)}
            debug_output["errors"].append(f"NLU: {str(e)}")
            return debug_output

        # Stage 2: Planning
        try:
            execution_plan = planner.plan(nlu_result)
            debug_output["stages"]["planning"] = {
                "status": "success",
                "output": execution_plan.to_dict()
            }
        except Exception as e:
            debug_output["stages"]["planning"] = {"status": "failed", "error": str(e)}
            debug_output["errors"].append(f"Planning: {str(e)}")
            return debug_output

        # Stage 3: Execution (let executor decide on unsupported/clarifications)
        try:
            execution_state = executor.execute(
                tool_chain=execution_plan.tool_chain,
                slots=nlu_result.slots,
                status=execution_plan.metadata.get("status", "OK"),
                message=execution_plan.metadata.get("message"),
                clarifications=execution_plan.clarifications,
                metadata=execution_plan.metadata
            )
            debug_output["stages"]["execution"] = {
                "status": "success",
                "output": execution_state.to_dict()
            }
        except Exception as e:
            debug_output["stages"]["execution"] = {"status": "failed", "error": str(e)}
            debug_output["errors"].append(f"Execution: {str(e)}")
            return debug_output

        # Stage 4: Composition
        try:
            response = compose_answer(
                {
                    "intent": nlu_result.intent,
                    "confidence": nlu_result.confidence,
                    "slots": nlu_result.slots,
                    "raw_query": nlu_result.raw_query
                },
                execution_state.to_dict(),
                execution_plan.metadata
            )
            debug_output["stages"]["composition"] = {
                "status": "success",
                "output": {
                    "answer_length": len(response.get("answer_md", "")),
                    "tables_count": len(response.get("tables", [])),
                    "charts_count": len(response.get("charts", [])),
                    "citations_count": len(response.get("citations", [])),
                    "status": response.get("status", "OK")
                }
            }
        except Exception as e:
            debug_output["stages"]["composition"] = {"status": "failed", "error": str(e)}
            debug_output["errors"].append(f"Composition: {str(e)}")

        return debug_output

    except Exception as e:
        return {"status": "error", "error": str(e), "stages": debug_output.get("stages", {})}


# ========== System Info ==========
@app.get("/info")
def system_info():
    """Get system information"""
    return {
        "version": "1.0.1",
        "architecture": {
            "pattern": "NLU → Planner → Executor → Composer",
            "layers": [
                {
                    "name": "NLU",
                    "responsibility": "Intent classification + Slot filling",
                    "model": "sentence-transformers/all-MiniLM-L6-v2"
                },
                {
                    "name": "Planner",
                    "responsibility": "Convert NLU results to execution plans",
                    "sql_templates": list(planner.sql_templates.keys())
                },
                {
                    "name": "Executor",
                    "responsibility": "Execute tool chains and collect evidence",
                    "tools": list(executor.tool_registry.keys())
                },
                {
                    "name": "Composer",
                    "responsibility": "Generate final user-facing response",
                    "llm": "Ollama (local)"
                }
            ]
        },
        "capabilities": {
            "intents": [
                "RAG",
                "SQL_tool",
                "RAG+SQL_tool",
                "CV_tool",
                "RAG+CV_tool"
            ],
            "domains": [
                "mowing",
                "field_standards",
                "generic"
            ],
            "sql_templates": list(planner.sql_templates.keys())
        }
    }


# ========== Run Server ==========
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)