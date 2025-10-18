# app.py

from __future__ import annotations
import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"
from datetime import datetime
from typing import Any, Dict, Optional

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from nlu import nlu_parse              # MiniLM + few-shot NLU (返回完整结果)
from executor import execute_plan      # ✅ 移除 build_route_plan 导入
from composer import compose_answer
from rag import RAG                    # 用于 health 返回 RAG 模式

# -------- FastAPI models --------
class NLUReq(BaseModel):
    text: str
    image_uri: Optional[str] = None

class AgentReq(BaseModel):
    text: str
    image_uri: Optional[str] = None
    nlu: Optional[Dict[str, Any]] = None

# -------- App --------
app = FastAPI(title="Parks Prototype API (Modular RAG/SQL/CV)", version="0.3.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/health")
def health():
    return {
        "status": "ok",
        "time": datetime.utcnow().isoformat(),
        "rag_mode": getattr(RAG, "mode", "none")
    }

@app.post("/nlu/parse")
def nlu_endpoint(req: NLUReq):
    """NLU 解析接口：返回意图、槽位和执行计划"""
    return nlu_parse(req.text, req.image_uri)

@app.post("/agent/answer")
def agent_answer(req: AgentReq):
    """
    智能代理接口：执行完整的 NLU → 工具调用 → 答案生成流程
    """
    # 1. NLU 解析（如果没有提供预解析的结果）
    nlu_result = req.nlu or nlu_parse(req.text, req.image_uri)
    
    # 2. 执行工具调用计划
    # nlu_result 已经包含 route_plan，直接使用
    plan = nlu_result.get("route_plan", [])
    slots = nlu_result.get("slots", {})
    
    state = execute_plan(plan, slots)
    
    # 3. 生成最终答案
    ans =  compose_answer(nlu_result, state)
    # print("Final Answer:", ans["answer_md"])
    return ans