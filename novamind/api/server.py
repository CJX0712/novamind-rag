"""FastAPI 服务层：薄装配，业务全在 core。作者：晨星

启动：uvicorn novamind.api.server:app --host 127.0.0.1 --port 8000
契约：docs/openapi.yaml
"""
from __future__ import annotations

from typing import Optional

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from ..config import get_settings
from ..core.agent import ReActAgent
from ..core.pipeline import RAGPipeline, build_offline_pipeline, build_pipeline
from ..eval.evaluate import evaluate

app = FastAPI(title="NovaMind RAG API", version="1.0.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

_settings = get_settings()
_pipeline: RAGPipeline = build_pipeline(_settings)
_agent = ReActAgent(_pipeline.llm, _pipeline)


# ---------- 请求/响应模型 ----------
class IngestRequest(BaseModel):
    text: Optional[str] = None
    path: Optional[str] = None
    doc_id: Optional[str] = None


class IngestResponse(BaseModel):
    doc_id: str
    chunks: int


class QueryRequest(BaseModel):
    query: str
    top_k: int = Field(default=5, ge=1, le=50)
    use_agent: bool = False


class Source(BaseModel):
    chunk_id: str
    doc_id: str
    text: str
    final_score: float


class QueryResponse(BaseModel):
    answer: str
    sources: list[Source]
    trace: Optional[dict] = None


class ChatRequest(BaseModel):
    message: str
    session_id: Optional[str] = None


class ToolCallOut(BaseModel):
    tool: str
    input: str
    output_preview: str


class ChatResponse(BaseModel):
    reply: str
    tool_calls: list[ToolCallOut]


class EvalRequest(BaseModel):
    k: int = Field(default=5, ge=1, le=50)


# ---------- 端点 ----------
@app.get("/health")
def health() -> dict:
    return _pipeline.health()


@app.post("/ingest", response_model=IngestResponse)
def ingest(req: IngestRequest) -> IngestResponse:
    if not req.text and not req.path:
        raise HTTPException(status_code=400, detail="text 与 path 至少提供一个")
    try:
        if req.path:
            doc_id, n = _pipeline.ingest_path(req.path, doc_id=req.doc_id)
        else:
            doc_id, n = _pipeline.ingest_text(req.text or "", doc_id=req.doc_id)
    except (ValueError, FileNotFoundError) as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    if n == 0:
        raise HTTPException(status_code=400, detail="文档为空或无法分块")
    return IngestResponse(doc_id=doc_id, chunks=n)


@app.post("/rag/query", response_model=QueryResponse)
def rag_query(req: QueryRequest) -> QueryResponse:
    if req.use_agent:
        result = _agent.run(req.query)
        return QueryResponse(answer=result.reply, sources=[], trace=None)
    result = _pipeline.query(req.query, top_k=req.top_k)
    sources = [
        Source(
            chunk_id=h.chunk.chunk_id,
            doc_id=h.chunk.doc_id,
            text=h.chunk.text,
            final_score=round(h.final_score, 6),
        )
        for h in result.sources
    ]
    return QueryResponse(answer=result.answer, sources=sources, trace=result.trace)


@app.post("/chat", response_model=ChatResponse)
def chat(req: ChatRequest) -> ChatResponse:
    result = _agent.run(req.message)
    return ChatResponse(
        reply=result.reply,
        tool_calls=[
            ToolCallOut(tool=t.tool, input=t.input, output_preview=t.output_preview)
            for t in result.tool_calls
        ],
    )


@app.post("/eval")
def run_eval(req: EvalRequest) -> dict:
    """黄金集评测。独立管道，绝不触生产索引。"""
    report = evaluate(build_offline_pipeline, k=req.k)
    return report.to_dict()


@app.post("/reset")
def reset() -> dict:
    cleared = _pipeline.reset()
    return {"cleared": cleared}


def main() -> None:
    import uvicorn

    uvicorn.run(app, host=_settings.host, port=_settings.port)


if __name__ == "__main__":
    main()
