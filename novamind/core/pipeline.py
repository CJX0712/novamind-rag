"""RAG 管道：摄取 + 混合检索（dense ∪ BM25 → RRF → 融合重排）+ 生成。作者：晨星"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from ..config import Settings, get_settings
from .bm25 import BM25Index
from .embed import Embedder, FallbackEmbedder
from .ingest import build_chunks, parse_file
from .llm import FallbackLLM, LLMProvider
from .rerank import FallbackReranker, Reranker, fuse_scores
from .store import Chunk, FallbackStore, VectorStore
from .types import BackendStatus, ScoredChunk


def rrf_merge(
    dense: list[tuple[Chunk, float]],
    sparse: list[tuple[Chunk, float]],
    k: int = 60,
) -> list[ScoredChunk]:
    """Reciprocal Rank Fusion 合并两路检索结果。"""
    merged: dict[str, ScoredChunk] = {}
    for rank, (chunk, score) in enumerate(dense):
        entry = merged.setdefault(chunk.chunk_id, ScoredChunk(chunk=chunk))
        entry.dense_score = score
        entry.rrf_score += 1.0 / (k + rank + 1)
    for rank, (chunk, score) in enumerate(sparse):
        entry = merged.setdefault(chunk.chunk_id, ScoredChunk(chunk=chunk))
        entry.bm25_score = score
        entry.rrf_score += 1.0 / (k + rank + 1)
    return sorted(merged.values(), key=lambda c: c.rrf_score, reverse=True)


RAG_PROMPT = (
    "你是知识库问答助手。只依据 [CONTEXT] 中的内容回答，"
    "不知道就明确说不知道。回答末尾用 [来源: chunk_id] 标注引用。\n"
    "[CONTEXT]\n{context}\n[/CONTEXT]"
)


@dataclass
class QueryResult:
    answer: str
    sources: list[ScoredChunk]
    trace: dict


class RAGPipeline:
    """完整 RAG 链路。所有后端经 Protocol 注入，可整体替换为离线兜底。"""

    def __init__(
        self,
        embedder: Embedder,
        store: VectorStore,
        reranker: Reranker,
        llm: LLMProvider,
        settings: Optional[Settings] = None,
    ) -> None:
        self.settings = settings or get_settings()
        self.embedder = embedder
        self.store = store
        self.reranker = reranker
        self.llm = llm
        self._bm25 = BM25Index()
        self._chunks_by_id: dict[str, Chunk] = {}

    # ---------- 摄取 ----------
    def ingest_text(self, text: str, doc_id: Optional[str] = None) -> tuple[str, int]:
        doc_id = doc_id or f"doc-{len({c.doc_id for c in self._chunks_by_id.values()}) + 1}"
        chunks = build_chunks(
            doc_id, text, self.settings.chunk_size, self.settings.chunk_overlap
        )
        if not chunks:
            return doc_id, 0
        vectors = self.embedder.embed([c.text for c in chunks])
        self.store.upsert(chunks, vectors)
        for c in chunks:
            self._chunks_by_id[c.chunk_id] = c
        self._bm25.rebuild(list(self._chunks_by_id.values()))
        return doc_id, len(chunks)

    def ingest_path(self, path: Path, doc_id: Optional[str] = None) -> tuple[str, int]:
        text = parse_file(Path(path))
        return self.ingest_text(text, doc_id=doc_id or Path(path).stem)

    # ---------- 检索 ----------
    def retrieve(self, query: str, top_k: Optional[int] = None) -> list[ScoredChunk]:
        s = self.settings
        top_k = top_k or s.default_top_k
        q_vec = self.embedder.embed([query])[0]
        dense = self.store.search(q_vec, s.dense_top_n)
        sparse = self._bm25.search(query, s.bm25_top_n)
        candidates = rrf_merge(dense, sparse, k=s.rrf_k)[: max(s.dense_top_n, top_k)]
        if not candidates:
            return []
        scores = self.reranker.score(query, [c.chunk.text for c in candidates])
        for c, sc in zip(candidates, scores):
            c.rerank_score = sc
        fused = fuse_scores(candidates, weight=s.rerank_weight)
        return fused[:top_k]

    # ---------- 问答 ----------
    def query(self, question: str, top_k: Optional[int] = None) -> QueryResult:
        hits = self.retrieve(question, top_k)
        if not hits:
            return QueryResult(
                answer="知识库中没有与该问题相关的内容，请先摄取文档。",
                sources=[],
                trace={"hits": 0},
            )
        context = "\n\n".join(
            f"[{h.chunk.chunk_id}] {h.chunk.text}" for h in hits
        )
        messages = [
            {"role": "system", "content": RAG_PROMPT.format(context=context)},
            {"role": "user", "content": question},
        ]
        answer = self.llm.generate(messages, max_tokens=self.settings.llm_max_tokens)
        trace = {
            "hits": len(hits),
            "ranking": [
                {
                    "chunk_id": h.chunk.chunk_id,
                    "rrf": round(h.rrf_score, 6),
                    "rerank": None if h.rerank_score is None else round(h.rerank_score, 6),
                    "final": round(h.final_score, 6),
                }
                for h in hits
            ],
        }
        return QueryResult(answer=answer, sources=hits, trace=trace)

    # ---------- 运维 ----------
    def reset(self) -> int:
        n = self.store.clear()
        self._chunks_by_id.clear()
        self._bm25.rebuild([])
        return n

    def health(self) -> dict:
        backends = {}
        for key, backend in (
            ("embedder", self.embedder),
            ("store", self.store),
            ("reranker", self.reranker),
            ("llm", self.llm),
        ):
            name, error = backend.status()
            backends[key] = BackendStatus(name=name, error=error)
        degraded = any(b.error for b in backends.values())
        return {
            "status": "degraded" if degraded else "ok",
            "backends": {
                k: {"name": b.name, "error": b.error} for k, b in backends.items()
            },
            "chunks": self.store.count(),
        }


def build_pipeline(settings: Optional[Settings] = None) -> RAGPipeline:
    """按配置组装生产管道（失败自动回退兜底，error 记入健康状态）。"""
    s = settings or get_settings()
    embedder = FallbackEmbedder(s.embed_model_dir, offline=s.offline, offline_dim=s.embed_dim_offline)
    store = FallbackStore(embedder.dim, s.index_dir, offline=s.offline)
    reranker = FallbackReranker(s.rerank_model_dir, offline=s.offline)
    llm = FallbackLLM(
        s.llm_model_path, offline=s.offline, n_ctx=s.llm_ctx, n_threads=s.llm_threads
    )
    return RAGPipeline(embedder, store, reranker, llm, settings=s)


def build_offline_pipeline(index_dim: int = 256) -> RAGPipeline:
    """全离线管道：评测/CI 专用，绝不读生产索引。"""
    s = Settings(offline=True)
    embedder = FallbackEmbedder(s.embed_model_dir, offline=True, offline_dim=index_dim)
    store = FallbackStore(embedder.dim, s.index_dir, offline=True)
    reranker = FallbackReranker(s.rerank_model_dir, offline=True)
    llm = FallbackLLM(s.llm_model_path, offline=True)
    return RAGPipeline(embedder, store, reranker, llm, settings=s)
