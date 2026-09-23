"""核心模块单测。作者：晨星

运行（沙箱纪律：--basetemp 指仓内目录，防系统临时根批量删除守卫假失败）：
    python -m pytest -q tests --basetemp=data/.pytest-tmp -p no:cacheprovider
"""
from __future__ import annotations

import numpy as np
import pytest

from novamind.core.agent import ReActAgent, safe_calc
from novamind.core.bm25 import BM25Index
from novamind.core.embed import HashEmbedder, tokenize
from novamind.core.ingest import build_chunks, chunk_text
from novamind.core.llm import MockLLM
from novamind.core.pipeline import RAGPipeline, build_offline_pipeline, rrf_merge
from novamind.core.rerank import NoopReranker, fuse_scores
from novamind.core.store import MemoryStore
from novamind.core.types import Chunk, ScoredChunk


# ---------- 分词 ----------
def test_tokenize_cjk_bigram():
    tokens = tokenize("检索增强生成")
    assert "检索" in tokens and "索增" in tokens  # bigram
    assert "检" in tokens and "索" in tokens      # unigram


def test_tokenize_mixed():
    tokens = tokenize("BM25 与 RAG")
    assert "bm25" in tokens and "rag" in tokens


# ---------- 哈希嵌入 ----------
def test_hash_embedder_normalized():
    emb = HashEmbedder(dim=64)
    vecs = emb.embed(["你好世界", ""])
    assert vecs.shape == (2, 64)
    assert abs(np.linalg.norm(vecs[0]) - 1.0) < 1e-5


def test_hash_embedder_similar_texts_closer():
    emb = HashEmbedder(dim=256)
    v = emb.embed(["机器学习是人工智能的分支", "机器学习属于人工智能领域", "今天天气晴朗适合出行"])
    sim_related = float(v[0] @ v[1])
    sim_unrelated = float(v[0] @ v[2])
    assert sim_related > sim_unrelated


# ---------- 分块 ----------
def test_chunk_text_respects_size():
    text = "\n\n".join(f"第{i}段。" + "内容" * 100 for i in range(10))
    chunks = chunk_text(text, chunk_size=300, overlap=40)
    assert len(chunks) >= 3
    assert all(len(c) <= 400 for c in chunks)  # 段落聚合可能略超 chunk_size


def test_chunk_text_empty():
    assert chunk_text("") == []
    assert build_chunks("d1", "") == []


# ---------- 内存向量库 ----------
def test_memory_store_cosine_order():
    store = MemoryStore(dim=4)
    chunks = [Chunk(f"c{i}", "d", f"t{i}") for i in range(3)]
    vecs = np.array([[1, 0, 0, 0], [0.9, 0.1, 0, 0], [0, 1, 0, 0]], dtype=np.float32)
    vecs /= np.linalg.norm(vecs, axis=1, keepdims=True)
    store.upsert(chunks, vecs)
    results = store.search(np.array([1, 0, 0, 0], dtype=np.float32), top_n=2)
    assert [c.chunk_id for c, _ in results] == ["c0", "c1"]
    assert store.count() == 3
    assert store.clear() == 3 and store.count() == 0


def test_memory_store_shape_guard():
    store = MemoryStore(dim=4)
    with pytest.raises(ValueError):
        store.upsert([Chunk("c", "d", "t")], np.zeros((1, 8), dtype=np.float32))


# ---------- BM25 ----------
def test_bm25_exact_match_wins():
    # 注意：rank_bm25 的 IDF 在「词出现在恰好一半文档」时为 ln(1)=0，
    # 2 文档语料下任何只出现 1 次的词得分都是 0 —— 测试语料必须 ≥3 篇。
    idx = BM25Index()
    chunks = [
        Chunk("c1", "d1", "向量检索使用嵌入模型计算语义相似度"),
        Chunk("c2", "d2", "BM25 基于词频和逆文档频率计算得分"),
        Chunk("c3", "d3", "大语言模型通过注意力机制处理长上下文"),
    ]
    idx.rebuild(chunks)
    results = idx.search("BM25 词频", top_n=3)
    assert results and results[0][0].chunk_id == "c2"


# ---------- RRF + 融合 ----------
def test_rrf_merge_combines_both_lists():
    a = Chunk("a", "d", "ta")
    b = Chunk("b", "d", "tb")
    merged = rrf_merge([(a, 0.9)], [(a, 5.0), (b, 3.0)])
    assert merged[0].chunk.chunk_id == "a"
    assert merged[0].dense_score == 0.9 and merged[0].bm25_score == 5.0


def test_fuse_scores_weighted_not_override():
    """重排分不能独自决定排序：粗排第 1 名在 0.3 权重下必须保住第 1。"""
    cands = [
        ScoredChunk(chunk=Chunk("strong", "d", "t"), rrf_score=0.03, rerank_score=0.1),
        ScoredChunk(chunk=Chunk("weak", "d", "t"), rrf_score=0.01, rerank_score=0.95),
    ]
    fused = fuse_scores(cands, weight=0.3)
    assert fused[0].chunk.chunk_id == "strong"


# ---------- 计算器 ----------
def test_safe_calc_blocks_injection():
    with pytest.raises(ValueError):
        safe_calc("__import__('os').system('x')")
    with pytest.raises(ValueError):
        safe_calc("open('f')")
    assert safe_calc("2**10") == 1024
    assert abs(safe_calc("1/3") - 0.3333) < 1e-3


# ---------- 端到端（离线管道） ----------
def test_offline_pipeline_end_to_end():
    pipeline = build_offline_pipeline()
    doc_id, n = pipeline.ingest_text(
        "混合检索同时使用稠密向量检索和 BM25 稀疏检索，再通过 RRF 融合排序。" * 3,
        doc_id="e2e-doc",
    )
    assert n >= 1
    hits = pipeline.retrieve("什么是混合检索", top_k=2)
    assert hits and hits[0].chunk.doc_id == "e2e-doc"
    result = pipeline.query("混合检索包含哪些部分", top_k=2)
    assert result.answer and len(result.sources) >= 1
    assert result.trace["hits"] >= 1


def test_agent_calculator_tool_used():
    pipeline = build_offline_pipeline()
    agent = ReActAgent(MockLLM(), pipeline)
    result = agent.run("请计算 7*8+4")
    assert any(t.tool == "calculator" for t in result.tool_calls)
    assert "60" in result.reply


def test_eval_isolated_from_production():
    """生产索引灌入干扰后，评测结果必须逐位不变。"""
    from novamind.eval.evaluate import evaluate

    before = evaluate(build_offline_pipeline, k=5).to_dict()
    prod = build_offline_pipeline()
    prod.ingest_text("无关干扰" * 500, doc_id="noise")
    after = evaluate(build_offline_pipeline, k=5).to_dict()
    assert before == after
    assert before["recall_at_k"] >= 0.75
