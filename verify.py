"""一键全链路自检：离线默认全绿，--online 走真实模型。作者：晨星

用法：
    python verify.py            # 离线路径（无网络无模型也能跑通全链路）
    python verify.py --online   # 真实模型路径（需先 scripts/download_models.py）

判据纪律：
- 汇总行（passed/failed）是唯一判据；无汇总行 + 非零码 = 运行中断，不是测试失败
- 断言后端真实状态（禁静默回退假成功）
- 评测使用独立索引，先灌干扰再断言指标逐位不变
"""
from __future__ import annotations

import argparse
import json
import sys
import tempfile
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

CHECKS: list[tuple[str, bool, str]] = []


def check(name: str, ok: bool, detail: str = "") -> bool:
    CHECKS.append((name, ok, detail))
    mark = "PASS" if ok else "FAIL"
    print(f"[{mark}] {name}" + (f" - {detail}" if detail else ""))
    return ok


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--online", action="store_true", help="使用真实模型")
    args = parser.parse_args()

    t0 = time.time()
    print(f"=== NovaMind 全链路自检（{'online' if args.online else 'offline'}）===")

    # 1. 配置与导入
    from novamind.config import Settings
    from novamind.core.agent import ReActAgent, safe_calc
    from novamind.core.embed import tokenize
    from novamind.core.pipeline import build_offline_pipeline, build_pipeline
    from novamind.eval.evaluate import evaluate

    settings = Settings(offline=not args.online)
    check("配置加载", settings.llm_threads <= 4, f"threads={settings.llm_threads}")

    # 2. 分词器
    tokens = tokenize("RAG检索增强生成 combined with BM25")
    check("中英文分词", any("检索" in t or "索增" in t for t in tokens) and "bm25" in tokens,
          f"tokens={len(tokens)}")

    # 3. 安全计算器
    try:
        safe_calc("__import__('os')")
        calc_safe = False
    except ValueError:
        calc_safe = True
    check("计算器拒绝危险表达式", calc_safe and abs(safe_calc("12*(3+4)") - 84) < 1e-9)

    # 4. 管道构建 + 后端状态（禁静默回退假成功）
    if args.online:
        pipeline = build_pipeline(settings)
    else:
        pipeline = build_offline_pipeline()
    health = pipeline.health()
    print(f"  后端: {json.dumps(health['backends'], ensure_ascii=False)}")
    if args.online:
        check("在线模式后端零回退", health["status"] == "ok",
              json.dumps({k: v["error"] for k, v in health["backends"].items()}, ensure_ascii=False))
    else:
        check("离线模式管道可用", health["chunks"] == 0)

    # 5. 摄取 + 检索
    doc_text = (
        "检索增强生成（RAG）将信息检索与大语言模型结合。"
        "系统先检索相关文档片段，再注入提示词生成回答。\n\n"
        "混合检索同时使用稠密向量检索和 BM25 稀疏检索，"
        "再通过 RRF 融合排序，兼顾语义与字面匹配。"
    )
    doc_id, n_chunks = pipeline.ingest_text(doc_text, doc_id="selftest-doc")
    check("摄取分块", n_chunks >= 1, f"chunks={n_chunks}")

    hits = pipeline.retrieve("什么是 RAG 混合检索", top_k=3)
    check("检索命中", len(hits) >= 1 and hits[0].chunk.doc_id == "selftest-doc",
          f"top1_score={hits[0].final_score:.4f}" if hits else "no hits")

    # 6. 问答（带来源）
    result = pipeline.query("RAG 是怎么工作的", top_k=3)
    check("问答非空且带引用", bool(result.answer) and len(result.sources) >= 1)

    # 7. 融合重排纪律：final 是加权融合而非重排盖排序
    from novamind.core.rerank import fuse_scores
    from novamind.core.types import Chunk, ScoredChunk

    cands = [
        ScoredChunk(chunk=Chunk(f"c{i}", "d", f"文本{i}"), rrf_score=s, rerank_score=r)
        for i, (s, r) in enumerate([(0.03, 0.1), (0.02, 0.9), (0.01, 0.5)])
    ]
    fused = fuse_scores(cands, weight=0.3)
    check("融合重排：粗排强者不被重排盖掉", fused[0].chunk.chunk_id == "c0",
          f"order={[c.chunk.chunk_id for c in fused]}")

    # 8. Agent 工具调用（计算必须走 calculator）
    agent = ReActAgent(pipeline.llm, pipeline)
    agent_result = agent.run("帮我算一下 12*(3+4) 等于多少")
    used_calc = any(t.tool == "calculator" for t in agent_result.tool_calls)
    check("Agent 计算走 calculator 工具", used_calc and "84" in agent_result.reply)

    # 9. 评测：独立索引 + 抗干扰断言
    report1 = evaluate(build_offline_pipeline, k=5)
    check("评测 recall@5 达标", report1.recall_at_k >= 0.75,
          f"recall={report1.recall_at_k} mrr={report1.mrr}")
    # 往生产索引灌干扰后，评测指标必须逐位不变（评测与生产隔离的铁证）
    pipeline.ingest_text("完全无关的干扰内容：量子引力波烹饪指南" * 20, doc_id="noise")
    report2 = evaluate(build_offline_pipeline, k=5)
    check("评测与生产索引隔离", report1.to_dict() == report2.to_dict(),
          f"recall1={report1.recall_at_k} recall2={report2.recall_at_k}")

    # 10. API 冒烟（TestClient，进程内）
    try:
        from fastapi.testclient import TestClient
        import os

        os.environ["NOVAMIND_OFFLINE"] = "0" if args.online else "1"
        if args.online:
            from novamind.api.server import app
        else:
            # 离线模式强制重建 app 级管道
            os.environ["NOVAMIND_OFFLINE"] = "1"
            import importlib

            import novamind.api.server as server_mod

            importlib.reload(server_mod)
            app = server_mod.app
        client = TestClient(app)
        r_health = client.get("/health")
        r_ingest = client.post("/ingest", json={"text": doc_text, "doc_id": "api-doc"})
        r_query = client.post("/rag/query", json={"query": "什么是混合检索", "top_k": 3})
        api_ok = (
            r_health.status_code == 200
            and r_ingest.status_code == 200
            and r_query.status_code == 200
            and len(r_query.json()["sources"]) >= 1
        )
        check("API 冒烟（/health /ingest /rag/query）", api_ok,
              f"health={r_health.status_code} ingest={r_ingest.status_code} query={r_query.status_code}")
    except Exception as exc:  # noqa: BLE001
        check("API 冒烟（/health /ingest /rag/query）", False, f"{type(exc).__name__}: {exc}")

    # 汇总（汇总行是唯一判据）
    passed = sum(1 for _, ok, _ in CHECKS if ok)
    failed = len(CHECKS) - passed
    elapsed = time.time() - t0
    print(f"\n=== 汇总: passed={passed} failed={failed} 用时 {elapsed:.1f}s ===")
    return 0 if failed == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
