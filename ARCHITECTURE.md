# NovaMind RAG 系统架构

> 作者：晨星 · v1.0.0

## 1. 总体架构

```
┌─────────────────────────────────────────────────────────────┐
│                     接入层                                   │
│   web/index.html（单文件控制台）   FastAPI（api/server.py）    │
└──────────────────────────┬──────────────────────────────────┘
                           │ HTTP（契约 docs/openapi.yaml）
┌──────────────────────────▼──────────────────────────────────┐
│                     编排层                                   │
│   RAGPipeline（core/pipeline.py）   ReActAgent（core/agent.py）│
│   - 摄取编排                     - 工具循环（≤6 步）           │
│   - 检索编排                     - AST 白名单安全计算           │
│   - 生成编排                     - 规则降级（LLM 缺席时）       │
└───┬───────┬───────┬───────┬─────────────────────────────────┘
    │       │       │       │
┌───▼───┐┌──▼───┐┌──▼────┐┌─▼─────┐   能力层（全部 Protocol 注入）
│Embedder││Vector││Reranker││LLM     │
│       ││Store ││       ││Provider│
└───┬───┘└──┬───┘└──┬────┘└──┬─────┘
    │       │       │        │
 生产:    生产:   生产:    生产:
 ONNX    FAISS   ONNX     llama-cpp
 BGE     FlatIP  reranker GGUF(线程锁4)
    │       │       │        │
 兜底:    兜底:   兜底:    兜底:
 哈希     内存    Noop     Mock模板
 bigram  余弦
```

**单一职责**：每个能力一个模块、一个 Protocol；编排层只依赖 Protocol，运行时注入实现。
**可独立验证**：每个模块有专属单测；`build_offline_pipeline()` 一键组装全兜底链路。

## 2. 模块接口定义

| 模块 | Protocol | 关键方法 | 生产实现 | 离线兜底 |
|------|----------|----------|----------|----------|
| core/embed.py | `Embedder` | `embed(texts) -> ndarray(n, dim)`，L2 归一 | OnnxEmbedder（CLS pooling，输入名运行时探测） | HashEmbedder（blake2b bigram） |
| core/store.py | `VectorStore` | `upsert / search / count / clear` | FaissStore（IndexFlatIP + JSONL 持久化） | MemoryStore（余弦） |
| core/bm25.py | `BM25Index` | `rebuild / search` | rank-bm25 库 | 内置纯 Python 实现 |
| core/rerank.py | `Reranker` | `score(query, texts) -> [0..1]` | OnnxReranker（sigmoid） | NoopReranker |
| core/llm.py | `LLMProvider` | `generate(messages) -> str` | LlamaCppLLM（GGUF，n_threads≤4） | MockLLM（引用模板） |
| core/ingest.py | 函数 | `parse_file / chunk_text / build_chunks` | pypdf + 段落聚合滑窗 | 同左（无外部依赖） |
| core/pipeline.py | `RAGPipeline` | `ingest_text / retrieve / query / reset / health` | — | — |
| core/agent.py | `ReActAgent` | `run(message) -> AgentResult` | LLM 驱动循环 | 规则降级 |
| eval/evaluate.py | 函数 | `evaluate(pipeline_factory, k)` | 黄金集 8 条 | 同左 |

每个 Fallback 包装器（FallbackEmbedder/Store/Reranker/LLM）在生产实现初始化失败时自动降级，并把异常写入 `status().error`——`/health` 原样暴露，**禁止静默回退**。

## 3. 调用关系（两条主链路）

### 摄取链路
```
text/path → ingest.parse_file → chunk_text(段落聚合,400字,重叠60)
→ build_chunks → Embedder.embed → VectorStore.upsert + BM25Index.rebuild
```

### 问答链路
```
query → Embedder.embed → VectorStore.search(top20)  ┐
     → BM25Index.search(top20)                      ├→ rrf_merge(k=60)
     → Reranker.score(candidates)                   ↓
     → fuse_scores(final = 0.7·rrf_norm + 0.3·rerank_norm) → top5
     → LLMProvider.generate(RAG_PROMPT + context) → answer + sources + trace
```

Agent 链路（/chat）：`ReActAgent.run` 循环解析 `Action: tool[input]` / `Final Answer:`；
工具集 = kb_search（即 pipeline.retrieve）/ calculator（AST 白名单，拒绝任意代码）/ memory_*。

## 4. 关键设计决策（详见 docs/decisions/）

| ADR | 决策 | 核心理由 |
|-----|------|----------|
| ADR-001 | faiss-cpu 而非 lancedb/Qdrant | Windows 零编译可装 |
| ADR-002 | ONNXRuntime + ModelScope 而非 transformers + HF | HF 不可达；免 torch |
| ADR-003 | llama-cpp 线程锁 4 | CPU 推理瓶颈在内存带宽，多线程慢 4 倍 |

## 5. 工程纪律（写进代码的护栏）

1. **评测隔离**：`evaluate()` 每次经 `pipeline_factory` 新建独立管道；测试断言「生产索引灌干扰后指标逐位不变」
2. **融合而非覆盖**：`fuse_scores` 加权融合，单测锁定「粗排第 1 在 0.3 权重下不被重排盖掉」
3. **输入名探测**：ONNX feeds 按 `sess.get_inputs()` 实际名字过滤，兼容不同导出
4. **打分留痕**：`ScoredChunk` 记录 dense/bm25/rrf/rerank/final 五路分数，`/rag/query` 的 trace 原样返回
5. **汇总行判据**：verify.py 的 `passed=N failed=M` 是唯一判据；无汇总行 = 运行中断 ≠ 测试失败

## 6. 部署形态

- **单机本地**：`uvicorn novamind.api.server:app` + 浏览器打开 web/index.html
- **干净环境复现**：`pip install -r requirements.txt && python verify.py`（离线全绿）
- **CI**：GitHub Actions Ubuntu runner 跑 P0 扫描 + 单测 + 离线 verify（见 .github/workflows/ci.yml）
- **模型分发**：不进 git；`scripts/download_models.py` 从 ModelScope 拉取，断点重跑自动跳过已完成文件
