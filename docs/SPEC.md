# Spec - NovaMind RAG v1.0.0

> 生成日期：2026-09-24
> 作者：晨星
> 状态：已确认

---

## 1. 产品定义

- **一句话描述**：基于全开源组件的本地知识库问答系统——混合检索（向量+BM25）+ 融合重排 + ReAct Agent，CPU 可跑，离线可验证。
- **目标用户**：需要在本地/私有环境构建知识库问答能力的开发者与小团队。
- **核心问题**：无 GPU、无外网依赖（HuggingFace 不可达）环境下，搭建一套端到端可运行、可复现、可评测的 RAG+Agent 系统。

## 2. MVP 范围（锁定）

| 优先级 | 功能 | 验收标准摘要 | RICE |
|--------|------|-------------|------|
| P0 | 文档摄取（PDF/TXT/MD → 分块 → 入库） | 摄取后可检索到对应分块 | 高 |
| P0 | 混合检索（FAISS 向量 ∪ BM25，RRF 粗排） | recall@5 ≥ 阈值（黄金集） | 高 |
| P0 | 融合重排（ONNX cross-encoder 分数加权融合，不盖排序） | top-1 命中率不低于粗排 | 高 |
| P0 | RAG 问答（带引用来源） | 回答含来源分块 ID | 高 |
| P0 | ReAct Agent（kb_search / calculator / memory 工具） | 多跳问题可正确调用工具 | 中 |
| P0 | FastAPI 服务（/health /ingest /rag/query /chat /eval） | OpenAPI 契约一致 | 高 |
| P0 | 离线兜底全链路（无网络无模型全绿） | verify.py 离线模式退出码 0 | 高 |
| P1 | 单文件 HTML 控制台 | 可调 /chat 完成问答 | 中 |
| P1 | 本地 GGUF 生成（llama-cpp，线程锁 4） | 真实模型可生成非空回答 | 中 |

## 3. 明确不做（Out-of-Scope）

| 不做的功能 | 原因 | 何时考虑 |
|------------|------|----------|
| GPU 加速 | 目标环境无 GPU | 有 GPU 环境后 |
| 多用户/权限体系 | MVP 单租户 | v2.0 |
| 分布式向量库（Qdrant Server/Milvus） | 本地单机优先，FAISS 足够 | 数据量超百万级 |
| 前端构建链（React/Vue） | 沙箱 npm registry 被劫持；单文件 HTML 满足控制台需求 | 产品化阶段 |
| 微调/训练 | 复用开源模型，不自研训练 | 有领域数据后 |

## 4. 技术架构（锁定 · 版本锚定）

| 层 | 技术 | 版本 | 锁定原因 |
|----|------|------|----------|
| 语言 | Python | 3.13.x | 托管运行时 |
| Web | FastAPI + uvicorn | 见 requirements.lock.txt | 异步、自动 OpenAPI |
| 向量库 | faiss-cpu | 见 lock | win 有 wheel，免编译 |
| 稀疏检索 | rank-bm25 | 见 lock | 纯 Python |
| 嵌入 | onnxruntime + tokenizers（BGE 系列 ONNX） | 见 lock | HF 不可达 → ModelScope 拉 ONNX |
| 重排 | onnxruntime（bge-reranker ONNX） | 见 lock | 同上 |
| LLM | llama-cpp-python（GGUF） | 见 lock | Ollama 未运行时的本地推理 |
| 文档解析 | pypdf | 见 lock | 纯 Python |
| 测试 | pytest | 见 lock | — |

**版本锚定**：`requirements.lock.txt` 由已验证可用的 worldai venv `pip freeze` 生成，干净环境 `pip install -r requirements.lock.txt` 一键复现。

## 5. API 端点清单（锁定，详见 docs/openapi.yaml）

| Method | Path | 功能 | 请求体 | 响应体 |
|--------|------|------|--------|--------|
| GET | /health | 健康检查 | — | {status, backends} |
| POST | /ingest | 摄取文档 | {path? , text? , doc_id?} | {doc_id, chunks} |
| POST | /rag/query | 检索问答 | {query, top_k?, use_agent?} | {answer, sources[], trace?} |
| POST | /chat | Agent 对话 | {message, session_id?} | {reply, tool_calls[]} |
| POST | /eval | 黄金集评测（独立索引） | {k?} | {recall@k, mrr, per_query[]} |
| POST | /reset | 清空索引 | — | {cleared} |

## 6. 数据模型

| 实体 | 核心字段 | 说明 |
|------|----------|------|
| Chunk | chunk_id, doc_id, text, meta, vector | 检索最小单元 |
| Document | doc_id, source, chunk_count | 摄取记录 |
| SearchResult | chunk, dense_rank, bm25_rank, rrf_score, rerank_score, final_score | 全链路打分会留痕 |

## 7. 页面清单

| 页面 | 文件 | 核心功能 | 对应 API |
|------|------|----------|----------|
| 控制台 | web/index.html（单文件，内联 CSS/JS） | 问答、来源展示、健康状态、评测触发 | /chat /rag/query /health /eval |

## 8. 设计规范（锁定）

- **图标**：内联 SVG（描边风格，16/20/24px），禁止 emoji 作功能图标
- **配色**：浅色主题，主色 Indigo #4F46E5（纯色，禁紫粉渐变）
- **文案**：中文为主，无空洞占位文案
- **颜色引用**：CSS 变量 token 化，禁硬编码（#fff/#000 除外）

## 9. 验收标准（EARS 格式）

| 编号 | 功能 | 验收标准 | 优先级 |
|------|------|----------|--------|
| AC-01 | 摄取 | When 用户提交合法文档，系统必须分块入库并返回 chunk 数 > 0 | P0 |
| AC-02 | 检索 | When 查询与库中文档相关，系统必须在前 top_k 返回相关分块 | P0 |
| AC-03 | 重排 | While 重排器可用，系统必须将其分数加权融合而非直接覆盖粗排顺序 | P0 |
| AC-04 | 问答 | When 检索命中，回答必须携带来源 chunk_id | P0 |
| AC-05 | 离线 | While 无网络且无本地模型，verify.py 必须以退出码 0 完成全链路自检 | P0 |
| AC-06 | 评测 | When 运行 /eval，系统必须使用独立新建索引，不读不写生产索引 | P0 |
| AC-07 | Agent | When 问题需要计算，Agent 必须调用 calculator 工具而非直接编造 | P0 |
| AC-08 | 健康 | GET /health 必须返回各后端真实状态（含错误字段，禁静默回退） | P0 |

## 10. 边界与约束

- Python 3.13+；Windows/Linux/macOS 均可运行
- 嵌入维度随模型锁定（bge-small-zh-v1.5 = 512）；离线哈希嵌入 256 维（独立索引，互不相混）
- CPU 推理线程数锁定 2–4（内存带宽瓶颈，非算力）
- 模型文件不进 git 仓库

## 11. 内嵌已知坑

| 坑 | 指纹 | 根因 | 修法 |
|----|------|------|------|
| 评测指标全错无报错 | eval | 复用生产索引 | 评测每次新建内存索引 |
| 静默回退假成功 | store/embed | 弃用参数致后端回退内存 | 断言 backend error 字段为 None |
| pytest 假失败 | sandbox | 系统临时根批量删除被守卫拦截 | --basetemp 指到仓内目录 |
| llama.cpp 慢 4 倍 | llm | 默认线程 cpu_count-1 超内存带宽 | 线程锁 2–4 |
| 中文重排被压 | rerank | 英文 reranker 直接盖排序 | 分数加权融合 |
| ONNX 输入名不一致 | embed/rerank | 导出差异 token_type_ids 有无 | 运行时探测输入名再喂 |
| HF 不可达 | models | 网络 | ModelScope 直链下载 |
| jieba/lancedb 无 win wheel | deps | 无预编译 | 自写 CJK 切分 / FAISS |

## 12. 端到端验证步骤

```bash
pip install -r requirements.lock.txt
python verify.py            # 离线路径全链路自检，退出码 0
python scripts/download_models.py   # 可选：拉取真实模型（ModelScope）
python verify.py --online   # 可选：真实模型路径验证
uvicorn novamind.api.server:app --port 8000
curl -X POST localhost:8000/ingest -H "Content-Type: application/json" -d '{"text":"...","doc_id":"d1"}'
curl -X POST localhost:8000/rag/query -H "Content-Type: application/json" -d '{"query":"..."}'
```

## 13. 变更记录

| 日期 | 变更内容 | 原因 | 影响范围 |
|------|----------|------|----------|
| 2026-09-24 | v1.0.0 初始锁定 | — | 全部 |
