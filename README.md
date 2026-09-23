# NovaMind RAG

<p align="center">
  <a href="https://github.com/CJX0712/novamind-rag/actions/workflows/ci.yml"><img src="https://github.com/CJX0712/novamind-rag/actions/workflows/ci.yml/badge.svg" alt="ci"></a>
  <a href="https://github.com/CJX0712/novamind-rag/releases"><img src="https://img.shields.io/github/v/release/CJX0712/novamind-rag?sort=semver" alt="release"></a>
  <a href="https://github.com/CJX0712/novamind-rag/blob/main/LICENSE"><img src="https://img.shields.io/github/license/CJX0712/novamind-rag" alt="license"></a>
  <img src="https://img.shields.io/badge/author-%E6%99%A8%E6%98%9F-1f6feb" alt="author">
</p>

> 端到端本地知识库问答系统：混合检索（FAISS 向量 ∪ BM25）+ 融合重排（ONNX cross-encoder）+ ReAct Agent + 本地 GGUF 生成。全开源组件，CPU 可跑，离线可验证。
>
> 作者：晨星 · License：MIT

## 特性

- **混合检索**：稠密向量（BGE ONNX）与 BM25 稀疏检索双路召回，RRF 融合粗排
- **融合重排**：cross-encoder 分数与粗排分数**加权融合**（默认 0.3），重排器永不独断排序——防止弱语言重排器压掉正确答案
- **ReAct Agent**：kb_search / calculator（AST 白名单安全计算）/ memory 工具，计算必走工具不心算
- **离线可复现**：每个外部依赖均为 Protocol + 可注入实现，内置离线兜底（哈希 bigram 嵌入 / 内存余弦 / MockLLM），`python verify.py` 无网络、无模型、无 Key 全链路 12 项自检全绿
- **真实后端状态透明**：`/health` 暴露每个后端的实现名与回退原因，杜绝静默降级假成功

## 快速开始

```bash
# 1. 安装核心依赖（全平台零编译）
pip install -r requirements.txt
# 或精确复现：pip install -r requirements.lock.txt

# 2. 离线自检（不需要任何模型/网络）
python verify.py

# 3. 可选：下载真实模型（ModelScope 通道，约 810MB）
python scripts/download_models.py

# 4. 可选：本地 GGUF 生成支持（Windows 需 MSVC 编译，见 requirements-llm.txt）
pip install -r requirements-llm.txt

# 5. 启动服务
uvicorn novamind.api.server:app --host 127.0.0.1 --port 8000

# 6. 打开控制台（或直接用 API）
#    浏览器打开 web/index.html
```

在线模式自检（需已下载模型）：`python verify.py --online`

## API 一览（契约：docs/openapi.yaml）

| Method | Path | 功能 |
|--------|------|------|
| GET | /health | 健康检查（含各后端真实状态与回退原因） |
| POST | /ingest | 摄取文档（text / path，支持 txt/md/pdf） |
| POST | /rag/query | 检索问答（带引用来源与全链路打分 trace） |
| POST | /chat | ReAct Agent 对话（工具调用可见） |
| POST | /eval | 黄金集评测（独立索引，不触生产数据） |
| POST | /reset | 清空生产索引 |

```bash
curl -X POST localhost:8000/ingest -H "Content-Type: application/json" \
  -d '{"text":"你的文档内容……","doc_id":"my-doc"}'
curl -X POST localhost:8000/rag/query -H "Content-Type: application/json" \
  -d '{"query":"你的问题","top_k":5}'
```

## 配置（环境变量）

| 变量 | 默认 | 说明 |
|------|------|------|
| NOVAMIND_OFFLINE | 0 | 1 = 强制全离线兜底实现 |
| NOVAMIND_EMBED_DIR | models/bge-small-zh-v1.5 | 嵌入模型目录（model.onnx + tokenizer.json） |
| NOVAMIND_RERANK_DIR | models/bge-reranker-base | 重排模型目录 |
| NOVAMIND_LLM_PATH | models/qwen2.5-0.5b-instruct-q4_k_m.gguf | GGUF 路径 |
| NOVAMIND_LLM_THREADS | min(4, cpu_count) | CPU 推理线程（锁 2-4，内存带宽瓶颈，见 ADR-003） |
| NOVAMIND_RERANK_WEIGHT | 0.3 | 重排分数融合权重 |
| NOVAMIND_INDEX_DIR | data/index | FAISS 索引持久化目录 |
| NOVAMIND_PORT | 8000 | 服务端口 |

## 测试与质量门禁

```bash
python -m pytest -q tests -p no:cacheprovider   # 15 项单测
python verify.py                                 # 12 项全链路自检（离线）
python verify.py --online                        # 12 项全链路自检（真实模型）
python scripts/smoke_live.py                     # 7 项真实进程冒烟（uvicorn + HTTP）
python tools/scan_emoji.py                       # P0 门禁：禁 emoji 功能图标
```

CI（GitHub Actions）在干净 Ubuntu runner 上跑同一套离线验证，证明与开发机无关。

## 文档

- [ARCHITECTURE.md](ARCHITECTURE.md) — 系统架构与模块接口
- [docs/SPEC.md](docs/SPEC.md) — 规格契约（范围/API/验收标准/已知坑）
- [docs/openapi.yaml](docs/openapi.yaml) — API 契约
- [docs/decisions/](docs/decisions/) — ADR 架构决策记录

## 技术栈

FastAPI · faiss-cpu · rank-bm25 · onnxruntime + tokenizers（BGE 嵌入/重排）· llama-cpp-python（Qwen2.5 GGUF，可选）· pypdf · pytest

模型来源（ModelScope）：Xenova/bge-small-zh-v1.5（int8 ONNX, 24MB）· Xenova/bge-reranker-base（int8 ONNX, 279MB）· Qwen/Qwen2.5-0.5B-Instruct-GGUF（Q4_K_M, 491MB）
