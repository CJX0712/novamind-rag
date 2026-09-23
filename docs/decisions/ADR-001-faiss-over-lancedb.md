# ADR-001: 向量库选 faiss-cpu 而非 lancedb / Qdrant Server

## Status: Accepted (2026-09-24)
## Author: 晨星

## Background
目标环境为 Windows 无 GPU 机器，要求干净环境零编译一键复现。lancedb 只发 manylinux/macOS 轮子，Windows 直接安装失败；Qdrant Server 需要额外进程，违背"单机一键"目标；faiss-cpu 的 pip 轮子在本环境已验证可装可用（worldai venv 实测）。

## Decision
生产向量库使用 faiss-cpu（IndexFlatIP + L2 归一化，等价余弦），离线兜底使用内存余弦实现。二者通过 VectorStore Protocol 注入互换。

## Consequences
- 正面：零编译、无外部进程、索引可持久化到文件
- 负面：百万级以上规模需迁移（Spec 已列为 Out-of-Scope）

## Related ADRs
- ADR-002（嵌入走 ONNXRuntime 而非 transformers）
