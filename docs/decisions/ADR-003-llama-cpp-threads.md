# ADR-003: 本地 LLM 用 llama-cpp-python（GGUF），线程锁 4

## Status: Accepted (2026-09-24)
## Author: 晨星

## Background
Ollama 在目标机器未运行；llama-cpp-python 已在 worldai venv 验证可 import。llama.cpp 系默认线程数 = cpu_count-1，在 CPU 推理小量化模型时瓶颈在内存带宽而非算力，过多线程导致争抢，实测吞吐可慢 4 倍。

## Decision
- LLMProvider 生产实现为 llama-cpp-python 加载 GGUF（ModelScope 下载，默认 Qwen2.5-0.5B-Instruct Q4_K_M 级别小模型）
- n_threads 锁定 min(4, cpu_count)，可通过 NOVAMIND_LLM_THREADS 覆盖
- 离线兜底为 MockLLM（模板化引用生成），CI 不依赖任何模型

## Consequences
- 正面：零额外服务、CPU 可跑、速度可控
- 负面：生成质量受小模型限制；换更大模型只需改配置

## Related ADRs
- ADR-002（模型下载通道）
