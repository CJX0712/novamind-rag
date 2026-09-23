# ADR-002: 嵌入/重排走 ONNXRuntime + ModelScope，而非 transformers + HuggingFace

## Status: Accepted (2026-09-24)
## Author: 晨星

## Background
huggingface.co 在本环境超时不可达；transformers 依赖重且 sentence-transformers 未安装。onnxruntime 与 tokenizers 均已在 worldai venv 验证可用，ModelScope（www.modelscope.cn）网络可达且提供 BGE 系列模型的 ONNX 与 tokenizer.json 直链。

## Decision
- 嵌入与重排统一用 onnxruntime.InferenceSession + tokenizers.Tokenizer.from_file()
- ONNX 输入名运行时探测（不同导出带/不带 token_type_ids），不硬编码
- 模型经 scripts/download_models.py 从 ModelScope 直链下载到 models/（.gitignore 排除）
- 离线兜底：哈希 bigram 嵌入 + BM25 归一化重排，保证 CI/无网络环境全链路可验证

## Consequences
- 正面：免 torch 重型依赖、免编译、国内网络可下载
- 负面：需自行处理 pooling 与归一化（约 20 行代码，已在 core/embed.py 实现并测试）

## Related ADRs
- ADR-001（faiss-cpu）
- ADR-003（llama-cpp 本地 GGUF）
