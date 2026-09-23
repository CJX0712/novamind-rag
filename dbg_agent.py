import sys
sys.path.insert(0, ".")
from novamind.config import Settings
from novamind.core.llm import LlamaCppLLM
from novamind.core.agent import AGENT_SYSTEM, ReActAgent
from novamind.core.pipeline import build_offline_pipeline
from pathlib import Path

s = Settings()
llm = LlamaCppLLM(s.llm_model_path, n_ctx=s.llm_ctx, n_threads=s.llm_threads)
pipeline = build_offline_pipeline()
agent = ReActAgent(llm, pipeline)
prompt = agent._system_prompt()
out = llm.generate([
    {"role": "system", "content": prompt},
    {"role": "user", "content": "帮我算一下 12*(3+4) 等于多少"},
], max_tokens=256)
print("=== RAW OUTPUT ===")
print(out)
print("=== AGENT RESULT ===")
r = agent.run("帮我算一下 12*(3+4) 等于多少")
print("reply:", r.reply)
print("tools:", [(t.tool, t.input, t.output_preview) for t in r.tool_calls])
