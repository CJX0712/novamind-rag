import sys, time
sys.path.insert(0, ".")
from pathlib import Path
import numpy as np
from novamind.core.embed import OnnxEmbedder

t0 = time.time()
emb = OnnxEmbedder(Path("models/bge-small-zh-v1.5"))
print("init:", round(time.time() - t0, 2), "s, dim =", emb.dim)
t0 = time.time()
v = emb.embed([
    "机器学习是人工智能的重要分支",
    "人工智能包含机器学习等领域",
    "今天天气晴朗适合出门散步",
])
print("embed 3 texts:", round(time.time() - t0, 2), "s")
print("shape:", v.shape, "norm:", [round(float(np.linalg.norm(x)), 4) for x in v])
print("sim(related):", round(float(v[0] @ v[1]), 4))
print("sim(unrelated):", round(float(v[0] @ v[2]), 4))
assert float(v[0] @ v[1]) > float(v[0] @ v[2]), "语义相似度关系错误"
print("EMBED OK")
