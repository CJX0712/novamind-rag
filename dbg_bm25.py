import sys
sys.path.insert(0, ".")
from novamind.core.bm25 import BM25Index
from novamind.core.types import Chunk
from novamind.core.embed import tokenize

idx = BM25Index()
print("lib_available:", idx._lib_available)
chunks = [
    Chunk("c1", "d1", "向量检索使用嵌入模型计算语义相似度"),
    Chunk("c2", "d2", "BM25 基于词频和逆文档频率计算得分"),
]
idx.rebuild(chunks)
print("use_lib:", idx._use_lib)
q = tokenize("BM25 词频")
print("q tokens:", q)
print("c2 tokens:", tokenize(chunks[1].text))
if idx._bm25 is not None:
    print("scores:", idx._bm25.get_scores(q))
print("search:", idx.search("BM25 词频", 2))
