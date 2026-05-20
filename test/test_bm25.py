"""BM25 混合检索 — query_with_bm25()

Dense 向量 + BM25 关键词混合检索 chunk。

用法:
    python test/test_bm25.py
"""
import os
import requests

SERVER = os.getenv("LIGHTSERVER_URL", "http://localhost:8001")

print("=== BM25 混合检索 ===\n")

r = requests.post(
    f"{SERVER}/api/v1/retrieval/bm25",
    json={"question": "冒充公检法", "top_k": 5},
    timeout=15,
)
data = r.json()

chunks = data.get("chunks", [])
print(f"匹配到 {len(chunks)} 个文本块\n")

for i, chunk in enumerate(chunks):
    content = chunk.get("content", "")[:150]
    score = chunk.get("rrf_score", 0)
    print(f"[{i+1}] score={score:.4f}")
    print(f"    {content}")
    print()
