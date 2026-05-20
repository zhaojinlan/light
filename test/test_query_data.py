"""纯数据检索 — query_data()

返回结构化的检索结果，不经过 LLM 生成回答。

用法:
    python test/test_query_data.py
"""
import os
import requests

SERVER = os.getenv("LIGHTSERVER_URL", "http://localhost:8001")

print("=== 纯数据检索 ===\n")

r = requests.post(
    f"{SERVER}/api/v1/retrieval/query_data",
    json={
        "question": "冒充公检法",
        "mode": "mix",
        "top_k": 5,
    },
    timeout=30,
)
data = r.json()

print(f"实体数: {len(data.get('entities', []))}")
print(f"关系数: {len(data.get('relationships', []))}")
print(f"文本块数: {len(data.get('chunks', []))}")

if data.get("entities"):
    print("\n实体:")
    for e in data["entities"][:3]:
        print(f"  - {e.get('entity_name', '')} ({e.get('entity_type', '')})")
