"""已处理文档列表 — get_processed_doc_ids()

获取所有已处理完成的文档 ID。

用法:
    python test/test_doc_ids.py
"""
import os
import sys
import requests

SERVER = os.getenv("LIGHTSERVER_URL", "http://localhost:8001")

print("=== 已处理文档列表 ===\n")

# 通过查询 chunk 的 full_doc_id 获取文档 ID
# summary 实体的 source_id 在 Neo4j 中未填充，改用 Qdrant chunks 中的 full_doc_id
r = requests.post(
    f"{SERVER}/api/v1/retrieval/bm25",
    json={"question": "", "top_k": 100},
    timeout=15,
)
data = r.json()

doc_ids = set()
for chunk in data.get("chunks", []):
    fid = chunk.get("payload", {}).get("full_doc_id", "")
    if fid:
        doc_ids.add(fid)

print(f"共 {len(doc_ids)} 个文档\n")
for doc_id in sorted(doc_ids):
    print(f"  - {doc_id}")
