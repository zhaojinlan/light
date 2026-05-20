"""类似案件检索 — query_similar_cases()

输入案件特征描述，查找案情相似的案例。

用法:
    python test/test_similar_cases.py
"""
import os
import requests

SERVER = os.getenv("LIGHTSERVER_URL", "http://localhost:8001")

print("=== 类似案件检索 ===\n")

r = requests.get(
    f"{SERVER}/api/v1/kg/search",
    params={"q": "诈骗", "top_k": 5},
    timeout=30,
)
data = r.json()

chunks = data.get("chunks", [])
summaries = data.get("summaries", [])
related = data.get("related_cases", [])

print(f"匹配到 {len(chunks)} 个文本块")
print(f"匹配到 {len(summaries)} 个案件摘要")
print(f"匹配到 {len(related)} 个关联案例")

if summaries:
    print("\n案件摘要:")
    for s in summaries[:5]:
        print(f"  - {s.get('entity_name', '')[:100]}")

if related:
    print("\n关联案例:")
    for c in related[:5]:
        print(f"  - {c.get('entity_name', '')[:100]}")

# 显示 chunk 对应的文档来源
doc_ids = set()
for chunk in chunks:
    fid = chunk.get("payload", {}).get("full_doc_id", "")
    if fid:
        doc_ids.add(fid)
if doc_ids:
    print(f"\n涉及文档: {len(doc_ids)} 个")
    for d in sorted(doc_ids)[:5]:
        print(f"  - {d}")
