"""案件摘要查询 — query_case_summary()

获取已入库案件的 summary 实体（案件简要描述）。

用法:
    python test/test_case_summary.py
"""
import os
import requests

SERVER = os.getenv("LIGHTSERVER_URL", "http://localhost:8001")

print("=== 案件摘要查询 ===\n")

r = requests.get(
    f"{SERVER}/api/v1/retrieval/by-type/summary",
    params={"top_k": 20},
    timeout=15,
)
data = r.json()
count = data.get("count", 0)
print(f"共 {count} 个案件摘要\n")

for item in data.get("items", []):
    name = item.get("entity_name", "")
    desc = item.get("description", "")
    source = item.get("source_id", "")
    # 兼容新旧格式：旧格式 entity_name=长摘要, source_id=空; 新格式 entity_name=doc_id, description=摘要
    if source:
        # 新格式：entity_name 是 doc_id，description 是摘要
        print(f"  [{name}] {desc[:120]}")
    else:
        # 旧格式：entity_name 直接是摘要文本
        print(f"  [] {name[:120]}")
    print()
