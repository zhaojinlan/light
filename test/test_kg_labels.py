"""实体标签列表 — get_graph_labels()

获取知识图谱中所有实体标签。

用法:
    python test/test_kg_labels.py
"""
import os
import requests

SERVER = os.getenv("LIGHTSERVER_URL", "http://localhost:8001")

print("=== 实体标签列表 ===\n")

r = requests.get(f"{SERVER}/api/v1/kg/labels", timeout=10)
data = r.json()
labels = data.get("labels", [])
print(f"实体总数: {len(labels)}\n")

# 按实体类型统计数量
entity_types = ["PreventionMeasure", "FraudScenario", "FraudMethod", "LawRegulation", "RelatedCase", "summary"]
print("按实体类型统计:")
for etype in entity_types:
    r2 = requests.get(
        f"{SERVER}/api/v1/retrieval/by-type/{etype}",
        params={"top_k": 500},
        timeout=10,
    )
    d = r2.json()
    count = d.get("count", 0)
    print(f"  {etype}: {count} 条")

print(f"\n实体名称示例（前 20）:")
for label in labels[:20]:
    print(f"  - {label}")
