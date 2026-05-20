"""按实体类型查询 — query_by_entity_type(type)

按实体类型列出实体，如"列出所有诈骗手法"。

用法:
    python test/test_by_type.py
"""
import os
import requests

SERVER = os.getenv("LIGHTSERVER_URL", "http://localhost:8001")

print("=== 按实体类型查询 ===\n")

for etype in ["PreventionMeasure", "FraudScenario", "FraudMethod", "LawRegulation", "RelatedCase", "summary"]:
    r = requests.get(
        f"{SERVER}/api/v1/retrieval/by-type/{etype}",
        params={"top_k": 5},
        timeout=15,
    )
    data = r.json()
    count = data.get("count", 0)
    print(f"{etype}: {count} 条")
    for item in data.get("items", [])[:3]:
        print(f"  - {item.get('entity_name', '')[:80]}")
    print()
