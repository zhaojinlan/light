"""防范措施查询 — query_prevention()

获取防范措施（PreventionMeasure）实体。

用法:
    python test/test_prevention.py
"""
import os
import requests

SERVER = os.getenv("LIGHTSERVER_URL", "http://localhost:8001")

print("=== 防范措施查询 ===\n")

# 无查询条件 — 返回所有防范措施
print("--- 所有防范措施 ---")
r = requests.post(
    f"{SERVER}/api/v1/retrieval/prevention",
    json={"top_k": 20},
    timeout=15,
)
data = r.json()
print(f"共 {data.get('count', 0)} 条\n")
for item in data.get("items", [])[:5]:
    print(f"  - {item.get('entity_name', '')}")
    print(f"    {item.get('description', '')[:100]}")
    print()

# 带查询条件 — 用语义搜索匹配相关防范措施
print("--- 查询 '虚假投资' ---")
r = requests.post(
    f"{SERVER}/api/v1/retrieval/prevention",
    json={"query": "虚假投资", "top_k": 20},
    timeout=15,
)
data = r.json()
print(f"共 {data.get('count', 0)} 条\n")
for item in data.get("items", [])[:5]:
    print(f"  - {item.get('entity_name', '')}")
    print(f"    {item.get('description', '')[:100]}")
    print()
