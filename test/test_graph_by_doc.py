"""按文档获取图谱 — get_kg_graph_by_doc()

按文档 ID 获取该文档对应的知识图谱子图。

用法:
    python test/test_graph_by_doc.py <doc_id>

示例:
    python test/test_graph_by_doc.py case_001
"""
import os
import sys
import requests

SERVER = os.getenv("LIGHTSERVER_URL", "http://localhost:8001")

doc_id = sys.argv[1] if len(sys.argv) > 1 else None
if not doc_id:
    print("用法: python test/test_graph_by_doc.py <doc_id>")
    print("\n可用文档 ID:")
    r = requests.get(f"{SERVER}/api/v1/kg/labels", timeout=10)
    sys.exit(1)

print(f"=== 文档图谱: {doc_id} ===\n")

r = requests.get(
    f"{SERVER}/api/v1/kg/graph/by-doc/{doc_id}",
    params={"depth": 2},
    timeout=15,
)
data = r.json()

center = data.get("center")
if not center:
    print(f"未找到文档 {doc_id} 的 summary 实体")
    sys.exit(0)

print(f"中心: {center[:100]}")
print(f"邻居数: {len(data.get('neighbors', []))}")
print(f"边数: {len(data.get('edges', []))}")

print("\n邻居节点:")
for nb in data.get("neighbors", [])[:10]:
    print(f"  - {nb.get('entity_name', '')[:80]} ({nb.get('entity_type', '')})")
