"""知识图谱子图 — get_knowledge_graph()

获取知识图谱的子图（nodes/edges）。

用法:
    python test/test_kg_graph.py
"""
import os
import requests

SERVER = os.getenv("LIGHTSERVER_URL", "http://localhost:8001")

print("=== 知识图谱子图 ===\n")

# 获取全图谱（depth=1）
r = requests.get(
    f"{SERVER}/api/v1/kg/graph",
    params={"entity_name": "*", "max_depth": 1, "max_nodes": 20},
    timeout=15,
)
data = r.json()
print(f"节点数: {data.get('total_nodes', 0)}")
print(f"边数: {data.get('total_edges', 0)}")

# 构建节点 ID → 实体名称的映射
id_to_name = {}
for node in data.get("nodes", []):
    nid = str(node.get("id", ""))
    label = node.get("label", "")
    if nid and label:
        id_to_name[nid] = label

print("\n节点示例:")
for node in data.get("nodes", [])[:5]:
    print(f"  - {node.get('label', '')[:80]} ({node.get('entity_type', '')})")

print("\n边示例:")
for edge in data.get("edges", [])[:5]:
    src = id_to_name.get(str(edge.get("source", "")), edge.get("source", ""))
    tgt = id_to_name.get(str(edge.get("target", "")), edge.get("target", ""))
    print(f"  - {src[:40]} -[{edge.get('relation_type', '')}]-> {tgt[:40]}")
