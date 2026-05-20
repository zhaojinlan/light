"""实体邻居 — get_entity_neighbors()

获取实体的 N 跳邻居节点。

用法:
    python test/test_neighbors.py <实体名称> [深度]

示例:
    python test/test_neighbors.py "冒充公检法类诈骗"
    python test/test_neighbors.py "冒充公检法类诈骗" 2
"""
import os
import sys
import requests

SERVER = os.getenv("LIGHTSERVER_URL", "http://localhost:8001")

name = sys.argv[1] if len(sys.argv) > 1 else "冒充公检法类诈骗"
depth = int(sys.argv[2]) if len(sys.argv) > 2 else 1

print(f"=== 实体邻居: {name} (depth={depth}) ===\n")

# 通过 by-doc 接口获取邻居
# 先找到该实体所在文档，或者通过 kg/graph 搜索
r = requests.get(
    f"{SERVER}/api/v1/kg/graph",
    params={"entity_name": name, "max_depth": depth, "max_nodes": 500},
    timeout=15,
)
data = r.json()

nodes = data.get("nodes", [])
edges = data.get("edges", [])

# 构建节点 ID → 实体名称的映射
id_to_name = {}
for node in nodes:
    nid = str(node.get("id", ""))
    label = node.get("label", "")
    if nid and label:
        id_to_name[nid] = label

print(f"中心实体: {name}")
print(f"邻居节点数: {len(nodes) - 1}")  # 减去中心节点
print(f"连接边数: {len(edges)}")

print("\n邻居节点:")
for node in nodes:
    label = node.get("label", "")
    etype = node.get("entity_type", "")
    if label != name:  # 跳过中心节点
        print(f"  - {label[:80]} ({etype})")

print("\n连接关系:")
for edge in edges[:10]:
    src = id_to_name.get(str(edge.get("source", "")), edge.get("source", ""))
    tgt = id_to_name.get(str(edge.get("target", "")), edge.get("target", ""))
    print(f"  {src[:40]} -[{edge.get('relation_type', '')}]-> {tgt[:40]}")
