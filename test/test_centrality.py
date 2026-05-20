"""实体中心性 — get_entity_centrality()

按连接数（degree）对实体降序排列，发现最核心的实体。

用法:
    python test/test_centrality.py [实体类型]

示例:
    python test/test_centrality.py FraudMethod
    python test/test_centrality.py
"""
import os
import sys
import requests

SERVER = os.getenv("LIGHTSERVER_URL", "http://localhost:8001")

etype = sys.argv[1] if len(sys.argv) > 1 else None
print(f"=== 实体中心性排名" + (f" ({etype})" if etype else " (全部)") + " ===\n")

# 获取所有节点
r = requests.get(
    f"{SERVER}/api/v1/kg/graph",
    params={"entity_name": "*", "max_depth": 1, "max_nodes": 1000},
    timeout=30,
)
data = r.json()

# 构建节点 ID → 实体名称的映射
id_to_name = {}
for node in data.get("nodes", []):
    nid = str(node.get("id", ""))
    label = node.get("label", "")
    if nid and label:
        id_to_name[nid] = label

# 计算每个实体的连接数（用实体名称而非数字 ID）
degree_map = {}
for edge in data.get("edges", []):
    src = str(edge.get("source", ""))
    tgt = str(edge.get("target", ""))
    src_name = id_to_name.get(src, src)
    tgt_name = id_to_name.get(tgt, tgt)
    if src_name:
        degree_map[src_name] = degree_map.get(src_name, 0) + 1
    if tgt_name:
        degree_map[tgt_name] = degree_map.get(tgt_name, 0) + 1

# 排序
sorted_entities = sorted(degree_map.items(), key=lambda x: x[1], reverse=True)

if etype:
    # 需要过滤类型，通过 by-type 获取该类型的实体
    r2 = requests.get(
        f"{SERVER}/api/v1/retrieval/by-type/{etype}",
        params={"top_k": 500},
        timeout=15,
    )
    type_names = {item["entity_name"] for item in r2.json().get("items", [])}
    sorted_entities = [(n, d) for n, d in sorted_entities if n in type_names]

print(f"{'排名':<4} {'实体名称':<50} {'连接数':>5}")
print("-" * 62)
for i, (name, degree) in enumerate(sorted_entities[:20], 1):
    print(f"{i:<4} {name[:50]:<50} {degree:>5}")
