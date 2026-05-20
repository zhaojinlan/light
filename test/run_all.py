"""一键运行所有检索方法测试。

用法:
    python test/run_all.py
"""
import os
import subprocess
import sys

PYTHON = sys.executable
TEST_DIR = os.path.dirname(os.path.abspath(__file__))

scripts = [
    ("健康检查", "test_health.py"),
    ("已处理文档", "test_doc_ids.py"),
    ("按实体类型查询", "test_by_type.py"),
    ("案件摘要", "test_case_summary.py"),
    ("知识图谱子图", "test_kg_graph.py"),
    ("实体标签", "test_kg_labels.py"),
    ("类似案件检索", "test_similar_cases.py"),
    ("防范措施查询", "test_prevention.py"),
    ("BM25 混合检索", "test_bm25.py"),
    ("实体详情", "test_entity_info.py"),
    ("实体中心性", "test_centrality.py"),
    ("实体邻居", "test_neighbors.py"),
    ("RAG 问答", "test_query.py"),
    ("纯数据检索", "test_query_data.py"),
]

ok = 0
fail = 0

for label, filename in scripts:
    print(f"\n{'='*60}")
    print(f">>> {label}")
    print(f"{'='*60}\n")
    result = subprocess.run(
        [PYTHON, os.path.join(TEST_DIR, filename)],
        timeout=60,
    )
    if result.returncode == 0:
        ok += 1
    else:
        fail += 1

print(f"\n{'='*60}")
print(f"总计: {ok} 通过, {fail} 失败")
print(f"{'='*60}")
