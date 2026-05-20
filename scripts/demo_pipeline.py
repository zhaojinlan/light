#!/usr/bin/env python
"""端到端演示脚本：插入案例文档 -> 查询知识图谱。

用法:
    python scripts/demo_pipeline.py              # 插入所有案例并查询
    python scripts/demo_pipeline.py --limit 3    # 只插入前 3 个文档
    python scripts/demo_pipeline.py --query-only # 跳过插入，直接查询
"""

import json
import os
import sys
import time
from pathlib import Path

import requests

SERVER = os.getenv("LIGHTSERVER_URL", "http://localhost:8001")
CASE_DIR = Path(__file__).resolve().parent.parent.parent / "case" / "反诈案例"


# ============================================================
# Phase 1: Insert documents
# ============================================================

def collect_documents(limit=None):
    """收集所有案例文档。"""
    docs = []
    for cat_dir in sorted(CASE_DIR.iterdir()):
        if not cat_dir.is_dir():
            continue
        for f in sorted(cat_dir.glob("*.md")):
            text = f.read_text(encoding="utf-8")
            doc_id = f"{cat_dir.name}/{f.stem}"
            docs.append({"doc_id": doc_id, "text": text, "file_path": str(f)})
    if limit:
        docs = docs[:limit]
    return docs


def submit_doc(doc):
    """提交一篇文档到 LightRAG。"""
    r = requests.post(
        f"{SERVER}/api/v1/analyze",
        json={"text": doc["text"], "doc_id": doc["doc_id"], "file_path": doc["file_path"]},
    )
    return r.json().get("task_id")


def wait_for_tasks(task_ids, timeout=600):
    """轮询等待所有任务完成。"""
    start = time.time()
    while time.time() - start < timeout:
        done = 0
        for tid in task_ids:
            r = requests.get(f"{SERVER}/api/v1/status/{tid}")
            s = r.json().get("status")
            if s in ("completed", "failed"):
                done += 1
        if done == len(task_ids):
            break
        remaining = len(task_ids) - done
        print(f"  等待中... {done}/{len(task_ids)} 完成，{remaining} 处理中")
        time.sleep(5)

    # 打印汇总
    for tid in task_ids:
        r = requests.get(f"{SERVER}/api/v1/status/{tid}")
        s = r.json()
        status = s.get("status")
        result = s.get("result", {})
        if status == "completed":
            entities = result.get("entity_count", 0)
            relations = result.get("relation_count", 0)
            print(f"  [OK] {tid[:8]}... entities={entities}, relations={relations}")
        else:
            print(f"  [FAIL] {tid[:8]}... {s.get('error', 'unknown')}")


# ============================================================
# Phase 2: Demo queries
# ============================================================

def get(path, params=None):
    r = requests.get(f"{SERVER}{path}", params=params)
    try:
        return r.json()
    except Exception:
        return {"error": r.text[:300]}


def post(path, data):
    r = requests.post(f"{SERVER}{path}", json=data)
    try:
        return r.json()
    except Exception:
        return {"error": r.text[:300]}


def truncate(s, n=100):
    s = str(s)
    return s[:n] + "..." if len(s) > n else s


def demo_queries():
    """运行一组演示查询。"""
    print("\n" + "=" * 70)

    # 1. RAG 问答
    print("\n[1] RAG 问答 — '这起案件属于哪类诈骗？'")
    print("-" * 70)
    r = post("/api/v1/retrieval/query", {
        "question": "这起案件属于哪类诈骗？", "mode": "mix", "top_k": 5
    })
    print(f"  回答: {truncate(r.get('answer', ''), 200)}")
    print()

    # 2. 知识图谱
    print("[2] 知识图谱子图")
    print("-" * 70)
    r = get("/api/v1/kg/graph", {"entity_name": "*", "max_depth": 2, "max_nodes": 20})
    nodes = r.get("nodes", [])
    edges = r.get("edges", [])
    print(f"  节点: {r.get('total_nodes')}, 边: {r.get('total_edges')}")
    for n in nodes[:5]:
        print(f"    [{n.get('entity_type')}] {truncate(n.get('label'), 60)}")
    for e in edges[:5]:
        print(f"    {truncate(e.get('source'), 30)} --{e.get('relation_type')}--> {truncate(e.get('target'), 30)}")
    print()

    # 3. 图谱标签
    print("[3] 知识图谱标签")
    print("-" * 70)
    r = get("/api/v1/kg/labels")
    labels = r.get("labels", [])
    print(f"  标签: {labels}")
    print()

    # 4. BM25 混合检索
    print("[4] BM25 混合检索 — '虚假投资'")
    print("-" * 70)
    r = post("/api/v1/retrieval/bm25", {"question": "虚假投资", "top_k": 3})
    chunks = r.get("chunks", [])
    print(f"  找到 {len(chunks)} 个 chunk")
    for c in chunks[:3]:
        score = c.get("rrf_score", 0)
        print(f"    [{score:.4f}] {truncate(c.get('content', ''), 100)}")
    print()

    # 5. 按实体类型查询
    print("[5] 按实体类型查询")
    print("-" * 70)
    for etype in ["FraudScenario", "FraudMethod", "PreventionMeasure"]:
        r = get(f"/api/v1/retrieval/by-type/{etype}", {"top_k": 5})
        items = r.get("items", [])
        print(f"  {etype}: {r.get('count')} 条")
        for it in items[:3]:
            print(f"    - {truncate(it.get('entity_name', ''), 70)}")
    print()

    # 6. 相似案例搜索
    print("[6] 相似案例搜索 — '虚假投资理财诈骗'")
    print("-" * 70)
    r = get("/api/v1/kg/search", {"q": "虚假投资理财诈骗", "top_k": 3})
    print(f"  chunks: {len(r.get('chunks', []))}, summaries: {len(r.get('summaries', []))}, cases: {len(r.get('related_cases', []))}")
    for s in r.get("summaries", [])[:2]:
        print(f"    案件: {truncate(s.get('entity_name', ''), 80)}")
    for c in r.get("related_cases", [])[:2]:
        print(f"    案例: {truncate(c.get('entity_name', ''), 80)}")
    print()

    # 7. 防范措施
    print("[7] 防范措施检索 — '投资'")
    print("-" * 70)
    r = post("/api/v1/retrieval/prevention", {"query": "投资", "top_k": 5})
    items = r.get("items", [])
    print(f"  共 {r.get('count')} 条")
    for it in items[:5]:
        print(f"    - {truncate(it.get('entity_name', ''), 70)}")
        print(f"      {truncate(it.get('description', ''), 100)}")
    print()


# ============================================================
# Main
# ============================================================

def main():
    import argparse
    parser = argparse.ArgumentParser(description="LightRAG 端到端演示")
    parser.add_argument("--limit", type=int, default=None, help="只插入前 N 个文档")
    parser.add_argument("--query-only", action="store_true", help="跳过插入，直接查询")
    parser.add_argument("--server", default=None, help="LightRAG 服务地址")
    args = parser.parse_args()

    if args.server:
        global SERVER
        SERVER = args.server

    print(f"LightRAG 服务器: {SERVER}")

    # 检查服务是否可达
    try:
        r = requests.get(f"{SERVER}/health", timeout=5)
        print(f"健康检查: {r.json()}")
    except Exception as e:
        print(f"无法连接服务器: {e}")
        print("请先启动服务: cd lightserver && python server.py")
        sys.exit(1)

    if not args.query_only:
        print(f"\n数据源: {CASE_DIR}")
        docs = collect_documents(limit=args.limit)
        print(f"收集到 {len(docs)} 篇文档")

        if not docs:
            print("没有找到文档，退出。")
            sys.exit(0)

        print("\n正在插入文档...")
        task_ids = []
        for i, doc in enumerate(docs):
            print(f"  提交 {i+1}/{len(docs)}: {doc['doc_id']}")
            tid = submit_doc(doc)
            task_ids.append(tid)
            time.sleep(0.5)  # 避免请求过快

        print(f"\n等待 {len(task_ids)} 个任务完成...")
        wait_for_tasks(task_ids)

    demo_queries()

    print("\n演示完成。")


if __name__ == "__main__":
    main()
