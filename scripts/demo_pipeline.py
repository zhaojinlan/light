#!/usr/bin/env python3
"""端到端批量上传脚本：将反诈案例全部通过主应用 API 上传。

与前端上传单个 .md 文件的流程完全一致：
  1. 登录获取 JWT token
  2. 通过 POST /api/v1/documents/upload 上传每篇文档（multipart/form-data）
  3. 后台 worker 自动从 MinIO 读取内容并提交到 lightserver 解析
  4. 轮询文档列表直到全部处理完成

用法:
    python scripts/demo_pipeline.py                  # 上传所有案例
    python scripts/demo_pipeline.py --limit 3        # 只上传前 3 篇
    python scripts/demo_pipeline.py --status-only    # 跳过上传，仅查看处理状态
    python scripts/demo_pipeline.py --dry-run        # 仅预览，不上传
"""

import os
import sys
import time
from pathlib import Path

import requests

# ============================================================
# 配置（可通过环境变量或命令行参数覆盖）
# ============================================================

APP_SERVER = os.getenv("APP_SERVER", "http://localhost:8000")
LIGHTSERVER_SERVER = os.getenv("LIGHTSERVER_SERVER", "http://localhost:8001")
ADMIN_USERNAME = os.getenv("ADMIN_USERNAME", "admin")
ADMIN_PASSWORD = os.getenv("ADMIN_PASSWORD", "admin123")
CASE_DIR = Path(__file__).resolve().parent.parent.parent / "case" / "反诈案例"


# ============================================================
# Phase 1: 认证
# ============================================================

def login(username: str, password: str) -> str:
    """登录获取 JWT token。"""
    r = requests.post(
        f"{APP_SERVER}/api/v1/auth/login",
        json={"username": username, "password": password},
        timeout=10,
    )
    if r.status_code != 200:
        detail = r.json().get("detail", "unknown")
        print(f"登录失败: {detail}")
        sys.exit(1)
    return r.json()["access_token"]


# ============================================================
# Phase 2: 收集文档
# ============================================================

def collect_documents(limit=None):
    """遍历反诈案例目录，收集所有 .md 文件。"""
    docs = []
    for cat_dir in sorted(CASE_DIR.iterdir()):
        if not cat_dir.is_dir():
            continue
        for f in sorted(cat_dir.glob("*.md")):
            docs.append({
                "file_path": f,
                "category": cat_dir.name,
                "filename": f.name,
            })
    if limit:
        docs = docs[:limit]
    return docs


# ============================================================
# Phase 3: 上传文档
# ============================================================

def upload_document(token: str, file_path: Path) -> dict:
    """通过主应用 API 上传一篇 .md 文档。

    与前端 KnowledgeBase.vue 的 handleUpload 调用方式完全一致。
    """
    headers = {"Authorization": f"Bearer {token}"}
    with open(file_path, "rb") as f:
        r = requests.post(
            f"{APP_SERVER}/api/v1/documents/upload",
            headers=headers,
            files={"file": (file_path.name, f, "text/markdown")},
            timeout=60,
        )
    if r.status_code == 409:
        detail = r.json().get("detail", "文件已存在")
        return {"status": "skipped", "reason": detail}
    if r.status_code != 200:
        detail = r.json().get("detail", "上传失败")
        return {"status": "error", "reason": detail}
    data = r.json()
    return {
        "status": "uploaded",
        "doc_id": data["id"],
        "filename": data["filename"],
    }


def list_documents(token: str) -> list[dict]:
    """获取文档列表及处理状态。"""
    headers = {"Authorization": f"Bearer {token}"}
    r = requests.get(
        f"{APP_SERVER}/api/v1/documents",
        headers=headers,
        params={"status": "all", "limit": 100},
        timeout=10,
    )
    if r.status_code != 200:
        print(f"获取文档列表失败: {r.text[:200]}")
        return []
    return r.json().get("documents", [])


# ============================================================
# Phase 4: 等待处理完成
# ============================================================

def wait_for_completion(token: str, timeout=21600, poll_interval=30):
    """轮询文档列表，直到所有文档处理完成或超时。"""
    start = time.time()
    last_status = {}
    pending = []  # 初始化，避免超时消息引用失败

    while time.time() - start < timeout:
        docs = list_documents(token)
        if not docs:
            print("  暂无文档记录")
            time.sleep(poll_interval)
            continue

        pending = [d for d in docs if d["status"] in ("pending", "processing")]
        completed = [d for d in docs if d["status"] == "completed"]
        failed = [d for d in docs if d["status"] == "failed"]

        # 状态变化时打印
        current_status = {d["original_filename"]: d["status"] for d in docs}
        if current_status != last_status:
            elapsed = int(time.time() - start)
            print(f"  [{elapsed}s] 已完成: {len(completed)}, 处理中: {len(pending)}, 失败: {len(failed)}, 总计: {len(docs)}")
            for d in docs:
                name = d["original_filename"]
                status = d["status"]
                meta = ""
                if status == "completed":
                    ec = d.get("entity_count", 0) or 0
                    rc = d.get("relation_count", 0) or 0
                    meta = f"  {ec}实体/{rc}关系"
                elif status == "failed":
                    meta = f"  {d.get('error_message', '未知错误')}"
                print(f"    [{status:>10}] {name}{meta}")
            print()

        if not pending:
            print(f"  全部文档处理完成（完成: {len(completed)}, 失败: {len(failed)}）")
            return True

        last_status = current_status
        time.sleep(poll_interval)

    print(f"  超时（{timeout}s），仍有 {len(pending)} 篇文档在处理中")
    return False


# ============================================================
# Phase 5: 演示查询（可选）
# ============================================================

def demo_queries(token: str):
    """运行一组演示查询验证知识图谱数据。"""
    headers = {"Authorization": f"Bearer {token}"}

    print("\n" + "=" * 70)
    print("验证知识图谱数据")
    print("=" * 70)

    # 1. 查看知识图谱子图
    print("\n[1] 知识图谱子图")
    print("-" * 70)
    r = requests.get(
        f"{APP_SERVER}/api/v1/kg/graph",
        headers=headers,
        params={"entity_name": "*", "max_depth": 2, "max_nodes": 10},
        timeout=60,
    )
    if r.status_code != 200:
        print(f"  请求失败: {r.text[:200]}")
    else:
        data = r.json()
        nodes = data.get("nodes", [])
        edges = data.get("edges", [])
        print(f"  节点: {data.get('total_nodes', 0)}, 边: {data.get('total_edges', 0)}")
        for n in nodes[:5]:
            print(f"    [{n.get('entity_type')}] {n.get('label', '')[:80]}")
        for e in edges[:5]:
            print(f"    {e.get('source', '')[:30]} --{e.get('relation_type')}--> {e.get('target', '')[:30]}")

    # 2. 实体类型统计
    print("\n[2] 按实体类型统计")
    print("-" * 70)
    for etype in ["FraudScenario", "FraudMethod", "FraudFeature",
                   "PreventionMeasure", "LawRegulation", "RelatedCase"]:
        r = requests.get(
            f"{LIGHTSERVER_SERVER}/api/v1/retrieval/by-type/{etype}",
            params={"top_k": 100},
            timeout=30,
        )
        if r.status_code != 200:
            print(f"  {etype}: 请求失败 ({r.status_code})")
        else:
            data = r.json()
            print(f"  {etype}: {data.get('count', 0)} 条")

    # 3. 相似案例搜索
    print("\n[3] 相似案例搜索 — '虚假投资理财'")
    print("-" * 70)
    r = requests.get(
        f"{APP_SERVER}/api/v1/kg/search",
        headers=headers,
        params={"q": "虚假投资理财", "top_k": 3},
        timeout=60,
    )
    if r.status_code != 200:
        print(f"  请求失败: {r.text[:200]}")
    else:
        data = r.json()
        for s in data.get("summaries", [])[:3]:
            name = s.get("entity_name", "")
            desc = s.get("description", "")
            print(f"    案件: {name[:70]}")
            print(f"      {desc[:100]}")


# ============================================================
# Main
# ============================================================

def main():
    import argparse
    parser = argparse.ArgumentParser(description="批量上传反诈案例到知识图谱")
    parser.add_argument("--limit", type=int, default=None, help="只上传前 N 篇文档")
    parser.add_argument("--status-only", action="store_true", help="跳过上传，仅查看处理状态")
    parser.add_argument("--dry-run", action="store_true", help="仅预览文档列表，不上传")
    parser.add_argument("--timeout", type=int, default=21600, help="等待处理完成的超时秒数（默认 21600=6 小时）")
    parser.add_argument("--poll-interval", type=int, default=30, help="轮询间隔秒数（默认 30）")
    parser.add_argument("--no-demo", action="store_true", help="跳过演示查询")
    parser.add_argument("--server", default=None, help="主应用服务器地址")
    parser.add_argument("--username", default=None, help="管理员用户名")
    parser.add_argument("--password", default=None, help="管理员密码")
    args = parser.parse_args()

    if args.server:
        global APP_SERVER
        APP_SERVER = args.server
    if args.username:
        global ADMIN_USERNAME
        ADMIN_USERNAME = args.username
    if args.password:
        global ADMIN_PASSWORD
        ADMIN_PASSWORD = args.password

    print("=" * 70)
    print("反诈案例批量上传工具")
    print("=" * 70)
    print(f"  服务器:     {APP_SERVER}")
    print(f"  案例目录:   {CASE_DIR}")
    print()

    # 检查服务是否可达
    try:
        r = requests.get(f"{APP_SERVER}/health", timeout=5)
        print(f"健康检查: {r.json()}")
    except Exception as e:
        print(f"无法连接服务器: {e}")
        print(f"请确保主应用已启动（默认 {APP_SERVER}）")
        sys.exit(1)

    # 收集文档
    docs = collect_documents(limit=args.limit)
    print(f"收集到 {len(docs)} 篇文档：")
    for d in docs:
        print(f"  [{d['category']}] {d['filename']}")
    print()

    if args.dry_run:
        print("预览模式，未执行上传。")
        return

    # 登录
    print("正在登录...")
    token = login(ADMIN_USERNAME, ADMIN_PASSWORD)
    print("登录成功。\n")

    # 上传
    if not args.status_only:
        print("=" * 70)
        print("上传文档")
        print("=" * 70)

        # 预检：查询已存在的文档，跳过无需上传的文件
        existing_docs = list_documents(token)
        existing_names = {d["original_filename"] for d in existing_docs}
        if existing_names:
            print(f"已存在 {len(existing_names)} 篇文档，将跳过重复上传\n")

        uploaded = 0
        skipped = 0
        errors = 0

        for i, d in enumerate(docs):
            if d["filename"] in existing_names:
                print(f"  [{i+1}/{len(docs)}] 跳过: {d['filename']} (已存在)")
                skipped += 1
                continue

            print(f"  [{i+1}/{len(docs)}] 上传: {d['filename']}", end=" ... ")
            result = upload_document(token, d["file_path"])

            if result["status"] == "uploaded":
                print(f"OK (id={result['doc_id'][:8]}...)")
                uploaded += 1
            elif result["status"] == "skipped":
                print(f"跳过 ({result['reason']})")
                skipped += 1
            else:
                print(f"失败 ({result['reason']})")
                errors += 1

            time.sleep(0.3)  # 避免请求过快

        print(f"\n上传完成: {uploaded} 篇成功, {skipped} 篇跳过, {errors} 篇失败\n")

    # 等待处理完成
    print("=" * 70)
    print("等待后台处理（lightserver 解析 -> Neo4j/Qdrant 写入）")
    print("=" * 70)
    wait_for_completion(token, timeout=args.timeout, poll_interval=args.poll_interval)

    # 演示查询验证
    if not args.no_demo:
        demo_queries(token)

    print("\n全部完成。")


if __name__ == "__main__":
    main()
