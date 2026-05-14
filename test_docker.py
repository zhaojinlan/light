"""LightRAG + Neo4j + Qdrant 测试验证脚本

验证步骤：
1. 检查 Docker 服务（Neo4j、Qdrant）是否可达
2. 创建 LightRAG 实例（Neo4j + Qdrant 后端）
3. 使用自定义实体 schema 插入测试文本
4. 查询知识图谱验证结果
"""

import sys
import json
import time
from datetime import datetime


def print_header(text: str):
    """打印分隔标题。"""
    print(f"\n{'=' * 60}")
    print(f"  {text}")
    print(f"{'=' * 60}\n")


def print_result(label: str, data):
    """格式化打印结果。"""
    print(f"  [{label}]")
    if isinstance(data, dict):
        print(json.dumps(data, ensure_ascii=False, indent=4))
    elif isinstance(data, list):
        for item in data:
            print(f"    - {json.dumps(item, ensure_ascii=False)}")
    else:
        print(f"    {data}")
    print()


# ============================================================
# 测试 1：检查 Docker 服务连通性
# ============================================================

def test_docker_connectivity():
    """测试 Neo4j 和 Qdrant 的连接。"""
    print_header("步骤 1: 检查 Docker 服务连通性")

    # 测试 Neo4j 连接
    print("  [Neo4j] 尝试连接 bolt://localhost:7687 ...")
    try:
        from neo4j import GraphDatabase

        driver = GraphDatabase.driver(
            "bolt://localhost:7687",
            auth=("neo4j", "LightRAG2026!"),
        )
        with driver.session() as session:
            result = session.run("RETURN 1 AS num")
            record = result.single()
            if record and record["num"] == 1:
                print("  [Neo4j] 连接成功！")

                # 检查 APOC 插件（core 版本没有 apoc.version()，用 apoc.util.validate 代替）
                try:
                    session.run("CALL apoc.util.validate(false, 'test', [])")
                    print("  [APOC] 已启用（apoc-core）")
                except Exception as e:
                    print(f"  [APOC] 未启用或不可用: {e}")

        driver.close()
    except Exception as e:
        print(f"  [Neo4j] 连接失败: {e}")
        print("  提示: 请先运行 docker-compose up -d 启动服务")
        return False

    # 测试 Qdrant 连接
    print("  [Qdrant] 尝试连接 http://localhost:6333 ...")
    try:
        import requests

        resp = requests.get("http://localhost:6333/healthz", timeout=5)
        if resp.status_code == 200:
            print("  [Qdrant] 连接成功！")
        else:
            print(f"  [Qdrant] 响应异常: {resp.status_code}")
            return False
    except Exception as e:
        print(f"  [Qdrant] 连接失败: {e}")
        print("  提示: 请先运行 docker-compose up -d 启动服务")
        return False

    return True


# ============================================================
# 测试 2：创建 LightRAG 实例
# ============================================================

def test_lightrag_creation():
    """测试创建 Neo4j + Qdrant 后端的 LightRAG 实例。"""
    print_header("步骤 2: 创建 LightRAG 实例")

    from service.kg_config import create_lightrag_neo4j_qdrant

    try:
        rag = create_lightrag_neo4j_qdrant(
            working_dir="./rag_storage_test",
            env_path="docker.env",
        )
        print("  LightRAG 实例创建成功")
        print(f"  图存储: {rag.graph_storage}")
        print(f"  向量存储: {rag.vector_storage}")
        print(f"  工作目录: {rag.working_dir}")
        print(f"  模型: {rag.llm_model_name}")
        return rag
    except Exception as e:
        print(f"  LightRAG 实例创建失败: {e}")
        import traceback
        traceback.print_exc()
        return None


# ============================================================
# 测试 3：自定义实体抽取与插入
# ============================================================

def load_test_text(path: str = "test.md") -> str:
    """从 test.md 文件读取内容作为测试文本。

    会自动剥离 Markdown 标题标记、表格 HTML、图片链接等格式噪音，
    保留纯文本内容供实体抽取使用。

    Args:
        path: test.md 文件路径

    Returns:
        清理后的纯文本内容
    """
    import re

    with open(path, "r", encoding="utf-8") as f:
        content = f.read()

    # 移除 Markdown 标题标记
    content = re.sub(r"^#+\s+", "", content, flags=re.MULTILINE)

    # 移除 HTML 标签（保留标签内的文字内容）
    content = re.sub(r"<table>.*?</table>", " [TABLE] ", content, flags=re.DOTALL | re.IGNORECASE)
    content = re.sub(r"<details>.*?</details>", "", content, flags=re.DOTALL | re.IGNORECASE)
    content = re.sub(r"<[^>]+>", "", content)

    # 移除图片链接
    content = re.sub(r"!\[.*?\]\(.*?\)", "", content)

    # 移除 Mermaid 代码块
    content = re.sub(r"```mermaid.*?```", "", content, flags=re.DOTALL)

    # 移除行内代码
    content = re.sub(r"`[^`]+`", "", content)

    # 移除多余空行，保留段落结构
    lines = [line.strip() for line in content.split("\n")]
    lines = [line for line in lines if line]
    content = "\n\n".join(line for line in lines if line)

    return content


# 测试文本从 test.md 加载
TEST_TEXT = load_test_text()


def test_custom_entity_insert(rag):
    """测试自定义实体抽取和插入。"""
    print_header("步骤 3: 自定义实体抽取与插入")

    print("  测试文本已加载")
    print(f"  来源: test.md")
    print(f"  长度: {len(TEST_TEXT)} 字符")

    from service.custom_entity_service import CustomEntityService

    service = CustomEntityService(rag=rag)

    try:
        result = service.insert_with_custom_schema(
            text=TEST_TEXT,
            doc_id="test_material_alloys_001",
            file_path="test.md",
        )

        print("  实体抽取结果:")
        for entity in result["entities"]:
            name = entity.get("normalized_name") or entity.get("entity_text", "?")
            etype = entity.get("entity_type", "?")
            print(f"    - {name} ({etype})")

        print("\n  关系抽取结果:")
        for rel in result["relationships"]:
            print(f"    - {rel['src_entity']} --[{rel['relation_type']}]-> {rel['tgt_entity']}")

        print(f"\n  文档 ID: {result['doc_id']}")
        print("  插入成功！")
        return True

    except Exception as e:
        print(f"  插入失败: {e}")
        import traceback
        traceback.print_exc()
        return False


# ============================================================
# 测试 4：知识图谱查询验证
# ============================================================

def test_query_verification(rag):
    """测试知识图谱查询。"""
    print_header("步骤 4: 知识图谱查询验证")

    # 查询图谱中的标签（使用已修补的同步方法，复用持久事件循环）
    try:
        labels = rag.get_graph_labels()
        print(f"  知识图谱中的实体标签 ({len(labels)} 个):")
        for label in labels[:20]:
            print(f"    - {label}")
    except Exception as e:
        print(f"  查询标签失败: {e}")

    # 获取子图（使用已修补的同步方法）
    try:
        graph = rag.get_knowledge_graph("*", max_depth=2, max_nodes=20)
        print(f"\n  知识图谱子图:")
        print(f"    节点数: {len(graph.nodes)}")
        print(f"    边数: {len(graph.edges)}")
        print(f"    是否截断: {graph.is_truncated}")

        print("\n    节点:")
        for node in graph.nodes[:10]:
            print(f"      - {node.labels}: {dict(list(node.properties.items())[:3])}")

        print("\n    边:")
        for edge in graph.edges[:10]:
            print(f"      - {edge.source} -> {edge.target}: {edge.type}")
    except Exception as e:
        print(f"  子图查询失败: {e}")


# ============================================================
# 主函数
# ============================================================

def main():
    """运行所有测试。"""
    print_header("LightRAG + Neo4j + Qdrant 测试")
    print(f"  时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    # 测试 1：Docker 连接
    if not test_docker_connectivity():
        print("\n  跳过后续测试，请先启动 Docker 服务:")
        print("    docker-compose up -d")
        return

    # 测试 2：创建 LightRAG 实例
    rag = test_lightrag_creation()
    if not rag:
        print("\n  LightRAG 实例创建失败，跳过后续测试")
        return

    # 测试 3：自定义实体插入
    if not test_custom_entity_insert(rag):
        print("\n  实体插入失败，跳过查询测试")
        return

    # 测试 4：查询验证
    test_query_verification(rag)

    print_header("测试完成")
    print("  清理测试数据...")

    # 关闭持久事件循环
    if hasattr(rag, "_persistent_loop"):
        rag._persistent_loop.close()
        print("  事件循环已关闭")

    import shutil
    import os

    try:
        if os.path.exists("./rag_storage_test"):
            shutil.rmtree("./rag_storage_test")
            print("  本地缓存已清理")
    except Exception as e:
        print(f"  清理失败: {e}")

    print("\n  提示: Neo4j 和 Qdrant 中的数据需要通过其管理界面手动清理")


if __name__ == "__main__":
    main()
