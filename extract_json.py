"""从 test.md 抽取实体和关系，输出到当前目录的 JSON 文件（不存入 LightRAG 存储）。

用法:
    python extract_json.py
"""

import json
import re
import os
from typing import Any

from lightrag import LightRAG, QueryParam
from service.kg_config import create_lightrag_neo4j_qdrant


# ============================================================
# Prompt 模板（与 main.py 一致）
# ============================================================

ENTITY_EXTRACT_PROMPT = """你是铜合金领域知识图谱的实体抽取专家。

你的任务是：
从{text}中，
抽取材料科学相关实体。

必须严格按照 schema 输出。

# 实体类型定义

1. MaterialSystem
铜合金体系名称
例如：
- CuNiSi
- CuCrZr
- CuNiSn
- CuFeP

2. Phase
物相名称
例如：
- α相
- β相
- δ-Ni2Si
- Al2Cu

3. ConductivityMechanism
导电机制
例如：
- 析出净化
- 溶质散射
- 晶界散射

4. StrengtheningMechanism
强化机制
例如：
- 固溶强化
- 析出强化
- 位错强化
- 细晶强化

5. PrecipitationMechanism
析出/相变机制
例如：
- 共格析出
- 失配析出
- spinodal decomposition

6. Property
材料性能
例如：
- 抗拉强度
- 屈服强度
- 导电率
- 延伸率
- 硬度

7. ServicePerformance
服役性能
例如：
- 抗氧化性
- 耐磨性
- 抗热应力松弛

8. Application
应用场景
例如：
- 电子工业
- 航空航天
- 汽车连接器

9. FailureMode
失效模式
例如：
- 应力腐蚀开裂
- 晶界蠕变
- 脱锌

# 抽取规则

1. 不允许臆造实体
2. 必须来自原文
3. 保留原文 evidence
4. 输出 normalized_name
5. 同义词统一：
   - 时效强化 → 析出强化
   - 沉淀强化 → 析出强化
6. 若不确定：
   confidence < 0.6
7. 不输出解释
8. 输出合法 JSON

# 输出格式

{{
  "entities": [
    {{
      "entity_text": "",
      "entity_type": "",
      "normalized_name": "",
      "aliases": [],
      "evidence": "",
      "confidence": 0.0
    }}
  ]
}}"""


def load_test_text(path: str = "test.md") -> str:
    """从 test.md 读取并清理文本。"""
    with open(path, "r", encoding="utf-8") as f:
        content = f.read()

    # 移除 Markdown 标题标记
    content = re.sub(r"^#+\s+", "", content, flags=re.MULTILINE)
    # 移除 HTML 标签
    content = re.sub(r"<[^>]+>", "", content)
    # 移除图片链接
    content = re.sub(r"!\[.*?\]\(.*?\)", "", content)
    # 移除代码块
    content = re.sub(r"```.*?```", "", content, flags=re.DOTALL)
    # 移除行内代码
    content = re.sub(r"`[^`]+`", "", content)
    # 清理多余空行
    lines = [line.strip() for line in content.split("\n")]
    lines = [line for line in lines if line]
    return "\n\n".join(lines)


def call_llm(prompt: str, rag: LightRAG) -> str:
    """通过 LightRAG 调用 LLM（不经过 RAG 检索）。"""
    param = QueryParam(mode="bypass")
    return rag.query(prompt, param)


def parse_json_response(text: str) -> dict:
    """从 LLM 返回中提取 JSON。"""
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    json_match = re.search(r"```(?:json)?\s*([\s\S]*?)```", text)
    if json_match:
        try:
            return json.loads(json_match.group(1))
        except json.JSONDecodeError:
            pass
    brace_match = re.search(r"\{[\s\S]*\}", text)
    if brace_match:
        try:
            return json.loads(brace_match.group(0))
        except json.JSONDecodeError:
            pass
    return {}


def split_text(text: str, max_chars: int = 4000) -> list[str]:
    """按段落切分长文本，每段不超过 max_chars。"""
    paragraphs = [p for p in text.split("\n\n") if p.strip()]
    chunks = []
    current = ""
    for p in paragraphs:
        if len(current) + len(p) > max_chars and current:
            chunks.append(current)
            current = p
        else:
            current = current + "\n\n" + p if current else p
    if current:
        chunks.append(current)
    return chunks


def extract_entities(text: str, rag: LightRAG) -> list[dict[str, Any]]:
    """抽取实体。"""
    prompt = ENTITY_EXTRACT_PROMPT.format(text=text)
    response = call_llm(prompt, rag)
    data = parse_json_response(response)
    return data.get("entities", [])


def main():
    os.chdir(os.path.dirname(os.path.abspath(__file__)) or ".")

    print("加载 test.md ...")
    text = load_test_text("test.md")
    print(f"  文本长度: {len(text)} 字符")

    chunks = split_text(text, max_chars=4000)
    print(f"  切分为 {len(chunks)} 段")

    print("创建 LightRAG 实例（仅用于 LLM 调用，不存储）...")
    rag = create_lightrag_neo4j_qdrant(
        working_dir="./rag_storage_tmp",
        env_path="docker.env",
    )

    all_entities = []
    for i, chunk in enumerate(chunks, 1):
        print(f"  抽取第 {i}/{len(chunks)} 段实体 ...")
        entities = extract_entities(chunk, rag)
        print(f"    抽取到 {len(entities)} 个实体")
        all_entities.extend(entities)

    # 去重（按 normalized_name）
    seen = set()
    unique_entities = []
    for e in all_entities:
        name = e.get("normalized_name", "")
        if name and name not in seen:
            seen.add(name)
            unique_entities.append(e)

    print(f"\n  去重后共 {len(unique_entities)} 个实体")

    # 按类型分组
    by_type = {}
    for e in unique_entities:
        etype = e.get("entity_type", "Unknown")
        by_type.setdefault(etype, []).append(e)

    # 输出 JSON
    output = {
        "source_file": "test.md",
        "total_entities": len(unique_entities),
        "entities_by_type": {k: len(v) for k, v in by_type.items()},
        "entities": unique_entities,
    }

    output_path = "extracted_entities.json"
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=2)

    print(f"\n  已保存到: {output_path}")

    # 打印统计
    print("\n  实体类型统计:")
    for etype, items in sorted(by_type.items()):
        names = [e["normalized_name"] for e in items]
        print(f"    {etype} ({len(items)}): {', '.join(names[:5])}{'...' if len(names) > 5 else ''}")

    # 清理临时目录
    import shutil
    if os.path.exists("./rag_storage_tmp"):
        shutil.rmtree("./rag_storage_tmp")


if __name__ == "__main__":
    main()
