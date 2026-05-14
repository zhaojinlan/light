"""LLM 实体与关系抽取测试脚本

读取 test.md 中的文本，使用 service/custom_entity_service.py 中定义的
prompt 和 schema 进行实体和关系抽取，输出到 test_extract/ 文件夹。

不存储任何数据到 LightRAG、Neo4j 或 Qdrant，仅调用 LLM 并展示结果。

用法:
    "C:/Users/Administrator/.conda/envs/ms/python.exe" test_extract/extract_only.py
"""

import json
import re
import os
import sys

# 添加父目录到路径，以便导入 service 模块
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from lightrag import LightRAG, QueryParam
from lightrag.utils import EmbeddingFunc
from lightrag.llm.openai import openai_complete, openai_embed
from functools import partial


# ============================================================
# Prompt 模板（从 service/custom_entity_service.py 复制）
# ============================================================

ENTITY_EXTRACT_PROMPT = """你是铜合金领域知识图谱的实体抽取专家。

你的任务是：
从{text}中，
抽取材料科学相关实体。

必须严格按照 schema 输出。

# 实体类型定义

1. MaterialSystem — 铜合金体系名称
   例如：CuNiSi, CuCrZr, CuNiSn, CuFeP
   建议属性：化学成分范围, 颜色, 典型用途, 特点

2. Phase — 物相名称
   例如：α相, β相, δ-Ni2Si, Al2Cu
   建议属性：晶格结构, 化学成分, 特性, 形成条件

3. ConductivityMechanism — 导电机制
   例如：析出净化, 溶质散射, 晶界散射
   建议属性：作用原理, 影响因素

4. StrengtheningMechanism — 强化机制
   例如：固溶强化, 析出强化, 位错强化, 细晶强化
   建议属性：作用原理, 适用条件

5. PrecipitationMechanism — 析出/相变机制
   例如：共格析出, 失配析出, spinodal decomposition
   建议属性：析出相, 共格关系, 温度范围

6. Property — 材料性能
   例如：抗拉强度, 屈服强度, 导电率, 延伸率, 硬度
   建议属性：单位, 测量条件, 数值范围

7. ServicePerformance — 服役性能
   例如：抗氧化性, 耐磨性, 抗热应力松弛
   建议属性：测试条件, 评价标准

8. Application — 应用场景
   例如：电子工业, 航空航天, 汽车连接器
   建议属性：适用合金类型, 优势

9. FailureMode — 失效模式
   例如：应力腐蚀开裂, 晶界蠕变, 脱锌
   建议属性：诱发条件, 表现特征

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
      "definition": "",
      "attributes": [
        {{"key": "", "value": ""}}
      ],
      "evidence": "",
      "confidence": 0.0
    }}
  ]
}}

说明：
- definition: 用一句话概括该实体是什么（基于原文）
- attributes: 从原文中提取该实体类型的关键属性，按建议属性填写
  如果原文中不存在某属性，跳过该属性即可，不强制填写全部
- evidence: 原文中支撑该实体的具体句子
- aliases: 原文中出现的同义词/别名
"""


RELATION_EXTRACT_PROMPT = """你是铜合金领域知识图谱的关系抽取专家。

你的任务是：
从{text}中，基于已抽取的实体列表，抽取实体之间的关系。

# 关系类型定义

1. has_phase（体系→物相）
   铜合金体系包含某个物相
   例：CuNiSi has_phase α相

2. strengthened_by（体系→强化机制）
   铜合金体系通过某种机制强化
   例：CuNiSi strengthened_by 析出强化

3. has_property（体系/物相→性能）
   材料具有某种性能
   例：CuNiSi has_property 高导电率

4. used_in（体系→应用场景）
   铜合金应用于某个场景
   例：CuNiSi used_in 电子工业

5. failure_mode_of（失效模式→体系）
   某种失效模式出现在某体系中
   例：脱锌 failure_mode_of CuNiSi

6. mechanism_of（机制→体系）
   某种机制在某体系中起作用
   例：析出净化 mechanism_of CuNiSi

7. contains（体系→体系，子体系关系）
   一个体系包含另一个体系
   例：铜合金 contains CuNiSi

# 抽取规则

1. 关系必须来自原文或可以合理推断
2. src_entity 和 tgt_entity 必须在已抽取的实体列表中存在
3. 使用实体的 normalized_name
4. 不输出解释
5. 输出合法 JSON

# 已抽取实体列表

{entities}

# 输出格式

{{
  "relationships": [
    {{
      "src_entity": "",
      "tgt_entity": "",
      "relation_type": "",
      "evidence": ""
    }}
  ]
}}"""


# ============================================================
# 工具函数
# ============================================================

def load_test_text(path: str = "test.md") -> str:
    """读取 test.md 并清理 Markdown 格式噪音。"""
    with open(path, "r", encoding="utf-8") as f:
        content = f.read()

    # 移除 Markdown 标题标记
    content = re.sub(r"^#+\s+", "", content, flags=re.MULTILINE)
    # 移除 HTML 标签
    content = re.sub(r"<table>.*?</table>", " [TABLE] ", content, flags=re.DOTALL | re.IGNORECASE)
    content = re.sub(r"<details>.*?</details>", "", content, flags=re.DOTALL | re.IGNORECASE)
    content = re.sub(r"<[^>]+>", "", content)
    # 移除图片链接
    content = re.sub(r"!\[.*?\]\(.*?\)", "", content)
    # 移除 Mermaid 代码块
    content = re.sub(r"```mermaid.*?```", "", content, flags=re.DOTALL)
    # 移除行内代码
    content = re.sub(r"`[^`]+`", "", content)
    # 清理多余空行
    lines = [line.strip() for line in content.split("\n")]
    lines = [line for line in lines if line]
    return "\n\n".join(line for line in lines if line)


def split_text(text: str, max_chars: int = 4000) -> list[str]:
    """按段落切分长文本。"""
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


def create_minimal_rag() -> LightRAG:
    """创建最小 LightRAG 实例（仅用于 LLM 调用，不存储数据）。

    使用默认 JSON 文件存储，不需要 Neo4j/Qdrant。
    """
    from dotenv import load_dotenv

    # 从项目根目录加载 docker.env（本项目根目录是 test_extract 的父目录）
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(script_dir)
    env_file = os.path.normpath(os.path.join(project_root, "docker.env"))
    if os.path.exists(env_file):
        load_dotenv(env_file, override=True)
        print(f"  已加载环境变量: {env_file}")
    else:
        print(f"  警告: 未找到 {env_file}，使用系统环境变量")

    rag = LightRAG(
        working_dir="./test_extract/tmp_rag",  # 临时目录，用完即删
        kv_storage="JsonKVStorage",
        doc_status_storage="JsonDocStatusStorage",
        graph_storage="NetworkXStorage",
        vector_storage="NanoVectorDBStorage",
        llm_model_func=partial(
            openai_complete,
            base_url=os.getenv("LLM_BASE_URL"),
            api_key=os.getenv("LLM_API_KEY"),
        ),
        llm_model_name=os.getenv("LLM_MODEL", "Qwen3-235B-A22B-Instruct"),
        embedding_func=EmbeddingFunc(
            embedding_dim=1024,
            func=partial(
                openai_embed.func,
                model=os.getenv("EMBEDDING_MODEL", "bge-m3"),
                base_url=os.getenv("EMBEDDING_BASE_URL"),
                api_key=os.getenv("EMBEDDING_API_KEY"),
            ),
            model_name=os.getenv("EMBEDDING_MODEL", "bge-m3"),
        ),
    )

    # 初始化存储
    import asyncio
    loop = asyncio.new_event_loop()
    try:
        loop.run_until_complete(rag.initialize_storages())
    finally:
        loop.close()

    return rag


def call_llm(prompt: str, rag: LightRAG) -> str:
    """调用 LLM（bypass 模式，不经过 RAG 检索）。"""
    param = QueryParam(mode="bypass")
    return rag.query(prompt, param)


def parse_json_response(text: str) -> dict:
    """从 LLM 返回的文本中提取 JSON。"""
    if not text or not isinstance(text, str):
        return {}
    # 尝试直接解析
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    # 从 markdown 代码块中提取
    json_match = re.search(r"```(?:json)?\s*([\s\S]*?)```", text)
    if json_match:
        try:
            return json.loads(json_match.group(1))
        except json.JSONDecodeError:
            pass

    # 查找最外层大括号
    brace_match = re.search(r"\{[\s\S]*\}", text)
    if brace_match:
        try:
            return json.loads(brace_match.group(0))
        except json.JSONDecodeError:
            pass

    return {}


def extract_entities(text: str, rag: LightRAG) -> list[dict]:
    """抽取实体。"""
    prompt = ENTITY_EXTRACT_PROMPT.format(text=text)
    response = call_llm(prompt, rag)
    data = parse_json_response(response)
    return data.get("entities", [])


def extract_relationships(text: str, entities: list[dict], rag: LightRAG) -> list[dict]:
    """抽取关系。"""
    entities_text = json.dumps(
        [{"name": e.get("normalized_name", e.get("entity_text", "")),
          "type": e.get("entity_type", "UNKNOWN")}
         for e in entities],
        ensure_ascii=False,
        indent=2,
    )
    prompt = RELATION_EXTRACT_PROMPT.format(text=text, entities=entities_text)
    response = call_llm(prompt, rag)
    data = parse_json_response(response)
    return data.get("relationships", [])


# ============================================================
# 主流程
# ============================================================

import sys
if sys.platform == 'win32':
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')


def main():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(script_dir)  # D:\LightRAG
    os.chdir(project_root)

    output_dir = script_dir
    os.makedirs(output_dir, exist_ok=True)

    # 1. 读取文本
    print("读取 test.md ...")
    test_md_path = os.path.join(project_root, "test.md")
    text = load_test_text(test_md_path)
    print(f"  文本长度: {len(text)} 字符")

    # 2. 分段
    chunks = split_text(text, max_chars=4000)
    print(f"  切分为 {len(chunks)} 段")

    # 3. 创建 LLM 实例
    print("创建 LightRAG 实例（仅 LLM 调用）...")
    rag = create_minimal_rag()

    # 4. 逐段抽取实体
    print("\n开始抽取实体...")
    all_entities = []
    for i, chunk in enumerate(chunks, 1):
        print(f"  [{i}/{len(chunks)}] 正在抽取... ", end="", flush=True)
        entities = extract_entities(chunk, rag)
        print(f"找到 {len(entities)} 个实体")
        all_entities.extend(entities)

    # 5. 去重（按 normalized_name）
    seen = set()
    unique_entities = []
    for e in all_entities:
        name = e.get("normalized_name", "")
        if name and name not in seen:
            seen.add(name)
            unique_entities.append(e)

    print(f"\n去重后共 {len(unique_entities)} 个实体")

    # 6. 抽取关系
    print("\n开始抽取关系...")
    relationships = extract_relationships(text, unique_entities, rag)
    print(f"  找到 {len(relationships)} 个关系")

    # 7. 输出 JSON
    entities_path = os.path.join(output_dir, "entities.json")
    relationships_path = os.path.join(output_dir, "relationships.json")
    summary_path = os.path.join(output_dir, "summary.json")

    # entities.json: 完整实体列表
    with open(entities_path, "w", encoding="utf-8") as f:
        json.dump(unique_entities, f, ensure_ascii=False, indent=2)
    print(f"\n实体已保存: {entities_path}")

    # relationships.json: 完整关系列表
    with open(relationships_path, "w", encoding="utf-8") as f:
        json.dump(relationships, f, ensure_ascii=False, indent=2)
    print(f"关系已保存: {relationships_path}")

    # summary.json: 统计信息
    by_type = {}
    for e in unique_entities:
        etype = e.get("entity_type", "UNKNOWN")
        by_type.setdefault(etype, []).append(e)

    summary = {
        "source": "test.md",
        "text_length": len(text),
        "chunks": len(chunks),
        "total_entities": len(unique_entities),
        "total_relationships": len(relationships),
        "entities_by_type": {k: len(v) for k, v in sorted(by_type.items())},
        "entity_type_details": {
            k: [e.get("normalized_name", "") for e in v]
            for k, v in sorted(by_type.items())
        },
        "relationship_type_counts": {},
    }

    rel_by_type = {}
    for r in relationships:
        rtype = r.get("relation_type", "UNKNOWN")
        rel_by_type.setdefault(rtype, []).append(r)
    summary["relationship_type_counts"] = {k: len(v) for k, v in sorted(rel_by_type.items())}

    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    print(f"统计已保存: {summary_path}")

    # 8. 打印摘要
    print("\n" + "=" * 60)
    print("  抽取结果摘要")
    print("=" * 60)
    print(f"\n实体总数: {len(unique_entities)}")
    for etype, items in sorted(by_type.items()):
        names = [e.get("normalized_name", "") for e in items]
        print(f"  {etype} ({len(items)}): {', '.join(names[:6])}{'...' if len(names) > 6 else ''}")

    # 展示部分实体的 definition 和 attributes
    print("\n  部分实体的结构化属性示例:")
    for e in unique_entities[:8]:
        name = e.get("normalized_name", "?")
        etype = e.get("entity_type", "?")
        definition = e.get("definition", "")
        attrs = e.get("attributes", [])
        if definition or attrs:
            print(f"    [{etype}] {name}")
            if definition:
                print(f"      定义: {definition}")
            if attrs:
                for a in attrs:
                    if isinstance(a, dict):
                        print(f"      {a.get('key', '?')}: {a.get('value', '?')}")

    print(f"\n关系总数: {len(relationships)}")
    for rtype, items in sorted(rel_by_type.items()):
        examples = [f"{r['src_entity']}--{r['relation_type']}-->{r['tgt_entity']}" for r in items[:3]]
        print(f"  {rtype} ({len(items)}): {'; '.join(examples)}{'...' if len(items) > 3 else ''}")

    # 9. 清理临时目录
    import shutil
    tmp_dir = os.path.join(output_dir, "tmp_rag")
    if os.path.exists(tmp_dir):
        shutil.rmtree(tmp_dir)
        print(f"\n临时目录已清理: {tmp_dir}")

    print("\n完成！请在 test_extract/ 文件夹中查看结果。")


if __name__ == "__main__":
    main()
