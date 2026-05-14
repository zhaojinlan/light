"""自定义实体抽取与知识图谱插入服务

使用 main.py 中定义的铜合金领域实体类型 schema，
通过 LLM 从文本中抽取实体和关系，
然后以自定义知识图谱的方式插入 LightRAG + Neo4j + Qdrant。

实体类型（9 种）:
    MaterialSystem, Phase, ConductivityMechanism, StrengtheningMechanism,
    PrecipitationMechanism, Property, ServicePerformance, Application, FailureMode
"""

import json
import os
from typing import Any

from lightrag import LightRAG, QueryParam

from .kg_config import create_lightrag_neo4j_qdrant


# ============================================================
# 实体类型定义（与 main.py 一致）
# ============================================================

ENTITY_TYPES = [
    "MaterialSystem",              # 铜合金体系名称，如 CuNiSi、CuCrZr
    "Phase",                       # 物相名称，如 α相、β相、δ-Ni2Si
    "ConductivityMechanism",       # 导电机制，如 析出净化、溶质散射
    "StrengtheningMechanism",      # 强化机制，如 固溶强化、析出强化
    "PrecipitationMechanism",      # 析出/相变机制，如 共格析出、spinodal decomposition
    "Property",                    # 材料性能，如 抗拉强度、导电率
    "ServicePerformance",          # 服役性能，如 抗氧化性、耐磨性
    "Application",                 # 应用场景，如 电子工业、汽车连接器
    "FailureMode",                 # 失效模式，如 应力腐蚀开裂、脱锌
]

# ============================================================
# 校验常量（用于过滤 LLM 输出中的非法数据）
# ============================================================

VALID_ENTITY_TYPES = set(ENTITY_TYPES)

VALID_RELATION_TYPES = {
    "has_phase",
    "strengthened_by",
    "has_property",
    "used_in",
    "failure_mode_of",
    "mechanism_of",
    "contains",
}


# ============================================================
# Prompt 模板
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


def _call_llm(prompt: str, rag: LightRAG) -> str:
    """通过 LightRAG 的 LLM 函数直接调用（不经过 RAG 检索）。

    Args:
        prompt: 完整的提示词
        rag: LightRAG 实例

    Returns:
        LLM 返回的文本
    """
    param = QueryParam(mode="bypass")
    return rag.query(prompt, param)


def _parse_json_response(text: str) -> dict:
    """从 LLM 返回的文本中解析 JSON。

    处理可能包含的思维链标记或多余文本。

    Args:
        text: LLM 返回的文本

    Returns:
        解析后的 JSON 字典
    """
    if not text or not isinstance(text, str):
        return {}

    # 尝试直接解析
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    # 尝试从 markdown 代码块中提取
    import re

    json_match = re.search(r"```(?:json)?\s*([\s\S]*?)```", text)
    if json_match:
        try:
            return json.loads(json_match.group(1))
        except json.JSONDecodeError:
            pass

    # 尝试查找最外层的大括号
    brace_match = re.search(r"\{[\s\S]*\}", text)
    if brace_match:
        try:
            return json.loads(brace_match.group(0))
        except json.JSONDecodeError:
            pass

    return {}


def extract_entities(text: str, rag: LightRAG) -> list[dict[str, Any]]:
    """从文本中抽取铜合金领域实体。

    Args:
        text: 输入的文本内容
        rag: LightRAG 实例

    Returns:
        实体列表，每项包含 entity_text, entity_type, normalized_name,
        aliases, evidence, confidence
    """
    prompt = ENTITY_EXTRACT_PROMPT.format(text=text)
    response = _call_llm(prompt, rag)
    data = _parse_json_response(response)
    return data.get("entities", [])


def extract_relationships(
    text: str, entities: list[dict[str, Any]], rag: LightRAG
) -> list[dict[str, Any]]:
    """从文本中抽取实体之间的关系。

    Args:
        text: 输入的文本内容
        entities: 已抽取的实体列表
        rag: LightRAG 实例

    Returns:
        关系列表，每项包含 src_entity, tgt_entity, relation_type, evidence
    """
    entities_text = json.dumps(
        [{"name": e.get("normalized_name", ""), "type": e.get("entity_type", "UNKNOWN")} for e in entities],
        ensure_ascii=False,
        indent=2,
    )
    prompt = RELATION_EXTRACT_PROMPT.format(text=text, entities=entities_text)
    response = _call_llm(prompt, rag)
    data = _parse_json_response(response)
    return data.get("relationships", [])


def build_custom_kg(
    text: str,
    entities: list[dict[str, Any]],
    relationships: list[dict[str, Any]],
    doc_id: str | None = None,
    file_path: str = "custom_kg",
) -> dict[str, Any]:
    """将抽取的实体和关系转换为 LightRAG insert_custom_kg 所需格式。

    包含校验逻辑：
    - 跳过 normalized_name / entity_type 为空的实体
    - 跳过 entity_type 不在预定义 9 种中的实体
    - attributes 中 key/value 缺失的条目被丢弃
    - 跳过 relation_type 不在预定义 7 种中的关系
    - 跳过 src/tgt 不在有效实体中的关系

    Args:
        text: 原始文本内容
        entities: 实体抽取结果
        relationships: 关系抽取结果
        doc_id: 文档 ID，若不提供则自动生成
        file_path: 文件路径标识

    Returns:
        符合 insert_custom_kg 接口格式的字典
    """
    import hashlib

    source_id = doc_id or f"doc-{hashlib.md5(text.encode()).hexdigest()[:32]}"

    # 构建 chunk 数据
    chunk_data = {
        "chunks": [
            {
                "content": text,
                "source_id": source_id,
                "chunk_order_index": 0,
                "file_path": file_path,
            }
        ],
    }

    # 构建实体数据（带校验 + 结构化属性展开）
    chunk_data["entities"] = []
    valid_names = set()
    skipped = []

    for entity in entities:
        name = (entity.get("normalized_name") or "").strip()
        etype = (entity.get("entity_type") or "").strip()

        if not name:
            skipped.append(f"(空名, type={entity.get('entity_type', '?')})")
            continue
        if etype not in VALID_ENTITY_TYPES:
            skipped.append(f"{name} (未知类型: {etype})")
            continue

        valid_names.add(name)

        etext = entity.get("entity_text") or name
        definition = (entity.get("definition") or "").strip()
        description = definition if definition else etext

        # 将 attributes 展开为 attr_xxx 键值对
        node_props = {
            "entity_name": name,
            "entity_type": etype,
            "description": description,
            "source_id": source_id,
            "file_path": file_path,
        }
        raw_attrs = entity.get("attributes")
        if isinstance(raw_attrs, list):
            for attr in raw_attrs:
                if isinstance(attr, dict):
                    k = (attr.get("key") or "").strip()
                    v = (attr.get("value") or "").strip()
                    if k and v:
                        node_props[f"attr_{k}"] = v

        chunk_data["entities"].append(node_props)

    if skipped:
        print(f"  build_custom_kg: 跳过 {len(skipped)} 个无效实体: {', '.join(skipped)}")

    # 构建关系数据（带校验）
    chunk_data["relationships"] = []
    rel_skipped = []

    for rel in relationships:
        src = (rel.get("src_entity") or "").strip()
        tgt = (rel.get("tgt_entity") or "").strip()
        rtype = (rel.get("relation_type") or "").strip()

        if not src or not tgt:
            rel_skipped.append(f"(空端点, type={rtype})")
            continue
        if rtype not in VALID_RELATION_TYPES:
            rel_skipped.append(f"{src}->{tgt} (未知关系: {rtype})")
            continue
        if src not in valid_names or tgt not in valid_names:
            rel_skipped.append(f"{src}->{tgt} (实体不存在)")
            continue

        evidence = (rel.get("evidence") or "").strip()
        chunk_data["relationships"].append(
            {
                "src_id": src,
                "tgt_id": tgt,
                "src_entity": src,       # 用于 Neo4jDirectWriter
                "tgt_entity": tgt,       # 用于 Neo4jDirectWriter
                "relation_type": rtype,  # 用于 Neo4j 边 label
                "evidence": evidence,    # 用于 Neo4j 边属性
                "description": f"{rtype}: {evidence}" if evidence else rtype,
                "keywords": rtype,
                "weight": 1.0,
                "source_id": source_id,
                "file_path": file_path,
            }
        )

    if rel_skipped:
        print(f"  build_custom_kg: 跳过 {len(rel_skipped)} 个无效关系: {', '.join(rel_skipped)}")

    return chunk_data


class CustomEntityService:
    """自定义实体抽取与知识图谱插入服务。

    工作流程：
    1. 使用 LLM + 自定义 prompt 从文本中抽取实体
    2. 使用 LLM + 关系 prompt 抽取实体间关系
    3. 直接入库（Qdrant 实体向量 + Neo4j 图 + KV 文档追踪）
    """

    def __init__(
        self,
        rag: LightRAG | None = None,
        env_path: str = "docker.env",
        working_dir: str = "./rag_storage",
    ):
        """初始化自定义实体服务。

        Args:
            rag: 已初始化的 LightRAG 实例。若为 None，则自动创建 Neo4j+Qdrant 版本
            env_path: 环境变量文件路径
            working_dir: 本地缓存目录
        """
        self.rag = rag or create_lightrag_neo4j_qdrant(
            working_dir=working_dir,
            env_path=env_path,
        )

    def insert_with_custom_schema(
        self,
        text: str,
        doc_id: str | None = None,
        file_path: str = "custom_kg",
    ) -> dict[str, Any]:
        """使用自定义 schema 插入文本到知识图谱。

        该方法会：
        1. 调用 LLM 抽取实体（使用 main.py 定义的 9 种实体类型）
        2. 调用 LLM 抽取实体间关系
        3. 转换为 LightRAG 格式并插入

        Args:
            text: 要插入的文本内容
            doc_id: 自定义文档 ID，若不指定则自动生成
            file_path: 文件路径标识，用于溯源

        Returns:
            包含抽取结果的字典：
            - entities: 抽取的实体列表
            - relationships: 抽取的关系列表
            - doc_id: 文档 ID
        """
        # 步骤 1：抽取实体
        entities = extract_entities(text, self.rag)

        # 步骤 1.5：过滤无效实体
        valid_entities = [
            e for e in entities
            if (e.get("normalized_name") or "").strip()
            and (e.get("entity_type") or "").strip() in VALID_ENTITY_TYPES
        ]
        if len(valid_entities) < len(entities):
            invalid = [e for e in entities if not (e.get("normalized_name") or "").strip() or (e.get("entity_type") or "").strip() not in VALID_ENTITY_TYPES]
            print(f"  插入前校验: 跳过 {len(invalid)} 个无效实体")
            for inv in invalid:
                print(f"    - {inv.get('normalized_name', '(空名)')} [type={inv.get('entity_type', '(空)')}]")

        # 步骤 2：抽取关系
        relationships = extract_relationships(text, valid_entities, self.rag)

        # 步骤 2.5：过滤无效关系
        valid_names = {e["normalized_name"] for e in valid_entities}
        valid_relationships = [
            r for r in relationships
            if (r.get("src_entity") or "").strip() in valid_names
            and (r.get("tgt_entity") or "").strip() in valid_names
        ]
        if len(valid_relationships) < len(relationships):
            print(f"  插入前校验: 跳过 {len(relationships) - len(valid_relationships)} 个无效关系（实体不存在）")

        # 步骤 3：直接入库（绕过 ainsert_custom_kg，避免重复节点）
        from .direct_ingestion import direct_ingest

        resolved_doc_id = doc_id
        stats = direct_ingest(self.rag, text, valid_entities, valid_relationships, resolved_doc_id, file_path)
        print(f"  入库完成: {stats['entities']} 个实体, {stats['relations']} 个关系, {stats['chunks']} 个 chunk")

        return {
            "entities": entities,
            "relationships": relationships,
            "doc_id": resolved_doc_id,
        }

    def query(self, question: str, mode: str = "mix") -> str:
        """执行知识图谱问答查询。

        Args:
            question: 用户提问
            mode: 检索模式（local/global/hybrid/naive/mix）

        Returns:
            LLM 生成的回答
        """
        param = QueryParam(mode=mode)
        return self.rag.query(question, param)

    def _neo4j_writer(self):
        """创建 Neo4jDirectWriter 实例。"""
        from .neo4j_writer import Neo4jDirectWriter
        return Neo4jDirectWriter(
            uri=os.getenv("NEO4J_URI", "bolt://localhost:7687"),
            username=os.getenv("NEO4J_USERNAME", "neo4j"),
            password=os.getenv("NEO4J_PASSWORD", "LightRAG2026!"),
        )

    def delete_by_doc_id(self, doc_id: str) -> dict:
        """按文档 ID 删除文档及其所有关联数据。

        同时清理：
        1. Neo4j 中的实体节点和关系边
        2. Qdrant 中的实体向量
        3. KV store 中的元数据

        Args:
            doc_id: 文档 ID

        Returns:
            删除统计
        """
        import asyncio

        # 从 full_entities KV store 获取该文档的实体列表
        loop = asyncio.new_event_loop()
        try:
            full_data = loop.run_until_complete(
                self.rag.full_entities.get_by_id(doc_id)
            )
        finally:
            loop.close()

        entity_names = full_data.get("entity_names", []) if full_data else []

        # 1. 先删除 Neo4j 中的实体节点（DETACH DELETE 会同时删除所有边）
        with self._neo4j_writer() as writer:
            for name in entity_names:
                writer.delete_by_entity(name)

        # 2. 再删除 LightRAG 的数据（Qdrant + KV + 旧 Neo4j）
        lr_result = self.rag.delete_by_doc_id(doc_id)
        return {
            "status": "deleted",
            "doc_id": doc_id,
            "entities_removed": entity_names,
            "lightrag_result": str(lr_result) if lr_result else "ok",
        }

    def delete_by_entity(self, entity_name: str) -> dict:
        """按实体名称删除实体及其所有关系。

        Args:
            entity_name: 实体名称

        Returns:
            删除统计
        """
        # 1. Neo4j: 删除 entity_name 节点 + 所有类型的边
        with self._neo4j_writer() as writer:
            writer.delete_by_entity(entity_name)

        # 2. LightRAG: 删除 entity_id 节点 + Qdrant
        lr_result = self.rag.delete_by_entity(entity_name)
        return {
            "status": "deleted",
            "entity_name": entity_name,
            "lightrag_result": str(lr_result) if lr_result else "ok",
        }

    def delete_by_relation(self, src_entity: str, tgt_entity: str) -> dict:
        """删除两个实体之间的所有关系边。

        Args:
            src_entity: 源实体名称
            tgt_entity: 目标实体名称

        Returns:
            删除统计
        """
        # 1. Neo4j: 删除所有类型的有向边
        with self._neo4j_writer() as writer:
            writer.delete_by_relation(src_entity, tgt_entity)

        # 2. LightRAG: 删除无向 DIRECTED 边 + Qdrant
        lr_result = self.rag.delete_by_relation(src_entity, tgt_entity)
        return {
            "status": "deleted",
            "src": src_entity,
            "tgt": tgt_entity,
            "lightrag_result": str(lr_result) if lr_result else "ok",
        }
