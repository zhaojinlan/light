"""法律判决文书实体抽取与知识图谱插入服务

通过 LLM 从法律判决文书中抽取诈骗案件相关实体和关系，
然后以自定义知识图谱的方式插入 LightRAG + Neo4j + Qdrant。

实体类型（7 种）:
    summary, FraudScenario, FraudFeature, FraudMethod,
    PreventionMeasure, LawRegulation, RelatedCase
"""

import json
import os
from typing import Any

from lightrag import LightRAG, QueryParam

from .kg_config import create_lightrag_neo4j_qdrant
from .entity_disambiguation import EntityDisambiguationService, _run_async_on_rag


# ============================================================
# 实体类型定义
# ============================================================

# 10 种预定义诈骗场景（仅可从中选择）
FRAUD_SCENARIOS = [
    "刷单返利类诈骗",
    "虚假网络投资理财类诈骗",
    "虚假购物、服务类诈骗",
    "冒充电商物流客服类诈骗",
    "贷款、征信类诈骗",
    "冒充领导、熟人类诈骗",
    "冒充公检法类诈骗",
    "婚恋、交友类诈骗",
    "网络游戏虚假交易类诈骗",
    "机票退改类诈骗",
]

ENTITY_TYPES = [
    "summary",                 # 案件简要描述（固定格式总结，每篇文书 1 个）
    "FraudScenario",          # 诈骗场景，从预定义 10 类中选择
    "FraudFeature",            # 骗局特征（欺骗性表现）
    "FraudMethod",             # 诈骗手法（技术手段）
    "PreventionMeasure",       # 防范建议（基于案情推理生成）
    "LawRegulation",           # 法律法规（保留但非核心）
    "RelatedCase",             # 关联案例（详细案例总结，用于相似案例匹配）
]

# ============================================================
# 校验常量（用于过滤 LLM 输出中的非法数据）
# ============================================================

VALID_ENTITY_TYPES = set(ENTITY_TYPES)

VALID_RELATION_TYPES = {
    # summary → FraudScenario / FraudFeature / FraudMethod
    "involves",      # summary → FraudScenario（案件摘要涉及某诈骗场景）
    "describes",     # summary → FraudFeature（摘要描述骗局特征）
    "mentions",      # summary → FraudMethod（摘要提及诈骗手法）

    # RelatedCase → FraudScenario / FraudFeature / FraudMethod
    "summarizes",    # RelatedCase → FraudScenario（关联案例涉及某诈骗场景）
    "has_feature",   # RelatedCase → FraudFeature（关联案例具有某骗局特征）
    "uses",          # RelatedCase → FraudMethod（关联案例使用某诈骗手法）

    # FraudMethod / FraudFeature → LawRegulation
    "violates",      # FraudMethod → LawRegulation（诈骗手法违反法律法规）

    # PreventionMeasure → FraudMethod / FraudFeature
    "prevents",      # PreventionMeasure → FraudMethod（防范措施预防某手法）
    "counters",      # PreventionMeasure → FraudFeature（防范措施应对某特征）
}


# ============================================================
# 属性 Schema（每个实体类型的预定义属性 key）
# ============================================================

ATTRIBUTE_SCHEMA: dict[str, list[str]] = {
    "FraudScenario": ["涉案金额", "被害人数量", "案件阶段"],
    "FraudFeature": ["欺骗性表现", "出现频次", "受害群体"],
    "FraudMethod": ["技术手段", "沟通渠道", "实施步骤"],
    "PreventionMeasure": ["防范对象", "适用人群", "可操作性"],
    "LawRegulation": ["法律条款号", "处罚标准"],
    "RelatedCase": ["案情经过", "判决结果", "涉案金额", "被告人", "受害人", "防范启示"],
}


# ============================================================
# Prompt 模板
# ============================================================

ENTITY_EXTRACT_PROMPT = """你是法律判决文书（诈骗案件）知识图谱的实体抽取专家。

你的任务是：
从{text}中，抽取诈骗案件相关实体。

必须严格按照 schema 输出。

# 实体类型定义

1. summary — 案件简要描述（固定格式总结，每篇文书 1 个）
   固定格式："{被告人身份+姓名}以{诈骗方式/理由}，通过{具体手段}骗取{被害人}共计{金额}，被判处{刑罚+罚金}"
   示例："张某以虚构境外期货交易平台为由，通过伪造交易截图和持续要求追加资金的手段骗取刘某、赵某共计609500元，被判处有期徒刑十年并处罚金二十万元"
   注意：normalized_name 必须使用文档的 doc_id（例如 "test_doc_001" 或 "冒充公检法类诈骗/冒充公检法类诈骗-案例一"），绝不使用摘要文本本身。完整的摘要内容应放入 definition 字段中。

2. FraudScenario — 诈骗场景（必须从以下 10 种预定义类型中选择，可多选）
   预定义类型：
   - 刷单返利类诈骗
   - 虚假网络投资理财类诈骗
   - 虚假购物、服务类诈骗
   - 冒充电商物流客服类诈骗
   - 贷款、征信类诈骗
   - 冒充领导、熟人类诈骗
   - 冒充公检法类诈骗
   - 婚恋、交友类诈骗
   - 网络游戏虚假交易类诈骗
   - 机票退改类诈骗
   normalized_name 必须与上述预定义名称完全一致，不得修改
   建议属性：涉案金额、被害人数量、案件阶段

3. FraudFeature — 骗局特征（从原文中抽取的欺骗性表现，2-8 个）
   例如：虚构高收益投资平台、伪造交易盈利截图、承诺月收益30%、
         持续以各种理由要求追加资金、设置无法提现的障碍
   建议属性：欺骗性表现、出现频次、受害群体

4. FraudMethod — 诈骗手法（从原文中抽取的技术手段，1-5 个）
   例如：社交软件诱导投资、虚假网站搭建、手动修改后台交易数据、
         伪造第三方支付渠道、冒充平台客服
   建议属性：技术手段、沟通渠道、实施步骤

5. PreventionMeasure — 防范建议（必生成项基于案情推理）
   说明：判决书通常不写防范措施，你需要根据案情主动推理。
   思考步骤：
   1. 识别本文涉及的所有诈骗手法（FraudMethod）和骗局特征（FraudFeature）
   2. 对每种手法/特征，问自己：普通人遇到这种情况应该怎么做才能避免被骗？
   3. 将答案提炼为具体的防范措施（3-5 条）
   示例：手法=伪造投资平台 → 防范=核实平台金融监管牌照
         特征=承诺高收益回报 → 防范=警惕超出正常水平的投资回报承诺
         手法=冒充公检法 → 防范=公检法机关不会电话要求转账或提供验证码
   建议属性：防范对象、适用人群、可操作性

6. LawRegulation — 法律法规（从原文中抽取，非核心）
   例如：刑法第266条（诈骗罪）、"虚构事实隐瞒真相"
   建议属性：法律条款号、处罚标准

7. RelatedCase — 关联案例（详细案例总结，每篇文书生成 1 个）
   normalized_name 格式："{{被告人姓名}}{{诈骗场景}}案"
   示例："张某虚假网络投资理财诈骗案"、"李某冒充客服退款诈骗案"
   与 summary 的区别：summary 是固定格式 1 句话，RelatedCase 是详细的案例总结，
   包含案情经过、诈骗手法、判决结果、防范启示等，用于相似案例匹配和知识库展示。
   建议属性：案情经过、判决结果、涉案金额、被告人、受害人、防范启示

# 抽取规则

1. summary 仅生成 1 个，必须严格遵循固定格式
2. FraudScenario 必须从预定义 10 类中选择，normalized_name 不得修改
3. 其他实体必须来自原文或可合理推断
4. 保留原文 evidence（summary 的 evidence 可为空）
5. 防范措施可基于案情推理生成，不必局限于原文
6. RelatedCase 需包含完整的案情经过和防范启示
7. 同义词统一（如"投资理财"与"投资平台"归入同一 normalized_name）
8. 若不确定，confidence < 0.6
9. 不输出解释
10. 输出合法 JSON

# 逐步思考（在输出 JSON 之前，请先按以下步骤思考）

请按以下顺序逐步推理（你的思考过程会放在 <thinking> 标签中，最终输出 JSON）：
1. 本文涉及哪些诈骗手法（FraudMethod）？每种手法的核心欺骗点是什么？
2. 本文有哪些骗局特征（FraudFeature）？这些特征为什么能骗到人？
3. 针对上面列出的每种手法和特征，普通人应该采取什么具体措施来防范？
4. 根据上述推理，生成 3-5 条 PreventionMeasure 实体
5. 整理所有实体，输出 JSON

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
- attributes: 从原文中提取该实体类型的关键属性，key 必须使用建议属性中列出的名称
- evidence: 原文中支撑该实体的具体句子（summary 可为空）
- aliases: 原文中出现的同义词/别名
"""


RELATION_EXTRACT_PROMPT = """你是法律判决文书（诈骗案件）知识图谱的关系抽取专家。

你的任务是：
从{text}中，基于已抽取的实体列表，抽取实体之间的关系。

# 关系类型定义

1. involves（summary → FraudScenario）
   案件摘要涉及某类诈骗场景
   例：(doc_id，如 "test_doc_001") involves 虚假网络投资理财类诈骗

2. describes（summary → FraudFeature）
   案件摘要描述了某个骗局特征
   例：(doc_id) describes 虚构高收益投资平台

3. mentions（summary → FraudMethod）
   案件摘要提及某种诈骗手法
   例：(doc_id) mentions 伪造交易截图

4. summarizes（RelatedCase → FraudScenario）
   关联案例涉及某类诈骗场景
   例：(关联案例) summarizes 虚假网络投资理财类诈骗

5. has_feature（RelatedCase → FraudFeature）
   关联案例具有某个骗局特征
   例：(关联案例) has_feature 持续要求追加资金

6. uses（RelatedCase → FraudMethod）
   关联案例使用某种诈骗手法
   例：(关联案例) uses 伪造交易盈利截图

7. violates（FraudMethod → LawRegulation）
   某种诈骗手法违反某条法律法规
   例：伪造投资平台 violates 刑法第266条

8. prevents（PreventionMeasure → FraudMethod）
   某种防范措施可以预防某种诈骗手法
   例：核实平台资质 prevents 虚假网站搭建

9. counters（PreventionMeasure → FraudFeature）
   某种防范措施可以应对某个骗局特征
   例：警惕高收益承诺 counters 承诺月收益30%

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
    """从法律判决文书中抽取诈骗案件相关实体。

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
    - 跳过 entity_type 不在预定义 7 种中的实体
    - attributes 中 key 必须在 ATTRIBUTE_SCHEMA 预定义列表中，否则丢弃
    - 跳过 relation_type 不在预定义 9 种中的关系
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

        # summary 实体使用 doc_id 作为 entity_name，摘要文本放入 description
        if etype == "summary":
            name = source_id  # 用 doc_id 作为 entity_name
            etext = entity.get("entity_text") or ""
            definition = (entity.get("definition") or "").strip()
            description = etext if etext else definition

        valid_names.add(name)

        etext = entity.get("entity_text") or name
        definition = (entity.get("definition") or "").strip()
        description = definition if definition else etext

        # 将 attributes 展开为 attr_xxx 键值对（仅保留预定义属性）
        node_props = {
            "entity_name": name,
            "entity_type": etype,
            "description": description,
            "source_id": source_id,
            "file_path": file_path,
        }
        valid_attrs = set(ATTRIBUTE_SCHEMA.get(etype, []))
        raw_attrs = entity.get("attributes")
        if isinstance(raw_attrs, list):
            for attr in raw_attrs:
                if isinstance(attr, dict):
                    k = (attr.get("key") or "").strip()
                    v = (attr.get("value") or "").strip()
                    if k and v and k in valid_attrs:
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
        1. 调用 LLM 抽取实体（7 种法律领域实体类型）
        2. 调用 LLM 抽取实体间关系（9 种关系类型）
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

        # 步骤 1.6：实体消歧（精确匹配 + embedding 相似度）
        # 注意：summary 和 FraudScenario 不做消歧
        #   - summary：每篇文书的案件摘要不同，不应合并
        #   - FraudScenario：预定义分类标签，多文书可指向同一场景，不应合并
        existing_names = set()
        disambig_service = EntityDisambiguationService(self.rag)

        # 获取已有图谱中的实体名称（通过 entities_vdb 查询）
        try:
            all_existing = _run_async_on_rag(
                self.rag,
                self.rag.entities_vdb.query("", top_k=5000),
            )
            for item in all_existing:
                name = item.get("entity_name", "")
                if name:
                    existing_names.add(name)
            print(f"  消歧: 图谱中已有 {len(existing_names)} 个实体")
        except Exception as e:
            print(f"  消歧: 获取已有实体失败 ({e})，跳过精确匹配，仅做相似度检测")

        # 精确匹配去重 + embedding 相似度消歧
        deduped_entities = []
        merged_count = 0
        no_disambig_types = {"summary", "FraudScenario"}
        for entity in valid_entities:
            name = (entity.get("normalized_name") or "").strip()
            etype = (entity.get("entity_type") or "").strip()

            # summary 和 FraudScenario 不做消歧，直接保留
            if etype in no_disambig_types:
                deduped_entities.append(entity)
                continue

            if name in existing_names:
                # 精确匹配命中，跳过
                print(f"  [精确去重] '{name}' 已存在于图谱中，跳过")
                continue

            # 精确匹配未命中，尝试 embedding 相似度消歧
            if disambig_service.disambiguate_and_merge(name, etype):
                # 发生了合并，该实体不再写入
                merged_count += 1
                continue

            # 无相似实体，保留该实体
            deduped_entities.append(entity)

        if merged_count > 0:
            print(f"  消歧完成: {merged_count} 个实体被自动合并到已有实体中")

        valid_entities = deduped_entities

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

    def delete_by_doc_id(self, doc_id: str) -> dict:
        """按文档 ID 删除文档及其所有关联数据（Neo4j + Qdrant + Redis）。"""
        result = self.rag.delete_by_doc_id(doc_id)
        return {
            "status": getattr(result, "status", "ok"),
            "doc_id": doc_id,
            "message": getattr(result, "message", str(result)),
        }

    def delete_by_entity(self, entity_name: str) -> dict:
        """按实体名称删除实体及其所有关系（Neo4j + Qdrant）。"""
        result = self.rag.delete_by_entity(entity_name)
        return {
            "status": getattr(result, "status", "ok"),
            "entity_name": entity_name,
            "message": getattr(result, "message", str(result)),
        }

    def delete_by_relation(self, src_entity: str, tgt_entity: str) -> dict:
        """删除两个实体之间的所有关系边（Neo4j + Qdrant）。"""
        result = self.rag.delete_by_relation(src_entity, tgt_entity)
        return {
            "status": getattr(result, "status", "ok"),
            "src": src_entity,
            "tgt": tgt_entity,
            "message": getattr(result, "message", str(result)),
        }
