"""Neo4j 直接写入器

绕过 LightRAG 的 ainsert_custom_kg 硬编码行为，直接向 Neo4j 写入：
- 实体节点使用 entity_type 作为 Label（同时保留 :base 兼容 LightRAG 查询层）
- 关系边使用 relation_type 作为边 Label，有向

工作流程：
1. ainsert_custom_kg 负责 Qdrant 向量存储
2. Neo4jDirectWriter 负责 Neo4j 正确结构的图数据
"""

import time
from collections import defaultdict

from neo4j import GraphDatabase


class Neo4jDirectWriter:
    """轻量级 Neo4j 直接写入器。

    按实体类型分组执行 MERGE（Cypher 不支持动态 label），
    按关系类型分组执行有向边写入。
    """

    WORKSPACE_LABEL = "base"

    def __init__(self, uri: str, username: str, password: str):
        self._driver = GraphDatabase.driver(uri, auth=(username, password))

    # ---------------------------------------------------------------
    # 实体写入
    # ---------------------------------------------------------------

    def upsert_entities(self, entities: list[dict]) -> int:
        """将实体写入 Neo4j，使用 entity_type 作为额外 Label。

        Args:
            entities: 实体列表，每项需包含 entity_type, normalized_name,
                      以及 build_custom_kg 产出的 props 字段（或可从实体推断）

        Returns:
            成功写入的实体数量
        """
        # 按 entity_type 分组
        by_type: dict[str, list[dict]] = defaultdict(list)
        for entity in entities:
            etype = (entity.get("entity_type") or "").strip()
            name = (entity.get("normalized_name") or "").strip()
            if not etype or not name:
                continue
            by_type[etype].append(entity)

        total = 0
        with self._driver.session() as session:
            for label, nodes in by_type.items():
                for e in nodes:
                    name = e["normalized_name"]
                    props = {
                        "entity_name": name,
                        "entity_id": name,       # LightRAG 查询层使用 entity_id
                        "entity_type": label,
                        "description": (e.get("description") or e.get("definition") or "").strip(),
                        "source_id": (e.get("source_id") or "").strip(),
                        "file_path": (e.get("file_path") or "").strip(),
                    }
                    # 展开 attr_xxx 动态属性
                    raw_attrs = e.get("attributes")
                    if isinstance(raw_attrs, list):
                        for attr in raw_attrs:
                            if isinstance(attr, dict):
                                k = (attr.get("key") or "").strip()
                                v = (attr.get("value") or "").strip()
                                if k and v:
                                    props[f"attr_{k}"] = v
                    # 保留 original_text
                    etext = (e.get("entity_text") or name).strip()
                    if etext != name:
                        props["original_text"] = etext
                    # 写入时间戳
                    props["created_at"] = int(time.time())

                    query = f"""
                    MERGE (n:`{self.WORKSPACE_LABEL}`:`{label}` {{entity_name: $entity_name}})
                    SET n += $props
                    """
                    session.run(query, entity_name=name, props=props)
                    total += 1

        return total

    # ---------------------------------------------------------------
    # 关系写入
    # ---------------------------------------------------------------

    def upsert_relations(self, relations: list[dict]) -> int:
        """将有向关系写入 Neo4j，使用 relation_type 作为边 Label。

        Args:
            relations: 关系列表，每项需包含 src_entity, tgt_entity, relation_type

        Returns:
            成功写入的关系数量
        """
        # 按 relation_type 分组
        by_type: dict[str, list[dict]] = defaultdict(list)
        for rel in relations:
            src = (rel.get("src_entity") or rel.get("src_id") or "").strip()
            tgt = (rel.get("tgt_entity") or rel.get("tgt_id") or "").strip()
            rtype = (rel.get("relation_type") or rel.get("keywords") or "").strip()
            if not src or not tgt or not rtype:
                continue
            by_type[rtype].append({
                "src_entity": src,
                "tgt_entity": tgt,
                "relation_type": rtype,
                "evidence": (rel.get("evidence") or "").strip(),
                "source_id": (rel.get("source_id") or "").strip(),
                "description": (rel.get("description") or "").strip(),
                "weight": rel.get("weight", 1.0),
                "file_path": (rel.get("file_path") or "").strip(),
            })

        total = 0
        with self._driver.session() as session:
            for label, edges in by_type.items():
                for e in edges:
                    props = {
                        "source_id": e["source_id"],
                        "description": e["description"],
                        "weight": e["weight"],
                        "created_at": int(time.time()),
                    }
                    if e["evidence"]:
                        props["evidence"] = e["evidence"]
                    if e["file_path"]:
                        props["file_path"] = e["file_path"]

                    query = f"""
                    MATCH (src:`{self.WORKSPACE_LABEL}` {{entity_name: $src}})
                    MATCH (tgt:`{self.WORKSPACE_LABEL}` {{entity_name: $tgt}})
                    MERGE (src)-[r:`{label}`]->(tgt)
                    SET r += $props
                    """
                    session.run(query, src=e["src_entity"], tgt=e["tgt_entity"], props=props)
                    total += 1

        return total

    # ---------------------------------------------------------------
    # 删除
    # ---------------------------------------------------------------

    def delete_by_entity(self, entity_name: str) -> int:
        """删除指定实体节点及其所有连接边。

        DETACH DELETE 会同时删除节点和所有类型的边（不依赖特定 label）。

        Returns:
            删除的节点数（0 或 1）
        """
        with self._driver.session() as session:
            result = session.run("""
            MATCH (n:`{label}` {entity_name: $name})
            DETACH DELETE n
            """.format(label=self.WORKSPACE_LABEL), name=entity_name)
            summary = result.consume()
            return summary.counters.nodes_deleted

    def delete_by_relation(self, src_entity: str, tgt_entity: str) -> int:
        """删除两个实体之间的所有关系边（不限关系类型）。

        Returns:
            删除的边数量
        """
        with self._driver.session() as session:
            result = session.run("""
            MATCH (src:`{label}` {{entity_name: $src}})-[r]-(tgt:`{label}` {{entity_name: $tgt}})
            DELETE r
            """.format(label=self.WORKSPACE_LABEL), src=src_entity, tgt=tgt_entity)
            summary = result.consume()
            return summary.counters.relationships_deleted

    def delete_by_doc_id(self, source_id: str) -> dict:
        """按文档 source_id 删除该文档写入的所有实体及其边。

        注意：只删除 entities 上的 source_id 属性匹配的节点，
        关系边的 source_id 会在节点删除时被 DETACH 一并删除。

        Returns:
            {"nodes_deleted": int, "relationships_deleted": int}
        """
        stats = {"nodes_deleted": 0, "relationships_deleted": 0}
        with self._driver.session() as session:
            result = session.run("""
            MATCH (n:`{label}` {source_id: $source_id})
            DETACH DELETE n
            """.format(label=self.WORKSPACE_LABEL), source_id=source_id)
            summary = result.consume()
            stats["nodes_deleted"] = summary.counters.nodes_deleted
            stats["relationships_deleted"] = summary.counters.relationships_deleted
        return stats

    # ---------------------------------------------------------------
    # 清理
    # ---------------------------------------------------------------

    def cleanup_wrong_labels(self) -> dict:
        """清理 ainsert_custom_kg 写入的错误结构数据。

        1. 删除所有 :DIRECTED 边（已被正确 label 的边替代）
        2. 返回清理统计

        注意：不移除 :base label，因为 LightRAG 查询层依赖它。
        实体节点使用双 label 策略（:base + :entity_type）。
        """
        stats = {"directed_edges_deleted": 0}

        with self._driver.session() as session:
            # 删除所有 DIRECTED 边
            result = session.run("""
            MATCH ()-[r:DIRECTED]-()
            WITH r LIMIT 10000
            DELETE r
            RETURN count(r) AS deleted
            """)
            record = result.single()
            if record:
                stats["directed_edges_deleted"] += record["deleted"]

            # 循环删除剩余（LIMIT 10000 避免单次事务过大）
            while stats["directed_edges_deleted"] > 0:
                result = session.run("""
                MATCH ()-[r:DIRECTED]-()
                WITH r LIMIT 10000
                DELETE r
                RETURN count(r) AS deleted
                """)
                record = result.single()
                count = record["deleted"] if record else 0
                stats["directed_edges_deleted"] += count
                if count == 0:
                    break

        return stats

    def close(self):
        """关闭驱动连接。"""
        self._driver.close()

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()
