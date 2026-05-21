"""实体消歧服务

在知识图谱构建时，对新增实体进行自动相似度检测和合并。
流程：
1. 精确字符串匹配去重（由调用方完成）
2. 对新实体计算 embedding，与已有同类型实体做余弦相似度比较
3. 超过阈值则自动调用 rag.amerge_entities() 合并
4. 合并成功后同步更新 full_entities / full_relations 追踪数据

复用 LightRAG 库已有能力：
- entities_vdb.query(query_embedding=...) 做向量相似度搜索
- rag.amerge_entities() 做实体合并（含关系迁移、向量更新）
- rag.embedding_func 计算实体 embedding
"""

import asyncio
import logging
from typing import Any

from lightrag import LightRAG

logger = logging.getLogger(__name__)

# 默认相似度阈值（cosine similarity，Qdrant 使用 COSINE 距离）
DEFAULT_SIMILARITY_THRESHOLD = 0.85


def _run_async_on_rag(rag: LightRAG, coro):
    """在 rag 的持久事件循环或临时循环上运行协程。"""
    if hasattr(rag, "_persistent_loop") and rag._persistent_loop is not None:
        return rag._persistent_loop.run_coroutine(coro)
    loop = asyncio.new_event_loop()
    try:
        return loop.run_until_complete(coro)
    finally:
        loop.close()


async def _get_entities_by_type(entities_vdb, target_type: str, workspace: str = "") -> dict[str, dict[str, Any]]:
    """从 entities_vdb 中获取指定类型的所有实体。

    由于 Qdrant query 只支持 top-K 相似度检索，不支持"获取全部"，
    我们通过多次查询 + 提高 top_k 来获取尽可能多的同类型实体。
    """
    all_entities: dict[str, dict[str, Any]] = {}

    # 用空查询获取尽可能多的实体（top_k 设大）
    results = await entities_vdb.query("", top_k=5000)

    for item in results:
        entity_type = item.get("entity_type", "")
        entity_name = item.get("entity_name", "")
        if entity_type == target_type and entity_name:
            all_entities[entity_name] = item

    return all_entities


class EntityDisambiguationService:
    """实体消歧服务。

    在实体插入前检测是否与已有实体高度相似，并自动合并。
    """

    def __init__(self, rag: LightRAG, threshold: float = DEFAULT_SIMILARITY_THRESHOLD):
        """初始化消歧服务。

        Args:
            rag: LightRAG 实例
            threshold: 余弦相似度阈值，超过此值认为是重复实体
        """
        self.rag = rag
        self.threshold = threshold

    def find_similar_entities(
        self,
        entity_name: str,
        entity_type: str,
        threshold: float | None = None,
        top_k: int = 10,
    ) -> list[dict[str, Any]]:
        """查找与给定实体相似的已有实体。

        Args:
            entity_name: 新实体名称
            entity_type: 新实体类型
            threshold: 相似度阈值（覆盖默认值）
            top_k: 返回的候选数量

        Returns:
            按相似度降序排列的候选列表：
            [{"name": str, "score": float, "entity_type": str}]
        """
        thresh = threshold if threshold is not None else self.threshold

        # 用实体名称和默认描述计算 embedding
        content = entity_name
        embedding_result = _run_async_on_rag(
            self.rag,
            self.rag.embedding_func([content], context="query", _priority=5),
        )
        embedding = embedding_result[0]

        # 查询 entities_vdb 获取相似实体
        results = _run_async_on_rag(
            self.rag,
            self.rag.entities_vdb.query("", top_k=top_k, query_embedding=embedding),
        )

        # 过滤：只保留同类型、非自身、超过阈值的实体
        candidates = []
        for item in results:
            if item.get("entity_name") == entity_name:
                continue
            if item.get("entity_type") != entity_type:
                continue
            score = item.get("distance", 0.0)
            if score >= thresh:
                candidates.append({
                    "name": item["entity_name"],
                    "score": round(score, 4),
                    "entity_type": item.get("entity_type", ""),
                })

        return sorted(candidates, key=lambda x: x["score"], reverse=True)

    def disambiguate_and_merge(
        self,
        entity_name: str,
        entity_type: str,
        threshold: float | None = None,
    ) -> bool:
        """检测并自动合并相似实体。

        如果找到相似度超过阈值的同类型实体，则调用 rag.amerge_entities()
        将相似实体合并到目标实体中。

        Args:
            entity_name: 新实体名称（合并后的目标名称）
            entity_type: 新实体类型
            threshold: 相似度阈值（覆盖默认值）

        Returns:
            True 如果发生了合并，False 如果没有需要合并的候选
        """
        thresh = threshold if threshold is not None else self.threshold

        candidates = self.find_similar_entities(entity_name, entity_type, threshold=thresh, top_k=5)
        if not candidates:
            return False

        # 取最高分的候选进行合并
        best = candidates[0]
        source_entity = best["name"]
        logger.info(
            "实体消歧: '%s' 与已有实体 '%s' 相似度 %.4f (类型: %s)，执行自动合并",
            entity_name, source_entity, best["score"], entity_type,
        )

        try:
            merge_result = _run_async_on_rag(
                self.rag,
                self.rag.amerge_entities(
                    source_entities=[source_entity],
                    target_entity=entity_name,
                ),
            )
            logger.info("实体消歧: 合并成功，'%s' 已并入 '%s'", source_entity, entity_name)

            # 合并成功后同步更新 full_entities 和 full_relations 追踪
            try:
                _run_async_on_rag(
                    self.rag,
                    self._update_tracking_after_merge(source_entity, entity_name),
                )
            except Exception as e:
                logger.warning("实体消歧: 合并追踪更新失败 (%s)，不影响删除正确性", e)

            return True
        except Exception as e:
            logger.warning("实体消歧: 合并失败 (%s)，保留两个独立实体", e)
            return False

    async def _update_tracking_after_merge(self, old_name: str, new_name: str):
        """合并后更新 full_entities 和 full_relations 中的实体名称引用。

        遍历所有文档的 full_entities 条目，将已删除的旧实体名替换为
        合并后的新实体名，确保删除文档时不会出现 stale 引用。

        Args:
            old_name: 被合并的源实体名称（已从图谱删除）
            new_name: 合并后的目标实体名称（保留在图谱中）
        """
        # 获取所有存在 full_entities 条目的文档 ID
        # 通过扫描 entity_chunks 中的 chunk source_ids 来发现相关文档
        # 这里用更简单的方式：遍历 doc_status 中所有文档
        try:
            doc_status_counts = await self.rag.doc_status.get_all_status_counts()
            # get_all_status_counts 返回 {status: count}，不包含 doc_id
            # 需要换一种方式：尝试从 full_entities 的存储中发现所有 doc_id
            # 使用 Redis SCAN 直接扫描 full_entities 命名空间下的所有 key
            await self._rename_entity_in_full_entities(old_name, new_name)
            await self._rename_entity_in_full_relations(old_name, new_name)
        except Exception as e:
            logger.warning("遍历文档状态失败: %s", e)

    async def _rename_entity_in_full_entities(self, old_name: str, new_name: str):
        """将 full_entities 中所有包含旧实体名的条目更新为新实体名。"""
        storage = self.rag.full_entities
        # Redis KV storage 的 key 模式: {final_namespace}:{doc_id}
        # 通过 filter_keys 无法枚举，需要直接访问 Redis
        # 但 BaseKVStorage 不提供 enumerate，我们用另一种方式：
        # 利用 entity_chunks 存储来发现哪些文档引用了旧实体名
        try:
            if self.rag.entity_chunks:
                # 检查旧实体名在 entity_chunks 中是否有记录
                old_chunks = await self.rag.entity_chunks.get_by_id(old_name)
                if old_chunks:
                    # 旧实体仍有 chunk 记录，说明合并时 chunk_ids 已迁移
                    # 这些 chunk 对应的 source_id 就是相关文档
                    # 但 entity_chunks 只存 chunk_ids，不存 doc_id
                    # 所以这里无法直接获取 doc_id
                    pass
        except Exception:
            pass

        # 最可靠的方式：直接在 Redis 中扫描 full_entities 命名空间
        # 由于我们在使用 RedisKVStorage，可以访问底层 Redis
        try:
            redis_conn = self._get_redis_connection()
            if redis_conn is None:
                return

            final_ns = getattr(storage, "final_namespace", "full_entities")
            pattern = f"{final_ns}:*"
            cursor = 0
            while True:
                cursor, keys = await redis_conn.scan(cursor, match=pattern, count=100)
                for key in keys:
                    if isinstance(key, bytes):
                        key = key.decode("utf-8")
                    data_str = await redis_conn.get(key)
                    if not data_str:
                        continue
                    import json
                    data = json.loads(data_str)
                    entity_names = data.get("entity_names", [])
                    if old_name in entity_names:
                        entity_names.remove(old_name)
                        if new_name not in entity_names:
                            entity_names.append(new_name)
                        data["entity_names"] = entity_names
                        data["count"] = len(entity_names)
                        await redis_conn.set(key, json.dumps(data, ensure_ascii=False))
                        # 从 key 中提取 doc_id 用于日志
                        doc_id = key.split(":", 1)[-1] if ":" in key else key
                        logger.info(
                            "full_entities 追踪更新: 文档 %s 的实体 '%s' → '%s'",
                            doc_id, old_name, new_name,
                        )
                if cursor == 0:
                    break
        except Exception as e:
            logger.warning("Redis 扫描更新 full_entities 失败: %s", e)

    async def _rename_entity_in_full_relations(self, old_name: str, new_name: str):
        """将 full_relations 中所有包含旧实体名的关系对更新为新实体名。"""
        storage = self.rag.full_relations
        try:
            redis_conn = self._get_redis_connection()
            if redis_conn is None:
                return

            final_ns = getattr(storage, "final_namespace", "full_relations")
            pattern = f"{final_ns}:*"
            cursor = 0
            while True:
                cursor, keys = await redis_conn.scan(cursor, match=pattern, count=100)
                for key in keys:
                    if isinstance(key, bytes):
                        key = key.decode("utf-8")
                    data_str = await redis_conn.get(key)
                    if not data_str:
                        continue
                    import json
                    data = json.loads(data_str)
                    relation_pairs = data.get("relation_pairs", [])
                    changed = False
                    for pair in relation_pairs:
                        if isinstance(pair, list) and len(pair) >= 2:
                            if pair[0] == old_name:
                                pair[0] = new_name
                                changed = True
                            if pair[1] == old_name:
                                pair[1] = new_name
                                changed = True
                    if changed:
                        data["relation_pairs"] = relation_pairs
                        data["count"] = len(relation_pairs)
                        await redis_conn.set(key, json.dumps(data, ensure_ascii=False))
                        doc_id = key.split(":", 1)[-1] if ":" in key else key
                        logger.info(
                            "full_relations 追踪更新: 文档 %s 的关系中 '%s' → '%s'",
                            doc_id, old_name, new_name,
                        )
                if cursor == 0:
                    break
        except Exception as e:
            logger.warning("Redis 扫描更新 full_relations 失败: %s", e)

    def _get_redis_connection(self):
        """获取 Redis 连接（用于 SCAN 操作）。"""
        try:
            storage = self.rag.full_entities
            # RedisKVStorage 有 _redis 属性
            if hasattr(storage, "_redis"):
                return storage._redis
        except Exception:
            pass
        return None
