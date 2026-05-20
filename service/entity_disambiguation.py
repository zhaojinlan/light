"""实体消歧服务

在知识图谱构建时，对新增实体进行自动相似度检测和合并。
流程：
1. 精确字符串匹配去重（由调用方完成）
2. 对新实体计算 embedding，与已有同类型实体做余弦相似度比较
3. 超过阈值则自动调用 rag.amerge_entities() 合并

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
            return True
        except Exception as e:
            logger.warning("实体消歧: 合并失败 (%s)，保留两个独立实体", e)
            return False
