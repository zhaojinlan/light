"""BM25 混合检索服务

在 Qdrant dense 向量检索基础上，增加 BM25 稀疏向量关键词匹配检索。
通过 RRF（Reciprocal Rank Fusion）融合 dense 和 BM25 的检索结果。

注意：需要 Qdrant 服务端（>=1.13.0）与客户端版本兼容。若版本不匹配，
BM25 索引会自动降级为纯 dense 检索。
"""

import logging
from typing import Any

logger = logging.getLogger(__name__)


class BM25RetrievalService:
    """BM25 混合检索服务。

    为 Qdrant chunks_vdb collection 添加 sparse vector (BM25) 索引，
    提供 dense + BM25 混合检索能力。
    """

    SPARSE_VECTOR_NAME = "bm25"

    def __init__(self, rag):
        """初始化 BM25 检索服务。

        Args:
            rag: LightRAG 实例，需已完成存储初始化
        """
        self.rag = rag
        self._client = None
        self._collection_name = None
        self._sparse_enabled = False  # 标记 sparse 向量是否可用

    def _ensure_client(self):
        """延迟获取 Qdrant client 和 collection name。"""
        if self._client is not None:
            return

        chunks_vdb = self.rag.chunks_vdb
        self._client = chunks_vdb._client
        self._collection_name = chunks_vdb.final_namespace

    def ensure_sparse_index(self):
        """尝试配置 BM25 sparse vector 索引。

        若配置失败（版本不兼容 / collection 不支持），则降级为纯 dense 检索。
        """
        from qdrant_client import models

        self._ensure_client()

        try:
            collection_info = self._client.get_collection(self._collection_name)
            sparse_config = getattr(
                collection_info.config.params, "sparse_vectors_config", None
            )
            if sparse_config and self.SPARSE_VECTOR_NAME in sparse_config:
                self._sparse_enabled = True
                logger.info(
                    "BM25 sparse vector 索引已存在: collection=%s",
                    self._collection_name,
                )
                print(
                    f"  [BM25] sparse vector 索引就绪: '{self._collection_name}'"
                )
                return

            # 尝试添加 sparse vector 配置
            sparse_params = models.SparseVectorParams(
                index=models.SparseIndexParams(on_disk=False),
            )
            self._client.update_collection(
                collection_name=self._collection_name,
                sparse_vectors_config={
                    self.SPARSE_VECTOR_NAME: sparse_params,
                },
            )
            self._sparse_enabled = True
            print(
                f"  [BM25] 已为 '{self._collection_name}' 创建 BM25 sparse vector 索引"
            )
        except Exception as e:
            logger.warning("BM25 sparse vector 索引不可用: %s", e)
            self._sparse_enabled = False
            print(
                f"  [BM25] sparse vector 索引不可用（{e}），"
                f"将使用纯 dense 向量检索"
            )

    def index_chunk_bm25(self, chunk_id: str, text: str):
        """为 chunk 生成并存储 BM25 稀疏向量。

        若 sparse 向量不可用则静默跳过。

        Args:
            chunk_id: chunk 的 ID
            text: chunk 原文内容
        """
        from qdrant_client import models

        self._ensure_client()
        if not self._sparse_enabled:
            return

        sparse_vector = self._build_bm25_sparse(text)
        if sparse_vector is None:
            return

        try:
            self._client.upsert(
                collection_name=self._collection_name,
                points=[
                    models.PointStruct(
                        id=chunk_id,
                        vector={self.SPARSE_VECTOR_NAME: sparse_vector},
                    )
                ],
                wait=False,
            )
        except Exception as e:
            logger.warning("BM25 索引写入失败 (chunk_id=%s): %s", chunk_id, e)

    def _build_bm25_sparse(self, text: str):
        """将文本转换为 BM25 稀疏向量。

        使用字符级 bi-gram 分词。

        Args:
            text: 原文内容

        Returns:
            SparseVector 或 None（文本为空时）
        """
        from qdrant_client import models

        if not text or not text.strip():
            return None

        text_clean = text.strip()
        tokens = [text_clean[i : i + 2] for i in range(len(text_clean) - 1)]

        if not tokens:
            return None

        unique_tokens = sorted(set(tokens))
        token_to_idx = {t: i for i, t in enumerate(unique_tokens)}

        term_freq = {}
        for token in tokens:
            term_freq[token] = term_freq.get(token, 0) + 1

        indices = [token_to_idx[t] for t in unique_tokens]
        values = [float(term_freq[t]) for t in unique_tokens]

        return models.SparseVector(indices=indices, values=values)

    def search(
        self,
        query: str,
        top_k: int = 10,
        dense_weight: float = 0.5,
        bm25_weight: float = 0.5,
    ) -> list[dict[str, Any]]:
        """执行 dense + BM25 混合检索。

        若 BM25 不可用，则退化为纯 dense 检索。

        Args:
            query: 查询文本
            top_k: 返回结果数量
            dense_weight: dense 搜索结果权重
            bm25_weight: BM25 搜索结果权重

        Returns:
            融合后的结果列表
        """
        from qdrant_client import models
        from qdrant_client.hybrid.fusion import reciprocal_rank_fusion

        self._ensure_client()

        # 1. Dense 搜索
        dense_hits = []
        try:
            embedding = _run_async_on_rag(
                self.rag,
                self.rag.embedding_func([query], context="query", _priority=5),
            )
            dense_results = self._client.query_points(
                collection_name=self._collection_name,
                query=embedding[0],
                limit=top_k,
                with_payload=True,
            ).points
            dense_hits = [
                {"id": p.id, "payload": p.payload, "score": p.score}
                for p in dense_results
            ]
        except Exception as e:
            logger.warning("dense 搜索失败: %s", e)

        # 2. BM25 sparse 搜索（仅当 sparse 可用时）
        bm25_hits = []
        if self._sparse_enabled:
            try:
                bm25_sparse = self._build_bm25_sparse(query)
                if bm25_sparse:
                    bm25_results = self._client.query_points(
                        collection_name=self._collection_name,
                        query=bm25_sparse,
                        using=self.SPARSE_VECTOR_NAME,
                        limit=top_k,
                        with_payload=True,
                    ).points
                    bm25_hits = [
                        {"id": p.id, "payload": p.payload, "score": p.score}
                        for p in bm25_results
                    ]
            except Exception as e:
                logger.warning("BM25 sparse 搜索失败: %s", e)

        # 3. 结果融合
        if not dense_hits and not bm25_hits:
            return []
        if not bm25_hits:
            # 无 BM25 结果，直接返回 dense 结果
            results = []
            for p in dense_hits:
                results.append({
                    "id": p["id"],
                    "payload": p["payload"],
                    "content": p["payload"].get("content", ""),
                    "score": p["score"],
                })
            return results[:top_k]
        if not dense_hits:
            return bm25_hits

        # RRF 融合
        dense_ranked = [p["id"] for p in dense_hits]
        bm25_ranked = [p["id"] for p in bm25_hits]

        fused = reciprocal_rank_fusion([dense_ranked, bm25_ranked], k=60)

        id_to_payload = {}
        for p in dense_hits + bm25_hits:
            if p["id"] not in id_to_payload:
                id_to_payload[p["id"]] = p["payload"]

        results = []
        for rank, point_id in enumerate(fused):
            payload = id_to_payload.get(point_id, {})
            results.append({
                "id": point_id,
                "payload": payload,
                "content": payload.get("content", ""),
                "rrf_score": 1.0 / (60 + rank + 1),
                "rank": rank + 1,
            })

        return results[:top_k]


def _run_async_on_rag(rag, coro):
    """在 rag 的持久事件循环或临时循环上运行协程。"""
    import asyncio

    if hasattr(rag, "_persistent_loop") and rag._persistent_loop is not None:
        return rag._persistent_loop.run_coroutine(coro)
    loop = asyncio.new_event_loop()
    try:
        return loop.run_until_complete(coro)
    finally:
        loop.close()
