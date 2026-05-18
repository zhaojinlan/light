"""自定义直接入库函数

绕过 ainsert_custom_kg，直接调用各存储层 API：
- Qdrant entities_vdb: 实体向量（语义搜索）
- Qdrant chunks_vdb: chunk 向量（检索上下文）
- JSON KV (text_chunks, full_entities, full_relations, doc_status): 元数据与文档追踪
- Neo4j (Neo4jDirectWriter): 实体节点（双 label）+ 有向关系边
- 不写入 relationships_vdb（关系不向量化）
"""

import asyncio
import os
import time
from typing import Any

from lightrag import LightRAG
from lightrag.base import DocStatus
from lightrag.utils import compute_mdhash_id


async def a_direct_ingest(
    rag: LightRAG,
    text: str,
    entities: list[dict[str, Any]],
    relationships: list[dict[str, Any]],
    doc_id: str | None = None,
    file_path: str = "custom_kg",
) -> dict[str, Any]:
    """异步直接入库所有存储层。

    Args:
        rag: LightRAG 实例（存储已初始化）
        text: 原始文本内容
        entities: LLM 抽取的实体列表（raw 格式）
        relationships: LLM 抽取的关系列表（raw 格式）
        doc_id: 文档 ID，若不提供则自动生成
        file_path: 文件路径标识

    Returns:
        包含写入统计的字典
    """
    source_id = doc_id or f"doc-{compute_mdhash_id(text)[:32]}"

    stats = {"entities": 0, "relations": 0, "chunks": 0}

    # ================================================================
    # 1. 处理 chunk → chunks_vdb + text_chunks
    # ================================================================
    chunk_content = text.strip()
    chunk_id = compute_mdhash_id(chunk_content, prefix="chunk-")
    tokens = len(rag.tokenizer.encode(chunk_content)) if hasattr(rag, "tokenizer") and rag.tokenizer else len(chunk_content) // 2

    chunk_entry = {
        "content": chunk_content,
        "source_id": source_id,
        "tokens": tokens,
        "chunk_order_index": 0,
        "full_doc_id": doc_id if doc_id is not None else source_id,
        "file_path": file_path,
        "status": DocStatus.PROCESSED,
    }
    chunks_data = {chunk_id: chunk_entry}

    if chunks_data:
        await asyncio.gather(
            rag.chunks_vdb.upsert(chunks_data),
            rag.text_chunks.upsert(chunks_data),
        )
        stats["chunks"] = len(chunks_data)

    chunk_to_source_map = {source_id: chunk_id}

    # ================================================================
    # 1.5 为 chunk 生成 BM25 稀疏向量索引（关键词检索）
    # ================================================================
    from .bm25_retrieval import BM25RetrievalService

    bm25_service = BM25RetrievalService(rag)
    bm25_service.ensure_sparse_index()
    bm25_service.index_chunk_bm25(chunk_id, chunk_content)

    # ================================================================
    # 2. 处理 entity → entities_vdb（Qdrant 向量）
    # ================================================================
    deduped_entities: dict[str, dict[str, Any]] = {}
    for entity_data in entities:
        entity_name = entity_data.get("normalized_name", "").strip()
        if entity_name:
            deduped_entities.pop(entity_name, None)
            deduped_entities[entity_name] = entity_data

    data_for_entities_vdb = {}
    entity_names_for_tracking = []
    for entity_name, entity_data in deduped_entities.items():
        entity_type = entity_data.get("entity_type", "UNKNOWN")
        description = (entity_data.get("definition") or entity_data.get("entity_text") or entity_name).strip()
        source_chunk_id = entity_data.get("source_id", source_id)
        resolved_source_id = chunk_to_source_map.get(source_chunk_id, source_id)

        ent_vdb_id = compute_mdhash_id(entity_name, prefix="ent-")
        data_for_entities_vdb[ent_vdb_id] = {
            "content": f"{entity_name}\n{description}",
            "entity_name": entity_name,
            "source_id": resolved_source_id,
            "description": description,
            "entity_type": entity_type,
            "file_path": file_path,
        }
        entity_names_for_tracking.append(entity_name)

    if data_for_entities_vdb:
        await rag.entities_vdb.upsert(data_for_entities_vdb)
        stats["entities"] = len(data_for_entities_vdb)

    # ================================================================
    # 3. 处理 relationship → 仅 Neo4j，不写入 relationships_vdb
    # ================================================================
    deduped_relations: dict[tuple[str, str], dict[str, Any]] = {}
    for rel_data in relationships:
        src = rel_data.get("src_entity", "").strip()
        tgt = rel_data.get("tgt_entity", "").strip()
        if src and tgt:
            rel_key = tuple(sorted((src, tgt)))
            deduped_relations.pop(rel_key, None)
            deduped_relations[rel_key] = rel_data

    relation_pairs_for_tracking = []
    for (src, tgt), _ in deduped_relations.items():
        relation_pairs_for_tracking.append([src, tgt])

    # ================================================================
    # 4. Neo4j → 实体节点（双 label）+ 有向关系边
    # ================================================================
    from .neo4j_writer import Neo4jDirectWriter

    with Neo4jDirectWriter(
        uri=os.getenv("NEO4J_URI", "bolt://localhost:7687"),
        username=os.getenv("NEO4J_USERNAME", "neo4j"),
        password=os.getenv("NEO4J_PASSWORD", "LightRAG2026!"),
    ) as writer:
        neo4j_entity_count = writer.upsert_entities(entities)
        neo4j_rel_count = writer.upsert_relations(relationships)
        stats["entities"] = neo4j_entity_count
        stats["relations"] = neo4j_rel_count
        cleanup_stats = writer.cleanup_wrong_labels()
        if cleanup_stats["directed_edges_deleted"] > 0:
            pass  # silent cleanup

    # ================================================================
    # 5. full_entities + full_relations → 文档追踪（用于删除）
    # ================================================================
    if entity_names_for_tracking:
        await rag.full_entities.upsert({
            source_id: {
                "entity_names": entity_names_for_tracking,
                "count": len(entity_names_for_tracking),
            }
        })

    if relation_pairs_for_tracking:
        await rag.full_relations.upsert({
            source_id: {
                "relation_pairs": relation_pairs_for_tracking,
                "count": len(relation_pairs_for_tracking),
            }
        })

    # ================================================================
    # 6. doc_status → 标记 PROCESSED（必须包含 chunks_list，用于 adelete_by_doc_id）
    # ================================================================
    await rag.doc_status.upsert({
        source_id: {
            "content_summary": (text[:100] + "...") if len(text) > 100 else text,
            "content_length": len(text),
            "status": DocStatus.PROCESSED,
            "created_at": time.time(),
            "updated_at": time.time(),
            "track_id": source_id,
            "chunks_list": list(chunks_data.keys()),
        }
    })

    # ================================================================
    # 7. 持久化
    # ================================================================
    await rag._insert_done()

    return stats


def direct_ingest(
    rag: LightRAG,
    text: str,
    entities: list[dict[str, Any]],
    relationships: list[dict[str, Any]],
    doc_id: str | None = None,
    file_path: str = "custom_kg",
) -> dict[str, Any]:
    """同步版本，通过 rag 的持久事件循环调用异步入库。

    如果 rag 有 _persistent_loop（kg_config 创建），使用它；
    否则创建临时事件循环。
    """
    if hasattr(rag, "_persistent_loop") and rag._persistent_loop is not None:
        return rag._persistent_loop.run_coroutine(
            a_direct_ingest(rag, text, entities, relationships, doc_id, file_path)
        )

    loop = asyncio.new_event_loop()
    try:
        return loop.run_until_complete(
            a_direct_ingest(rag, text, entities, relationships, doc_id, file_path)
        )
    finally:
        loop.close()
