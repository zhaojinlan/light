"""LightRAG HTTP 服务入口

提供文档分析接口，供主项目通过 HTTP 调用。
处理完成后通过 Redis Stream 异步回调结果。
"""

import asyncio
import logging
import os
import sys
import uuid
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from redis.asyncio import Redis

logger = logging.getLogger("lightserver")
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)

# 确保 service 包可导入
sys.path.insert(0, str(Path(__file__).parent))

# ============================================================
# Redis Stream 配置
# ============================================================

REDIS_DOC_STREAM = os.getenv("REDIS_DOC_STREAM", "doc:result")
REDIS_STREAM_MAXLEN = 10000  # 防止 Stream 无限增长
DOC_IMPORT_CONCURRENCY = 5  # 同时处理文档的上限

_redis: Redis | None = None
_doc_sem = asyncio.Semaphore(DOC_IMPORT_CONCURRENCY)


async def _get_redis() -> Redis:
    """获取 Redis 连接（延迟初始化）。"""
    global _redis
    if _redis is None:
        _redis = Redis.from_url(os.getenv("REDIS_URI", "redis://localhost:6379/0"), decode_responses=True)
    return _redis


# ============================================================
# Request / Response Models
# ============================================================


class AnalyzeRequest(BaseModel):
    text: str
    doc_id: str
    file_path: str = ""


class AnalyzeResponse(BaseModel):
    task_id: str


class SetIsanalysisRequest(BaseModel):
    doc_id: str
    isanalysis: bool = True


class RetrievalQueryRequest(BaseModel):
    question: str
    mode: str = "mix"
    response_type: str = "Multiple Paragraphs"
    top_k: int = 10


class RetrievalDataRequest(BaseModel):
    question: str
    mode: str = "mix"
    top_k: int = 10


class BM25RetrievalRequest(BaseModel):
    question: str
    top_k: int = 10
    dense_weight: float = 0.5
    bm25_weight: float = 0.5


class PreventionRequest(BaseModel):
    query: str | None = None
    top_k: int = 20


# ============================================================
# FastAPI 应用
# ============================================================


@asynccontextmanager
async def lifespan(app: FastAPI):
    """启动时创建 LightRAG 实例，关闭时释放资源。"""
    from service.kg_config import create_lightrag_neo4j_qdrant

    logger.info("Initializing LightRAG (Neo4j + Qdrant + Redis)...")
    rag = create_lightrag_neo4j_qdrant()
    app.state.rag = rag
    logger.info("LightRAG initialized successfully.")
    yield
    # 关闭时释放资源
    if hasattr(rag, "_persistent_loop"):
        rag._persistent_loop.close()
        logger.info("LightRAG resources released.")
    if _redis is not None:
        await _redis.aclose()
        logger.info("Redis connection closed.")


app = FastAPI(title="LightRAG Server", version="0.1.0", lifespan=lifespan)


@app.get("/health")
async def health():
    return {"status": "ok"}


@app.post("/api/v1/analyze", response_model=AnalyzeResponse)
async def analyze(req: AnalyzeRequest):
    """提交文档到 LightRAG 进行知识图谱构建。

    处理完成后通过 Redis Stream "doc:result" 异步回调结果。
    并发数受限于 DOC_IMPORT_CONCURRENCY（默认 5），超限时返回 503。
    """
    if _doc_sem.locked():
        raise HTTPException(status_code=503, detail="文档处理队列已满，请稍后重试")

    task_id = str(uuid.uuid4())

    # 在后台执行耗时的 insert 操作
    asyncio.create_task(_run_insert(task_id, req.text, req.doc_id, req.file_path))

    return AnalyzeResponse(task_id=task_id)


async def _push_to_redis_stream(fields: dict, max_retries: int = 3) -> None:
    """推送结果到 Redis Stream，失败时指数退避重试。

    Args:
        fields: 要推送的字段字典
        max_retries: 最大重试次数，默认 3 次

    Raises:
        Exception: 所有重试均失败时抛出最后一次的异常
    """
    last_error = None
    for attempt in range(max_retries):
        try:
            redis = await _get_redis()
            await redis.xadd(REDIS_DOC_STREAM, fields, maxlen=REDIS_STREAM_MAXLEN, approximate=True)
            return
        except Exception as e:
            last_error = e
            if attempt < max_retries - 1:
                delay = 2 ** attempt
                logger.warning("Redis 推送失败(第%d次), %ds 后重试: %s", attempt + 1, delay, e)
                await asyncio.sleep(delay)
    logger.critical("Redis 推送结果失败(重试%d次): %s", max_retries, last_error)
    raise last_error


async def _run_insert(task_id: str, text: str, doc_id: str, file_path: str):
    """在后台执行 LightRAG insert 操作，完成后通过 Redis Stream 推送结果。"""
    async with _doc_sem:
        try:
            def _insert_sync():
                from service.custom_entity_service import CustomEntityService

                svc = CustomEntityService(rag=app.state.rag)
                result = svc.insert_with_custom_schema(
                    text=text,
                    doc_id=doc_id,
                    file_path=file_path if file_path else "custom_kg",
                )
                return {
                    "track_id": result.get("doc_id", doc_id),
                    "entity_count": len(result.get("entities", [])),
                    "relation_count": len(result.get("relationships", [])),
                }

            # 在线程池中执行同步的 insert
            insert_result = await asyncio.to_thread(_insert_sync)

            track_id = insert_result.get("track_id", doc_id)
            entity_count = insert_result.get("entity_count", 0)
            relation_count = insert_result.get("relation_count", 0)

            logger.info("Task %s completed: track_id=%s, entities=%d, relations=%d",
                        task_id, track_id, entity_count, relation_count)

            # 推送结果到 Redis Stream（带重试）
            await _push_to_redis_stream({
                "task_id": task_id,
                "status": "completed",
                "entity_count": str(entity_count),
                "relation_count": str(relation_count),
            })
            logger.info("Result pushed to Redis Stream '%s'", REDIS_DOC_STREAM)

        except Exception as e:
            logger.error("Task %s failed: %s", task_id, e, exc_info=True)

            # 推送失败结果到 Redis Stream（带重试）
            try:
                await _push_to_redis_stream({
                    "task_id": task_id,
                    "status": "failed",
                    "error": str(e),
                })
            except Exception as redis_err:
                logger.critical("无法推送失败结果到 Redis，文档 %s 将永久卡在 processing: %s", task_id, redis_err)


@app.post("/api/v1/doc/set-isanalysis")
async def set_doc_isanalysis(req: SetIsanalysisRequest):
    """设置 doc_status 中某条文档记录的 isanalysis 字段（Redis KV 存储）。"""
    rag = app.state.rag
    try:
        ploop = rag._persistent_loop
        future = asyncio.run_coroutine_threadsafe(
            rag.doc_status.upsert({req.doc_id: {"isanalysis": req.isanalysis}}),
            ploop._loop,
        )
        await asyncio.wrap_future(future)
        logger.info("Set isanalysis=%s for doc_id=%s", req.isanalysis, req.doc_id)
        return {"status": "ok"}
    except Exception as e:
        logger.error("Failed to set isanalysis for %s: %s", req.doc_id, e)
        raise HTTPException(status_code=500, detail=str(e))


# ============================================================
# 知识图谱删除端点
# ============================================================


@app.delete("/api/v1/kg/doc/{doc_id}")
async def delete_kg_doc(doc_id: str):
    """删除文档关联的所有知识图谱数据（Neo4j + Qdrant + Redis）。

    由主项目 app 的 DELETE /api/v1/documents/{id} 端点调用，
    确保文档删除时知识图谱数据也被清理。
    """
    from service.custom_entity_service import CustomEntityService

    rag = app.state.rag
    svc = CustomEntityService(rag=rag)
    try:
        result = svc.delete_by_doc_id(doc_id)
        status = result.get("status", "ok")
        if status == "not_found":
            return {
                "status": "ok",
                "doc_id": doc_id,
                "detail": "doc_status not found, skipping KG cleanup",
            }
        return {"status": "ok", "doc_id": doc_id, "detail": result}
    except Exception as e:
        error_msg = str(e).lower()
        if "not found" in error_msg or "not_found" in error_msg:
            return {
                "status": "ok",
                "doc_id": doc_id,
                "detail": "not found in KG, skipping",
            }
        logger.error("KG delete failed for doc %s: %s", doc_id, e, exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


# ============================================================
# 知识图谱查询端点
# ============================================================


@app.get("/api/v1/kg/graph")
async def get_kg_graph(
    entity_name: str = "*",
    max_depth: int = 3,
    max_nodes: int = 500,
):
    """获取知识图谱子图（nodes/edges）。"""
    from service.query_service import QueryService

    svc = QueryService(rag=app.state.rag)
    try:
        graph = svc.get_knowledge_graph(entity_name, max_depth, max_nodes)
        nodes = [
            {
                "id": n.id,
                "label": n.properties.get("entity_name", n.id),
                "entity_type": n.properties.get("entity_type", ""),
                "description": n.properties.get("description", ""),
            }
            for n in graph.nodes
        ]
        edges = [
            {
                "source": e.source,
                "target": e.target,
                "relation_type": e.type,
                "description": e.properties.get("description", ""),
            }
            for e in graph.edges
        ]
        return {"nodes": nodes, "edges": edges, "total_nodes": len(nodes), "total_edges": len(edges)}
    except Exception as e:
        logger.error("KG graph error: %s", e, exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/v1/kg/entity/{entity_name}")
async def get_kg_entity(entity_name: str):
    """获取单个实体的详细信息。"""
    from service.query_service import QueryService

    svc = QueryService(rag=app.state.rag)
    try:
        info = svc.get_entity_info(entity_name)
        if not info:
            raise HTTPException(status_code=404, detail=f"Entity '{entity_name}' not found")
        return info
    except HTTPException:
        raise
    except Exception as e:
        logger.error("KG entity error for %s: %s", entity_name, e, exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/v1/kg/labels")
async def get_kg_labels():
    """获取知识图谱所有实体标签。"""
    from service.query_service import QueryService

    svc = QueryService(rag=app.state.rag)
    try:
        return {"labels": svc.get_graph_labels()}
    except Exception as e:
        logger.error("KG labels error: %s", e, exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/v1/kg/graph/by-doc/{doc_id}")
async def get_kg_graph_by_doc(doc_id: str, depth: int = 2):
    """按文档 ID 获取该文档对应的知识图谱子图。

    先找到该文档的 summary 实体，再展开 N 跳邻居。
    """
    from service.query_service import QueryService

    svc = QueryService(rag=app.state.rag)
    try:
        # 找到该文档的 summary 实体
        summaries = svc.query_case_summary(doc_id)
        if not summaries:
            return {"center": None, "neighbors": [], "edges": [], "message": f"No summary found for doc_id={doc_id}"}

        summary_name = summaries[0]["entity_name"]

        # 展开 N 跳邻居
        result = svc.get_entity_neighbors(summary_name, depth=depth)
        if not result:
            return {"center": summary_name, "neighbors": [], "edges": []}

        return {
            "center": result["center"],
            "neighbors": result.get("neighbors", []),
            "edges": result.get("edges", []),
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error("KG by-doc error for %s: %s", doc_id, e, exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/v1/kg/search")
async def kg_search(q: str, top_k: int = 10):
    """全文搜索：BM25 混合检索，返回 chunk、summary、related_cases。"""
    from service.query_service import QueryService

    svc = QueryService(rag=app.state.rag)
    try:
        result = svc.query_similar_cases(q, top_k=top_k)
        return result
    except Exception as e:
        logger.error("KG search error: %s", e, exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


# ============================================================
# 检索端点 — RAG问答 / 向量检索 / BM25混合 / 防范建议
# ============================================================


@app.post("/api/v1/retrieval/query")
async def retrieval_query(req: RetrievalQueryRequest):
    """RAG 知识图谱问答（经过 LLM 生成）。

    支持 5 种检索模式：
      - local: 局部实体上下文检索
      - global: 全局知识检索
      - hybrid: local + global
      - naive: 简单向量搜索
      - mix: KG + 向量混合（推荐）
    """
    from service.query_service import QueryService

    svc = QueryService(rag=app.state.rag)
    try:
        answer = svc.query(
            question=req.question,
            mode=req.mode,
            response_type=req.response_type,
            top_k=req.top_k,
        )
        return {
            "answer": answer,
            "question": req.question,
            "mode": req.mode,
        }
    except Exception as e:
        logger.error("Retrieval query error: %s", e, exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/v1/retrieval/query_data")
async def retrieval_query_data(req: RetrievalDataRequest):
    """纯数据检索，不经过 LLM 生成。

    返回结构化结果：entities / relationships / chunks / references。
    """
    from service.query_service import QueryService

    svc = QueryService(rag=app.state.rag)
    try:
        result = svc.query_data(
            question=req.question,
            mode=req.mode,
            top_k=req.top_k,
        )
        # LightRAG may wrap the response as {"status": "...", "data": {...}}
        # Unwrap to return the actual data at top level
        if isinstance(result, dict) and "data" in result:
            return result["data"]
        return result
    except Exception as e:
        logger.error("Retrieval query_data error: %s", e, exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/v1/retrieval/bm25")
async def retrieval_bm25(req: BM25RetrievalRequest):
    """Dense 向量 + BM25 稀疏向量混合检索（RRF 融合）。

    不经过 LLM，返回原始 chunk 列表及其 RRF 融合分数。
    """
    from service.query_service import QueryService

    svc = QueryService(rag=app.state.rag)
    try:
        result = svc.query_with_bm25(
            question=req.question,
            top_k=req.top_k,
            dense_weight=req.dense_weight,
            bm25_weight=req.bm25_weight,
        )
        return result
    except Exception as e:
        logger.error("BM25 retrieval error: %s", e, exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/v1/retrieval/by-type/{entity_type}")
async def retrieval_by_type(entity_type: str, top_k: int = 50):
    """按实体类型从向量库检索。

    类型: FraudScenario / FraudFeature / FraudMethod /
          PreventionMeasure / LawRegulation / RelatedCase / summary
    """
    from service.query_service import QueryService

    svc = QueryService(rag=app.state.rag)
    try:
        items = svc.query_by_entity_type(entity_type, top_k=top_k)
        return {"items": items, "entity_type": entity_type, "count": len(items)}
    except Exception as e:
        logger.error("Retrieval by-type error: %s", e, exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/v1/retrieval/prevention")
async def retrieval_prevention(req: PreventionRequest):
    """防范措施检索。

    先查向量库获取所有 PreventionMeasure 实体，
    如有 query 则用 BM25 混合检索按相关性排序。
    """
    from service.query_service import QueryService

    svc = QueryService(rag=app.state.rag)
    try:
        items = svc.query_prevention(query=req.query, top_k=req.top_k)
        return {"items": items, "query": req.query, "count": len(items)}
    except Exception as e:
        logger.error("Prevention retrieval error: %s", e, exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


# ============================================================
# 图遍历端点 — 邻居探索 / 最短路径
# ============================================================


@app.get("/api/v1/kg/entity/{entity_name}/neighbors")
async def get_kg_entity_neighbors(entity_name: str, depth: int = 1):
    """获取实体的 N 跳邻居节点（用于多跳图遍历推理）。

    Args:
        entity_name: 实体名称（精确匹配）
        depth: 搜索深度（跳数），1=直接邻居，2=邻居的邻居。默认 1
    """
    from service.query_service import QueryService

    svc = QueryService(rag=app.state.rag)
    try:
        if not (1 <= depth <= 4):
            raise HTTPException(status_code=400, detail="depth must be between 1 and 4")
        result = svc.get_entity_neighbors(entity_name, depth=depth)
        if not result:
            raise HTTPException(status_code=404, detail=f"Entity '{entity_name}' not found")
        return result
    except HTTPException:
        raise
    except Exception as e:
        logger.error("KG neighbors error for %s: %s", entity_name, e, exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/v1/kg/shortest-path")
async def get_kg_shortest_path(src: str, tgt: str, max_depth: int = 4):
    """查找两个实体之间的最短关系路径（用于关联链路推理）。

    Args:
        src: 起始实体名称
        tgt: 目标实体名称
        max_depth: 最大搜索深度（边数），默认 4
    """
    from service.query_service import QueryService

    svc = QueryService(rag=app.state.rag)
    try:
        if not (1 <= max_depth <= 6):
            raise HTTPException(status_code=400, detail="max_depth must be between 1 and 6")
        result = svc.get_shortest_path(src, tgt, max_depth=max_depth)
        if not result:
            return {"nodes": [], "edges": [], "message": f"No path found between '{src}' and '{tgt}'"}
        return result
    except HTTPException:
        raise
    except Exception as e:
        logger.error("KG shortest-path error: %s", e, exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("server:app", host="0.0.0.0", port=8001, reload=False)
