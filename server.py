"""LightRAG HTTP 服务入口

提供文档分析接口，供主项目通过 HTTP 调用。
"""

import asyncio
import logging
import sys
import uuid
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

logger = logging.getLogger("lightserver")
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)

# 确保 service 包可导入
sys.path.insert(0, str(Path(__file__).parent))

# ============================================================
# Task 状态跟踪（内存）
# ============================================================

tasks: dict[str, dict] = {}


class TaskStatus:
    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"


# ============================================================
# Request / Response Models
# ============================================================


class AnalyzeRequest(BaseModel):
    text: str
    doc_id: str
    file_path: str = ""


class AnalyzeResponse(BaseModel):
    task_id: str


class StatusResponse(BaseModel):
    status: str
    result: dict | None = None
    error: str | None = None


class SetIsanalysisRequest(BaseModel):
    doc_id: str
    isanalysis: bool = True


# ============================================================
# FastAPI 应用
# ============================================================


@asynccontextmanager
async def lifespan(app: FastAPI):
    """启动时创建 LightRAG 实例，关闭时释放资源。"""
    from service.kg_config import create_lightrag_neo4j_qdrant

    logger.info("Initializing LightRAG (Neo4j + Qdrant + MongoDB)...")
    rag = create_lightrag_neo4j_qdrant()
    app.state.rag = rag
    logger.info("LightRAG initialized successfully.")
    yield
    # 关闭时释放资源
    if hasattr(rag, "_persistent_loop"):
        rag._persistent_loop.close()
        logger.info("LightRAG resources released.")


app = FastAPI(title="LightRAG Server", version="0.1.0", lifespan=lifespan)


@app.get("/health")
async def health():
    return {"status": "ok"}


@app.post("/api/v1/analyze", response_model=AnalyzeResponse)
async def analyze(req: AnalyzeRequest):
    """提交文档到 LightRAG 进行知识图谱构建。"""
    task_id = str(uuid.uuid4())
    tasks[task_id] = {
        "status": TaskStatus.PENDING,
        "result": None,
        "error": None,
    }

    # 在后台线程执行耗时的 insert 操作
    asyncio.create_task(_run_insert(task_id, req.text, req.doc_id, req.file_path))

    return AnalyzeResponse(task_id=task_id)


@app.get("/api/v1/status/{task_id}", response_model=StatusResponse)
async def get_status(task_id: str):
    """查询文档处理任务状态。"""
    if task_id not in tasks:
        raise HTTPException(status_code=404, detail=f"Task {task_id} not found")

    task = tasks[task_id]
    return StatusResponse(
        status=task["status"],
        result=task["result"],
        error=task["error"],
    )


async def _run_insert(task_id: str, text: str, doc_id: str, file_path: str):
    """在后台执行 LightRAG insert 操作。"""
    try:
        tasks[task_id]["status"] = TaskStatus.PROCESSING

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
        result = {
            "entity_count": insert_result.get("entity_count", 0),
            "relation_count": insert_result.get("relation_count", 0),
            "track_id": track_id,
        }

        tasks[task_id]["status"] = TaskStatus.COMPLETED
        tasks[task_id]["result"] = result
        logger.info("Task %s completed: track_id=%s, entities=%d, relations=%d",
                    task_id, track_id, result["entity_count"], result["relation_count"])

    except Exception as e:
        logger.error("Task %s failed: %s", task_id, e, exc_info=True)
        tasks[task_id]["status"] = TaskStatus.FAILED
        tasks[task_id]["error"] = str(e)


@app.post("/api/v1/doc/set-isanalysis")
async def set_doc_isanalysis(req: SetIsanalysisRequest):
    """设置 MongoDB doc_status 中某条文档记录的 isanalysis 字段。"""
    rag = app.state.rag
    try:
        # MongoDB client 绑定在 _PersistentLoop 线程上，
        # 需要将协程提交到该循环执行，避免事件循环冲突
        if hasattr(rag, "_persistent_loop") and rag._persistent_loop is not None:
            ploop = rag._persistent_loop
            future = asyncio.run_coroutine_threadsafe(
                rag.doc_status.upsert({req.doc_id: {"isanalysis": req.isanalysis}}),
                ploop._loop,
            )
            await asyncio.wrap_future(future)
        else:
            await rag.doc_status.upsert({req.doc_id: {"isanalysis": req.isanalysis}})
        logger.info("Set isanalysis=%s for doc_id=%s", req.isanalysis, req.doc_id)
        return {"status": "ok"}
    except Exception as e:
        logger.error("Failed to set isanalysis for %s: %s", req.doc_id, e)
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
                "label": n.label,
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


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("server:app", host="0.0.0.0", port=8001, reload=False)
