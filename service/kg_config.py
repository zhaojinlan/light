"""Neo4j + Qdrant 存储配置模块

构建使用 Neo4j（图存储）和 Qdrant（向量存储）的 LightRAG 实例。
环境变量从 docker.env 或 .env 文件读取。

注意：Neo4j 和 Qdrant 等异步驱动的连接绑定到创建时的事件循环。
为避免每次同步调用创建新事件循环导致连接失效，本模块维护一个
持久运行的事件循环线程，所有同步操作都复用该循环。
"""

import asyncio
import os
import threading
from functools import partial
from typing import Any

from dotenv import load_dotenv
from lightrag import LightRAG
from lightrag.utils import EmbeddingFunc
from lightrag.llm.openai import openai_complete, openai_embed
from lightrag.rerank import generic_rerank_api


def load_env(env_path: str = "docker.env") -> None:
    """加载环境变量配置文件。

    优先使用 docker.env，若不存在则回退到 .env。
    环境变量不会被已有的系统环境变量覆盖。

    Args:
        env_path: 环境变量文件路径
    """
    # 尝试加载 docker.env
    docker_env = os.path.join(os.path.dirname(os.path.dirname(__file__)), env_path)
    if os.path.exists(docker_env):
        load_dotenv(docker_env, override=False)
    else:
        load_dotenv(override=False)


def build_llm_func(
    api_key: str | None = None,
    base_url: str | None = None,
    model: str | None = None,
) -> partial:
    """构建 LLM 调用函数。

    Args:
        api_key: LLM API 密钥
        base_url: LLM API 基础 URL
        model: LLM 模型名称

    Returns:
        已绑定配置的 openai_complete 函数
    """
    return partial(
        openai_complete,
        base_url=base_url or os.getenv("LLM_BASE_URL"),
        api_key=api_key or os.getenv("LLM_API_KEY"),
    )


def build_embedding_func(
    api_key: str | None = None,
    base_url: str | None = None,
    model: str | None = None,
) -> EmbeddingFunc:
    """构建 Embedding 函数。

    Args:
        api_key: Embedding API 密钥
        base_url: Embedding API 基础 URL
        model: Embedding 模型名称

    Returns:
        EmbeddingFunc 实例
    """
    return EmbeddingFunc(
        embedding_dim=1024,  # bge-m3 的向量维度
        func=partial(
            openai_embed.func,  # 使用 .func 避免双重包装
            model=model or os.getenv("EMBEDDING_MODEL", "bge-m3"),
            base_url=base_url or os.getenv("EMBEDDING_BASE_URL"),
            api_key=api_key or os.getenv("EMBEDDING_API_KEY"),
        ),
        max_token_size=8192,
        model_name=model or os.getenv("EMBEDDING_MODEL", "bge-m3"),
    )


def build_rerank_func(
    api_key: str | None = None,
    base_url: str | None = None,
    model: str | None = None,
) -> partial:
    """构建 Rerank 函数。

    Args:
        api_key: Rerank API 密钥
        base_url: Rerank API 基础 URL
        model: Rerank 模型名称

    Returns:
        已绑定配置的 generic_rerank_api 函数
    """
    return partial(
        generic_rerank_api,
        model=model or os.getenv("RERANK_MODEL", "bge-reranker-v2-m3"),
        base_url=base_url or os.getenv("RERANK_BASE_URL"),
        api_key=api_key or os.getenv("RERANK_API_KEY"),
    )


class _PersistentLoop:
    """持久事件循环管理器。

    在后台线程中运行一个永不关闭的事件循环，
    所有同步操作都通过 run_coroutine_threadsafe 在该循环上执行。
    这避免了每次创建新事件循环导致异步驱动（Neo4j）连接失效的问题。
    """

    def __init__(self):
        self._loop: asyncio.AbstractEventLoop | None = None
        self._thread: threading.Thread | None = None

    def start(self) -> asyncio.AbstractEventLoop:
        """启动后台事件循环线程。

        Returns:
            新创建的事件循环
        """
        if self._loop is not None and self._loop.is_running():
            return self._loop

        self._loop = asyncio.new_event_loop()
        self._thread = threading.Thread(
            target=self._run_loop, daemon=True, name="lightrag-event-loop"
        )
        self._thread.start()
        return self._loop

    def _run_loop(self):
        """在后台线程中运行事件循环。"""
        asyncio.set_event_loop(self._loop)
        self._loop.run_forever()

    def run_coroutine(self, coro):
        """在持久事件循环上运行协程并等待结果。

        Args:
            coro: 要运行的协程

        Returns:
            协程的返回值
        """
        if self._loop is None or not self._loop.is_running():
            self.start()
        future = asyncio.run_coroutine_threadsafe(coro, self._loop)
        return future.result()

    def close(self):
        """停止事件循环并清理资源。"""
        if self._loop is not None and self._loop.is_running():
            # Cancel all pending tasks before stopping
            pending = asyncio.all_tasks(self._loop)
            for task in pending:
                task.cancel()
            # Give tasks a moment to finish cancellation
            import concurrent.futures
            def _cancel_and_stop():
                for task in pending:
                    task.cancel()
                self._loop.stop()
            self._loop.call_soon_threadsafe(_cancel_and_stop)
            self._thread.join(timeout=10)
            if self._loop.is_running():
                self._loop.close()
        self._loop = None
        self._thread = None


def _patch_sync_methods(rag: LightRAG, ploop: _PersistentLoop):
    """替换 LightRAG 的同步方法，使其使用持久事件循环。

    LightRAG 的同步方法（如 insert）内部通过 always_get_an_event_loop()
    创建新事件循环，导致 Neo4j 等异步驱动连接失效。
    本函数将它们替换为直接在持久循环上运行对应异步方法的版本。

    处理方法分为两类：
    1. 已经是协程的方法（如 get_graph_labels）— 直接在持久循环上运行
    2. 同步方法有异步对应（如 insert→ainsert）— 调用异步版本
    """
    import inspect

    def run_on_ploop(async_coro):
        """运行协程并等待结果。"""
        return ploop.run_coroutine(async_coro)

    # 第一类：已经是协程的方法，直接在持久循环上包装
    coroutine_methods = [
        "get_graph_labels",
        "get_knowledge_graph",
        "get_entity_info",
        "get_relation_info",
        "get_processing_status",
        "initialize_storages",
        "finalize_storages",
    ]

    for method_name in coroutine_methods:
        if hasattr(rag, method_name):
            original_async = getattr(rag, method_name)
            if inspect.iscoroutinefunction(original_async):

                def make_wrapper(orig):
                    def wrapper(*args, **kwargs):
                        return run_on_ploop(orig(*args, **kwargs))
                    return wrapper

                setattr(rag, method_name, make_wrapper(original_async))

    # 第二类：同步方法，有对应的异步版本（a前缀）
    sync_to_async_map = {
        "insert": "ainsert",
        "query": "aquery",
        "query_data": "aquery_data",
        "delete_by_doc_id": "adelete_by_doc_id",
        "delete_by_entity": "adelete_by_entity",
        "delete_by_relation": "adelete_by_relation",
        "insert_custom_kg": "ainsert_custom_kg",
        "export_data": "aexport_data",
        "get_docs_by_status": "aget_docs_by_status",
    }

    for sync_name, async_name in sync_to_async_map.items():
        if hasattr(rag, async_name):
            async_method = getattr(rag, async_name)

            def make_wrapper2(async_m):
                def wrapper(*args, **kwargs):
                    return run_on_ploop(async_m(*args, **kwargs))
                return wrapper

            setattr(rag, sync_name, make_wrapper2(async_method))

    # 保存关闭方法供外部调用
    rag._persistent_loop = ploop


def create_lightrag_neo4j_qdrant(
    working_dir: str = "./rag_storage",
    env_path: str = "docker.env",
    workspace: str = "",
    **extra_kwargs: Any,
) -> LightRAG:
    """创建使用 Neo4j 图存储 + Qdrant 向量存储的 LightRAG 实例。

    该函数会：
    1. 加载环境变量（从 docker.env 或 .env）
    2. 构建 LLM / Embedding / Rerank 函数
    3. 配置 Neo4JStorage 作为图存储后端
    4. 配置 QdrantVectorDBStorage 作为向量存储后端
    5. 启动持久事件循环并调用 initialize_storages()
    6. 替换所有同步方法以使用持久循环

    Args:
        working_dir: 本地缓存目录（KV 存储和文档状态使用 JSON 文件）
        env_path: 环境变量配置文件路径
        workspace: 工作空间标识，用于数据隔离
        **extra_kwargs: 其他 LightRAG 构造函数参数

    Returns:
        已配置且存储已初始化的 LightRAG 实例。
        调用 rag._persistent_loop.close() 可释放资源。

    Example:
        >>> from service.kg_config import create_lightrag_neo4j_qdrant
        >>> rag = create_lightrag_neo4j_qdrant()
        >>> rag.insert("张某以虚构投资平台为由骗取他人财物...")
        >>> rag._persistent_loop.close()  # 使用完毕后释放资源
    """
    load_env(env_path)

    # 从 custom_entity_service 读取实体类型（保证单一来源）
    from .custom_entity_service import ENTITY_TYPES as _ENTITY_TYPES

    # 启动持久事件循环
    ploop = _PersistentLoop()
    ploop.start()

    rag = LightRAG(
        working_dir=working_dir,
        workspace=workspace,
        # 存储后端配置
        graph_storage="Neo4JStorage",
        vector_storage="QdrantVectorDBStorage",
        kv_storage="MongoKVStorage",
        doc_status_storage="MongoDocStatusStorage",
        # LLM 配置
        llm_model_func=build_llm_func(),
        llm_model_name=os.getenv("LLM_MODEL", "Qwen3-235B-A22B-Instruct"),
        # Embedding 配置
        embedding_func=build_embedding_func(),
        # Rerank 配置
        rerank_model_func=build_rerank_func(),
        # 自定义实体类型引导
        addon_params={
            "language": "Chinese",
            "entity_types": _ENTITY_TYPES,
        },
        # 分块参数
        chunk_token_size=1200,
        chunk_overlap_token_size=100,
        **extra_kwargs,
    )

    # 在持久事件循环上初始化存储（使用线程安全方式）
    ploop.run_coroutine(rag.initialize_storages())

    # 替换同步方法以使用持久循环
    _patch_sync_methods(rag, ploop)

    return rag
