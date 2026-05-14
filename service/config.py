"""LightRAG 配置与初始化模块

负责从 config.yaml 读取配置，构建 LightRAG 实例所需的
LLM 函数、Embedding 函数、Rerank 函数，并返回初始化好的 LightRAG 对象。
"""

import asyncio
from functools import partial
from typing import Any

import yaml
from lightrag import LightRAG, QueryParam
from lightrag.utils import EmbeddingFunc
from lightrag.llm.openai import openai_complete, openai_embed
from lightrag.rerank import generic_rerank_api


def load_config(path: str = "config.yaml") -> dict[str, Any]:
    """加载项目根目录下的 config.yaml 配置文件。

    Args:
        path: 配置文件路径，默认为 config.yaml

    Returns:
        配置文件内容字典
    """
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def build_llm_func(cfg: dict[str, Any]) -> partial:
    """构建 LLM 调用函数。

    使用 functools.partial 预先绑定 API 地址和密钥，
    LightRAG 内部会自动注入 hashing_kv 等额外参数。

    Args:
        cfg: config.yaml 中 llm 部分的配置

    Returns:
        已绑定 base_url 的 openai_complete 函数
    """
    return partial(
        openai_complete,
        base_url=cfg["base_url"],
        api_key=cfg["api_key"],
    )


def build_embedding_func(cfg: dict[str, Any]) -> EmbeddingFunc:
    """构建 Embedding 函数。

    注意：openai_embed 已被 @wrap_embedding_func_with_attrs 装饰，
    需要使用 .func 访问原始函数以避免双重包装。

    Args:
        cfg: config.yaml 中 embedding 部分的配置

    Returns:
        EmbeddingFunc 实例
    """
    return EmbeddingFunc(
        embedding_dim=1024,  # bge-m3 的向量维度
        func=partial(
            openai_embed.func,  # 使用 .func 避免双重包装
            model=cfg["model"],
            base_url=cfg["base_url"],
            api_key=cfg["api_key"],
        ),
        max_token_size=8192,
        model_name=cfg["model"],
    )


def build_rerank_func(cfg: dict[str, Any]) -> partial:
    """构建 Rerank 函数。

    Args:
        cfg: config.yaml 中 rerank 部分的配置

    Returns:
        已绑定配置的 generic_rerank_api 函数
    """
    return partial(
        generic_rerank_api,
        model=cfg["model"],
        base_url=cfg["base_url"],
        api_key=cfg["api_key"],
    )


def create_lightrag(
    working_dir: str = "./rag_storage",
    config_path: str = "config.yaml",
    **extra_kwargs: Any,
) -> LightRAG:
    """创建并返回一个已初始化的 LightRAG 实例。

    从 config.yaml 中读取 LLM、Embedding、Rerank 配置，
    自动构建所需的函数对象并传入 LightRAG。
    创建后自动调用 initialize_storages() 完成存储初始化。

    Args:
        working_dir: LightRAG 工作目录，用于存储向量数据和缓存
        config_path: 配置文件路径
        **extra_kwargs: 其他 LightRAG 构造函数参数

    Returns:
        已配置且存储已初始化的 LightRAG 实例

    Example:
        >>> from service import create_lightrag
        >>> rag = create_lightrag()
        >>> rag.insert("这是一段测试文本")
        >>> result = rag.query("测试内容是什么？")
    """
    cfg = load_config(config_path)

    rag = LightRAG(
        working_dir=working_dir,
        llm_model_func=build_llm_func(cfg["llm"]),
        llm_model_name=cfg["llm"]["model"],
        embedding_func=build_embedding_func(cfg["embedding"]),
        rerank_model_func=build_rerank_func(cfg["rerank"]),
        # 从配置中读取 chunk 参数
        chunk_token_size=1200,
        chunk_overlap_token_size=100,
        **extra_kwargs,
    )

    # 初始化存储（LightRAG 要求显式调用）
    loop = asyncio.new_event_loop()
    try:
        loop.run_until_complete(rag.initialize_storages())
    finally:
        loop.close()

    return rag
