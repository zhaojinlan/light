"""查询服务模块

封装 LightRAG 的知识图谱查询、实体查询、关系查询等方法。
支持多种检索模式（local / global / hybrid / naive / mix）。
"""

import asyncio
from typing import Any

from lightrag import LightRAG, QueryParam
from lightrag.types import KnowledgeGraph
from lightrag.base import DocStatus

from .config import create_lightrag


class QueryService:
    """知识图谱查询服务。

    封装 LightRAG 的查询操作，包括：
    - RAG 问答查询（支持多种检索模式）
    - 纯数据检索（不经过 LLM 生成）
    - 实体/关系详情查询
    - 知识图谱子图查询
    """

    def __init__(self, rag: LightRAG | None = None, config_path: str = "config.yaml"):
        """初始化查询服务。

        Args:
            rag: 已初始化的 LightRAG 实例。若为 None，则自动从配置文件创建
            config_path: 配置文件路径，仅在 rag 为 None 时使用
        """
        self.rag = rag or create_lightrag(config_path=config_path)

    def query(
        self,
        question: str,
        mode: str = "mix",
        response_type: str = "Multiple Paragraphs",
        top_k: int = 10,
    ) -> str:
        """执行知识图谱问答查询。

        根据问题从知识图谱中检索相关实体、关系和文本块，
        然后由 LLM 生成自然语言回答。

        Args:
            question: 用户提问
            mode: 检索模式，可选值:
                - "local": 基于局部上下文的检索，关注与问题直接相关的实体
                - "global": 全局知识检索，利用全局知识结构
                - "hybrid": 结合 local 和 global 两种策略
                - "naive": 简单向量搜索，不使用高级技术
                - "mix": 知识图谱 + 向量检索混合（推荐，默认）
            response_type: 回答格式，如 "Multiple Paragraphs" / "Single Paragraph" / "Bullet Points"
            top_k: 检索的实体/关系数量上限

        Returns:
            LLM 生成的回答文本

        Example:
            >>> svc = QueryService()
            >>> answer = svc.query("铜合金有哪些强化机制？", mode="mix")
            >>> print(answer)
        """
        param = QueryParam(
            mode=mode,
            response_type=response_type,
            top_k=top_k,
        )
        return self.rag.query(question, param)

    def query_data(
        self,
        question: str,
        mode: str = "mix",
        top_k: int = 10,
    ) -> dict[str, Any]:
        """执行纯数据检索，不经过 LLM 生成回答。

        返回结构化的检索结果，包括实体、关系、文本块等原始数据，
        适用于需要进一步处理检索结果的场景。

        Args:
            question: 查询文本
            mode: 检索模式（同 query 方法）
            top_k: 检索数量上限

        Returns:
            结构化数据字典，包含:
                - entities: 检索到的实体列表
                - relationships: 检索到的关系列表
                - chunks: 检索到的文本块列表
                - references: 引用来源列表
        """
        param = QueryParam(
            mode=mode,
            top_k=top_k,
        )
        return self.rag.query_data(question, param)

    def get_entity_info(
        self, entity_name: str, include_vector_data: bool = False
    ) -> dict[str, str | None | dict[str, str]]:
        """获取指定实体的详细信息。

        Args:
            entity_name: 实体名称
            include_vector_data: 是否包含向量数据库中的额外信息

        Returns:
            实体信息字典，包含实体描述、类型、关联的文本块等

        Example:
            >>> svc = QueryService()
            >>> info = svc.get_entity_info("CuNiSi")
            >>> print(info)
        """
        loop = asyncio.new_event_loop()
        try:
            return loop.run_until_complete(
                self.rag.get_entity_info(entity_name, include_vector_data)
            )
        finally:
            loop.close()

    def get_relation_info(
        self,
        source_entity: str,
        target_entity: str,
        include_vector_data: bool = False,
    ) -> dict[str, str | None | dict[str, str]]:
        """获取两个实体之间关系的详细信息。

        Args:
            source_entity: 源实体名称
            target_entity: 目标实体名称
            include_vector_data: 是否包含向量数据库中的额外信息

        Returns:
            关系信息字典，包含关系描述、权重、关联文本块等
        """
        loop = asyncio.new_event_loop()
        try:
            return loop.run_until_complete(
                self.rag.get_relation_info(source_entity, target_entity, include_vector_data)
            )
        finally:
            loop.close()

    def get_graph_labels(self) -> list[str]:
        """获取知识图谱中所有实体标签（名称）。

        Returns:
            实体名称列表，按字母顺序排序
        """
        loop = asyncio.new_event_loop()
        try:
            return loop.run_until_complete(self.rag.get_graph_labels())
        finally:
            loop.close()

    def get_knowledge_graph(
        self,
        entity_name: str = "*",
        max_depth: int = 3,
        max_nodes: int = 1000,
    ) -> KnowledgeGraph:
        """获取知识图谱的子图。

        返回以指定实体为起点的连通子图，包含节点和边的完整信息。

        Args:
            entity_name: 起始实体名称，使用 "*" 表示获取全部节点的子图
            max_depth: 子图最大深度（边数），默认 3
            max_nodes: 最大返回节点数，默认 1000

        Returns:
            KnowledgeGraph 对象，包含 nodes 和 edges 两个列表，
            以及 is_truncated 标记表示是否因节点数限制而被截断

        Example:
            >>> svc = QueryService()
            >>> graph = svc.get_knowledge_graph("CuNiSi", max_depth=2)
            >>> for node in graph.nodes:
            ...     print(node.labels, node.properties)
        """
        loop = asyncio.new_event_loop()
        try:
            return loop.run_until_complete(
                self.rag.get_knowledge_graph(entity_name, max_depth, max_nodes)
            )
        finally:
            loop.close()

    def get_processed_doc_ids(self) -> list[str]:
        """获取所有已处理完成的文档 ID 列表。

        Returns:
            已处理文档的 ID 列表
        """
        loop = asyncio.new_event_loop()
        try:
            docs = loop.run_until_complete(
                self.rag.get_docs_by_status(DocStatus.PROCESSED)
            )
            return list(docs.keys())
        finally:
            loop.close()
