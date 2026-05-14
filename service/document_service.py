"""文档管理服务模块

封装 LightRAG 的文档插入、删除、状态查询等管理操作。
所有方法均为同步接口，内部自动处理异步事件循环。
"""

import asyncio
from typing import Any

from lightrag import LightRAG
from lightrag.base import DocStatus, DocProcessingStatus, DeletionResult

from .config import create_lightrag, load_config


class DocumentService:
    """文档管理服务，封装 LightRAG 文档管理相关方法。

    负责文档的插入、删除、状态查询等生命周期管理操作。
    """

    def __init__(self, rag: LightRAG | None = None, config_path: str = "config.yaml"):
        """初始化文档服务。

        Args:
            rag: 已初始化的 LightRAG 实例。若为 None，则自动从配置文件创建
            config_path: 配置文件路径，仅在 rag 为 None 时使用
        """
        self.rag = rag or create_lightrag(config_path=config_path)

    def insert(
        self,
        text: str | list[str],
        split_by_character: str | None = None,
        ids: str | list[str] | None = None,
        file_paths: str | list[str] | None = None,
    ) -> str:
        """插入文档到知识库。

        将文本内容分块、抽取实体与关系，并写入向量数据库和知识图谱。

        Args:
            text: 单个文档文本或文档文本列表
            split_by_character: 按指定字符切分文本；若为 None 则按 token 数量自动分块
            ids: 文档唯一 ID，若不提供则自动生成 MD5 哈希 ID
            file_paths: 文件路径列表，用于引用溯源

        Returns:
            跟踪 ID（track_id），可用于查询处理进度

        Example:
            >>> svc = DocumentService()
            >>> track_id = svc.insert("铜合金具有优异的导电性和强度...")
            >>> print(svc.get_processing_status())
        """
        return self.rag.insert(
            input=text,
            split_by_character=split_by_character,
            ids=ids,
            file_paths=file_paths,
        )

    def get_processing_status(self) -> dict[str, int]:
        """获取当前文档处理状态统计。

        Returns:
            各状态的文档数量字典，例如:
            {"pending": 0, "processing": 1, "processed": 10, "failed": 0}
        """
        loop = asyncio.new_event_loop()
        try:
            return loop.run_until_complete(self.rag.get_processing_status())
        finally:
            loop.close()

    def get_docs_by_status(self, status: DocStatus) -> dict[str, DocProcessingStatus]:
        """按状态获取文档详情。

        Args:
            status: 文档状态枚举值（PENDING / PROCESSING / PROCESSED / FAILED）

        Returns:
            文档 ID 到处理状态对象的映射字典
        """
        loop = asyncio.new_event_loop()
        try:
            return loop.run_until_complete(self.rag.get_docs_by_status(status))
        finally:
            loop.close()

    def get_processed_docs(self) -> dict[str, DocProcessingStatus]:
        """获取所有已处理完成的文档。

        Returns:
            已处理文档的 ID 到状态对象的映射字典
        """
        return self.get_docs_by_status(DocStatus.PROCESSED)

    def get_failed_docs(self) -> dict[str, DocProcessingStatus]:
        """获取所有处理失败的文档。

        Returns:
            失败文档的 ID 到状态对象的映射字典，包含错误信息
        """
        return self.get_docs_by_status(DocStatus.FAILED)

    def delete_by_doc_id(self, doc_id: str, delete_llm_cache: bool = False) -> DeletionResult:
        """按文档 ID 删除文档及其所有关联数据。

        会删除该文档对应的的文本块、实体、关系、向量存储等数据。
        如果实体/关系被部分影响，会用剩余文档的 LLM 缓存重建。

        Args:
            doc_id: 文档的唯一标识符
            delete_llm_cache: 是否同时删除关联的 LLM 缓存

        Returns:
            删除结果对象，包含 status、message、status_code 等字段
        """
        loop = asyncio.new_event_loop()
        try:
            return loop.run_until_complete(
                self.rag.adelete_by_doc_id(doc_id, delete_llm_cache)
            )
        finally:
            loop.close()

    def delete_by_entity(self, entity_name: str) -> DeletionResult:
        """按实体名称删除实体及其所有关系。

        Args:
            entity_name: 实体名称

        Returns:
            删除结果对象
        """
        return self.rag.delete_by_entity(entity_name)

    def delete_by_relation(
        self, source_entity: str, target_entity: str
    ) -> DeletionResult:
        """删除两个实体之间的关系。

        Args:
            source_entity: 源实体名称
            target_entity: 目标实体名称

        Returns:
            删除结果对象
        """
        return self.rag.delete_by_relation(source_entity, target_entity)
