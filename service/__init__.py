"""LightRAG 服务模块

提供对 LightRAG 知识库操作的高级封装，包括：
- 配置管理与实例创建（config）
- Neo4j + Qdrant 存储配置（kg_config）
- 文档管理：插入、删除、状态查询（document_service）
- 知识图谱查询：问答、实体查询、子图检索（query_service）
- 自定义实体抽取与插入（custom_entity_service）

使用方式:
    # 默认模式（JSON 文件存储）
    from service import DocumentService, QueryService

    # Neo4j + Qdrant 模式
    from service import create_lightrag_neo4j_qdrant, CustomEntityService
"""

from .config import create_lightrag, load_config, build_llm_func, build_embedding_func, build_rerank_func
from .kg_config import create_lightrag_neo4j_qdrant
from .document_service import DocumentService
from .query_service import QueryService
from .custom_entity_service import CustomEntityService, extract_entities, extract_relationships

__all__ = [
    # config.py
    "create_lightrag",
    "load_config",
    "build_llm_func",
    "build_embedding_func",
    "build_rerank_func",
    # kg_config.py
    "create_lightrag_neo4j_qdrant",
    # document_service.py
    "DocumentService",
    # query_service.py
    "QueryService",
    # custom_entity_service.py
    "CustomEntityService",
    "extract_entities",
    "extract_relationships",
]
