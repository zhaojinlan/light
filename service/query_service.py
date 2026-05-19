"""反诈领域知识图谱查询服务

封装 LightRAG 的知识图谱查询、实体查询、关系查询等方法。
支持多种检索模式（local / global / hybrid / naive / mix）。
提供 BM25 关键词混合检索、实体中心性排名、最短路径、N 跳邻居等图分析功能。

预定义的 7 种实体类型：
    summary — 案件简要描述（固定格式 1 句话）
    FraudScenario — 诈骗场景（10 种预定义类型）
    FraudFeature — 骗局特征（欺骗性表现）
    FraudMethod — 诈骗手法（技术手段）
    PreventionMeasure — 防范建议（基于案情推理）
    LawRegulation — 法律法规
    RelatedCase — 关联案例（详细案例总结，用于相似案例匹配）

9 种关系类型：
    involves、describes、mentions、summarizes、has_feature、
    uses、violates、prevents、counters
"""

import asyncio
from typing import Any

from lightrag import LightRAG, QueryParam
from lightrag.types import KnowledgeGraph
from lightrag.base import DocStatus

from .kg_config import create_lightrag_neo4j_qdrant


class QueryService:
    """知识图谱查询服务。

    封装 LightRAG 的查询操作，包括：
    - RAG 问答查询（支持多种检索模式）
    - 纯数据检索（不经过 LLM 生成）
    - 实体/关系详情查询
    - 知识图谱子图查询
    """

    def __init__(self, rag: LightRAG | None = None):
        """初始化查询服务。

        Args:
            rag: 已初始化的 LightRAG 实例。若为 None，则自动使用 Neo4j+Qdrant+MongoDB 存储创建
        """
        self.rag = rag or create_lightrag_neo4j_qdrant()

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
            >>> answer = svc.query("这起案件属于哪类诈骗？", mode="mix")
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
        """
        return self.rag.get_entity_info(entity_name, include_vector_data)

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
        return self.rag.get_relation_info(source_entity, target_entity, include_vector_data)

    def get_graph_labels(self) -> list[str]:
        """获取知识图谱中所有实体标签（名称）。"""
        return self.rag.get_graph_labels()

    def get_knowledge_graph(
        self,
        entity_name: str = "*",
        max_depth: int = 3,
        max_nodes: int = 1000,
    ) -> KnowledgeGraph:
        """获取知识图谱的子图。

        Args:
            entity_name: 起始实体名称，"*" 表示获取全部节点
            max_depth: 子图最大深度
            max_nodes: 最大返回节点数

        Returns:
            KnowledgeGraph 对象，包含 nodes 和 edges
        """
        return self.rag.get_knowledge_graph(entity_name, max_depth, max_nodes)

    def get_processed_doc_ids(self) -> list[str]:
        """获取所有已处理完成的文档 ID 列表。"""
        docs = self._run_async(
            self.rag.get_docs_by_status(DocStatus.PROCESSED)
        )
        return list(docs.keys())

    def query_with_bm25(
        self,
        question: str,
        top_k: int = 10,
        dense_weight: float = 0.5,
        bm25_weight: float = 0.5,
    ) -> dict[str, Any]:
        """使用 dense + BM25 混合检索返回结果（不经过 LLM 生成回答）。

        同时搜索 dense 向量（语义相似度）和 BM25 稀疏向量（关键词匹配），
        通过 RRF（Reciprocal Rank Fusion）融合两种检索结果。

        Args:
            question: 查询文本
            top_k: 返回结果数量上限
            dense_weight: dense 搜索权重（RRF 融合比例）
            bm25_weight: BM25 关键词匹配权重

        Returns:
            包含以下字段的字典:
                - chunks: 检索到的 chunk 列表，每项包含 content 和 rrf_score
                - query: 原始查询文本
                - mode: "bm25_hybrid"
        """
        from .bm25_retrieval import BM25RetrievalService

        bm25_service = BM25RetrievalService(self.rag)
        bm25_service.ensure_sparse_index()

        chunks = bm25_service.search(
            query=question,
            top_k=top_k,
            dense_weight=dense_weight,
            bm25_weight=bm25_weight,
        )

        return {
            "chunks": chunks,
            "query": question,
            "mode": "bm25_hybrid",
        }

    def query_by_entity_type(
        self,
        entity_type: str,
        top_k: int = 50,
    ) -> list[dict[str, Any]]:
        """按实体类型检索实体。

        从 entities_vdb 中获取指定类型的所有实体信息。
        适用于按类别浏览图谱内容，如获取所有诈骗手法、防范措施等。

        Args:
            entity_type: 实体类型，可选值：
                - "FraudScenario": 诈骗场景（10 种预定义类型）
                - "FraudFeature": 骗局特征（欺骗性表现）
                - "FraudMethod": 诈骗手法（技术手段）
                - "PreventionMeasure": 防范建议（基于案情推理）
                - "LawRegulation": 法律法规
                - "RelatedCase": 关联案例（详细案例总结）
                - "summary": 案件简要描述（固定格式 1 句话）
            top_k: 返回数量上限

        Returns:
            实体信息列表，每项包含 entity_name、description、entity_type 字段

        Example:
            >>> svc = QueryService()
            >>> methods = svc.query_by_entity_type("FraudMethod")
            >>> for m in methods:
            ...     print(m["entity_name"])
        """
        results = self._run_async(
            self.rag.entities_vdb.query("", top_k=top_k)
        )
        return [
            {
                "entity_name": item.get("entity_name", ""),
                "description": item.get("description", ""),
                "entity_type": item.get("entity_type", ""),
            }
            for item in results
            if item.get("entity_type") == entity_type
        ]

    def query_case_summary(self, doc_id: str | None = None) -> list[dict[str, Any]]:
        """获取已入库案件的 summary 实体（案件简要描述）。

        summary 实体采用固定格式："{被告人身份+姓名}以{诈骗方式/理由}，
        通过{具体手段}骗取{被害人}共计{金额}，被判处{刑罚+罚金}"，
        用于快速查看案件基本信息（被告人、手法、金额、判决结果等）。
        每个文书仅包含 1 个 summary 实体。

        Args:
            doc_id: 若指定则仅返回该文档关联的 summary，否则返回所有

        Returns:
            summary 实体列表，每项包含 entity_name（摘要文本）、description（同摘要）、
            entity_type（"summary"）、source_id（文档 ID）
        """
        results = self._run_async(
            self.rag.entities_vdb.query("", top_k=1000)
        )
        summaries = [
            {
                "entity_name": item.get("entity_name", ""),
                "description": item.get("description", ""),
                "entity_type": item.get("entity_type", ""),
                "source_id": item.get("source_id", ""),
            }
            for item in results
            if item.get("entity_type") == "summary"
        ]
        if doc_id:
            summaries = [s for s in summaries if s.get("source_id") == doc_id]
        return summaries

    def query_similar_cases(
        self,
        question: str,
        top_k: int = 5,
    ) -> dict[str, Any]:
        """类似案件检索。

        使用 BM25 混合检索（dense 向量语义相似度 + sparse 关键词匹配）
        查找与查询相似的案件文本块（chunk），
        再提取这些 chunk 关联的 summary 实体（案件简要描述）和 RelatedCase 实体（关联案例）。

        Args:
            question: 查询描述，如"虚假投资理财诈骗"、"冒充客服退款"
            top_k: 返回相似案件数量上限

        Returns:
            包含以下字段的字典：
                - chunks: 相关文本块列表（content + rrf_score）
                - summaries: 匹配的案件简要描述列表
                - related_cases: 匹配的关联案例列表（含案情经过、判决结果等详细信息）
                - query: 原始查询文本
                - count: 匹配案件总数

        Example:
            >>> svc = QueryService()
            >>> cases = svc.query_similar_cases("虚假投资理财诈骗")
            >>> print(f"找到 {cases['count']} 个类似案件")
        """
        chunks = self.query_with_bm25(question, top_k=top_k)["chunks"]

        # 从相关 chunk 中提取关联的 summary 和 related_case
        source_ids = set()
        for chunk in chunks:
            sid = chunk.get("payload", {}).get("source_id", "")
            if sid:
                source_ids.add(sid)

        summaries = self.query_case_summary()
        matched_summaries = [
            s for s in summaries if s.get("source_id") in source_ids
        ]

        related_cases = self.query_related_cases()
        matched_cases = [
            c for c in related_cases if c.get("source_id") in source_ids
        ]

        return {
            "chunks": chunks,
            "summaries": matched_summaries,
            "related_cases": matched_cases,
            "query": question,
            "count": len(matched_summaries),
        }

    def query_prevention(
        self,
        query: str | None = None,
        top_k: int = 20,
    ) -> list[dict[str, Any]]:
        """防范措施查询。

        获取防范措施（PreventionMeasure）实体，如"核实平台资质"、"警惕高收益承诺"等。
        防范措施基于案情推理生成，不限于原文，适用于学习反诈知识。

        Args:
            query: 查询文本，如"虚假投资平台"、"冒充客服"。若为 None 则返回所有防范措施
            top_k: 返回数量上限

        Returns:
            防范措施列表，每项包含 entity_name（措施名称）、description（措施说明）

        Example:
            >>> svc = QueryService()
            >>> items = svc.query_prevention("虚假投资")
            >>> for item in items:
            ...     print(item["entity_name"], "-", item["description"])
        """
        # 获取所有防范措施实体
        all_entities = self._run_async(
            self.rag.entities_vdb.query("", top_k=top_k)
        )
        preventions = [
            item for item in all_entities
            if item.get("entity_type") == "PreventionMeasure"
        ]

        # 如果有查询条件，用 BM25 混合检索排序
        if query:
            bm25_results = self.query_with_bm25(query, top_k=top_k)
            bm25_content_ids = {c.get("content", "") for c in bm25_results.get("chunks", [])}

            # 对防范措施按相关性排序：与 BM25 结果内容匹配的排前面
            def _score(p):
                desc = p.get("description", "")
                return 1.0 if any(desc in c or c in desc for c in bm25_content_ids) else 0.0

            preventions = sorted(preventions, key=_score, reverse=True)

        return [
            {
                "entity_name": item.get("entity_name", ""),
                "description": item.get("description", ""),
                "entity_type": "PreventionMeasure",
            }
            for item in preventions
        ]

    # ============================================================
    # 按实体类型快捷查询
    # ============================================================

    def query_fraud_scenarios(
        self,
        top_k: int = 20,
    ) -> list[dict[str, Any]]:
        """查询所有预定义的诈骗场景。

        当前系统预定义了 10 种诈骗场景类型，每篇文书可关联多个场景。
        返回结果为 FraudScenario 类型的实体列表。

        Args:
            top_k: 返回数量上限

        Returns:
            诈骗场景列表，每项包含 entity_name、description、entity_type。
            示例："虚假网络投资理财类诈骗"、"刷单返利类诈骗" 等。
        """
        return self.query_by_entity_type("FraudScenario", top_k=top_k)

    def query_fraud_features(
        self,
        top_k: int = 50,
    ) -> list[dict[str, Any]]:
        """查询所有骗局特征实体。

        骗局特征（FraudFeature）是从原文中抽取的欺骗性表现，
        如"虚构高收益投资平台"、"伪造交易盈利截图"、"承诺高额回报"等。

        Args:
            top_k: 返回数量上限

        Returns:
            骗局特征列表，每项包含 entity_name、description、entity_type。
        """
        return self.query_by_entity_type("FraudFeature", top_k=top_k)

    def query_fraud_methods(
        self,
        top_k: int = 50,
    ) -> list[dict[str, Any]]:
        """查询所有诈骗手法实体。

        诈骗手法（FraudMethod）是从原文中抽取的技术手段，
        如"社交软件诱导投资"、"虚假网站搭建"、"手动修改后台交易数据"等。

        Args:
            top_k: 返回数量上限

        Returns:
            诈骗手法列表，每项包含 entity_name、description、entity_type。
        """
        return self.query_by_entity_type("FraudMethod", top_k=top_k)

    def query_related_cases(
        self,
        top_k: int = 50,
    ) -> list[dict[str, Any]]:
        """查询所有关联案例实体。

        关联案例（RelatedCase）是每篇文书的详细案例总结，
        包含案情经过、诈骗手法、判决结果、防范启示等，
        用于相似案例匹配和知识库展示。
        与 summary 的区别：summary 是固定格式 1 句话，RelatedCase 是完整的案例详情。

        Args:
            top_k: 返回数量上限

        Returns:
            关联案例列表，每项包含 entity_name、description、entity_type、source_id。
            entity_name 格式："{{被告人姓名}}{{诈骗场景}}案"，如"张某虚假网络投资理财诈骗案"。
        """
        return self.query_by_entity_type("RelatedCase", top_k=top_k)

    def query_laws(
        self,
        top_k: int = 50,
    ) -> list[dict[str, Any]]:
        """查询所有法律法规实体。

        法律法规（LawRegulation）是从原文中抽取的法律条款，
        如"刑法第266条（诈骗罪）"、"虚构事实隐瞒真相"等。

        Args:
            top_k: 返回数量上限

        Returns:
            法律法规列表，每项包含 entity_name、description、entity_type。
        """
        return self.query_by_entity_type("LawRegulation", top_k=top_k)

    def _run_async(self, coro):
        """在 rag 的持久事件循环或临时循环上运行协程。"""
        if hasattr(self.rag, "_persistent_loop") and self.rag._persistent_loop is not None:
            return self.rag._persistent_loop.run_coroutine(coro)
        loop = asyncio.new_event_loop()
        try:
            return loop.run_until_complete(coro)
        finally:
            loop.close()

    # ============================================================
    # 图分析（Graph Analytics）
    # ============================================================

    def get_entity_centrality(
        self,
        entity_type: str | None = None,
        top_k: int = 20,
    ) -> list[dict[str, Any]]:
        """实体中心性排名：按连接数（degree）对实体降序排列。

        连接数越多的实体在图谱中越"核心"，可用于发现最常见的
        诈骗手法、最关键的法律条款等。

        Args:
            entity_type: 实体类型过滤，如 "FraudMethod"、"LawRegulation"。
                         若为 None 则对所有实体排名
            top_k: 返回前 N 名

        Returns:
            实体中心性列表，每项包含 entity_name、entity_type、degree

        Example:
            >>> svc = QueryService()
            >>> top = svc.get_entity_centrality("FraudMethod", top_k=10)
            >>> for t in top:
            ...     print(t["entity_name"], t["degree"])
        """
        graph = self.rag.chunk_entity_relation_graph

        # 获取所有节点
        all_nodes = self._run_async(graph.get_all_nodes())
        # 获取所有边的度数
        all_edges = self._run_async(graph.get_all_edges())

        # 计算每个节点的度数
        degree_map: dict[str, int] = {}
        for edge in all_edges:
            src = edge.get("source", "")
            tgt = edge.get("target", "")
            if src:
                degree_map[src] = degree_map.get(src, 0) + 1
            if tgt:
                degree_map[tgt] = degree_map.get(tgt, 0) + 1

        # 构建结果
        results = []
        for node in all_nodes:
            name = node.get("entity_id", "") or node.get("entity_name", "")
            if not name:
                continue
            etype = node.get("entity_type", "")
            if entity_type and etype != entity_type:
                continue
            degree = degree_map.get(name, 0)
            results.append({
                "entity_name": name,
                "entity_type": etype,
                "description": node.get("description", ""),
                "degree": degree,
            })

        results.sort(key=lambda x: x["degree"], reverse=True)
        return results[:top_k]

    def get_shortest_path(
        self,
        src_entity: str,
        tgt_entity: str,
        max_depth: int = 4,
    ) -> dict[str, Any] | None:
        """查找两个实体之间的最短路径。

        可用于分析两个诈骗手法/防范措施/法律条款之间的关联链路。

        Args:
            src_entity: 起始实体名称
            tgt_entity: 目标实体名称
            max_depth: 最大搜索深度（边数），默认 4

        Returns:
            路径信息字典，包含 nodes（实体序列）和 edges（关系序列）。
            若两个实体相同或无法连通则返回 None。

        Example:
            >>> svc = QueryService()
            >>> path = svc.get_shortest_path("伪造交易截图", "核实平台资质")
            >>> if path:
            ...     for n in path["nodes"]:
            ...         print(n["entity_name"])
        """
        if src_entity == tgt_entity:
            return None

        # 从 Neo4j 执行最短路径 Cypher 查询
        cypher = """
        MATCH (src:base {entity_name: $src}), (tgt:base {entity_name: $tgt})
        MATCH path = shortestPath((src)-[*1..$max_depth]-(tgt))
        RETURN path
        LIMIT 1
        """
        params = {"src": src_entity, "tgt": tgt_entity, "max_depth": max_depth}

        graph = self.rag.chunk_entity_relation_graph
        driver = graph._driver
        database = getattr(graph, "_DATABASE", "neo4j")

        def _execute():
            import neo4j

            with driver.session(
                database=database, default_access_mode=neo4j.READ_ACCESS
            ) as session:
                result = session.run(cypher, params)
                record = result.single()
                if not record:
                    return None

                path = record["path"]
                nodes = []
                edges = []

                for node in path.nodes:
                    nodes.append({
                        "entity_name": dict(node).get("entity_name", ""),
                        "entity_type": dict(node).get("entity_type", ""),
                        "description": dict(node).get("description", ""),
                    })

                for rel in path.relationships:
                    edges.append({
                        "relation_type": rel.type,
                        "source": rel.start_node.get("entity_name", ""),
                        "target": rel.end_node.get("entity_name", ""),
                        "description": dict(rel).get("description", ""),
                    })

                return {"nodes": nodes, "edges": edges}

        return self._run_async(asyncio.to_thread(_execute))

    def get_entity_neighbors(
        self,
        entity_name: str,
        depth: int = 1,
    ) -> dict[str, Any] | None:
        """获取实体的 N 跳邻居节点。

        Args:
            entity_name: 实体名称
            depth: 搜索深度（跳数），1 表示直接邻居，2 表示邻居的邻居

        Returns:
            邻居信息字典，包含 center（中心实体）、neighbors（邻居实体列表）、
            edges（连接关系列表）。若无该实体则返回 None。

        Example:
            >>> svc = QueryService()
            >>> nb = svc.get_entity_neighbors("虚假网络投资理财类诈骗", depth=2)
            >>> if nb:
            ...     print(f"中心: {nb['center']}")
            ...     print(f"邻居数: {len(nb['neighbors'])}")
        """
        cypher = """
        MATCH (center:base {entity_name: $name})
        MATCH path = (center)-[*1..$depth]-(neighbor)
        WHERE neighbor <> center
        RETURN DISTINCT neighbor, relationships(path) AS rels
        LIMIT 500
        """
        params = {"name": entity_name, "depth": depth}

        graph = self.rag.chunk_entity_relation_graph
        driver = graph._driver
        database = getattr(graph, "_DATABASE", "neo4j")

        def _execute():
            import neo4j

            with driver.session(
                database=database, default_access_mode=neo4j.READ_ACCESS
            ) as session:
                result = session.run(cypher, params)
                records = list(result)
                if not records:
                    # 检查中心实体是否存在
                    center_exists = session.run(
                        "MATCH (n:base {entity_name: $name}) RETURN count(n) as cnt",
                        name=entity_name,
                    ).single()["cnt"]
                    if center_exists == 0:
                        return None
                    return {"center": entity_name, "neighbors": [], "edges": []}

                neighbors = {}
                edge_set = set()
                edges = []

                for record in records:
                    node = dict(record["neighbor"])
                    neighbor_name = node.get("entity_name", "")
                    if neighbor_name:
                        neighbors[neighbor_name] = {
                            "entity_name": neighbor_name,
                            "entity_type": node.get("entity_type", ""),
                            "description": node.get("description", ""),
                        }
                    for rel in record["rels"]:
                        src = rel.start_node.get("entity_name", "")
                        tgt = rel.end_node.get("entity_name", "")
                        edge_key = (src, tgt, rel.type)
                        if edge_key not in edge_set:
                            edge_set.add(edge_key)
                            edges.append({
                                "source": src,
                                "target": tgt,
                                "relation_type": rel.type,
                            })

                return {
                    "center": entity_name,
                    "neighbors": list(neighbors.values()),
                    "edges": edges,
                }

        return self._run_async(asyncio.to_thread(_execute))
