# LightRAG 反诈知识图谱项目

基于 [HKUDS/LightRAG](https://github.com/HKUDS/LightRAG) 框架，构建诈骗案件法律判决文书的知识图谱与智能问答系统，用于参加金融犯罪反诈竞赛。

## 项目简介

本项目将 LightRAG 框架应用于法律判决文书（诈骗案件）领域，实现：

- **实体/关系抽取**：通过 LLM 从判决书中自动抽取 7 种领域实体和 9 种关系
- **知识图谱构建**：使用 Neo4j 存储实体节点（双 label）和有向关系边
- **向量检索**：使用 Qdrant 存储实体向量和 chunk 向量，支持语义搜索
- **智能问答**：基于知识图谱 + 向量混合检索，由 LLM 生成回答
- **BM25 关键词检索**：稀疏向量索引，补全语义搜索的关键词匹配能力

### 技术栈

| 组件 | 技术 | 用途 |
|------|------|------|
| 框架 | LightRAG 1.4+ | RAG + 知识图谱 |
| 图数据库 | Neo4j 5.26-community | 实体节点和关系边存储 |
| 向量数据库 | Qdrant v1.12.5 | 实体向量 + chunk 向量 |
| KV 存储 | MongoDB (AsyncMongoKV) | 文档追踪（full_entities, full_relations, doc_status） |
| LLM | Qwen3-235B（可替换） | 实体/关系抽取、问答生成 |
| Embedding | bge-m3 | 文本向量化 |
| Rerank | bge-reranker-v2-m3 | 检索结果重排序 |

### 实体类型（7 种）

| 实体类型 | 含义 | 数量 |
|---------|------|------|
| `summary` | 案件简要描述（固定格式 1 句话）| 每篇文书 1 个 |
| `FraudScenario` | 诈骗场景（从 10 种预定义类型中选）| 1-N 个 |
| `FraudFeature` | 骗局特征（欺骗性表现）| 2-8 个 |
| `FraudMethod` | 诈骗手法（技术手段）| 1-5 个 |
| `PreventionMeasure` | 防范建议（基于案情推理生成）| 1-N 个 |
| `LawRegulation` | 法律法规（保留但降低权重）| 0-N 个 |
| `RelatedCase` | 关联案例（详细案例总结，用于相似案例匹配）| 每篇文书 1 个 |

### 10 种预定义诈骗场景

`FraudScenario` 必须从以下预定义类型中选择：
刷单返利类诈骗、虚假网络投资理财类诈骗、虚假购物/服务类诈骗、冒充电商物流客服类诈骗、贷款/征信类诈骗、冒充领导/熟人类诈骗、冒充公检法类诈骗、婚恋/交友类诈骗、网络游戏虚假交易类诈骗、机票退改类诈骗。

### 关系类型（9 种）

| 关系 | 源 → 目标 | 说明 |
|------|----------|------|
| `involves` | summary → FraudScenario | 案件摘要涉及某诈骗场景 |
| `describes` | summary → FraudFeature | 摘要描述骗局特征 |
| `mentions` | summary → FraudMethod | 摘要提及诈骗手法 |
| `summarizes` | RelatedCase → FraudScenario | 关联案例涉及某诈骗场景 |
| `has_feature` | RelatedCase → FraudFeature | 关联案例具有某骗局特征 |
| `uses` | RelatedCase → FraudMethod | 关联案例使用某诈骗手法 |
| `violates` | FraudMethod → LawRegulation | 诈骗手法违反法律法规 |
| `prevents` | PreventionMeasure → FraudMethod | 防范措施预防某手法 |
| `counters` | PreventionMeasure → FraudFeature | 防范措施应对某特征 |

## 快速开始

### 1. 环境要求

- Python 3.10+
- Docker + Docker Compose
- 外部 LLM / Embedding / Rerank API

### 2. 安装依赖

```bash
pip install lightrag neo4j qdrant-client python-dotenv requests pymongo
```

### 3. 启动数据库服务

```bash
docker-compose up -d
```

等待服务就绪（约 30 秒）：

```bash
docker ps
# 等待 lightrag-neo4j 状态变为 (healthy)
```

验证连通性：

```bash
# Neo4j 浏览器：http://localhost:7474
# Neo4j 连接：bolt://localhost:7687（用户名: neo4j，密码: LightRAG2026!）
# Qdrant 浏览器：http://localhost:6333/dashboard
```

### 4. 配置环境变量

编辑 `docker.env` 文件，配置 API 地址和密钥：

```bash
# LLM 配置
LLM_API_KEY=your_api_key
LLM_MODEL=your_model_name
LLM_BASE_URL=http://your_llm_host:port/v1

# Embedding 配置
EMBEDDING_API_KEY=your_api_key
EMBEDDING_MODEL=bge-m3
EMBEDDING_BASE_URL=http://your_embedding_host:port/v1/

# Rerank 配置
RERANK_API_KEY=your_api_key
RERANK_MODEL=bge-reranker-v2-m3
RERANK_BASE_URL=http://your_rerank_host:port/v1/rerank

# Neo4j 配置（默认无需修改）
NEO4J_URI=bolt://localhost:7687
NEO4J_USERNAME=neo4j
NEO4J_PASSWORD=LightRAG2026!
NEO4J_DATABASE=neo4j

# Qdrant 配置（默认无需修改）
QDRANT_URL=http://localhost:6333
QDRANT_API_KEY=LightRAG2026Qdrant!
```

## 使用方式

### 方式一：自定义实体抽取 + 插入（推荐）

使用反诈领域自定义的 7 种实体类型 + 9 种关系类型，输出结构化属性。

```python
from service.custom_entity_service import CustomEntityService

svc = CustomEntityService()

result = svc.insert_with_custom_schema(
    text="张某以虚构境外期货交易平台为由...",
    doc_id="case_001",         # 可选，自定义文档 ID
    file_path="judgment.md",   # 可选，溯源文件路径
)

print(f"抽取到 {len(result['entities'])} 个实体")
print(f"抽取到 {len(result['relationships'])} 个关系")
```

该方式会在插入前自动校验：
- 跳过 `normalized_name` 为空的实体
- 跳过实体类型不在预定义 7 种中的实体
- 跳过属性 key 不在 ATTRIBUTE_SCHEMA 中的条目
- 跳过关系类型不在预定义 9 种中的关系
- 跳过 src/tgt 不在有效实体中的关系
- 自动实体消歧（精确匹配 + embedding 相似度，阈值 0.85）

### 方式二：纯 LLM 抽取（不落库）

只调用 LLM 抽取实体和关系，输出到 JSON 文件，不写入任何数据库。适合调试和检查结果。

```bash
python test_extract/extract_only.py
```

## 查询知识图谱

```python
from service.query_service import QueryService

svc = QueryService()

# 问答模式（推荐 mix 模式）
answer = svc.query("这起案件的诈骗手法是什么？", mode="mix")
print(answer)

# 纯数据检索（不经过 LLM 生成回答）
data = svc.query_data("诈骗手法")
print(data["entities"])
print(data["relationships"])

# 获取实体详情
info = svc.get_entity_info("虚假网络投资理财类诈骗")
print(info)

# 获取关系详情
info = svc.get_relation_info("summary", "虚假网络投资理财类诈骗")
print(info)

# 获取知识图谱子图
graph = svc.get_knowledge_graph("*", max_depth=2, max_nodes=100)
print(f"节点数: {len(graph.nodes)}, 边数: {len(graph.edges)}")

# 查看所有实体标签
labels = svc.get_graph_labels()
print(labels)
```

### 快捷查询方法（按实体类型）

```python
# 查询所有预定义的诈骗场景（10 种）
scenarios = svc.query_fraud_scenarios()

# 查询所有骗局特征
features = svc.query_fraud_features()

# 查询所有诈骗手法
methods = svc.query_fraud_methods()

# 查询所有关联案例
cases = svc.query_related_cases()

# 查询所有法律法规
laws = svc.query_laws()

# 查询案件摘要
summary = svc.query_case_summary()

# 查询防范措施
preventions = svc.query_prevention()
```

### 按实体类型查询

```python
# 通用方法：按任意实体类型查询
entities = svc.query_by_entity_type("FraudMethod", top_k=50)
```

### 检索模式说明

| 模式 | 适用场景 |
|------|---------|
| `mix`（推荐） | 知识图谱 + 向量混合检索，大多数问答场景 |
| `local` | 查询具体实体的详细信息 |
| `global` | 查询跨实体的关联和趋势 |
| `hybrid` | local + global 合并 |
| `naive` | 纯向量搜索，不使用知识图谱 |
| `bypass` | 不调用检索，直接问 LLM |

### 全部 20 个查询方法

| 方法 | 说明 |
|------|------|
| `query(question, mode)` | 问答模式 |
| `query_data(query_text)` | 纯数据检索 |
| `get_entity_info(name)` | 实体详情 |
| `get_relation_info(src, tgt)` | 关系详情 |
| `get_graph_labels()` | 实体标签列表 |
| `get_knowledge_graph(query, max_depth, max_nodes)` | 子图 |
| `get_processed_doc_ids()` | 已处理文档列表 |
| `query_with_bm25(text, top_k)` | BM25 关键词检索 |
| `query_by_entity_type(type, top_k)` | 按实体类型查询 |
| `query_case_summary()` | 案件摘要 |
| `query_similar_cases()` | 相似案例 |
| `query_prevention()` | 防范建议 |
| `query_fraud_scenarios()` | 诈骗场景 |
| `query_fraud_features()` | 骗局特征 |
| `query_fraud_methods()` | 诈骗手法 |
| `query_related_cases()` | 关联案例 |
| `query_laws()` | 法律法规 |
| `get_entity_centrality()` | 实体中心性 |
| `get_shortest_path(src, tgt)` | 最短路径 |
| `get_entity_neighbors(name)` | 实体邻居 |

## 文档管理 / 删除

```python
from service.custom_entity_service import CustomEntityService

svc = CustomEntityService()

# 按文档 ID 删除（删除该文档创建的所有实体、关系、向量）
svc.delete_by_doc_id("case_001")

# 按实体名删除（实体及其所有关系边）
svc.delete_by_entity("虚假网络投资理财类诈骗")

# 删除两个实体之间的所有关系边
svc.delete_by_relation("summary", "虚假网络投资理财类诈骗")
```

删除流程会同时清理 Neo4j、Qdrant 和 MongoDB 中的数据：
1. Neo4j：`DETACH DELETE` 删除实体节点和所有连接边
2. Qdrant：删除实体向量和 chunk 向量
3. MongoDB：清理 full_entities、full_relations、doc_status、text_chunks

## 项目结构

```
d:\LightRAG/
├── docker-compose.yml          # Docker 服务编排（Neo4j + Qdrant）
├── docker.env                  # 环境变量配置（API 地址、密钥等）
├── test.md                     # 测试用法律判决文书（诈骗案件）
├── .gitignore
│
├── service/                    # 业务服务层
│   ├── __init__.py             # 模块导出（CustomEntityService, FRAUD_SCENARIOS）
│   ├── kg_config.py            # Neo4j + Qdrant 版 LightRAG 实例创建
│   ├── neo4j_writer.py         # Neo4j 直接写入器（双 label + 有向边）
│   ├── direct_ingestion.py     # 自定义直接入库（绕过 ainsert_custom_kg）
│   ├── custom_entity_service.py # 自定义实体抽取（7 种反诈实体类型）
│   ├── query_service.py        # 查询服务（20 个查询方法）
│   ├── document_service.py     # 文档管理服务
│   ├── entity_disambiguation.py # 实体消歧（精确匹配 + embedding 相似度）
│   └── bm25_retrieval.py       # BM25 稀疏向量检索
│
├── script/                     # 运维脚本
├── doc/                        # 文档资料
└── test_extract/               # 纯 LLM 抽取测试（不落库）
```

## 如何修改实体、关系和 Prompt

> 项目的基础设施（Docker、存储、服务层、校验逻辑）**无需修改**。
> 需要调整时，只需编辑以下位置：

### 需要修改的文件

| 修改内容 | 修改文件 |
|---|---|
| **实体类型列表**（唯一来源） | `service/custom_entity_service.py` → `ENTITY_TYPES` |
| **实体属性 Schema** | `service/custom_entity_service.py` → `ATTRIBUTE_SCHEMA` |
| **实体抽取 Prompt** | `service/custom_entity_service.py` → `ENTITY_EXTRACT_PROMPT` |
| **关系抽取 Prompt** | `service/custom_entity_service.py` → `RELATION_EXTRACT_PROMPT` |
| **关系类型校验** | `service/custom_entity_service.py` → `VALID_RELATION_TYPES` |
| **预定义诈骗场景** | `service/custom_entity_service.py` → `FRAUD_SCENARIOS` |

### 替换为新领域

只需修改 `custom_entity_service.py` 中的五个部分：
- `ENTITY_TYPES` — 改为新领域的实体类型
- `ATTRIBUTE_SCHEMA` — 改为各实体类型的预定义属性
- `VALID_RELATION_TYPES` — 改为对应关系类型
- `ENTITY_EXTRACT_PROMPT` — 改为新领域的实体抽取 prompt
- `RELATION_EXTRACT_PROMPT` — 改为新领域的关系抽取 prompt

其余代码（校验逻辑、存储映射、服务封装、`kg_config.py`、`direct_ingestion.py`、`neo4j_writer.py`）**完全复用**。

## 存储架构

### Neo4j 实体节点（双 Label 策略）

每个实体节点同时拥有两个 Label：
- `:base` — LightRAG 查询层使用，保持框架兼容
- `:FraudScenario` / `:FraudFeature` / ... — 语义查询使用，对应 entity_type

```cypher
(n:base:FraudScenario {
    entity_name: "虚假网络投资理财类诈骗",
    entity_type: "FraudScenario",
    description: "以虚构投资理财平台为名的诈骗方式",
    attr_涉案金额: "609500元",
    source_id: "doc-xxx",
    file_path: "test.md",
    created_at: 1715000000
})
```

### Neo4j 关系边（有向 + 类型 Label）

关系边使用 relation_type 作为边 Label，保留方向语义：

```cypher
(a:base:summary {entity_name: "张某以虚构...诈骗案"})
  -[:involves {
      evidence: "原文依据...",
      source_id: "doc-xxx",
      weight: 1.0,
      created_at: 1715000000
  }]->
(b:base:FraudScenario {entity_name: "虚假网络投资理财类诈骗"})
```

### 写入流程

`insert_with_custom_schema()` 不调用 LightRAG 的 `ainsert_custom_kg`，而是使用 `direct_ingest()` 直接编排所有存储层：

| 存储层 | 写入内容 | 说明 |
|--------|---------|------|
| Qdrant `entities_vdb` | 实体向量 | 语义搜索 |
| Qdrant `chunks_vdb` | chunk 向量 | 检索上下文 |
| Qdrant BM25 | 稀疏向量索引 | 关键词检索 |
| MongoDB `text_chunks` | chunk 元数据 | 分块内容、source_id 等 |
| MongoDB `full_entities` | 文档→实体映射 | 用于按文档 ID 删除 |
| MongoDB `full_relations` | 文档→关系映射 | 用于按文档 ID 删除 |
| MongoDB `doc_status` | 文档状态 | 标记为 PROCESSED |
| Neo4j | 实体节点 + 有向边 | 双 label + 类型边 label |
| ~~Qdrant `relationships_vdb`~~ | ~~关系向量~~ | **不写入**（关系不向量化） |

写入后自动清理历史残留的 `:DIRECTED` 边。

### Neo4j 查询示例

```cypher
-- 按实体类型查询
MATCH (n:FraudScenario) RETURN n.entity_name, n.description LIMIT 5

-- 按动态属性查询
MATCH (n) WHERE n.attr_涉案金额 CONTAINS '万元' RETURN n.entity_name

-- 按关系类型查询
MATCH (a:summary)-[r:involves]->(b:FraudScenario)
RETURN a.entity_name AS 案件, b.entity_name AS 诈骗场景

-- 查询实体的所有出边关系
MATCH (n {entity_name: '张某虚假网络投资理财诈骗案'})-[r]->(m)
RETURN type(r) AS 关系类型, m.entity_name AS 目标实体

-- 双 label 兼容查询（LightRAG 查询层使用）
MATCH (n:base) RETURN n.entity_name, n.entity_type LIMIT 5
```

## 常见问题

### 1. Docker 端口被占用

Neo4j 使用 7474（Browser）和 7687（Bolt），Qdrant 使用 6333（REST）和 6334（gRPC）。

```bash
# 检查端口占用
netstat -ano | findstr "7687"
netstat -ano | findstr "6333"

# 如有冲突，修改 docker-compose.yml 中的 ports 映射
```

### 2. Qdrant 连接失败

```bash
# 检查 Qdrant 是否启动
curl http://localhost:6333/healthz

# 检查 Docker 状态
docker ps | grep qdrant
```

### 3. LLM 返回 502 或超时

```bash
# 直接测试 LLM 服务连通性
curl http://your_llm_host:port/v1/models

# LightRAG 默认超时为 180 秒，长文本抽取可能需要更长时间
```

### 4. 清理测试数据

```bash
# Neo4j 中删除所有数据
docker exec lightrag-neo4j cypher-shell -u neo4j -p "LightRAG2026!" \
  "MATCH (n) DETACH DELETE n"

# Qdrant 中删除所有 collection
# 通过 Qdrant Dashboard: http://localhost:6333/dashboard

# 清理本地缓存
rm -rf rag_storage/
```

### 5. 在 Linux 上运行

`docker-compose.yml` 完全适配 Linux，无需修改。如果 Neo4j 数据目录遇到权限问题：

```yaml
# 在 neo4j service 下添加
user: "1000:1000"
```
