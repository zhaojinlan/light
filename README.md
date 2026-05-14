# LightRAG 铜合金知识图谱项目

基于 [HKUDS/LightRAG](https://github.com/HKUDS/LightRAG) 框架，构建铜合金材料领域的知识图谱与智能问答系统。

## 项目简介

本项目将 LightRAG 框架应用于铜合金材料科学领域，实现：

- **文档处理**：从材料文献/文档中自动抽取实体和关系
- **知识图谱构建**：使用 Neo4j 存储实体节点和关系边
- **向量检索**：使用 Qdrant 存储实体/关系/文本块向量
- **智能问答**：基于知识图谱 + 向量混合检索，由 LLM 生成回答

### 技术栈

| 组件 | 技术 | 用途 |
|------|------|------|
| 框架 | LightRAG 1.4+ | RAG + 知识图谱 |
| 图数据库 | Neo4j 5.26-community | 实体和关系存储 |
| 向量数据库 | Qdrant v1.12.5 | 向量相似度检索 |
| LLM | Qwen3-235B（可替换） | 实体/关系抽取、问答生成 |
| Embedding | bge-m3 | 文本向量化 |
| Rerank | bge-reranker-v2-m3 | 检索结果重排序 |

## 快速开始

### 1. 环境要求

- Python 3.10+
- Docker + Docker Compose
- 外部 LLM / Embedding / Rerank API

### 2. 安装依赖

```bash
pip install lightrag neo4j qdrant-client python-dotenv requests
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

编辑 `docker.env` 文件，配置你的 API 地址和密钥：

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

### 5. 运行测试

```bash
python test_docker.py
```

验证 Docker 服务连通性、LightRAG 实例创建、实体抽取和插入全流程。

## 使用方式

### 方式一：标准文档插入（自动抽取）

LightRAG 内置的实体/关系抽取流程，使用框架默认的 prompt 和实体类型（11 种通用类型）。

```python
from service.document_service import DocumentService

svc = DocumentService()

# 插入文本
track_id = svc.insert("铜合金CuNiSi具有优异的导电性和强度...")
print(f"插入成功，track_id: {track_id}")

# 查询处理状态
status = svc.get_processing_status()
print(status)  # {"pending": 0, "processing": 0, "processed": 1, "failed": 0}
```

### 方式二：自定义实体抽取 + 插入（推荐）

使用铜合金领域自定义的 9 种实体类型 + 7 种关系类型，输出结构化属性。

```python
from service.custom_entity_service import CustomEntityService

svc = CustomEntityService()

result = svc.insert_with_custom_schema(
    text="铜合金CuNiSi具有优异的导电性和强度，常用于电子工业...",
    doc_id="my_doc_001",        # 可选，自定义文档 ID
    file_path="my_paper.md",    # 可选，溯源文件路径
)

print(f"抽取到 {len(result['entities'])} 个实体")
print(f"抽取到 {len(result['relationships'])} 个关系")
```

该方式会在插入前自动校验：
- 跳过 `normalized_name` 为空的实体
- 跳过实体类型不在预定义 9 种中的实体
- 跳过关系中 src/tgt 不在有效实体中的条目

### 方式三：纯 LLM 抽取（不落库）

只调用 LLM 抽取实体和关系，输出到 JSON 文件，不写入任何数据库。适合调试和检查结果。

```bash
python test_extract/extract_only.py
```

输出文件：
- `test_extract/entities.json` — 实体列表（含 definition、attributes、evidence）
- `test_extract/relationships.json` — 关系列表
- `test_extract/summary.json` — 统计信息

## 查询知识图谱

```python
from service.query_service import QueryService

svc = QueryService()

# 问答模式（推荐 mix 模式）
answer = svc.query("铜合金有哪些强化机制？", mode="mix")
print(answer)

# 纯数据检索（不经过 LLM 生成回答）
data = svc.query_data("铜合金强化机制")
print(data["entities"])
print(data["relationships"])

# 获取实体详情
info = svc.get_entity_info("CuNiSi")
print(info)

# 获取关系详情
info = svc.get_relation_info("CuNiSi", "析出强化")
print(info)

# 获取知识图谱子图
graph = svc.get_knowledge_graph("*", max_depth=2, max_nodes=100)
print(f"节点数: {len(graph.nodes)}, 边数: {len(graph.edges)}")

# 查看所有实体标签
labels = svc.get_graph_labels()
print(labels)
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

## 文档管理

### 标准文档管理（`rag.insert()` 插入）

```python
from service.document_service import DocumentService

svc = DocumentService()

# 查看已处理完成的文档
docs = svc.get_processed_docs()
for doc_id, status in docs.items():
    print(f"{doc_id}: {status}")

# 查看失败的文档
failed = svc.get_failed_docs()
for doc_id, status in failed.items():
    print(f"{doc_id} 失败: {status.error}")

# 按文档 ID 删除
svc.delete_by_doc_id("my_doc_001")

# 按实体名删除（实体及其所有关系）
svc.delete_by_entity("CuNiSi")

# 删除关系
svc.delete_by_relation("CuNiSi", "析出强化")
```

### 自定义 schema 文档管理（`insert_with_custom_schema()` 插入）

```python
from service.custom_entity_service import CustomEntityService

svc = CustomEntityService()

# 按文档 ID 删除
svc.delete_by_doc_id("my_doc_001")

# 按实体名删除（实体及其所有关系边）
svc.delete_by_entity("CuNiSi")

# 删除两个实体之间的所有关系边
svc.delete_by_relation("CuNiSi", "析出强化")
```

> 两种方式的区别：`DocumentService` 适用于标准 `rag.insert()` 插入的文档；`CustomEntityService` 适用于自定义 schema 插入的文档，删除时同时清理 Neo4j 正确结构的图数据。

## 项目结构

```
d:\LightRAG/
├── docker-compose.yml          # Docker 服务编排（Neo4j + Qdrant）
├── docker.env                  # 环境变量配置（API 地址、密钥等）
├── test_docker.py              # 端到端测试脚本
├── LightRAG_技术说明.md         # LightRAG 框架底层架构文档
├── test.md                     # 测试用铜合金材料文档
│
├── service/                    # 业务服务层
│   ├── __init__.py
│   ├── config.py               # 基于 config.yaml 的 LightRAG 实例创建
│   ├── kg_config.py            # Neo4j + Qdrant 版 LightRAG 实例创建
│   ├── neo4j_writer.py         # Neo4j 直接写入器（双 label + 有向边）
│   ├── direct_ingestion.py     # 自定义直接入库（绕过 ainsert_custom_kg）
│   ├── document_service.py     # 文档管理服务（插入、删除、状态查询）
│   ├── query_service.py        # 查询服务（问答、实体查询、关系查询）
│   └── custom_entity_service.py # 自定义实体抽取（9 种铜合金实体类型）
│
├── test_extract/               # 纯 LLM 抽取测试（不落库）
│   └── extract_only.py
│
└── test.py                     # 其他测试脚本
```

## 如何修改实体、关系和 Prompt

> 项目的基础设施（Docker、存储、服务层、校验逻辑）**无需修改**。
> 需要调整时，只需编辑以下位置：

### 需要修改的文件

| 修改内容 | 修改文件 | 位置 |
|---|---|---|
| **实体类型列表**（唯一来源） | `service/custom_entity_service.py` | `ENTITY_TYPES`（第 25-35 行） |
| **实体抽取 Prompt** | `service/custom_entity_service.py` | `ENTITY_EXTRACT_PROMPT`（第 58-143 行） |
| **关系抽取 Prompt** | `service/custom_entity_service.py` | `RELATION_EXTRACT_PROMPT`（第 146-204 行） |
| **关系类型校验** | `service/custom_entity_service.py` | `VALID_RELATION_TYPES`（第 43-51 行） |
| **测试用 Prompt（不落库）** | `test_extract/extract_only.py` | `ENTITY_EXTRACT_PROMPT` |

> **`kg_config.py` 会自动引用 `custom_entity_service.py` 中的 `ENTITY_TYPES`，无需重复修改。**

### 修改示例

**新增一种实体类型（如 `ProcessingMethod`）：**

1. 在 `custom_entity_service.py` 的 `ENTITY_TYPES` 中添加：
```python
ENTITY_TYPES = [
    "MaterialSystem",
    "Phase",
    # ... 其他类型 ...
    "ProcessingMethod",  # 新增
]
```
2. 在 `ENTITY_EXTRACT_PROMPT` 中添加该类型的定义、示例和建议属性
3. 完成。`VALID_ENTITY_TYPES` 由 `set(ENTITY_TYPES)` 自动生成，`kg_config.py` 也自动引用此列表。

**新增一种关系类型（如 `produced_from`）：**

1. 在 `RELATION_EXTRACT_PROMPT` 中添加类型定义
2. 在 `VALID_RELATION_TYPES` 中添加：
```python
VALID_RELATION_TYPES = {
    "has_phase",
    # ... 其他类型 ...
    "produced_from",  # 新增
}
```

**替换为新领域（如高分子材料）：**

只需修改 `custom_entity_service.py` 中的四个部分：
- `ENTITY_TYPES` — 改为新领域的实体类型
- `VALID_RELATION_TYPES` — 改为对应关系类型
- `ENTITY_EXTRACT_PROMPT` — 改为新领域的实体抽取 prompt
- `RELATION_EXTRACT_PROMPT` — 改为新领域的关系抽取 prompt

其余代码（校验逻辑、存储映射、服务封装、`kg_config.py`）**完全复用**。

---

## 实体类型定义（9 种）

| 实体类型 | 含义 | 示例 |
|---|---|---|
| `MaterialSystem` | 铜合金体系名称 | Cu, CuZn, CuNiSi, CuCrZr |
| `Phase` | 物相名称 | α相, β相, δ-Ni2Si |
| `ConductivityMechanism` | 导电机制 | 析出净化, 溶质散射 |
| `StrengtheningMechanism` | 强化机制 | 固溶强化, 析出强化, 细晶强化 |
| `PrecipitationMechanism` | 析出/相变机制 | 共格析出, spinodal decomposition |
| `Property` | 材料性能 | 抗拉强度, 导电率, 硬度 |
| `ServicePerformance` | 服役性能 | 抗氧化性, 耐磨性 |
| `Application` | 应用场景 | 电子工业, 汽车连接器 |
| `FailureMode` | 失效模式 | 应力腐蚀开裂, 脱锌 |

### 实体输出结构

```json
{
  "entity_text": "α相",
  "entity_type": "Phase",
  "normalized_name": "α相",
  "aliases": ["α-solid solution"],
  "definition": "锡在铜中的固溶体，具有面心立方晶格结构。",
  "attributes": [
    {"key": "晶格结构", "value": "面心立方晶格"},
    {"key": "化学成分", "value": "Sn在Cu中的固溶体"}
  ],
  "evidence": "α相是锡在铜中的固溶体，面心立方晶格。",
  "confidence": 0.95
}
```

## 关系类型定义（7 种）

| 关系类型 | 方向 | 示例 |
|---|---|---|
| `has_phase` | 体系 → 物相 | CuNiSi → α相 |
| `strengthened_by` | 体系 → 强化机制 | CuNiSi → 析出强化 |
| `has_property` | 体系/物相 → 性能 | CuSn → 耐蚀性 |
| `used_in` | 体系 → 应用场景 | CuNiSi → 电子工业 |
| `failure_mode_of` | 失效模式 → 体系 | 脱锌 → CuNiSi |
| `mechanism_of` | 机制 → 体系 | 析出净化 → CuNiSi |
| `contains` | 体系 → 子体系 | 铜合金 → CuNiSi |

## 存储映射说明

### Neo4j 实体节点（双 Label 策略）

每个实体节点同时拥有两个 Label：
- `:base` — LightRAG 查询层使用，保持框架兼容
- `:Phase` / `:MaterialSystem` / ... — 语义查询使用，对应 entity_type

```cypher
(n:base:Phase {
    entity_name: "α相",          -- normalized_name
    entity_type: "Phase",        -- entity_type
    description: "锡在铜中的固溶体", -- definition
    attr_晶格结构: "面心立方晶格",  -- attributes 展开
    attr_化学成分: "Sn在Cu中的固溶体",
    source_id: "doc-xxx",        -- 自动生成
    file_path: "test.md",        -- 溯源文件路径
    created_at: 1715000000       -- 自动生成的时间戳
})
```

### Neo4j 关系边（有向 + 类型 Label）

关系边使用 relation_type 作为边 Label，保留方向语义：

```cypher
(a:base:MaterialSystem {entity_name: "CuNiSi"})
  -[:strengthened_by {
      evidence: "通过δ-Ni2Si析出相实现强化",
      source_id: "doc-xxx",
      weight: 1.0,
      created_at: 1715000000
  }]->
(b:base:StrengtheningMechanism {entity_name: "析出强化"})
```

### LLM 字段 → Neo4j 映射表

| LLM 字段 | Neo4j 节点属性 | 说明 |
|---|---|---|
| `normalized_name` | `entity_name` | 实体唯一标识（MERGE key） |
| `entity_type` | `entity_type` + Label | 实体类型，同时作为 Neo4j Label |
| `definition` | `description` | 实体描述 |
| `entity_text` | `original_text` | 原文中的原始文本（若与 normalized_name 不同） |
| `attributes[].{key, value}` | `attr_{key}` | 动态属性（如 `attr_晶格结构`） |
| 自动生成 | `source_id` | 文档溯源 ID |
| 自动生成 | `file_path` | 文件路径标识 |
| 自动生成 | `created_at` | 写入时间戳 |

| LLM 字段 | Neo4j 边属性 | 说明 |
|---|---|---|
| `src_entity` / `tgt_entity` | MERGE 匹配 key | 通过 entity_name 匹配两端节点 |
| `relation_type` | 边 Label | 如 `:has_phase`, `:strengthened_by` |
| `evidence` | `evidence` | 原文依据 |
| `description` | `description` | 关系描述 |
| 自动生成 | `source_id` | 文档溯源 ID |
| 自动生成 | `weight` | 权重（默认 1.0） |
| 自动生成 | `created_at` | 写入时间戳 |

### 写入流程

`insert_with_custom_schema()` 不调用 LightRAG 的 `ainsert_custom_kg`，而是使用 `direct_ingest()` 直接编排所有存储层：

| 存储层 | 写入内容 | 说明 |
|--------|---------|------|
| Qdrant `entities_vdb` | 实体向量 | `entity_name + "\n" + description` 的 embedding，用于语义搜索 |
| Qdrant `chunks_vdb` | chunk 向量 | 原文内容的 embedding，用于检索上下文 |
| JSON KV `text_chunks` | chunk 元数据 | 分块内容、source_id、token 数等 |
| JSON KV `full_entities` | 文档→实体映射 | 用于按文档 ID 删除 |
| JSON KV `full_relations` | 文档→关系映射 | 用于按文档 ID 删除 |
| JSON KV `doc_status` | 文档状态 | 标记为 PROCESSED |
| Neo4j `Neo4jDirectWriter` | 实体节点 + 有向边 | 双 label + 类型边 label |
| ~~Qdrant `relationships_vdb`~~ | ~~关系向量~~ | **不写入**（关系不向量化） |

写入后自动清理历史残留的 `:DIRECTED` 边。

### Neo4j 查询示例

```cypher
-- 按实体类型查询（语义查询）
MATCH (n:Phase) RETURN n.entity_name, n.description LIMIT 5

-- 按动态属性查询
MATCH (n) WHERE n.attr_晶格结构 = '面心立方晶格' RETURN n.entity_name

-- 按关系类型查询（有向边）
MATCH (a:MaterialSystem)-[r:has_phase]->(b:Phase)
RETURN a.entity_name AS 体系, b.entity_name AS 物相

-- 查询实体的所有出边关系
MATCH (n {entity_name: 'CuNiSi'})-[r]->(m)
RETURN type(r) AS 关系类型, m.entity_name AS 目标实体

-- 双 label 兼容查询（LightRAG 查询层使用）
MATCH (n:base) RETURN n.entity_name, n.entity_type LIMIT 5
```

### 删除方法

`CustomEntityService` 提供三个删除方法，同时清理 Neo4j、Qdrant 和 KV store：

```python
svc = CustomEntityService()

# 按文档 ID 删除（删除该文档创建的所有实体、关系、向量）
svc.delete_by_doc_id("my_doc_001")

# 按实体名删除（实体及其所有关系边）
svc.delete_by_entity("CuNiSi")

# 删除两个实体之间的所有关系边
svc.delete_by_relation("CuNiSi", "析出强化")
```

删除流程：
1. 先从 `full_entities` KV store 获取该文档的实体列表
2. 通过 `Neo4jDirectWriter` 删除 Neo4j 中的实体节点（`DETACH DELETE` 自动清理所有边）
3. 调用 LightRAG 原方法清理 Qdrant 向量和 KV store 数据

> **注意：** `DocumentService` 的删除方法只适用于 `rag.insert()` 插入的文档。`CustomEntityService` 的删除方法用于自定义 schema 插入的文档。

## 常见问题

### 1. Docker 端口被占用

Neo4j 使用 7474（Browser）和 7687（Bolt），Qdrant 使用 6333（REST）和 6334（gRPC）。

```bash
# 检查端口占用
netstat -ano | findstr "7687"
netstat -ano | findstr "6333"

# 如有冲突，修改 docker-compose.yml 中的 ports 映射，并同步修改 docker.env
```

### 2. Neo4j APOC 插件未加载

```bash
# 验证 APOC 是否可用
docker exec lightrag-neo4j cypher-shell -u neo4j -p "LightRAG2026!" \
  "CALL apoc.util.validate(false, 'test', [])"

# 如果报错，检查插件目录
docker exec lightrag-neo4j ls /plugins/

# 重新加载
docker-compose restart neo4j
```

### 3. Qdrant 连接失败

```bash
# 检查 Qdrant 是否启动
curl http://localhost:6333/healthz

# 检查 Docker 状态
docker ps | grep qdrant

# 如果端口未映射，可能是旧容器未应用最新配置
docker-compose down && docker-compose up -d
```

### 4. LLM 返回 502 或超时

```bash
# 直接测试 LLM 服务连通性
curl http://your_llm_host:port/v1/models

# LightRAG 默认超时为 180 秒，长文本抽取可能需要更长时间
# 可通过环境变量调整：
# export LLM_TIMEOUT=300
```

### 5. 清理测试数据

```bash
# Neo4j 中删除所有数据
docker exec lightrag-neo4j cypher-shell -u neo4j -p "LightRAG2026!" \
  "MATCH (n) DETACH DELETE n"

# Qdrant 中删除所有 collection
# 通过 Qdrant Dashboard: http://localhost:6333/dashboard

# 清理本地缓存
rm -rf rag_storage/ rag_storage_test/
```

### 6. 在 Linux 上运行

`docker-compose.yml` 完全适配 Linux，无需修改。如果 Neo4j 数据目录遇到权限问题：

```yaml
# 在 neo4j service 下添加
user: "1000:1000"
```
