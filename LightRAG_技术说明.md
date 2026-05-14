# LightRAG 技术文档

> 版本：基于 HKUDS/LightRAG 源码分析
> 作者：Zirui Guo (https://github.com/HKUDS/LightRAG)
> 文档目标：帮助 Python 初学者理解 LightRAG 的架构与使用方式

---

## 目录

1. [什么是 LightRAG？](#1-什么是-lightrag)
2. [核心架构](#2-核心架构)
3. [模块详细说明](#3-模块详细说明)
4. [数据存储方式](#4-数据存储方式)
5. [知识图谱构建流程](#5-知识图谱构建流程)
6. [查询模式详解](#6-查询模式详解)
7. [可插拔存储后端](#7-可插拔存储后端)
8. [实体类型与 Prompt 模板](#8-实体类型与-prompt-模板)

---

## 1. 什么是 LightRAG？

LightRAG 是一个**检索增强生成（RAG）+ 知识图谱**框架。它的工作流程可以概括为：

```
输入文档 → 文本分块 → LLM 抽取实体/关系 → 存入知识图谱 + 向量数据库
                                                          ↓
用户提问 → 从图谱/向量库检索相关知识 → LLM 生成回答
```

与传统 RAG 的区别：**它不仅存文本向量，还构建了实体和关系的知识图谱**，使得回答既能利用全局知识结构，也能利用局部上下文。

---

## 2. 核心架构

```
┌─────────────────────────────────────────────────────┐
│                   LightRAG（主类）                    │
│                                                      │
│  ┌──────────┐  ┌──────────┐  ┌──────────┐          │
│  │ KV 存储   │  │ 向量存储  │  │ 图存储    │          │
│  │ Json/    │  │ Nano/    │  │ NetworkX │          │
│  │ Redis/PG │  │ Qdrant/  │  │ Neo4j/PG │          │
│  │ Mongo    │  │ Milvus   │  │ Mongo    │          │
│  └──────────┘  └──────────┘  └──────────┘          │
│                                                      │
│  ┌──────────────────────────────────────────┐       │
│  │         operate.py（核心操作）             │       │
│  │  chunking → extract → merge → query      │       │
│  └──────────────────────────────────────────┘       │
│                                                      │
│  ┌──────────────┐  ┌──────────────┐                │
│  │ LLM 函数      │  │ Embedding 函数 │                │
│  │ OpenAI/Ollama│  │ OpenAI/本地   │                │
│  └──────────────┘  └──────────────┘                │
└─────────────────────────────────────────────────────┘
```

**三个关键抽象：**

| 抽象 | 对应类 | 作用 |
|------|--------|------|
| **KV 存储** | `BaseKVStorage` | 存储文档原文、分块结果、实体数据、关系数据、LLM 缓存 |
| **向量存储** | `BaseVectorStorage` | 存储实体/关系/文本块的向量，用于相似度检索 |
| **图存储** | `BaseGraphStorage` | 存储实体节点和关系边，用于图谱遍历和查询 |

---

## 3. 模块详细说明

### 3.1 `lightrag.py` — 主类 LightRAG

这是你日常使用的核心类，是一个 `@dataclass`（数据类），拥有 50+ 个可配置参数。

**主要参数分组：**

| 分组 | 关键参数 | 默认值 | 说明 |
|------|---------|--------|------|
| 目录 | `working_dir` | `"./rag_storage"` | 缓存和临时文件目录 |
| 图存储 | `graph_storage` | `"NetworkXStorage"` | 图数据库后端 |
| 向量存储 | `vector_storage` | `"NanoVectorDBStorage"` | 向量数据库后端 |
| KV 存储 | `kv_storage` | `"JsonKVStorage"` | 键值存储后端 |
| 文档状态 | `doc_status_storage` | `"JsonDocStatusStorage"` | 文档处理状态追踪 |
| 分块 | `chunk_token_size` | 1200 | 每块最大 token 数 |
| 分块 | `chunk_overlap_token_size` | 100 | 块间重叠 token 数 |
| Embedding | `embedding_func` | **必须设置** | 文本向量化函数 |
| LLM | `llm_model_func` | **必须设置** | 大语言模型调用函数 |
| LLM | `llm_model_name` | `"gpt-4o-mini"` | 模型名称 |
| 查询 | `top_k` | 40 | 检索的实体/关系数量 |
| 查询 | `chunk_top_k` | 20 | 检索的文本块数量 |
| 查询 | `max_total_tokens` | 30000 | 查询上下文最大 token 数 |
| 重排序 | `rerank_model_func` | 可选 | 检索结果重排序函数 |

**核心方法：**

```
插入文档：
  insert(text)          → 同步插入，返回 track_id
  ainsert(text)         → 异步插入，返回 track_id
  insert_custom_kg(kg)  → 直接插入自定义知识图谱

查询：
  query(question)       → 同步查询，返回回答文本
  aquery(question)      → 异步查询，返回回答文本
  query_data(question)  → 仅检索数据，不生成回答（返回结构化 dict）

删除：
  delete_by_doc_id(id)  → 按文档 ID 删除
  delete_by_entity(name)→ 按实体名删除
  delete_by_relation(src, tgt) → 删除关系

知识图谱管理：
  get_entity_info(name) → 获取实体详情
  get_relation_info(src, tgt) → 获取关系详情
  get_knowledge_graph(label) → 获取子图
  edit_entity() / create_entity() / merge_entities() → CRUD 操作
  export_data() → 导出数据

生命周期：
  initialize_storages() → 初始化所有存储（使用 Neo4j/Qdrant 时必须调用）
  finalize_storages() → 清理资源
```

### 3.2 `base.py` — 基础类和数据类型

定义了所有存储后端的抽象基类和核心数据结构。

**QueryParam — 查询配置：**

```python
QueryParam(
    mode="mix",              # 检索模式：local/global/hybrid/naive/mix/bypass
    top_k=40,                # 检索实体/关系数
    chunk_top_k=20,          # 检索文本块数
    stream=False,            # 是否流式输出
    response_type="Multiple Paragraphs",  # 回答格式
    max_entity_tokens=6000,  # 实体上下文 token 上限
    max_relation_tokens=8000,# 关系上下文 token 上限
    max_total_tokens=30000,  # 总 token 上限
)
```

**DocStatus — 文档处理状态枚举：**

| 状态 | 含义 |
|------|------|
| `PENDING` | 等待处理 |
| `PROCESSING` | 正在处理 |
| `PREPROCESSED` | 预处理完成 |
| `PROCESSED` | 处理完成 |
| `FAILED` | 处理失败 |

### 3.3 `operate.py` — 核心操作流程

包含了整个知识图谱构建和查询的流水线逻辑：

**插入流水线（两阶段）：**

```
第一阶段（apipeline_enqueue_documents）：
  1. 验证输入，去重，过滤已处理的文档
  2. 将新文档加入队列，标记状态为 PENDING

第二阶段（apipeline_process_enqueue_documents）：
  3. 文本分块（chunking_by_token_size）
  4. 对每个块抽取实体和关系（extract_entities）
  5. 合并到图存储和向量存储（merge_nodes_and_edges）
  6. 更新文档状态为 PROCESSED
```

**查询流程：**

```
1. 根据 query 提取关键词
2. 在实体向量库中检索相关实体
3. 在关系向量库中检索相关关系
4. 获取关联的文本块
5. 组装上下文（实体描述 + 关系描述 + 文本块）
6. 调用 LLM 生成回答
```

### 3.4 `utils.py` — 工具函数

关键工具类和函数：

| 工具 | 说明 |
|------|------|
| `EmbeddingFunc` | 包装 embedding 函数，包含维度、最大 token 等元信息 |
| `Tokenizer` / `TiktokenTokenizer` | 分词器接口和 tiktoken 实现 |
| `compute_mdhash_id()` | 计算内容的 MD5 哈希值作为唯一 ID |
| `priority_limit_async_func_call()` | 异步调用限流装饰器 |
| `always_get_an_event_loop()` | 获取或创建 asyncio 事件循环 |
| `wrap_embedding_func_with_attrs` | 装饰器：为 embedding 函数附加属性 |

### 3.5 `prompt.py` — Prompt 模板

所有内置提示词都存储在 `PROMPTS` 字典中：

| 模板 Key | 用途 |
|----------|------|
| `entity_extraction_system_prompt` | 实体/关系抽取的系统提示词 |
| `entity_extraction_user_prompt` | 实体抽取的用户提示词模板 |
| `summarize_entity_descriptions` | 实体描述摘要合并提示词 |
| `rag_response` | RAG 回答模板（支持引用标注） |
| `naive_rag_response` | 朴素 RAG 回答模板 |
| `fail_response` | 找不到相关知识时的回答模板 |

### 3.6 `constants.py` — 配置常量

所有默认值的集中管理。大部分可以通过环境变量覆盖：

```
TOP_K=40              → 检索数量
CHUNK_SIZE=1200       → 分块大小
CHUNK_OVERLAP_SIZE=100→ 分块重叠
MAX_ASYNC=4           → 最大并发数
LLM_TIMEOUT=180       → LLM 超时（秒）
EMBEDDING_TIMEOUT=30  → Embedding 超时（秒）
```

### 3.7 `namespace.py` — 存储命名空间

定义了 12 个存储命名空间常量，每个对应一种数据类型：

```
KV_STORE_FULL_DOCS          → 完整文档内容
KV_STORE_TEXT_CHUNKS        → 文本分块
KV_STORE_LLM_RESPONSE_CACHE → LLM 响应缓存
KV_STORE_FULL_ENTITIES      → 完整实体数据
KV_STORE_FULL_RELATIONS     → 完整关系数据
KV_STORE_ENTITY_CHUNKS      → 实体→分块映射
KV_STORE_RELATION_CHUNKS    → 关系→分块映射
VECTOR_STORE_ENTITIES       → 实体向量
VECTOR_STORE_RELATIONSHIPS  → 关系向量
VECTOR_STORE_CHUNKS         → 分块向量
GRAPH_STORE_CHUNK_ENTITY_RELATION → 知识图谱
DOC_STATUS                  → 文档处理状态
```

---

## 4. 数据存储方式

### 4.1 默认存储（无外部依赖）

```
./rag_storage/
├── kv_store_full_docs.json          # 文档内容
├── kv_store_text_chunks.json        # 分块内容
├── kv_store_full_entities.json      # 实体数据
├── kv_store_full_relations.json     # 关系数据
├── kv_store_entity_chunks.json      # 实体→分块映射
├── kv_store_relation_chunks.json    # 关系→分块映射
├── kv_store_llm_response_cache.json # LLM 缓存
├── doc_status.json                  # 文档处理状态
├── vdb_entities.json                # 实体向量（nano-vectordb）
├── vdb_relationships.json           # 关系向量
├── vdb_chunks.json                  # 分块向量
└── graph_chunk_entity_relation.graphml  # 知识图谱（GraphML 格式）
```

### 4.2 生产级存储（需外部服务）

| 类型 | 后端 | 配置方式 | 需要环境变量 |
|------|------|---------|-------------|
| 图存储 | Neo4j | `graph_storage="Neo4JStorage"` | NEO4J_URI, NEO4J_USERNAME, NEO4J_PASSWORD |
| 向量存储 | Qdrant | `vector_storage="QdrantVectorDBStorage"` | QDRANT_URL |
| 向量存储 | Milvus | `vector_storage="MilvusVectorDBStorage"` | MILVUS_URI, MILVUS_DB_NAME |
| 向量存储 | PostgreSQL | `vector_storage="PGVectorStorage"` | POSTGRES_USER, POSTGRES_PASSWORD, POSTGRES_DATABASE |
| KV 存储 | Redis | `kv_storage="RedisKVStorage"` | REDIS_URI |

---

## 5. 知识图谱构建流程

以插入一段文本为例，完整流程如下：

```
1. 用户调用 rag.insert("铜合金CuNiSi具有优异的导电性...")

2. 第一阶段：入队
   - 计算文档 MD5 ID
   - 检查是否已处理过（去重）
   - 将文档加入队列，状态设为 PENDING
   - 返回 track_id

3. 第二阶段：处理队列
   a) 分块：
      "铜合金CuNiSi..." → [chunk_1, chunk_2, ...]
      按 token 数量（默认 1200）切分，重叠 100 token

   b) 实体抽取（每个 chunk）：
      调用 LLM，使用 PROMPTS["entity_extraction_system_prompt"]
      抽取结果示例：
      {
        "entities": [
          {"entity_text": "CuNiSi", "entity_type": "MaterialSystem", ...},
          {"entity_text": "析出强化", "entity_type": "StrengtheningMechanism", ...},
        ],
        "relationships": [
          {"src": "CuNiSi", "tgt": "析出强化", "description": "..."},
        ]
      }

   c) 合并到存储：
      - 实体节点 → 图存储（NetworkX/Neo4j）
      - 实体描述 → 向量存储（NanoVectorDB/Qdrant）
      - 关系边 → 图存储
      - 关系描述 → 向量存储
      - 文本块 → KV 存储 + 向量存储

   d) 更新文档状态为 PROCESSED
```

**实体关系合并策略：**

当同一实体/关系出现在多个文档中时：
- 实体描述会被 LLM 摘要合并（当描述数量超过 `force_llm_summary_on_merge`，默认 8 条）
- 关系描述同样会被摘要合并
- 源 ID 列表会累积记录（默认使用 FIFO 策略限制数量）

---

## 6. 查询模式详解

| 模式 | 检索范围 | 适用场景 |
|------|---------|---------|
| **local** | 以实体为中心，检索与问题直接相关的实体 | 查询具体实体的详细信息 |
| **global** | 以关系为中心，利用全局知识结构 | 查询跨实体的关联和趋势 |
| **hybrid** | local + global 轮询合并 | 综合检索 |
| **mix**（推荐） | 知识图谱 + 向量检索混合 | 大多数问答场景 |
| **naive** | 纯向量搜索，不使用知识图谱 | 简单文本匹配 |
| **bypass** | 不调用检索，直接问 LLM | 不需要 RAG 的通用问答 |

**查询内部流程：**

```
1. 调用 rag.query("铜合金有哪些强化机制？", param=QueryParam(mode="mix"))

2. 关键词提取：
   - 调用 LLM 从 query 中提取高层关键词（hl_keywords）
   - 和低层关键词（ll_keywords）

3. 实体检索：
   - 用关键词在实体向量库中检索 top_k 实体

4. 关系检索：
   - 用关键词在关系向量库中检索 top_k 关系

5. 文本块检索：
   - 获取与实体/关系关联的文本块

6. 上下文组装：
   - 实体描述 + 关系描述 + 文本块内容
   - 使用 PROMPTS["rag_response"] 作为模板

7. LLM 生成回答
```

---

## 7. 可插拔存储后端

LightRAG 的存储设计采用了**策略模式**：

```python
# 注册表
STORAGE_IMPLEMENTATIONS = {
    "KV_STORAGE": {
        "implementations": ["JsonKVStorage", "RedisKVStorage", "PGKVStorage", ...],
    },
    "GRAPH_STORAGE": {
        "implementations": ["NetworkXStorage", "Neo4JStorage", "PGGraphStorage", ...],
    },
    "VECTOR_STORAGE": {
        "implementations": ["NanoVectorDBStorage", "QdrantVectorDBStorage", ...],
    },
    "DOC_STATUS_STORAGE": {
        "implementations": ["JsonDocStatusStorage", "RedisDocStatusStorage", ...],
    },
}
```

每个存储后端必须实现对应抽象基类的方法。例如图存储必须实现：
- `upsert_node()` / `upsert_edge()` — 插入/更新
- `get_node()` / `get_edge()` — 查询
- `has_node()` / `has_edge()` — 存在性检查
- `get_all_labels()` / `get_knowledge_graph()` — 图遍历
- `initialize()` / `finalize()` — 生命周期

---

## 8. 实体类型与 Prompt 模板

### 8.1 默认实体类型（11 种）

LightRAG 默认抽取以下 11 种实体：

```
Person（人物）、Creature（生物）、Organization（组织）、
Location（地点）、Event（事件）、Concept（概念）、
Method（方法）、Content（内容）、Data（数据）、
Artifact（人工制品）、NaturalObject（自然物体）
```

### 8.2 自定义实体类型

在 `addon_params` 中指定：

```python
rag = LightRAG(
    addon_params={
        "language": "Chinese",
        "entity_types": ["MaterialSystem", "Phase", "Property", ...],
    },
    ...
)
```

### 8.3 Prompt 模板结构

```python
PROMPTS = {
    # 分隔符
    "DEFAULT_TUPLE_DELIMITER": "<|＃|>",        # 字段分隔
    "DEFAULT_RECORD_DELIMITER": "<|##|>",      # 记录分隔
    "DEFAULT_COMPLETION_DELIMITER": "<|COMPLETE|>",

    # 系统提示词（定义角色和规则）
    "entity_extraction_system_prompt": "...",
    "summarize_entity_descriptions": "...",
    "rag_response": "...",

    # 用户提示词模板（使用 {input_text} 等占位符）
    "entity_extraction_user_prompt": "...",

    # 失败响应
    "fail_response": "...",
}
```

---

*文档结束。继续阅读使用手册了解如何实际操作。*
