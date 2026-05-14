# LightRAG 使用手册

> 面向初学者的完整使用指南，包含安装、配置、示例和常见问题

---

## 目录

1. [快速开始](#1-快速开始)
2. [核心概念](#2-核心概念)
3. [LLM 和 Embedding 函数配置](#3-llm-和-embedding-函数配置)
4. [文档插入](#4-文档插入)
5. [知识查询](#5-知识查询)
6. [自定义知识图谱插入](#6-自定义知识图谱插入)
7. [知识图谱管理](#7-知识图谱管理)
8. [文档删除](#8-文档删除)
9. [使用 Neo4j + Qdrant](#9-使用-neo4j--qdrant)
10. [自定义实体类型](#10-自定义实体类型)
11. [环境变量配置](#11-环境变量配置)
12. [异步操作](#12-异步操作)
13. [常见问题排查](#13-常见问题排查)
14. [项目实战：铜合金知识图谱](#14-项目实战铜合金知识图谱)

---

## 1. 快速开始

### 1.1 安装

```bash
# 使用 conda 环境
conda create -n lightrag python=3.12
conda activate lightrag

# 安装 LightRAG
pip install lightrag-hku
```

### 1.2 最简示例

```python
from lightrag import LightRAG, QueryParam
from lightrag.llm.openai import openai_complete, openai_embed
from lightrag.utils import EmbeddingFunc
from functools import partial

# 创建 LightRAG 实例
rag = LightRAG(
    working_dir="./demos",                       # 数据存储目录
    llm_model_func=partial(                      # LLM 调用函数
        openai_complete,
        api_key="your-api-key",
        base_url="https://api.openai.com/v1",
    ),
    embedding_func=EmbeddingFunc(               # Embedding 函数
        embedding_dim=1536,
        func=partial(
            openai_embed.func,
            model="text-embedding-3-small",
            api_key="your-api-key",
        ),
    ),
)

# 初始化存储
await rag.initialize_storages()

# 插入文档
rag.insert("铜是一种优良的导电材料。")

# 查询
result = rag.query("铜有什么特点？")
print(result)
```

### 1.3 同步 vs 异步

LightRAG 的所有操作都有同步和异步两个版本：

```python
# 同步版本（推荐新手使用，简单直接）
rag.insert(text)
rag.query(question)
rag.delete_by_doc_id(doc_id)

# 异步版本（适合高并发场景）
await rag.ainsert(text)
await rag.aquery(question)
await rag.adelete_by_doc_id(doc_id)
```

**内部实现**：同步方法内部通过 `asyncio.new_event_loop().run_until_complete()` 调用对应的异步方法。

---

## 2. 核心概念

### 2.1 LightRAG 处理的是什么？

传统 RAG 和 LightRAG 的区别：

```
传统 RAG:
  文档 → 分块 → 向量库 → 相似度检索 → LLM 回答

LightRAG:
  文档 → 分块 → LLM 抽取{实体, 关系} → 知识图谱 + 向量库 → 检索 → LLM 回答
                                              ↑
                                    多了结构化知识层
```

**优势：**
- **全局知识理解**：不仅找到"最相似"的文本，还能通过图谱遍历发现关联
- **减少幻觉**：基于结构化的实体和关系回答，而非单纯文本匹配

### 2.2 存储分层

```
┌─────────────────────────────────────────────────┐
│ 你的问题："CuNiSi 的强化机制有哪些？"              │
├─────────────────────────────────────────────────┤
│ 1. 向量检索（找相似实体/关系）                      │
│    → CuNiSi（相似度 0.85）                         │
│    → 析出强化（相似度 0.78）                       │
│    → 固溶强化（相似度 0.72）                       │
├─────────────────────────────────────────────────┤
│ 2. 图遍历（找关联关系）                            │
│    CuNiSi --strengthened_by--> 析出强化            │
│    CuNiSi --strengthened_by--> 固溶强化            │
│    CuNiSi --has_phase--> δ-Ni2Si                  │
├─────────────────────────────────────────────────┤
│ 3. 获取关联文本块                                  │
│    "CuNiSi合金通过析出Ni2Si相产生强烈的析出强化..." │
├─────────────────────────────────────────────────┤
│ 4. 组装上下文，调用 LLM 生成回答                    │
└─────────────────────────────────────────────────┘
```

---

## 3. LLM 和 Embedding 函数配置

### 3.1 LLM 函数

LightRAG 需要调用 LLM 做三件事：
1. **实体抽取**：从文本中识别实体和关系
2. **摘要合并**：合并重复实体的描述
3. **回答生成**：基于检索结果生成自然语言回答

**使用 OpenAI 兼容 API（推荐）：**

```python
from functools import partial
from lightrag.llm.openai import openai_complete

llm_func = partial(
    openai_complete,
    api_key="your-key",           # API 密钥
    base_url="https://api.openai.com/v1",  # API 地址
)

# 用于非 OpenAI 的兼容服务（如 vLLM、Ollama 等）
llm_func = partial(
    openai_complete,
    api_key="not-needed",
    base_url="http://localhost:8000/v1",    # 本地服务
)
```

### 3.2 Embedding 函数

Embedding 用于将文本转换为向量，是检索的核心。

```python
from lightrag.utils import EmbeddingFunc
from lightrag.llm.openai import openai_embed

embed_func = EmbeddingFunc(
    embedding_dim=1536,              # 向量维度（必须与模型匹配）
    func=partial(
        openai_embed.func,           # 注意使用 .func 避免双重包装
        model="text-embedding-3-small",
        api_key="your-key",
        base_url="https://api.openai.com/v1",
    ),
)
```

**⚠️ 重要**：使用 `openai_embed.func` 而不是 `openai_embed`，因为 `openai_embed` 已经被 `@wrap_embedding_func_with_attrs` 装饰器包装过，直接使用会导致双重包装。

### 3.3 使用 bge-m3 示例

```python
embed_func = EmbeddingFunc(
    embedding_dim=1024,              # bge-m3 的向量维度
    func=partial(
        openai_embed.func,
        model="bge-m3",
        api_key="your-key",
        base_url="http://your-server:8000/v1/",
    ),
    model_name="bge-m3",
)
```

### 3.4 Rerank 函数（可选）

对检索结果进行二次排序，提升精度：

```python
from lightrag.rerank import generic_rerank_api

rerank_func = partial(
    generic_rerank_api,
    model="bge-reranker-v2-m3",
    base_url="http://your-server:8000/v1/rerank",
    api_key="your-key",
)
```

---

## 4. 文档插入

### 4.1 基本插入

```python
# 单条文本
track_id = rag.insert("铜是一种优良的导电材料。")
print(f"跟踪 ID: {track_id}")

# 多条文本
track_id = rag.insert(["文本1", "文本2", "文本3"])
```

### 4.2 带自定义 ID 和文件路径

```python
track_id = rag.insert(
    text,
    ids="my_doc_001",                        # 自定义文档 ID
    file_paths="/path/to/source.md",          # 源文件路径（用于溯源）
)
```

### 4.3 按指定字符分块

```python
# 按换行符分块
rag.insert(text, split_by_character="\n")

# 仅按字符分块（不按 token 再分）
rag.insert(text, split_by_character="\n", split_by_character_only=True)
```

### 4.4 查询处理状态

```python
status = rag.get_processing_status()
# {'pending': 0, 'processing': 1, 'processed': 5, 'failed': 0}

# 获取失败的文档
from lightrag.base import DocStatus
failed_docs = rag.get_docs_by_status(DocStatus.FAILED)
for doc_id, status_obj in failed_docs.items():
    print(f"文档: {doc_id}, 错误: {status_obj.error_msg}")
```

---

## 5. 知识查询

### 5.1 基本查询

```python
# 默认 mix 模式（推荐）
answer = rag.query("铜合金有哪些类型？")
print(answer)

# 流式输出
param = QueryParam(stream=True)
result = rag.query("铜合金有哪些类型？", param=param)
for chunk in result:
    print(chunk, end="", flush=True)
```

### 5.2 查询模式对比

```python
# local 模式：关注具体实体
param = QueryParam(mode="local")
rag.query("CuNiSi 是什么材料？", param)

# global 模式：关注全局关系
param = QueryParam(mode="global")
rag.query("铜合金的强化机制有哪些共性？", param)

# mix 模式：综合（默认）
param = QueryParam(mode="mix")
rag.query("CuNiSi 的强化机制有哪些？", param)

# bypass 模式：不检索，直接问 LLM
param = QueryParam(mode="bypass")
rag.query("什么是量子力学？", param)
```

### 5.3 只检索数据，不生成回答

```python
# 返回结构化数据，不调用 LLM 生成
data = rag.query_data("CuNiSi 的特点？")
print(data)
# {
#   "status": "success",
#   "data": {
#     "entities": [...],      # 检索到的实体
#     "relationships": [...], # 检索到的关系
#     "chunks": [...],        # 检索到的文本块
#     "references": [...]     # 参考文献/来源
#   }
# }
```

### 5.4 自定义系统提示词

```python
# 在查询时自定义回答模板
custom_prompt = "请用要点列表的方式回答，并标注来源。"
answer = rag.query("问题...", system_prompt=custom_prompt)
```

---

## 6. 自定义知识图谱插入

### 6.1 适用场景

当你已经知道实体和关系，不想让 LLM 重新抽取时，使用 `insert_custom_kg`。

### 6.2 数据格式

```python
custom_kg = {
    "chunks": [
        {
            "content": "CuNiSi 是一种铜合金，通过 Ni2Si 相的析出产生强化效果。",
            "source_id": "doc_001",
            "chunk_order_index": 0,
        }
    ],
    "entities": [
        {
            "entity_name": "CuNiSi",
            "entity_type": "MaterialSystem",
            "description": "铜镍硅合金",
            "source_id": "doc_001",
        },
        {
            "entity_name": "析出强化",
            "entity_type": "StrengtheningMechanism",
            "description": "通过第二相析出提高材料强度",
            "source_id": "doc_001",
        },
    ],
    "relationships": [
        {
            "src_id": "CuNiSi",
            "tgt_id": "析出强化",
            "description": "CuNiSi 通过 Ni2Si 相析出产生强化",
            "keywords": "析出强化",
            "weight": 1.0,
            "source_id": "doc_001",
        }
    ],
}

rag.insert_custom_kg(custom_kg)
```

### 6.3 注意事项

- `entities` 中的 `entity_name` 必须与 `relationships` 中的 `src_id`/`tgt_id` 对应
- `source_id` 必须与 `chunks` 中的 `source_id` 一致
- `entity_type` 可以是任意字符串（但建议保持一致的命名规范）

---

## 7. 知识图谱管理

### 7.1 查询实体信息

```python
# 获取同步版本需要事件循环
import asyncio
loop = asyncio.new_event_loop()
info = loop.run_until_complete(rag.get_entity_info("CuNiSi"))
loop.close()

print(info)
# {
#   "entity_name": "CuNiSi",
#   "entity_type": "MaterialSystem",
#   "description": "铜镍硅合金...",
#   "source_id": "chunk-xxx",
# }
```

### 7.2 查询关系信息

```python
loop = asyncio.new_event_loop()
info = loop.run_until_complete(rag.get_relation_info("CuNiSi", "析出强化"))
loop.close()
```

### 7.3 获取知识图谱子图

```python
loop = asyncio.new_event_loop()
graph = loop.run_until_complete(
    rag.get_knowledge_graph("CuNiSi", max_depth=2, max_nodes=100)
)
loop.close()

print(f"节点数: {len(graph.nodes)}")
print(f"边数: {len(graph.edges)}")

for node in graph.nodes:
    print(f"节点: {node.labels}, 属性: {node.properties}")

for edge in graph.edges:
    print(f"边: {edge.source} -> {edge.target}, 类型: {edge.type}")
```

### 7.4 获取所有实体标签

```python
loop = asyncio.new_event_loop()
labels = loop.run_until_complete(rag.get_graph_labels())
loop.close()
print(labels)  # ['CuNiSi', '析出强化', '固溶强化', ...]
```

---

## 8. 文档删除

### 8.1 按文档 ID 删除

```python
result = rag.delete_by_doc_id("doc_001")
print(result.status)    # "success" / "not_found" / "fail"
print(result.message)   # 详细信息
```

### 8.2 按实体名删除

```python
result = rag.delete_by_entity("CuNiSi")
# 会同时删除该实体的所有关系
```

### 8.3 删除关系

```python
result = rag.delete_by_relation("CuNiSi", "析出强化")
```

---

## 9. 使用 Neo4j + Qdrant

### 9.1 启动服务

```bash
cd d:\LightRAG
docker-compose up -d
```

### 9.2 创建实例

```python
from service.kg_config import create_lightrag_neo4j_qdrant

rag = create_lightrag_neo4j_qdrant(
    working_dir="./rag_storage",
    env_path="docker.env",
)
```

### 9.3 手动创建（不通过 kg_config）

```python
import os
from functools import partial
from lightrag import LightRAG
from lightrag.utils import EmbeddingFunc
from lightrag.llm.openai import openai_complete, openai_embed

# 设置环境变量
os.environ["NEO4J_URI"] = "bolt://localhost:7687"
os.environ["NEO4J_USERNAME"] = "neo4j"
os.environ["NEO4J_PASSWORD"] = "LightRAG2026!"
os.environ["QDRANT_URL"] = "http://localhost:6333"

rag = LightRAG(
    working_dir="./rag_storage",
    graph_storage="Neo4JStorage",
    vector_storage="QdrantVectorDBStorage",
    kv_storage="JsonKVStorage",
    doc_status_storage="JsonDocStatusStorage",
    llm_model_func=partial(openai_complete, base_url="...", api_key="..."),
    embedding_func=EmbeddingFunc(
        embedding_dim=1024,
        func=partial(openai_embed.func, model="bge-m3", base_url="...", api_key="..."),
    ),
)

# 必须手动初始化存储
import asyncio
loop = asyncio.new_event_loop()
loop.run_until_complete(rag.initialize_storages())
loop.close()
```

---

## 10. 自定义实体类型

### 10.1 在创建时指定

```python
rag = LightRAG(
    ...,
    addon_params={
        "language": "Chinese",
        "entity_types": [
            "MaterialSystem",       # 材料体系
            "Phase",                # 物相
            "StrengtheningMechanism",  # 强化机制
            "Property",             # 性能
            "Application",          # 应用
        ],
    },
)
```

### 10.2 完全自定义 Prompt 抽取

如果默认的实体抽取 prompt 不满足需求，可以：
1. 先用 LLM 按自己的 schema 抽取实体
2. 用 `insert_custom_kg` 直接插入

参见第 6 节示例。

---

## 11. 环境变量配置

几乎所有默认值都可以通过环境变量覆盖：

```bash
# 在 .env 文件中设置
TOP_K=40                    # 检索数量
CHUNK_SIZE=1200             # 分块大小
CHUNK_OVERLAP_SIZE=100      # 分块重叠
MAX_ASYNC=4                 # 最大并发数
LLM_TIMEOUT=180             # LLM 超时（秒）
EMBEDDING_TIMEOUT=30        # Embedding 超时
COSINE_THRESHOLD=0.2        # 向量相似度阈值
WORKSPACE="my_project"      # 工作空间（数据隔离）
ENTITY_TYPES=["Type1", "Type2"]  # 实体类型
SUMMARY_LANGUAGE="Chinese"  # 摘要语言
```

---

## 12. 异步操作

### 12.1 基本用法

```python
import asyncio
from lightrag import LightRAG, QueryParam

async def main():
    rag = LightRAG(...)
    await rag.initialize_storages()

    # 异步插入
    track_id = await rag.ainsert("文本内容")

    # 异步查询
    answer = await rag.aquery("问题")

asyncio.run(main())
```

### 12.2 流式查询

```python
async def stream_query():
    param = QueryParam(stream=True)
    result = await rag.aquery("问题", param=param)

    async for chunk in result:
        print(chunk, end="", flush=True)

asyncio.run(stream_query())
```

---

## 13. 常见问题排查

### 13.1 `StorageNotInitializedError`

**原因**：使用 Neo4j/Qdrant 等外部存储时没有调用 `initialize_storages()`。

**解决**：
```python
import asyncio
loop = asyncio.new_event_loop()
loop.run_until_complete(rag.initialize_storages())
loop.close()
```

### 13.2 Embedding 函数报错

**原因**：使用了被装饰器包装过的函数（如 `openai_embed`）而没有用 `.func` 访问原始函数。

**解决**：
```python
# ❌ 错误
embedding_func=EmbeddingFunc(func=openai_embed, ...)

# ✅ 正确
embedding_func=EmbeddingFunc(func=partial(openai_embed.func, ...), ...)
```

### 13.3 Neo4j APOC 插件不可用

**原因**：Neo4j 5.x 社区版默认不带完整 APOC，或插件未正确加载。

**解决**：
1. 检查 `docker-compose.yml` 中是否设置了 `NEO4J_PLUGINS=["apoc"]`
2. 重启容器：`docker-compose down && docker-compose up -d`
3. 进入容器检查：`docker exec lightrag-neo4j ls /plugins/`

### 13.4 Qdrant 端口被占用

**解决**：
```bash
# 查看占用端口的容器
docker ps --filter publish=6333

# 停止旧容器
docker stop <container_name>

# 重新启动
docker-compose up -d
```

### 13.5 实体抽取为空

**可能原因**：
1. LLM 模型能力不足，无法识别自定义实体类型
2. Prompt 中的实体类型与输入文本不匹配
3. 文本太短或信息太少

**排查方法**：
```python
# 用 bypass 模式直接问 LLM
param = QueryParam(mode="bypass")
answer = rag.query("请列出以下文本中的所有材料名称：{text}", param)
```

### 13.6 查询回答为空或"找不到信息"

**可能原因**：
1. 知识图谱中没有相关实体
2. 向量检索的 cosine 阈值太高
3. 文档还没处理完成（状态仍是 PENDING）

**排查方法**：
```python
# 检查处理状态
print(rag.get_processing_status())

# 用 query_data 查看检索到了什么
data = rag.query_data("你的问题")
print(data.get("data", {}).get("entities", []))
print(data.get("data", {}).get("relationships", []))
```

---

## 14. 项目实战：铜合金知识图谱

### 14.1 完整流程

```python
# 1. 创建实例
from service import CustomEntityService

service = CustomEntityService()

# 2. 插入文档（使用自定义实体 schema）
text = """铜合金CuNiSi具有优异的导电性和机械强度，
广泛应用于电子工业中的连接器制造。其主要强化机制包括
固溶强化和析出强化。在时效处理过程中，CuNiSi合金中
会析出纳米级的Ni2Si相（δ相），产生强烈的析出强化效果。"""

result = service.insert_with_custom_schema(
    text=text,
    doc_id="cunisi_001",
    file_path="materials.md",
)

print(f"抽取了 {len(result['entities'])} 个实体")
print(f"抽取了 {len(result['relationships'])} 个关系")

# 3. 查询
answer = service.query("CuNiSi 有哪些强化机制？")
print(answer)

# 4. 查看知识图谱子图
from service import create_lightrag_neo4j_qdrant
import asyncio

rag = create_lightrag_neo4j_qdrant()
loop = asyncio.new_event_loop()
graph = loop.run_until_complete(rag.get_knowledge_graph("*", max_depth=2))
loop.close()

print(f"知识图谱包含 {len(graph.nodes)} 个节点，{len(graph.edges)} 条边")
```

### 14.2 批量处理文件

```python
import os

# 读取目录下所有 .md 文件
docs = []
for filename in os.listdir("materials/"):
    if filename.endswith(".md"):
        with open(f"materials/{filename}", "r", encoding="utf-8") as f:
            docs.append(f.read())

# 批量插入
service = CustomEntityService()
for i, doc in enumerate(docs):
    print(f"处理第 {i+1}/{len(docs)} 个文档...")
    result = service.insert_with_custom_schema(
        text=doc,
        doc_id=f"doc_{i:03d}",
        file_path=f"materials/{filename}",
    )
    print(f"  实体: {len(result['entities'])}, 关系: {len(result['relationships'])}")
```

---

*文档结束。祝使用愉快！*
