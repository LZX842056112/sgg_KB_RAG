# 掌柜智库项目(RAG)实战

## 5. 导入数据节点实现与测试

### 5.7 存入 Milvus (node_import_milvus)

**文件**: `app/import_process/node_import_milvus.py`

#### 节点作用与实现思路

**节点作用**: 数据加载流程的终点，负责将处理好的结构化数据（切片内容、元数据、向量）持久化存储到向量数据库中，构建可供即时查询的索引。

**实现思路**:

1.  **幂等性设计**: 在插入新数据前，根据 `item_name` 或文件 ID 清理旧数据，防止重复导入导致的数据污染。
2.  **Schema 适配**: 严格按照 Milvus 集合的 Schema 定义（主键、Dense字段、Sparse字段、JSON元数据字段）组织数据，确保插入成功率。
3.  **混合索引构建**: 确保存入的数据能够支持 Milvus 的 Hybrid Search（Dense + Sparse 加权），最大化检索效果。

#### 1. 导入与配置 

- **目的**: 导入必要的库（如 `pymilvus`）和项目工具类，配置 Milvus 集合名称。
- **关键点**:
  - `CHUNKS_COLLECTION_NAME`: 从环境变量获取集合名。
  - `add_running_task`: 记录任务执行状态。

```python
# 导入Milvus相关依赖
from pymilvus import DataType
# 导入自定义模块
from app.import_process.agent.state import ImportGraphState
from app.clients.milvus_utils import get_milvus_client
from app.utils.task_utils import add_running_task, add_done_task
from app.core.logger import logger,node_log,step_log
from app.conf.milvus_config import milvus_config

# 从配置文件读取切片集合名称，与配置解耦，便于环境切换
CHUNKS_COLLECTION_NAME = milvus_config.chunks_collection
```

#### 2. 核心辅助函数 

**功能**: 处理幂等性清理（删除旧数据）和字符串转义。

```python
from app.utils.escape_milvus_string_utils import escape_milvus_string
```

#### 3. 主流程定义

**函数**: `node_import_milvus`

**逻辑**:

1.  **Step 1**: 检查输入 (`step_1_check_input`)。
2.  **Step 2**: 准备环境 (`step_2_prepare_collection`)。
3.  **Step 3**: 清理旧数据 (`step_3_clean_old_data`)。
4.  **Step 4**: 插入数据 (`step_4_insert_data`)。

```python
"""
  1. 检查数据 chunks是否存在
  2. 前置准备工作 准备 milvus的集合和字段等
  3. 删除旧数据
  4. 查询chunks的数据即可
"""
@node_log("node_import_milvus")
def node_import_milvus(state: ImportGraphState) -> ImportGraphState:
    """
    节点: 导入向量库 (node_import_milvus)
    为什么叫这个名字: 将处理好的向量数据写入 Milvus 数据库。
    """
    # 准备日志和任务列表
    add_running_task(state['task_id'], "node_import_milvus")
    # 1. 检查数据 chunks是否存在
    chunks = state.get('chunks')
    if not chunks:
        logger.error("node_import_milvus: chunks数据不存在")
        raise ValueError("node_import_milvus: chunks数据不存在")
    # 2. 前置准备工作 创建 Milvus 集合和字段
    milvus_client = get_milvus_client()
    step_2_prepare_collection(milvus_client)
    # 3. 删除旧数据
    step_3_delete_old_data(milvus_client, state['item_name'])
    # 4. 插入chunks的数据即可
    with_id_chunks = step_4_insert_collections(milvus_client, chunks)
    state['chunks'] = with_id_chunks
    add_done_task(state['task_id'], 'node_import_milvus')
    return state
```

#### 4. 步骤 1: 检查输入

**功能**: 验证 `chunks` 是否存在，并提取 `dense_vector` 维度和 `item_name`。

```python
 # 1. 检查数据 chunks是否存在
    chunks = state.get('chunks')
    if not chunks:
        logger.error("node_import_milvus: chunks数据不存在")
        raise ValueError("node_import_milvus: chunks数据不存在")
```

#### 5. 步骤 2: 准备集合 

**功能**: 获取 Milvus 客户端，如果集合不存在则创建。

```python
@step_log("step_2_prepare_collection")
def step_2_prepare_collection(milvus_client):
    """
    准备和创建chunks对应的集合
    :param milvus_client:
    :return:
    """
    # 2. 判断是否存在集合（表），存在创建集合（表）
    if not milvus_client.has_collection(collection_name=milvus_config.chunks_collection):
        # 创建集合
        # 3.1. 创建集合对应的列的信息
        schema = milvus_client.create_schema(
            auto_id=True,  # 主键自增长
            enable_dynamic_field=True,  # 动态字段
        )

        # 3.2. Add fields to schema
        # pk file_title item_name dense_vector sparse_vector
        schema.add_field(field_name="chunk_id", datatype=DataType.INT64, is_primary=True, auto_id=True)
        schema.add_field(field_name="file_title", datatype=DataType.VARCHAR, max_length=65535)
        schema.add_field(field_name="item_name", datatype=DataType.VARCHAR, max_length=65535)
        schema.add_field(field_name="content", datatype=DataType.VARCHAR, max_length=65535)
        schema.add_field(field_name="title", datatype=DataType.VARCHAR, max_length=65535)
        schema.add_field(field_name="parent_title", datatype=DataType.VARCHAR, max_length=65535)
        schema.add_field(field_name="part", datatype=DataType.INT8)
        schema.add_field(field_name="dense_vector", datatype=DataType.FLOAT_VECTOR, dim=1024)
        schema.add_field(field_name="sparse_vector", datatype=DataType.SPARSE_FLOAT_VECTOR)
        # 3.3 查询快，配置索引
        index_params = milvus_client.prepare_index_params()

        index_params.add_index(
            field_name="dense_vector",  # 给哪个列创建索引 稠密
            index_name="dense_vector_index",  # 索引的名字
            index_type="HNSW",  # 配置查找所用的算法
            metric_type="COSINE",  # 配置向量匹配和对比的 IP COSINE
            params={"M": 32,  # Maximum number of neighbors each node can connect to in the graph
                    "efConstruction": 300},  # or "DAAT_WAND" or "TAAT_NAIVE"
        )
        """
           10000  M = 16  efConstruction = 200
           50000  M = 32  efConstruction = 300
           100000  M = 64  efConstruction = 400
           M:图中每个节点在层次结构的每个层级所能拥有的最大边数或连接数。M 越高，图的密度就越大，搜索结果的召回率和准确率也就越高，因为有更多的路径可以探索，但同时也会消耗更多内存，并由于连接数的增加而减慢插入时间。如上图所示，M = 5表示 HNSW 图中的每个节点最多与 5 个其他节点直接相连。这就形成了一个中等密度的图结构，节点有多条路径到达其他节点。
           efConstruction:索引构建过程中考虑的候选节点数量。efConstruction 越高，图的质量越好，但需要更多时间来构建。
        """
        index_params.add_index(
            field_name="sparse_vector",  # Name of the vector field to be indexed
            index_type="SPARSE_INVERTED_INDEX",  # Type of the index to create
            index_name="sparse_vector_index",  # Name of the index to create
            metric_type="IP",  # Metric type used to measure similarity
            # 只计算可能得高分的向量，跳过大量的 0
            params={"inverted_index_algo": "DAAT_MAXSCORE"},  # Algorithm used for building and querying the index
        )

        milvus_client.create_collection(
            collection_name=milvus_config.chunks_collection,
            schema=schema,  # 字段
            index_params=index_params  # 索引
        )
    return milvus_client
```

#### 6. 步骤 3: 清理旧数据 

**功能**: 根据 `item_name` 删除已存在的切片，确保幂等性。

```python
@step_log("step_3_delete_old_data")
def step_3_delete_old_data(milvus_client, item_name):
    """
    删除旧数据 根据item_name删除
    :param milvus_client:
    :param item_name:
    :return:
    """
    milvus_client.delete(collection_name=CHUNKS_COLLECTION_NAME,
                         filter=f"item_name=='{item_name}'")
    # 调用 load_collection() 会触发 Milvus 重新加载集合数据、刷新索引、清理已标记删除的数据，确保删除操作真正生效，避免新旧数据混杂导致检索错误。
    milvus_client.load_collection(collection_name=CHUNKS_COLLECTION_NAME)
```

#### 7. 步骤 4: 插入数据

**功能**: 移除临时 `chunk_id`，批量插入数据，并回填生成的 ID。

```python
@step_log("step_4_insert_collections")
def step_4_insert_collections(milvus_client,chunks):
    """
    插入集合的数据！
    :param milvus_client
    :param chunks:
    :return:  chunks -> 主键回显
    """
    insert_result = milvus_client.insert(collection_name=CHUNKS_COLLECTION_NAME, data=chunks)
    # 成功插入了几条
    insert_count = insert_result.get("insert_count",0)
    logger.info(f"完成了数据插入，成功插入了 {insert_count} 条数据")

    # 获取回显的ids
    ids = insert_result.get("ids",[])

    if ids and len(ids) == len(chunks):
        for index,chunk in enumerate(chunks):
            chunk['chunk_id'] = ids[index]

    return chunks
```

#### 8. 单元测试

您可以在 `node_import_milvus.py` 文件底部直接运行以下测试代码：

```python
if __name__ == '__main__':
    # --- 单元测试 ---
    # 目的：验证 Milvus 导入节点的完整流程，包括连接、创建集合、清理旧数据和插入新数据。
    import sys
    import os
    from dotenv import load_dotenv

    # 加载环境变量 (自动寻找项目根目录的 .env)
    current_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(os.path.dirname(current_dir))
    load_dotenv(os.path.join(project_root, ".env"))

    # 构造测试数据
    dim = 1024
    test_state = {
        "task_id": "test_milvus_task",
        "item_name":"测试项目_Milvus",
        "chunks": [
            {
                "content": "Milvus 测试文本 1",
                "title": "测试标题",
                "item_name": "测试项目_Milvus",  # 必须有 item_name，用于幂等清理
                "parent_title":"test.pdf",
                "part":1,
                "file_title": "test.pdf",
                "dense_vector": [0.1] * dim,  # 模拟 Dense Vector
                "sparse_vector": {1: 0.5, 10: 0.8}  # 模拟 Sparse Vector
            }
,
            {
                "content": "Milvus 测试文本 2",
                "title": "测试标题2",
                "item_name": "测试项目_Milvus2",  # 必须有 item_name，用于幂等清理
                "parent_title": "test.pdf2",
                "part": 1,
                "file_title": "test.pdf2",
                "dense_vector": [0.1] * dim,  # 模拟 Dense Vector
                "sparse_vector": {1: 0.5, 10: 0.8}  # 模拟 Sparse Vector
            }
        ]
    }

    print("正在执行 Milvus 导入节点测试...")
    try:
        # 检查必要的环境变量
        if not os.getenv("MILVUS_URL"):
            print("❌ 未设置 MILVUS_URL，无法连接 Milvus")
        elif not os.getenv("CHUNKS_COLLECTION"):
            print("❌ 未设置 CHUNKS_COLLECTION")
        else:
            # 执行节点函数
            result_state = node_import_milvus(test_state)

            # 验证结果
            chunks = result_state.get("chunks", [])
            if chunks and chunks[0].get("chunk_id"):
                print(f"✅ Milvus 导入测试通过，生成 ID: {chunks[0]['chunk_id']}")
            else:
                print("❌ 测试失败：未能获取 chunk_id")

    except Exception as e:
        print(f"❌ 测试失败: {e}")
```

