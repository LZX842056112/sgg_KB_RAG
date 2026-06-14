# 掌柜智库 RAG 系统 - Code Wiki

## 目录

1. [项目概述](#项目概述)
2. [整体架构](#整体架构)
3. [目录结构](#目录结构)
4. [核心模块详解](#核心模块详解)
   - 4.1 [导入流水线 (Import Process)](#导入流水线-import-process)
   - 4.2 [查询流水线 (Query Process)](#查询流水线-query-process)
   - 4.3 [Web 服务层](#web-服务层)
5. [状态管理](#状态管理)
6. [配置管理](#配置管理)
7. [工具类说明](#工具类说明)
8. [依赖关系](#依赖关系)
9. [运行方式](#运行方式)
10. [技术栈总结](#技术栈总结)

---

## 项目概述

### 项目定位

**掌柜智库** 是一套基于 RAG（Retrieval-Augmented Generation，检索增强生成）技术构建的**企业级私有知识库智能问答系统**。系统采用 LangGraph 作为工作流编排核心，围绕"先检索、再生成"的核心理念，为企业提供高可信、可追溯、可运维的智能问答能力。

### 核心价值

- **降低幻觉**：基于事实文档生成答案，显著降低大模型幻觉
- **私有知识管理**：支持企业内部私有知识库的安全管理
- **多格式文档支持**：自动解析 PDF、Markdown 等格式文档
- **高精度检索**：融合多路召回、混合向量、重排序等高级策略
- **流式交互**：支持 SSE 流式输出，提供丝滑的用户体验

### 核心能力

1. **数据结构化**：PDF/Markdown → 语义化 Markdown
2. **多模态理解**：图片 OCR + VLM 视觉理解
3. **语义切片**：基于标题层级的智能文档切分
4. **混合向量**：BGE-M3 生成稠密 + 稀疏向量
5. **多路召回**：向量检索 + HyDE + 网络搜索
6. **重排序**：Cross-Encoder 精准打分

---

## 整体架构

### 架构图

```
┌─────────────────────────────────────────────────────────────┐
│                      Web 服务层 (FastAPI)                    │
│  ┌──────────────────┐         ┌──────────────────┐         │
│  │  导入服务 API    │         │  查询服务 API    │         │
│  │  (port: 8000)    │         │  (port: 8001)    │         │
│  └──────────────────┘         └──────────────────┘         │
└─────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────┐
│                   LangGraph 工作流编排层                     │
│  ┌──────────────────┐         ┌──────────────────┐         │
│  │  导入流水线      │         │  查询流水线      │         │
│  │  (7个节点)       │         │  (7个节点)       │         │
│  └──────────────────┘         └──────────────────┘         │
└─────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────┐
│                      核心能力层                              │
│  ┌──────────┐ ┌──────────┐ ┌──────────┐ ┌──────────┐      │
│  │ 文档解析 │ │ 向量化   │ │ 检索     │ │ 生成     │      │
│  │ (MinerU) │ │ (BGE-M3) │ │ (Milvus) │ │ (LLM)    │      │
│  └──────────┘ └──────────┘ └──────────┘ └──────────┘      │
└─────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────┐
│                      存储层                                  │
│  ┌──────────┐ ┌──────────┐ ┌──────────┐ ┌──────────┐      │
│  │  Milvus  │ │  MinIO   │ │ MongoDB  │ │ 本地文件 │      │
│  │ (向量库) │ │ (对象存储)│ │ (历史记录)│ │ (中间文件)│      │
│  └──────────┘ └──────────┘ └──────────┘ └──────────┘      │
└─────────────────────────────────────────────────────────────┘
```

### 两大核心流水线

#### 1. 知识库构建流水线（导入）

```
START → 任务分发 → PDF解析 → 图片处理 → 文档切分 → 主体识别 → 向量化 → Milvus入库 → END
```

**节点说明**：
1. **node_entry**：任务分发，根据文件类型（PDF/MD）路由
2. **node_pdf_to_md**：PDF → Markdown（MinerU 引擎）
3. **node_md_img**：图片 VLM 理解 + MinIO 存储
4. **node_document_split**：语义切片（基于标题层级）
5. **node_item_name_recognition**：LLM 识别文档主体名称
6. **node_bge_embedding**：BGE-M3 生成稠密 + 稀疏向量
7. **node_import_milvus**：数据持久化到 Milvus

#### 2. 智能检索流水线（查询）

```
用户提问 → 意图识别 → 多路召回 → RRF融合 → 重排序 → 答案生成 → SSE输出
```

**节点说明**：
1. **node_item_name_confirm**：意图识别 + 问题重写 + 实体确认
2. **node_search_embedding**：向量检索（稠密 + 稀疏混合）
3. **node_search_embedding_hyde**：HyDE 假设性文档检索
4. **node_web_search_mcp**：网络搜索补充
5. **node_rrf**：RRF（Reciprocal Rank Fusion）融合排序
6. **node_rerank**：Cross-Encoder 重排序
7. **node_answer_output**：LLM 生成答案 + SSE 流式输出

---

## 目录结构

```
sgg_KB_RAG/
├── app/                          # 核心应用代码
│   ├── import_process/           # 导入流水线模块
│   │   ├── agent/                # LangGraph 工作流
│   │   │   ├── nodes/            # 7个导入节点
│   │   │   │   ├── node_entry.py
│   │   │   │   ├── node_pdf_to_md.py
│   │   │   │   ├── node_md_img.py
│   │   │   │   ├── node_document_split.py
│   │   │   │   ├── node_item_name_recognition.py
│   │   │   │   ├── node_bge_embedding.py
│   │   │   │   └── node_import_milvus.py
│   │   │   ├── main_graph.py     # 导入流程图定义
│   │   │   └── state.py          # 导入状态定义
│   │   ├── api/                  # Web API
│   │   │   └── server.py         # FastAPI 导入服务
│   │   └── page/                 # 前端页面
│   │       └── import.html
│   │
│   ├── query_process/            # 查询流水线模块
│   │   ├── agent/                # LangGraph 工作流
│   │   │   ├── nodes/            # 7个查询节点
│   │   │   │   ├── node_item_name_confirm.py
│   │   │   │   ├── node_search_embedding.py
│   │   │   │   ├── node_search_embedding_hyde.py
│   │   │   │   ├── node_web_search_mcp.py
│   │   │   │   ├── node_rrf.py
│   │   │   │   ├── node_rerank.py
│   │   │   │   └── node_answer_output.py
│   │   │   ├── main_graph.py     # 查询流程图定义
│   │   │   └── state.py          # 查询状态定义
│   │   ├── api/                  # Web API
│   │   │   └── server.py         # FastAPI 查询服务
│   │   └── page/                 # 前端页面
│   │       └── chat.html
│   │
│   ├── conf/                     # 配置文件
│   │   ├── milvus_config.py      # Milvus 配置
│   │   ├── minio_config.py       # MinIO 配置
│   │   ├── lm_config.py          # LLM 配置
│   │   ├── embedding_config.py   # Embedding 配置
│   │   ├── mineru_config.py      # MinerU 配置
│   │   └── reranker_config.py    # Reranker 配置
│   │
│   ├── clients/                  # 数据库客户端
│   │   ├── milvus_utils.py       # Milvus 客户端工具
│   │   ├── minio_utils.py        # MinIO 客户端工具
│   │   └── mongo_history_utils.py # MongoDB 历史记录工具
│   │
│   ├── lm/                       # 大模型工具
│   │   ├── lm_utils.py           # LLM 客户端工具
│   │   ├── embedding_utils.py    # BGE-M3 向量化工具
│   │   └── reranker_utils.py     # Reranker 工具
│   │
│   ├── utils/                    # 通用工具
│   │   ├── task_utils.py         # 任务状态管理
│   │   ├── sse_utils.py          # SSE 流式推送工具
│   │   ├── path_util.py          # 路径工具
│   │   ├── format_utils.py       # 格式化工具
│   │   ├── rate_limit_utils.py   # API 限速工具
│   │   ├── escape_milvus_string_utils.py  # Milvus 字符串转义
│   │   └── normalize_sparse_vector.py     # 稀疏向量归一化
│   │
│   ├── core/                     # 核心工具
│   │   ├── logger.py             # 日志工具
│   │   └── load_prompt.py        # Prompt 加载工具
│   │
│   ├── sse/                      # SSE 示例代码
│   └── tool/                     # 模型下载工具
│
├── prompts/                      # Prompt 模板文件
│   ├── answer_out.prompt
│   ├── hyde_prompt.prompt
│   ├── image_summary.prompt
│   ├── item_name_recognition.prompt
│   ├── product_recognition_system.prompt
│   └── rewritten_query_and_itemnames.prompt
│
├── doc/                          # 测试文档（PDF）
├── output/                       # 中间文件输出目录
├── test/                         # 测试代码
├── 笔记/                         # 项目文档和笔记
├── .env                          # 环境变量配置（需自行创建）
└── pyproject.toml                # 项目依赖配置（如使用 uv）
```

---

## 核心模块详解

### 导入流水线 (Import Process)

#### 状态定义 (`app/import_process/agent/state.py`)

```python
class ImportGraphState(TypedDict):
    task_id: str                    # 任务唯一ID
    is_md_read_enabled: bool        # 是否启用 MD 路径
    is_pdf_read_enabled: bool       # 是否启用 PDF 路径
    local_dir: str                  # 工作目录
    local_file_path: str            # 输入文件路径
    file_title: str                 # 文件标题（去后缀）
    pdf_path: str                   # PDF 文件路径
    md_path: str                    # Markdown 文件路径
    md_content: str                 # Markdown 全文内容
    chunks: list                    # 切片列表（含 metadata）
    item_name: str                  # 识别的主体名称
    embeddings_content: list        # 含向量的数据列表
```

#### 节点详解

##### 1. node_entry - 任务分发节点

**文件**：`app/import_process/agent/nodes/node_entry.py`

**职责**：
- 接收输入文件路径
- 识别文件类型（PDF/MD）
- 设置流程控制标记
- 提取文件标题

**核心逻辑**：
```python
@node_log("node_entry")
def node_entry(state: ImportGraphState) -> ImportGraphState:
    # 1. 获取文件路径
    local_file_path = state["local_file_path"]
    
    # 2. 根据文件类型设置标记
    if local_file_path.endswith(".md"):
        state["md_path"] = local_file_path
        state["is_md_read_enabled"] = True
    elif local_file_path.endswith(".pdf"):
        state["pdf_path"] = local_file_path
        state["is_pdf_read_enabled"] = True
    
    # 3. 提取文件标题
    state["file_title"] = Path(local_file_path).stem
    
    return state
```

**路由逻辑**：
```python
def after_entry_node(state: ImportGraphState):
    if state['is_md_read_enabled']:
        return "node_md_img"      # MD → 直接图片处理
    elif state['is_pdf_read_enabled']:
        return "node_pdf_to_md"   # PDF → 先转 MD
    else:
        return END                # 不支持的类型 → 结束
```

---

##### 2. node_pdf_to_md - PDF 解析节点

**文件**：`app/import_process/agent/nodes/node_pdf_to_md.py`

**职责**：
- 调用 MinerU 服务将 PDF 转换为 Markdown
- 保留文档层级结构（标题、段落、表格）
- 提取图片占位符

**核心步骤**：
1. **step_1_validate_paths**：校验 PDF 路径和输出目录
2. **step_2_upload_and_poll**：上传 PDF 到 MinerU，轮询解析状态
3. **step_3_download_and_extract**：下载解析结果 ZIP，解压获取 MD

**MinerU 交互流程**：
```python
# 1. 申请上传地址
POST {mineru_base_url}/file-urls/batch
→ 返回 file_upload_url, batch_id

# 2. 上传 PDF 文件
PUT file_upload_url
→ 上传成功

# 3. 轮询解析结果
GET {mineru_base_url}/extract-results/batch/{batch_id}
→ 返回 full_zip_url (解析完成)

# 4. 下载并解压
GET full_zip_url → ZIP → 解压 → MD 文件
```

**输出**：
- `state["md_path"]`：Markdown 文件路径
- `state["md_content"]`：Markdown 全文内容

---

##### 3. node_md_img - 图片处理节点

**文件**：`app/import_process/agent/nodes/node_md_img.py`

**职责**：
- 扫描 Markdown 中的图片引用
- 提取图片上下文（前后 100 字符）
- 调用 VLM（视觉语言模型）生成图片描述
- 上传图片到 MinIO
- 替换 Markdown 中的图片链接

**核心步骤**：
1. **step_1_get_content**：获取 MD 内容和图片目录
2. **step_2_scan_images**：正则匹配图片，提取上下文
3. **step_3_image_summary**：VLM 生成图片描述
4. **step_3_upload_and_replace**：上传 MinIO + 替换链接
5. **step_4_backup_md**：备份新 MD 文件

**图片替换示例**：
```markdown
# 替换前
![image](./images/xxx.jpg)

# 替换后
![这是一张电路结构图，左侧标识为电源模块...](http://minio:9000/bucket/images/xxx.jpg)
```

**VLM 调用**：
```python
# 构建多模态消息
message = HumanMessage(
    content=[
        {
            "type": "image_url",
            "image_url": {
                "url": f"data:{mime_type};base64,{image_base64}"
            }
        },
        {
            "type": "text",
            "text": prompt  # 包含图片上下文
        }
    ]
)

# 调用 VLM
chains = vm_model | StrOutputParser()
summary = chains.invoke([message])
```

---

##### 4. node_document_split - 文档切分节点

**文件**：`app/import_process/agent/nodes/node_document_split.py`

**职责**：
- 基于标题层级进行语义切分
- 超长段落二次切分（递归字符切分器）
- 保留标题路径作为元数据

**切分策略**：
1. **一级切分**：基于 Markdown 标题（`#` ~ `######`）
2. **二级切分**：超过 `CHUNK_MAX_SIZE`（500字符）的段落，使用 `RecursiveCharacterTextSplitter`

**配置参数**：
```python
CHUNK_MAX_SIZE = 500    # 触发二次切分的阈值
CHUNK_SIZE = 200        # 二次切分块大小
CHUNK_OVERLAP = 20      # 块之间重叠长度
```

**输出数据结构**：
```python
{
    "content": "切片内容",
    "title": "# 标题路径",
    "parent_title": "父级标题",
    "part": 1,           # 分段标识（二次切分后）
    "file_title": "文件名"
}
```

**切分逻辑**：
```python
# 1. 按标题切分
for line in lines:
    if reg.match(line) and not is_code_block:
        # 新标题 → 结算上一个标题的内容
        chunks.append({...})
        current_title = line
    else:
        # 普通行 → 追加到当前标题
        current_title_lines.append(line)

# 2. 超长段落二次切分
if len(content) > CHUNK_MAX_SIZE:
    splitter = RecursiveCharacterTextSplitter(
        separators=["\n\n", "\n", "。", "！", "；", " "],
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP
    )
    sub_chunks = splitter.split_text(content)
```

---

##### 5. node_item_name_recognition - 主体识别节点

**文件**：`app/import_process/agent/nodes/node_item_name_recognition.py`

**职责**：
- 提取文档前 5 个切片作为上下文
- 调用 LLM 识别文档主体名称（如"华为 Mate60 Pro"）
- 将 `item_name` 附加到所有切片
- 生成 `item_name` 的向量并存入 Milvus

**核心步骤**：
1. **step_1_check_content**：校验 chunks 和 file_title
2. **step_2_document_context**：拼接前 5 个切片为上下文
3. **step_3_call_lm**：调用 LLM 识别主体名称
4. **step_4_insert_milvus**：生成向量并存入 `kb_item_names` 集合

**LLM 提示词**：
```python
# System Prompt
"你是一个专业的文档分析助手，擅长从文档中提取核心主体名称..."

# User Prompt
"文件标题：{file_title}\n上下文：{context}\n请识别文档描述的主体名称..."
```

**Milvus 存储**：
```python
# kb_item_names 集合 Schema
{
    "pk": Int64 (自增主键),
    "file_title": VarChar(512),
    "item_name": VarChar(512),
    "dense_vector": FloatVector(1024),
    "sparse_vector": SparseFloatVector
}
```

---

##### 6. node_bge_embedding - 向量化节点

**文件**：`app/import_process/agent/nodes/node_bge_embedding.py`

**职责**：
- 批量生成切片的稠密 + 稀疏向量
- 语义增强：拼接 `item_name + content`
- 异常处理：单批次失败不影响整体

**向量化策略**：
```python
# 语义增强拼接
item_str = f"主体:{item_name},内容:{content}" if item_name else content

# 批量生成（每批 5 个）
for index in range(0, total, step):
    step_chunks = chunks[index:index + step]
    result = generate_embeddings(vector_str_list)
    # result = {"dense": [...], "sparse": [...]}
```

**BGE-M3 模型配置**：
```python
# app/lm/embedding_utils.py
_bge_m3_ef = BGEM3EmbeddingFunction(
    model_name="BAAI/bge-m3",
    device="cuda:0",        # 或 "cpu"
    use_fp16=True,          # 半精度加速
    normalize_embeddings=True  # L2 归一化
)
```

**输出格式**：
```python
{
    "dense": [0.1, 0.2, ...],      # 1024 维稠密向量
    "sparse": {1: 0.5, 10: 0.8}    # 稀疏向量字典
}
```

---

##### 7. node_import_milvus - Milvus 入库节点

**文件**：`app/import_process/agent/nodes/node_import_milvus.py`

**职责**：
- 创建 Milvus 集合（如不存在）
- 删除旧数据（基于 `item_name`）
- 插入新数据

**Milvus 集合 Schema**：
```python
# kb_chunks 集合
schema.add_field("chunk_id", DataType.INT64, is_primary=True, auto_id=True)
schema.add_field("file_title", DataType.VARCHAR, max_length=512)
schema.add_field("item_name", DataType.VARCHAR, max_length=512)
schema.add_field("title", DataType.VARCHAR, max_length=512)
schema.add_field("parent_title", DataType.VARCHAR, max_length=512)
schema.add_field("part", DataType.INT8)
schema.add_field("content", DataType.VARCHAR, max_length=65535)
schema.add_field("dense_vector", DataType.FLOAT_VECTOR, dim=1024)
schema.add_field("sparse_vector", DataType.SPARSE_FLOAT_VECTOR)

# 索引配置
index_params.add_index(
    field_name="dense_vector",
    index_type="AUTOINDEX",
    metric_type="IP"  # 内积（已归一化）
)
index_params.add_index(
    field_name="sparse_vector",
    index_type="SPARSE_INVERTED_INDEX",
    metric_type="IP",
    params={"inverted_index_algo": "DAAT_MAXSCORE"}
)
```

**幂等性设计**：
```python
# 先删除旧数据（基于 item_name）
milvus_client.delete(
    collection_name=CHUNKS_COLLECTION_NAME,
    filter=f"item_name=='{item_name}'"
)

# 再插入新数据
milvus_client.insert(
    collection_name=CHUNKS_COLLECTION_NAME,
    data=chunks
)
```

---

### 查询流水线 (Query Process)

#### 状态定义 (`app/query_process/agent/state.py`)

```python
class QueryGraphState(TypedDict):
    session_id: str              # 会话唯一标识
    original_query: str          # 用户原始问题
    embedding_chunks: list       # 向量检索结果
    hyde_embedding_chunks: list  # HyDE 检索结果
    web_search_docs: list        # 网络搜索结果
    rrf_chunks: list             # RRF 融合后结果
    reranked_docs: list          # 重排序后 Top-K
    prompt: str                  # 组装的 Prompt
    answer: str                  # 最终答案
    item_names: List[str]        # 提取的主体名称
    rewritten_query: str         # 改写后的问题
    history: list                # 历史对话记录
    is_stream: bool              # 是否流式输出
```

#### 节点详解

##### 1. node_item_name_confirm - 意图识别节点

**文件**：`app/query_process/agent/nodes/node_item_name_confirm.py`

**职责**：
- 获取历史聊天记录
- 调用 LLM 识别主体名称 + 重写问题
- 向量数据库验证主体名称
- 确认/可选/无匹配三级判定

**核心步骤**：
1. **step_1_data_validates**：校验 `original_query` 和 `session_id`
2. **step_2_chat_history**：获取 MongoDB 历史记录
3. **step_3_llm_itemnames_and_rewrite**：LLM 识别 + 重写
4. **step_4_vector_query_item_name**：向量库验证
5. **step_5_select_item_name_list**：三级判定
6. **step_6_deal_state**：处理 state（answer/item_names）
7. **step_7_save_user_chat_message**：保存用户消息

**三级判定逻辑**：
```python
# 确认：score >= 0.65
high_list = [item for item in item_name_list if item["score"] >= 0.65]
if len(high_list) > 0:
    confirmed_item_name_list.append(high_list[0]["item_name"])

# 可选：0.50 <= score < 0.65
low_list = [item for item in item_name_list if 0.50 <= item["score"] < 0.65]
if len(low_list) > 0:
    options_item_name_list.extend([item['item_name'] for item in low_list[:2]])

# 无匹配：score < 0.50
```

**路由逻辑**：
```python
def node_item_name_confirm_after_router(state: QueryGraphState):
    if state['answer']:
        # 有 answer → 需要用户确认 → 直接输出
        return "node_answer_output"
    # 无 answer → 继续多路召回
    return "node_search_embedding", "node_search_embedding_hyde", "node_web_search_mcp"
```

---

##### 2. node_search_embedding - 向量检索节点

**文件**：`app/query_process/agent/nodes/node_search_embedding.py`

**职责**：
- 基于 `rewritten_query` 生成向量
- 在 `kb_chunks` 集合中进行混合检索
- 返回 Top-K 切片

**混合检索流程**：
```python
# 1. 生成查询向量
result = generate_embeddings([rewritten_query])
dense_vector = result['dense'][0]
sparse_vector = result['sparse'][0]

# 2. 构建混合搜索请求
reqs = create_hybrid_search_requests(
    dense_vector, 
    sparse_vector,
    limit=20
)

# 3. 执行混合搜索
response = hybrid_search(
    client=milvus_client,
    collection_name=milvus_config.chunks_collection,
    reqs=reqs,
    ranker_weights=(0.8, 0.2),  # 稠密 0.8，稀疏 0.2
    norm_score=True,
    limit=20
)
```

---

##### 3. node_search_embedding_hyde - HyDE 检索节点

**文件**：`app/query_process/agent/nodes/node_search_embedding_hyde.py`

**职责**：
- 调用 LLM 生成假设性答案
- 对假设性答案进行向量化
- 基于假设性向量检索相关切片

**HyDE 流程**：
```python
# 1. LLM 生成假设性答案
prompt = load_prompt("hyde_prompt", query=rewritten_query)
hypothetical_answer = llm.invoke(prompt)

# 2. 对假设性答案向量化
result = generate_embeddings([hypothetical_answer])

# 3. 混合检索
response = hybrid_search(...)
```

**优势**：
- 提升对隐式意图的召回能力
- 弥补用户提问与文档表述的差异

---

##### 4. node_web_search_mcp - 网络搜索节点

**文件**：`app/query_process/agent/nodes/node_web_search_mcp.py`

**职责**：
- 调用外部搜索引擎（如 Bing、Google）
- 补充知识库中可能缺失的信息
- 返回相关文档片段

**应用场景**：
- 知识库覆盖不足
- 需要实时信息补充
- 用户问题涉及外部知识

---

##### 5. node_rrf - RRF 融合节点

**文件**：`app/query_process/agent/nodes/node_rrf.py`

**职责**：
- 融合多路召回结果（向量、HyDE、Web）
- 使用 RRF（Reciprocal Rank Fusion）算法
- 生成统一的候选文档列表

**RRF 算法**：
```python
# RRF 公式
score(d) = Σ 1 / (k + rank_i(d))

# k 通常为 60
# rank_i(d) 是文档 d 在第 i 路召回中的排名
```

**融合逻辑**：
```python
# 1. 收集所有召回结果
all_chunks = embedding_chunks + hyde_embedding_chunks + web_search_docs

# 2. 按 chunk_id 分组，计算 RRF 分数
rrf_scores = {}
for chunk in all_chunks:
    chunk_id = chunk['chunk_id']
    rank = chunk['rank']  # 在该路召回中的排名
    rrf_scores[chunk_id] = rrf_scores.get(chunk_id, 0) + 1.0 / (60 + rank)

# 3. 按 RRF 分数排序，取 Top-N
sorted_chunks = sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)
top_n_chunks = sorted_chunks[:20]
```

---

##### 6. node_rerank - 重排序节点

**文件**：`app/query_process/agent/nodes/node_rerank.py`

**职责**：
- 使用 Cross-Encoder 模型精准打分
- 过滤低相关噪声
- 筛选 Top-K 文档（如 Top 5）

**重排序流程**：
```python
# 1. 构建 (query, chunk) 对
pairs = [(rewritten_query, chunk['content']) for chunk in rrf_chunks]

# 2. Cross-Encoder 打分
scores = reranker_model.predict(pairs)

# 3. 按分数排序，取 Top-K
sorted_indices = np.argsort(scores)[::-1][:5]
top_k_chunks = [rrf_chunks[i] for i in sorted_indices]
```

**BGE-Reranker 配置**：
```python
# app/lm/reranker_utils.py
reranker_model = FlagReranker(
    model_name="BAAI/bge-reranker-v2-m3",
    device="cuda:0",
    use_fp16=True
)
```

---

##### 7. node_answer_output - 答案生成节点

**文件**：`app/query_process/agent/nodes/node_answer_output.py`

**职责**：
- 组装 Prompt（问题 + 上下文）
- 调用 LLM 生成答案
- 支持流式/非流式输出
- 保存 AI 回复到 MongoDB

**Prompt 组装**：
```python
# 加载 Prompt 模板
prompt = load_prompt(
    "answer_out",
    query=rewritten_query,
    context="\n\n".join([chunk['content'] for chunk in reranked_docs])
)

# 调用 LLM
chains = llm | StrOutputParser()
answer = chains.invoke(prompt)
```

**流式输出**：
```python
if is_stream:
    for ch in answer:
        push_to_session(session_id, SSEEvent.DELTA, {"delta": ch})
        time.sleep(0.03)
    
    push_to_session(session_id, SSEEvent.FINAL, {
        "answer": answer,
        "status": "completed"
    })
```

---

### Web 服务层

#### 导入服务 API (`app/import_process/api/server.py`)

**端口**：8000

**接口列表**：

| 接口 | 方法 | 功能 | 参数 | 响应 |
|------|------|------|------|------|
| `/import/html` | GET | 返回导入页面 | 无 | HTML 文件 |
| `/upload` | POST | 上传文件并触发导入 | `files`: 文件列表 | `{"code": 200, "task_ids": [...]}` |
| `/status/{task_id}` | GET | 查询任务状态 | `task_id` | `{"status": "...", "done_list": [...]}` |

**上传流程**：
```python
@app.post("/upload")
async def upload(backgroundtasks: BackgroundTasks, files: List[UploadFile] = File(...)):
    task_ids = []
    for file in files:
        # 1. 生成 task_id
        task_id = str(uuid.uuid4())
        task_ids.append(task_id)
        
        # 2. 保存文件到 output/YYYYMMDD/task_id/
        local_file_path = save_file(file, task_id)
        
        # 3. 异步执行导入流程
        backgroundtasks.add_task(
            invoke_import_graph,
            task_id=task_id,
            local_file_path=local_file_path
        )
    
    return {"code": 200, "task_ids": task_ids}
```

**任务状态管理**：
```python
# app/utils/task_utils.py
_task_status = {}      # {task_id: status}
_done_list = {}        # {task_id: [done_nodes]}
_running_list = {}     # {task_id: [running_nodes]}

def update_task_status(task_id, status):
    _task_status[task_id] = status

def add_done_task(task_id, node_name):
    _done_list.setdefault(task_id, []).append(node_name)
```

---

#### 查询服务 API (`app/query_process/api/server.py`)

**端口**：8001

**接口列表**：

| 接口 | 方法 | 功能 | 参数 | 响应 |
|------|------|------|------|------|
| `/query/html` | GET | 返回查询页面 | 无 | HTML 文件 |
| `/health` | GET | 健康检查 | 无 | `{"ok": true}` |
| `/query` | POST | 提交查询 | `query`, `session_id`, `is_stream` | 流式/非流式响应 |
| `/stream/{session_id}` | GET | SSE 流式获取结果 | `session_id` | SSE 事件流 |
| `/history/{session_id}` | GET | 查询历史记录 | `session_id`, `limit` | `{"items": [...]}` |
| `/history/{session_id}` | DELETE | 清空历史记录 | `session_id` | `{"delete_count": N}` |

**查询流程**：
```python
@app.post("/query")
async def query(query_request: QueryRequest, backgroundtasks: BackgroundTasks):
    if is_stream:
        # 流式：创建 SSE 队列，异步执行
        create_sse_queue(session_id)
        backgroundtasks.add_task(run_query_graph, session_id, query, is_stream)
        return {"message": "结果正在处理中...", "session_id": session_id}
    else:
        # 非流式：同步执行，等待结果
        run_query_graph(session_id, query, is_stream)
        answer = get_task_result(session_id, "answer")
        return {"answer": answer, "session_id": session_id}
```

**SSE 流式推送**：
```python
# app/utils/sse_utils.py
_session_stream = {}  # {session_id: queue.Queue}

def push_to_session(session_id, event_type, data):
    queue = _session_stream[session_id]
    queue.put(SSEEvent(event_type, data))

def sse_generator(session_id, request):
    queue = _session_stream[session_id]
    while True:
        event = queue.get()
        yield f"event: {event.type}\ndata: {json.dumps(event.data)}\n\n"
        if event.type == SSEEvent.FINAL:
            break
```

---

## 状态管理

### 导入状态 (`ImportGraphState`)

```python
class ImportGraphState(TypedDict):
    # 任务标识
    task_id: str
    
    # 流程控制
    is_md_read_enabled: bool
    is_pdf_read_enabled: bool
    
    # 路径信息
    local_dir: str
    local_file_path: str
    file_title: str
    pdf_path: str
    md_path: str
    
    # 内容数据
    md_content: str
    chunks: list
    item_name: str
    
    # 向量数据
    embeddings_content: list
```

**状态流转**：
```
node_entry
  ↓ 设置 is_pdf_read_enabled / is_md_read_enabled
node_pdf_to_md (或跳过)
  ↓ 设置 md_path, md_content
node_md_img
  ↓ 更新 md_content (图片替换)
node_document_split
  ↓ 设置 chunks
node_item_name_recognition
  ↓ 设置 item_name, 更新 chunks
node_bge_embedding
  ↓ 更新 chunks (添加 dense_vector, sparse_vector)
node_import_milvus
  ↓ 数据持久化
END
```

---

### 查询状态 (`QueryGraphState`)

```python
class QueryGraphState(TypedDict):
    # 会话标识
    session_id: str
    original_query: str
    
    # 检索结果
    embedding_chunks: list
    hyde_embedding_chunks: list
    web_search_docs: list
    
    # 排序结果
    rrf_chunks: list
    reranked_docs: list
    
    # 生成结果
    prompt: str
    answer: str
    
    # 辅助信息
    item_names: List[str]
    rewritten_query: str
    history: list
    is_stream: bool
```

**状态流转**：
```
node_item_name_confirm
  ↓ 设置 item_names, rewritten_query (或 answer)
  ↓ (如果有 answer) → node_answer_output
  ↓ (如果无 answer) → 并发执行三路召回
node_search_embedding ─┐
node_search_embedding_hyde ─┤→ node_rrf
node_web_search_mcp ─┘       ↓
                         node_rerank
                              ↓
                         node_answer_output
                              ↓
                           END
```

---

## 配置管理

### 配置文件位置

所有配置文件位于 `app/conf/` 目录，使用 `@dataclass` 定义，从 `.env` 读取环境变量。

### Milvus 配置 (`milvus_config.py`)

```python
@dataclass
class MilvusConfig:
    milvus_url: str              # Milvus 连接地址
    chunks_collection: str       # 切片集合名
    entity_name_collection: str  # 实体集合名（预留）
    item_name_collection: str    # 主体名称集合名

milvus_config = MilvusConfig(
    milvus_url=os.getenv("MILVUS_URL"),
    chunks_collection=os.getenv("CHUNKS_COLLECTION"),
    entity_name_collection=os.getenv("ENTITY_NAME_COLLECTION"),
    item_name_collection=os.getenv("ITEM_NAME_COLLECTION")
)
```

**环境变量示例**：
```bash
MILVUS_URL=http://localhost:19530
CHUNKS_COLLECTION=kb_chunks
ENTITY_NAME_COLLECTION=kb_entities
ITEM_NAME_COLLECTION=kb_item_names
```

---

### MinIO 配置 (`minio_config.py`)

```python
@dataclass
class MinIOConfig:
    endpoint: str       # MinIO 服务地址
    access_key: str     # 访问密钥
    secret_key: str     # 秘钥
    bucket_name: str    # 存储桶名
    minio_img_dir: str  # 图片存储目录
    minio_secure: bool  # 是否使用 SSL

minio_config = MinIOConfig(
    endpoint=os.getenv("MINIO_ENDPOINT"),
    access_key=os.getenv("MINIO_ACCESS_KEY"),
    secret_key=os.getenv("MINIO_SECRET_KEY"),
    bucket_name=os.getenv("MINIO_BUCKET_NAME"),
    minio_img_dir=os.getenv("MINIO_IMG_DIR"),
    minio_secure=os.getenv("MINIO_SECURE") == "True"
)
```

**环境变量示例**：
```bash
MINIO_ENDPOINT=localhost:9000
MINIO_ACCESS_KEY=minioadmin
MINIO_SECRET_KEY=minioadmin
MINIO_BUCKET_NAME=knowledge-base
MINIO_IMG_DIR=/upload-images
MINIO_SECURE=False
```

---

### LLM 配置 (`lm_config.py`)

```python
@dataclass
class LLMConfig:
    base_url: str         # OpenAI API 地址
    api_key: str          # API Key
    lv_model: str         # 视觉语言模型
    llm_model: str        # 默认 LLM 模型
    llm_temperature: float # 温度参数

lm_config = LLMConfig(
    base_url=os.getenv("OPENAI_BASE_URL"),
    api_key=os.getenv("OPENAI_API_KEY"),
    lv_model=os.getenv("VL_MODEL"),
    llm_model=os.getenv("LLM_DEFAULT_MODEL"),
    llm_temperature=float(os.getenv("LLM_DEFAULT_TEMPERATURE"))
)
```

**环境变量示例**：
```bash
OPENAI_BASE_URL=https://api.openai.com/v1
OPENAI_API_KEY=sk-xxx
VL_MODEL=gpt-4-vision-preview
LLM_DEFAULT_MODEL=gpt-4
LLM_DEFAULT_TEMPERATURE=0.7
```

---

### Embedding 配置 (`embedding_config.py`)

```python
@dataclass
class EmbeddingConfig:
    bge_m3_path: str  # 本地模型路径
    bge_m3: str       # 模型仓库标识
    bge_device: str   # 运行设备
    bge_fp16: bool    # 是否开启半精度

embedding_config = EmbeddingConfig(
    bge_m3_path=os.getenv("BGE_M3_PATH"),
    bge_m3=os.getenv("BGE_M3"),
    bge_device=os.getenv("BGE_DEVICE"),
    bge_fp16=os.getenv("BGE_FP16") in ("1", "True", "true", 1)
)
```

**环境变量示例**：
```bash
BGE_M3_PATH=/models/bge-m3
BGE_M3=BAAI/bge-m3
BGE_DEVICE=cuda:0
BGE_FP16=1
```

---

## 工具类说明

### Milvus 工具 (`app/clients/milvus_utils.py`)

#### 核心函数

**1. get_milvus_client()**
```python
def get_milvus_client():
    """
    获取 Milvus 客户端单例
    :return: MilvusClient 实例
    """
    global _milvus_client
    if _milvus_client is None:
        _milvus_client = MilvusClient(uri=milvus_config.milvus_url)
    return _milvus_client
```

**2. create_hybrid_search_requests()**
```python
def create_hybrid_search_requests(
    dense_vector, 
    sparse_vector, 
    limit=5
):
    """
    构建混合搜索请求
    :return: [dense_req, sparse_req]
    """
    dense_req = AnnSearchRequest(
        data=[dense_vector],
        anns_field="dense_vector",
        param={"metric_type": "IP"},
        limit=limit
    )
    sparse_req = AnnSearchRequest(
        data=[sparse_vector],
        anns_field="sparse_vector",
        param={"metric_type": "IP"},
        limit=limit
    )
    return [dense_req, sparse_req]
```

**3. hybrid_search()**
```python
def hybrid_search(
    client, 
    collection_name, 
    reqs, 
    ranker_weights=(0.5, 0.5),
    norm_score=False,
    limit=5
):
    """
    执行混合搜索
    :return: 搜索结果列表
    """
    rerank = WeightedRanker(
        ranker_weights[0], 
        ranker_weights[1], 
        norm_score=norm_score
    )
    res = client.hybrid_search(
        collection_name=collection_name,
        reqs=reqs,
        ranker=rerank,
        limit=limit
    )
    return res
```

---

### Embedding 工具 (`app/lm/embedding_utils.py`)

#### 核心函数

**1. get_bge_m3_ef()**
```python
def get_bge_m3_ef():
    """
    获取 BGE-M3 模型单例
    :return: BGEM3EmbeddingFunction 实例
    """
    global _bge_m3_ef
    if _bge_m3_ef is None:
        _bge_m3_ef = BGEM3EmbeddingFunction(
            model_name=embedding_config.bge_m3_path,
            device=embedding_config.bge_device,
            use_fp16=embedding_config.bge_fp16,
            normalize_embeddings=True
        )
    return _bge_m3_ef
```

**2. generate_embeddings()**
```python
def generate_embeddings(texts):
    """
    生成稠密 + 稀疏向量
    :param texts: 文本列表
    :return: {"dense": [...], "sparse": [...]}
    """
    model = get_bge_m3_ef()
    embeddings = model.encode_documents(texts)
    
    # 解析稀疏向量（CSR → Dict）
    processed_sparse = []
    for i in range(len(texts)):
        sparse_indices = embeddings["sparse"].indices[
            embeddings["sparse"].indptr[i]:embeddings["sparse"].indptr[i+1]
        ].tolist()
        sparse_data = embeddings["sparse"].data[
            embeddings["sparse"].indptr[i]:embeddings["sparse"].indptr[i+1]
        ].tolist()
        sparse_dict = {k: v for k, v in zip(sparse_indices, sparse_data)}
        processed_sparse.append(sparse_dict)
    
    return {
        "dense": [emb.tolist() for emb in embeddings["dense"]],
        "sparse": processed_sparse
    }
```

---

### 任务工具 (`app/utils/task_utils.py`)

#### 核心函数

**1. 任务状态管理**
```python
_task_status = {}
_done_list = {}
_running_list = {}
_task_results = {}

def update_task_status(task_id, status, is_stream=False):
    """更新任务状态"""
    _task_status[task_id] = status
    if is_stream:
        push_to_session(task_id, SSEEvent.STATUS, {"status": status})

def get_task_status(task_id):
    """获取任务状态"""
    return _task_status.get(task_id, "unknown")

def add_done_task(task_id, node_name, is_stream=False):
    """添加已完成节点"""
    _done_list.setdefault(task_id, []).append(node_name)
    _running_list[task_id].remove(node_name)
    if is_stream:
        push_to_session(task_id, SSEEvent.NODE_DONE, {"node": node_name})

def add_running_task(task_id, node_name, is_stream=False):
    """添加运行中节点"""
    _running_list.setdefault(task_id, []).append(node_name)
    if is_stream:
        push_to_session(task_id, SSEEvent.NODE_RUNNING, {"node": node_name})
```

**2. 任务结果管理**
```python
def set_task_result(task_id, key, value):
    """设置任务结果"""
    _task_results.setdefault(task_id, {})[key] = value

def get_task_result(task_id, key):
    """获取任务结果"""
    return _task_results.get(task_id, {}).get(key)
```

---

### SSE 工具 (`app/utils/sse_utils.py`)

#### 核心类与函数

**1. SSEEvent 枚举**
```python
class SSEEvent(Enum):
    STATUS = "status"           # 状态更新
    NODE_RUNNING = "node_running"  # 节点运行中
    NODE_DONE = "node_done"     # 节点完成
    DELTA = "delta"             # 流式文本增量
    FINAL = "final"             # 最终结果
```

**2. 队列管理**
```python
_session_stream = {}  # {session_id: queue.Queue}

def create_sse_queue(session_id):
    """创建 SSE 队列"""
    _session_stream[session_id] = queue.Queue()

def push_to_session(session_id, event_type, data):
    """推送事件到队列"""
    queue = _session_stream.get(session_id)
    if queue:
        queue.put({"type": event_type.value, "data": data})
```

**3. SSE 生成器**
```python
def sse_generator(session_id, request):
    """SSE 事件流生成器"""
    queue = _session_stream[session_id]
    while True:
        try:
            event = queue.get(timeout=30)
            yield f"event: {event['type']}\ndata: {json.dumps(event['data'])}\n\n"
            if event['type'] == SSEEvent.FINAL.value:
                break
        except queue.Empty:
            yield ": heartbeat\n\n"  # 心跳保活
```

---

### 日志工具 (`app/core/logger.py`)

#### 核心功能

**1. 日志配置**
```python
from loguru import logger

# 控制台输出
logger.add(
    sys.stderr,
    format="<green>{time}</green> | <level>{level}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan> - <level>{message}</level>",
    level="INFO"
)

# 文件输出（自动轮转）
logger.add(
    "logs/app_{time}.log",
    rotation="10 MB",
    retention="7 days",
    level="DEBUG"
)
```

**2. 节点日志装饰器**
```python
def node_log(node_name):
    """节点日志装饰器"""
    def decorator(func):
        @wraps(func)
        def wrapper(state, *args, **kwargs):
            logger.info(f"🚀 节点 [{node_name}] 开始执行")
            try:
                result = func(state, *args, **kwargs)
                logger.success(f"✅ 节点 [{node_name}] 执行成功")
                return result
            except Exception as e:
                logger.error(f"❌ 节点 [{node_name}] 执行失败: {e}")
                raise
        return wrapper
    return decorator
```

**3. 步骤日志装饰器**
```python
def step_log(step_name):
    """步骤日志装饰器"""
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            logger.debug(f"  📍 步骤 [{step_name}] 开始")
            result = func(*args, **kwargs)
            logger.debug(f"  ✔️ 步骤 [{step_name}] 完成")
            return result
        return wrapper
    return decorator
```

---

### Prompt 加载工具 (`app/core/load_prompt.py`)

#### 核心函数

```python
def load_prompt(prompt_name, root_folder=None, **kwargs):
    """
    加载 Prompt 模板
    :param prompt_name: Prompt 文件名（不含 .prompt 后缀）
    :param root_folder: 根文件夹（用于变量替换）
    :param kwargs: 模板变量
    :return: 渲染后的 Prompt 字符串
    """
    prompt_path = PROJECT_ROOT / "prompts" / f"{prompt_name}.prompt"
    template = prompt_path.read_text(encoding="utf-8")
    
    # 变量替换
    for key, value in kwargs.items():
        template = template.replace(f"{{{{{key}}}}}", str(value))
    
    return template
```

**Prompt 模板示例**：
```
# prompts/answer_out.prompt

你是一个专业的客服助手，请基于以下参考资料回答用户问题。

参考资料：
{{context}}

用户问题：{{query}}

请给出准确、简洁的回答：
```

---

## 依赖关系

### 核心框架

| 依赖包 | 版本 | 用途 |
|--------|------|------|
| `langgraph` | latest | 工作流编排 |
| `langchain` | latest | LLM 应用开发框架 |
| `langchain_openai` | latest | OpenAI API 兼容 |
| `langchain_community` | latest | 社区扩展工具 |
| `langchain_text_splitters` | latest | 文本切分器 |

### 向量与模型

| 依赖包 | 版本 | 用途 |
|--------|------|------|
| `torch` | 2.x | 深度学习框架 |
| `FlagEmbedding` | latest | BGE-M3 向量模型 |
| `pymilvus` | 2.4+ | Milvus 客户端 |
| `pymilvus-model` | 2.4+ | Milvus 模型工具 |

### 存储服务

| 依赖包 | 版本 | 用途 |
|--------|------|------|
| `minio` | latest | MinIO 对象存储 |
| `pymongo` | latest | MongoDB 客户端 |

### Web 服务

| 依赖包 | 版本 | 用途 |
|--------|------|------|
| `fastapi` | latest | Web 框架 |
| `uvicorn` | latest | ASGI 服务器 |
| `python-multipart` | latest | 文件上传处理 |
| `sse-starlette` | latest | SSE 支持 |

### 工具库

| 依赖包 | 版本 | 用途 |
|--------|------|------|
| `python-dotenv` | latest | 环境变量管理 |
| `loguru` | latest | 日志工具 |
| `regex` | latest | 增强正则表达式 |
| `requests` | latest | HTTP 客户端 |
| `pydantic` | latest | 数据验证 |

### PDF 解析

| 依赖包 | 版本 | 用途 |
|--------|------|------|
| `magic-pdf` | latest | MinerU PDF 解析 |

---

### 依赖安装

#### 使用 uv（推荐）

```bash
# 安装 uv
pip install uv

# 创建虚拟环境
uv venv

# 激活虚拟环境
# Windows
.venv\Scripts\activate
# Linux/Mac
source .venv/bin/activate

# 安装依赖
uv sync
```

#### 使用 pip

```bash
# 创建虚拟环境
python -m venv .venv

# 激活虚拟环境
# Windows
.venv\Scripts\activate
# Linux/Mac
source .venv/bin/activate

# 安装依赖
pip install -r requirements.txt
```

---

## 运行方式

### 环境准备

#### 1. 克隆项目

```bash
git clone <repository_url>
cd sgg_KB_RAG
```

#### 2. 安装依赖

```bash
# 使用 uv（推荐）
uv sync

# 或使用 pip
pip install -r requirements.txt
```

#### 3. 配置环境变量

创建 `.env` 文件（参考 `.env.example`）：

```bash
# Milvus 配置
MILVUS_URL=http://localhost:19530
CHUNKS_COLLECTION=kb_chunks
ITEM_NAME_COLLECTION=kb_item_names

# MinIO 配置
MINIO_ENDPOINT=localhost:9000
MINIO_ACCESS_KEY=minioadmin
MINIO_SECRET_KEY=minioadmin
MINIO_BUCKET_NAME=knowledge-base
MINIO_IMG_DIR=/upload-images
MINIO_SECURE=False

# LLM 配置
OPENAI_BASE_URL=https://api.openai.com/v1
OPENAI_API_KEY=sk-xxx
VL_MODEL=gpt-4-vision-preview
LLM_DEFAULT_MODEL=gpt-4
LLM_DEFAULT_TEMPERATURE=0.7

# Embedding 配置
BGE_M3_PATH=/models/bge-m3
BGE_M3=BAAI/bge-m3
BGE_DEVICE=cuda:0
BGE_FP16=1

# MinerU 配置
MINERU_BASE_URL=http://localhost:8000
MINERU_API_KEY=xxx

# MongoDB 配置
MONGODB_URI=mongodb://localhost:27017
MONGODB_DB_NAME=knowledge_base
```

#### 4. 启动外部服务

**Milvus**：
```bash
# Docker 启动
docker run -d --name milvus-standalone \
  -p 19530:19530 \
  -p 9091:9091 \
  milvusdb/milvus:latest
```

**MinIO**：
```bash
docker run -d --name minio \
  -p 9000:9000 \
  -p 9001:9001 \
  -e "MINIO_ROOT_USER=minioadmin" \
  -e "MINIO_ROOT_PASSWORD=minioadmin" \
  minio/minio server /data --console-address ":9001"
```

**MongoDB**：
```bash
docker run -d --name mongodb \
  -p 27017:27017 \
  mongo:latest
```

---

### 启动服务

#### 1. 启动导入服务

```bash
# 方式一：直接运行
python -m app.import_process.api.server

# 方式二：使用 uvicorn
uvicorn app.import_process.api.server:app --host 0.0.0.0 --port 8000 --reload
```

**访问**：
- API: http://localhost:8000
- 页面: http://localhost:8000/import/html

---

#### 2. 启动查询服务

```bash
# 方式一：直接运行
python -m app.query_process.api.server

# 方式二：使用 uvicorn
uvicorn app.query_process.api.server:app --host 0.0.0.0 --port 8001 --reload
```

**访问**：
- API: http://localhost:8001
- 页面: http://localhost:8001/query/html

---

### 使用示例

#### 1. 导入文档

**API 调用**：
```bash
curl -X POST http://localhost:8000/upload \
  -F "files=@/path/to/document.pdf"
```

**响应**：
```json
{
  "code": 200,
  "message": "文件已经上传成功,正在处理解析中....",
  "task_ids": ["550e8400-e29b-41d4-a716-446655440000"]
}
```

**查询状态**：
```bash
curl http://localhost:8000/status/550e8400-e29b-41d4-a716-446655440000
```

**响应**：
```json
{
  "code": 200,
  "task_id": "550e8400-e29b-41d4-a716-446655440000",
  "status": "completed",
  "done_list": [
    "node_entry",
    "node_pdf_to_md",
    "node_md_img",
    "node_document_split",
    "node_item_name_recognition",
    "node_bge_embedding",
    "node_import_milvus"
  ],
  "running_list": []
}
```

---

#### 2. 查询问答

**非流式查询**：
```bash
curl -X POST http://localhost:8001/query \
  -H "Content-Type: application/json" \
  -d '{
    "query": "华为 Mate60 Pro 的电池容量是多少？",
    "session_id": "test_session_001",
    "is_stream": false
  }'
```

**响应**：
```json
{
  "message": "处理完成！",
  "session_id": "test_session_001",
  "answer": "华为 Mate60 Pro 的电池容量为 5000mAh，支持 88W 有线超级快充和 50W 无线超级快充。",
  "done_list": [
    "node_item_name_confirm",
    "node_search_embedding",
    "node_search_embedding_hyde",
    "node_web_search_mcp",
    "node_rrf",
    "node_rerank",
    "node_answer_output"
  ]
}
```

---

**流式查询**：
```bash
# 1. 提交查询
curl -X POST http://localhost:8001/query \
  -H "Content-Type: application/json" \
  -d '{
    "query": "华为 Mate60 Pro 的电池容量是多少？",
    "session_id": "test_session_002",
    "is_stream": true
  }'

# 响应
{
  "message": "结果正在处理中...",
  "session_id": "test_session_002"
}

# 2. SSE 流式获取结果
curl http://localhost:8001/stream/test_session_002
```

**SSE 事件流**：
```
event: status
data: {"status": "processing"}

event: node_running
data: {"node": "node_item_name_confirm"}

event: node_done
data: {"node": "node_item_name_confirm"}

event: delta
data: {"delta": "华"}

event: delta
data: {"delta": "为"}

event: delta
data: {"delta": " "}

...

event: final
data: {"answer": "华为 Mate60 Pro 的电池容量为 5000mAh...", "status": "completed"}
```

---

#### 3. 查询历史记录

```bash
curl http://localhost:8001/history/test_session_001?limit=10
```

**响应**：
```json
{
  "session_id": "test_session_001",
  "items": [
    {
      "_id": "65a1b2c3d4e5f6g7h8i9j0k1",
      "session_id": "test_session_001",
      "role": "user",
      "text": "华为 Mate60 Pro 的电池容量是多少？",
      "rewritten_query": "华为 Mate60 Pro 电池容量",
      "item_names": ["华为 Mate60 Pro"],
      "ts": "2026-06-13T10:30:00Z"
    },
    {
      "_id": "65a1b2c3d4e5f6g7h8i9j0k2",
      "session_id": "test_session_001",
      "role": "assistant",
      "text": "华为 Mate60 Pro 的电池容量为 5000mAh...",
      "rewritten_query": "",
      "item_names": [],
      "ts": "2026-06-13T10:30:05Z"
    }
  ]
}
```

---

### 测试运行

#### 1. 单元测试

```bash
# 测试导入节点
python -m app.import_process.agent.nodes.node_entry
python -m app.import_process.agent.nodes.node_pdf_to_md
python -m app.import_process.agent.nodes.node_md_img
python -m app.import_process.agent.nodes.node_document_split
python -m app.import_process.agent.nodes.node_item_name_recognition
python -m app.import_process.agent.nodes.node_bge_embedding
python -m app.import_process.agent.nodes.node_import_milvus

# 测试查询节点
python -m app.query_process.agent.nodes.node_item_name_confirm
```

#### 2. 集成测试

```bash
# 测试导入全流程
python -m app.import_process.agent.main_graph

# 测试查询全流程
python -m app.query_process.agent.main_graph
```

---

## 技术栈总结

### 核心技术

| 技术 | 用途 | 优势 |
|------|------|------|
| **LangGraph** | 工作流编排 | 状态管理、条件路由、可视化 |
| **LangChain** | LLM 应用开发 | 生态丰富、工具链完整 |
| **BGE-M3** | 向量化 | 支持稠密+稀疏、多语言、长文本 |
| **Milvus** | 向量数据库 | 高性能、云原生、混合检索 |
| **MinerU** | PDF 解析 | 高精度、保留结构、多模态 |
| **FastAPI** | Web 服务 | 高性能、异步、自动文档 |
| **SSE** | 流式输出 | 实时推送、低延迟 |

### 设计模式

| 模式 | 应用场景 |
|------|----------|
| **单例模式** | Milvus 客户端、BGE-M3 模型 |
| **工厂模式** | LLM 客户端创建 |
| **策略模式** | 多路召回策略 |
| **装饰器模式** | 日志装饰器 |
| **观察者模式** | SSE 事件推送 |

### 核心亮点

1. **双层索引架构**：文档级（item_name）+ 切片级（chunks），兼顾效率与精度
2. **混合向量检索**：稠密（语义）+ 稀疏（关键词），覆盖全面
3. **多路召回 + RRF**：向量 + HyDE + Web，避免信息遗漏
4. **重排序机制**：Cross-Encoder 精准打分，过滤噪声
5. **流式交互**：SSE 实时推送，用户体验丝滑
6. **幂等性设计**：基于 item_name 清理旧数据，支持重复导入
7. **任务状态管理**：全程可追踪，支持异步监控

---

## 附录

### A. 常见问题

**Q1: 如何切换 LLM 模型？**
- 修改 `.env` 中的 `LLM_DEFAULT_MODEL` 和 `OPENAI_BASE_URL`
- 支持 OpenAI、Azure OpenAI、国产大模型（千问、智谱等）

**Q2: 如何优化检索精度？**
- 调整 `CHUNK_SIZE` 和 `CHUNK_OVERLAP`
- 调整 RRF 权重和 Rerank Top-K
- 优化 Prompt 模板

**Q3: 如何部署到生产环境？**
- 使用 Docker Compose 编排服务
- 配置 Nginx 反向代理
- 使用 Gunicorn + Uvicorn 部署

**Q4: 如何监控任务进度？**
- 查询 `/status/{task_id}` 接口
- 查看日志文件 `logs/app_*.log`
- 前端页面实时展示

---

### B. 性能优化建议

1. **GPU 加速**：使用 CUDA 运行 BGE-M3 和 Reranker
2. **批量处理**：向量化时批量处理（默认 5 个/批）
3. **索引优化**：Milvus 使用 `AUTOINDEX` 自动优化
4. **缓存策略**：对高频查询结果进行缓存
5. **异步处理**：使用 BackgroundTasks 异步执行导入

---

### C. 扩展方向

1. **知识图谱**：集成 Neo4j，构建实体关系图
2. **多模态检索**：支持图片、视频检索
3. **对话管理**：增强多轮对话上下文管理
4. **权限控制**：基于角色的文档访问控制
5. **联邦学习**：支持多租户隔离

---

## 结语

掌柜智库 RAG 系统是一套完整的企业级知识库解决方案，涵盖从文档解析、向量化、存储到智能检索的全链路能力。通过本 Code Wiki，您可以深入了解系统的架构设计、核心模块、关键技术和运行方式，为二次开发和生产部署提供有力支持。

如有问题或建议，欢迎反馈！

---

**文档版本**：v1.0  
**最后更新**：2026-06-13  
**维护者**：掌柜智库团队
