# 掌柜智库项目(RAG)实战

## 5. 导入数据节点实现与测试

### 5.5 主体识别 (node_item_name_recognition)

**文件**: `app/import_process/agent/nodes/node_item_name_recognition.py`

#### 1. 节点作用与实现思路

作为文档结构化解析与差异化分类的**核心关键节点**，依托大语言模型深度语义理解能力，精准萃取文档**核心主体**、业务实体与专属概念，快速判定文档所属品类与内容属性。

通过全局主体标识绑定，实现多源文档精准区分、内容归类、数据去重与精细化管控，搭建**实体与文本切片的强关联映射体系**，为后续实体级检索、语义对齐、定向过滤、结构化问答筑牢底层支撑，大幅提升知识库检索精准度与数据治理能力。

<img src="assets/image-20260425093305664.png" alt="image-20260425093305664" style="zoom:40%;" />

**实现思路**：

1. **关键上下文精准裁剪**：优先截取文档高价值头部切片，涵盖标题、概述等核心摘要信息，精简输入上下文，在控制推理成本的同时，保障大模型主体识别的准确率。
2. **大模型语义萃取识别**：结合定制化业务提示词，依托 LLM 深层语义解析能力，智能提取文档核心主体、专属名词与业务标识，完成文档类型自动判别与内容切面划分。
3. **全链路容错兜底设计**：针对大模型输出不稳定、识别异常、返回空值等场景，配置异常捕获与默认兜底策略，保障导入流程稳定运行，避免单点故障中断全链路任务。
4. **实体向量化预处理**：将识别后的标准主体实体统一完成向量编码，对接向量库实现跨表述语义匹配，打通别名关联、语义联动能力，实现模糊检索与精准召回。

#### 2. 步骤分解

1.  **导入与配置**: 引入必要的库（LangChain, Milvus, etc.）及配置参数。
2.  **核心辅助函数**: 包含字符串安全转义等辅助逻辑。
3.  **主流程定义**: LangGraph 节点的入口函数，串联各个步骤。
4.  **步骤 1: 获取输入**: 校验 State 中的 `file_title` 和 `chunks`。
5.  **步骤 2: 构建上下文**: 截取前 K 个切片作为 LLM 的识别素材。
6.  **步骤 3: 调用 LLM**: 使用大模型识别商品名称，包含错误重试与兜底。
7.  **步骤 4: 回填数据**: 将识别结果更新回 State 和 Chunks 元数据。
8.  **步骤 5: 生成向量**: 调用 Embedding 模型生成 Dense/Sparse 向量。
9.  **步骤 6: 保存结果**: 将数据写入 Milvus 向量库，并处理幂等性。
10.  **单元测试**: 独立运行的测试代码，验证核心流程。

#### 3. 准备Embedding模型和工具

##### 3.1 什么是 “生成词向量”？

词向量（Word Vector/Embedding）就是把**文字（比如 “苏泊尔 5000W 大功率电磁炉”）转换成计算机能理解的数字列表（向量）** 的过程。

打个比方

- 人类理解文字：“苹果手机”= 品牌（苹果）+ 品类（手机）；
- 计算机理解文字：没法直接懂 “苹果手机”，但能懂 `[0.23, -0.56, 1.89, ...]` 这样的数字列表；
- 词向量的作用：把文字的**语义信息**（含义、特征、关联度）编码成数字，让计算机能 “计算文字相似度”“分类文字”“检索相似内容”。

举个简单例子

|    文字    | 对应的词向量（简化版，实际是几百 / 几千维） |
| :--------: | :-----------------------------------------: |
|  苹果手机  |          [0.23, -0.56, 1.89, 0.78]          |
|  华为手机  |          [0.21, -0.58, 1.91, 0.76]          |
| 苹果笔记本 |          [0.22, -0.55, 0.87, 0.79]          |

计算机通过对比这些数字列表的相似度，就能判断：

- “苹果手机” 和 “华为手机” 更像（数字差异小）；
- “苹果手机” 和 “苹果笔记本” 相似度低（数字差异大）。

##### 3.2  “稀疏向量 + 稠密向量” 

代码是基于 `BGE-M3` 模型生成两种词向量（这是当前主流的多模态嵌入方案）拆解：

|           类型            |                             特点                             |                             用途                             |
| :-----------------------: | :----------------------------------------------------------: | :----------------------------------------------------------: |
| 稠密向量（Dense Vector）  | 长度固定（比如 768 维 / 1024 维），每个位置都是连续数值（如 0.23、-0.56） | 捕捉文字的**语义信息**（比如 “苹果手机” 的核心含义），适合相似度计算 |
| 稀疏向量（Sparse Vector） | 长度极长（比如几十万维），但只有少数位置有非 0 值，其余都是 0 | 捕捉文字的**关键词 / 字面特征**（比如 “苹果”“5000W”“电磁炉”），适合精准检索 |

BGE-M3 模型同时输出这两种向量，结合使用能兼顾 “语义理解” 和 “精准匹配”。

##### 3.3 安装Python依赖库

在使用模型之前，需要安装相关的 Python 依赖库。

```cmd
# ===================== 环境安装命令（适配BGE-M3+Milvus，GPU/CPU版区分）=====================
# 【GPU版】安装CUDA 12.4版PyTorch（含torchvision/torchaudio，NVIDIA显卡GPU加速必备）
# 适配：有NVIDIA独显且驱动≥551.61，后续BGE-M3可开启FP16半精度推理
uv pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124

# 【备用-CPU版】无NVIDIA显卡（AMD/Intel集显）请用此命令，直接安装CPU版PyTorch
# 注释掉上方GPU版命令，取消注释下方即可
uv add torch torchvision torchaudio

# 安装Milvus和BGE-M3核心依赖（所有环境必装，无GPU/CPU区分）
# pymilvus[model]：Milvus Python客户端（带模型相关依赖，适配向量入库/检索）
# FlagEmbedding：BGE-M3向量生成模型的核心依赖（不可替代）
# transformers：FlagEmbedding底层依赖，Hugging Face模型运行库
uv add  pymilvus[model] FlagEmbedding transformers

#⚠️：安装FlagEmbedding的时候会自动安装一个cpu版本的torch替换掉之前的gpu版本的torch，
# 要解决这个问题需要做以下几个步骤
# 步骤1 先删除已经安装的FlagEmbedding（先在pyproject.toml中确定一下自己安装的版本）
uv remove FlagEmbeddin

# 步骤2 将以下内容配置在pyproject.toml中
dependencies = [
     其他之前安装过的配置,
    "flagembedding>=v1.3.5",
    "torch>=2.10.0",
    "torchvision>=0.25.0",
    "torchaudio>=2.10.0",
]

[tool.uv.sources]
# 强制从 NVIDIA 源安装
torch = { index = "pytorch-cuda" }
torchvision = { index = "pytorch-cuda" }
torchaudio = { index = "pytorch-cuda" }

[[tool.uv.index]]
name = "pytorch-cuda"
url = "https://download.pytorch.org/whl/cu128"
explicit = true

# 步骤3 删除锁文件并重新锁定
rm uv.lock
uv lock

# 步骤4：重新同步环境
uv sync --reinstall

# 步骤5：验证
uv run python -c "import torch; print('GPU:', torch.cuda.is_available())"
```

CUDA 每个版本都有**最低算力要求**，CUDA 12.4 要求显卡的**CUDA 算力≥3.5**（几乎 2016 年之后的 NVIDIA 独显都满足，老款如 GTX 750 Ti 也达标），**主流显卡（RTX30/40 系、GTX16/20 系）全兼容**，几乎不用担心里程碑。

直接打开 NVIDIA 官方算力表，搜索自己的显卡型号，看对应的**Compute Capability（算力）** 数值：

[NVIDIA 显卡 CUDA 算力官方查询地址](https://developer.nvidia.com/cuda-gpus)

- 桌面显卡看**GeForce**栏，笔记本显卡看**GeForce Notebook**栏；
- 示例：RTX 3060 算力 8.6、GTX 1650 算力 7.5、RTX 4090 算力 8.9，都远大于 3.5，完美适配 CUDA 12.4。

 ##### 3.4 Embedding下载模型

如果访问 HuggingFace 较慢，可以使用阿里云(阿里巴巴通义实验室（原达摩院）)的 ModelScope 社区下载。

https://www.modelscope.cn/models/BAAI/bge-m3 

**1.** **安装 modelscope 库**：

```python
uv add modelscope
```

**2.** **运行 Python 脚本下载**： 创建一个临时的 Python 脚本（例如 download_bge.py）并运行：

```python
from modelscope.hub.snapshot_download import snapshot_download

# 下载模型到当前目录下的 models/bge-m3 文件夹
model_dir = snapshot_download('BAAI/bge-m3', cache_dir='D:/ai_models/modelscope_cache/models')
print(f"模型已下载到: {model_dir}")
```

**3. .env配置**

```ini
#embedding配置
# BGE-M3模型本地缓存/部署路径（本地加载模型时使用，指向ModelScope下载的模型目录）
BGE_M3_PATH=D:\ai_models\modelscope_cache\models\BAAI\bge-m3
# BGE-M3模型官方标识（ModelScope/HuggingFace通用，拉取模型时使用）
BGE_M3=BAAI/bge-m3
# BGE-M3运行设备，cuda:0表示使用第1块GPU，cpu表示使用CPU，cuda:N表示第N+1块GPU
BGE_DEVICE=cuda:0 
# BGE-M3是否开启FP16半精度推理，1=开启（GPU加速更高效），0=关闭（兼容低版本GPU/CPU）
BGE_FP16=1
```

**4. 配置参数读取**

文件：`app.config.embedding_config.py`

```py
# 导入核心依赖：数据类、环境变量读取、路径处理
from dataclasses import dataclass
import os
from dotenv import load_dotenv

# 提前加载.env配置文件（保持和原代码一致，只需执行一次）
load_dotenv()

# 定义Embedding配置（适配BGE-M3的所有配置，类名embedding_config）
@dataclass
class EmbeddingConfig:
    bge_m3_path: str  # 本地模型路径
    bge_m3: str       # 模型仓库标识
    bge_device: str   # 运行设备(cuda:0/cpu)
    bge_fp16: bool    # 是否开启半精度（1=True/0=False）

# 实例化配置对象，和原代码lm_config风格保持一致
embedding_config = EmbeddingConfig(
    bge_m3_path=os.getenv("BGE_M3_PATH"),
    bge_m3=os.getenv("BGE_M3"),
    bge_device=os.getenv("BGE_DEVICE"),
    # 特殊处理：将.env中的1/0转为布尔值，兼容常见的数字/字符串格式
    bge_fp16=os.getenv("BGE_FP16") in ("1", "True", "true", 1)
)
```

##### 3.5 工具代码导入

文件：`app.lm.embedding_utils.py`

```python
from pymilvus.model.hybrid import BGEM3EmbeddingFunction
from app.core.logger import logger
from app.conf.embedding_config import embedding_config

# 模型单例对象，避免重复初始化
_bge_m3_ef = None

def get_bge_m3_ef():
    """
    获取BGE-M3模型单例对象，自动加载环境变量配置
    :return: 初始化完成的BGEM3EmbeddingFunction实例
    """
    global _bge_m3_ef
    # 单例模式：已初始化则直接返回，避免重复加载模型
    if _bge_m3_ef is not None:
        logger.debug("BGE-M3模型单例已存在，直接返回实例")
        return _bge_m3_ef

    # 从环境变量加载配置，无配置则使用默认值
    # 本地有可以使用本地地址！ 没有使用 "BAAI/bge-m3" 会自动下载！ 如果云端部署也可以使用url地址！
    model_name = embedding_config.bge_m3_path or "BAAI/bge-m3"
    device = embedding_config.bge_device or "cpu"
    use_fp16 = embedding_config.bge_fp16 or False

    # 打印模型初始化配置，便于问题排查
    logger.info(
        "开始初始化BGE-M3模型",
        extra={
            "model_name": model_name,
            "device": device,
            "use_fp16": use_fp16,
            "normalize_embeddings": True
        }
    )

    try:
        # 初始化BGE-M3模型，开启原生L2归一化（适配Milvus IP内积检索）
        # pymilvus.model.hybrid.BGEM3EmbeddingFunction ，在工程上最大的好处是： 
        # 和 Milvus 检索链路天然对齐 ，上线更稳更省事。
        _bge_m3_ef = BGEM3EmbeddingFunction(
            model_name=model_name,
            device=device,
            use_fp16=use_fp16,
            normalize_embeddings=True  # 模型原生对稠密+稀疏向量做L2归一化
        )
        logger.success("BGE-M3模型初始化成功，已开启原生L2归一化")
        # “它把所有向量拉伸到统一长度（模长为1），让我们能在数据库中放心使用最快的内积（IP）检索，既提速又不丢精度。”
        return _bge_m3_ef
    except Exception as e:
        logger.error(f"BGE-M3模型初始化失败：{str(e)}", exc_info=True)
        raise  # 向上抛出异常，由调用方处理


def generate_embeddings(texts):
    """
    为文本列表生成稠密+稀疏混合向量嵌入（模型原生L2归一化）
    :param texts: 要生成嵌入的文本列表，单文本也需封装为列表
    :return: 字典格式的向量结果，key为dense/sparse，对应嵌套列表/字典列表
    :raise: 向量生成过程中的异常，由调用方捕获处理
    """
    # 入参合法性校验
    if not isinstance(texts, list) or len(texts) == 0:
        logger.warning("生成向量入参不合法，texts必须为非空列表")
        raise ValueError("参数texts必须是包含文本的非空列表")

    logger.info(f"开始为{len(texts)}条文本生成混合向量嵌入")
    try:
        # 加载BGE-M3模型单例
        model = get_bge_m3_ef()
        # 模型编码生成向量，返回dense（稠密向量）+sparse（CSR格式稀疏向量）
        embeddings = model.encode_documents(texts)
        logger.debug(f"模型编码完成，开始解析稀疏向量格式，共{len(texts)}条")

        # 初始化稀疏向量处理结果，解析为字典格式（适配序列化/存储）
        processed_sparse = []
      	# 把模型输出的 CSR 稀疏矩阵 ，按“每条文本一行”拆成 {特征索引: 权重} 字典
        # - indices ：非零元素的“列号（特征ID）”
		# - data ：对应列号的权重值
		# - indptr ：每一行在 indices/data 里的起止位置指针 
        # 数据示例:
        # indices = [3, 8, 20, 1, 9]
		# data    = [0.7, 0.2, 0.1, 0.6, 0.4]
        # indptr  = [0, 3, 5]
        # 获取对应的数据
        # - 第0条文本用 0:3 => indices=[3,8,20] , data=[0.7,0.2,0.1]
		# - 第1条文本用 3:5 => indices=[1,9] , data=[0.6,0.4]
        for i in range(len(texts)):
            # 提取第i个文本的稀疏向量索引：np.int64 → Python int（满足字典key可哈希要求）
            sparse_indices = embeddings["sparse"].indices[
                embeddings["sparse"].indptr[i]:embeddings["sparse"].indptr[i + 1]
            ].tolist()
            # 提取第i个文本的稀疏向量权重：np.float32 → Python float（适配JSON序列化/接口返回）
            sparse_data = embeddings["sparse"].data[
                embeddings["sparse"].indptr[i]:embeddings["sparse"].indptr[i + 1]
            ].tolist()
            # 构造{特征索引: 归一化权重}的稀疏向量字典
            sparse_dict = {k: v for k, v in zip(sparse_indices, sparse_data)}
            processed_sparse.append(sparse_dict)

        # 构造最终返回结果，稠密向量转列表（解决numpy数组不可序列化问题）
        result = {
            "dense": [emb.tolist() for emb in embeddings["dense"]],  # 嵌套列表，与输入文本一一对应
            "sparse": processed_sparse  # 字典列表，模型已做L2归一化
        }
        logger.success(f"{len(texts)}条文本向量生成完成，格式已适配工业级使用")
        return result

    except Exception as e:
        logger.error(f"文本向量生成失败：{str(e)}", exc_info=True)
        raise  # 不吞异常，向上传递让调用方做重试/降级处理


"""
核心设计亮点&适配说明：
1. normalize_embeddings=True 的价值：
- 检索更稳定 ：不同文本长短、词频差异不会把分数拉偏。
- IP 可近似 cosine ：向量都归一化后， Inner Product 和余弦相似度等价，Milvus 用 IP 检索就很合适。
- dense/sparse 都统一标尺 ：混合检索时两路分数更容易做融合，不容易一边压死另一边。
- 减少异常高分 ：防止“模长大”的向量仅靠长度拿高分。
2. 彻底解决NumPy类型做key问题：sparse_indices加.tolist()，将np.int64转为Python原生int，满足字典key的可哈希要求，无报错风险；
3. 稀疏值适配序列化：sparse_data加.tolist()，将np.float32转为Python原生float，支持JSON写入/接口返回/Milvus入库等所有场景；
4. 单例模式优化：模型仅初始化一次，避免重复加载耗时耗资源，提升批量处理效率；
5. 格式匹配业务调用：返回dense嵌套列表、sparse字典列表，与vector_result["dense"][0]/sparse_vector["sparse"][0]取值逻辑完美契合；
6. 分级日志覆盖：从模型初始化、向量生成到异常报错，全流程日志记录，便于生产环境问题排查；
7. 入参合法性校验：防止空列表/非列表入参导致的内部报错，提升工具类健壮性。
"""
```

#### 4. 准备Milvues向量数据库和工具

##### 4.1 安装 Milvus 并完成连接测试

- 操作系统：Linux（CentOS/RHEL/Ubuntu 通用，`yum` 命令适用于 CentOS/RHEL，Ubuntu 需替换为 `apt-get`）
- 核心目标：编译安装 Python3.8（Milvus 客户端适配）+ 部署 Milvus 2.4.11 单机版 + 解决 MinIO 端口冲突 + 部署 Attu 可视化客户端
- 前置要求：服务器已安装 `docker` + `docker compose`（未安装可先执行：`yum install -y docker docker-compose-plugin && systemctl start docker && systemctl enable docker`）

**第一步：Python3.8 环境编译安装（客户端基础 ！使用Docker安装无需要此步骤）**

```bash
# 1. 安装 Python 编译必备的工具和依赖库
yum install -y zlib-devel bzip2-devel openssl-devel ncurses-devel sqlite-devel readline-devel tk-devel gcc make libffi-devel

# 2. 创建 Python3.8 专属安装目录，避免和系统Python冲突
mkdir -p /usr/local/python3.8

# 3. 下载 Python3.8.16 源码包（稳定版，适配Milvus客户端）
wget https://www.python.org/ftp/python/3.8.16/Python-3.8.16.tgz

# 4. 解压源码包
tar -zxvf Python-3.8.16.tgz

# 5. 进入解压目录，配置编译参数（指定安装路径+关联openssl）
cd Python-3.8.16
./configure --prefix=/usr/local/python3.8 --with-openssl=/usr/local/openssl

# 6. 多线程编译并安装（-j $(nproc) 调用所有CPU核心，加速编译）
make -j $(nproc) && make install

# 7. 验证安装成功（输出版本号即成功，若提示命令不存在，需配置环境变量：ln -s /usr/local/python3.8/bin/python3 /usr/bin/python3）
python3 --version
```

> 注：Ubuntu 系统替换第一步命令为：`apt-get update && apt-get install -y zlib1g-dev libbz2-dev libssl-dev libncurses5-dev libsqlite3-dev libreadline-dev libtk8.6 libgdm-dev libdb4o-cil-dev libffi-dev gcc make`

**第二步：部署 Milvus 2.4.11 单机版（从旧版本升级 / 全新部署通用）**

```bash
# 1. 创建并进入Milvus工作目录（无则新建，有则进入原有目录）
mkdir -p ~/milvus && cd ~/milvus

# 2. 停止当前运行的旧版Milvus（全新部署执行此命令无影响）
docker compose down

# 3. 备份原有数据（关键！防止数据丢失，若为全新部署，无volumes目录可跳过此步）
mv volumes volumes.bak_2.3.5

# 4. 下载 Milvus 2.4.11 官方单机版docker-compose配置文件（覆盖原有文件）
wget https://github.com/milvus-io/milvus/releases/download/v2.4.11/milvus-standalone-docker-compose.yml -O docker-compose.yml

# 5. 启动 Milvus 2.4.11（-d 后台运行）
docker compose up -d

# 6. 检查Milvus运行状态（等待3-5秒，所有容器状态为 Up 即启动成功）
docker compose ps

# 【常用运维命令】后续管理Milvus可使用
# 停止Milvus：docker compose stop
# 重启Milvus：docker compose restart
# 查看运行日志（排查问题用）：docker compose logs milvus-standalone
```

**第三步：解决 MinIO 端口冲突问题（修改 docker-compose.yml）**

Milvus 内置 MinIO 用于存储向量数据，默认端口 `9000/9001` 若被本地服务占用，需修改映射端口，**编辑 `~/milvus/docker-compose.yml`**，找到 `minio` 节点，修改 ports 配置：

```yml
minio:
    container_name: milvus-minio
    image: minio/minio:RELEASE.2023-03-20T20-16-18Z
    environment:
      MINIO_ACCESS_KEY: minioadmin  # 内置账号，无需修改
      MINIO_SECRET_KEY: minioadmin  # 内置密码，无需修改
    ports:
      - "9003:9001"  # 控制台端口：原9001→改为9003（避开占用）
      - "9002:9000"  # 数据端口：原9000→改为9002（避开占用）
    volumes:
      - ${DOCKER_VOLUME_DIRECTORY:-.}/volumes/minio:/minio_data  # 数据持久化目录
    command: minio server /minio_data --console-address ":9001"  # 容器内端口不变，仅改宿主机映射
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:9000/minio/health/live"]
      interval: 30s
      timeout: 20s
      retries: 3
```

> 改完端口后，需重启 Milvus 生效：`cd ~/milvus && docker compose restart`

**第四步：部署 Attu 2.4.0 可视化客户端（适配 Milvus 2.4.11）**

```bash
# 1. 先删除旧版Attu（若未安装，此命令无影响）
docker rm -f attu

# 2. 启动新版Attu 2.4.0（适配Milvus 2.4.11，后台运行）
# 关键：MILVUS_URL 填写Milvus服务器的IP+默认端口19530（本地部署填127.0.0.1:19530，远程服务器填公网/内网IP）
docker run -d --name attu \
    -p 8000:3000 \  # 宿主机8000端口映射容器3000端口
    -e MILVUS_URL=47.94.86.115:19530 \
    zilliz/attu:v2.4.0
```

**第五步：Attu 连接测试（验证 Milvus 部署成功）**

1. 服务器开放端口：若为云服务器 / 防火墙开启状态，需放行 8000 端口（Milvus 默认 19530 端口无需外部访问，Attu 用 8000 端口）

   ```
   # CentOS 放行8000端口
   firewall-cmd --add-port=8000/tcp --permanent
   firewall-cmd --reload
   ```

2. 浏览器访问：打开本地浏览器，输入地址 

   ```
   http://<Linux服务器IP>:8000
   ```

   - 本地虚拟机 / 服务器本地访问：`http://127.0.0.1:8000`
   - 远程服务器访问：`http://服务器公网/内网IP:8000`

3. 连接 Milvus：页面无需输入账号密码，直接点击「Connect」按钮，若能进入 Attu 可视化界面，即**Milvus 部署 + Attu 连接全部成功**。

##### 4.2 定义Milvus客户端工具类

**步骤1：定义.env配置参数**

文件：`.env`

```ini
# Milvus 配置
# 切换成你milvus的url地址
MILVUS_URL=http://47.94.86.115:19530
# 存储切片集合
CHUNKS_COLLECTION=kb_chunks
# 预留
ENTITY_NAME_COLLECTION=kb_graph_entity_names
# 存储每个文档对应的实体类
ITEM_NAME_COLLECTION=kb_item_names
```

**步骤2：读取配置参数**

文件：`app.config.milvus_config.py`

```python
# 导入核心依赖（和其他配置类共用，只需导入一次）
from dataclasses import dataclass
import os
from dotenv import load_dotenv

# 提前加载.env配置文件（全局执行一次即可，无需重复写）
load_dotenv()

# ===================== 其他配置类（LLM/Embedding）可放在上方，保持原有代码不变 =====================
# ... 你的LLMConfig、EmbeddingConfig代码 ...

# 定义Milvus向量数据库配置类
@dataclass
class MilvusConfig:
    milvus_url: str          # Milvus服务端连接地址
    chunks_collection: str   # 存储切片的集合名称
    entity_name_collection: str  # 预留-实体名称集合
    item_name_collection: str    # 存储文档对应实体类的集合名称

# 实例化Milvus配置对象（和其他配置对象命名风格统一）
milvus_config = MilvusConfig(
    milvus_url=os.getenv("MILVUS_URL"),
    chunks_collection=os.getenv("CHUNKS_COLLECTION"),
    entity_name_collection=os.getenv("ENTITY_NAME_COLLECTION"),
    item_name_collection=os.getenv("ITEM_NAME_COLLECTION")
)
```

**步骤3：定义Milvus客户端类**

```python
import os
from pymilvus import MilvusClient, AnnSearchRequest, WeightedRanker
from app.conf.milvus_config import milvus_config
from app.core.logger import logger

# 全局Milvus客户端实例，实现单例复用
_milvus_client = None


def get_milvus_client():
    """
    Milvus客户端单例获取方法
    实现客户端连接复用，避免重复创建连接消耗资源
    :return: MilvusClient实例，连接失败返回None
    """
    try:
        global _milvus_client
        # 单例判断：未初始化则创建新连接
        if _milvus_client is None:
            milvus_uri = milvus_config.milvus_url
            # 校验Milvus连接地址配置
            if not milvus_uri:
                logger.error("Milvus客户端连接失败：缺少MILVUS_URL环境变量配置")
                return None
            # 初始化Milvus客户端
            _milvus_client = MilvusClient(uri=milvus_uri)
            logger.info("Milvus客户端连接成功")
        return _milvus_client
    except Exception as e:
        logger.error(f"Milvus客户端连接异常：{str(e)}", exc_info=True)
        return None


def _coerce_int64_ids(ids):
    """
    转换chunk_id为Milvus要求的INT64类型（主键字段schema为INT64）
    过滤无效ID，分离可转换/不可转换的ID
    :param ids: 待转换的chunk_id列表
    :return: 元组(ok_ids, bad_ids)，ok_ids为可转换的int64类型ID列表，bad_ids为无效ID列表
    """
    ok, bad = [], []
    for x in (ids or []):
        if x is None:
            continue
        try:
            ok.append(int(x))
        except Exception:
            bad.append(x)
    return ok, bad


def fetch_chunks_by_chunk_ids(
        client,
        collection_name: str,
        chunk_ids,
        *,
        output_fields=None,
        batch_size: int = 100,
):
    """
    通过chunk_id主键批量查询Milvus中的切片数据
    用于补全「仅拥有chunk_id无文本内容」场景的切片信息
    优先使用get方法（主键直查，性能最优），失败则回退query过滤查询
    :param client: MilvusClient实例
    :param collection_name: 集合名称
    :param chunk_ids: 待查询的chunk_id列表
    :param output_fields: 需要返回的字段列表，默认返回核心切片字段
    :param batch_size: 分批查询大小，避免单次查询数据量过大，默认100
    :return: List[dict]，Milvus实体字典列表，查询失败返回空列表
    """
    # 前置校验：客户端/集合名无效直接返回空
    if client is None:
        return []
    if not collection_name:
        return []
    # 默认返回字段：核心切片标识与内容字段
    if output_fields is None:
        output_fields = ["chunk_id", "content", "title", "parent_title", "item_name"]

    # 转换ID为INT64类型，分离有效/无效ID
    ok_ids, bad_ids = _coerce_int64_ids(chunk_ids)
    if bad_ids:
        # 记录无效ID，跳过查询
        logger.warning(f"存在无法转换为INT64的chunk_id，将跳过查询：{bad_ids}")

    # 无有效ID直接返回空
    if not ok_ids:
        return []

    results = []
    # 分批查询：按batch_size切分有效ID，循环查询
    for i in range(0, len(ok_ids), batch_size):
        batch = ok_ids[i: i + batch_size]

        # 方式1：优先使用主键get方法查询（性能最优）
        if hasattr(client, "get"):
            try:
                got = client.get(collection_name=collection_name, ids=batch, output_fields=output_fields)
                if got:
                    results.extend(got)
                continue
            except Exception as e:
                logger.warning(f"Milvus get方法查询失败，将回退至query方法：{str(e)}")

        # 方式2：get方法失败，回退使用filter过滤查询
        try:
            expr = f"chunk_id in [{', '.join(str(x) for x in batch)}]"
            q = client.query(collection_name=collection_name, filter=expr, output_fields=output_fields)
            if q:
                results.extend(q)
        except Exception as e:
            logger.error(f"Milvus query方法批量查询chunk_id失败：{str(e)}", exc_info=True)

    return results


def create_hybrid_search_requests(dense_vector, sparse_vector, dense_params=None, sparse_params=None, expr=None,
                                  limit=5):
    """
    构建Milvus混合搜索请求对象
    分别创建稠密/稀疏向量的搜索请求，用于后续混合搜索融合
    :param dense_vector: 文本生成的稠密向量
    :param sparse_vector: 文本生成的稀疏向量
    :param dense_params: 稠密向量搜索参数，默认使用余弦相似度
    :param sparse_params: 稀疏向量搜索参数，默认使用内积相似度
    :param expr: 搜索过滤表达式，用于精准筛选数据
    :param limit: 单向量搜索返回结果数量，默认5
    :return: 搜索请求列表，包含[dense_req, sparse_req]
    """
    # 稠密向量默认搜索参数：余弦相似度（COSINE），适配BGE-M3稠密向量
    if dense_params is None:
        dense_params = {"metric_type": "COSINE"}
    # 稀疏向量默认搜索参数：内积（IP），适配BGE-M3稀疏向量
    if sparse_params is None:
        sparse_params = {"metric_type": "IP"}

    # 构建稠密向量搜索请求，关联Milvus的dense_vector字段
    dense_req = AnnSearchRequest(
        data=[dense_vector],
        anns_field="dense_vector",
        param=dense_params,
        expr=expr,
        limit=limit
    )

    # 构建稀疏向量搜索请求，关联Milvus的sparse_vector字段
    sparse_req = AnnSearchRequest(
        data=[sparse_vector],
        anns_field="sparse_vector",
        param=sparse_params,
        expr=expr,
        limit=limit
    )

    return [dense_req, sparse_req]


def hybrid_search(client, collection_name, reqs, ranker_weights=(0.5, 0.5), norm_score=False, limit=5,
                  output_fields=None, search_params=None):
    """
    执行Milvus稠密+稀疏向量混合搜索
    基于WeightedRanker实现双向量搜索结果加权融合，提升检索准确性
    :param client: MilvusClient实例
    :param collection_name: 集合名称
    :param reqs: 搜索请求列表，固定为[dense_req, sparse_req]
    :param ranker_weights: 加权融合权重，默认(0.5,0.5)，依次对应稠密/稀疏向量
    :param norm_score: 是否归一化评分后再融合，避免评分量级差异导致权重失效
    :param limit: 混合搜索最终返回结果数量，默认5
    :param output_fields: 需要返回的字段列表，默认返回item_name
    :param search_params: 搜索参数，如ef/topk等，默认None
    :return: 混合搜索结果列表，搜索失败返回None
    """
    try:
        # 初始化加权排名器：按权重融合稠密/稀疏向量的搜索结果
        # norm_score=True：先将两个向量评分归一化到0~1区间，再加权计算
        rerank = WeightedRanker(ranker_weights[0], ranker_weights[1], norm_score=norm_score)

        # 默认返回字段：文档标识字段
        if output_fields is None:
            output_fields = ["item_name"]

        # 执行混合搜索：融合稠密+稀疏向量结果，按权重重新排序
        res = client.hybrid_search(
            collection_name=collection_name,
            reqs=reqs,
            ranker=rerank,
            limit=limit,
            output_fields=output_fields,
            search_params=search_params
        )

        logger.info(f"Milvus混合搜索完成，集合[{collection_name}]共检索到{len(res[0])}条结果")
        return res
    except Exception as e:
        logger.error(f"Milvus混合搜索执行失败，集合[{collection_name}]：{str(e)}", exc_info=True)
        return None
```

#### 5. 导入与配置

引入必要的依赖库，并定义默认配置参数。

```python
from langchain_core.messages import SystemMessage, HumanMessage

from app.conf.milvus_config import milvus_config
# 导入自定义模块：
# 1. 流程状态载体：ImportGraphState为LangGraph流程的统一状态管理对象
from app.import_process.agent.state import ImportGraphState
# 2. Milvus工具：获取单例Milvus客户端，实现连接复用
from app.clients.milvus_utils import get_milvus_client
# 3. 大模型工具：获取大模型客户端，统一模型调用入口
from app.lm.lm_utils import get_llm_client
# 4. 向量工具：BGE-M3模型实例、向量生成方法（稠密+稀疏向量）
from app.lm.embedding_utils import get_bge_m3_ef, generate_embeddings
# 5. 稀疏向量工具：归一化处理，保证向量长度为1，提升检索准确性
from app.utils.normalize_sparse_vector import normalize_sparse_vector
# 6. 任务工具：更新任务运行状态，用于任务监控和管理
from app.utils.task_utils import add_running_task, add_done_task
# 7. 日志工具：项目统一日志入口，分级输出（info/warning/error）
from app.core.logger import logger,node_log,step_log
# 8. 提示词工具：加载本地prompt模板，实现提示词与代码解耦
from app.core.load_prompt import load_prompt

from app.utils.escape_milvus_string_utils import escape_milvus_string

# --- 配置参数 (Configuration) ---
# 大模型识别商品名称的上下文切片数：取前5个切片，避免上下文过长导致大模型输入超限
DEFAULT_ITEM_NAME_CHUNK_K = 5
# 单个切片内容截断长度：防止单切片内容过长，占满大模型上下文
SINGLE_CHUNK_CONTENT_MAX_LEN = 800
# 大模型上下文总字符数上限：适配主流大模型输入限制，默认2500
CONTEXT_TOTAL_MAX_CHARS = 2500
```

#### 6. 核心辅助函数

处理 Milvus 字符串转义等底层逻辑。

```python
from app.utils.escape_milvus_string_utils import escape_milvus_string
```

#### 7. 主流程定义

LangGraph 节点的入口函数，负责串联所有步骤。

```python
"""
  主要目标：
     1. 录用文本大模型识别当前chunks对应的item_name！用于区分不同的文档
     2. 使用嵌入式模型，将item_name生成向量存储到向量数据库 
     3. 修改state[chunks] -> chunk {title parent_title part file_title content item_name => 每个赋值 }
  实现步骤：
     1. 校验和取值 （file_title,chunks）
     2. 构建上下文环境  chunks -> top 5 -> 拼接成context文本 
     3. 调用模型，拼接提示词，识别chunks对应item_name
     4. 修改state chunks -》 item_name 
     5. item_name生成向量（稠密/稀疏）
     6. 存储向量到向量数据库 kb_item_name (id / file_title / item_name / 稠密 和 稀疏)
 """
@node_log("node_item_name_recognition")
def node_item_name_recognition(state: ImportGraphState) -> ImportGraphState:
    """
    节点: 主体识别 (node_item_name_recognition)
    为什么叫这个名字: 识别文档核心描述的物品/商品名称 (Item Name)。
    """
    # 日志和任务处理
    add_running_task(state['task_id'],'node_item_name_recognition')
    # 1. 校验和取值 （file_title,chunks）
    # 获取前置的材料！ file_title = 为了兜底，没有识别到item_name
    chunks, file_title = step_1_get_chunks_and_file_title(state)
    # 2. 构建上下文环境  chunks -> top 5 -> 拼接成context文本
    context = step_2_build_context(chunks)
    # 3. 调用模型，拼接提示词，识别chunks对应item_name
    item_name = step_3_call_llm(context, file_title)
    # 4. 修改state chunks -》 item_name -> chunks [{title parent_title context part item_name [没有值]}]
    step_4_update_chunks_and_state(state, item_name, chunks)
    # 5. item_name生成向量（稠密/稀疏）
    dense_vector, sparse_vector = step_5_generate_embeddings(item_name)
    # 6. 将向量存储到向量数据库 kb_item_name (id / file_title / item_name / 稠密 和 稀疏)
    step_6_save_to_vector_db(file_title, item_name, dense_vector, sparse_vector)
    
    add_done_task(state['task_id'], 'node_item_name_recognition')
    return state
```

#### 8. 步骤 1: 获取输入 

从 State 中提取文件名和切片数据，并进行基础校验。

```python
@step_log("step_1_get_chunks_and_file_title")
def step_1_get_chunks_and_file_title(state) -> Tuple[str, str]:
    """
    继续进行参数校验和处理!
    :param state:
    :return:
    """
    chunks = state.get('chunks')
    file_title = state.get('file_title')

    if not chunks:
        raise ValueError("chunks没有值，无法继续进行，抛出异常处理！")
    if not file_title:
        # file_title没有值！
        # md_path中获取文件名即可 (字符串处理更方便)
        file_title = os.path.basename(state.get('md_path'))
        state['file_title'] = file_title
    return chunks, file_title
```

#### 9. 步骤 2: 构建上下文

截取文档的前 K 个切片，拼接成用于 LLM 识别的 Context。

```python
@step_log("step_2_build_context")
def step_2_build_context(chunks) -> str:
    """
    构建提示词上下文环境
    根据chunks切面的content内容进行分拼接！ （2000）
    截取内容限制： 1. 最多截取前top个 （5） 2. 最多字符不能超过 CONTEXT_TOTAL_MAX_CHARS
    截取内容处理：
          切片：{1}，标题:{title},内容：{content} \n\n
          切片：{2}，标题:{title},内容：{content} \n\n
          切片：{3}，标题:{title},内容：{content} \n\n
          切片：{4}，标题:{title},内容：{content} \n\n
          切片：{5}，标题:{title},内容：{content} \n\n
    :param chunks:
    :return:
    """
    #1. 前置准备工作
    parts = []  # 存储处理后的切片：{1}，标题:{title},内容：{content} \n\n
    total_chars = 0  # 记录已经加入列表的字符串数量
    #2. 循环处理 content + 判断
    for index,chunk in enumerate(chunks[:DEFAULT_ITEM_NAME_CHUNK_K], start=1):
        chunk_title = chunk['title']
        chunk_content = chunk['content']
        # 先处理一下！！
        # if len(chunk_content) + total_chars > SINGLE_CHUNK_CONTENT_MAX_LEN:
        #     chunk_content = chunk_content[:SINGLE_CHUNK_CONTENT_MAX_LEN-total_chars]
        data = f"切片：{index}，标题:{chunk_title},内容：{chunk_content}"
        parts.append(data)
        total_chars += len(data)
        # 第一次的content已经超标了但是完成了拼接！！！
        if total_chars >= CONTEXT_TOTAL_MAX_CHARS:
            logger.info(f"已经达到最大字符数:{total_chars}，停止拼接！")
            break
    # 结果的转化
    context = "\n\n".join(parts)
    # 兜底处理下
    final_context = context[:SINGLE_CHUNK_CONTENT_MAX_LEN]
    # 返回结果
    return final_context
```

#### 10. 步骤 3: 调用 

构造 Prompt 并调用大模型，识别商品名称。

```python
@step_log("step_3_call_llm")
def step_3_call_llm(context, file_title) -> str:
    """
    想模型调用! 获取item_name!
    使用file_tile进行兜底！！
    :param context:
    :param file_title:
    :return:
    """
    # 1. 构建提示词
    human_prompt = load_prompt("item_name_recognition", file_title=file_title, context=context)
    system_prompt = load_prompt("product_recognition_system")
    messages = [
        HumanMessage(content=human_prompt),
        SystemMessage(content=system_prompt)
    ]
    # 2. 获取模型对象
    llm = get_llm_client(json_mode=False)
    # 3. 组建调用链
    chain = llm | StrOutputParser ()
    item_name = chain.invoke(messages)

    # 4. 进行判断
    if not item_name:
        item_name = file_title
    # 5. 返回结果
    return item_name
```

#### 11. 步骤 4: 回填数据

将识别到的 `item_name` 回填到 State 和 Chunks 中。

```python
@step_log("step_4_update_chunks_and_state")
def step_4_update_chunks_and_state(state, item_name, chunks):
    """
    state[item_name] = item_name
    chunks -> {item_name:item_name}
    :param state:
    :param item_name:
    :param chunks:
    :return:
    """
    state['item_name'] = item_name

    for chunk in chunks:
        chunk['item_name'] = item_name
    state['chunks'] = chunks
    logger.info(f"完成了chunks和state[item_name]的赋值和修改！！")
```

#### 12. 步骤 5: 生成向量

使用 Embedding 模型为商品名称生成向量。

BGE-M3 模型同时输出这两种向量，结合使用能兼顾 “语义理解” 和 “精准匹配”。

```python
@step_log("step_5_generate_embeddings")
def step_5_generate_embeddings(item_name):
    """
    根据item_name生成向量 -》 稠密 + 稀疏
    :param item_name:
    :return: dense_vector [稠密] ,  sparse_vector [稀疏]
    """
    """
    generate_embeddings 自己封装的嵌入式模式生成向量的函数！！ 
          embeddings list对应的向量 = model.encode_documents(texts) 传入的字符串list 
          参数：生成向量的字符串 ["1","2","3"] 
          返回结果： 
             result = {
                        "dense": [1的稠密,2的稠密,3的稠密],  #稠密向量
                        "sparse": [1的稀疏,2的稀疏,3的稀疏], #稀疏向量
                      }
    """
    result = generate_embeddings([item_name])
    dense_vector,sparse_vector = result['dense'][0],result['sparse'][0]
    return dense_vector,sparse_vector
```

#### 13. 步骤 6: 保存结果 

将识别结果及向量保存到 Milvus 数据库。

```python
@step_log("step_6_save_to_vector_db")
def step_6_save_to_vector_db(file_title, item_name, dense_vector, sparse_vector):
    """
    将向量和对应的字段保存到向量数据库中！！
    :param file_title:
    :param item_name:
    :param dense_vector:
    :param sparse_vector:
    :return:
    """
    # 1. 获取milvus的客户端
    milvus_client = get_milvus_client()
    # 2. 判断是否存在集合（表），存在创建集合（表）
    if not milvus_client.has_collection(collection_name=milvus_config.item_name_collection):
        # 创建集合
        # 3.1. 创建集合对应的列的信息
        schema = milvus_client.create_schema(
            auto_id=True, # 主键自增长
            enable_dynamic_field=True, # 动态字段
        )

        # 3.2. Add fields to schema
        # pk file_title item_name dense_vector sparse_vector
        schema.add_field(field_name="pk", datatype=DataType.INT64, is_primary=True,auto_id=True)
        schema.add_field(field_name="file_title", datatype=DataType.VARCHAR, max_length =65535)
        schema.add_field(field_name="item_name", datatype=DataType.VARCHAR, max_length =65535)
        schema.add_field(field_name="dense_vector", datatype=DataType.FLOAT_VECTOR, dim=1024)
        schema.add_field(field_name="sparse_vector", datatype=DataType.SPARSE_FLOAT_VECTOR)
        # 3.3 查询快，配置索引
        index_params = milvus_client.prepare_index_params()

        index_params.add_index(
            field_name="dense_vector",  #给哪个列创建索引 稠密
            index_name="dense_vector_index",  # 索引的名字
            index_type="HNSW",  # 配置查找所用的算法
            metric_type="COSINE", # 配置向量匹配和对比的 IP COSINE
            params={"M": 16, # Maximum number of neighbors each node can connect to in the graph
                    "efConstruction": 200 },  # or "DAAT_WAND" or "TAAT_NAIVE"
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
            collection_name=milvus_config.item_name_collection,
            schema=schema, #字段
            index_params=index_params # 索引
        )
    # 3. 先删除之前存在的item_name
    # 加载和选中集合
    milvus_client.load_collection(collection_name=milvus_config.item_name_collection)
    milvus_client.delete(collection_name=milvus_config.item_name_collection,
                         filter=f"item_name=='{item_name}'")
    # 4. 向集合插入最新的item_name数据和对应的向量即可
    item = {
        "file_title": file_title,
        "item_name": item_name,
        "dense_vector": dense_vector,
        "sparse_vector": sparse_vector
    }
    milvus_client.insert(collection_name=milvus_config.item_name_collection,
                         data=[item])
    milvus_client.load_collection(collection_name=milvus_config.item_name_collection)
    logger.info(f"保存了item_name:{item_name}的数据到向量数据库中！！")
```

#### 15. 单元测试

模拟数据测试核心流程。

```python
# ===================== 本地测试方法（直接运行调试，无需启动LangGraph） =====================
def test_node_item_name_recognition():
    """
    商品名称识别节点本地测试方法
    功能：模拟LangGraph流程输入，独立测试node_item_name_recognition节点全链路逻辑
    适用场景：本地开发、调试、单节点功能验证，无需启动整个LangGraph流程
    测试前准备：
        1. 确保项目环境变量配置完成（MILVUS_URL/ITEM_NAME_COLLECTION等）
        2. 确保大模型、Milvus、BGE-M3服务均可正常访问
        3. 确保prompt模板（item_name_recognition/product_recognition_system）已存在
    使用方法：
        直接运行该函数：if __name__ == "__main__": test_node_item_name_recognition()
    """
    logger.info("=== 开始执行商品名称识别节点本地测试 ===")
    try:
        # 1. 构造模拟的ImportGraphState状态（模拟上游节点产出数据）
        mock_state = ImportGraphState({
            "task_id": "test_task_123456",  # 测试任务ID
            "file_title": "华为Mate60 Pro手机使用说明书",  # 模拟文件标题
            "file_name": "华为Mate60Pro说明书.pdf",  # 模拟原始文件名（兜底用）
            # 模拟文本切片列表（上游切片节点产出，含title/content字段）
            "chunks": [
                {
                    "title": "产品简介",
                    "content": "华为Mate60 Pro是华为公司2023年发布的旗舰智能手机，搭载麒麟9000S芯片，支持卫星通话功能，屏幕尺寸6.82英寸，分辨率2700×1224。"
                },
                {
                    "title": "拍照功能",
                    "content": "华为Mate60 Pro后置5000万像素超光变摄像头+1200万像素超广角摄像头+4800万像素长焦摄像头，支持5倍光学变焦，100倍数字变焦。"
                },
                {
                    "title": "电池参数",
                    "content": "电池容量5000mAh，支持88W有线超级快充，50W无线超级快充，反向无线充电功能。"
                }
            ]
        })

        # 2. 调用商品名称识别核心节点
        result_state = node_item_name_recognition(mock_state)

        # 3. 打印测试结果（调试用）
        logger.info("=== 商品名称识别节点本地测试完成 ===")
        logger.info(f"测试任务ID：{result_state.get('task_id')}")
        logger.info(f"最终识别商品名称：{result_state.get('item_name')}")
        logger.info(f"切片数量：{len(result_state.get('chunks', []))}")
        logger.info(f"第一个切片商品名称：{result_state.get('chunks', [{}])[0].get('item_name')}")

        # 4. 验证Milvus存储（可选）
        milvus_client = get_milvus_client()
        collection_name = os.environ.get("ITEM_NAME_COLLECTION")
        if milvus_client and collection_name:
            milvus_client.load_collection(collection_name)
            # 检索测试结果
            item_name = result_state.get('item_name')
            safe_name = _escape_milvus_string(item_name)
            res = milvus_client.query(
                collection_name=collection_name,
                filter=f'item_name=="{safe_name}"',
                output_fields=["file_title", "item_name"]
            )
            logger.info(f"Milvus中检索到的数据：{res}")

    except Exception as e:
        logger.error(f"商品名称识别节点本地测试失败，原因：{str(e)}", exc_info=True)


# 测试方法运行入口：直接执行该文件即可触发测试
if __name__ == "__main__":
    # 执行本地测试
    test_node_item_name_recognition()
```

