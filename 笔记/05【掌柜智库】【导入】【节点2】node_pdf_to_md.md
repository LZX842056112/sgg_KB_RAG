# 掌柜智库项目(RAG)实战

## 5. 导入数据节点实现与测试

### 5.2 PDF 转 Markdown (node_pdf_to_md)

**文件**: `app/import_process/agent/nodes/node_pdf_to_md.py`
**相关工具类位置**: `app/utils/task_utils.py`

#### 节点作用与实现思路

解决**非结构化 PDF 无法直接用于 RAG 知识库**的核心痛点：

- 普通 PDF 文本乱码、分段错乱、表格丢失、图片无法识别；
- 通过 **MinerU 企业级高精度解析**，将 PDF 一键转为 **结构完整、带图片、带表格、带排版** 的标准 Markdown 文档；
- 为后续**文本分块、向量入库、多模态理解**提供高质量数据源。

**为什么选择 MinerU?**

<img src="assets/image-20260422135823601.png" alt="image-20260422135823601" style="zoom:50%;" />

- **超高精度解析**：文字、标题、目录、表格、公式全还原
- **强 Layout 分析**：保留原文结构，不打乱段落
- **内置 OCR**：图片版 PDF、扫描件也能转文字
- **多模态输出**：自动提取图片并生成引用，支持图文一体知识库
- **云端异步解析**：不占本地 GPU / 内存，大文件不崩溃
- **企业级稳定**：适合工程化、批量入库、生产环境使用
- 官网地址: https://mineru.net/
- 文档地址: https://opendatalab.github.io/MinerU/zh/quick_start/

**实现思路**:

1.  **云端算力卸载**: 考虑到本地 OCR 和布局分析的资源消耗巨大，本节点采用“云端 API”模式，将繁重的解析任务卸载到 MinerU 服务端。
2.  **异步轮询机制**: 针对大文件解析耗时较长的问题，设计了“上传 -> 提交 -> 轮询 -> 下载”的异步交互流程，通过**指数退避**或固定间隔轮询状态，避免长连接超时。
3.  **完整性保障**: 解析完成后，不仅提取 Markdown 文本，还同步下载并解压关联的图片资源，保持文档的多模态完整性。

#### 步骤分解

1.  **准备参数**: 获取 PDF 路径和输出目录。
2.  **请求上传**: 调用 MinerU 在线 API (`/file-urls/batch`) 获取上传链接。
3.  **上传文件**: 将 PDF 文件 PUT 到签名 URL。
4.  **轮询结果**: 循环查询任务状态 (`/extract-results/batch/{batch_id}`)，直到完成。
5.  **获取结果**: 下载生成的 ZIP 包，解压并读取 `.md` 文件内容到 state。

**注意**: 需要在 `.env` 中配置 `MINERU_BASE_API_TOKEN`。

API申请位置：https://mineru.net/apiManage/token （有效期90天）

参考官方文档：https://mineru.net/doc/docs/index.html?theme=light&v=1.0#%E6%89%B9%E9%87%8F%E6%96%87%E4%BB%B6%E8%A7%A3%E6%9E%90

#### 1. 导入与配置 (Imports & Config)

添加全局配置 .env文件

```ini
# mineru api以及url处理地址！
MINERU_API_TOKEN=eyJ0eXBlIjoiSldUIiwiYWxnIjoiSFM1MTIifQ.eyJqdGkiOiI3NjUwMDY4NiIsInJvbCI6IlJPTEVfUkVHSVNURVIiLCJpc3MiOiJPcGVuWExhYiIsImlhdCI6MTc3MzY3MDk2OSwiY2xpZW50SWQiOiJsa3pkeDU3bnZ5MjJqa3BxOXgydyIsInBob25lIjoiIiwib3BlbklkIjpudWxsLCJ1dWlkIjoiMTBkYTM3MTgtMzA3YS00ODdkLWJhNDEtZjNkYzlmMGQ3ZmE5IiwiZW1haWwiOiIiLCJleHAiOjE3ODE0NDY5Njl9.9DF0FHvQYaK6NRnOVG87V_IYToHPRcyHcQd-tuXETuPIJzVV6h6PMc3RtY5zeaIN7LiPRobziXqAKWoDNl3ueA
MINERU_BASE_URL=https://mineru.net/api/v4
```

定义配置类，读取配置文件 

位置：`app/import_process/config/mineru_config.py`

```python
# 导入核心依赖：数据类、环境变量读取、路径处理
from dataclasses import dataclass
import os
from dotenv import load_dotenv

# 提前加载.env配置文件（必须在读取环境变量前执行，确保os.getenv能获取到值）
# 若.env不在项目根目录，可指定路径：load_dotenv(dotenv_path=Path(__file__).parent / ".env")
load_dotenv()

# 定义minerU服务配置
@dataclass
class MineruConfig:
    base_url: str
    api_token : str

mineru_config = MineruConfig(
    base_url=os.getenv("MINERU_BASE_URL"),
    api_token=os.getenv("MINERU_API_TOKEN")
)
```

首先引入必要的库，并加载环境变量配置。

```python
import os
import shutil
import sys
import time
import zipfile
from pathlib import Path

import requests

from app.core.logger import logger, node_log, step_log
from app.import_process.agent.state import ImportGraphState, create_default_state
from app.utils.path_util import PROJECT_ROOT
from app.utils.task_utils import add_running_task, add_done_task
from app.conf.mineru_config import mineru_config
```

#### 2. 主流程定义 (Main Flow)

定义 `node_pdf_to_md` 入口函数，清晰展示三个主要步骤的调用关系。

```python
"""
   node_pdf_to_md 
     参数： state [is_pdf_read_enabled = True | pdf_path = xxx.pdf | local_dir = output ] 
     返回： state [md_path = 地址 | md_content = 内容 ]
     1. 日志和任务状态
     2. step_1_validate_paths路径校验
     3. step_2_upload_and_poll minerU的交互
     4. step_3_download_and_extract 下载和解压
     5. 日志和任务状态 return state 
   step_1_validate_paths
     参数：state pdf_path = xxx.pdf | local_dir = output
     返回： pdf_path_obj Path  local_dir_obj Path 
     1. 非空校验 
     2. 文件校验 pdf_path_obj 没有抛异常 local_dir_obj 没有给与默认
     3. 返回完成可用的Path对象即可
   step_2_upload_and_poll
     参数：pdf对应Path  pdf_path_obj
     返回：str zip url地址
     1. 进行申请，获取要上传文件的地址 
     2. 进行文件上传 session | requests.put 
     3. 轮询获取返回结果 zip_url  （确定一个最大等待时间 1页pdf 1s 间隔时间3 错误码 200 -》 500能容忍）
     4. 返回地址即可
   step_3_download_and_extract
     参数：zip_url , out_dir_obj , 原文件名 path.stem
     返回：解压后的.md的str地址 
     1. zip下载 get    output / stem_result.zip
     2. 检查解压的文件夹地址  output / stem 
     3. 检查解压的文件夹进行防重复处理
     4. 进行解压 zipFile  extractall(解压的目标文件夹)
     5. 考虑文件名字 原文件件名 还是 full 还是其他 
     6. 重命名处理
     7. 路径转成字符串 获取绝对路径最终返回即可！
"""
@node_log("node_pdf_to_md")
def node_pdf_to_md(state: ImportGraphState) -> ImportGraphState:
    """
    节点: PDF转Markdown (node_pdf_to_md)
    为什么叫这个名字: 核心任务是将 PDF 非结构化数据转换为 Markdown 结构化数据。
    """
    # 1.注解日志和进行和完成任务状态处理
    add_running_task(state['task_id'], 'node_pdf_to_md')
    # 2. step_1_validate_paths 校验路径完整以及是否真实存在
    pdf_path_obj , output_dir_pbj = step_1_validate_paths(state)
    # 3. step_2_upload_and_poll
    zip_url = step_2_upload_and_poll(pdf_path_obj, output_dir_pbj)
    # 4. step_3_download_and_extract
    md_path = step_3_download_and_extract(zip_url, output_dir_pbj, pdf_path_obj.stem)
    # 5. 处理响应结果 state进行赋值处理
    state['md_path'] = md_path
    # Path方案 使用read_text()
    # 原始方案
    with open(md_path, 'r', encoding='utf-8') as f:
        state['md_content'] = f.read()
    add_done_task(state['task_id'], 'node_pdf_to_md')
    return state
```



#### 3. 步骤 1: 校验路径 (Step 1: Validate Paths)

这一步负责检查 `state` 中的参数，并确保文件存在。

```python
@step_log("step_1_validate_paths")
def step_1_validate_paths(state):
    """
    步骤1：路径校验与初始化
    校验PDF输入文件与输出目录的有效性，遵循「输入严格校验、输出自动修复」的鲁棒性设计原则：
    1. 校验PDF路径非空且文件真实存在，不存在则直接抛出异常（快速失败）
    2. 校验输出目录，为空则赋予默认值，不存在则自动创建（自动容错）
    3. 统一转换为Path对象处理，保证路径操作的规范性与跨平台兼容性
    :param state: 流程状态字典，包含pdf_path、local_dir
    :return: 无返回值，直接修改状态字典
    """
    # 1. 获取路径参数
    pdf_path = state.get("pdf_path", "").strip()
    local_dir = state.get("local_dir", "").strip()
    # 2. 参数非空校验
    if not pdf_path:
        raise ValueError("pdf_path 不能为空，请提供有效的PDF文件路径")
    if not local_dir:
        local_dir = PROJECT_ROOT / "output"
        state["local_dir"] = str(local_dir)
        logger.warning(f"未指定输出目录，使用默认路径：{local_dir}")
    # 3. 统一转换为Path对象，标准化路径处理
    pdf_path_obj = Path(pdf_path)
    local_dir_obj = Path(local_dir)
    # 4. 路径有效性校验（差异化处理：输入严格校验，输出自动修复）
    if not pdf_path_obj.exists():
        raise FileNotFoundError(f"PDF文件不存在：{pdf_path_obj}，请检查文件路径是否正确")
    if not local_dir_obj.exists():
        logger.warning(f"输出目录不存在，自动创建：{local_dir_obj}")
        local_dir_obj.mkdir(parents=True, exist_ok=True)
    return pdf_path_obj, local_dir_obj
```

**总结 : from pathlib import Path**

传统路径操作依赖`os.path`模块（如`os.path.exists()`/`os.path.join()`），需要对**字符串路径**做各种处理，而`Path`对象是**面向对象的路径操作方式**，把路径的「属性」和「操作方法」封装在对象中，相比字符串路径 +`os.path`，优势极其明显，也是 Python3.4 + 官方推荐的路径操作方式：

1. **跨平台兼容性强，无需手动处理路径分隔符**

不同系统的路径分隔符不同：Windows 用`\`，Linux/Mac 用`/`，用字符串路径时需要手动适配（如`os.path.join()`），而`Path`对象会**自动识别当前系统**，生成对应分隔符的路径，代码一次编写，多平台运行。

```python
# 字符串路径（需手动用os.path.join适配）
import os
path1 = os.path.join("D:", "test", "pdf.pdf")  # Windows→D:\test\pdf.pdf，Linux→D:/test/pdf.pdf

# Path对象（自动适配，更简洁）
from pathlib import Path
path2 = Path("D:") / "test" / "pdf.pdf"  # 用/直接拼接，自动适配系统分隔符
```

2. **路径拼接更直观，支持「/」运算符**

字符串路径拼接需要使用`os.path.join()`，语法繁琐且容易出错（如忘记加分隔符）；`Path`对象支持**直接用 / 运算符拼接路径**，和日常书写路径的习惯一致，代码更简洁、可读性更高。

3. **封装了路径的「属性」，直接获取路径信息，无需手动解析**

字符串路径获取「文件名、后缀、纯名称、父目录」等信息，需要结合`os.path.basename()`/`os.path.splitext()`等多个函数，而`Path`对象内置了这些**只读属性**，直接点取即可，无需额外处理：

```python
p = Path("D:/test/report_v1.2.pdf")
print(p.name)      # 获取完整文件名：report_v1.2.pdf
print(p.stem)      # 获取文件纯名称（无后缀）：report_v1.2（你代码中核心使用的属性）
print(p.suffix)    # 获取文件后缀：.pdf
print(p.parent)    # 获取父目录：D:/test
print(p.absolute())# 获取绝对路径：D:\test\report_v1.2.pdf（Windows）
```

4. **封装了路径的「操作方法」，直接调用，无需传参**

`os.path`的所有操作函数（如判断存在、判断是文件 / 目录）都需要**将路径字符串作为参数传入**，而`Path`对象将这些操作封装为**自身方法**，直接调用即可，无需重复传参，语法更简洁：

```python
p = Path("D:/test.pdf")
d = Path("D:/output")

# Path对象（直接调用方法）
p.exists()    # 判断路径是否存在，等价于os.path.exists(p)
p.is_file()   # 判断是否是文件，等价于os.path.isfile(p)
d.mkdir(parents=True, exist_ok=True)  # 创建目录，等价于os.makedirs(d, exist_ok=True)

# 字符串路径+os.path（需传参，繁琐）
import os
os.path.exists(str(p))
os.path.isfile(str(p))
os.makedirs(str(d), exist_ok=True)
```

5. **方法链式调用，简化多步路径操作**

`Path`对象的多数方法执行后，**返回的还是 Path 对象**，支持**链式调用**，可以把多个路径操作写在一行，相比`os.path`的多次函数调用，代码更简洁：

```python
# 链式调用：创建父目录 → 拼接子文件路径 → 获取绝对路径
new_file = Path("D:/temp/test").parent.mkdir(parents=True, exist_ok=True) / "new.pdf"
```

6. **与 Python 内置函数无缝兼容，无需频繁类型转换**

Python 的内置文件操作函数（如`open()`）、第三方库（如`shutil`/`zipfile`）都**直接支持 Path 对象作为参数**，无需手动转换为字符串，避免了`str(path)`的频繁类型转换，减少代码冗余：

```python
p = Path("D:/test.md")
# 直接传入Path对象，无需str(p)
with open(p, "r", encoding="utf-8") as f:
    content = f.read()

# 字符串路径（需直接传字符串）
with open(str(p), "r", encoding="utf-8") as f:
    content = f.read()
```



#### 4. 步骤 2: 上传与轮询 (Upload & Poll)

这一步实现了与 MinerU API 的交互，包括申请上传链接、上传文件、提交任务和轮询状态。

**1. 配置校验** → 2. **获取上传链接 + batch_id**（批量接口自动生成，无需手动提交） → 3. **文件上传**（含重试 + 代理禁用） → 4. **轮询状态**（根据 batch_id 持续查询，直至完成 / 失败 / 超时）

```python
@step_log("step_2_upload_and_poll")
def step_2_upload_and_poll(pdf_path_obj, output_dir_obj):
    """
    步骤2：上传PDF至MinerU并轮询解析任务状态
    核心流程：配置校验 → 获取上传链接 → 文件上传（含重试） → 任务轮询（直至完成/失败/超时）
    参数：pdf_path_obj-已校验的PDF Path对象；output_dir_obj-输出目录Path对象
    返回：解析结果ZIP包下载链接full_zip_url
    异常：ValueError(配置缺失)、RuntimeError(请求/上传失败)、TimeoutError(任务超时)
    """
    # 1. 前期配置校验，拦截无效配置
    if not mineru_config.base_url or not mineru_config.api_key:
        raise ValueError("MinerU服务地址(base_url)或API密钥(api_key)未配置，请检查配置文件！")

    # 2. 构造请求头，调用批量接口获取预签名上传地址与批次ID
    request_headers = {
        "Authorization": f"Bearer {mineru_config.api_key}",
        "Content-Type": "application/json"
    }
    url_get_upload = f"{mineru_config.base_url}/file-urls/batch"
    req_data = {
        "files": [{"name": pdf_path_obj.name}],
        "model_version": "vlm"
    }

    # 发起获取上传链接请求
    response = requests.post(url_get_upload, json=req_data, headers=request_headers)
    if response.status_code != 200:
        raise RuntimeError(f"请求MinerU服务失败，状态码：{response.status_code}，响应内容：{response.text}")

    resp_data = response.json()
    if resp_data["code"] != 0:
        raise RuntimeError(f"MinerU接口返回失败，code：{resp_data['code']}，msg：{resp_data['msg']}")

    # 提取预签名上传地址与任务批次ID
    signed_url = resp_data["data"]["file_urls"][0]
    batch_id = resp_data["data"]["batch_id"]

    # 3. 读取PDF文件二进制内容
    file_data = pdf_path_obj.read_bytes()

    # 4. 使用稳定Session上传文件（关闭系统环境变量，避免OSS预签名URL校验失败）
    with requests.Session() as session:
        session.trust_env = False
        put_response = session.put(signed_url, data=file_data, timeout=60)

        if put_response.status_code != 200:
            raise RuntimeError(f"PDF文件上传失败，状态码：{put_response.status_code}，响应：{put_response.text}")
    # 5. 轮询解析任务状态（带超时控制 + 服务端异常自动重试）
    poll_url = f"{mineru_config.base_url}/extract-results/batch/{batch_id}"
    timeout_seconds = 600  # 最大超时时间10分钟
    poll_interval = 3      # 轮询间隔3秒
    start_time = time.time()

    logger.debug("开始轮询MinerU解析结果...")

    while True:
        # 超时判断
        if time.time() - start_time > timeout_seconds:
            raise TimeoutError(f"MinerU解析任务超时（超时时间{timeout_seconds}秒），请检查服务或文件大小")
        try:
            poll_response = requests.get(poll_url, headers=request_headers, timeout=10)
        except Exception as e:
            logger.warning(f"轮询请求异常，将重试：{str(e)}")
            time.sleep(poll_interval)
            continue
        status_code = poll_response.status_code
        # 服务端5xx异常 → 降级重试
        if status_code != 200:
            if 500 <= status_code < 600:
                logger.warning(f"MinerU服务端异常{status_code}，休眠后自动重试")
                time.sleep(poll_interval)
                continue
            else:
                raise RuntimeError(f"轮询请求失败，客户端异常{status_code}，请检查API_KEY与服务地址")
        # 解析业务响应结果
        poll_data = poll_response.json()
        if poll_data["code"] != 0:
            raise RuntimeError(f"MinerU轮询接口异常，code：{poll_data['code']}，msg：{poll_data['msg']}")
        extract_results = poll_data["data"]["extract_result"]
        if not extract_results:
            logger.debug("暂无解析结果，继续轮询...")
            time.sleep(poll_interval)
            continue
        # 6. 根据任务状态执行对应逻辑
        task_state = extract_results[0]["state"]
        if task_state == "done":
            full_zip_url = extract_results[0]["full_zip_url"]
            if not full_zip_url:
                raise RuntimeError("MinerU解析完成，但未返回有效的ZIP下载链接")
            logger.info("PDF解析任务完成，准备下载结果包")
            return full_zip_url
        elif task_state == "failed":
            raise RuntimeError("MinerU解析任务执行失败，请检查PDF文件是否损坏或内容异常")
        else:
            logger.debug(f"任务处理中，当前状态：{task_state}，继续轮询...")
            time.sleep(poll_interval)
```

总结：为什么 PUT 需要 Session 而 POST/GET 不需要？

1. POST/GET 不需要 ：
   它们是普通的业务接口，只要 Token 对了就能过。即使你的请求经过了代理（比如开了 VPN），Header 里多了一些代理信息（如 Via ），服务端通常会 忽略 这些杂质，照样处理，所以直接用 requests.post/get 也没事。
2. PUT 必须需要 Session (trust_env=False) ：它使用的是 预签名 URL (Presigned URL) 直连对象存储（如 OSS/S3）。这种接口有 极其严格的签名校验机制 ：
   
   - 不开 Session (trust_env=True) ：如果你的电脑有代理，代理服务器可能会偷偷往 Header 里加东西（如 Via , Connection: close ）。服务端收到后，发现请求头和签名里约定的不一样，判定为 请求被篡改 ，直接报错 403 SignatureDoesNotMatch 。
   - 开了 Session (trust_env=False) ：强制绕过系统代理，发送 纯净无杂质 的原始请求。服务端校验签名一致，上传成功。
   一句话 ：POST/GET 是宽进宽出，怎么发都行；PUT 是 字节级校验 ，必须用 Session 确保请求头一个标点符号都不被代理修改。

#### 5. 步骤 3: 下载与解压 (Step 3: Download & Extract)

这一步负责下载 MinerU 返回的 ZIP 包，解压并提取目标 Markdown 文件。

```python
@step_log("step_3_download_and_extract")
def step_3_download_and_extract(zip_url:str, output_dir_pbj:Path, stem:str):
    """
    步骤3：下载MinerU解析结果ZIP包并解压，提取目标MD文件（重命名统一规范）
    核心流程：下载ZIP → 清理旧目录并解压 → 查找MD文件（按优先级） → 重命名统一为PDF同名
    参数：zip_url-ZIP包下载链接；output_dir_obj-输出目录Path；pdf_stem-PDF无后缀纯名称
    返回：最终MD文件的字符串格式绝对路径
    异常：RuntimeError(下载失败)、FileNotFoundError(无MD文件)
    """
    # 1.下载zip_url对应的资源
    response = requests.get(zip_url, timeout=120)
    if response.status_code != 200:
        raise RuntimeError(f"ZIP包下载失败，状态码：{response.status_code}，响应内容：{response.text}")

    # 2.保存zip文件
    # 文件保存和命名 output_dir_obj / 源文件名_result.zip
    zip_save_path =output_dir_pbj / f"{stem}_result.zip"
    zip_save_path.write_bytes(response.content)

    # 3.清理旧目录并解压zip包
    # 定义要解压的目录地址
    extract_target_dir = output_dir_pbj / stem
    if extract_target_dir.exists():
        # shutil.copy(源, 目标)   # 复制文件
        # shutil.move(源, 目标)   # 移动/重命名
        # extract_target_dir.unlink(missing_ok=True)   # 删除单个文件
        # extract_target_dir.rmdir()                   # 删除【空文件夹】
        shutil.rmtree(extract_target_dir)

    # 确保输出目录存在,重新创建下
    extract_target_dir.mkdir(parents=True, exist_ok=True)
    # 利用zipFile进行解压!
    with zipfile.ZipFile(zip_save_path, 'r') as zip_ref:
        zip_ref.extractall(extract_target_dir)
    # with ZipFile("test.zip", "w") as zip_write:
    #     zip_write.write("a.txt")   # 把 a.txt 打进压缩包
	# shutil.unpack_archive(zip_file_path_obj, extract_dir_path_obj)
    # shutil.make_archive("压缩后的文件名", "zip", "要压缩的文件夹路径")
    # 4. 处理下md文件,统一姓名,并且返回md的字符串地址
    # 获取全部文件
    md_file_list = list(extract_target_dir.rglob("*.md"))
    if not md_file_list or len(md_file_list) == 0:
        raise FileNotFoundError("未找到PDF对应的MD文件")

    target_md_file = None
    # 先读取同名的
    for md_file in md_file_list:
        if md_file.stem == stem:
            target_md_file = md_file
            break
    # 再读取叫full.md的
    if  not target_md_file :
        for md_file in md_file_list:
            if md_file.name.lower() == "full.md":
                target_md_file = md_file
                break

    if not target_md_file:
        target_md_file = md_file_list[0]

    # 如果不是文件名需要重命名,方便后续处理
    # md文件名  二狗子.md  full.md  不知道.md
    # 统一改成  原文件名（stem）.md
    # 不是原名字的时候，我才重命名
    if target_md_file.stem != stem:
        # 进行重命名
        # target_md_file.with_name(f"{stem}.md") 修改path对象 （不涉及文件操作） 返回结果是修改后path对象
        # target_md_file.rename(target_md_file.with_name(f"{stem}.md")) 修改磁盘中的文件名称（修改名称了） return 新的路径path
        target_md_file = target_md_file.rename(target_md_file.with_name(f"{stem}.md"))

        # 最终的md文件获取绝对路径，并且返回字符串类型
    final_md_str_path = str(target_md_file.resolve())
    return final_md_str_path
```

技术细节说明：

| 函数 / 方法                 | 所属模块 / 对象 |                           核心作用                           |                           关键特点                           |
| :-------------------------- | :-------------: | :----------------------------------------------------------: | :----------------------------------------------------------: |
| `zipfile.ZipFile(..., 'r')` |     zipfile     |         以只读模式打开 ZIP 压缩包，创建 ZIP 操作对象         |     需配合`with`语句使用，自动释放文件句柄，避免资源泄漏     |
| `zip_file_obj.extractall()` | zipfile.ZipFile | 将 ZIP 包内**所有文件 / 目录**解压至指定目录，保留原内部目录结构 |    一键解压，无需遍历文件，适配 MinerU 返回的 ZIP 包格式     |
| `shutil.rmtree()`           |     shutil      |            递归删除指定目录及所有子文件 / 子目录             | 比`os.rmdir`更彻底（os.rmdir 仅删除空目录），适合清理旧解压目录 |
| `Path.rglob("*.md")`        |  pathlib.Path   |    递归遍历当前目录及**所有子目录**，匹配指定通配符的文件    | 相比`glob()`（仅当前目录），更适配解压后可能的多层子目录结构 |
| `Path.with_name(新文件名)`  |  pathlib.Path   | 基于原路径，**仅修改文件名**，保留父目录，返回**新的 Path 对象** | 不修改实际文件，仅生成路径；自动保留父目录，无需手动拼接路径 |
| `Path.rename(新路径)`       |  pathlib.Path   | 将文件 / 目录从原路径**重命名 / 移动**至新路径，实际修改文件系统 | 若新路径已存在会报错；需捕获异常，处理文件被占用 / 权限不足等场景 |



#### 6. 单元测试 (Unit Test)

您可以在 `node_pdf_to_md.py` 文件底部直接运行以下测试代码：

```python
if __name__ == "__main__":

    # 单元测试：验证PDF转MD全流程
    logger.info("===== 开始node_pdf_to_md节点单元测试 =====")

    from app.utils.path_util import PROJECT_ROOT
    logger.info(f"测试获取根地址：{PROJECT_ROOT}")

    test_pdf_name = os.path.join("doc", "hak180产品安全手册.pdf")
    test_pdf_path = os.path.join(PROJECT_ROOT, test_pdf_name)

    # 构造测试状态
    test_state = create_default_state(
        task_id="test_pdf2md_task_001",
        pdf_path=test_pdf_path,
        local_dir=os.path.join(PROJECT_ROOT, "output")
    )

    node_pdf_to_md(test_state)

    logger.info("===== 结束node_pdf_to_md节点单元测试 =====")
```

测试结果：

<img src="images/image-20260203165149857.png" alt="image-20260203165149857" />

