# 掌柜智库项目(RAG)实战

## 5. 导入数据节点实现与测试

### 5.1 入口与类型判断 (node_entry)

**文件**: `app/import_process/agent/nodes/node_entry.py`
**相关工具类位置**: `app/utils/task_utils.py`

#### 节点作用与实现思路

**节点作用**: 作为数据加载流程的“总调度员”，负责接收外部输入的文件，识别文件类型（PDF或Markdown），并根据类型开启相应的处理分支。同时，它提取文件名作为全局元数据，并初始化任务追踪状态，确保后续流程可追溯、可监控。

**实现思路**:

1.  **路由分发**: 采用轻量级的条件判断逻辑，通过文件后缀 (`.pdf` / `.md`) 决定激活 `is_pdf_read_enabled` 还是 `is_md_read_enabled` 状态位，实现不同格式文件的差异化处理。
2.  **元数据提取**: 在入口处统一提任务标记(`task_id`)，作为贯穿整个知识库构建流程的唯一标识，避免后续节点重复解析。
3.  **任务监控初始化**: 集成 `task_utils`，记录当前任务 ID 和初始状态，为前端提供实时的进度反馈。

#### 步骤分解

1.  **接收状态**: 获取 `local_file_path`。
2.  **判断类型**: 检查文件后缀是 `.pdf` 还是 `.md`。
3.  **设置标记**: 更新 state 中的 `is_pdf_read_enabled` 或 `is_md_read_enabled`，供主图路由使用。
4.  **提取标题**: 从文件名中提取 `file_title`，后续作为元数据。

####  工具类解读：任务追踪

**文件**: `app/utils/task_utils.py`

**实现思路**:

1.  **内存管理**: 使用简单的内存字典 `_tasks_running_list` 和 `_tasks_done_list` 记录任务状态，轻量高效。
2.  **状态映射**: 维护 `_NODE_NAME_TO_CN` 字典，将技术性的节点名称（如 `node_entry`）映射为用户友好的中文名称（如 `检查文件`），方便前端展示。
3.  **SSE 集成**: 集成 SSE (Server-Sent Events) 推送机制，允许实时将任务进度推送到前端。
4.  **操作封装**: 提供 `add_running_task` 和 `add_done_task` 接口，方便各节点调用，屏蔽底层状态管理细节。

#### 代码实现

```python
import os

from pathlib import Path
from app.core.logger import logger, node_log
from app.import_process.agent.state import ImportGraphState, create_default_state
from app.utils.task_utils import add_running_task, add_done_task

@node_log("node_entry")
def node_entry(state: ImportGraphState) -> ImportGraphState:
    """
    节点: 入口节点 (node_entry)
    为什么叫这个名字: 作为图的 Entry Point，负责接收外部输入并决定流程走向。
    未来要实现:
        1. 进行任务状态记录,开始和结束列表记录
        2. 根据state中 local_file_path属性判断数据类型进而修改
           相关参数, is_md_read_enabled 或者 is_pdf_read_enabled
                    md_path 或者 pdf_path
        3. 不可解析结果类型不可用,直接输出对应警告日志! 逻辑路由节点会自动处理
        4. 获取file_tile标识,用于后期识别pdf对应的主体(item_name)进行兜底
    """

    # 1. 任务状态记录处理
    add_running_task(state['task_id'],'node_entry')

    # 2. 判断文件类型
    local_file_path = state['local_file_path']
    if not local_file_path:
        logger.warning(f"没有输入文件地址,无法处理,直接跳转到结束节点!")
        add_done_task(state['task_id'], 'node_entry')
        return state

    if local_file_path.endswith(".md"):
        state['is_md_read_enabled'] = True
        state['md_path'] = local_file_path
    elif local_file_path.endswith(".pdf"):
        state['is_pdf_read_enabled'] = True
        state['pdf_path'] = local_file_path
    else:
        logger.warning(f"虽然输出了loclal_file_path,但是无法识别文件类型,请检查输入文件类型是否正确,目前只支持md和pdf文件,请检查! {local_file_path}")
        add_done_task(state['task_id'], 'node_entry')
        return state

    # 3. 获取文件标识
    # 基于os.path处理
    file_title_os = os.path.basename(local_file_path).split(".")[0]
    # 基于pathlib处理
    file_title = Path(local_file_path).stem # 文件名 .name  文件夹名 .parent   文件后缀 .suffix
    state['file_title']= file_title

    add_done_task(state['task_id'], 'node_entry')
    return state
```

关键语法补充说明

| 语法 / 函数                       | 具体作用                            | 执行示例                                        |
| :-------------------------------- | :---------------------------------- | :---------------------------------------------- |
| `os.path.basename(document_path)` | 从完整路径提取文件名                | `/data/kb/test.pdf` → `test.pdf` p.stem         |
| `splitext(file_name)`             | 拆分文件名和后缀                    | `test.pdf` → `('test', '.pdf')`                 |
| `state.get("key", "")`            | 安全提取状态值，无 key 时返回默认值 | `state.get("a", "")` → 无 "a" 则返回 ""         |
| `sys._getframe().f_code.co_name`  | 动态获取当前函数名                  | 本节点中返回 `node_entry`                       |
| `add_running_task/add_done_task`  | 记录任务的节点运行状态              | 用于任务监控面板，展示节点执行进度(fastapi使用) |

#### 单元测试

您可以在 `node_entry.py` 文件底部直接运行以下测试代码：

```python
if __name__ == '__main__':

    # 单元测试：覆盖不支持类型、MD、PDF三种场景
    logger.info("===== 开始node_entry节点单元测试 =====")

    # 测试1: 不支持的TXT文件
    test_state1 = create_default_state(
        task_id="test_task_001",
        local_file_path="联想海豚用户手册.txt"
    )
    node_entry(test_state1)

    # 测试2: MD文件
    test_state2 = create_default_state(
        task_id="test_task_002",
        local_file_path="小米用户手册.md"
    )
    node_entry(test_state2)

    # 测试3: PDF文件
    test_state3 = create_default_state(
        task_id="test_task_003",
        local_file_path="万用表的使用.pdf"
    )
    node_entry(test_state3)

    logger.info("===== 结束node_entry节点单元测试 =====")
```



